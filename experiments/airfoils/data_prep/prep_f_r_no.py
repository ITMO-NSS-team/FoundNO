"""
prep_f_r_no.py - FNO/RNO data preparation (AirfRANS).

Converts raw AirfRANS .pth graphs into the fno_data npz dataset used by
both FNO and RNO training.

Pipeline (ported from FNO/convert_to_fno_dataset.py):
    1. Load raw graph (pos, x [N,5], y [N,4])
    2. Build airfoil profile polygon from wall points (wall_dist ~ 0)
    3. Interpolate all 4 fields + mask dummy onto a regular grid
       (LinearNDInterpolator, one triangulation per file)
    4. Mask: 0 inside the profile polygon / where interpolation failed

Output npz per sample:
    vx_in, vy_in          - scalar inlet velocities
    vel_x, vel_y, pressure, nut - [H, W] each
    mask                  - [H, W] float32, 1 = fluid, 0 = inside profile

Grid: 128x256, box [-0.5, 1.5] x [-0.5, 0.5].
File naming: global index = (batch - 1) * 100 + local index -> fno_data_{0001..1000}.npz

Usage:
    python -m experiments.airfoils.data_prep.prep_f_r_no \\
        --raw_root /media/.../airfrans \\
        --out_root /media/.../airfrans/fno_dataset
"""

import os
import re
import glob
import gc
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator
from matplotlib.path import Path


# Grid and bounding box
GRID_H, GRID_W = 128, 256
X_MIN, X_MAX = -0.5, 1.5
Y_MIN, Y_MAX = -0.5, 0.5


def build_airfoil_polygon(pos: np.ndarray, wall_dist: np.ndarray,
                          eps: float = 1e-6) -> Optional[Path]:
    """
    Build a Path polygon of the airfoil profile from points with wall_dist ~ 0.

    Points are ordered by polar angle around the profile center
    (chord midpoint at the mean camber line).
    """
    mask = wall_dist < eps
    wp = pos[mask]
    if len(wp) < 3:
        return None

    x_min, x_max = wp[:, 0].min(), wp[:, 0].max()
    x_c = (x_min + x_max) / 2.0
    chord = x_max - x_min

    slice_mask = np.abs(wp[:, 0] - x_c) < 0.02 * chord
    if slice_mask.sum() < 2:
        closest = np.argsort(np.abs(wp[:, 0] - x_c))[:10]
        y_top = wp[closest, 1].max()
        y_bot = wp[closest, 1].min()
    else:
        y_top = wp[slice_mask, 1].max()
        y_bot = wp[slice_mask, 1].min()
    y_c = (y_top + y_bot) / 2.0
    center = np.array([x_c, y_c])

    angles = np.arctan2(wp[:, 1] - center[1], wp[:, 0] - center[0])
    wp_sorted = wp[np.argsort(angles)]
    return Path(wp_sorted)


def file_sort_key(fname: str) -> int:
    """Extract trailing number from a file name for sorting."""
    nums = re.findall(r'(\d+)', fname)
    return int(nums[-1]) if nums else 0


def process_one_file(fpath: str, out_path: str,
                     airfoil_cache: dict) -> None:
    """
    Convert one raw graph to an fno_data npz.
    """
    data = torch.load(fpath, weights_only=False)
    pos = data.pos.numpy()
    x_feat = data.x.numpy()   # [N, 5]
    y_feat = data.y.numpy()   # [N, 4]

    vx_in = float(x_feat[0, 0])
    vy_in = float(x_feat[0, 1])

    # Airfoil polygon (cached by node count)
    cache_key = len(pos)
    if cache_key not in airfoil_cache:
        airfoil_cache[cache_key] = build_airfoil_polygon(pos, x_feat[:, 2])
    airfoil = airfoil_cache[cache_key]

    # Regular grid (built once per call; cheap)
    x_1d = np.linspace(X_MIN, X_MAX, GRID_W)
    y_1d = np.linspace(Y_MIN, Y_MAX, GRID_H)
    grid_x, grid_y = np.meshgrid(x_1d, y_1d)
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Interpolate all 4 fields + a dummy channel for the mask at once
    all_vals = np.column_stack([
        y_feat[:, 0],  # vel_x
        y_feat[:, 1],  # vel_y
        y_feat[:, 2],  # pressure
        y_feat[:, 3],  # nut
        np.ones(len(pos), dtype=np.float32),  # dummy for mask
    ])
    interp = LinearNDInterpolator(pos, all_vals, fill_value=0.0)
    result = interp(grid_pts)  # [N_grid, 5]

    vel_x = result[:, 0].reshape(GRID_H, GRID_W)
    vel_y = result[:, 1].reshape(GRID_H, GRID_W)
    pressure = result[:, 2].reshape(GRID_H, GRID_W)
    nut = result[:, 3].reshape(GRID_H, GRID_W)
    mask_val = result[:, 4].reshape(GRID_H, GRID_W)

    # Mask: 1 = fluid, 0 = hole
    mask = np.ones((GRID_H, GRID_W), dtype=np.float32)
    mask[mask_val < 0.5] = 0.0
    if airfoil is not None:
        inside = airfoil.contains_points(grid_pts).reshape(GRID_H, GRID_W)
        mask[inside] = 0.0

    np.savez_compressed(out_path,
                        vx_in=vx_in, vy_in=vy_in,
                        vel_x=vel_x, vel_y=vel_y,
                        pressure=pressure, nut=nut,
                        mask=mask)


def generate_fno_dataset(raw_root: str, out_root: str,
                         batches: Optional[list] = None,
                         resume: bool = True,
                         verbose: bool = True) -> dict:
    """
    Convert all raw graphs to fno_data npz files.

    Input layout: {raw_root}/graph_airfrans_data_batch_{1..10}/*.pth
    Output:       {out_root}/fno_data_{global_idx:04d}.npz

    Returns {'ok': int, 'skipped': int, 'failed': int}
    """
    os.makedirs(out_root, exist_ok=True)
    batches = batches or list(range(1, 11))
    airfoil_cache = {}

    total_ok = total_err = total_skip = 0
    for batch_num in batches:
        folder = f'graph_airfrans_data_batch_{batch_num}'
        folder_path = os.path.join(raw_root, folder)
        if not os.path.isdir(folder_path):
            if verbose:
                print(f'[skip] missing directory {folder_path}')
            continue

        pth_files = sorted(glob.glob(os.path.join(folder_path, '*.pth')),
                           key=file_sort_key)
        if verbose:
            print(f'  {folder}: {len(pth_files)} files')

        for local_idx, fpath in enumerate(pth_files, start=1):
            global_idx = (batch_num - 1) * 100 + local_idx
            out_name = f'fno_data_{global_idx:04d}.npz'
            out_path = os.path.join(out_root, out_name)

            if resume and os.path.exists(out_path):
                total_skip += 1
                continue

            try:
                process_one_file(fpath, out_path, airfoil_cache)
                total_ok += 1
            except Exception as e:
                total_err += 1
                print(f'  ERROR {os.path.basename(fpath)} -> {out_name}: {e}')

            gc.collect()

    if verbose:
        print(f'Done: ok={total_ok}, skipped={total_skip}, failed={total_err}')
        print(f'Saved to: {out_root}')
    return {'ok': total_ok, 'skipped': total_skip, 'failed': total_err}


def main():
    parser = argparse.ArgumentParser(
        description='FNO/RNO dataset preparation: raw AirfRANS graphs -> fno_data npz.')
    parser.add_argument('--raw_root', type=str, required=True,
                        help='Root containing graph_airfrans_data_batch_{1..10}')
    parser.add_argument('--out_root', type=str, required=True,
                        help='Where to write fno_data_*.npz')
    parser.add_argument('--batches', type=int, nargs='*', default=None,
                        help='Batch numbers (default: 1..10)')
    parser.add_argument('--no_resume', action='store_true',
                        help='Overwrite already existing files')
    args = parser.parse_args()

    result = generate_fno_dataset(
        raw_root=args.raw_root,
        out_root=args.out_root,
        batches=args.batches,
        resume=not args.no_resume,
    )
    print(result)


if __name__ == '__main__':
    main()

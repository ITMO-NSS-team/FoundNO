#!/usr/bin/env python3
"""
airfoil_eval.py - evaluate trained FNO / RNO / DNO / Geo-FNO models on the
test split and compare them on a COMMON reference grid.

Why a common reference: each method predicts on its own grid (FNO/RNO:
128x256 regular box, DNO: 256x256 universal square, Geo-FNO: 137x1128
C-grid). To make per-field errors comparable, every prediction is
interpolated (LinearNDInterpolator) onto the SAME reference points and the
rel-L2 is computed there - exactly like the airrans comparison
(R_D_Geo-F_F/add_stage2.py).

Reference per test sample (global 901..1000 -> batch_10, local 101..200):
    first N_ROWS (137) rows from index_raw_data_w -> physical coords
    (global_id_to_xy) + ground truth from the original graph (g.y).

Output per model:
    native:   test_loss (FieldLpLoss) + per-field on its own grid
    ref:      per-field rel-L2 + coverage on the common reference
    summary:  mean over the samples evaluated for every model (intersection)

Usage:
    python experiments/scripts/airfoil_eval.py \\
        --data.root_dir /media/.../airfrans \\
        --out runs_airfoils/leaderboard.json

    python experiments/scripts/airfoil_eval.py --limit 10   # quick check
    python experiments/scripts/airfoil_eval.py \\
        --checkpoint fno=/path/to/model_best.pth
"""

import argparse
import json
import os
import sys

sys.path.append('.')

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator

from fnofound.data.data.datasets.airfoil_datasets import (
    get_airfoil_dataset,
    collate_fn,
)
from fnofound.models import FNO2d, RNO2d, DNOAirfoil
from fnofound.utils.airfoil_trainer import AirfoilTrainer

DATA_ROOT_DEFAULT = '/media/expiwt/f2d8c27a-bdac-4400-b1be-cdb75b6b5a40/airfrans'
DEFAULT_CHECKPOINTS = {
    'fno':    '{root}/fno_dataset/runs/fno_m16_w64_l4_b4/model_best.pth',
    'rno':    '{root}/fno_dataset/runs_rno/rno_m16_w64_l4_b4/model_best.pth',
    'dno':    '{root}/DNO_data/dno_small/runs_dno_small/dno_m16_w64_l4_b4/model_best.pth',
    'geofno': '{root}/Geo-FNO_data/runs/geofno_m16_w32_l4_b4/model_best.pth',
}
CASE_DIRS = {
    'fno':    'fno_dataset',
    'dno':    'DNO_data/dno_small',
    'geofno': 'Geo-FNO_data',
}
FIELDS = ['vel_x', 'vel_y', 'pressure', 'nu_t']
FNO_BOX = (-0.5, 1.5, -0.5, 0.5)  # X_MIN, X_MAX, Y_MIN, Y_MAX


# Checkpoint inference

def infer_arch(state: dict) -> dict:
    """Infer (kind, hyperparameters) from a checkpoint state dict."""
    fc0 = state['fc0.weight'].shape
    width, fc_in = int(fc0[0]), int(fc0[1])

    if fc_in == 4:
        in_channels, use_grid = 2, True
    elif fc_in == 8:
        in_channels, use_grid = 8, False
    else:
        in_channels, use_grid = fc_in - 2, True

    if any(k.startswith('riesz_conductors') for k in state):
        n_layers = sum(1 for k in state if k.startswith('riesz_conductors')) // 6
        modes = int(state['riesz_conductors.0.w_global_lo'].shape[2])
        return {'kind': 'rno', 'in_channels': in_channels, 'modes': modes,
                'width': width, 'n_layers': n_layers, 'use_grid': use_grid,
                'use_geom': False, 'padding': 0}
    if any(k.startswith('blocks.') for k in state):
        n_layers = sum(1 for k in state if k.startswith('blocks.')) // 4
        modes = int(state['blocks.0.spectral.weights1'].shape[2])
        return {'kind': 'geofno', 'in_channels': in_channels, 'modes': modes,
                'width': width, 'n_layers': n_layers, 'use_grid': use_grid,
                'use_geom': False, 'padding': 8}
    n_layers = sum(1 for k in state if k.startswith('spectral_convs')) // 2
    modes = int(state['spectral_convs.0.weights1'].shape[2])
    if any(k.startswith('conv_grids') for k in state):
        use_geom = any(k.startswith('conv_meshes') for k in state)
        return {'kind': 'dno', 'in_channels': in_channels, 'modes': modes,
                'width': width, 'n_layers': n_layers, 'use_grid': use_grid,
                'use_geom': use_geom, 'padding': 0}
    return {'kind': 'fno', 'in_channels': in_channels, 'modes': modes,
            'width': width, 'n_layers': n_layers, 'use_grid': use_grid,
            'use_geom': False, 'padding': 0}


def build_model(arch: dict) -> torch.nn.Module:
    kind = arch['kind']
    if kind == 'rno':
        return RNO2d(in_channels=arch['in_channels'], out_channels=4,
                     modes=arch['modes'], width=arch['width'],
                     n_layers=arch['n_layers'], use_grid=arch['use_grid'])
    if kind == 'dno':
        return DNOAirfoil(in_channels=arch['in_channels'], out_channels=4,
                          modes=arch['modes'], width=arch['width'],
                          n_layers=arch['n_layers'], use_grid=arch['use_grid'],
                          use_geom=arch['use_geom'], padding=arch['padding'])
    return FNO2d(in_channels=arch['in_channels'], out_channels=4,
                 modes=arch['modes'], width=arch['width'],
                 n_layers=arch['n_layers'], use_grid=arch['use_grid'],
                 padding=arch['padding'])


# Common reference (like airrans add_stage2)

def global_to_batch_local(g: int) -> tuple:
    """Map a global sample index to (batch, local file number)."""
    group = (g - 1) // 400
    local = g - group * 400
    batch = group * 4 + 1 + (local - 1) // 100
    return batch, local


def load_reference(index_root: str, raw_root: str, batch: int, local: int,
                   n_rows: int = 137):
    """
    Reference points for one sample: coords + ground truth from the first
    n_rows full rows (index_raw_data_w) at the original graph nodes.
    """
    idx = torch.load(os.path.join(index_root, f'batch_{batch}',
                                  f'graph_airfrans_data_{local}.pth'),
                     weights_only=False)
    g = torch.load(os.path.join(raw_root, f'graph_airfrans_data_batch_{batch}',
                                f'graph_airfrans_data_{local}.pth'),
                   weights_only=False)

    ids = np.array([int(x) for r in idx['global_wing_rows'][:n_rows] for x in r],
                   dtype=np.int64)
    ids, uniq = np.unique(ids, return_index=True)
    ids = ids[np.argsort(uniq)]
    coords = np.array([idx['global_id_to_xy'][int(i)] for i in ids], dtype=np.float64)
    y_true = np.asarray(g['y'].numpy()[ids], dtype=np.float64)  # [N, 4]
    return coords, y_true


def grid_coords(case: str, sample: dict) -> np.ndarray:
    """Physical coordinates of the model's prediction points."""
    if case == 'dno':
        gm = sample['grid_mesh'].numpy()      # [H, W, 2]
        m = sample['mask'].numpy() > 0
        return gm[m]
    if case == 'geofno':
        return sample['pos'].numpy().reshape(-1, 2)
    # fno / rno: regular box grid, masked points only
    H, W = sample['x'].shape[:2]
    m = sample['mask'].numpy() > 0
    x_min, x_max, y_min, y_max = FNO_BOX
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, W),
                         np.linspace(y_min, y_max, H))
    return np.stack([gx[m], gy[m]], -1)


def interp_to_ref(points, values, ref_coords):
    """points [N,2], values [N,C] -> values at ref_coords [M,2] -> [M,C]."""
    out = np.full((len(ref_coords), values.shape[1]), np.nan)
    for c in range(values.shape[1]):
        itp = LinearNDInterpolator(points, values[:, c])
        out[:, c] = itp(ref_coords)
    return out


def rel_l2_per_field(pred, true) -> dict:
    """pred/true [N,4] -> {field: rel-L2}, coverage."""
    res = {}
    for i, name in enumerate(FIELDS):
        ok = np.isfinite(pred[:, i])
        num = np.linalg.norm(pred[ok, i] - true[ok, i])
        den = np.linalg.norm(true[ok, i])
        res[name] = float(num / max(den, 1e-12))
    res['coverage'] = float(np.isfinite(pred).all(axis=1).mean())
    return res


# Main

def main():
    parser = argparse.ArgumentParser(
        description='AirfRANS comparison on a common reference grid.')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['fno', 'rno', 'dno', 'geofno'],
                        choices=['fno', 'rno', 'dno', 'geofno'])
    parser.add_argument('--checkpoint', action='append', default=[],
                        metavar='MODEL=PATH')
    parser.add_argument('--data.root_dir', type=str, default=DATA_ROOT_DEFAULT)
    parser.add_argument('--n_rows', type=int, default=137,
                        help='Rows of the reference grid')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0,
                        help='Evaluate on the first N test samples only')
    parser.add_argument('--out', type=str, default='./runs_airfoils/leaderboard.json')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    root = getattr(args, 'data.root_dir', DATA_ROOT_DEFAULT)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'device: {device} | reference rows: {args.n_rows}')

    checkpoints = dict(DEFAULT_CHECKPOINTS)
    for kv in args.checkpoint:
        model, path = kv.split('=', 1)
        checkpoints[model] = path

    results = {}
    for model_name in args.models:
        ckpt_path = checkpoints[model_name].format(root=root)
        if not os.path.exists(ckpt_path):
            print(f'[skip] {model_name}: no checkpoint at {ckpt_path}')
            continue

        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        arch = infer_arch(state)
        model = build_model(arch)
        if arch['kind'] == 'geofno':
            from fnofound.models.fno2d import remap_geofno_keys
            state = remap_geofno_keys(state)
        model.load_state_dict(state, strict=True)
        model.eval().to(device)

        case = 'fno' if model_name in ('fno', 'rno') else model_name
        ds = get_airfoil_dataset(case, os.path.join(root, CASE_DIRS[case]),
                                 split='test')
        n_eval = min(len(ds), args.limit) if args.limit else len(ds)

        params = sum(p.numel() for p in model.parameters())
        ref_metrics = {}
        for k in range(n_eval):
            g = ds.indices[k]
            batch, local = global_to_batch_local(g)
            idx_path = os.path.join(root, 'index_raw_data_w', f'batch_{batch}',
                                    f'graph_airfrans_data_{local}.pth')
            if not os.path.exists(idx_path):
                continue
            coords, y_true = load_reference(
                os.path.join(root, 'index_raw_data_w'), root, batch, local,
                n_rows=args.n_rows)

            sample = ds[k]
            x = sample['x'].unsqueeze(0).to(device)
            mask = None if sample['mask'] is None else \
                sample['mask'].unsqueeze(0).unsqueeze(-1).to(device)
            grid_mesh = None if sample['grid_mesh'] is None else \
                sample['grid_mesh'].unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x, mask=mask, grid_mesh=grid_mesh)
            pred = pred[0].cpu().numpy()                # [H, W, 4]

            if case == 'geofno':
                values = pred.reshape(-1, 4)
            else:
                m = sample['mask'].numpy() > 0
                values = pred[m]
            pts = grid_coords(case, sample)
            y_pred = interp_to_ref(pts, values, coords)
            ref_metrics[g] = rel_l2_per_field(y_pred, y_true)

        # native metrics on the model's own grid (quick, single pass)
        from torch.utils.data import DataLoader, Subset
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)
        trainer = AirfoilTrainer(model, device=device)
        native = trainer.evaluate(loader)

        results[model_name] = {
            'checkpoint': ckpt_path,
            'arch': arch,
            'params_count': int(params),
            'native': {'test_loss': native['loss'],
                       'per_field': native['per_field']},
            'ref': ref_metrics,
        }

        # mean over the evaluated samples
        vals = list(ref_metrics.values())
        mean = {f: float(np.mean([v[f] for v in vals])) for f in FIELDS}
        mean['coverage'] = float(np.mean([v['coverage'] for v in vals]))
        results[model_name]['ref_mean'] = mean
        pf = ' | '.join(f'{f}={mean[f]:.4f}' for f in FIELDS)
        print(f"{model_name}: ref {pf} | cov={mean['coverage']:.3f} | "
              f"native={native['loss']:.4f} | params={params:,} "
              f"| samples={len(vals)}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {args.out}')

    # summary table on the common reference (intersection of samples)
    if results:
        common = None
        for r in results.values():
            s = set(r['ref'])
            common = s if common is None else (common & s)
        print(f'\n=== Summary on common reference ({len(common)} samples) ===')
        print(f'{"model":<8} {"vel_x":<8} {"vel_y":<8} {"pressure":<9} '
              f'{"nu_t":<8} {"cov":<6} {"params":<10}')
        for name in sorted(results, key=lambda n: results[n]['ref_mean']['vel_x']):
            r = results[name]
            vals = [r['ref'][g] for g in common]
            mean = {f: float(np.mean([v[f] for v in vals])) for f in FIELDS}
            cov = float(np.mean([v['coverage'] for v in vals]))
            print(f'{name:<8} {mean["vel_x"]:<8.4f} {mean["vel_y"]:<8.4f} '
                  f'{mean["pressure"]:<9.4f} {mean["nu_t"]:<8.4f} '
                  f'{cov:<6.3f} {r["params_count"]:<10,}')


if __name__ == '__main__':
    main()

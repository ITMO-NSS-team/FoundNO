#!/usr/bin/env python3
"""
airfoil_compare_fl4.py - AirfRANS comparison on the ORIGINAL MESH reference
points with QUADRATIC interpolation.

Port of the airrans comparison (compare_fl4/compare_fl4.py). Compared to
airfoil_eval.py:

  - interpolation onto the reference points is QUADRATIC MLS (moving least
    squares, degree-2 polynomial basis + ridge) with an NN fallback, instead
    of linear (LinearNDInterpolator);
  - default checkpoints are the fl4 models (trained with nu_t in the loss,
    airrans runs_fl4 / runs_rno_fl4 / runs_dno_small_fl4);
  - test set: batch_10 samples 101..200 except 133, 155 (98 samples);
  - extra outputs: pooled rel-L2, per-sample CSVs, per-sample plots and
    run_config.json.

Reference per sample: the first N_ROWS (137) rows of global_wing_rows
(index_raw_data_w) -> physical coords (global_id_to_xy) + ground truth from
the original graph; points are cropped to the FNO/RNO box.

Usage:
    python experiments/scripts/airfoil_compare_fl4.py \\
        --data.root_dir /media/.../airfrans \\
        --out runs_airfoils/compare_fl4

    python experiments/scripts/airfoil_compare_fl4.py --no-plots --limit 10
    python experiments/scripts/airfoil_compare_fl4.py \\
        --checkpoint fno=/path/to/model_best.pth
"""

import argparse
import json
import os
import sys
import time

sys.path.append('.')

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from airfoil_eval import (                  # same directory (shared helpers)
    infer_arch,
    build_model,
    grid_coords,
    load_reference,
    global_to_batch_local,
    FIELDS,
    FNO_BOX,
)
from fnofound.data.data.datasets.airfoil_datasets import get_airfoil_dataset

DATA_ROOT_DEFAULT = '/media/expiwt/f2d8c27a-bdac-4400-b1be-cdb75b6b5a40/airfrans'
DEFAULT_FL4_CHECKPOINTS = {
    'fno':    '{root}/fno_dataset/runs_fl4/fno_m24_w64_l5_b4/model_best.pth',
    'rno':    '{root}/fno_dataset/runs_rno_fl4/rno_m24_w64_l5_b4/model_best.pth',
    'dno':    '{root}/DNO_data/dno_small/runs_dno_small_fl4/dno_m8_w64_l5_b4/model_best.pth',
    'geofno': '{root}/Geo-FNO_data/runs_fl4/geofno_m24_w64_l5_b4/model_best.pth',
}
CASE_DIRS = {
    'fno':    'fno_dataset',
    'dno':    'DNO_data/dno_small',
    'geofno': 'Geo-FNO_data',
}
N_ROWS = 137                     # full rows of the reference (as Geo-FNO N_ROW)
X_MIN, X_MAX, Y_MIN, Y_MAX = FNO_BOX
ALL_GLOBAL = [g for g in range(901, 1001) if g not in (933, 955)]  # 98 samples
PLOT_GLOBAL = list(range(901, 911))
MLS_K = 16                      # neighbors for the quadratic MLS
MLS_RIDGE = 1e-10               # ridge regularization
COV_RADIUS = 0.02               # NN fallback radius (coverage metric)
COVERAGE_FILTER = False         # False: ALL reference points enter the metrics


def interp_to_ref(points, values, ref_coords):
    """Quadratic MLS interpolation onto the reference points.

    points [N,2], values [N,C] -> values at ref_coords [M,2] -> [M,C].
    For each reference point: k nearest source points, weights 1/d^2,
    weighted least squares over the basis [1, x, y, x^2, xy, y^2] with ridge.
    NN fallback where the reference point is farther than COV_RADIUS from the
    nearest source cell (quadratic extrapolation there would be garbage).
    Returns (out [M, C], covered [M]).
    """
    pts = points.astype(np.float64)
    vals = values.astype(np.float64)
    ref = ref_coords.astype(np.float64)

    # deduplicate source points (zero distances -> inf weights otherwise)
    _, uniq_idx = np.unique(pts, axis=0, return_index=True)
    if uniq_idx.shape[0] < pts.shape[0]:
        pts = pts[uniq_idx]
        vals = vals[uniq_idx]

    k = min(MLS_K, pts.shape[0])
    tree = cKDTree(pts)
    d, idx = tree.query(ref, k=k)          # [M, k]
    if k == 1:
        d = d[:, None]
        idx = idx[:, None]

    if COVERAGE_FILTER:
        covered = d[:, 0] <= COV_RADIUS
    else:
        covered = np.ones(len(ref), dtype=bool)

    X = pts[idx]                            # [M, k, 2]
    V = vals[idx]                           # [M, k, C]

    w = 1.0 / (d * d + 1e-16)               # [M, k]
    w /= w.sum(axis=1, keepdims=True)

    x0, y0 = X[..., 0], X[..., 1]
    A = np.stack([np.ones_like(x0), x0, y0, x0 * x0, x0 * y0, y0 * y0], -1)
    AW = A * w[..., None]
    G = np.einsum('mki,mkj->mij', AW, A)    # [M, 6, 6]
    G += MLS_RIDGE * np.eye(6)
    B = np.einsum('mki,mkc->mic', AW, V)    # [M, 6, C]
    c = np.linalg.solve(G, B)               # [M, 6, C]

    rx, ry = ref[:, 0], ref[:, 1]
    R = np.stack([np.ones_like(rx), rx, ry, rx * rx, rx * ry, ry * ry], -1)
    out = np.einsum('mi,mic->mc', R, c)

    far = d[:, 0] > COV_RADIUS
    if far.any():
        out[far] = vals[idx[far, 0]]
    return out, covered


def rel_l2_per_field(pred, true, covered):
    res = {}
    for i, name in enumerate(FIELDS):
        ok = covered & np.isfinite(pred[:, i])
        num = np.linalg.norm(pred[ok, i] - true[ok, i])
        den = np.linalg.norm(true[ok, i])
        res[name] = float(num / max(den, 1e-12))
    res['coverage'] = float(covered.mean())
    return res


def plot_true_pred_error(sample, coords, y_true, y_pred, covered, method, save_path):
    fig, axes = plt.subplots(4, 3, figsize=(15, 17))
    fig.suptitle(f'{method} (fl4) | sample {sample}', fontsize=14)
    for i, name in enumerate(FIELDS):
        t = y_true[:, i]
        p = y_pred[:, i]
        m_pred = covered & np.isfinite(p)
        m_true = np.isfinite(t)
        vmin = min(t[m_true].min(), p[m_pred].min())
        vmax = max(t[m_true].max(), p[m_pred].max())
        for j, (vals, m, title, vlo, vhi) in enumerate([
                (p, m_pred, f'predict {name}', vmin, vmax),
                (t, m_true, f'true {name}', vmin, vmax),
                (np.abs(p - t), m_pred, f'|error| {name}', 0, max(vmax - vmin, 1e-9))]):
            ax = axes[i, j]
            sc = ax.scatter(coords[m, 0], coords[m, 1], c=vals[m], s=0.4,
                            cmap='RdBu_r' if name == 'pressure' else 'jet',
                            vmin=vlo, vmax=vhi)
            ax.set_title(title, fontsize=10)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def run_method(name, ckpt_path, root, device, do_plots, out_dir, n_rows, samples):
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    arch = infer_arch(state)
    model = build_model(arch)
    if arch['kind'] == 'geofno':
        from fnofound.models.fno2d import remap_geofno_keys
        state = remap_geofno_keys(state)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[{name}] {arch} | params={n_params:,} | {ckpt_path}', flush=True)

    case = 'fno' if name in ('fno', 'rno') else name
    ds = get_airfoil_dataset(case, os.path.join(root, CASE_DIRS[case]),
                             indices=samples)

    metrics = {}
    pooled_num = np.zeros(len(FIELDS))
    pooled_den = np.zeros(len(FIELDS))
    t0 = time.time()
    for k, g in enumerate(ds.indices):
        batch, local = global_to_batch_local(g)
        idx_path = os.path.join(root, 'index_raw_data_w', f'batch_{batch}',
                                f'graph_airfrans_data_{local}.pth')
        if not os.path.exists(idx_path):
            continue
        coords, y_true = load_reference(
            os.path.join(root, 'index_raw_data_w'), root, batch, local,
            n_rows=n_rows)

        sample = ds[k]
        x = sample['x'].unsqueeze(0).to(device)
        mask = None if sample['mask'] is None else \
            sample['mask'].unsqueeze(0).unsqueeze(-1).to(device)
        grid_mesh = None if sample['grid_mesh'] is None else \
            sample['grid_mesh'].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x, mask=mask, grid_mesh=grid_mesh)
        pred = pred[0].cpu().numpy()                    # [H, W, 4]

        if case == 'geofno':
            values = pred.reshape(-1, 4)
        else:
            m = sample['mask'].numpy() > 0
            values = pred[m]
        pts = grid_coords(case, sample)
        y_pred, covered = interp_to_ref(pts, values, coords)
        metrics[g] = rel_l2_per_field(y_pred, y_true, covered)
        for i in range(len(FIELDS)):
            ok = covered & np.isfinite(y_pred[:, i])
            pooled_num[i] += np.sum((y_pred[ok, i] - y_true[ok, i]) ** 2)
            pooled_den[i] += np.sum(y_true[ok, i] ** 2)
        if do_plots and g in PLOT_GLOBAL:
            plot_true_pred_error(g, coords, y_true, y_pred, covered, name,
                                 os.path.join(out_dir, f'{name.lower()}_s{g}.png'))
        if (k + 1) % 10 == 0:
            print(f'  [{name}] {k+1}/{len(ds.indices)} ({time.time()-t0:.0f}s)', flush=True)
    print(f'[{name}] done in {time.time()-t0:.0f}s', flush=True)
    pooled = {FIELDS[i]: float(np.sqrt(pooled_num[i] / max(pooled_den[i], 1e-12)))
              for i in range(len(FIELDS))}
    return metrics, pooled


def save_outputs(all_metrics, all_pooled, root, out_dir):
    # metrics per sample (json)
    with open(os.path.join(out_dir, 'metrics_all98.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2, default=float)

    # mean +/- std over samples
    rows = []
    for name, md in all_metrics.items():
        df = pd.DataFrame(md).T
        row = {'architecture': name}
        for f in FIELDS:
            row[f'{f}_mean'] = df[f].mean()
            row[f'{f}_std'] = df[f].std()
        row['coverage'] = df['coverage'].mean()
        rows.append(row)
    summary = pd.DataFrame(rows).set_index('architecture')
    summary.to_csv(os.path.join(out_dir, 'summary_all98.csv'), float_format='%.5f')

    # pooled rel-L2 over all points of all samples
    pooled_rows = []
    for name, p in all_pooled.items():
        row = {'architecture': name}
        for f in FIELDS:
            row[f] = p[f]
        row['coverage'] = np.mean([md['coverage'] for md in all_metrics[name].values()])
        pooled_rows.append(row)
    pd.DataFrame(pooled_rows).set_index('architecture').to_csv(
        os.path.join(out_dir, 'summary_by_architecture.csv'), float_format='%.5f')

    # per-sample physical summary (vx_in/vy_in/alpha/speed + per-method errors)
    samples = sorted({int(g) for md in all_metrics.values() for g in md})
    rows = []
    for g in samples:
        row = {'sample': g}
        d = np.load(os.path.join(root, 'fno_dataset', f'fno_data_{g:04d}.npz'))
        vxi, vyi = float(d['vx_in']), float(d['vy_in'])
        row['vx_in'] = vxi
        row['vy_in'] = vyi
        row['alpha_deg'] = float(np.degrees(np.arctan2(vyi, vxi)))
        row['speed'] = float(np.hypot(vxi, vyi))
        for name, md in all_metrics.items():
            for f in FIELDS:
                row[f'{name}_{f}'] = md[g][f]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'summary_per_sample_phys.csv'),
                              index=False, float_format='%.5f')

    run_cfg = {
        'interpolation': ('quadratic MLS (k=%d, ridge=%g) + NN fallback '
                          'beyond r=%.2f from the grid') % (MLS_K, MLS_RIDGE, COV_RADIUS),
        'coverage_filter': COVERAGE_FILTER,
        'coverage_radius': COV_RADIUS,
        'reference': 'batch_10, %d rows, box x in [%g,%g] y in [%g,%g]'
                     % (N_ROWS, X_MIN, X_MAX, Y_MIN, Y_MAX),
        'samples': samples,
    }
    with open(os.path.join(out_dir, 'run_config.json'), 'w') as f:
        json.dump(run_cfg, f, indent=2)

    print('\n=== SUMMARY (98 samples, rel-L2 mean +/- std) ===')
    for name in summary.index:
        parts = [f'{f}={summary.loc[name, f"{f}_mean"]:.4f}±'
                 f'{summary.loc[name, f"{f}_std"]:.4f}' for f in FIELDS]
        print(f'{name:8s} | ' + ' | '.join(parts) +
              f' | cov={summary.loc[name, "coverage"]:.3f}')
    print('\nsaved to', out_dir, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='AirfRANS fl4 comparison on the original mesh points '
                    '(quadratic MLS).')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['fno', 'rno', 'dno', 'geofno'],
                        choices=['fno', 'rno', 'dno', 'geofno'])
    parser.add_argument('--checkpoint', action='append', default=[],
                        metavar='MODEL=PATH')
    parser.add_argument('--data.root_dir', type=str, default=DATA_ROOT_DEFAULT)
    parser.add_argument('--n_rows', type=int, default=N_ROWS)
    parser.add_argument('--limit', type=int, default=0,
                        help='Use only the first N test samples')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--out', type=str, default='./runs_airfoils/compare_fl4')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    root = getattr(args, 'data.root_dir', DATA_ROOT_DEFAULT)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    print(f'device: {device} | reference: batch_10, {args.n_rows} rows, '
          f'box x in [{X_MIN},{X_MAX}] y in [{Y_MIN},{Y_MAX}]', flush=True)

    samples = ALL_GLOBAL[:args.limit] if args.limit else ALL_GLOBAL
    print(f'samples: {len(samples)} | interpolation: quadratic MLS '
          f'(k={MLS_K}, ridge={MLS_RIDGE:g}) + NN fallback beyond '
          f'r={COV_RADIUS}', flush=True)

    checkpoints = dict(DEFAULT_FL4_CHECKPOINTS)
    for kv in args.checkpoint:
        model, path = kv.split('=', 1)
        checkpoints[model] = path

    # reference sanity check on the first sample
    c, y = load_reference(os.path.join(root, 'index_raw_data_w'), root, 10, 101,
                          n_rows=args.n_rows)
    print(f'reference on sample 101: {c.shape[0]} points', flush=True)

    all_metrics, all_pooled = {}, {}
    for name in args.models:
        ckpt_path = checkpoints[name].format(root=root)
        if not os.path.exists(ckpt_path):
            print(f'[skip] {name}: no checkpoint at {ckpt_path}')
            continue
        m, p = run_method(name, ckpt_path, root, device, not args.no_plots,
                          out_dir, args.n_rows, samples)
        all_metrics[name] = m
        all_pooled[name] = p

    if not all_metrics:
        raise SystemExit('no checkpoints found - nothing to compare')

    save_outputs(all_metrics, all_pooled, root, out_dir)


if __name__ == '__main__':
    main()

"""
prep_geofno.py - Geo-FNO data preparation (AirfRANS).

Builds the unfolded C-grid dataset from the common index stage
(index_raw_data_w, see index.py).

Pipeline (ported from Geo_FNO/data_prep.ipynb):
    1. Filter pass over index files:
       - number of rows within [MIN_ROWS, MAX_ROWS]
       - at least N_ROW rows
       - row N_ROW-1 has >= THRESHOLD_FRAC points of the first row
       -> collect nl = len(row N_ROW-1) per kept sample,
          n_dots = median(nl) over the dataset
    2. Per sample: take rows[:N_ROW], symmetrically trim each row to nl
       (cut_start = (nk-nl)//2 + (nk-nl)%2, cut_end = (nk-nl)//2)
    3. interp_fun: resample each row to exactly n_dots points via
       arc-length CubicSpline (adds new global ids on the curve or
       removes points uniformly)
    4. Gather matrices and save as .pth

Output per sample (torch.save dict):
    name, ids (N_ROW, n_dots), pos (N_ROW, n_dots, 2),
    x_feats (N_ROW, n_dots, 5), fields (N_ROW, n_dots, 4),
    field_names, vx_in, vy_in, n_dots, N_ROW

Usage:
    python -m experiments.airfoils.data_prep.prep_geofno \\
        --index_root /media/.../airfrans/index_raw_data_w \\
        --graph_root /media/.../airfrans \\
        --out_root   /media/.../airfrans/Geo-FNO_data
"""

import os
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import CubicSpline


N_ROW = 137                      # rows to keep per sample
MIN_ROWS, MAX_ROWS = 120, 180    # sample filter: number of rows
THRESHOLD_FRAC = 0.9             # row N_ROW-1 must have >= 90% of first row points
FIELD_NAMES = ['vel_x', 'vel_y', 'pressure', 'nut']


# Row resampling helpers (ported from Geo_FNO/data_prep.ipynb, cell 20)


def _row_spline(t: np.ndarray, vals: np.ndarray) -> List:
    """Cubic splines over row points. vals: (n, F) -> list of F splines."""
    return [CubicSpline(t, vals[:, f]) for f in range(vals.shape[1])]


def _eval_spline(spl: List, t_new: np.ndarray) -> np.ndarray:
    """Spline values at t_new. Returns (len(t_new), F)."""
    return np.stack([s(t_new) for s in spl], axis=-1)


def _remove_points(ids: List[int], m: int) -> List[int]:
    """Uniformly remove m nodes: step = n // m (integer arithmetic)."""
    n = len(ids)
    drop = {(k * n) // m for k in range(m)}
    return [id_ for i, id_ in enumerate(ids) if i not in drop]


def _add_points(ids: List[int], t: np.ndarray, m: int, next_id: Callable):
    """Add m nodes uniformly: k-th new point goes into gap (k*(n-1))//m.

    If c points land in the same gap, they split it into fractions
    1/(c+1), 2/(c+1), ..., c/(c+1).

    Returns (new ids, new t, {id_new: (id_left, id_right)}).
    """
    n = len(ids)
    gap_cnt = {}
    for k in range(m):
        g = (k * (n - 1)) // m
        gap_cnt[g] = gap_cnt.get(g, 0) + 1

    new_pts = []  # (t_new, id_left, id_right)
    for g, c in gap_cnt.items():
        a, b = ids[g], ids[g + 1]
        ta, tb = t[g], t[g + 1]
        for q in range(1, c + 1):
            frac = q / (c + 1)
            new_pts.append((ta + frac * (tb - ta), a, b))
    new_pts.sort(key=lambda x: x[0])

    new_id_list = [next_id() for _ in new_pts]

    merged = ([(t[i], ids[i]) for i in range(n)] +
              [(tp[0], nid) for tp, nid in zip(new_pts, new_id_list)])
    merged.sort(key=lambda x: x[0])

    cur = [x[1] for x in merged]
    cur_t = [x[0] for x in merged]
    inserted = {nid: (a, b) for (_, a, b), nid in zip(new_pts, new_id_list)}
    return cur, cur_t, inserted


def interp_fun(row_ids: List[int], pos: np.ndarray, fields: np.ndarray,
               n_dots: int, next_id: Callable, x_feats: Optional[np.ndarray] = None):
    """
    Bring one row to exactly n_dots points.

    row_ids : list[int]  - global ids of the row nodes (ordered along the row)
    pos     : (n, 2)     - node coordinates
    fields  : (n, F)     - node field values (vel_x, vel_y, pressure, nut)
    n_dots  : int        - target number of points
    next_id : callable   - generator of NEW global ids (not used in the sample)
    x_feats : (n, X) | None - node features; new points get the mean of neighbors

    Returns (new_ids, new_pos, new_fields[, new_x_feats]) - all of size n_dots.
    """
    n = len(row_ids)
    if n == n_dots:
        if x_feats is not None:
            return list(row_ids), pos, fields, x_feats
        return list(row_ids), pos, fields
    if n < 2:
        raise ValueError(f'row with {n} points cannot be interpolated')

    # parameter: cumulative arc length, normalized to [0, 1]
    diff = np.diff(pos, axis=0)
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diff, axis=1))])
    if t[-1] > 0:
        t = t / t[-1]
    else:
        t = np.arange(n, dtype=float)

    id2pos = {id_: i for i, id_ in enumerate(row_ids)}

    if n < n_dots:
        # add points on the curve, in the middle of gaps
        spl_xy = _row_spline(t, pos)
        spl_f = _row_spline(t, fields)
        new_ids, new_t, inserted = _add_points(row_ids, t, n_dots - n, next_id)
        new_pos = _eval_spline(spl_xy, np.array(new_t))
        new_fields = _eval_spline(spl_f, np.array(new_t))
        if x_feats is not None:
            new_xf = np.zeros((n_dots, x_feats.shape[1]), dtype=np.float32)
            for k, id_ in enumerate(new_ids):
                if id_ in id2pos:
                    new_xf[k] = x_feats[id2pos[id_]]
                else:
                    a, b = inserted[id_]
                    new_xf[k] = 0.5 * (x_feats[id2pos[a]] + x_feats[id2pos[b]])
            return new_ids, new_pos, new_fields, new_xf
        return new_ids, new_pos, new_fields
    else:
        # remove points uniformly
        keep = _remove_points(row_ids, n - n_dots)
        keep_idx = [id2pos[id_] for id_ in keep]
        if x_feats is not None:
            return keep, pos[keep_idx], fields[keep_idx], x_feats[keep_idx]
        return keep, pos[keep_idx], fields[keep_idx]


# Per-sample processing


def _row_sizes(idx_path: str) -> List[int]:
    idx = torch.load(idx_path, weights_only=False)
    return [len(r) for r in idx['global_wing_rows']]


def _process_sample(args: tuple) -> Tuple[str, str]:
    idx_path, graph_path, out_path, n_dots = args
    idx = torch.load(idx_path, weights_only=False)
    rows = idx['global_wing_rows'][:N_ROW]

    # symmetric trim of each row to nl (length of the last kept row)
    nl = len(rows[-1])
    trimmed = []
    for r in rows:
        nk = len(r)
        if nk < nl:
            return ('bad_row', os.path.basename(idx_path))
        cut_start = (nk - nl) // 2 + (nk - nl) % 2
        cut_end = (nk - nl) // 2
        trimmed.append(r[cut_start: nk - cut_end])
    assert all(len(r) == nl for r in trimmed)

    # graph data (global id == row index in pos/x/y)
    g = torch.load(graph_path, weights_only=False)
    pos_all = g['pos'].numpy()
    x_all = g['x'].numpy()      # (N, 5)
    y_all = g['y'].numpy()      # (N, 4)

    # new global ids continue after the last graph node
    counter = [int(pos_all.shape[0])]

    def next_id():
        counter[0] += 1
        return counter[0]

    ids_mat, pos_mat, xf_mat, yf_mat = [], [], [], []
    for r in trimmed:
        ids = [int(i) for i in r]
        pos = pos_all[np.asarray(r)]
        xf = x_all[np.asarray(r)]
        yf = y_all[np.asarray(r)]
        new_ids, new_pos, new_yf, new_xf = interp_fun(ids, pos, yf, n_dots, next_id, xf)
        ids_mat.append(new_ids)
        pos_mat.append(new_pos)
        xf_mat.append(new_xf)
        yf_mat.append(new_yf)

    ids_mat = np.array(ids_mat, dtype=np.int64)        # (N_ROW, n_dots)
    pos_mat = np.array(pos_mat, dtype=np.float32)      # (N_ROW, n_dots, 2)
    xf_mat = np.array(xf_mat, dtype=np.float32)        # (N_ROW, n_dots, 5)
    yf_mat = np.array(yf_mat, dtype=np.float32)        # (N_ROW, n_dots, 4)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        'name': idx['name'],
        'ids': ids_mat,
        'pos': pos_mat,
        'x_feats': xf_mat,
        'fields': yf_mat,
        'field_names': FIELD_NAMES,
        'vx_in': float(x_all[0, 0]), 'vy_in': float(x_all[0, 1]),
        'n_dots': n_dots, 'N_ROW': N_ROW,
    }, out_path)
    return ('ok', os.path.basename(out_path))


def generate_geofno_dataset(index_root: str, graph_root: str, out_root: str,
                            n_row: int = N_ROW,
                            min_rows: int = MIN_ROWS, max_rows: int = MAX_ROWS,
                            threshold_frac: float = THRESHOLD_FRAC,
                            workers: int = 8,
                            resume: bool = True,
                            verbose: bool = True) -> dict:
    """
    Build the Geo-FNO C-grid dataset from index_raw_data_w.

    index_root : root with batch_N/ index files (from index.py)
    graph_root : root with graph_airfrans_data_batch_{N} raw graphs
    out_root   : where to write Geo-FNO_data/batch_N/*.pth

    Returns {'saved': int, 'filtered': int, 'bad_row': int, 'n_dots': int}
    """
    global N_ROW
    N_ROW = n_row

    batch_dirs = sorted([d for d in os.listdir(index_root) if d.startswith('batch_')],
                        key=lambda x: int(x.split('_')[1]))

    samples = []  # (idx_path, graph_path, out_path)
    for bd in batch_dirs:
        bn = int(bd.split('_')[1])
        idx_batch = os.path.join(index_root, bd)
        graph_batch = os.path.join(graph_root, f'graph_airfrans_data_batch_{bn}')
        for f in sorted(os.listdir(idx_batch)):
            if f.endswith('.pth'):
                samples.append((os.path.join(idx_batch, f),
                                os.path.join(graph_batch, f),
                                os.path.join(out_root, bd, f)))

    # Pass 1: filters + nl per sample (index files only)
    nls, skip_reasons = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for sp, sizes in zip(samples, ex.map(_row_sizes, [s[0] for s in samples])):
            if not (min_rows <= len(sizes) <= max_rows):
                skip_reasons.append(('rows_out_of_range', sp[0], len(sizes)))
            elif len(sizes) < n_row:
                skip_reasons.append(('too_few_rows', sp[0], len(sizes)))
            elif sizes[n_row - 1] < threshold_frac * sizes[0]:
                skip_reasons.append(('row_N_ROW_small', sp[0],
                                     f'{sizes[n_row - 1]} < {threshold_frac:.0%} of {sizes[0]}'))
            else:
                nls.append(sizes[n_row - 1])

    nls = np.array(nls)
    n_dots = int(np.median(nls)) if len(nls) else 0
    if verbose:
        print(f'Filtered out: {len(skip_reasons)} | kept: {len(nls)}')
        print(f'n_dots (median of nl): {n_dots}')

    # Pass 2: resample to n_dots and save
    skipped_paths = {p for _, p, _ in skip_reasons}
    tasks = [(idx_p, gr_p, out_p, n_dots)
             for idx_p, gr_p, out_p in samples if idx_p not in skipped_paths]
    if resume:
        tasks = [t for t in tasks if not os.path.exists(t[2])]

    results = {'ok': 0, 'bad_row': 0}
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for status, fname in ex.map(_process_sample, tasks):
                results[status] = results.get(status, 0) + 1
                if verbose and (results['ok'] + results['bad_row']) % 50 == 0:
                    print(f'  {results["ok"] + results["bad_row"]}/{len(tasks)}')

    if verbose:
        print(f'Saved: {results["ok"]} | bad_row: {results["bad_row"]}')
        print(f'Output: {out_root}')
    return {'saved': results['ok'], 'filtered': len(skip_reasons),
            'bad_row': results['bad_row'], 'n_dots': n_dots}


def main():
    parser = argparse.ArgumentParser(
        description='Geo-FNO dataset preparation: rows -> unfolded C-grid.')
    parser.add_argument('--index_root', type=str, required=True,
                        help='index_raw_data_w root (output of index.py)')
    parser.add_argument('--graph_root', type=str, required=True,
                        help='Root containing graph_airfrans_data_batch_{1..10}')
    parser.add_argument('--out_root', type=str, required=True,
                        help='Where to write Geo-FNO_data')
    parser.add_argument('--n_row', type=int, default=N_ROW)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--no_resume', action='store_true',
                        help='Overwrite already existing files')
    args = parser.parse_args()

    result = generate_geofno_dataset(
        index_root=args.index_root,
        graph_root=args.graph_root,
        out_root=args.out_root,
        n_row=args.n_row,
        workers=args.workers,
        resume=not args.no_resume,
    )
    print(result)


if __name__ == '__main__':
    main()

"""
prep_dno.py - DNO data preparation (AirfRANS).

Two stages, both consuming the common index stage (index_raw_data_w, index.py):

Stage A (zonal row resampling) - ported from DNO/data_prep.ipynb:
    index_raw_data_w + raw graphs -> DNO_data/batch_N/*.pth
    - filter samples by number of full rows in [MIN_FULL, MAX_FULL]
    - keep rows[:N_ROW], thin them per zone (ZONES config)
    - walk each row: add/remove points based on inter-row distance ratios
      (new global ids generated via splines)
    - wall (inner boundary) becomes "row -1" with zone-1 treatment
    Output dict: pos, x, y, row_id, node_id, global_wing_rows,
                 ordered_wall_indices, kept, stats

Stage B (diffeomorphism) - ported from gen_dno_small_all.py + phys_dif.py:
    DNO_data/batch_N/*.pth -> dno_small/batch_N/*.npz
    - Delaunay triangulation filtered by wall and outer row
    - remove isolated nodes
    - Dirichlet BCs: wall -> circle (TE=0, LE=pi), outer row -> square,
      row banks -> right side
    - cotangent Laplacian -> harmonic map (xi, eta)
    - barycentric sampling on a GRID x GRID regular grid (cKDTree candidates)
    - mask = outside the central hole (circle of radius RADIUS)
    - interpolate 5 input features + 4 output fields
    Output npz: grid_x, grid_y, input [G,G,5], output [G,G,4], mask [G,G]

Usage:
    python -m experiments.airfoils.data_prep.prep_dno --stage a \\
        --index_root /media/.../airfrans/index_raw_data_w \\
        --graph_root /media/.../airfrans \\
        --dno_root   /media/.../airfrans/DNO_data
    python -m experiments.airfoils.data_prep.prep_dno --stage b \\
        --dno_root  /media/.../airfrans/DNO_data \\
        --out_root  /media/.../airfrans/DNO_data/dno_small
    python -m experiments.airfoils.data_prep.prep_dno --stage all ...
"""

import os
import math
import time
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree, Delaunay
from matplotlib.path import Path


# Stage A: zonal row resampling

N_ROW = 137              # rows to take (0..136)
MIN_FULL = 137           # sample filter: at least this many full rows
MAX_FULL = 180           # ... and at most this many
FULL_FRAC = 0.9          # a row is "full" if it has >= 90% of first row points

RATIO_ADD = 2.0          # add a point if gap > RATIO_ADD * d
RATIO_REM = 1.0          # drop a node if gap < RATIO_REM * d
WALL_MIN_D = 1e-4        # rows closer than this to the wall are "coincident"
DROP_ROWS = [0, 45, 57]  # rows removed entirely after processing

# Zones (0-INDEXED rows). mode:
#   'rows+add'    - thin rows (rows_step) + add points (ratio_add)
#   'rows_only'   - thin rows only
#   'none'        - leave as is
#   'points_only' - only remove points (ratio_rem)
ZONES = [
    {'r0': 0,   'r1': 44,  'mode': 'rows+add',    'rows_step': 35, 'ratio_add': 2.0, 'ratio_rem': 0.0, 'keep_end': True},
    {'r0': 45,  'r1': 57,  'mode': 'rows+add',    'rows_step': 5,  'ratio_add': 2.0, 'ratio_rem': 0.0, 'keep_end': True},
    {'r0': 58,  'r1': 75,  'mode': 'rows+add',    'rows_step': 2,  'ratio_add': 2.0, 'ratio_rem': 0.0, 'keep_end': True},
    {'r0': 76,  'r1': 84,  'mode': 'rows+add',    'rows_step': 1,  'ratio_add': 2.0, 'ratio_rem': 0.0, 'keep_end': True},
    {'r0': 85,  'r1': 112, 'mode': 'points_only', 'rows_step': 1,  'ratio_add': 0.0, 'ratio_rem': 1.0, 'keep_end': False},
    {'r0': 113, 'r1': 137, 'mode': 'points_only', 'rows_step': 1,  'ratio_add': 0.0, 'ratio_rem': 0.5, 'keep_end': False},
]

FIELD_NAMES = ['vel_x', 'vel_y', 'pressure', 'nut']
XFEAT_NAMES = ['vx_in', 'vy_in', 'dist', 'norm_x', 'norm_y']


def zone_cfg(k: int, zones: List[dict] = ZONES) -> Optional[dict]:
    """Config of the zone containing row k (0-indexed), or None."""
    for z in zones:
        if z['r0'] <= k <= z['r1']:
            return z
    return None


def select_rows(n_row: int = N_ROW, zones: List[dict] = ZONES) -> List[int]:
    """Rows (0-indexed) kept after zone thinning.

    In 'rows+add'/'rows_only' zones every rows_step-th row is kept;
    the first zone row is always kept, and with keep_end=True the last one too.
    """
    kept = []
    for k in range(n_row):
        z = zone_cfg(k, zones)
        if z is None:
            kept.append(k)
            continue
        if z['mode'] in ('rows+add', 'rows_only') and z['rows_step'] > 1:
            rel = (k - z['r0']) % z['rows_step']
            if rel != 0 and not (z.get('keep_end', False) and k == min(z['r1'], n_row - 1)):
                continue
        kept.append(k)
    return kept


def _row_splines(t, pos, fields, x_feats):
    spl_x = CubicSpline(t, pos[:, 0])
    spl_y = CubicSpline(t, pos[:, 1])
    spl_f = [CubicSpline(t, fields[:, f]) for f in range(fields.shape[1])]
    spl_xf = [CubicSpline(t, x_feats[:, f]) for f in range(x_feats.shape[1])]
    return spl_x, spl_y, spl_f, spl_xf


def _eval_row(spl, tt):
    spl_x, spl_y, spl_f, spl_xf = spl
    xy = np.stack([spl_x(tt), spl_y(tt)], axis=-1).astype(np.float32)
    f = np.stack([s(tt) for s in spl_f], axis=-1).astype(np.float32)
    xf = np.stack([s(tt) for s in spl_xf], axis=-1).astype(np.float32)
    return xy, f, xf


def resample_row_walk(ids, pos, fields, x_feats, d_arr, next_id=None,
                      ratio_add: float = RATIO_ADD,
                      ratio_rem: float = RATIO_REM):
    """Walk along a row (zonal version).

    ids     : global ids of the row nodes
    pos     : (n, 2) coordinates
    fields  : (n, 4) fields (vel_x, vel_y, pressure, nut)
    x_feats : (n, 5) graph input features
    d_arr   : (n,) inter-row distance at each node (to the next kept row;
              last row - to the previous one)
    ratio_add : if the gap to the next node > ratio_add * d, insert points
                spaced by d (0 = never add)
    ratio_rem : if the gap to the next node < ratio_rem * d, drop the node
                (0 = never drop)

    Returns (new_ids, new_pos, new_fields, new_xfeats, n_new).
    """
    n = len(pos)
    if n < 2:
        return list(ids), pos, fields, x_feats, 0

    diff = np.diff(pos, axis=0)
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diff, axis=1))])
    # guard: coincident points create duplicate t -> spline fails;
    # keep strictly increasing t, drop duplicates
    keep = np.concatenate([[True], np.diff(t) > 0])
    if not keep.all():
        t = t[keep]
        pos = pos[keep]
        fields = fields[keep]
        x_feats = x_feats[keep]
        d_arr = d_arr[keep]
        ids = [ids[i] for i in range(len(ids)) if keep[i]]
        n = len(pos)
        if n < 2:
            return list(ids), pos, fields, x_feats, 0
    spl = _row_splines(t, pos, fields, x_feats)

    def dd(k):
        v = float(d_arr[k])
        return v if v > 0 else 1e-6

    out_ids = [int(ids[0])]
    out_pos = [pos[0][None].astype(np.float32)]
    out_f = [fields[0][None].astype(np.float32)]
    out_xf = [x_feats[0][None].astype(np.float32)]

    node, cur, d, n_new = 0, 0.0, dd(0), 0
    while node < n - 1:
        g = t[node + 1] - cur
        if ratio_add > 0 and g > ratio_add * d:
            # insert new points spaced by d (coords/fields via spline,
            # x-feats as the mean of neighbors)
            m = int(math.floor(g / d - 1))
            if m > 0:
                tt = cur + d * np.arange(1, m + 1)
                xy, f, _ = _eval_row(spl, tt)
                xf = np.tile(0.5 * (x_feats[node] + x_feats[node + 1]), (m, 1)).astype(np.float32)
                out_pos.append(xy)
                out_f.append(f)
                out_xf.append(xf)
                if next_id is not None:
                    out_ids.extend(next_id() for _ in range(m))
                else:
                    out_ids.extend([-1] * m)
                n_new += m
                cur = float(tt[-1])
            g = t[node + 1] - cur
        # decide the fate of the next node
        if g < ratio_rem * d and node + 1 < n - 1:
            node += 1                       # node too close - drop it
            d = dd(node)
        else:
            node += 1                       # keep the node
            out_ids.append(int(ids[node]))
            out_pos.append(pos[node][None].astype(np.float32))
            out_f.append(fields[node][None].astype(np.float32))
            out_xf.append(x_feats[node][None].astype(np.float32))
            cur = t[node]
            d = dd(node)

    out_ids = np.asarray(out_ids, dtype=np.int64)
    out_pos = np.concatenate(out_pos)
    out_f = np.concatenate(out_f)
    out_xf = np.concatenate(out_xf)
    return list(out_ids), out_pos, out_f, out_xf, n_new


def _n_full_rows(idx) -> int:
    sizes = [len(r) for r in idx['global_wing_rows']]
    if not sizes:
        return 0
    thr = FULL_FRAC * sizes[0]
    return sum(1 for s in sizes if s >= thr)


def process_sample_resample(idx, g, zones: List[dict] = ZONES, n_row: int = N_ROW,
                            wall_min_d: float = WALL_MIN_D,
                            drop_rows: List[int] = DROP_ROWS):
    """Transform one sample: zone thinning -> zonal walk -> wall ("row -1").

    idx : index dict (global_wing_rows, ordered_wall_indices)
    g   : raw graph (pos, x, y)

    Returns (rows_data, kept, n_new, info):
      rows_data : list[dict] - {'row', 'ids', 'pos', 'fields', 'xfeats'}
                  (row = -1 is the wall, 0..n_row-1 are rows)
      kept      : list[int]  - kept rows (0-indexed)
      n_new     : int        - number of new global ids created
      info      : list[dict] - per-object stats
    """
    rows = idx['global_wing_rows'][:n_row]
    pos_all = g['pos'].numpy()
    x_all = g['x'].numpy()
    y_all = g['y'].numpy()

    kept = [k for k in select_rows(n_row, zones) if k not in drop_rows]

    counter = [int(pos_all.shape[0])]

    def next_id():
        counter[0] += 1
        return counter[0]

    rows_data, info = [], []

    def _intra(pos):
        return float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1))) if len(pos) >= 2 else float('nan')

    # wall (inner boundary) - "row -1", zone-1 mode
    # wall node order differs from row order -> d via nearest neighbors
    # (cKDTree). The neighbor is the first kept row NOT coincident with the
    # wall (closer than wall_min_d rows are skipped).
    wall_ids = [int(i) for i in idx['ordered_wall_indices']]
    wpos = pos_all[wall_ids].astype(np.float64)
    nbr, w_d_arr = kept[0], None
    for kk0 in kept:
        pn = pos_all[np.asarray(rows[kk0])].astype(np.float64)
        dd = cKDTree(pn).query(wpos, k=1)[0]
        if np.median(dd) > wall_min_d:
            nbr, w_d_arr = kk0, dd
            break
    if w_d_arr is None:
        nbr = kept[-1]
        w_d_arr = cKDTree(pos_all[np.asarray(rows[nbr])].astype(np.float64)).query(wpos, k=1)[0]
    z1 = zone_cfg(kept[0], zones)
    if z1['mode'] in ('none', 'rows_only'):
        w_ids, w_pos, w_f, w_xf = (wall_ids, pos_all[wall_ids].astype(np.float32),
                                   y_all[wall_ids].astype(np.float32),
                                   x_all[wall_ids].astype(np.float32))
    else:
        ra = z1['ratio_add'] if z1['mode'] == 'rows+add' else 0.0
        rr = z1['ratio_rem'] if z1['mode'] == 'points_only' else 0.0
        w_ids, w_pos, w_f, w_xf, _ = resample_row_walk(
            wall_ids, pos_all[wall_ids], y_all[wall_ids], x_all[wall_ids], w_d_arr,
            next_id=next_id, ratio_add=ra, ratio_rem=rr)
    rows_data.append({'row': -1, 'ids': w_ids, 'pos': w_pos, 'fields': w_f, 'xfeats': w_xf})
    info.append({'row': -1, 'ref': nbr, 'pts0': len(wall_ids), 'pts1': len(w_ids),
                 'intra_med': _intra(w_pos), 'inter_med': float(np.median(w_d_arr))})

    # rows
    for j, k in enumerate(kept):
        z = zone_cfg(k, zones)
        ids = [int(i) for i in rows[k]]
        pos = pos_all[np.asarray(rows[k])]
        xf = x_all[np.asarray(rows[k])]
        yf = y_all[np.asarray(rows[k])]

        d_arr, ref_k = None, None
        if z['mode'] == 'none':
            new_ids, new_pos, new_yf, new_xf = (ids, pos.astype(np.float32),
                                                yf.astype(np.float32), xf.astype(np.float32))
        else:
            # reference row for d: start/middle of zone -> next kept row in the
            # SAME zone; end of zone -> previous one (never across zones)
            if j + 1 < len(kept) and zone_cfg(kept[j + 1], zones) is z:
                ref_k = kept[j + 1]
            elif j - 1 >= 0 and zone_cfg(kept[j - 1], zones) is z:
                ref_k = kept[j - 1]
            else:
                ref_k = kept[j + 1] if j + 1 < len(kept) else kept[j - 1]
            pn = pos_all[np.asarray(rows[ref_k])]
            kk = np.minimum(np.arange(len(ids)), len(pn) - 1)
            d_arr = np.linalg.norm(pos - pn[kk], axis=1)
            if z['mode'] == 'rows_only':
                new_ids, new_pos, new_yf, new_xf = (ids, pos.astype(np.float32),
                                                    yf.astype(np.float32), xf.astype(np.float32))
            else:
                ra = z['ratio_add'] if z['mode'] == 'rows+add' else 0.0
                rr = z['ratio_rem'] if z['mode'] == 'points_only' else 0.0
                new_ids, new_pos, new_yf, new_xf, _ = resample_row_walk(
                    ids, pos, yf, xf, d_arr, next_id=next_id, ratio_add=ra, ratio_rem=rr)

        rows_data.append({'row': k, 'ids': new_ids, 'pos': new_pos,
                          'fields': new_yf, 'xfeats': new_xf})
        info.append({'row': k, 'ref': ref_k, 'pts0': len(ids), 'pts1': len(new_ids),
                     'intra_med': _intra(new_pos),
                     'inter_med': float(np.median(d_arr)) if d_arr is not None else float('nan')})

    return rows_data, kept, counter[0] - pos_all.shape[0], info


def _process_one_resample(args):
    batch, fname, index_root, graph_root, dno_root, apply_filter = args
    try:
        ipath = os.path.join(index_root, batch, fname)
        gpath = os.path.join(graph_root, f'graph_airfrans_data_batch_{int(batch.split("_")[1])}', fname)
        idx = torch.load(ipath, weights_only=False)
        g = torch.load(gpath, weights_only=False)

        nf = _n_full_rows(idx)
        if apply_filter and not (MIN_FULL <= nf <= MAX_FULL):
            return (fname, 'skipped', {'n_full': nf})

        rows_data, kept, n_new, info = process_sample_resample(idx, g)

        pos_parts, x_parts, y_parts, rid_parts, nid_parts = [], [], [], [], []
        rows_out, wall_out = [], None
        for r in rows_data:
            pos_parts.append(r['pos'])
            x_parts.append(r['xfeats'])
            y_parts.append(r['fields'])
            rid_parts.append(np.full(len(r['pos']), r['row'], dtype=np.int32))
            nid_parts.append(np.asarray(r['ids'], dtype=np.int64))
            if r['row'] == -1:
                wall_out = list(r['ids'])
            else:
                rows_out.append(list(r['ids']))

        out = {
            'name': fname,
            'pos': torch.from_numpy(np.concatenate(pos_parts).astype(np.float32)),
            'x': torch.from_numpy(np.concatenate(x_parts).astype(np.float32)),
            'y': torch.from_numpy(np.concatenate(y_parts).astype(np.float32)),
            'row_id': torch.from_numpy(np.concatenate(rid_parts)),
            'node_id': torch.from_numpy(np.concatenate(nid_parts)),
            'global_wing_rows': rows_out,
            'ordered_wall_indices': wall_out,
            'kept': kept,
            'stats': {'n_full_rows': nf, 'n_points': sum(len(p) for p in pos_parts),
                      'n_new_ids': n_new, 'batch': batch},
        }

        odir = os.path.join(dno_root, batch)
        os.makedirs(odir, exist_ok=True)
        tmp = os.path.join(odir, fname + '.tmp')
        torch.save(out, tmp)
        os.replace(tmp, os.path.join(odir, fname))
        return (fname, 'ok', out['stats'])
    except Exception as e:
        return (fname, 'error', f'{type(e).__name__}: {e}')


def run_resample(index_root: str, graph_root: str, dno_root: str,
                 batches: Optional[List[str]] = None,
                 workers: int = 4,
                 apply_filter: bool = True,
                 resume: bool = True,
                 verbose: bool = True) -> dict:
    """Stage A: zonal resampling -> DNO_data/batch_N/*.pth."""
    os.makedirs(dno_root, exist_ok=True)
    if batches is None:
        batches = sorted([d for d in os.listdir(index_root) if d.startswith('batch_')],
                         key=lambda x: int(x.split('_')[1]))
    tasks = [(b, f) for b in batches
             for f in sorted(os.listdir(os.path.join(index_root, b)))
             if f.endswith('.pth')]
    if resume:
        tasks = [t for t in tasks
                 if not os.path.exists(os.path.join(dno_root, t[0], t[1]))]
    if verbose:
        print(f'Stage A: {len(tasks)} samples to process')

    results = []
    t0 = time.time()
    if workers and workers > 1 and tasks:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, res in enumerate(ex.map(_process_one_resample,
                                           [(b, f, index_root, graph_root, dno_root, apply_filter)
                                            for b, f in tasks], chunksize=4), 1):
                results.append(res)
                if verbose and i % 50 == 0:
                    print(f'  {i}/{len(tasks)} ({time.time() - t0:.0f}s)')
    else:
        for i, t in enumerate(tasks, 1):
            results.append(_process_one_resample((*t, index_root, graph_root, dno_root, apply_filter)))
            if verbose and i % 50 == 0:
                print(f'  {i}/{len(tasks)} ({time.time() - t0:.0f}s)')

    ok = [r for r in results if r[1] == 'ok']
    sk = [r for r in results if r[1] == 'skipped']
    er = [r for r in results if r[1] == 'error']
    if verbose:
        print(f'Stage A done in {time.time() - t0:.0f}s: ok={len(ok)}, '
              f'skipped(filter)={len(sk)}, errors={len(er)}')
        for f, _, e in er[:10]:
            print(f'  ERROR {f}: {e}')
    return {'ok': len(ok), 'skipped': len(sk), 'errors': len(er)}


# Stage B: diffeomorphism (square with hole)

GRID = 256                # regular grid resolution
RADIUS = 0.01             # hole radius in the universal square
CENTER = (0.5, 0.5)       # hole center
K = 256                   # triangle candidates per grid point


def load_sample(dno_root: str, batch: str, name: str):
    """Load a resampled sample from DNO_data.

    Returns (d, pos, wall_loc, rows):
      d        - dict from .pth (pos, x, y, row_id, node_id, ...)
      pos      - (N, 2) float64 physical coordinates
      wall_loc - local wall indices (global ids translated via node_id)
      rows     - list of rows (local indices); rows[-1] is the outer boundary
    """
    d = torch.load(os.path.join(dno_root, batch, name), weights_only=False)
    pos = d['pos'].numpy().astype(np.float64)
    node_id = d['node_id'].numpy()
    id2loc = {int(g): i for i, g in enumerate(node_id)}
    wall_loc = np.array([id2loc[g] for g in d['ordered_wall_indices']], dtype=int)
    rows_loc = [np.array([id2loc[g] for g in r], dtype=int) for r in d['global_wing_rows']]
    return d, pos, wall_loc, rows_loc


def mesh_boundary_nodes(faces: np.ndarray, n_nodes: int) -> np.ndarray:
    """Local indices of mesh boundary nodes (edges with a single triangle)."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    cnt = {}
    for e in map(tuple, edges):
        cnt[e] = cnt.get(e, 0) + 1
    bnd_edges = [e for e, c in cnt.items() if c == 1]
    return np.unique(np.array(bnd_edges)) if bnd_edges else np.array([], dtype=int)


def build_mesh_scipy(pos: np.ndarray, wall_loc: np.ndarray,
                     outer_loc: np.ndarray) -> np.ndarray:
    """Delaunay (scipy) + filter: drop triangles inside the profile and
    outside the outer row. Returns faces (M, 3) only."""
    tri = Delaunay(pos)
    faces = tri.simplices
    cent = pos[faces].mean(axis=1)
    in_wall = Path(pos[wall_loc]).contains_points(cent)
    in_outer = Path(pos[outer_loc]).contains_points(cent)
    keep = (~in_wall) & in_outer
    return faces[keep]


def isolated_nodes(faces: np.ndarray, n: int) -> np.ndarray:
    """Nodes not present in any triangle (outside the domain) - to remove."""
    present = np.zeros(n, dtype=bool)
    present[faces.ravel()] = True
    return np.nonzero(~present)[0]


def remove_points(d, pos, wall_loc, rows, remove, all_idx):
    """Remove nodes and everything referencing them; renumber indices.

    Returns (d, pos, wall_loc, rows, keep, node_id):
      keep    - indices of remaining nodes in the old numbering
      node_id - new local ids (0..N-1)
    """
    keep = np.setdiff1d(all_idx, remove)
    new_idx = np.full(len(pos), -1, dtype=int)
    new_idx[keep] = np.arange(len(keep))

    wall_loc_new = new_idx[wall_loc]
    rows_new = []
    keep_b = np.zeros(len(pos), dtype=bool)
    keep_b[keep] = True
    for r in rows:
        r = r[keep_b[r]]
        if len(r):
            rows_new.append(new_idx[r])

    pos = pos[keep]
    d['pos'] = torch.from_numpy(pos.astype(np.float32))
    d['x'] = d['x'][keep]
    d['y'] = d['y'][keep]
    d['row_id'] = d['row_id'][keep]
    node_id = new_idx[keep]
    d['node_id'] = torch.from_numpy(node_id.astype(np.int64))
    d['global_wing_rows'] = [r.tolist() for r in rows_new]
    d['ordered_wall_indices'] = wall_loc_new.tolist()
    return d, pos, wall_loc_new, rows_new, keep, node_id


def _arc_param(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc length normalized to [0, 1]."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if s[-1] == 0:
        return np.zeros(len(pts))
    return s / s[-1]


def wall_to_circle(wall_loc, pos, center=CENTER, radius=RADIUS):
    """Wall -> circle: TE (1st point) at angle 0, clockwise;
    LE (min x) at pi. Inside segments - by arc length."""
    wp = pos[wall_loc]
    i_le = int(np.argmin(wp[:, 0]))
    s = _arc_param(wp)
    s_le = s[i_le]
    ang = np.where(s <= s_le,
                   -np.pi * s / s_le,                            # TE->LE: 0 -> -pi
                   -np.pi - np.pi * (s - s_le) / (1.0 - s_le))   # LE->TE: -pi -> -2pi
    return np.column_stack([center[0] + radius * np.cos(ang),
                            center[1] + radius * np.sin(ang)])


def row_to_square(outer_loc, pos):
    """Last row -> square perimeter (1,0)->(0,0)->(0,1)->(1,1), by arc length."""
    t = _arc_param(pos[outer_loc])
    d = t * 3.0
    uv = np.empty((len(outer_loc), 2))
    m1 = d < 1.0                                        # bottom: (1,0)->(0,0)
    m2 = (d >= 1.0) & (d < 2.0)                         # left: (0,0)->(0,1)
    m3 = d >= 2.0                                       # top: (0,1)->(1,1)
    uv[m1] = np.column_stack([1.0 - d[m1], np.zeros(m1.sum())])
    uv[m2] = np.column_stack([np.zeros(m2.sum()), d[m2] - 1.0])
    uv[m3] = np.column_stack([d[m3] - 2.0, np.ones(m3.sum())])
    return uv


def banks_to_right(rows, pos):
    """Row ends -> right side (1,1)->(1,0) top to bottom, by arc length."""
    seq = [rows[i][-1] for i in range(len(rows) - 1, -1, -1)]
    seq.append(rows[0][0])
    seq += [rows[i][0] for i in range(1, len(rows))]
    seq = np.array(seq, dtype=int)
    t = _arc_param(pos[seq])
    return seq, np.column_stack([np.ones(len(seq)), 1.0 - t])


def build_bcs(pos, wall_loc, rows, center=CENTER, radius=RADIUS):
    """Dirichlet BCs for all our points. Returns (bnd_idx, bnd_uv).

    Guarantees:
      - row ends (banks) are always x = 1 (right side of the square)
      - outer row (rows[-1]): first point exactly (1, 0),
        last point exactly (1, 1) - corners pinned explicitly
    """
    outer = rows[-1]
    uv_wall = wall_to_circle(wall_loc, pos, center, radius)
    uv_outer = row_to_square(outer, pos)
    seq, uv_right = banks_to_right(rows, pos)
    bnd_idx = np.concatenate([wall_loc, outer, seq])
    bnd_uv = np.concatenate([uv_wall, uv_outer, uv_right])
    i0 = len(wall_loc)                        # outer[0] in bnd_idx
    i1 = len(wall_loc) + len(outer) - 1       # outer[-1] in bnd_idx
    bnd_uv[i0] = [1.0, 0.0]
    bnd_uv[i1] = [1.0, 1.0]
    return bnd_idx, bnd_uv


def extra_boundary_bcs(node_pos, faces, our_idx, wall_loc, outer_loc, rows,
                       center=CENTER, radius=RADIUS):
    """Boundary nodes of the triangulation without Dirichlet BCs get the
    BC of their nearest node from the full set (wall/outer row/banks).
    Returns (idx, uv)."""
    bnd_nodes = mesh_boundary_nodes(faces, len(node_pos))
    our = set(np.asarray(our_idx).tolist())
    extra = np.array([n for n in bnd_nodes if n not in our], dtype=int)
    if len(extra) == 0:
        return np.array([], dtype=int), np.empty((0, 2))
    bnd_all, bnd_all_uv = build_bcs(node_pos, wall_loc, rows, center, radius)
    tree = cKDTree(node_pos[bnd_all])
    _, nn = tree.query(node_pos[extra])
    return extra, bnd_all_uv[nn]


def _cotangents(X, faces):
    """Raw cotangents (M, 3): [cot(angle at v0), cot(at v1), cot(at v2)]."""
    v0, v1, v2 = X[faces[:, 0]], X[faces[:, 1]], X[faces[:, 2]]

    def cotan(u, v):
        dot = np.sum(u * v, axis=1)
        cross = np.maximum(np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]), 1e-12)
        return dot / cross

    return np.column_stack([cotan(v1 - v0, v2 - v0),
                            cotan(v2 - v1, v0 - v1),
                            cotan(v0 - v2, v1 - v2)])


def build_cotangent_laplacian(X, faces, clamp_negative=True, return_raw=False):
    """Cotangent Laplacian on the triangulation (L * u = 0).

    clamp_negative=True - zero out negative weights (stability).
    return_raw=True     - also return raw cotangents (M, 3).
    """
    import scipy.sparse as sp
    from scipy.sparse import coo_matrix, diags
    n = X.shape[0]
    cots = _cotangents(X, faces)
    raw = cots.copy() if return_raw else None
    if clamp_negative:
        cots = np.maximum(cots, 0)
    rows, cols, data = [], [], []
    for k, c in enumerate(cots.T):
        i = faces[:, (k + 1) % 3]
        j = faces[:, (k + 2) % 3]
        w = c * 0.5
        rows += list(i)
        cols += list(j)
        data += list(w)
        rows += list(j)
        cols += list(i)
        data += list(w)
    W = coo_matrix((data, (rows, cols)), shape=(n, n))
    deg = np.array(W.sum(axis=1)).flatten()
    L = (sp.diags(deg) - W).tocsr()
    return (L, raw) if return_raw else L


def solve_harmonic_map(L, bnd_idx, bnd_uv, eps=1e-12):
    """Solve Laplace for xi and eta with Dirichlet BCs. Returns uv (N, 2).

    NOTE: bnd_idx is deduplicated (np.unique). Without this, nodes that
    appear twice in the boundary set (e.g. square corners from the outer
    row AND from the bank chain) would be counted twice in the rhs and the
    harmonic map would leave the square (uv > 1).
    """
    from scipy.sparse.linalg import spsolve
    import scipy.sparse as sp
    n = L.shape[0]
    bnd, uniq_idx = np.unique(np.asarray(bnd_idx, dtype=int), return_index=True)
    bnd_uv = np.asarray(bnd_uv, dtype=float)[uniq_idx]
    inner = np.setdiff1d(np.arange(n), bnd)
    Lii = L[inner, :][:, inner].tocsr() + eps * sp.eye(len(inner), format='csr')
    Lib = L[inner, :][:, bnd].tocsr()
    rhs = -Lib.dot(bnd_uv)
    sol = spsolve(Lii, rhs)                 # (n_inner, 2)
    uv = np.zeros((n, 2))
    uv[inner] = sol
    uv[bnd] = bnd_uv
    return uv


def process_sample_diffeo(dno_root: str, batch: str, name: str,
                          out_root: str, grid: int = GRID,
                          radius: float = RADIUS, center: Tuple[float, float] = CENTER,
                          k_cand: int = K, write_vtk: bool = False) -> dict:
    """Full Stage B pipeline for one resampled sample -> dno_small npz."""
    t0 = time.time()

    d, pos, wall_loc, rows = load_sample(dno_root, batch, name)
    faces = build_mesh_scipy(pos, wall_loc, rows[-1])
    iso = isolated_nodes(faces, len(pos))
    d, pos, wall_loc, rows, keep, _ = remove_points(
        d, pos, wall_loc, rows, iso, np.arange(len(pos)))
    faces = build_mesh_scipy(pos, wall_loc, rows[-1])

    bnd_idx, bnd_uv = build_bcs(pos, wall_loc, rows, radius=radius)
    eidx, euv = extra_boundary_bcs(pos, faces, bnd_idx, wall_loc, rows[-1], rows,
                                   radius=radius)
    if len(eidx):
        bnd_idx = np.concatenate([bnd_idx, eidx])
        bnd_uv = np.concatenate([bnd_uv, euv])
    L, raw = build_cotangent_laplacian(pos, faces, return_raw=True)
    uv = solve_harmonic_map(L, bnd_idx, bnd_uv)

    # regular grid: barycentric sampling
    g = np.linspace(0.0, 1.0, grid)
    Xi, Eta = np.meshgrid(g, g)
    grid_pts = np.column_stack([Xi.ravel(), Eta.ravel()])

    cent = uv[faces].mean(axis=1)
    tree = cKDTree(cent)
    _, cand = tree.query(grid_pts, k=k_cand)

    f0, f1, f2 = faces[cand, 0], faces[cand, 1], faces[cand, 2]
    t0v = uv[f0]
    t1 = uv[f1] - t0v
    t2 = uv[f2] - t0v
    p = grid_pts[:, None, :] - t0v
    cr = t1[..., 0] * t2[..., 1] - t1[..., 1] * t2[..., 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        lam1 = (p[..., 0] * t2[..., 1] - p[..., 1] * t2[..., 0]) / cr
        lam2 = (t1[..., 0] * p[..., 1] - t1[..., 1] * p[..., 0]) / cr
        lam0 = 1.0 - lam1 - lam2
    inside = (lam0 >= -1e-9) & (lam1 >= -1e-9) & (lam2 >= -1e-9) & (np.abs(cr) > 1e-14)
    hit = inside.argmax(axis=1)
    ok = inside.any(axis=1)
    G = len(grid_pts)
    gg = np.arange(G)
    lam = np.stack([lam0[gg, hit], lam1[gg, hit], lam2[gg, hit]], axis=-1)

    miss = ~ok
    if miss.any():
        ntree = cKDTree(uv)
        _, nn = ntree.query(grid_pts[miss])
        lam[miss] = np.array([1.0, 0.0, 0.0])
        f0[gg[miss], hit[miss]] = nn
        f1[gg[miss], hit[miss]] = nn
        f2[gg[miss], hit[miss]] = nn

    phys = (lam[:, 0, None] * pos[f0[gg, hit]] +
            lam[:, 1, None] * pos[f1[gg, hit]] +
            lam[:, 2, None] * pos[f2[gg, hit]])

    hole = np.linalg.norm(grid_pts - center, axis=1) < radius + 1e-9
    mask = (~hole).reshape(grid, grid)

    grid_x = phys[:, 0].reshape(grid, grid)
    grid_y = phys[:, 1].reshape(grid, grid)
    grid_x[~mask] = np.nan
    grid_y[~mask] = np.nan

    # interpolate fields/features
    Xf = d['x'].numpy()
    Yf = d['y'].numpy()

    def interp_fields(V):
        return (lam[:, 0, None] * V[f0[gg, hit]] +
                lam[:, 1, None] * V[f1[gg, hit]] +
                lam[:, 2, None] * V[f2[gg, hit]])

    fields_x = interp_fields(Xf).reshape(grid, grid, 5)
    fields_y = interp_fields(Yf).reshape(grid, grid, 4)
    fields_x[~mask] = np.nan
    fields_y[~mask] = np.nan

    out_dir = os.path.join(out_root, batch)
    os.makedirs(out_dir, exist_ok=True)
    m = re_search_number(name)
    npz_path = os.path.join(out_dir, f'graph_airfrans_data_{m}_dno_small.npz')
    np.savez_compressed(npz_path,
                        grid_x=np.nan_to_num(grid_x), grid_y=np.nan_to_num(grid_y),
                        input=np.nan_to_num(fields_x),
                        output=np.nan_to_num(fields_y),
                        mask=mask.astype(np.float32))

    if write_vtk:
        write_vtk_sample(out_dir, name, m, pos, faces, uv, d, Yf, raw)

    dt = time.time() - t0
    return {'batch': batch, 'sample': m, 'nodes': len(pos), 'tris': len(faces),
            'coverage': 100.0 * ok.mean(), 'fallback': int(miss.sum()),
            'hole': int(hole.sum()), 'time': dt, 'npz': npz_path}


def re_search_number(name: str) -> int:
    import re
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else 0


def write_vtk_tri(path, pts, faces, pscal, cscal):
    """VTK POLYDATA: POINTS + POLYGONS, POINT_DATA (pscal), CELL_DATA (cscal)."""
    n = len(pts)
    m = len(faces)

    def dump(f, values, chunk=200_000):
        buf = []
        for v in values:
            buf.append(v)
            if len(buf) >= chunk:
                f.write(''.join(buf))
                buf = []
        f.write(''.join(buf))

    def dump_faces(f, chunk=50_000):
        buf = []
        for t in faces:
            buf.append(f'3 {t[0]} {t[1]} {t[2]}\n')
            if len(buf) >= chunk:
                f.write(''.join(buf))
                buf = []
        f.write(''.join(buf))

    def dump_scalars(f, arr, chunk=200_000):
        arr = np.asarray(arr)
        if arr.dtype.kind == 'f':
            dump(f, (f'{v:.6f}\n' for v in arr), chunk)
        else:
            dump(f, (f'{int(v)}\n' for v in arr), chunk)

    with open(path, 'w') as f:
        f.write('# vtk DataFile Version 3.0\nDNO diffeo (direct)\nASCII\nDATASET POLYDATA\n')
        f.write(f'POINTS {n} float\n')
        dump(f, (f'{p[0]:.6f} {p[1]:.6f} 0.0\n' for p in pts))
        f.write(f'POLYGONS {m} {m * 4}\n')
        dump_faces(f)
        f.write(f'POINT_DATA {n}\n')
        for name_, arr in pscal.items():
            f.write('SCALARS %s float 1\nLOOKUP_TABLE default\n' % name_
                    if np.asarray(arr).dtype.kind == 'f'
                    else 'SCALARS %s int 1\nLOOKUP_TABLE default\n' % name_)
            dump_scalars(f, arr)
        f.write(f'CELL_DATA {m}\n')
        for name_, arr in cscal.items():
            f.write(f'SCALARS {name_} float 1\nLOOKUP_TABLE default\n')
            dump_scalars(f, arr)


def write_vtk_sample(out_root, name, m, pos, faces, uv, d, Yf, raw):
    """Optional VTK output (physical + universal space) for debugging."""
    vtk_dir = os.path.join(out_root, 'vtk_small', f'batch_{os.path.basename(os.path.dirname(name))}')
    os.makedirs(vtk_dir, exist_ok=True)
    q, cots, jac = triangle_quality(pos, faces, uv)
    pscal = {'xi': uv[:, 0], 'eta': uv[:, 1],
             'row_id': d['row_id'].numpy(), 'node_id': d['node_id'].numpy(),
             'u': Yf[:, 0], 'v': Yf[:, 1], 'p': Yf[:, 2], 'nu_t': Yf[:, 3]}
    cscal = {'quality': q, 'min_cot': cots.min(axis=1), 'jacobian': jac}
    write_vtk_tri(os.path.join(vtk_dir, f'dif_phys_{m}.vtk'), pos, faces, pscal, cscal)
    write_vtk_tri(os.path.join(vtk_dir, f'dif_uv_{m}.vtk'), uv, faces, pscal, cscal)


def triangle_quality(pos, faces, uv):
    """Shape quality q (physical triangles), raw cotangents, map jacobian."""
    v0, v1, v2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    a = np.linalg.norm(v1 - v2, axis=1)
    b = np.linalg.norm(v2 - v0, axis=1)
    c = np.linalg.norm(v0 - v1, axis=1)
    A = 0.5 * np.abs((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) -
                     (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0]))
    with np.errstate(divide='ignore', invalid='ignore'):
        q = 4.0 * np.sqrt(3.0) * A / (a * a + b * b + c * c)
    cots = _cotangents(pos, faces)
    u0, u1, u2 = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    area_uv = 0.5 * ((u1[:, 0] - u0[:, 0]) * (u2[:, 1] - u0[:, 1]) -
                     (u1[:, 1] - u0[:, 1]) * (u2[:, 0] - u0[:, 0]))
    with np.errstate(divide='ignore', invalid='ignore'):
        jac = area_uv / A
    return q, cots, jac


def run_diffeo(dno_root: str, out_root: str,
               batches: Optional[List[str]] = None,
               workers: int = 3,
               resume: bool = True,
               verbose: bool = True) -> dict:
    """Stage B: diffeomorphism -> dno_small/batch_N/*.npz."""
    os.makedirs(out_root, exist_ok=True)
    if batches is None:
        batches = sorted([d for d in os.listdir(dno_root) if d.startswith('batch_')],
                         key=lambda x: int(x.split('_')[1]))

    items = []
    for bd in batches:
        bpath = os.path.join(dno_root, bd)
        if not os.path.isdir(bpath):
            continue
        for f in sorted(os.listdir(bpath)):
            if f.endswith('.pth'):
                items.append((bd, f))

    todo = []
    for bd, f in items:
        m = re_search_number(f)
        npz_path = os.path.join(out_root, bd, f'graph_airfrans_data_{m}_dno_small.npz')
        if resume and os.path.exists(npz_path):
            continue
        todo.append((bd, f))
    if verbose:
        print(f'Stage B: {len(todo)} samples to process (of {len(items)})')

    t_start = time.time()
    stats = []
    if todo:
        if workers and workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for i, st in enumerate(ex.map(
                        lambda t: process_sample_diffeo(dno_root, t[0], t[1], out_root),
                        todo, chunksize=1), 1):
                    stats.append(st)
                    if verbose:
                        print(f'[{i}/{len(todo)}] {st["batch"]}/{st["sample"]}: '
                              f'{st["nodes"]:,} nodes, {st["tris"]:,} tris, '
                              f'cov={st["coverage"]:.1f}%, fallback={st["fallback"]}, '
                              f'hole={st["hole"]} | {st["time"]:.0f}s', flush=True)
        else:
            for i, (bd, f) in enumerate(todo, 1):
                st = process_sample_diffeo(dno_root, bd, f, out_root)
                stats.append(st)
                if verbose:
                    print(f'[{i}/{len(todo)}] {st["batch"]}/{st["sample"]}: '
                          f'{st["nodes"]:,} nodes, {st["tris"]:,} tris, '
                          f'cov={st["coverage"]:.1f}% | {st["time"]:.0f}s', flush=True)

    tot = time.time() - t_start
    if verbose:
        print(f'Stage B done: {len(stats)} samples in {tot / 60:.1f} min')
    with open(os.path.join(out_root, 'generation_log.json'), 'w') as f:
        json.dump({'n': len(stats), 'total_sec': tot, 'stats': stats}, f, indent=1)
    return {'ok': len(stats), 'total_sec': tot}


def main():
    parser = argparse.ArgumentParser(
        description='DNO dataset preparation: zonal resampling (a) + diffeomorphism (b).')
    parser.add_argument('--stage', choices=['a', 'b', 'all'], default='all')
    parser.add_argument('--index_root', type=str,
                        help='index_raw_data_w root (stage a)')
    parser.add_argument('--graph_root', type=str,
                        help='Root with graph_airfrans_data_batch_{1..10} (stage a)')
    parser.add_argument('--dno_root', type=str, required=True,
                        help='DNO_data root (intermediate for stage a output, input for b)')
    parser.add_argument('--out_root', type=str,
                        help='dno_small root (stage b output; default: {dno_root}/dno_small)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--no_filter', action='store_true',
                        help='Disable sample filter in stage a')
    parser.add_argument('--no_resume', action='store_true',
                        help='Overwrite already existing files')
    parser.add_argument('--vtk', action='store_true',
                        help='Write VTK debug output in stage b')
    args = parser.parse_args()

    if args.stage in ('a', 'all'):
        if not args.index_root or not args.graph_root:
            parser.error('--index_root and --graph_root are required for stage a')
        run_resample(args.index_root, args.graph_root, args.dno_root,
                     workers=args.workers,
                     apply_filter=not args.no_filter,
                     resume=not args.no_resume)

    if args.stage in ('b', 'all'):
        out_root = args.out_root or os.path.join(args.dno_root, 'dno_small')
        run_diffeo(args.dno_root, out_root,
                   workers=max(1, args.workers // 2),
                   resume=not args.no_resume)


if __name__ == '__main__':
    main()

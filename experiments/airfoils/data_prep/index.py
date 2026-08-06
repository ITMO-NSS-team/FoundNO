"""
index.py - COMMON preprocessing stage for Geo-FNO and DNO (AirfRANS).

What it does:
    Raw AirfRANS graph (.pth PyG: pos, x, y) -> "blocks -> rows" structure,
    saved to index_raw_data_w/ (per batch).

This stage is COMMON for Geo-FNO and DNO; afterwards the pipeline branches:
    prep_geofno.py  : rows -> unfolded C-grid (N_ROW x n_dots)
    prep_dno.py     : rows -> diffeomorphism -> regular grid (npz)

Algorithm (ported from airfrans_data/index/data_index.ipynb, cell 27):
    1. Bounding box filtering
    2. Raw block detection: index gaps + angle jumps + Y jumps
    3. Smart merging: blocks with wall points are "anchors",
       the rest are attached by angle (polar anchoring to wing center)
    4. Clockwise sorting of blocks starting from the "seam"
    5. Inner boundary removal (wall points, wall_distance == 0)
    6. Splitting each block into rows: detecting "bends" in the point path
    7. Row orientation and global row stitching across blocks
    8. Wall point ordering (nearest-neighbor chains per block)

Output format (torch.save dict per sample):
    name, global_wing_rows, global_id_to_xy, cleaned_blocks, blocks,
    ordered_wall_indices, wall_per_block, stats, bbox

Usage:
    python -m experiments.airfoils.data_prep.index \\
        --raw_root  /media/.../airfrans \\
        --out_root  /media/.../airfrans/index_raw_data_w
"""

import os
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# 1. Row splitting of a cleaned block

def get_rows_from_cleaned_block(b_info: dict, bx_arr: np.ndarray,
                                by_arr: np.ndarray, diff_threshold: float = 0.5):
    """
    Split a cleaned block (without wall points) into rows.

    Walk along the block points; as soon as the movement direction "bends"
    (|cos1 - cos2| > diff_threshold), the current row is closed and a new
    one starts. A row is a list of original point indices.

    Returns
    -------
    rows : list[list[int]] - rows (original point indices)
    bx, by : np.ndarray - block point coordinates
    id_to_local : dict - original index -> position in b_idx
    """
    b_origs = b_info['clean_orig_indices']
    s, e = b_info['span']
    mask = b_info['valid_mask']

    bx = bx_arr[s:e + 1][mask]
    by = by_arr[s:e + 1][mask]
    b_idx = b_origs

    rows = []
    if len(bx) < 4:
        return [list(b_idx)], bx, by, {idx: i for i, idx in enumerate(b_idx)}

    current_row = [b_idx[0], b_idx[1], b_idx[2]]
    i = 2
    while i < len(bx) - 3:
        v1 = np.array([bx[i - 1] - bx[i - 2], by[i - 1] - by[i - 2]])
        v2 = np.array([bx[i] - bx[i - 1], by[i] - by[i - 1]])
        v3 = np.array([bx[i + 1] - bx[i], by[i + 1] - by[i]])

        def norm(v): return np.linalg.norm(v) + 1e-9
        v1, v2, v3 = v1 / norm(v1), v2 / norm(v2), v3 / norm(v3)

        cos1 = np.dot(v1, v2)
        cos2 = np.dot(v2, v3)

        if np.abs(cos1 - cos2) > diff_threshold:
            rows.append(current_row)
            current_row = [b_idx[i + 1], b_idx[i + 2], b_idx[i + 3]]
            i += 2
        else:
            current_row.append(b_idx[i + 1])
        i += 1

    if current_row and current_row not in rows:
        rows.append(current_row)

    id_to_local = {idx: idx_loc for idx_loc, idx in enumerate(b_idx)}
    return rows, bx, by, id_to_local


# 2. Wall point ordering (inner boundary)

def extract_ordered_inner_boundary(blocks, orig_indices, wall_dist, x, y,
                                   oriented: Optional[dict] = None,
                                   global_id_to_xy: Optional[dict] = None):
    """
    Collect wall point indices (wall_distance == 0) in block traversal order.

    Inside each block, points are ordered via a nearest-neighbor chain,
    forming a spatially ordered sequence along the wing contour.

    If oriented (dict block_idx -> list of oriented rows) and
    global_id_to_xy are passed, the block orientation (first point of the
    first oriented row) is used to pick the wall traversal start point in
    each block - aligning wall traversal direction with row orientation.

    Returns
    -------
    ordered_wall_indices : list[int] - flat wall index list
    wall_per_block : list[list[int]] - same, split by block
    """
    # Step 1: raw wall points per block
    raw_wall = []
    for s, e in blocks:
        b_origs = orig_indices[s:e + 1]
        b_wd = wall_dist[s:e + 1]
        b_x = x[s:e + 1]
        b_y = y[s:e + 1]
        wall_mask = (b_wd == 0)
        raw_wall.append((b_origs[wall_mask], b_x[wall_mask], b_y[wall_mask]))

    # Step 2: nearest-neighbor chain for each block
    chains = []
    for block_idx, (wall_idx, wall_x, wall_y) in enumerate(raw_wall):
        n = len(wall_idx)
        if n == 0:
            chains.append([])
            continue

        coords = np.column_stack([wall_x, wall_y])

        use_orient = (oriented is not None and global_id_to_xy is not None
                      and block_idx in oriented and len(oriented[block_idx]) > 0)

        if use_orient:
            ref_id = oriented[block_idx][0][0]
            ref_pt = np.array(global_id_to_xy[ref_id])
            dists_to_ref = np.sum((coords - ref_pt) ** 2, axis=1)
            start = int(np.argmin(dists_to_ref))
        else:
            start = 0

        if n == 1:
            chains.append([0])
            continue

        visited = np.zeros(n, dtype=bool)
        chain = [start]
        visited[start] = True
        current = start

        for _ in range(n - 1):
            dists = np.sum((coords - coords[current]) ** 2, axis=1)
            dists[visited] = np.inf
            nxt = int(np.argmin(dists))
            chain.append(nxt)
            visited[nxt] = True
            current = nxt

        chains.append(chain)

    # Step 3: fix direction for blocks without orientation info
    # (so that the last point of block N is close to the first of block N+1)
    for i in range(len(chains)):
        if len(chains[i]) <= 1:
            continue

        has_orient = (oriented is not None and i in oriented and len(oriented[i]) > 0)
        if has_orient:
            continue

        wall_idx_i, wx_i, wy_i = raw_wall[i]
        coords_i = np.column_stack([wx_i, wy_i])
        chain_i = chains[i]

        cur_first = coords_i[chain_i[0]]
        cur_last = coords_i[chain_i[-1]]

        if i + 1 < len(chains) and len(chains[i + 1]) > 0:
            wall_idx_next, wx_next, wy_next = raw_wall[i + 1]
            coords_next = np.column_stack([wx_next, wy_next])
            next_first = coords_next[chains[i + 1][0]]

            d_first = np.sum((cur_first - next_first) ** 2)
            d_last = np.sum((cur_last - next_first) ** 2)

            if d_first < d_last:
                chains[i] = chain_i[::-1]
        elif i > 0 and len(chains[i - 1]) > 0:
            wall_idx_prev, wx_prev, wy_prev = raw_wall[i - 1]
            coords_prev = np.column_stack([wx_prev, wy_prev])
            prev_last = coords_prev[chains[i - 1][-1]]

            d_first = np.sum((cur_first - prev_last) ** 2)
            d_last = np.sum((cur_last - prev_last) ** 2)

            if d_last < d_first:
                chains[i] = chain_i[::-1]

    # Step 4: assemble result
    wall_per_block = []
    ordered_wall_indices = []
    for block_idx, chain in enumerate(chains):
        wall_idx = raw_wall[block_idx][0]
        ordered = [int(wall_idx[i]) for i in chain]
        wall_per_block.append(ordered)
        ordered_wall_indices.extend(ordered)

    return ordered_wall_indices, wall_per_block


# 3. Full per-sample pipeline: blocks -> rows -> boundary

def process_sample_v2(data, x_min: float, x_max: float, y_min: float, y_max: float,
                      center_x: float = 0.5, center_y: float = 0.0,
                      diff_threshold: float = 0.5, check_row: int = 130,
                      y_jump_threshold: float = 0.5):
    """
    Full preprocessing pipeline for one AirfRANS sample.

    data - PyG graph: data.pos [N, 2], data.x [N, 5] (column 2 = wall_distance).

    Returns
    -------
    global_wing_rows, global_id_to_xy, cleaned_blocks, blocks,
    ordered_wall_indices, wall_per_block, stats
    """
    # 1. Load data
    x_full = data.pos[:, 0].cpu().numpy()
    y_full = data.pos[:, 1].cpu().numpy()
    indices_full = np.arange(len(x_full))
    wall_distance_full = data.x[:, 2].cpu().numpy()

    # 2. Bounding box
    mask = (x_full >= x_min) & (x_full <= x_max) & (y_full >= y_min) & (y_full <= y_max)
    x = x_full[mask]
    y = y_full[mask]
    orig_indices = indices_full[mask]
    wall_dist = wall_distance_full[mask]

    # 3. Raw block detection
    gaps = np.diff(orig_indices)
    break_indices = np.where(gaps > 1)[0]

    y_diffs = np.abs(np.diff(y))
    y_jump_between_mask = np.zeros(len(y), dtype=bool)
    y_jump_positions = np.where(y_diffs > y_jump_threshold)[0]
    y_jump_between_mask[y_jump_positions] = True

    block_boundaries = []
    for i in range(1, len(break_indices)):
        idx_curr = break_indices[i]
        idx_prev = break_indices[i - 1]

        vx_curr = x[idx_curr + 1] - x[idx_curr]
        vy_curr = y[idx_curr + 1] - y[idx_curr]
        angle_curr = np.arctan2(vy_curr, vx_curr)

        vx_prev = x[idx_prev + 1] - x[idx_prev]
        vy_prev = y[idx_prev + 1] - y[idx_prev]
        angle_prev = np.arctan2(vy_prev, vx_prev)

        da = np.abs(angle_curr - angle_prev)
        da = min(da, 2 * np.pi - da)
        gap_size = gaps[idx_curr]

        y_jump_between = y_jump_between_mask[idx_prev:idx_curr + 1].any()

        if da > 0.5 or gap_size > 5000 or y_jump_between:
            block_boundaries.append(idx_curr)

    block_boundaries = np.unique(block_boundaries)
    block_starts = np.concatenate([[0], block_boundaries + 1])
    block_ends = np.concatenate([block_boundaries, [len(x) - 1]])
    raw_blocks = [(s, e) for s, e in zip(block_starts, block_ends) if e - s > 50]

    # 4. Smart merging (anchors = blocks containing wall points)
    def get_block_angle(s, e, cx=center_x, cy=center_y):
        bx_mean = np.mean(x[s:e + 1])
        by_mean = np.mean(y[s:e + 1])
        return np.arctan2(by_mean - cy, bx_mean - cx)

    anchors = []
    orphans = []

    for idx, (s, e) in enumerate(raw_blocks):
        if np.any(wall_dist[s:e + 1] < 0.02):
            anchors.append({'idx': idx, 'span': (s, e),
                            'angle': get_block_angle(s, e),
                            'merged_spans': [(s, e)]})
        else:
            orphans.append({'idx': idx, 'span': (s, e),
                            'angle': get_block_angle(s, e)})

    for orphan in orphans:
        min_angle_diff = float('inf')
        best_anchor = None
        for anchor in anchors:
            diff = np.abs(orphan['angle'] - anchor['angle'])
            diff = min(diff, 2 * np.pi - diff)
            if diff < min_angle_diff:
                min_angle_diff = diff
                best_anchor = anchor
        if min_angle_diff < 2 and best_anchor is not None:
            best_anchor['merged_spans'].append(orphan['span'])
        else:
            anchors.append({'idx': orphan['idx'], 'span': orphan['span'],
                            'angle': orphan['angle'],
                            'merged_spans': [orphan['span']]})

    blocks = []
    for anchor in anchors:
        if not anchor['merged_spans']:
            continue
        all_starts = [span[0] for span in anchor['merged_spans']]
        all_ends = [span[1] for span in anchor['merged_spans']]
        blocks.append((min(all_starts), max(all_ends)))

    # 4.4 Clockwise sorting starting from the seam
    block_info_list = []
    for s, e in blocks:
        cx_blk = np.mean(x[s:e + 1])
        cy_blk = np.mean(y[s:e + 1])
        math_angle = np.arctan2(cy_blk - center_y, cx_blk - center_x)
        cw_angle = np.mod(-math_angle, 2 * np.pi)
        block_info_list.append({'span': (s, e), 'cw_angle': cw_angle, 'size': e - s + 1})

    start_angle = 0.0
    for info in block_info_list:
        if info['size'] < 100:
            start_angle = info['cw_angle']
            break

    block_info_list = sorted(block_info_list,
                             key=lambda info: np.mod(info['cw_angle'] - start_angle, 2 * np.pi))
    blocks = [info['span'] for info in block_info_list]

    # 4.5 Remove inner boundary (wall points)
    inner_boundary = orig_indices[wall_dist == 0]
    inner_set = set(inner_boundary.tolist())

    cleaned_blocks = []
    for s, e in blocks:
        b_origs = orig_indices[s:e + 1]
        valid_mask = np.array([idx not in inner_set for idx in b_origs])
        cleaned_blocks.append({
            'span': (s, e),
            'valid_mask': valid_mask,
            'clean_orig_indices': b_origs[valid_mask]
        })

    # 6. Row stitching (skipping block 0)
    all_blocks_rows = []
    for b_info in cleaned_blocks:
        rows, _, _, _ = get_rows_from_cleaned_block(b_info, x, y,
                                                    diff_threshold=diff_threshold)
        all_blocks_rows.append(rows)

    # ID -> coordinates map
    global_id_to_xy = {}
    for s_b, e_b in blocks:
        bx_full_b = x[s_b:e_b + 1]
        by_full_b = y[s_b:e_b + 1]
        b_ids_b = orig_indices[s_b:e_b + 1]
        for px, py, pid in zip(bx_full_b, by_full_b, b_ids_b):
            global_id_to_xy[pid] = (px, py)

    def dist(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    # Orientation of blocks 1..N
    if len(all_blocks_rows) < 2:
        global_wing_rows = []
        for b_rows in all_blocks_rows[1:] if len(all_blocks_rows) > 1 else []:
            global_wing_rows.extend(b_rows)

        ordered_wall_indices, wall_per_block = extract_ordered_inner_boundary(
            blocks, orig_indices, wall_dist, x, y
        )

        return (global_wing_rows, global_id_to_xy, cleaned_blocks, blocks,
                ordered_wall_indices, wall_per_block,
                {
                    'n_raw_blocks': len(raw_blocks),
                    'n_inner_boundary': len(ordered_wall_indices),
                    'n_final_blocks': len(blocks),
                    'n_global_rows': len(global_wing_rows),
                    'row_sizes': [len(r) for r in global_wing_rows],
                    'block_row_counts': [len(r) for r in all_blocks_rows],
                })

    oriented = {}
    oriented[1] = [row[:] for row in all_blocks_rows[1]]

    for block_idx in range(2, len(all_blocks_rows)):
        prev_rows = oriented[block_idx - 1]
        prev_last_pt = global_id_to_xy[prev_rows[0][-1]]

        curr_row0 = all_blocks_rows[block_idx][0]
        first_pt = global_id_to_xy[curr_row0[0]]
        last_pt = global_id_to_xy[curr_row0[-1]]

        d_first = dist(prev_last_pt, first_pt)
        d_last = dist(prev_last_pt, last_pt)

        is_forward = d_first <= d_last
        if is_forward:
            oriented[block_idx] = [row[:] for row in all_blocks_rows[block_idx]]
        else:
            oriented[block_idx] = [row[::-1] for row in all_blocks_rows[block_idx]]

    # Order the boundary AFTER orientation
    ordered_wall_indices, wall_per_block = extract_ordered_inner_boundary(
        blocks, orig_indices, wall_dist, x, y,
        oriented=oriented, global_id_to_xy=global_id_to_xy
    )

    # Global calibration (row shifts between blocks)
    raw_offsets = {1: 0}
    for block_idx in range(1, len(all_blocks_rows) - 1):
        b_curr = oriented[block_idx]
        b_next = oriented[block_idx + 1]

        idx = min(check_row, len(b_curr) - 1, len(b_next) - 1)
        ref_id = b_curr[idx][-1]
        ref_pt = global_id_to_xy[ref_id]

        best_dist = float('inf')
        best_match_idx = idx
        search_start = max(0, idx - 10)
        search_end = min(len(b_next), idx + 11)

        for j in range(search_start, search_end):
            cand_id = b_next[j][0]
            cand_pt = global_id_to_xy[cand_id]
            d = dist(ref_pt, cand_pt)
            if d < best_dist:
                best_dist = d
                best_match_idx = j

        local_shift = best_match_idx - idx
        raw_offsets[block_idx + 1] = raw_offsets[block_idx] + local_shift

    # Trimming and stitching
    min_offset = min(raw_offsets.values())
    final_cuts = {b: raw_offsets[b] - min_offset for b in raw_offsets}

    aligned_blocks = {}
    for b in range(1, len(all_blocks_rows)):
        cut = final_cuts[b]
        aligned_blocks[b] = oriented[b][cut:]

    max_rows = max((len(rows) for rows in aligned_blocks.values()), default=0)
    global_wing_rows = []
    for k in range(max_rows):
        current_global_row = []
        for block_idx in range(1, len(all_blocks_rows)):
            rows = aligned_blocks[block_idx]
            if k < len(rows):
                current_global_row.extend(rows[k])
        if current_global_row:
            global_wing_rows.append(current_global_row)

    stats = {
        'n_raw_blocks': len(raw_blocks),
        'n_inner_boundary': len(ordered_wall_indices),
        'n_final_blocks': len(blocks),
        'n_global_rows': len(global_wing_rows),
        'row_sizes': [len(r) for r in global_wing_rows],
        'block_row_counts': [len(r) for r in all_blocks_rows],
    }

    return (global_wing_rows, global_id_to_xy, cleaned_blocks, blocks,
            ordered_wall_indices, wall_per_block, stats)


# 4. index_raw_data_w generator over all batches

# Generator defaults (as in data_index.ipynb, cell 27)
DEFAULT_BBOX = (-1.0, 2.5, -0.75, 0.75)   # X_MIN, X_MAX, Y_MIN, Y_MAX
DEFAULT_BATCHES = list(range(1, 11))
DEFAULT_CENTER = (0.5, 0.0)
DEFAULT_DIFF_THRESHOLD = 0.5
DEFAULT_CHECK_ROW = 130
DEFAULT_Y_JUMP_THRESHOLD = 0.5


def generate_index_raw_data(raw_root: str, out_root: str,
                            batches: Optional[List[int]] = None,
                            bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
                            center: Tuple[float, float] = DEFAULT_CENTER,
                            diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
                            check_row: int = DEFAULT_CHECK_ROW,
                            y_jump_threshold: float = DEFAULT_Y_JUMP_THRESHOLD,
                            resume: bool = True,
                            verbose: bool = True) -> Dict[str, int]:
    """
    Run process_sample_v2 over all samples of all batches and save the
    result to out_root/batch_N/ (files keep their input names).

    Input layout:
        {raw_root}/graph_airfrans_data_batch_{1..10}/*.pth
        File names are local to each group of 4 batches (see module docstring).

    Returns
    -------
    {'processed': int, 'skipped': int, 'failed': int}
    """
    os.makedirs(out_root, exist_ok=True)
    batches = batches or DEFAULT_BATCHES
    x_min, x_max, y_min, y_max = bbox
    center_x, center_y = center

    processed = skipped = failed = 0
    for batch_num in batches:
        in_dir = os.path.join(raw_root, f'graph_airfrans_data_batch_{batch_num}')
        out_dir = os.path.join(out_root, f'batch_{batch_num}')
        if not os.path.isdir(in_dir):
            if verbose:
                print(f'[skip] missing directory {in_dir}')
            continue
        os.makedirs(out_dir, exist_ok=True)

        # Local file names (as in cell 27: block_idx = (batch-1) % 4)
        block_idx = (batch_num - 1) % 4
        start_file_idx = block_idx * 100
        file_names = [f'graph_airfrans_data_{start_file_idx + i}.pth'
                      for i in range(1, 101)]

        for fname in file_names:
            fpath = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, fname)
            if not os.path.exists(fpath):
                continue
            if resume and os.path.exists(out_path):
                skipped += 1
                continue
            try:
                data = torch.load(fpath, weights_only=False)
                (global_wing_rows, global_id_to_xy, cleaned_blocks, blocks,
                 ordered_wall_indices, wall_per_block, stats) = process_sample_v2(
                    data,
                    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                    center_x=center_x, center_y=center_y,
                    diff_threshold=diff_threshold,
                    check_row=check_row, y_jump_threshold=y_jump_threshold,
                )
                torch.save({
                    'name': fname,
                    'global_wing_rows': global_wing_rows,
                    'global_id_to_xy': global_id_to_xy,
                    'cleaned_blocks': cleaned_blocks,
                    'blocks': blocks,
                    'ordered_wall_indices': ordered_wall_indices,
                    'wall_per_block': wall_per_block,
                    'stats': stats,
                    'bbox': (x_min, x_max, y_min, y_max),
                }, out_path)
                processed += 1
                if verbose and processed % 50 == 0:
                    print(f'  [{batch_num}] {fname}: {stats["n_global_rows"]} rows, '
                          f'{stats["n_final_blocks"]} blocks, '
                          f'{len(ordered_wall_indices)} wall points')
            except Exception as e:
                failed += 1
                print(f'  ERROR {fname}: {e}')

    if verbose:
        print(f'Done: processed {processed}, skipped {skipped}, failed {failed}')
        print(f'Saved to: {out_root}')
    return {'processed': processed, 'skipped': skipped, 'failed': failed}


# CLI

def main():
    parser = argparse.ArgumentParser(
        description='Common AirfRANS preprocessing stage: blocks -> rows (index_raw_data_w).')
    parser.add_argument('--raw_root', type=str, required=True,
                        help='Root containing graph_airfrans_data_batch_{1..10}')
    parser.add_argument('--out_root', type=str, required=True,
                        help='Where to write index_raw_data_w')
    parser.add_argument('--batches', type=int, nargs='*', default=None,
                        help='Batch numbers (default: 1..10)')
    parser.add_argument('--bbox', type=float, nargs=4, default=list(DEFAULT_BBOX),
                        help='X_MIN X_MAX Y_MIN Y_MAX')
    parser.add_argument('--diff_threshold', type=float, default=DEFAULT_DIFF_THRESHOLD)
    parser.add_argument('--check_row', type=int, default=DEFAULT_CHECK_ROW)
    parser.add_argument('--no_resume', action='store_true',
                        help='Overwrite already existing files')
    args = parser.parse_args()

    result = generate_index_raw_data(
        raw_root=args.raw_root,
        out_root=args.out_root,
        batches=args.batches,
        bbox=tuple(args.bbox),
        diff_threshold=args.diff_threshold,
        check_row=args.check_row,
        resume=not args.no_resume,
    )
    print(result)


if __name__ == '__main__':
    main()

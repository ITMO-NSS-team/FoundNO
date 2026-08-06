"""
airfoil_datasets.py - dataset loaders for the AirfRANS experiments.

Three cases, one common interface:

    FnoDataset     ('fno' / 'rno')    - fno_data_{0001..1000}.npz, grid 128x256
    DnoDataset     ('dno')            - dno_small/batch_N/*.npz, grid 256x256
    GeofnoDataset  ('geofno')         - Geo-FNO_data/batch_N/*.pth, C-grid 137x1128

Each sample is a dict:
    x         : [H, W, C_in]   input channels (mask included where applicable)
    y         : [H, W, 4]      output fields (vel_x, vel_y, pressure, nu_t)
    mask      : [H, W] or None mask of valid points (for the loss);
                               None for Geo-FNO (no mask)
    grid_mesh : [H, W, 2] or None physical coordinates (DNO only)
    pos       : [H, W, 2] or None physical coordinates (Geo-FNO only)

Global file indexing (names repeat in groups of 4 batches):
    global_idx = ((batch - 1) // 4) * 400 + local_number

Default split by global index: train 1..800, val 801..900, test 901..1000
(use file_range or indices to override).
"""

import glob
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

OUT_FIELDS = ('vel_x', 'vel_y', 'pressure', 'nu_t')


def _global_index(batch: int, local: int) -> int:
    """Global sample index from batch number and local file number."""
    return ((batch - 1) // 4) * 400 + local


def _parse_name(path: str):
    """Extract (batch, local) from a path containing batch_N and a number."""
    m = re.search(r'batch_(\d+)/[^/]*?(\d+)[^/]*$', path)
    if not m:
        raise ValueError(f'cannot parse batch/local from {path}')
    return int(m.group(1)), int(m.group(2))


def _split_indices(indices: List[int], train: Tuple[int, int] = (1, 800),
                   val: Tuple[int, int] = (801, 900),
                   test: Tuple[int, int] = (901, 1000)):
    """Split sorted global indices into train/val/test by inclusive ranges."""
    tr = [i for i in indices if train[0] <= i <= train[1]]
    va = [i for i in indices if val[0] <= i <= val[1]]
    te = [i for i in indices if test[0] <= i <= test[1]]
    return tr, va, te


class FnoDataset(Dataset):
    """FNO/RNO dataset: fno_data_{0001..1000}.npz (regular 128x256 grid).

    Input:  [vx_in (broadcast), vy_in (broadcast), mask] -> [H, W, 3]
    Output: [vel_x, vel_y, pressure, nu_t]               -> [H, W, 4]
    """

    def __init__(self, paths: List[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        H, W = data['mask'].shape

        vx = np.full((H, W, 1), data['vx_in'], dtype=np.float32)
        vy = np.full((H, W, 1), data['vy_in'], dtype=np.float32)
        mask = data['mask'].reshape(H, W, 1).astype(np.float32)
        x = np.concatenate([vx, vy, mask], axis=-1)

        y = np.stack([data['vel_x'], data['vel_y'],
                      data['pressure'], data['nut']], axis=-1).astype(np.float32)

        return {
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
            'mask': torch.from_numpy(data['mask'].astype(np.float32)),
            'grid_mesh': None,
            'pos': None,
        }


class DnoDataset(Dataset):
    """DNO dataset: dno_small/batch_N/graph_*_dno_small.npz (256x256).

    Input:  [vx_in (input[...,0]), vy_in (input[...,1]), mask] -> [H, W, 3]
    Output: output (vel_x, vel_y, pressure, nu_t)              -> [H, W, 4]
    Geometry: grid_mesh = [grid_x, grid_y]                     -> [H, W, 2]
    """

    def __init__(self, paths: List[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        H, W = data['mask'].shape

        x = np.stack([
            data['input'][..., 0],   # vx_in
            data['input'][..., 1],   # vy_in
            data['mask'],
        ], axis=-1).astype(np.float32)                       # [H, W, 3]

        grid_mesh = np.stack([data['grid_x'], data['grid_y']],
                             axis=-1).astype(np.float32)     # [H, W, 2]
        y = data['output'].astype(np.float32)                # [H, W, 4]

        return {
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
            'mask': torch.from_numpy(data['mask'].astype(np.float32)),
            'grid_mesh': torch.from_numpy(grid_mesh),
            'pos': None,
        }


class GeofnoDataset(Dataset):
    """Geo-FNO dataset: Geo-FNO_data/batch_N/*.pth (C-grid, no mask).

    Input:  [vx_in (broadcast), vy_in (broadcast)] -> [H, W, 2]
    Output: fields (vel_x, vel_y, pressure, nu_t)  -> [H, W, 4]
    """

    def __init__(self, paths: List[str], cache: bool = False):
        self.paths = paths
        self.data = None
        if cache:
            self.data = [self._load(p) for p in paths]

    def __len__(self):
        return len(self.paths)

    def _load(self, p):
        d = torch.load(p, weights_only=False)
        fields = d['fields']                     # (H, W, 4) float32
        H, W = fields.shape[:2]

        vx = np.full((H, W, 1), float(d['vx_in']), dtype=np.float32)
        vy = np.full((H, W, 1), float(d['vy_in']), dtype=np.float32)
        x = np.concatenate([vx, vy], axis=-1)    # (H, W, 2)

        return {
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(fields.astype(np.float32)),
            'mask': None,                        # no mask for the C-grid
            'grid_mesh': None,
            'pos': torch.from_numpy(d['pos'].astype(np.float32)),
        }

    def __getitem__(self, idx):
        if self.data is not None:
            return self.data[idx]
        return self._load(self.paths[idx])


def collate_fn(batch):
    """Stack a batch of sample dicts into batched tensors.

    Returns a dict with x [B,H,W,C], y [B,H,W,4],
    mask [B,H,W] or None, grid_mesh [B,H,W,2] or None, pos [B,H,W,2] or None.
    """
    out = {'x': torch.stack([b['x'] for b in batch], dim=0),
           'y': torch.stack([b['y'] for b in batch], dim=0)}
    for key in ('mask', 'grid_mesh', 'pos'):
        vals = [b[key] for b in batch]
        out[key] = torch.stack(vals, dim=0) if vals[0] is not None else None
    return out


def get_airfoil_dataset(case: str, root_dir: str, split: str = 'train',
                        file_range: Optional[Tuple[int, int]] = None,
                        indices: Optional[List[int]] = None,
                        cache: bool = False):
    """
    Build a dataset for the given case and split.

    Parameters
    ----------
    case : 'fno' | 'rno' | 'dno' | 'geofno'
        Dataset kind (fno/rno share the fno_data npz).
    root_dir : str
        Root with the case data:
        - fno/rno:   .../fno_dataset
        - dno:       .../DNO_data/dno_small
        - geofno:    .../Geo-FNO_data
    split : 'train' | 'val' | 'test'
        Default ranges by global index: train 1..800, val 801..900, test 901..1000.
    file_range : tuple (lo, hi) or None
        Override split with an inclusive global-index range.
    indices : list[int] or None
        Exact global indices to use (takes precedence over split/range).
    cache : bool
        Preload all files into RAM (Geo-FNO only).
    """
    if case in ('fno', 'rno'):
        pattern = os.path.join(root_dir, 'fno_data_*.npz')
        files = sorted(glob.glob(pattern))
        pairs = [(int(re.search(r'(\d+)', os.path.basename(f)).group(1)), f)
                 for f in files]
        pairs.sort()
        global_to_path = {g: p for g, p in pairs}
        cls = FnoDataset
    elif case == 'dno':
        pattern = os.path.join(root_dir, 'batch_*', '*.npz')
        files = glob.glob(pattern)
        pairs = [(_global_index(*_parse_name(f)), f) for f in files]
        pairs.sort()
        global_to_path = {g: p for g, p in pairs}
        cls = DnoDataset
    elif case == 'geofno':
        pattern = os.path.join(root_dir, 'batch_*', '*.pth')
        files = glob.glob(pattern)
        pairs = [(_global_index(*_parse_name(f)), f) for f in files]
        pairs.sort()
        global_to_path = {g: p for g, p in pairs}
        cls = GeofnoDataset
    else:
        raise ValueError(f'unknown case: {case}')

    all_indices = sorted(global_to_path)

    if indices is not None:
        selected = [(i, global_to_path[i]) for i in indices if i in global_to_path]
    elif file_range is not None:
        lo, hi = file_range
        selected = [(i, global_to_path[i]) for i in all_indices if lo <= i <= hi]
    else:
        tr, va, te = _split_indices(all_indices)
        ranges = {'train': tr, 'val': va, 'test': te}
        selected = [(i, global_to_path[i]) for i in ranges[split]]

    if not selected:
        raise FileNotFoundError(f'no files for case={case}, split={split} in {root_dir}')

    paths = [p for _, p in selected]
    ds = cls(paths, cache=cache) if case == 'geofno' else cls(paths)
    ds.indices = [g for g, _ in selected]  # global sample indices
    return ds

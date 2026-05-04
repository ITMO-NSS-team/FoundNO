"""
Unified Dataset for Diffeomorphism Neural Operator (DNO) tasks.

Supports three data cases:
  - Case 1 (darcy):    Loads from CSV (train_C.csv, train_x_data.csv, etc.)
                        directly into RAM. Small datasets (30-1000 samples).
                        Creates a mask from NaN values (holes).

  - Case 2 (fluid):    Loads Navier-Stokes CSV data into RAM.
                        Uses StandardScaler (Z-score) normalization.

  - Case 3 (reservoir): Lazy-loads from large HDF5 files (~15-20 GB).
                        Seq2Seq: 39 input channels → 72 output channels.
"""

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class StandardScaler:
    """Z-score normalizer on numpy arrays (ignores zeros as former NaN)."""
    def __init__(self, data):
        valid = data[data != 0]
        if len(valid) == 0:
            valid = data
        self.mean = float(np.mean(valid))
        self.std = float(np.std(valid))
        if self.std == 0:
            self.std = 1.0

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


class DiffeoDataset(Dataset):
    """
    Unified dataset for DNO tasks.

    Parameters
    ----------
    data_path : str
        Path to data folder (CSV) or HDF5 file (reservoir).
    indices : list of int
        Sample indices after train/val split.
    case_type : str
        One of "darcy", "fluid", "reservoir".
    p_scale, s_scale : float
        Scaling factors for reservoir case (pressure and source).
    """
    def __init__(self, data_path, indices, case_type="fluid",
                 p_scale=100.0, s_scale=1000.0):
        self.data_path = data_path
        self.indices = indices
        self.case_type = case_type
        self.p_scale = p_scale
        self.s_scale = s_scale

        self._h5_file = None
        self.ram_cache = {}

        if self.case_type == "darcy":
            self._preload_darcy_csv()
        elif self.case_type == "fluid":
            self._preload_fluid_csv()

    def __len__(self):
        return len(self.indices)

    # CASE 1: DARCY (Polygons with holes from CSV)
    def _preload_darcy_csv(self):
        """
        Load Darcy data with DNO masking logic:
          1. Create domain mask (1 = domain, 0 = hole) from NaN.
          2. Clean NaN → 0.0 (NNs cannot handle NaN).
          3. Reshape from flat vectors to 2D grids.
        """
        print("Loading Darcy data with DNO masking logic...")

        raw_c = np.loadtxt(os.path.join(self.data_path, 'train_C.csv'),
                           delimiter=',')
        raw_x = np.loadtxt(os.path.join(self.data_path, 'train_x_data.csv'),
                           delimiter=',')
        raw_y = np.loadtxt(os.path.join(self.data_path, 'train_y_data.csv'),
                           delimiter=',')
        raw_u = np.loadtxt(os.path.join(self.data_path, 'train_U.csv'),
                           delimiter=',')

        # Dynamic mask from NaN: mask=0 where C=NaN (hole)
        mask = (~np.isnan(raw_c)).astype(np.float32)

        c_clean = np.nan_to_num(raw_c, nan=0.0)
        x_clean = np.nan_to_num(raw_x, nan=0.0)
        y_clean = np.nan_to_num(raw_y, nan=0.0)
        u_clean = np.nan_to_num(raw_u, nan=0.0)

        # Reshape to N×128×128
        self.ram_cache['c'] = c_clean.reshape(-1, 128, 128)
        self.ram_cache['x'] = x_clean.reshape(-1, 128, 128)
        self.ram_cache['y'] = y_clean.reshape(-1, 128, 128)
        self.ram_cache['mask'] = mask.reshape(-1, 128, 128)
        self.ram_cache['u'] = u_clean.reshape(-1, 128, 128)

        print(f"Preloaded {self.ram_cache['c'].shape[0]} Darcy samples.")

    def _get_darcy_sample(self, scenario_idx):
        """
        Case 1: Darcy flow.
        Input:  [C, X, Y, Mask] (4 channels)
        Output: [U * 10]       (1 channel)
        """
        c = self.ram_cache['c'][scenario_idx]
        x = self.ram_cache['x'][scenario_idx]
        y = self.ram_cache['y'][scenario_idx]
        mask = self.ram_cache['mask'][scenario_idx]
        u = self.ram_cache['u'][scenario_idx]

        x_input = np.stack([c, x, y, mask], axis=-1)
        y_output = u[..., None] * 10.0

        return {
            "x": torch.tensor(x_input, dtype=torch.float32),
            "y": torch.tensor(y_output, dtype=torch.float32),
        }
    # CASE 2: FLUID (Navier-Stokes from CSV)
    def _preload_fluid_csv(self):
        """
        Load Navier-Stokes data (step geometries):
          1. Geometry: x_data, y_data (diffeomorphism maps).
          2. Physics: velocity fields u, v and pressure p.
          3. Parameter: Reynolds number Re (broadcast to 2D).
        """
        print(f"Loading fluid data from {self.data_path}...")

        raw_x  = np.loadtxt(os.path.join(self.data_path, "x_data.csv"),
                            delimiter=",")
        raw_y  = np.loadtxt(os.path.join(self.data_path, "y_data.csv"),
                            delimiter=",")
        raw_u  = np.loadtxt(os.path.join(self.data_path, "u_data.csv"),
                            delimiter=",")
        raw_v  = np.loadtxt(os.path.join(self.data_path, "v_data.csv"),
                            delimiter=",")
        raw_p  = np.loadtxt(os.path.join(self.data_path, "p_data.csv"),
                            delimiter=",")
        raw_re = np.loadtxt(os.path.join(self.data_path, "re_data.csv"),
                            delimiter=",")

        num_samples = raw_u.shape[0]
        S = 128

        self.ram_cache['x'] = raw_x.reshape(num_samples, S, S)
        self.ram_cache['y'] = raw_y.reshape(num_samples, S, S)

        if raw_re.ndim == 1:
            raw_re = raw_re.reshape(-1, 1)
        re_expanded = np.repeat(raw_re, S * S, axis=1).reshape(
            num_samples, S, S)
        self.ram_cache['re'] = re_expanded

        self.ram_cache['u'] = raw_u.reshape(num_samples, S, S)
        self.ram_cache['v'] = raw_v.reshape(num_samples, S, S)
        self.ram_cache['p'] = raw_p.reshape(num_samples, S, S)

        # Compute StandardScaler for each field
        self.scalers = {
            k: StandardScaler(self.ram_cache[k].reshape(-1))
            for k in ['x', 'y', 're', 'u', 'v', 'p']
        }

        print(f"[OK] Loaded {num_samples} fluid scenarios.")

    def _get_fluid_sample(self, scenario_idx):
        """
        Case 2: Navier-Stokes with Z-score normalization.
        Input:  [X, Y, Re] (3 channels)
        Output: [U, V, P] (3 channels)
        """
        x_raw = self.ram_cache['x'][scenario_idx]
        y_raw = self.ram_cache['y'][scenario_idx]
        re_raw = self.ram_cache['re'][scenario_idx]
        u_raw = self.ram_cache['u'][scenario_idx]
        v_raw = self.ram_cache['v'][scenario_idx]
        p_raw = self.ram_cache['p'][scenario_idx]

        x_input = np.stack([
            self.scalers['x'].encode(x_raw),
            self.scalers['y'].encode(y_raw),
            self.scalers['re'].encode(re_raw),
        ], axis=-1)
        y_output = np.stack([
            self.scalers['u'].encode(u_raw),
            self.scalers['v'].encode(v_raw),
            self.scalers['p'].encode(p_raw),
        ], axis=-1)

        return {
            "x": torch.tensor(x_input, dtype=torch.float32),
            "y": torch.tensor(y_output, dtype=torch.float32),
        }

    # CASE 3: RESERVOIR (Well simulation from HDF5)
    def _get_reservoir_sample(self, scenario_idx):
        """
        Case 3: Seq2Seq reservoir simulation.

        Input  (39 channels): [X, Y, P0, S0, Src_0...Src_34]
        Output (72 channels): [P1, S1, P2, S2, ..., P36, S36]
                               (interleaved: even = P, odd = S)
        """
        data = self._h5_file['dataset_128'][:, scenario_idx, :, :, :]

        x_map = self._h5_file['x_map'][scenario_idx]
        y_map = self._h5_file['y_map'][scenario_idx]

        p0 = data[0, :, :, 0] / self.p_scale
        s0 = data[0, :, :, 1]
        sources = data[:, :, :, 2] / self.s_scale
        sources_part = sources[:35]

        x_input = np.concatenate([
            x_map[..., None], y_map[..., None],
            p0[..., None], s0[..., None],
            np.transpose(sources_part, (1, 2, 0)),
        ], axis=-1).astype(np.float32)

        p_seq = data[1:, :, :, 0] / self.p_scale
        s_seq = data[1:, :, :, 1]

        y_output = np.zeros((128, 128, 72), dtype=np.float32)
        y_output[..., 0::2] = np.transpose(p_seq, (1, 2, 0))
        y_output[..., 1::2] = np.transpose(s_seq, (1, 2, 0))

        return {
            "x": torch.tensor(x_input, dtype=torch.float32),
            "y": torch.tensor(y_output, dtype=torch.float32),
        }

    def __getitem__(self, idx):
        scenario_idx = self.indices[idx]

        if self.case_type == "reservoir" and self._h5_file is None:
            self._h5_file = h5py.File(self.data_path, 'r', swmr=True)

        if self.case_type == "darcy":
            return self._get_darcy_sample(scenario_idx)
        elif self.case_type == "fluid":
            return self._get_fluid_sample(scenario_idx)
        elif self.case_type == "reservoir":
            return self._get_reservoir_sample(scenario_idx)
        else:
            raise ValueError(f"Unknown case type: {self.case_type}")

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parents[2]))

from datasets import load_from_disk
from neuralop.layers.channel_mlp import ChannelMLP


def load_post_lift_mamba_lifting():
    module_path = Path(__file__).resolve().parents[2] / "muno" / "models" / "mamba_fno.py"
    spec = importlib.util.spec_from_file_location("_muno_mamba_fno", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load mamba_fno module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PostLiftMambaLifting


PostLiftMambaLifting = load_post_lift_mamba_lifting()


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_index(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def tensor_stats(tensor):
    return {
        "shape": list(tensor.shape),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def format_seconds(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{int(rem):02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m"


class CombustionObservedWindows(Dataset):
    def __init__(
        self,
        root,
        source,
        index_entries,
        time_window,
        time_stride=1,
        max_samples=None,
        spatial_size=None,
        mean=None,
        std=None,
        cache_rows=2,
        seed=0,
    ):
        if source != "real":
            raise NotImplementedError(
                "This script currently trains on combustion real/observed. "
                "Numerical fields are stored as ~2GB row blobs and need a separate streaming path."
            )

        self.root = Path(root)
        self.source = source
        self.time_window = int(time_window)
        self.time_stride = int(time_stride)
        self.spatial_size = spatial_size
        self.mean = mean
        self.std = std
        self.cache_rows = max(1, int(cache_rows))
        self._row_cache = OrderedDict()

        self.ds = load_from_disk(str(self.root / "hf_dataset" / source))
        self.row_by_sim_id = {self.ds[i]["sim_id"]: i for i in range(len(self.ds))}

        entries = [
            e for e in index_entries
            if isinstance(e, dict) and e.get("sim_id") in self.row_by_sim_id
        ]
        if not entries:
            entries = []
            for row_idx in range(len(self.ds)):
                row = self.ds[row_idx]
                t = int(row["shape_t"])
                step = max(1, self.time_window)
                for start in range(0, max(1, t - self.time_window), step):
                    entries.append({"sim_id": row["sim_id"], "time_id": start})

        rng = random.Random(seed)
        rng.shuffle(entries)
        if max_samples is not None and max_samples > 0:
            entries = entries[:max_samples]
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def _decode_observed(self, row_idx):
        if row_idx in self._row_cache:
            arr = self._row_cache.pop(row_idx)
            self._row_cache[row_idx] = arr
            return arr

        row = self.ds[row_idx]
        shape = (int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"]))
        arr = np.frombuffer(row["observed"], dtype=np.float32).reshape(shape)
        self._row_cache[row_idx] = arr
        while len(self._row_cache) > self.cache_rows:
            self._row_cache.popitem(last=False)
        return arr

    def _window(self, arr, start):
        start = int(start)
        if start < 0:
            start = 0

        needed = self.time_window * self.time_stride
        if start + needed > arr.shape[0]:
            start = max(0, arr.shape[0] - needed)

        idx = start + np.arange(self.time_window) * self.time_stride
        idx = np.clip(idx, 0, arr.shape[0] - 1)
        return arr[idx]

    def __getitem__(self, idx):
        entry = self.entries[idx]
        row_idx = self.row_by_sim_id[entry["sim_id"]]
        arr = self._decode_observed(row_idx)
        window = self._window(arr, entry.get("time_id", 0))

        y = torch.from_numpy(np.array(window, copy=True)).float().unsqueeze(0)
        if self.spatial_size is not None and y.shape[-1] != self.spatial_size:
            y = F.interpolate(
                y.unsqueeze(0),
                size=(self.time_window, self.spatial_size, self.spatial_size),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

        if self.mean is not None and self.std is not None:
            y = (y - self.mean) / self.std

        x0 = y[:, 0:1].expand(-1, y.shape[1], -1, -1)
        return {
            "x": x0,
            "y": y,
            "sim_id": entry["sim_id"],
            "time_id": int(entry.get("time_id", 0)),
        }


class MemmapWindowDataset(Dataset):
    def __init__(self, cache_dir, split, x_stats=None, y_stats=None, mask_prob=0.0):
        self.cache_dir = Path(cache_dir)
        self.split = split
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing cache metadata: {metadata_path}")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        split_meta = self.metadata["splits"][split]
        self.x_shape = tuple(split_meta["x_shape"])
        self.y_shape = tuple(split_meta["y_shape"])
        self.x_path = self.cache_dir / split_meta["x_file"]
        self.y_path = self.cache_dir / split_meta["y_file"]
        self.source = split_meta.get("source", self.metadata.get("source", ""))
        self.mask_prob = float(mask_prob)

        self.x = np.memmap(self.x_path, dtype=np.float32, mode="r", shape=self.x_shape)
        self.y = np.memmap(self.y_path, dtype=np.float32, mode="r", shape=self.y_shape)
        self.x_stats = x_stats
        self.y_stats = y_stats

    def __len__(self):
        return self.x_shape[0]

    @property
    def input_channels(self):
        return self.x_shape[1]

    @property
    def output_channels(self):
        return self.y_shape[1]

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.x[idx], copy=True)).float()
        y = torch.from_numpy(np.array(self.y[idx], copy=True)).float()

        if (
            self.split == "train"
            and self.source == "numerical"
            and self.mask_prob > 0
            and x.shape[0] > 1
            and random.random() < self.mask_prob
        ):
            x[1:] = 0
            y[1:] = 0

        if self.x_stats is not None:
            mean, std = self.x_stats
            x = (x - mean) / std
        if self.y_stats is not None:
            mean, std = self.y_stats
            y = (y - mean) / std

        return {"x": x, "y": y}


def estimate_dataset_stats(dataset, max_samples=64, label="dataset"):
    x_sum = None
    x_sq_sum = None
    y_sum = None
    y_sq_sum = None
    x_count = 0
    y_count = 0
    n = min(len(dataset), max_samples)
    start_time = time.time()
    print(f"[stats:{label}] start samples={n}", flush=True)
    for i in range(n):
        sample_start = time.time()
        sample = dataset[i]
        x = sample["x"].double()
        y = sample["y"].double()
        x_reduce_dims = tuple(d for d in range(x.ndim) if d != 0)
        y_reduce_dims = tuple(d for d in range(y.ndim) if d != 0)

        cur_x_sum = x.sum(dim=x_reduce_dims)
        cur_x_sq_sum = x.square().sum(dim=x_reduce_dims)
        cur_y_sum = y.sum(dim=y_reduce_dims)
        cur_y_sq_sum = y.square().sum(dim=y_reduce_dims)

        x_sum = cur_x_sum if x_sum is None else x_sum + cur_x_sum
        x_sq_sum = cur_x_sq_sum if x_sq_sum is None else x_sq_sum + cur_x_sq_sum
        y_sum = cur_y_sum if y_sum is None else y_sum + cur_y_sum
        y_sq_sum = cur_y_sq_sum if y_sq_sum is None else y_sq_sum + cur_y_sq_sum
        x_count += int(x[0].numel())
        y_count += int(y[0].numel())

        elapsed = time.time() - start_time
        per_sample = elapsed / max(1, i + 1)
        eta = per_sample * max(0, n - i - 1)
        print(
            f"[stats:{label}] {i + 1}/{n} "
            f"sample={format_seconds(time.time() - sample_start)} "
            f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
            flush=True,
        )

    x_mean = x_sum / max(1, x_count)
    y_mean = y_sum / max(1, y_count)
    x_var = x_sq_sum / max(1, x_count) - x_mean.square()
    y_var = y_sq_sum / max(1, y_count) - y_mean.square()
    x_std = torch.where(x_var <= 1e-12, torch.ones_like(x_var), torch.sqrt(torch.clamp(x_var, min=1e-12)))
    y_std = torch.where(y_var <= 1e-12, torch.ones_like(y_var), torch.sqrt(torch.clamp(y_var, min=1e-12)))
    return (
        x_mean.float().view(-1, 1, 1, 1),
        x_std.float().view(-1, 1, 1, 1),
    ), (
        y_mean.float().view(-1, 1, 1, 1),
        y_std.float().view(-1, 1, 1, 1),
    )


def estimate_observed_stats(dataset, max_batches=64, label="observed"):
    values_sum = 0.0
    values_sq_sum = 0.0
    count = 0
    n = min(len(dataset), max_batches)
    start_time = time.time()
    print(f"[stats:{label}] start samples={n}", flush=True)
    for i in range(n):
        sample_start = time.time()
        sample = dataset[i]
        y = sample["y"]
        values_sum += y.double().sum().item()
        values_sq_sum += y.double().square().sum().item()
        count += y.numel()
        elapsed = time.time() - start_time
        per_sample = elapsed / max(1, i + 1)
        eta = per_sample * max(0, n - i - 1)
        print(
            f"[stats:{label}] {i + 1}/{n} "
            f"sample={format_seconds(time.time() - sample_start)} "
            f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
            flush=True,
        )
    mean = values_sum / max(1, count)
    var = values_sq_sum / max(1, count) - mean * mean
    std = math.sqrt(max(var, 1e-12))
    return float(mean), float(std)


def decode_observed(row):
    shape = (int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"]))
    return np.frombuffer(row["observed"], dtype=np.float32).reshape(shape)


def decode_numerical(row):
    shape = (
        int(row["shape_t"]),
        int(row["shape_h"]),
        int(row["shape_w"]),
        int(row["numerical_channels"]),
    )
    return np.frombuffer(row["numerical"], dtype=np.float32).reshape(shape)


class CombustionRealPDEWindows(Dataset):
    def __init__(
        self,
        root,
        source,
        split,
        in_step=20,
        out_step=20,
        n_autoregressive=1,
        sub_s=2,
        mask_prob=0.5,
        max_samples=None,
        x_stats=None,
        y_stats=None,
        seed=0,
        verbose=False,
        hdf5_rdcc_nbytes=1048576,
        hdf5_read_retries=3,
    ):
        self.root = Path(root)
        self.source = source
        self.split = split
        self.in_step = int(in_step)
        self.out_step = int(out_step) * int(n_autoregressive)
        self.n_autoregressive = int(n_autoregressive)
        self.horizon = self.in_step + self.out_step
        self.sub_s = int(sub_s)
        self.mask_prob = float(mask_prob)
        self.x_stats = x_stats
        self.y_stats = y_stats
        self.verbose = bool(verbose)
        self.hdf5_rdcc_nbytes = int(hdf5_rdcc_nbytes)
        self.hdf5_read_retries = int(hdf5_read_retries)
        self._h5_cache = OrderedDict()
        self._h5_cache_size = 4
        self._real_trajectory_cache = OrderedDict()
        self._real_trajectory_cache_size = 2

        if self.source not in {"real", "numerical"}:
            raise ValueError(f"Unsupported combustion source: {self.source}")
        if self.in_step != self.out_step:
            raise ValueError(
                "The frozen UNO core keeps the temporal length unchanged. "
                "Use --in-step equal to --out-step * --n-autoregressive."
            )

        self.ds = load_from_disk(str(self.root / "hf_dataset" / self.source))
        sim_ids = list(self.ds["sim_id"])
        shape_ts = list(self.ds["shape_t"])
        self.row_by_sim_id = {sim_id: i for i, sim_id in enumerate(sim_ids)}
        self.shape_t_by_sim_id = {sim_id: int(shape_t) for sim_id, shape_t in zip(sim_ids, shape_ts)}

        hf_root = self.root / "hf_dataset"
        entries = read_index(hf_root / f"{self.split}_index_{self.source}.json")
        if not entries and self.split == "val" and self.source == "numerical":
            entries = read_index(hf_root / "train_index_numerical.json")

        entries = [
            e
            for e in entries
            if isinstance(e, dict)
            and e.get("sim_id") in self.row_by_sim_id
            and int(e.get("time_id", 0)) + self.horizon < self.shape_t_by_sim_id[e["sim_id"]]
        ]
        rng = random.Random(seed)
        rng.shuffle(entries)
        if max_samples is not None and max_samples > 0:
            entries = entries[:max_samples]
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    @property
    def input_channels(self):
        return 16

    @property
    def output_channels(self):
        return 16

    def __getitem__(self, idx):
        entry = self.entries[idx]
        time_id = int(entry.get("time_id", 0))
        item_start = time.time()
        row = self.ds[self.row_by_sim_id[entry["sim_id"]]]
        observed = decode_observed(row)[
            time_id : time_id + self.horizon,
            :: self.sub_s,
            :: self.sub_s,
        ]
        observed = np.asarray(observed, dtype=np.float32)[..., None]

        if self.source == "real" or random.random() < self.mask_prob:
            numerical = np.zeros((*observed.shape[:3], 15), dtype=np.float32)
        else:
            if self.verbose:
                print(
                    f"[dataset:{self.split}/{self.source}] idx={idx} sim_id={entry['sim_id']} "
                    f"time_id={time_id} reading numerical trajectory blob...",
                    flush=True,
                )
            num_start = time.time()
            numerical = decode_numerical(row)[
                time_id : time_id + self.horizon,
                :: self.sub_s,
                :: self.sub_s,
                :,
            ]
            numerical = np.asarray(numerical, dtype=np.float32)
            if self.verbose:
                print(
                    f"[dataset:{self.split}/{self.source}] idx={idx} numerical read "
                    f"done in {format_seconds(time.time() - num_start)}",
                    flush=True,
                )

        data = np.concatenate([observed, numerical], axis=-1)
        x = torch.from_numpy(np.moveaxis(data[: self.in_step], -1, 0).copy()).float()
        y = torch.from_numpy(np.moveaxis(data[self.in_step :], -1, 0).copy()).float()

        if self.x_stats is not None:
            mean, std = self.x_stats
            x = (x - mean) / std
        if self.y_stats is not None:
            mean, std = self.y_stats
            y = (y - mean) / std

        if self.verbose:
            print(
                f"[dataset:{self.split}/{self.source}] idx={idx} total={format_seconds(time.time() - item_start)} "
                f"x_shape={tuple(x.shape)} y_shape={tuple(y.shape)}",
                flush=True,
            )

        return {
            "x": x,
            "y": y,
            "sim_id": entry["sim_id"],
            "time_id": time_id,
        }


class CombustionRealPDEHDF5Windows(Dataset):
    def __init__(
        self,
        root,
        source,
        split,
        in_step=20,
        out_step=20,
        n_autoregressive=1,
        sub_s=2,
        mask_prob=0.5,
        max_samples=None,
        x_stats=None,
        y_stats=None,
        seed=0,
        verbose=False,
        hdf5_rdcc_nbytes=1048576,
        hdf5_read_retries=3,
    ):
        self.root = Path(root)
        self.source = source
        self.split = split
        self.in_step = int(in_step)
        self.out_step = int(out_step) * int(n_autoregressive)
        self.n_autoregressive = int(n_autoregressive)
        self.horizon = self.in_step + self.out_step
        self.sub_s = int(sub_s)
        self.mask_prob = float(mask_prob)
        self.x_stats = x_stats
        self.y_stats = y_stats
        self.verbose = bool(verbose)
        self.hdf5_rdcc_nbytes = int(hdf5_rdcc_nbytes)
        self.hdf5_read_retries = int(hdf5_read_retries)
        self._h5_cache = OrderedDict()
        self._h5_cache_size = 4

        if self.source not in {"real", "numerical"}:
            raise ValueError(f"Unsupported combustion source: {self.source}")
        if self.in_step != self.out_step:
            raise ValueError(
                "The frozen UNO core keeps the temporal length unchanged. "
                "Use --in-step equal to --out-step * --n-autoregressive."
            )

        self.shape_t_by_sim_id = self._scan_shape_t()
        hf_root = self.root / "hf_dataset"
        entries = read_index(hf_root / f"{self.split}_index_{self.source}.json")
        if not entries and self.split == "val" and self.source == "numerical":
            entries = read_index(hf_root / "train_index_numerical.json")

        entries = [
            e
            for e in entries
            if isinstance(e, dict)
            and e.get("sim_id") in self.shape_t_by_sim_id
            and int(e.get("time_id", 0)) + self.horizon < self.shape_t_by_sim_id[e["sim_id"]]
        ]
        rng = random.Random(seed)
        rng.shuffle(entries)
        if max_samples is not None and max_samples > 0:
            entries = entries[:max_samples]
        self.entries = entries

    def _file_name(self, sim_id):
        sim_id = str(sim_id)
        return sim_id if sim_id.endswith(".h5") else f"{sim_id}.h5"

    def _scan_shape_t(self):
        shape_t_by_sim_id = {}
        data_dir = self.root / ("real" if self.source == "real" else "surrogate")
        dataset_name = "trajectory" if self.source == "real" else "measured_data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Missing HDF5 data directory: {data_dir}")

        for path in data_dir.glob("*.h5"):
            try:
                with h5py.File(path, "r") as f:
                    if dataset_name not in f:
                        continue
                    shape_t_by_sim_id[path.name] = int(f[dataset_name].shape[0])
            except OSError as exc:
                raise OSError(f"Cannot read HDF5 file {path}: {exc}") from exc
        return shape_t_by_sim_id

    def __len__(self):
        return len(self.entries)

    @property
    def input_channels(self):
        return 16

    @property
    def output_channels(self):
        return 16

    def _open_h5(self, path):
        if not hasattr(self, "_h5_cache"):
            self._h5_cache = OrderedDict()
            self._h5_cache_size = 4
        path = Path(path)
        key = str(path)
        if key in self._h5_cache:
            f = self._h5_cache.pop(key)
            self._h5_cache[key] = f
            return f

        f = h5py.File(path, "r", rdcc_nbytes=self.hdf5_rdcc_nbytes)
        self._h5_cache[key] = f
        while len(self._h5_cache) > self._h5_cache_size:
            _, old_f = self._h5_cache.popitem(last=False)
            old_f.close()
        return f

    def close(self):
        while self._h5_cache:
            _, f = self._h5_cache.popitem(last=False)
            f.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _read_h5_window(self, path, dataset_name, time_id):
        last_error = None
        for attempt in range(self.hdf5_read_retries + 1):
            try:
                f = self._open_h5(path)
                return np.asarray(
                    f[dataset_name][
                        time_id : time_id + self.horizon,
                        :: self.sub_s,
                        :: self.sub_s,
                        ...,
                    ],
                    dtype=np.float32,
                )
            except OSError as exc:
                last_error = exc
                self._h5_cache.pop(str(Path(path)), None)
                if attempt >= self.hdf5_read_retries:
                    break
                wait_s = 0.25 * (attempt + 1)
                if self.verbose:
                    print(
                        f"[hdf5:{self.split}/{self.source}] retry {attempt + 1}/"
                        f"{self.hdf5_read_retries} after read error: {path.name} "
                        f"time_id={time_id} error={exc}",
                        flush=True,
                    )
                time.sleep(wait_s)
        raise OSError(
            f"Failed to read HDF5 window path={path}, dataset={dataset_name}, "
            f"time_id={time_id}, horizon={self.horizon}, sub_s={self.sub_s}: {last_error}"
        )

    def _read_real_window_cached(self, path, dataset_name, time_id):
        if not hasattr(self, "_real_trajectory_cache"):
            self._real_trajectory_cache = OrderedDict()
            self._real_trajectory_cache_size = 2

        path = Path(path)
        key = str(path)
        if key in self._real_trajectory_cache:
            arr = self._real_trajectory_cache.pop(key)
            self._real_trajectory_cache[key] = arr
        else:
            f = self._open_h5(path)
            arr = np.asarray(
                f[dataset_name][
                    :,
                    :: self.sub_s,
                    :: self.sub_s,
                    ...,
                ],
                dtype=np.float32,
            )
            self._real_trajectory_cache[key] = arr
            while len(self._real_trajectory_cache) > self._real_trajectory_cache_size:
                self._real_trajectory_cache.popitem(last=False)

        return arr[time_id : time_id + self.horizon]

    def __getitem__(self, idx):
        entry = self.entries[idx]
        sim_id = entry["sim_id"]
        file_name = self._file_name(sim_id)
        time_id = int(entry.get("time_id", 0))
        item_start = time.time()

        if self.source == "real":
            observed = self._read_real_window_cached(self.root / "real" / file_name, "trajectory", time_id)
            observed = observed[..., None]
            numerical = np.zeros((*observed.shape[:3], 15), dtype=np.float32)
        else:
            observed = self._read_h5_window(self.root / "surrogate" / file_name, "measured_data", time_id)
            observed = observed[..., None]
            if random.random() < self.mask_prob:
                numerical = np.zeros((*observed.shape[:3], 15), dtype=np.float32)
            else:
                if self.verbose:
                    print(
                        f"[hdf5:{self.split}/{self.source}] idx={idx} sim_id={sim_id} "
                        f"time_id={time_id} reading numerical h5 window...",
                        flush=True,
                    )
                num_start = time.time()
                numerical = self._read_h5_window(
                    self.root / "numerical" / file_name,
                    "measured_data",
                    time_id,
                )
                if self.verbose:
                    print(
                        f"[hdf5:{self.split}/{self.source}] idx={idx} numerical window read "
                        f"done in {format_seconds(time.time() - num_start)}",
                        flush=True,
                    )

        data = np.concatenate([observed, numerical], axis=-1)
        x = torch.from_numpy(np.moveaxis(data[: self.in_step], -1, 0).copy()).float()
        y = torch.from_numpy(np.moveaxis(data[self.in_step :], -1, 0).copy()).float()

        if self.x_stats is not None:
            mean, std = self.x_stats
            x = (x - mean) / std
        if self.y_stats is not None:
            mean, std = self.y_stats
            y = (y - mean) / std

        if self.verbose:
            print(
                f"[hdf5:{self.split}/{self.source}] idx={idx} total={format_seconds(time.time() - item_start)} "
                f"x_shape={tuple(x.shape)} y_shape={tuple(y.shape)}",
                flush=True,
            )

        return {
            "x": x,
            "y": y,
            "sim_id": sim_id,
            "time_id": time_id,
        }


def select_entries(index_entries, row_by_sim_id, max_samples, seed):
    entries = [
        e for e in index_entries
        if isinstance(e, dict) and e.get("sim_id") in row_by_sim_id
    ]
    rng = random.Random(seed)
    rng.shuffle(entries)
    if max_samples is not None and max_samples > 0:
        entries = entries[:max_samples]
    return entries


def make_window_indices(start, total_t, time_window, time_stride):
    needed = time_window * time_stride
    if start + needed > total_t:
        start = max(0, total_t - needed)
    idx = start + np.arange(time_window) * time_stride
    return np.clip(idx, 0, total_t - 1)


def resize_window_np(arr_cthw, time_window, spatial_size):
    if spatial_size is None or arr_cthw.shape[-1] == spatial_size:
        return arr_cthw
    tensor = torch.from_numpy(np.array(arr_cthw, copy=True)).float().unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(time_window, spatial_size, spatial_size),
        mode="trilinear",
        align_corners=False,
    )
    return tensor.squeeze(0).numpy().astype(np.float32, copy=False)


def build_memmap_cache(
    data_root,
    cache_dir,
    source,
    input_mode,
    time_window,
    time_stride,
    spatial_size,
    max_train_samples,
    max_val_samples,
    seed,
):
    if source != "numerical":
        raise NotImplementedError("Memmap cache builder is intended for source='numerical'.")
    if input_mode not in {"numerical", "observed_numerical", "observed"}:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    data_root = Path(data_root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(str(data_root / "hf_dataset" / source))
    row_by_sim_id = {ds[i]["sim_id"]: i for i in range(len(ds))}
    hf_root = data_root / "hf_dataset"
    train_entries = select_entries(
        read_index(hf_root / f"train_index_{source}.json"),
        row_by_sim_id,
        max_train_samples,
        seed,
    )
    val_entries = select_entries(
        read_index(hf_root / f"val_index_{source}.json"),
        row_by_sim_id,
        max_val_samples,
        seed + 1,
    )
    if not val_entries:
        holdout = max(1, min(len(train_entries) // 10, max_val_samples))
        val_entries = train_entries[:holdout]
        train_entries = train_entries[holdout:]
    if not train_entries or not val_entries:
        raise RuntimeError(f"Empty cache entries: train={len(train_entries)}, val={len(val_entries)}")

    if input_mode == "observed":
        in_channels = 1
    elif input_mode == "numerical":
        in_channels = 15
    else:
        in_channels = 16
    out_channels = 1

    def write_split(split_name, entries):
        x_shape = (len(entries), in_channels, time_window, spatial_size, spatial_size)
        y_shape = (len(entries), out_channels, time_window, spatial_size, spatial_size)
        x_file = f"{split_name}_x.dat"
        y_file = f"{split_name}_y.dat"
        x_mm = np.memmap(cache_dir / x_file, dtype=np.float32, mode="w+", shape=x_shape)
        y_mm = np.memmap(cache_dir / y_file, dtype=np.float32, mode="w+", shape=y_shape)

        grouped = OrderedDict()
        for out_idx, entry in enumerate(entries):
            grouped.setdefault(entry["sim_id"], []).append((out_idx, entry))

        for sim_idx, (sim_id, sim_entries) in enumerate(grouped.items(), start=1):
            row = ds[row_by_sim_id[sim_id]]
            observed = decode_observed(row)
            numerical = None
            if input_mode in {"numerical", "observed_numerical"}:
                numerical = decode_numerical(row)
            print(f"{split_name}: {sim_idx}/{len(grouped)} {sim_id} windows={len(sim_entries)}")

            for out_idx, entry in sim_entries:
                idx = make_window_indices(
                    int(entry.get("time_id", 0)),
                    observed.shape[0],
                    time_window,
                    time_stride,
                )
                y = observed[idx][None, ...]
                y = resize_window_np(y, time_window, spatial_size)

                if input_mode == "observed":
                    x = np.broadcast_to(y[:, 0:1], y.shape).astype(np.float32, copy=True)
                elif input_mode == "numerical":
                    x = np.moveaxis(numerical[idx], -1, 0).astype(np.float32, copy=False)
                    x = resize_window_np(x, time_window, spatial_size)
                else:
                    obs_x = np.broadcast_to(y[:, 0:1], y.shape).astype(np.float32, copy=True)
                    num_x = np.moveaxis(numerical[idx], -1, 0).astype(np.float32, copy=False)
                    num_x = resize_window_np(num_x, time_window, spatial_size)
                    x = np.concatenate([obs_x, num_x], axis=0)

                x_mm[out_idx] = x
                y_mm[out_idx] = y

            x_mm.flush()
            y_mm.flush()

        return {
            "x_file": x_file,
            "y_file": y_file,
            "x_shape": list(x_shape),
            "y_shape": list(y_shape),
            "entries": entries,
        }

    metadata = {
        "format": "combustion_memmap_windows_v1",
        "data_root": str(data_root),
        "source": source,
        "input_mode": input_mode,
        "time_window": time_window,
        "time_stride": time_stride,
        "spatial_size": spatial_size,
        "splits": {
            "train": write_split("train", train_entries),
            "val": write_split("val", val_entries),
        },
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Built cache at {cache_dir}")
    return cache_dir


def build_realpde_memmap_cache(
    data_root,
    cache_dir,
    source,
    val_source,
    in_step,
    out_step,
    n_autoregressive,
    sub_s_real,
    sub_s_numerical,
    max_train_samples,
    max_val_samples,
    seed,
):
    data_root = Path(data_root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    in_step = int(in_step)
    out_step = int(out_step) * int(n_autoregressive)
    if in_step != out_step:
        raise ValueError(
            "The frozen UNO core keeps the temporal length unchanged. "
            "Use --in-step equal to --out-step * --n-autoregressive."
        )
    horizon = in_step + out_step

    def load_entries(ds, split_source, split, max_samples, split_seed):
        sim_ids = list(ds["sim_id"])
        shape_ts = list(ds["shape_t"])
        shape_hs = list(ds["shape_h"])
        shape_ws = list(ds["shape_w"])
        row_by_sim_id = {sim_id: i for i, sim_id in enumerate(sim_ids)}
        shape_t_by_sim_id = {sim_id: int(shape_t) for sim_id, shape_t in zip(sim_ids, shape_ts)}
        shape_hw_by_sim_id = {
            sim_id: (int(shape_h), int(shape_w))
            for sim_id, shape_h, shape_w in zip(sim_ids, shape_hs, shape_ws)
        }

        entries = read_index(data_root / "hf_dataset" / f"{split}_index_{split_source}.json")
        if not entries and split == "val" and split_source == "numerical":
            entries = read_index(data_root / "hf_dataset" / "train_index_numerical.json")

        entries = [
            e
            for e in entries
            if isinstance(e, dict)
            and e.get("sim_id") in row_by_sim_id
            and int(e.get("time_id", 0)) + horizon < shape_t_by_sim_id[e["sim_id"]]
        ]
        rng = random.Random(split_seed)
        rng.shuffle(entries)
        if max_samples is not None and max_samples > 0:
            entries = entries[:max_samples]
        return entries, row_by_sim_id, shape_hw_by_sim_id

    def write_split(split, split_source, max_samples, split_seed):
        split_start = time.time()
        ds = load_from_disk(str(data_root / "hf_dataset" / split_source))
        entries, row_by_sim_id, shape_hw_by_sim_id = load_entries(ds, split_source, split, max_samples, split_seed)
        if not entries:
            raise RuntimeError(f"Empty RealPDEBench cache split: split={split}, source={split_source}")

        sub_s = int(sub_s_real if split_source == "real" else sub_s_numerical)
        first_h, first_w = shape_hw_by_sim_id[entries[0]["sim_id"]]
        spatial_h = len(range(0, first_h, sub_s))
        spatial_w = len(range(0, first_w, sub_s))

        x_shape = (len(entries), 16, in_step, spatial_h, spatial_w)
        y_shape = (len(entries), 16, out_step, spatial_h, spatial_w)
        x_file = f"{split}_x.dat"
        y_file = f"{split}_y.dat"
        x_mm = np.memmap(cache_dir / x_file, dtype=np.float32, mode="w+", shape=x_shape)
        y_mm = np.memmap(cache_dir / y_file, dtype=np.float32, mode="w+", shape=y_shape)

        grouped = OrderedDict()
        for out_idx, entry in enumerate(entries):
            grouped.setdefault(entry["sim_id"], []).append((out_idx, entry))

        print(
            f"[cache:{split}/{split_source}] start windows={len(entries)} sims={len(grouped)} "
            f"x_shape={x_shape} y_shape={y_shape}",
            flush=True,
        )
        processed = 0
        for sim_idx, (sim_id, sim_entries) in enumerate(grouped.items(), start=1):
            sim_start = time.time()
            row = ds[row_by_sim_id[sim_id]]
            observed = decode_observed(row)
            numerical = None
            if split_source == "numerical":
                print(
                    f"[cache:{split}/{split_source}] sim {sim_idx}/{len(grouped)} {sim_id} "
                    f"reading numerical trajectory...",
                    flush=True,
                )
                num_start = time.time()
                numerical = decode_numerical(row)
                print(
                    f"[cache:{split}/{split_source}] sim {sim_id} numerical ready in "
                    f"{format_seconds(time.time() - num_start)}",
                    flush=True,
                )

            for out_idx, entry in sim_entries:
                time_id = int(entry.get("time_id", 0))
                obs = observed[
                    time_id : time_id + horizon,
                    ::sub_s,
                    ::sub_s,
                ][..., None]
                if numerical is None:
                    num = np.zeros((*obs.shape[:3], 15), dtype=np.float32)
                else:
                    num = numerical[
                        time_id : time_id + horizon,
                        ::sub_s,
                        ::sub_s,
                        :,
                    ]
                window = np.concatenate([obs, num], axis=-1).astype(np.float32, copy=False)
                x_mm[out_idx] = np.moveaxis(window[:in_step], -1, 0)
                y_mm[out_idx] = np.moveaxis(window[in_step:], -1, 0)
                processed += 1

            x_mm.flush()
            y_mm.flush()
            elapsed = time.time() - split_start
            per_window = elapsed / max(1, processed)
            eta = per_window * max(0, len(entries) - processed)
            print(
                f"[cache:{split}/{split_source}] sim {sim_idx}/{len(grouped)} done "
                f"windows={processed}/{len(entries)} sim_time={format_seconds(time.time() - sim_start)} "
                f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
                flush=True,
            )

        return {
            "source": split_source,
            "x_file": x_file,
            "y_file": y_file,
            "x_shape": list(x_shape),
            "y_shape": list(y_shape),
            "entries": entries,
        }

    metadata = {
        "format": "combustion_realpde_memmap_windows_v1",
        "data_root": str(data_root),
        "source": source,
        "val_source": val_source,
        "in_step": in_step,
        "out_step": out_step,
        "n_autoregressive": int(n_autoregressive),
        "sub_s_real": int(sub_s_real),
        "sub_s_numerical": int(sub_s_numerical),
        "splits": {
            "train": write_split("train", source, max_train_samples, seed),
            "val": write_split("val", val_source, max_val_samples, seed + 1),
        },
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Built RealPDEBench cache at {cache_dir}", flush=True)
    return cache_dir


class AdapterModel(torch.nn.Module):
    def __init__(self, input_adapter, main_model, output_adapter):
        super().__init__()
        self.input_adapter = input_adapter
        self.main_model = main_model
        self.output_adapter = output_adapter
        for p in self.main_model.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        self.main_model.eval()
        z = self.input_adapter(x)
        z = self.main_model(z)
        return self.output_adapter(z)


def build_model(main_ckpt, input_channels, output_channels, device):
    main = torch_load(main_ckpt, map_location=device)
    for p in main.parameters():
        p.requires_grad_(False)
    main.eval()

    latent_in = int(getattr(main, "in_channels"))
    latent_out = int(getattr(main, "out_channels"))
    if latent_in != latent_out:
        raise ValueError(f"Expected square latent core, got {latent_in}->{latent_out}")

    input_adapter = PostLiftMambaLifting(
        in_channels=input_channels,
        out_channels=latent_in,
        width=latent_in,
        n_dim=3,
        padding=0,
        use_mamba_kwargs=None,
        mamba_fallback_kernel=9,
        positional_embedding="grid",
        non_linearity=F.gelu,
    )
    output_adapter = ChannelMLP(
        in_channels=latent_out,
        out_channels=output_channels,
        hidden_channels=latent_out,
        n_layers=2,
        n_dim=3,
        non_linearity=F.gelu,
    )
    return AdapterModel(input_adapter, main, output_adapter).to(device)


def save_checkpoint(path, model, optimizer, epoch, val_loss, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_adapter": model.input_adapter,
            "output_adapter": model.output_adapter,
            "input_adapter_state_dict": model.input_adapter.state_dict(),
            "output_adapter_state_dict": model.output_adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "metadata": metadata,
        },
        path,
    )


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    train,
    loss_channels=None,
    epoch=0,
    log_every=10,
):
    model.train(train)
    model.main_model.eval()
    if train:
        model.input_adapter.train()
        model.output_adapter.train()
    else:
        model.input_adapter.eval()
        model.output_adapter.eval()

    total_loss = 0.0
    total_items = 0
    phase = "train" if train else "val"
    epoch_start = time.time()
    n_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        batch_start = time.time()
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pred = model(x)
            if loss_channels is not None:
                pred_loss = pred[:, :loss_channels]
                y_loss = y[:, :loss_channels]
            else:
                pred_loss = pred
                y_loss = y
            loss = F.mse_loss(pred_loss, y_loss)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.input_adapter.parameters()) + list(model.output_adapter.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_items += batch_size
        running_loss = total_loss / max(1, total_items)

        should_log = (
            log_every > 0
            and (batch_idx == 1 or batch_idx == n_batches or batch_idx % log_every == 0)
        )
        if should_log:
            elapsed = time.time() - epoch_start
            per_batch = elapsed / max(1, batch_idx)
            eta = per_batch * max(0, n_batches - batch_idx)
            print(
                f"[{phase} epoch {epoch:04d}] {batch_idx}/{n_batches} "
                f"loss={running_loss:.6e} batch={format_seconds(time.time() - batch_start)} "
                f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
                flush=True,
            )

    return total_loss / max(1, total_items)


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def run_updates(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_update,
    val_every,
    log_every,
    loss_channels,
    out_dir,
    metadata,
):
    model.main_model.eval()
    train_iter = infinite_loader(train_loader)
    best_val = float("inf")
    running_loss = 0.0
    running_items = 0
    start_time = time.time()

    for update in range(1, num_update + 1):
        batch_start = time.time()
        model.train(True)
        model.input_adapter.train()
        model.output_adapter.train()
        model.main_model.eval()

        batch = next(train_iter)
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.input_adapter.parameters()) + list(model.output_adapter.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = x.shape[0]
        running_loss += loss.item() * batch_size
        running_items += batch_size

        should_log = (
            log_every > 0
            and (update == 1 or update == num_update or update % log_every == 0)
        )
        if should_log:
            elapsed = time.time() - start_time
            per_update = elapsed / max(1, update)
            eta = per_update * max(0, num_update - update)
            print(
                f"[train update {update:04d}/{num_update:04d}] "
                f"loss={running_loss / max(1, running_items):.6e} "
                f"batch={format_seconds(time.time() - batch_start)} "
                f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
                flush=True,
            )

        if update % val_every == 0 or update == num_update:
            train_loss = running_loss / max(1, running_items)
            val_loss = run_epoch(
                model,
                val_loader,
                optimizer,
                device,
                train=False,
                loss_channels=loss_channels,
                epoch=update,
                log_every=log_every,
            )
            print(
                f"update {update:04d} | train_mse={train_loss:.6e} | val_mse={val_loss:.6e}",
                flush=True,
            )

            save_checkpoint(out_dir / "last_adapters.pt", model, optimizer, update, val_loss, metadata)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(out_dir / "best_adapters.pt", model, optimizer, update, val_loss, metadata)

            running_loss = 0.0
            running_items = 0

    return best_val


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune input/output adapters for RealPDEBench combustion observed data while keeping a UNO core frozen."
    )
    parser.add_argument("--data-root", required=True, help="Path to RealPDEBench/combustion directory.")
    parser.add_argument(
        "--hdf5-root",
        default="",
        help="Optional HDF5 combustion directory with real/, numerical/, surrogate/. Used for --sample-mode realpde.",
    )
    parser.add_argument("--main-ckpt", required=True, help="Path to frozen UNO core checkpoint.")
    parser.add_argument("--out-dir", required=True, help="Directory for adapter checkpoints.")
    parser.add_argument("--source", default="real", choices=["real", "numerical"], help="Combustion HF split to use.")
    parser.add_argument(
        "--val-source",
        default="real",
        choices=["real", "numerical"],
        help="Validation source for --sample-mode realpde. Official combustion validation uses real.",
    )
    parser.add_argument(
        "--sample-mode",
        default="initial",
        choices=["initial", "realpde"],
        help="initial repeats the first observed frame. realpde matches RealPDEBench combustion windows.",
    )
    parser.add_argument(
        "--input-mode",
        default="observed",
        choices=["observed", "numerical", "observed_numerical"],
        help="Input channels for cache-backed training. observed uses initial observed frame; numerical uses 15 fields; observed_numerical uses both.",
    )
    parser.add_argument("--cache-dir", default="", help="Memmap cache directory. Required for numerical training.")
    parser.add_argument("--build-cache", action="store_true", help="Build memmap cache before training.")
    parser.add_argument("--cache-only", action="store_true", help="Build cache and exit without training.")
    parser.add_argument("--time-window", type=int, default=64)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--spatial-size", type=int, default=128)
    parser.add_argument("--in-step", type=int, default=20)
    parser.add_argument("--out-step", type=int, default=20)
    parser.add_argument("--n-autoregressive", type=int, default=1)
    parser.add_argument("--sub-s-real", type=int, default=2)
    parser.add_argument("--sub-s-numerical", type=int, default=2)
    parser.add_argument("--mask-prob", type=float, default=0.5)
    parser.add_argument(
        "--val-loss-channels",
        type=int,
        default=0,
        help="Validation channels to score. 0 means auto: RealPDEBench real validation scores only observed channel.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--num-update",
        type=int,
        default=0,
        help="RealPDEBench-style training updates. If >0, ignores --epochs and trains for this many optimizer steps.",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=0,
        help="Validate every N updates in --num-update mode. 0 means num_update/50, as in RealPDEBench.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"], help="Learning-rate scheduler.")
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=128)
    parser.add_argument("--max-stats-samples", type=int, default=64)
    parser.add_argument("--cache-rows", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--hdf5-rdcc-nbytes", type=int, default=1048576)
    parser.add_argument("--hdf5-read-retries", type=int, default=3)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--log-every", type=int, default=10, help="Print train/val progress every N batches. 0 disables.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose-data", action="store_true", help="Print per-sample dataset read timings.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root)
    hdf5_root = Path(args.hdf5_root) if args.hdf5_root else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    if args.build_cache:
        if cache_dir is None:
            raise ValueError("--cache-dir is required with --build-cache")
        if args.sample_mode == "realpde":
            build_realpde_memmap_cache(
                data_root=data_root,
                cache_dir=cache_dir,
                source=args.source,
                val_source=args.val_source,
                in_step=args.in_step,
                out_step=args.out_step,
                n_autoregressive=args.n_autoregressive,
                sub_s_real=args.sub_s_real,
                sub_s_numerical=args.sub_s_numerical,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                seed=args.seed,
            )
        else:
            build_memmap_cache(
                data_root=data_root,
                cache_dir=cache_dir,
                source=args.source,
                input_mode=args.input_mode,
                time_window=args.time_window,
                time_stride=args.time_stride,
                spatial_size=args.spatial_size,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                seed=args.seed,
            )
        if args.cache_only:
            return

    cache_metadata = None
    if cache_dir is not None and (cache_dir / "metadata.json").exists():
        cache_metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))

    if (
        args.sample_mode == "realpde"
        and cache_metadata is not None
        and cache_metadata.get("format") == "combustion_realpde_memmap_windows_v1"
    ):
        train_raw = MemmapWindowDataset(cache_dir, "train", mask_prob=args.mask_prob)
        (x_mean, x_std), (y_mean, y_std) = estimate_dataset_stats(
            train_raw,
            max_samples=args.max_stats_samples,
            label="realpde-cache-train",
        )
        print(
            "RealPDEBench cache normalization:",
            f"x_mean_shape={tuple(x_mean.shape)}",
            f"y_mean_shape={tuple(y_mean.shape)}",
        )
        train_ds = MemmapWindowDataset(
            cache_dir,
            "train",
            x_stats=(x_mean, x_std),
            y_stats=(y_mean, y_std),
            mask_prob=args.mask_prob,
        )
        val_ds = MemmapWindowDataset(cache_dir, "val", x_stats=(x_mean, x_std), y_stats=(y_mean, y_std))
        input_channels = train_ds.input_channels
        output_channels = train_ds.output_channels
        norm_metadata = {
            "x_mean": x_mean.flatten().tolist(),
            "x_std": x_std.flatten().tolist(),
            "y_mean": y_mean.flatten().tolist(),
            "y_std": y_std.flatten().tolist(),
        }
    elif args.sample_mode == "realpde" and hdf5_root is not None:
        sub_s_train = args.sub_s_real if args.source == "real" else args.sub_s_numerical
        sub_s_val = args.sub_s_real if args.val_source == "real" else args.sub_s_numerical

        train_raw = CombustionRealPDEHDF5Windows(
            hdf5_root,
            args.source,
            "train",
            in_step=args.in_step,
            out_step=args.out_step,
            n_autoregressive=args.n_autoregressive,
            sub_s=sub_s_train,
            mask_prob=args.mask_prob,
            max_samples=args.max_stats_samples,
            seed=args.seed,
            verbose=True,
            hdf5_rdcc_nbytes=args.hdf5_rdcc_nbytes,
            hdf5_read_retries=args.hdf5_read_retries,
        )
        if len(train_raw) == 0:
            raise RuntimeError(
                f"Empty RealPDEBench HDF5 stats dataset for source={args.source}, "
                f"in_step={args.in_step}, out_step={args.out_step}, "
                f"n_autoregressive={args.n_autoregressive}"
            )
        (x_mean, x_std), (y_mean, y_std) = estimate_dataset_stats(
            train_raw,
            max_samples=args.max_stats_samples,
            label=f"realpde-hdf5-{args.source}-train",
        )
        print(
            "RealPDEBench HDF5 normalization:",
            f"x_mean_shape={tuple(x_mean.shape)}",
            f"y_mean_shape={tuple(y_mean.shape)}",
        )

        train_ds = CombustionRealPDEHDF5Windows(
            hdf5_root,
            args.source,
            "train",
            in_step=args.in_step,
            out_step=args.out_step,
            n_autoregressive=args.n_autoregressive,
            sub_s=sub_s_train,
            mask_prob=args.mask_prob,
            max_samples=args.max_train_samples,
            x_stats=(x_mean, x_std),
            y_stats=(y_mean, y_std),
            seed=args.seed,
            verbose=args.verbose_data,
            hdf5_rdcc_nbytes=args.hdf5_rdcc_nbytes,
            hdf5_read_retries=args.hdf5_read_retries,
        )
        val_ds = CombustionRealPDEHDF5Windows(
            hdf5_root,
            args.val_source,
            "val",
            in_step=args.in_step,
            out_step=args.out_step,
            n_autoregressive=args.n_autoregressive,
            sub_s=sub_s_val,
            mask_prob=1.0 if args.val_source == "real" else args.mask_prob,
            max_samples=args.max_val_samples,
            x_stats=(x_mean, x_std),
            y_stats=(y_mean, y_std),
            seed=args.seed + 1,
            verbose=args.verbose_data,
            hdf5_rdcc_nbytes=args.hdf5_rdcc_nbytes,
            hdf5_read_retries=args.hdf5_read_retries,
        )
        input_channels = train_ds.input_channels
        output_channels = train_ds.output_channels
        norm_metadata = {
            "x_mean": x_mean.flatten().tolist(),
            "x_std": x_std.flatten().tolist(),
            "y_mean": y_mean.flatten().tolist(),
            "y_std": y_std.flatten().tolist(),
        }
    elif args.sample_mode == "realpde":
        sub_s_train = args.sub_s_real if args.source == "real" else args.sub_s_numerical
        sub_s_val = args.sub_s_real if args.val_source == "real" else args.sub_s_numerical

        train_raw = CombustionRealPDEWindows(
            data_root,
            args.source,
            "train",
            in_step=args.in_step,
            out_step=args.out_step,
            n_autoregressive=args.n_autoregressive,
            sub_s=sub_s_train,
            mask_prob=args.mask_prob,
            max_samples=args.max_stats_samples,
            seed=args.seed,
            verbose=True,
        )
        if len(train_raw) == 0:
            raise RuntimeError(
                f"Empty RealPDEBench stats dataset for source={args.source}, "
                f"in_step={args.in_step}, out_step={args.out_step}, "
                f"n_autoregressive={args.n_autoregressive}"
            )
        (x_mean, x_std), (y_mean, y_std) = estimate_dataset_stats(
            train_raw,
            max_samples=args.max_stats_samples,
            label=f"realpde-{args.source}-train",
        )
        print(
            "RealPDEBench normalization:",
            f"x_mean_shape={tuple(x_mean.shape)}",
            f"y_mean_shape={tuple(y_mean.shape)}",
        )

        train_ds = CombustionRealPDEWindows(
            data_root,
            args.source,
            "train",
            in_step=args.in_step,
            out_step=args.out_step,
            n_autoregressive=args.n_autoregressive,
            sub_s=sub_s_train,
            mask_prob=args.mask_prob,
            max_samples=args.max_train_samples,
            x_stats=(x_mean, x_std),
            y_stats=(y_mean, y_std),
            seed=args.seed,
            verbose=args.verbose_data,
        )
        val_ds = CombustionRealPDEWindows(
            data_root,
            args.val_source,
            "val",
            in_step=args.in_step,
            out_step=args.out_step,
            n_autoregressive=args.n_autoregressive,
            sub_s=sub_s_val,
            mask_prob=1.0 if args.val_source == "real" else args.mask_prob,
            max_samples=args.max_val_samples,
            x_stats=(x_mean, x_std),
            y_stats=(y_mean, y_std),
            seed=args.seed + 1,
            verbose=args.verbose_data,
        )
        input_channels = train_ds.input_channels
        output_channels = train_ds.output_channels
        norm_metadata = {
            "x_mean": x_mean.flatten().tolist(),
            "x_std": x_std.flatten().tolist(),
            "y_mean": y_mean.flatten().tolist(),
            "y_std": y_std.flatten().tolist(),
        }
    elif cache_metadata is not None:
        train_raw = MemmapWindowDataset(cache_dir, "train")
        (x_mean, x_std), (y_mean, y_std) = estimate_dataset_stats(
            train_raw,
            max_samples=args.max_stats_samples,
            label="memmap-train",
        )
        print(
            "Cache normalization:",
            f"x_mean_shape={tuple(x_mean.shape)}",
            f"y_mean_shape={tuple(y_mean.shape)}",
        )
        train_ds = MemmapWindowDataset(cache_dir, "train", x_stats=(x_mean, x_std), y_stats=(y_mean, y_std))
        val_ds = MemmapWindowDataset(cache_dir, "val", x_stats=(x_mean, x_std), y_stats=(y_mean, y_std))
        input_channels = train_ds.input_channels
        output_channels = train_ds.output_channels
        norm_metadata = {
            "x_mean": x_mean.flatten().tolist(),
            "x_std": x_std.flatten().tolist(),
            "y_mean": y_mean.flatten().tolist(),
            "y_std": y_std.flatten().tolist(),
        }
    else:
        if args.source != "real":
            raise ValueError("source='numerical' requires --cache-dir, usually with --build-cache")

        hf_root = data_root / "hf_dataset"
        train_index = read_index(hf_root / f"train_index_{args.source}.json")
        val_index = read_index(hf_root / f"val_index_{args.source}.json")
        if not val_index:
            holdout = max(1, min(len(train_index) // 10, args.max_val_samples))
            val_index = train_index[:holdout]
            train_index = train_index[holdout:]

        stats_ds = CombustionObservedWindows(
            data_root,
            args.source,
            train_index,
            args.time_window,
            args.time_stride,
            max_samples=args.max_stats_samples,
            spatial_size=args.spatial_size,
            cache_rows=args.cache_rows,
            seed=args.seed,
        )
        mean, std = estimate_observed_stats(
            stats_ds,
            args.max_stats_samples,
            label=f"observed-{args.source}-train",
        )
        print(f"Observed normalization: mean={mean:.6e}, std={std:.6e}")

        train_ds = CombustionObservedWindows(
            data_root,
            args.source,
            train_index,
            args.time_window,
            args.time_stride,
            max_samples=args.max_train_samples,
            spatial_size=args.spatial_size,
            mean=mean,
            std=std,
            cache_rows=args.cache_rows,
            seed=args.seed,
        )
        val_ds = CombustionObservedWindows(
            data_root,
            args.source,
            val_index,
            args.time_window,
            args.time_stride,
            max_samples=args.max_val_samples,
            spatial_size=args.spatial_size,
            mean=mean,
            std=std,
            cache_rows=args.cache_rows,
            seed=args.seed + 1,
        )
        input_channels = 1
        output_channels = 1
        norm_metadata = {"mean": mean, "std": std}

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty train/val datasets: train={len(train_ds)}, val={len(val_ds)}")

    val_loss_channels = args.val_loss_channels if args.val_loss_channels > 0 else None
    if val_loss_channels is None and args.sample_mode == "realpde" and args.val_source == "real":
        val_loss_channels = 1

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(args.main_ckpt, input_channels=input_channels, output_channels=output_channels, device=device)
    optimizer = torch.optim.AdamW(
        list(model.input_adapter.parameters()) + list(model.output_adapter.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler_t_max = args.num_update if args.num_update > 0 else max(1, args.epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_t_max)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda" and not args.no_pin_memory),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, args.prefetch_factor)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    out_dir = Path(args.out_dir)
    metadata = {
        "data_root": str(data_root),
        "hdf5_root": str(hdf5_root) if hdf5_root is not None else "",
        "source": args.source,
        "val_source": args.val_source,
        "sample_mode": args.sample_mode,
        "input_mode": args.input_mode,
        "cache_dir": str(cache_dir) if cache_dir is not None else "",
        "main_ckpt": str(args.main_ckpt),
        "main_type": type(model.main_model).__module__ + "." + type(model.main_model).__name__,
        "main_in_channels": int(getattr(model.main_model, "in_channels")),
        "main_out_channels": int(getattr(model.main_model, "out_channels")),
        "input_channels": input_channels,
        "output_channels": output_channels,
        "time_window": args.time_window,
        "time_stride": args.time_stride,
        "spatial_size": args.spatial_size,
        "in_step": args.in_step,
        "out_step": args.out_step,
        "n_autoregressive": args.n_autoregressive,
        "sub_s_real": args.sub_s_real,
        "sub_s_numerical": args.sub_s_numerical,
        "mask_prob": args.mask_prob,
        "num_update": args.num_update,
        "val_every": args.val_every,
        "scheduler": args.scheduler,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "hdf5_rdcc_nbytes": args.hdf5_rdcc_nbytes,
        "hdf5_read_retries": args.hdf5_read_retries,
        "pin_memory": (device.type == "cuda" and not args.no_pin_memory),
        "val_loss_channels": val_loss_channels,
        "normalization": norm_metadata,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("First train sample:", tensor_stats(train_ds[0]["x"]), tensor_stats(train_ds[0]["y"]))
    print(
        f"Training adapters on {device}; train={len(train_ds)}, val={len(val_ds)}, "
        f"val_loss_channels={val_loss_channels or 'all'}"
    )

    if args.num_update > 0:
        val_every = args.val_every if args.val_every > 0 else max(1, int(args.num_update / 50))
        print(
            f"RealPDEBench-style updates: num_update={args.num_update}, "
            f"val_every={val_every}, batch_size={args.batch_size}",
            flush=True,
        )
        best_val = run_updates(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_update=args.num_update,
            val_every=val_every,
            log_every=args.log_every,
            loss_channels=val_loss_channels,
            out_dir=out_dir,
            metadata=metadata,
        )
    else:
        best_val = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                train=True,
                epoch=epoch,
                log_every=args.log_every,
            )
            if scheduler is not None:
                scheduler.step()
            val_loss = run_epoch(
                model,
                val_loader,
                optimizer,
                device,
                train=False,
                loss_channels=val_loss_channels,
                epoch=epoch,
                log_every=args.log_every,
            )
            print(f"epoch {epoch:04d} | train_mse={train_loss:.6e} | val_mse={val_loss:.6e}")

            save_checkpoint(out_dir / "last_adapters.pt", model, optimizer, epoch, val_loss, metadata)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(out_dir / "best_adapters.pt", model, optimizer, epoch, val_loss, metadata)

    print(f"Done. Best val_mse={best_val:.6e}. Saved to {out_dir}")


if __name__ == "__main__":
    main()

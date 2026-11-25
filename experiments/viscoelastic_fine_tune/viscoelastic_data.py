"""Official viscoelastic_instability_v2 splits, first twenty consecutive frames."""
from pathlib import Path
import hashlib
import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

FIELD_NAMES = ("c_zz", "pressure", "velocity_x", "velocity_y", "C_xx", "C_xy", "C_yx", "C_yy")
FIELD_COMPONENTS = (("t0_fields/c_zz", ()), ("t0_fields/pressure", ()),
                    ("t1_fields/velocity", (0,)), ("t1_fields/velocity", (1,)),
                    ("t2_fields/C", (0, 0)), ("t2_fields/C", (0, 1)),
                    ("t2_fields/C", (1, 0)), ("t2_fields/C", (1, 1)))


def inventory(root):
    root = Path(root)
    result = {}
    for split in ("train", "valid", "test"):
        records = []
        for path in sorted((root / "data" / split).glob("*.hdf5")):
            with h5py.File(path, "r") as h:
                assert h.attrs["dataset_name"] == "viscoelastic_instability_v2"
                assert h["dimensions/time"].shape[-1] >= 20
                for i in range(int(h.attrs["n_trajectories"])):
                    times = np.asarray(h["dimensions/time"][i, :20], dtype=np.float64)
                    assert np.all(np.diff(times) > 0)
                    records.append({"file": str(path), "trajectory": i,
                                    "regime": path.stem.removeprefix("viscoelastic_instability_v2_"),
                                    "time_indices": list(range(20)), "times": times.tolist(),
                                    "relative_times": (times - times[0]).tolist()})
        assert len(records) == {"train": 210, "valid": 22, "test": 22}[split]
        result[split] = records
    # Original file directories are kept; trajectories are never re-split.
    return result


class ViscoelasticDataset(Dataset):
    def __init__(self, records, resolution, cache_dir):
        self.records = records
        self.resolution = int(resolution)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.records)

    def cache_path(self, index):
        record = self.records[index]
        source = Path(record["file"])
        signature = json.dumps({"path": str(source), "size": source.stat().st_size,
                                "mtime_ns": source.stat().st_mtime_ns, "trajectory": record["trajectory"],
                                "indices": record["time_indices"], "resolution": self.resolution,
                                "fields": FIELD_NAMES, "resize": "area", "version": 1}, sort_keys=True)
        return self.cache_dir / (hashlib.sha256(signature.encode()).hexdigest() + ".npy")

    def materialize(self, index):
        path = self.cache_path(index)
        if path.exists():
            array = np.load(path, mmap_mode="r")
            assert array.shape == (8, 20, self.resolution, self.resolution)
            return path
        record = self.records[index]
        fields = []
        with h5py.File(record["file"], "r") as h:
            # Read complete vector/tensor datasets once. Strided HDF5 component
            # reads are particularly expensive on /mnt/d and repeat disk IO.
            for name in ("t0_fields/c_zz", "t0_fields/pressure", "t1_fields/velocity", "t2_fields/C"):
                array = np.asarray(h[name][record["trajectory"], :20], dtype=np.float32)
                value = torch.from_numpy(array).reshape(20, 512, 512, -1).permute(0, 3, 1, 2)
                if value.shape[-2:] != (self.resolution, self.resolution):
                    value = F.interpolate(value, size=(self.resolution, self.resolution), mode="area")
                fields.append(value.permute(1, 0, 2, 3).contiguous().numpy())
        result = np.concatenate(fields, axis=0)
        assert np.isfinite(result).all()
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as f:
            np.save(f, result, allow_pickle=False)
        temporary.replace(path)
        return path

    def __getitem__(self, index):
        y = np.load(self.materialize(index))
        return {"y": torch.from_numpy(y),
                "times": torch.tensor(self.records[index]["relative_times"], dtype=torch.float32),
                "index": index}


def fit_normalization(dataset):
    total = np.zeros(8, dtype=np.float64)
    total_sq = np.zeros(8, dtype=np.float64)
    count = 0
    for i in range(len(dataset)):
        value = np.load(dataset.materialize(i), mmap_mode="r")
        total += value.sum(axis=(1, 2, 3), dtype=np.float64)
        total_sq += np.square(value, dtype=np.float64).sum(axis=(1, 2, 3))
        count += int(np.prod(value.shape[1:]))
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 1e-12))
    return {"field_names": list(FIELD_NAMES), "mean": mean.tolist(), "std": std.tolist(),
            "fitted_split": "train", "trajectories": len(dataset), "time_indices": list(range(20)),
            "spatial_resolution": dataset.resolution,
            "time_scale": max(r["relative_times"][-1] for r in dataset.records)}


def prepare_batch(batch, stats, device):
    y = batch["y"].to(device)
    mean = torch.tensor(stats["mean"], device=device, dtype=y.dtype).view(1, 8, 1, 1, 1)
    std = torch.tensor(stats["std"], device=device, dtype=y.dtype).view(1, 8, 1, 1, 1)
    target = (y - mean) / std
    x = target[:, :, :1].expand_as(target)
    t = (batch["times"].to(device) / stats["time_scale"])[:, None, :, None, None]
    t = t.expand(y.shape[0], 1, 20, y.shape[-2], y.shape[-1])
    return torch.cat((x, t), dim=1), target, y, mean, std

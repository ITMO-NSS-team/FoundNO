from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from muno.data.benchmarks.pipeline import build_benchmark_loaders


se_af_config = {
    "source": {
        "location": "huggingface",
        "format": "netcdf",
        "repo_id": "camlab-ethz/SE-AF",
        "filename": "SE-AF.nc",
        "cache_dir": r"D:\datasets_cache",
        "variable_name": "solution",
        "sample_dim": "sample",
    },
    "adapter": {
        "type": "index",
        "variable_name": None,
        "data_order": "CHW",
        "input_indices": [0],
        "output_indices": [1],
        "benchmark_name": "POSEIDON",
        "physics_name": "SE-AF",
    },
    "split": {
        "train": [0, 32],
        "val": [32, 48],
        "test": [48, 64],
    },
    "max_samples_per_split": {
        "train": 16,
        "val": 16,
        "test": 16,
    },
    "loaders": {
        "train": {
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": True,
            "drop_last": True,
        },
        "val": {
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": False,
            "drop_last": False,
        },
        "test": {
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": False,
            "drop_last": False,
        },
    },
}


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_benchmark_loaders(se_af_config)
    print(len(train_loader), len(val_loader), len(test_loader))

    batch = next(iter(train_loader))
    print(batch.keys())
    print(batch["x"].shape)
    print(batch["y"].shape)
    print(batch["benchmark_name"])
    print(batch["physics_name"])

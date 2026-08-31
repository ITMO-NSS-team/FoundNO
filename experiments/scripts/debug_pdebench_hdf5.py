from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import h5py
import numpy as np

from muno.data.benchmarks.config_io import load_yaml_config
from muno.data.benchmarks.multiphysics_loaders import (
    build_multitask_loaders,
    get_loaders_channels,
)


CONFIG_PATH = PROJECT_ROOT / "experiments" / "configs" / "pdebench_debug.yaml"


def create_hdf5_if_missing(path, datasets):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return

    with h5py.File(path, "w") as file:
        for name, data in datasets.items():
            file.create_dataset(name, data=data.astype("float32"))

    print(f"created debug PDEBench HDF5: {path}")


def ensure_debug_pdebench_files():
    create_hdf5_if_missing(
        r"D:\PDEBench_data\debug_diff_sorp.hdf5",
        {
            "data": np.random.randn(32, 6, 2, 64),
        },
    )

    create_hdf5_if_missing(
        r"D:\PDEBench_data\debug_darcy.hdf5",
        {
            "coeff": np.random.randn(32, 32, 32),
            "solution": np.random.randn(32, 32, 32),
        },
    )

    create_hdf5_if_missing(
        r"D:\PDEBench_data\debug_swe.hdf5",
        {
            "data": np.random.randn(32, 6, 3, 32, 32),
        },
    )


def main():
    ensure_debug_pdebench_files()

    config = load_yaml_config(CONFIG_PATH)
    task_configs = config["tasks"]

    train_loaders, val_loaders, test_loaders, task_metadata = build_multitask_loaders(task_configs)

    print(len(train_loaders), len(val_loaders), len(test_loaders))
    print(task_metadata)

    for loader_idx, loader in enumerate(train_loaders):
        batch = next(iter(loader))
        print("loader", loader_idx)
        print(batch["x"].shape)
        print(batch["y"].shape)
        print(batch["eq_idx"])
        print(batch["benchmark_name"])
        print(batch["physics_name"])

    channels = get_loaders_channels(train_loaders)
    print(channels)


if __name__ == "__main__":
    main()

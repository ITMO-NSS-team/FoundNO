from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from muno.data.benchmarks.config_io import load_yaml_config
from muno.data.benchmarks.multiphysics_loaders import (
    build_multitask_loaders,
    get_loaders_channels,
)


config = load_yaml_config(PROJECT_ROOT / "experiments" / "configs" / "poseidon_debug.yaml")
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
    print(batch["physics_name"])

channels = get_loaders_channels(train_loaders)
print(channels)

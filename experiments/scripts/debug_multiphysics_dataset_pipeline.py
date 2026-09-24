from pathlib import Path
import sys

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from muno.data.benchmarks.config_io import load_yaml_config
from muno.data.benchmarks.multiphysics_loaders import (
    build_multitask_loaders,
    build_multitask_datasets,
    get_loaders_channels,
)
from muno.data.benchmarks.datasets import MultiPhysicsDataset

a = None

config = load_yaml_config(
    "D:/PycharmProjects/FoundNO-main/experiments/configs/extra_channels_test.yaml"
)

task_configs = config["tasks"]

old_train_loaders, old_val_loaders, old_test_loaders, old_metadata = build_multitask_loaders(
    task_configs,
    seed=42,
)

print("old channels:", get_loaders_channels(old_train_loaders))

train_datasets, val_datasets, test_datasets, metadata = build_multitask_datasets(task_configs)

print("\ntrain dataset lengths:", [len(ds) for ds in train_datasets])
print("val dataset lengths:", [len(ds) for ds in val_datasets])
print("test dataset lengths:", [len(ds) for ds in test_datasets])

train_set = MultiPhysicsDataset(train_datasets)
train_loader = DataLoader(train_set, batch_size=2, shuffle=False)

print("new channels:", get_loaders_channels(train_loader))

lengths = [len(ds) for ds in train_datasets]
min_len = min(lengths)
max_len = max(lengths)

for idx in [0, min_len - 1, min_len, max_len - 1]:
    print("checking idx:", idx)
    sample = train_set[idx]
    print({key: tuple(value["x"].shape) for key, value in sample.items()})

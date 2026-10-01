import torch
from functools import singledispatch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Callable, List, Tuple, Set
import warnings

from muno.layers.skips import SkipLike
from muno.data.benchmarks.datasets import MultiPhysicsDataset
from muno.data.benchmarks.pipeline import (
    resolve_split,
    resolve_trajectory_indices,
    resolve_index_split,
    build_adapter,
    build_datasets,
    build_indexed_datasets,
    build_loaders,
    build_source,
)


def count_extra_channels(adapter_config):
    x_extra = 0
    y_extra = 0

    for extra_config in adapter_config.get("extra_channels", []):
        if "name" not in extra_config:
            raise ValueError("extra_channels entry must define a name")

        channel_type = extra_config["type"]

        if channel_type == "constant":
            count = 1
        elif channel_type == "coordinates":
            count = len(extra_config["axes"])
        else:
            raise ValueError(f"Unknown extra channel type: {channel_type}")

        target = extra_config.get("target", "x")
        if target == "x":
            x_extra += count
        elif target == "y":
            y_extra += count
        elif target == "both":
            x_extra += count
            y_extra += count
        else:
            raise ValueError(f"Unknown extra channel target: {target}")

    return x_extra, y_extra


class EqIndexDataset(Dataset):
    def __init__(self, dataset, eq_idx):
        self.dataset = dataset
        self.eq_idx = eq_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        item["eq_idx"] = self.eq_idx
        return item


def build_multitask_datasets(task_configs, seed=None):
    train_datasets = []
    val_datasets   = []
    test_datasets  = []
    task_metadata  = []

    for eq_idx, task_config in enumerate(task_configs):
        task_name = task_config.get("name", f"task_{eq_idx}")

        print(f"\n[{task_name}]")
        print(f"  eq_idx: {eq_idx}")

        source = build_source(task_config["source"])
        adapter = build_adapter(task_config["adapter"])

        source_length = len(source)
        selected_trajectory_count = source_length

        if "trajectory_selection" in task_config:
            trajectory_indices = resolve_trajectory_indices(
                source,
                task_config.get("trajectory_selection"),
            )
            selected_trajectory_count = len(trajectory_indices)
            split = resolve_index_split(
                trajectory_indices,
                task_config["split"],
                task_config.get("max_samples_per_split"),
            )
            train_dataset, val_dataset, test_dataset = build_indexed_datasets(
                source,
                adapter,
                split,
                seed=seed,
                eq_idx=eq_idx
            )
        else:
            split = resolve_split(
                source,
                task_config["split"],
                task_config.get("max_samples_per_split"),
            )
            train_dataset, val_dataset, test_dataset = build_datasets(
                source,
                adapter,
                split,
                seed=seed,
                eq_idx=eq_idx
            )

        train_datasets.append(EqIndexDataset(train_dataset, eq_idx))
        val_datasets.append(EqIndexDataset(val_dataset, eq_idx))
        test_datasets.append(EqIndexDataset(test_dataset, eq_idx))

        sample = train_dataset[0]
        x_channels = sample["x"].shape[0]
        y_channels = sample["y"].shape[0]

        x_extra, y_extra = count_extra_channels(task_config["adapter"])

        print(
            f"  x channels: original={x_channels - x_extra}, "
            f"extra={x_extra}, final={x_channels}"
        )
        print(
            f"  y channels: original={y_channels - y_extra}, "
            f"extra={y_extra}, final={y_channels}"
        )

        print(f"  source_length: {source_length}")
        print(f"  trajectory_selection: {task_config.get('trajectory_selection')}")
        print(f"  selected_trajectory_count: {selected_trajectory_count}")

        task_metadata.append({"eq_idx": eq_idx,
                              "name": task_config.get("name", f"task_{eq_idx}"),
                              "source": task_config["source"],
                              "adapter": task_config["adapter"],
                              "trajectory_selection": task_config.get("trajectory_selection"),
                              "source_length": source_length,
                              "selected_trajectory_count": selected_trajectory_count,
                              "dataset_lengths": {"train": len(train_dataset),
                                                  "val": len(val_dataset),
                                                  "test": len(test_dataset),
                                                  },
                              })

    return train_datasets, val_datasets, test_datasets, task_metadata


def resolve_loader_config(tasks_config):
    if not tasks_config:
        raise ValueError("tasks_config is empty!")

    loader_config = tasks_config[0]["loaders"]

    for task_idx, task_config in enumerate(tasks_config[1:], start=1):
        current_loader_config = task_config["loaders"]
        if current_loader_config != loader_config:
            raise ValueError(
                f"All tasks must use the same loaders config for MUNO DataLoader. "
                f"Task 0 has {loader_config}, task {task_idx} has {current_loader_config}."
            )

    return loader_config


def build_multitask_loaders(task_configs, seed=None):
    train_datasets, val_datasets, test_datasets, task_metadata = build_multitask_datasets(
        task_configs,
        seed=seed
    )

    train_set = MultiPhysicsDataset(train_datasets)
    val_set = MultiPhysicsDataset(val_datasets)
    test_set = MultiPhysicsDataset(test_datasets)

    loader_config = resolve_loader_config(task_configs)

    train_loader, val_loader, test_loader = build_loaders(
        train_set,
        val_set,
        test_set,
        loader_config,
        seed=seed
    )

    return train_loader, val_loader, test_loader, task_metadata


# def get_loader_channels(loader, skips=None):
#     batch = next(iter(loader))
#     print([key for key in batch.keys()])

#     in_channels = batch["x"].shape[1]

#     if skips:
#         in_channels = filterChannelsBySkips(batch["x"], skips)

#     return in_channels, batch["y"].shape[1]


def getSkipsChannel(skips: Dict[int, SkipLike] = None, condition: Callable[[SkipLike,], bool] = None) -> Set[int]:
    if skips is None:
        return set()
    
    if condition is None:
        condition = lambda x: True # lambda x: x.origin == -2

    skipped_channels = set()
    for skip in skips.values():
        if condition(skip):
            skipped_channels = skipped_channels | set(skip._channels)

    return skipped_channels

def filterChannelsBySkips(batch: torch.Tensor, skips: Dict[int, SkipLike] = None) -> int:
    dim = batch.shape[1]
    skipped_channels = getSkipsChannel(skips, lambda x: x.origin == -2)

    return dim - len(skipped_channels)

@singledispatch
def getLoadersChannels(loaders):
    raise NotImplementedError('Calling getLoadersChannels of a default unimplemented type.')

@getLoadersChannels.register
def _(loaders: list, skips: List[Dict[int, SkipLike]] = None) -> List[List[Tuple[int, int]]]:
    warnings.warn("Calling legacy implementation of getLoadersChannels, unexpected behavior.")

    assert all([isinstance(loader, DataLoader) for loader in loaders]), 'Loaders have to be a list of DataLoader objects.'
    if skips is None:
        skips = [None,] * len(loaders)
    return [getLoadersChannels(loader, skips[idx]) for idx, loader in enumerate(loaders)]

@getLoadersChannels.register
def _(loaders: DataLoader, skips: Dict[int, SkipLike] = None) -> List[Tuple[int, int]]:
    batch = next(iter(loaders))
    assert isinstance(batch, dict), \
        'loader has to return a dict with keys - multiphysics problems idx, values - dicts {"x": torch.Tensor, "y": ...}.'

    return [(filterChannelsBySkips(subbatch["x"], skips), subbatch["y"].shape[1]) for subbatch in batch.values()]

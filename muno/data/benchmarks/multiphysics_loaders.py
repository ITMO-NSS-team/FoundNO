from functools import singledispatch
from torch.utils.data import Dataset, DataLoader

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

def build_multitask_datasets(task_configs):
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
            )

        train_datasets.append(EqIndexDataset(train_dataset, eq_idx))
        val_datasets.append(EqIndexDataset(val_dataset, eq_idx))
        test_datasets.append(EqIndexDataset(test_dataset, eq_idx))

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
    

def build_multitask_loaders(task_configs, seed=None):
    train_loaders = []
    val_loaders = []
    test_loaders = []

    train_datasets, val_datasets, test_datasets, task_metadata = build_multitask_datasets(task_configs)

    for eq_idx, task_config in enumerate(task_configs):

        train_dataset = train_datasets[eq_idx]  # EqIndexDataset(train_dataset, eq_idx)
        val_dataset   = val_datasets[eq_idx]    # EqIndexDataset(val_dataset, eq_idx)
        test_dataset  = test_datasets[eq_idx]   # EqIndexDataset(test_dataset, eq_idx)

        loader_seed = None if seed is None else int(seed) + eq_idx * 1000

        train_loader, val_loader, test_loader = build_loaders(
            train_dataset,
            val_dataset,
            test_dataset,
            task_config["loaders"],
            seed=loader_seed
        )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

    return train_loaders, val_loaders, test_loaders, task_metadata


def build_multitask_loaders_old(task_configs, seed=None):
    train_loaders = []
    val_loaders = []
    test_loaders = []
    task_metadata = []

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
            )

        train_dataset = EqIndexDataset(train_dataset, eq_idx)
        val_dataset = EqIndexDataset(val_dataset, eq_idx)
        test_dataset = EqIndexDataset(test_dataset, eq_idx)

        loader_seed = None if seed is None else int(seed) + eq_idx * 1000

        train_loader, val_loader, test_loader = build_loaders(
            train_dataset,
            val_dataset,
            test_dataset,
            task_config["loaders"],
            seed=loader_seed
        )

        first_batch = next(iter(train_loader))

        print(f"  train batch x: {tuple(first_batch['x'].shape)}")
        print(f"  train batch y: {tuple(first_batch['y'].shape)}")
        print(f"  benchmark_name: {first_batch['benchmark_name'][0]}")
        print(f"  physics_name: {first_batch['physics_name'][0]}")
        print(f"  trajectories_full: {source_length}")
        print(f"  trajectories_selection: {task_config.get('trajectory_selection')}")
        print(f"  selected_trajectory_count: {selected_trajectory_count}")

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

        task_metadata.append({
            "eq_idx": eq_idx,
            "name": task_config.get("name", f"task_{eq_idx}"),
            "source": task_config["source"],
            "adapter": task_config["adapter"],
            "trajectory_selection": task_config.get("trajectory_selection"),
            "source_length": source_length,
            "selected_trajectory_count": selected_trajectory_count,
            "dataset_lengths": {
                "train": len(train_dataset),
                "val": len(val_dataset),
                "test": len(test_dataset),
            },
        })

    return train_loaders, val_loaders, test_loaders, task_metadata


def get_loader_channels(loader):
    batch = next(iter(loader))
    return batch["x"].shape[1], batch["y"].shape[1]


@singledispatch
def get_loaders_channels(loaders):
    raise NotImplementedError('Calling get_loaders_channels of a default unimplemented type.')

@get_loaders_channels.register
def _(loaders: list):
    assert all([isinstance(loader, DataLoader) for loader in loaders]), 'Loaders have to be a list of DataLoader objects.'
    return [get_loader_channels(loader) for loader in loaders]

@get_loaders_channels.register
def _(loaders: DataLoader):
    batch = next(iter(loaders))
    assert isinstance(batch, dict), \
        'loader has to return a dict with keys - multiphysics problems idx, values - dicts {"x": torch.Tensor, "y": ...}.'

    # print('batch is ', batch)
    # print('Shapes are: ', [(subbatch['eq_idx'], subbatch["x"].shape, subbatch["y"].shape) for subbatch in batch.values()])
    return [(subbatch["x"].shape[1], subbatch["y"].shape[1]) for subbatch in batch.values()]

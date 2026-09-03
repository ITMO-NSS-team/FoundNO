import torch
import numpy as np
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_files(files, selection_config=None):
    files = list(files)

    if selection_config is None:
        return files

    max_files = selection_config.get("max_files")
    strategy = selection_config.get("strategy", "first")
    seed = selection_config.get("seed", 0)

    if max_files is None or max_files >= len(files):
        return files

    if strategy == "first":
        return files[:max_files]

    if strategy == "random":
        rng = random.Random(seed)
        return rng.sample(files, max_files)

    raise ValueError(f"Unknown file selection strategy: {strategy}")


def get_config_value(key, *configs):
    for config in configs:
        if config is not None and key in config:
            return config[key]
    return None


def validate_batch(loader):
    batch = next(iter(loader))

    assert "x" in batch
    assert "y" in batch
    assert "benchmark_name" in batch
    assert "physics_name" in batch

    assert isinstance(batch["x"], torch.Tensor)
    assert isinstance(batch["y"], torch.Tensor)

    assert batch["x"].dtype == torch.float32
    assert batch["y"].dtype == torch.float32

    assert batch["x"].ndim >= 3
    assert batch["y"].ndim >= 3

    assert batch["x"].shape[0] == batch["y"].shape[0]
    assert batch["x"].shape[2:] == batch["y"].shape[2:]

    assert len(batch["benchmark_name"]) == batch["x"].shape[0]
    assert len(batch["physics_name"]) == batch["x"].shape[0]


def validate_split_keys(split):
    required = ("train", "val", "test")
    missing = [name for name in required if name not in split]
    if missing:
        raise ValueError(f"split is missing required keys: {missing}")


def validate_range_split(split, source_length):
    validate_split_keys(split)
    for split_name in ("train", "val", "test"):
        value = split[split_name]

        if not isinstance(value, (tuple, list)):
            raise ValueError(f"{split_name} split must be a pair (start, end), got {value}")

        if len(value) != 2:
            raise ValueError(f"{split_name} split must be a pair (start, end), got {value}")

        start, end = value
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"{split_name} split start/end must be int, got {value}")
        if start < 0:
            raise ValueError(f"{split_name} split start must be >= 0, got {start}")
        if end <= start:
            raise ValueError(f"{split_name} split must be non-empty, got start={start}, end={end}")
        if end > source_length:
            raise ValueError(f"{split_name} split end={end} exceeds source length {source_length}")


def validate_index_split(split):
    validate_split_keys(split)
    seen = {}

    for split_name in ("train", "val", "test"):
        indices = list(split[split_name])
        if not indices:
            raise ValueError(f"{split_name} split is empty")

        for idx in indices:
            if not isinstance(idx, int):
                raise ValueError(f"{split_name} split indices must be int, got {idx!r}")

        if len(indices) != len(set(indices)):
            raise ValueError(f"{split_name} split contains duplicated indices")

        for idx in indices:
            if idx in seen:
                raise ValueError(
                    f"trajectory index {idx} appears in both "
                    f"{seen[idx]} and {split_name} splits"
                )
            seen[idx] = split_name


def apply_max_samples(split, max_samples_per_split):
    if max_samples_per_split is None:
        return split
    new_split = {}
    for split_name in split:
        start, end = split[split_name]
        max_samples = max_samples_per_split[split_name]
        new_end = min(end, start + max_samples)
        new_split[split_name] = (start, new_end)
    return new_split


def apply_max_samples_to_index_split(split, max_samples_per_split):
    if max_samples_per_split is None:
        return split
    new_split = {}
    for split_name, indices in split.items():
        max_samples = max_samples_per_split[split_name]
        new_split[split_name] = indices[:max_samples]
    return new_split


def resolve_trajectory_indices(source, selection_config=None):
    modes = ["count", "fraction", "indices"]
    strategies = ["first", "random"]
    total = len(source)

    if selection_config is None:
        return list(range(total))

    selection_modes = [key for key in modes if key in selection_config]

    if len(selection_modes) != 1:
        raise ValueError(
            "trajectory_selection must define exactly one of: "
            "count, fraction, indices"
        )

    selection_mode = selection_modes[0]

    if selection_mode == "indices":
        indices = list(selection_config["indices"])
        if not indices:
            raise ValueError("trajectory_selection indices must not be empty")
        if len(indices) != len(set(indices)):
            raise ValueError("trajectory_selection indices must be unique")
        for idx in indices:
            if idx < 0 or idx >= total:
                raise ValueError(
                    f"trajectory_selection index {idx} is out of range "
                    f"for source length {total}"
                )
        return indices

    if selection_mode == "count":
        count = int(selection_config["count"])
    else:
        fraction = float(selection_config["fraction"])
        if fraction <= 0 or fraction > 1:
            raise ValueError(
                f"trajectory_selection fraction must be in (0, 1], got {fraction}"
            )

        count = int(total * fraction)

    if count <= 0:
        raise ValueError(f"trajectory_selection count must be positive, got {count}")
    if count > total:
        raise ValueError(f"trajectory_selection count={count} exceeds source length {total}")

    strategy = selection_config.get("strategy", "first")

    if strategy == "first":
        return list(range(count))
    elif strategy == "random":
        rng = random.Random(selection_config.get("seed", 0))
        return rng.sample(range(total), count)
    else:
        raise ValueError(
            f"Unknown trajectory_selection strategy: {strategy}. "
            f"Valid strategies: {strategies}"
        )


def resolve_split(source, split_config, max_samples_per_split=None):
    split_type = split_config.get("type", "explicit")

    if split_type == "explicit":
        split = {}
        for split_name in ("train", "val", "test"):
            start, end = split_config[split_name]
            start = int(start)
            end = int(end)
            split[split_name] = (start, end)

    elif split_type == "ratios":
        total = len(source)
        train_ratio = float(split_config["train"])
        val_ratio = float(split_config["val"])
        test_ratio = float(split_config["test"])
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        split = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, total)
        }

    else:
        raise ValueError(f"Unknown split type: {split_type}")

    split = apply_max_samples(split, max_samples_per_split)
    validate_range_split(split, len(source))
    return split


def resolve_index_split(indices, split_config, max_samples_per_split=None):
    indices = list(indices)
    total = len(indices)
    split_type = split_config.get("type", "explicit")

    if total == 0:
        raise ValueError("Cannot split empty trajectory index list")

    if split_type == "explicit":
        split = {}
        for split_name in ("train", "val", "test"):
            start, end = split_config[split_name]
            start = int(start)
            end = int(end)
            if start < 0 or end <= start or end > total:
                raise ValueError(
                    f"Invalid explicit {split_name} split [{start}, {end}] "
                    f"for selected trajectory count {total}"
                )
            split[split_name] = indices[start:end]

    elif split_type == "ratios":
        train_ratio = float(split_config["train"])
        val_ratio = float(split_config["val"])
        test_ratio = float(split_config["test"])
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        split = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:]
        }

    else:
        raise ValueError(f"Unknown split type: {split_type}")

    split = apply_max_samples_to_index_split(split, max_samples_per_split)
    validate_index_split(split)
    return split

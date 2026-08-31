from pathlib import Path
import re

import matplotlib.pyplot as plt
import torch

from muno.data.benchmarks.pipeline import build_source, build_adapter


SPATIAL_AXES = {"X", "Y", "H", "W"}


def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name))


def describe_shape(shape, axis_names=None):
    if axis_names is None or len(shape) != len(axis_names):
        return f"{tuple(shape)}"

    parts = [
        f"{axis_name}={size}"
        for axis_name, size in zip(axis_names, shape)
    ]
    return "(" + ", ".join(parts) + ")"


def tensor_summary(value, axis_names=None):
    if isinstance(value, torch.Tensor):
        return {
            "shape": describe_shape(value.shape, axis_names),
            "dtype": str(value.dtype),
            "min": float(value.min()),
            "max": float(value.max()),
            "mean": float(value.mean()),
            "std": float(value.std()),
        }

    if isinstance(value, dict):
        return {
            key: tensor_summary(item, axis_names)
            for key, item in value.items()
        }

    return type(value).__name__


def print_tensor_summary(prefix, value, axis_names=None):
    summary = tensor_summary(value, axis_names=axis_names)

    if isinstance(summary, dict) and "shape" in summary:
        print(
            f"  {prefix}: shape={summary['shape']}, dtype={summary['dtype']}, "
            f"min={summary['min']:.6g}, max={summary['max']:.6g}, "
            f"mean={summary['mean']:.6g}, std={summary['std']:.6g}"
        )
        return

    if isinstance(summary, dict):
        print(f"  {prefix}:")
        for key, item in summary.items():
            if isinstance(item, dict) and "shape" in item:
                print(
                    f"    {key}: shape={item['shape']}, dtype={item['dtype']}, "
                    f"min={item['min']:.6g}, max={item['max']:.6g}, "
                    f"mean={item['mean']:.6g}, std={item['std']:.6g}"
                )
            else:
                print(f"    {key}: {item}")
        return

    print(f"  {prefix}: {summary}")


def image_from_order(tensor, data_order, time_index=0, channel_index=0):
    data_order = data_order.upper()

    if len(data_order) != tensor.ndim:
        return first_spatial_image(tensor)

    selection = []
    for axis in data_order:
        if axis in SPATIAL_AXES:
            selection.append(slice(None))
        elif axis == "T":
            selection.append(time_index)
        elif axis == "C":
            selection.append(channel_index)
        else:
            selection.append(0)

    image = tensor[tuple(selection)]

    while image.ndim > 2:
        image = image[0]

    return image


def first_spatial_image(tensor):
    image = tensor

    while image.ndim > 2:
        image = image[0]

    return image


def canonical_image(tensor, channel_index=0, time_index=0):
    if tensor.ndim == 2:
        return tensor

    if tensor.ndim == 3:
        # [C, H, W]
        return tensor[channel_index]

    if tensor.ndim == 4:
        # [C, T, H, W]
        return tensor[channel_index, time_index]

    if tensor.ndim == 5:
        # [B, C, T, H, W]
        return tensor[0, channel_index, time_index]

    return first_spatial_image(tensor)


def adapter_raw_axis_names(adapter_config):
    data_order = adapter_config.get("data_order")

    if data_order is None:
        return None

    return list(data_order.upper())


def canonical_axis_names(tensor):
    if tensor.ndim == 3:
        return ["C", "X", "Y"]

    if tensor.ndim == 4:
        return ["C", "T", "X", "Y"]

    if tensor.ndim == 5:
        return ["C", "T", "X", "Y", "Z"]

    return None


def save_image(image, path, title):
    image = image.detach().cpu()

    plt.figure(figsize=(5, 4))
    plt.imshow(image, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_raw_images(raw_sample, output_prefix, data_order=None):
    if isinstance(raw_sample, dict):
        for key, tensor in raw_sample.items():
            if not isinstance(tensor, torch.Tensor):
                continue

            if data_order is not None:
                image = image_from_order(tensor, data_order, time_index=0, channel_index=0)
            else:
                image = first_spatial_image(tensor)

            save_image(
                image,
                output_prefix.with_name(output_prefix.name + f"_raw_{safe_name(key)}.png"),
                f"{output_prefix.name} raw {key}",
            )
        return

    if data_order is not None:
        image = image_from_order(raw_sample, data_order, time_index=0, channel_index=0)
    else:
        image = first_spatial_image(raw_sample)

    save_image(
        image,
        output_prefix.with_name(output_prefix.name + "_raw.png"),
        f"{output_prefix.name} raw",
    )


def save_canonical_images(canonical_sample, output_prefix):
    x = canonical_sample["x"]
    y = canonical_sample["y"]

    save_image(
        canonical_image(x, channel_index=0, time_index=0),
        output_prefix.with_name(output_prefix.name + "_x_t0_c0.png"),
        f"{output_prefix.name} x t0 c0",
    )

    save_image(
        canonical_image(y, channel_index=0, time_index=0),
        output_prefix.with_name(output_prefix.name + "_y_t0_c0.png"),
        f"{output_prefix.name} y t0 c0",
    )

    if x.ndim >= 4:
        save_image(
            canonical_image(x, channel_index=0, time_index=x.shape[1] - 1),
            output_prefix.with_name(output_prefix.name + "_x_tlast_c0.png"),
            f"{output_prefix.name} x t_last c0",
        )

    if y.ndim >= 4:
        save_image(
            canonical_image(y, channel_index=0, time_index=y.shape[1] - 1),
            output_prefix.with_name(output_prefix.name + "_y_tlast_c0.png"),
            f"{output_prefix.name} y t_last c0",
        )


def get_inspection_sample_index(task_config, source):
    inspection_config = task_config.get("inspection", {})

    if "sample_index" in inspection_config:
        return inspection_config["sample_index"]

    split_config = task_config["split"]
    split_type = split_config.get("type", "explicit")

    if split_type == "explicit":
        return split_config["train"][0]

    if split_type == "ratios":
        return 0

    raise ValueError(f"Unknown split type for inspection: {split_type}")


def inspect_task(task_config, output_dir, eq_idx):
    task_name = task_config.get("name", f"task_{eq_idx}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = build_source(task_config["source"])
    adapter = build_adapter(task_config["adapter"])

    sample_index = get_inspection_sample_index(task_config, source)
    raw_sample = source.get_sample(sample_index)
    canonical_sample = adapter.canonize(raw_sample)

    print(f"\n[inspection] {task_name}")
    print(f"  sample index: {sample_index}")

    raw_axis_names = adapter_raw_axis_names(task_config["adapter"])
    x_axis_names = canonical_axis_names(canonical_sample["x"])
    y_axis_names = canonical_axis_names(canonical_sample["y"])

    print_tensor_summary("raw", raw_sample, axis_names=raw_axis_names)
    print_tensor_summary("canonical x", canonical_sample["x"], axis_names=x_axis_names)
    print_tensor_summary("canonical y", canonical_sample["y"], axis_names=y_axis_names)

    print(f"  benchmark_name: {canonical_sample['benchmark_name']}")
    print(f"  physics_name: {canonical_sample['physics_name']}")

    output_prefix = output_dir / f"{eq_idx}_{safe_name(task_name)}"
    data_order = task_config["adapter"].get("data_order")

    save_raw_images(raw_sample, output_prefix, data_order=data_order)
    save_canonical_images(canonical_sample, output_prefix)


def inspect_tasks(task_configs, output_dir):
    for eq_idx, task_config in enumerate(task_configs):
        inspect_task(task_config, output_dir, eq_idx)

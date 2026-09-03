import torch
import random
import numpy as np

from muno.data.benchmarks.sources import (
    NetCDFSource,
    MultiNetCDFSource,
    ConcatSource,
    HDF5Source,
    HDF5GroupSource,
    TheWellHDF5Source
)
from muno.data.benchmarks.adapters import IndexAdapter, TemporalAdapter, InputOutputAdapter
from muno.data.benchmarks.datasets import (
    LazyCanonicalDataset,
    SlidingWindowCanonicalDataset,
    IndexedCanonicalDataset,
    IndexedSlidingWindowCanonicalDataset,
)
from muno.data.benchmarks.data_access import get_access_data_path
from muno.data.benchmarks.pipeline_utils import (
    seed_worker,
    select_files,
    get_config_value,
    validate_batch,
    resolve_index_split,
    resolve_split,
    resolve_trajectory_indices
)


def build_source(config):
    allowed_formats = [
        "netcdf", "multi_netcdf", "concat_netcdf",
        "hdf5", "concat_hdf5", "hdf5_grouped", "concat_hdf5_grouped",
        "thewell_hdf5", "concat_thewell_hdf5"
    ]
    if config["format"] == "netcdf":
        path = get_access_data_path(config)
        return NetCDFSource(
            path,
            config["variable_name"],
            config["sample_dim"],
            axis_selection=config.get("axis_selection")
        )

    elif config["format"] == "concat_netcdf":
        sources = []
        selected_files = select_files(
            config["files"],
            config.get("file_selection"),
        )

        print("selected NetCDF files:")
        for file_config in selected_files:
            print("  ", file_config.get("filename") or file_config.get("path"))
            full_file_config = {
                "location": get_config_value("location", file_config, config),
                "format": "netcdf",
                "repo_id": get_config_value("repo_id", file_config, config),
                "cache_dir": get_config_value("cache_dir", file_config, config),
                "path": file_config.get("path"),
                "filename": file_config.get("filename"),
                "variable_name": get_config_value("variable_name", file_config, config),
                "sample_dim": get_config_value("sample_dim", file_config, config),
                "token_env": get_config_value("token_env", file_config, config),
                "axis_selection": get_config_value("axis_selection", file_config, config)
            }
            local_path = get_access_data_path(full_file_config)
            sources.append(
                NetCDFSource(
                    local_path,
                    full_file_config.get("variable_name"),
                    full_file_config.get("sample_dim"),
                    axis_selection=full_file_config.get("axis_selection")
                )
            )
        return ConcatSource(sources)

    elif config["format"] == "hdf5":
        path = get_access_data_path(config)
        return HDF5Source(
            path,
            variable_path=config.get("variable_path", config.get("variable_name")),
            sample_dim=config.get("sample_dim", 0),
            axis_selection=config.get("axis_selection"),
            variables=config.get("variables"),
        )

    elif config["format"] == "hdf5_grouped":
        path = get_access_data_path(config)
        return HDF5GroupSource(
            path,
            variable_path=config.get("variable_path", config.get("variable_name", "data")),
            variables=config.get("variables"),
            axis_selection=config.get("axis_selection"),
        )

    elif config["format"] == "concat_hdf5":
        sources = []
        selected_files = select_files(
            config["files"],
            config.get("file_selection"),
        )

        print("selected HDF5 files:")
        for file_config in selected_files:
            print("  ", file_config.get("filename") or file_config.get("path"))

            full_file_config = dict(config)
            full_file_config.update(file_config)
            full_file_config["format"] = "hdf5"

            path = get_access_data_path(full_file_config)

            sources.append(
                HDF5Source(
                    path,
                    variable_path=file_config.get(
                        "variable_path",
                        config.get("variable_path", config.get("variable_name")),
                    ),
                    sample_dim=file_config.get("sample_dim", config.get("sample_dim", 0)),
                    axis_selection=config.get("axis_selection"),
                    variables=file_config.get("variables", config.get("variables"))
                )
            )
        return ConcatSource(sources)

    elif config["format"] == "concat_hdf5_grouped":
        sources = []
        selected_files = select_files(
            config["files"],
            config.get("file_selection")
        )

        print("selected grouped HDF5 files:")
        for file_config in selected_files:
            print("  ", file_config.get("filename") or file_config.get("path"))

            full_file_config = dict(config)
            full_file_config.update(file_config)
            full_file_config["format"] = "hdf5_grouped"

            path = get_access_data_path(full_file_config)

            sources.append(
                HDF5GroupSource(
                    path,
                    variable_path=file_config.get(
                        "variable_path",
                        config.get("variable_path", config.get("variable_name", "data"))
                    ),
                    variables=file_config.get("variables", config.get("variables")),
                    axis_selection=config.get("axis_selection"),
                )
            )

        return ConcatSource(sources)

    elif config["format"] == "multi_netcdf":
        components = {}
        for component_name, component_config in config["components"].items():
            if "files" in component_config:
                sources = []
                for file_config in component_config["files"]:
                    full_file_config = {
                        "location": config["location"],
                        "format": "netcdf",
                        "repo_id": config.get("repo_id"),
                        "cache_dir": config.get("cache_dir"),
                        "path": file_config.get("path"),
                        "filename": file_config.get("filename"),
                        "variable_name": component_config["variable_name"],
                        "sample_dim": component_config["sample_dim"],
                        "token_env": get_config_value(
                            "token_env", file_config, component_config, config
                        ),
                        "axis_selection": get_config_value(
                            "axis_selection", file_config, component_config, config
                        )
                    }
                    local_path = get_access_data_path(full_file_config)
                    sources.append(
                        NetCDFSource(
                            local_path,
                            full_file_config["variable_name"],
                            full_file_config["sample_dim"],
                            axis_selection=full_file_config.get("axis_selection")
                        )
                    )
                components[component_name] = ConcatSource(sources)
            else:
                full_component_config = {
                    "location": config["location"],
                    "format": "netcdf",
                    "repo_id": config.get("repo_id"),
                    "cache_dir": config.get("cache_dir"),
                    "path": component_config.get("path"),
                    "filename": component_config.get("filename"),
                    "variable_name": component_config["variable_name"],
                    "sample_dim": component_config["sample_dim"],
                    "token_env": get_config_value(
                        "token_env", component_config, config
                    ),
                    "axis_selection": get_config_value(
                        "axis_selection", component_config, config
                    )
                }
                local_path = get_access_data_path(full_component_config)
                components[component_name] = NetCDFSource(
                    local_path,
                    component_config["variable_name"],
                    component_config["sample_dim"],
                    axis_selection=full_component_config.get("axis_selection")
                )
        return MultiNetCDFSource(
            components,
            length_component=config.get("length_component"),
        )

    elif config["format"] == "thewell_hdf5":
        path = get_access_data_path(config)
        return TheWellHDF5Source(
            path,
            field_groups=config.get("field_groups"),
            sample_dim=config.get("sample_dim", 0),
            axis_selection=config.get("axis_selection"),
            output_key=config.get("output_key", "data"),
        )

    elif config["format"] == "concat_thewell_hdf5":
        sources = []
        selected_files = select_files(
            config["files"],
            config.get("file_selection"),
        )

        print("selected The Well HDF5 files:")
        for file_config in selected_files:
            print("  ", file_config.get("filename") or file_config.get("path"))

            full_file_config = dict(config)
            full_file_config.update(file_config)
            full_file_config["format"] = "thewell_hdf5"

            path = get_access_data_path(full_file_config)

            sources.append(
                TheWellHDF5Source(
                    path,
                    field_groups=file_config.get(
                        "field_groups",
                        config.get("field_groups"),
                    ),
                    sample_dim=file_config.get(
                        "sample_dim",
                        config.get("sample_dim", 0),
                    ),
                    axis_selection=file_config.get(
                        "axis_selection",
                        config.get("axis_selection"),
                    ),
                    output_key=file_config.get(
                        "output_key",
                        config.get("output_key", "data"),
                    ),
                )
            )

        return ConcatSource(sources)

    else:
        raise ValueError(
            f"Unknown source format {config['format']}. "
            f"Allowed formats: {allowed_formats}"
        )


def build_adapter(config):
    allowed_adapters = ["index", "temporal", "input_output"]
    adapter_type = config["type"]
    common_kwargs = {
        "benchmark_name": config.get("benchmark_name"),
        "physics_name": config.get("physics_name"),
        "metadata": config.get("metadata"),
        "ensure_2d": config.get("ensure_2d", False),
    }

    if adapter_type == "index":
        return IndexAdapter(
            variable_name=config.get("variable_name"),
            data_order=config.get("data_order", "CHW"),
            input_indices=config["input_indices"],
            output_indices=config["output_indices"],
            **common_kwargs,
        )
    if adapter_type == "temporal":
        return TemporalAdapter(
            variable_name=config.get("variable_name"),
            variable_names=config.get("variable_names"),
            data_order=config.get("data_order", "TCHW"),
            temporal_mode=config.get("temporal_mode", "window"),
            input_time_indices=config.get("input_time_indices"),
            output_time_indices=config.get("output_time_indices"),
            window_start_indices=config.get("window_start_indices"),
            input_time_index=config.get("input_time_index", 0),
            input_channel_indices=config.get("input_channel_indices"),
            output_channel_indices=config.get("output_channel_indices"),
            static_inputs=config.get("static_inputs"),
            flatten_time_to_channels=config.get("flatten_time_to_channels", True),
            **common_kwargs,
        )
    if adapter_type == "input_output":
        return InputOutputAdapter(
            input_variable_name=config["input_variable_name"],
            output_variable_name=config["output_variable_name"],
            input_order=config.get("input_order", "CHW"),
            output_order=config.get("output_order", "CHW"),
            **common_kwargs,
        )

    raise ValueError(f"Unknown adapter {adapter_type}. Allowed adapters: {allowed_adapters}")


def build_datasets(source, adapter, split):
    window_start_indices = getattr(adapter, "window_start_indices", None)

    if window_start_indices is None:
        dataset_cls = LazyCanonicalDataset
        dataset_kwargs = {}
    else:
        dataset_cls = SlidingWindowCanonicalDataset
        dataset_kwargs = {
            "window_start_indices": window_start_indices,
        }

    train_dataset = dataset_cls(
        source,
        adapter,
        start=split["train"][0],
        end=split["train"][1],
        **dataset_kwargs,
    )
    val_dataset = dataset_cls(
        source,
        adapter,
        start=split["val"][0],
        end=split["val"][1],
        **dataset_kwargs,
    )
    test_dataset = dataset_cls(
        source,
        adapter,
        start=split["test"][0],
        end=split["test"][1],
        **dataset_kwargs,
    )

    return train_dataset, val_dataset, test_dataset


def build_indexed_datasets(source, adapter, split):
    window_start_indices = getattr(adapter, "window_start_indices", None)

    if window_start_indices is None:
        dataset_cls = IndexedCanonicalDataset
        dataset_kwargs = {}
    else:
        dataset_cls = IndexedSlidingWindowCanonicalDataset
        dataset_kwargs = {
            "window_start_indices": window_start_indices,
        }

    train_dataset = dataset_cls(
        source,
        adapter,
        indices=split["train"],
        **dataset_kwargs,
    )
    val_dataset = dataset_cls(
        source,
        adapter,
        indices=split["val"],
        **dataset_kwargs,
    )
    test_dataset = dataset_cls(
        source,
        adapter,
        indices=split["test"],
        **dataset_kwargs,
    )

    return train_dataset, val_dataset, test_dataset


def build_loader(dataset, loader_config, seed=None):
    generator = None
    worker_init_fn = None

    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        worker_init_fn = seed_worker

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=loader_config["batch_size"],
        shuffle=loader_config["shuffle"],
        drop_last=loader_config["drop_last"],
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def build_loaders(train_dataset, val_dataset, test_dataset, config, seed=None):
    train_loader = build_loader(
        train_dataset,
        config["train"],
        seed=None if seed is None else int(seed) + 0,
    )
    val_loader = build_loader(
        val_dataset,
        config["val"],
        seed=None if seed is None else int(seed) + 1,
    )
    test_loader = build_loader(
        test_dataset,
        config["test"],
        seed=None if seed is None else int(seed) + 2,
    )

    validate_batch(train_loader)
    validate_batch(val_loader)
    validate_batch(test_loader)

    return train_loader, val_loader, test_loader


def build_benchmark_loaders(config):
    source = build_source(config["source"])
    adapter = build_adapter(config["adapter"])

    if "trajectory_selection" in config:
        trajectory_indices = resolve_trajectory_indices(
            source,
            config.get("trajectory_selection"),
        )
        split = resolve_index_split(
            trajectory_indices,
            config["split"],
            config.get("max_samples_per_split"),
        )
        train_dataset, val_dataset, test_dataset = build_indexed_datasets(
            source, adapter, split
        )
    else:
        split = resolve_split(
            source,
            config["split"],
            config.get("max_samples_per_split"),
        )
        train_dataset, val_dataset, test_dataset = build_datasets(
            source, adapter, split
        )

    train_loader, val_loader, test_loader = build_loaders(
        train_dataset, val_dataset, test_dataset, config["loaders"]
    )
    return train_loader, val_loader, test_loader

from muno.data.benchmarks.sources import TheWellHDF5Source

path = r"C:\Users\dimerf\Downloads\turbulent_radiative_layer_tcool_0.03_train.hdf5"

source = TheWellHDF5Source(path)
print(len(source))

file = source._open()
print(source._get_field_groups(file))

file = source._open()

density = source._read_field(file, "t0_fields", "density", 0)
pressure = source._read_field(file, "t0_fields", "pressure", 0)
velocity = source._read_field(file, "t1_fields", "velocity", 0)

print(density.shape)
print(pressure.shape)
print(velocity.shape)

density_ch = source._field_to_channels(density, "t0_fields", "density", 2)
velocity_ch = source._field_to_channels(velocity, "t1_fields", "velocity", 2)

print(density_ch.shape)
print(velocity_ch.shape)

file = source._open()

density = source._read_field(file, "t0_fields", "density", 0)
velocity = source._read_field(file, "t1_fields", "velocity", 0)

density_ch = source._field_to_channels(
    density,
    "t0_fields",
    "density",
    2,
)

velocity_ch = source._field_to_channels(
    velocity,
    "t1_fields",
    "velocity",
    2,
)

print(density_ch.shape)
print(velocity_ch.shape)

sample = source.get_sample(0)

print(sample.keys())
print(sample["data"].shape)
print(sample["channel_names"])
print(sample["metadata"])

source_small = TheWellHDF5Source(
    path,
    axis_selection={
        0: {"linspace": [0, 100, 16]},
        1: {"slice": [0, 128, 2]},
        2: {"slice": [0, 384, 3]},
    },
)

sample_small = source_small.get_sample(0)

print(sample_small["data"].shape)
print(sample_small["channel_names"])

# FROM PIPELINE
from muno.data.benchmarks.pipeline import build_source

config = {
    "location": "local",
    "format": "thewell_hdf5",
    "path": path,
    "field_groups": {
        "t0_fields": ["density", "pressure"],
        "t1_fields": ["velocity"],
        "t2_fields": [],
    },
    "axis_selection": {
        0: {"linspace": [0, 100, 16]},
        1: {"slice": [0, 128, 2]},
        2: {"slice": [0, 384, 3]},
    },
}

source_from_pipeline = build_source(config)
sample_from_pipeline = source_from_pipeline.get_sample(0)

print("pipeline len:", len(source_from_pipeline))
print("pipeline sample:", sample_from_pipeline["data"].shape)
print("pipeline channels:", sample_from_pipeline["channel_names"])

from muno.data.benchmarks.pipeline import build_adapter

adapter_config = {
    "type": "temporal",
    "variable_name": "data",
    "data_order": "TCHW",
    "temporal_mode": "window",
    "input_time_indices": [0, 1, 2, 3],
    "output_time_indices": [1, 2, 3, 4],
    "flatten_time_to_channels": False,
    "benchmark_name": "TheWell",
    "physics_name": "turbulent_radiative_layer_2d",
}

adapter = build_adapter(adapter_config)

raw_sample = source_from_pipeline.get_sample(0)
canonical = adapter.canonize(raw_sample, window_start=0)

print("x:", canonical["x"].shape)
print("y:", canonical["y"].shape)
print("benchmark_name:", canonical["benchmark_name"])
print("physics_name:", canonical["physics_name"])

from muno.data.benchmarks.pipeline import build_benchmark_loaders

task_config = {
    "name": "thewell_turbulent_radiative_layer_2d_debug",
    "source": config,
    "adapter": adapter_config,
    "split": {
        "type": "ratios",
        "train": 0.75,
        "val": 0.125,
        "test": 0.125,
    },
    "loaders": {
        "train": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": True,
            "drop_last": True,
        },
        "val": {
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": False,
            "drop_last": False,
        },
        "test": {
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": False,
            "drop_last": False,
        },
    },
}

train_loader, val_loader, test_loader = build_benchmark_loaders(task_config)

batch = next(iter(train_loader))

print("batch x:", batch["x"].shape)
print("batch y:", batch["y"].shape)
print("batch benchmark:", batch["benchmark_name"])
print("batch physics:", batch["physics_name"])

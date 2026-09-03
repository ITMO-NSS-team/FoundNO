import torch
import xarray as xr
import h5py

from bisect import bisect_right


def _selector_from_config(config):
    if config is None or config == "all":
        return slice(None)

    if isinstance(config, list):
        return config

    if isinstance(config, dict):
        if "indices" in config:
            return list(config["indices"])

        if "slice" in config:
            values = list(config["slice"])
            while len(values) < 3:
                values.append(None)
            return slice(values[0], values[1], values[2])

        if "linspace" in config:
            start, stop, count = config["linspace"]
            if count <= 1:
                return [int(start)]
            step = (stop - start) / (count - 1)
            return [int(round(start + i * step)) for i in range(count)]

    raise ValueError(f"Unknown axis selection config: {config}")


def _normalize_axis_selection(axis_selection):
    if axis_selection is None:
        return {}

    return {
        int(axis): _selector_from_config(config)
        for axis, config in axis_selection.items()
    }


def _apply_axis_selection(selection, axis_selection, sample_dim=None):
    if axis_selection is None:
        return selection

    for axis, axis_selector in axis_selection.items():
        if sample_dim is None:
            dataset_axis = axis
        else:
            dataset_axis = axis if axis < sample_dim else axis + 1

        selection[dataset_axis] = axis_selector

    return selection


class DictSource:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        key = next(iter(self.data))
        return self.data[key].shape[0]

    def get_sample(self, idx):
        return {
            key: value[idx] for key, value in self.data.items()
        }


class NetCDFSource:
    def __init__(self, path, variable_name, sample_dim, axis_selection=None, ds=None, array=None):
        self.path = path
        self.variable_name = variable_name
        self.sample_dim = sample_dim
        self.axis_selection = _normalize_axis_selection(axis_selection)
        self._ds = ds
        self._array = array

    def __len__(self):
        self._open()
        return self._array.sizes[self.sample_dim]

    def get_sample(self, idx):
        self._open()
        sample = self._array.isel({self.sample_dim: idx})

        idx_dict = {}
        for axis, axis_selector in self.axis_selection.items():
            if axis < 0 or axis >= sample.ndim:
                raise ValueError(
                    f"axis_selection axis {axis} is out of range for NetCDF sample "
                    f"with ndim={sample.ndim}, dims={sample.dims}"
                )
            dim_name = sample.dims[axis]
            idx_dict[dim_name] = axis_selector
        sample = sample.isel(idx_dict)

        return torch.from_numpy(sample.to_numpy()).float()

    def _open(self):
        if self._ds is None:
            self._ds = xr.open_dataset(self.path)
        if self._array is None:
            if self.variable_name not in self._ds:
                available = list(self._ds.data_vars)
                raise KeyError(
                    f"Variable '{self.variable_name}' not found in {self.path}. "
                    f"Available variables: {available}"
                )
            self._array = self._ds[self.variable_name]


class MultiNetCDFSource:
    def __init__(self, components, length_component=None):
        self.components = components
        self.length_component = length_component

    def __len__(self):
        if not self.components:
            raise ValueError("MultiNetCDFSource must contain at least one component")

        if self.length_component is not None:
            if self.length_component not in self.components:
                raise ValueError(
                    f"length_component '{self.length_component}' not found. "
                    f"Available components: {list(self.components.keys())}"
                )
            return len(self.components[self.length_component])

        lengths = {
            name: len(source)
            for name, source in self.components.items()
        }

        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(f"All components must have the same length, got: {lengths}")

        return next(iter(unique_lengths))

    def get_sample(self, idx):
        return {
            name: source.get_sample(idx)
            for name, source in self.components.items()
        }


class ConcatSource:
    def __init__(self, sources):
        self.sources = list(sources)
        self._lengths = None
        self._offsets = None
        self._total_length = None

    def _build_index(self):
        if self._lengths is not None:
            return

        if not self.sources:
            raise ValueError("ConcatSource must contain at least one source")

        lengths = [len(source) for source in self.sources]

        for length in lengths:
            if length <= 0:
                raise ValueError(f"ConcatSource cannot use empty source, got length={length}")

        offsets = []
        total = 0

        for length in lengths:
            offsets.append(total)
            total += length

        self._lengths = lengths
        self._offsets = offsets
        self._total_length = total

    def __len__(self):
        self._build_index()
        return self._total_length

    def get_sample(self, idx):
        self._build_index()

        if idx < 0:
            idx = self._total_length + idx

        if idx < 0 or idx >= self._total_length:
            raise IndexError(f"Index {idx} out of range for ConcatSource with length {self._total_length}")

        source_idx = bisect_right(self._offsets, idx) - 1
        local_idx = idx - self._offsets[source_idx]

        return self.sources[source_idx].get_sample(local_idx)


class HDF5Source:
    def __init__(self, path, variable_path=None, sample_dim=0, variables=None, file=None, axis_selection=None):
        self.path = path
        self.variable_path = variable_path
        self.sample_dim = sample_dim
        self.variables = variables
        self._file = file
        self.axis_selection = _normalize_axis_selection(axis_selection)

    def __len__(self):
        dataset = self._get_length_dataset()
        return dataset.shape[self.sample_dim]

    def get_sample(self, idx):
        if self.variables is not None:
            return {
                name: self._read_dataset(variable_path, idx)
                for name, variable_path in self.variables.items()
            }

        return self._read_dataset(self.variable_path, idx)

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def _get_dataset(self, variable_path):
        file = self._open()

        if variable_path not in file:
            available = list(file.keys())
            raise KeyError(
                f"Variable '{variable_path}' not found in {self.path}. "
                f"Top-level keys: {available}"
            )

        dataset = file[variable_path]

        if not hasattr(dataset, "shape"):
            raise TypeError(f"HDF5 path '{variable_path}' is not a dataset")

        return dataset

    def _get_length_dataset(self):
        if self.variables is not None:
            first_variable_path = next(iter(self.variables.values()))
            return self._get_dataset(first_variable_path)

        return self._get_dataset(self.variable_path)

    def _read_dataset(self, variable_path, idx):
        dataset = self._get_dataset(variable_path)

        if idx < 0:
            idx = dataset.shape[self.sample_dim] + idx

        if idx < 0 or idx >= dataset.shape[self.sample_dim]:
            raise IndexError(
                f"Index {idx} out of range for HDF5 dataset '{variable_path}' "
                f"with length {dataset.shape[self.sample_dim]}"
            )

        selection = [slice(None)] * len(dataset.shape)
        selection[self.sample_dim] = idx
        selection = _apply_axis_selection(selection, self.axis_selection, sample_dim=self.sample_dim)

        return torch.from_numpy(dataset[tuple(selection)]).float()

    def __del__(self):
        if self._file is not None:
            self._file.close()


class HDF5GroupSource:
    def __init__(self, path, variable_path="data", variables=None, file=None, axis_selection=None):
        self.path = path
        self.variable_path = variable_path
        self.variables = variables
        self._file = file
        self._sample_keys = None
        self.axis_selection = _normalize_axis_selection(axis_selection)

    def __len__(self):
        self._ensure_sample_keys()
        return len(self._sample_keys)

    def get_sample(self, idx):
        self._ensure_sample_keys()

        if idx < 0:
            idx = len(self._sample_keys) + idx

        if idx < 0 or idx >= len(self._sample_keys):
            raise IndexError(f"Index {idx} out of range for HDF5GroupSource with length {len(self)}")

        sample_key = self._sample_keys[idx]

        if self.variables is not None:
            return {
                name: self._read_sample_dataset(sample_key, variable_path)
                for name, variable_path in self.variables.items()
            }

        return self._read_sample_dataset(sample_key, self.variable_path)

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def _ensure_sample_keys(self):
        if self._sample_keys is not None:
            return

        file = self._open()
        keys = sorted(file.keys())

        if not keys:
            raise ValueError(f"HDF5 file has no top-level sample groups: {self.path}")

        self._sample_keys = keys

    def _read_sample_dataset(self, sample_key, variable_path):
        file = self._open()
        full_path = f"{sample_key}/{variable_path}"

        if full_path not in file:
            available = list(file[sample_key].keys())
            raise KeyError(
                f"Variable '{variable_path}' not found in sample group '{sample_key}' "
                f"of {self.path}. Available keys: {available}"
            )

        # [5, :, :, :, :] --> [5, [0, ..., 999], 0:512:4, 0:512:4, :]
        #
        # for this axis_selection parameter:
        #       axis_selection:
        #         0:
        #           linspace: [0, 999, 16]
        #         1:
        #           slice: [0, 512, 4]
        #         2:
        #           slice: [0, 512, 4]

        dataset = file[full_path]
        selection = [slice(None)] * len(dataset.shape)
        selection = _apply_axis_selection(selection, self.axis_selection, sample_dim=None)

        return torch.from_numpy(dataset[tuple(selection)]).float()

    def __del__(self):
        if self._file is not None:
            self._file.close()


# The Well processing: description

# raw HDF5:
#   t0_fields/density      [B,T,X,Y]
#   t0_fields/pressure     [B,T,X,Y]
#   t1_fields/velocity     [B,T,X,Y,2]
#
# source output:
#   data                   [T,C,X,Y]
#   channel_names          [density, pressure, velocity_0, velocity_1]
#
# adapter output:
#   x/y                    [C,T,X,Y]


class TheWellHDF5Source:
    def __init__(
            self,
            path,
            field_groups=None,
            sample_dim=0,
            axis_selection=None,
            file=None,
            output_key="data",
    ):
        self.path = path
        self.field_groups = field_groups
        self.sample_dim = sample_dim
        self.axis_selection = _normalize_axis_selection(axis_selection)
        self._file = file
        self.output_key = output_key
        self.channel_names = None

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self):
        file = self._open()
        return int(file.attrs["n_trajectories"])

    def __del__(self):
        if self._file is not None:
            self._file.close()

    def _decode_name(self, name):
        if isinstance(name, bytes):
            return name.decode("utf-8")
        return str(name)

    def _get_field_groups(self, file):
        if self.field_groups is not None:
            return {
                group_name: [self._decode_name(name) for name in field_names]
                for group_name, field_names in self.field_groups.items()
            }

        groups = {}

        for group_name in ("t0_fields", "t1_fields", "t2_fields"):
            if group_name not in file:
                groups[group_name] = []
                continue

            field_names = file[group_name].attrs.get("field_names", [])
            groups[group_name] = [
                self._decode_name(name)
                for name in field_names
            ]

        return groups

    def _read_field(self, file, group_name, field_name, idx):
        if group_name not in file:
            raise KeyError(
                f"The Well group '{group_name}' not found in {self.path}. "
                f"Available groups: {list(file.keys())}"
            )

        group = file[group_name]
        if field_name not in group:
            raise KeyError(
                f"The Well field '{field_name}' not found in group '{group_name}' "
                f"of {self.path}. Available fields: {list(group.keys())}"
            )

        dataset = group[field_name]

        if idx < 0:
            idx = dataset.shape[self.sample_dim] + idx

        if idx < 0 or idx >= dataset.shape[self.sample_dim]:
            raise IndexError(
                f"Index {idx} out of range for The Well field "
                f"'{group_name}/{field_name}' with length "
                f"{dataset.shape[self.sample_dim]}"
            )

        selection = [slice(None)] * len(dataset.shape)
        selection[self.sample_dim] = idx

        selection = _apply_axis_selection(
            selection,
            self.axis_selection,
            sample_dim=self.sample_dim,
        )

        return torch.from_numpy(dataset[tuple(selection)]).float()

    def _field_to_channels(self, field, group_name, field_name, n_spatial_dims):
        spatial_shape = field.shape[1:1 + n_spatial_dims]
        component_shape = field.shape[1 + n_spatial_dims:]

        if group_name == "t0_fields":
            if len(component_shape) != 0:
                raise ValueError(
                    f"Expected scalar The Well field '{field_name}' to have no "
                    f"component axes, got shape {tuple(field.shape)}"
                )
            return field.unsqueeze(1)

        if group_name in ("t1_fields", "t2_fields"):
            if len(component_shape) == 0:
                raise ValueError(
                    f"Expected vector/tensor The Well field '{field_name}' to have "
                    f"component axes, got shape {tuple(field.shape)}"
                )

            n_components = 1
            for size in component_shape:
                n_components *= size

            field = field.reshape(
                field.shape[0],
                *spatial_shape,
                n_components,
            )

            permute_order = [0, len(spatial_shape) + 1]
            permute_order += list(range(1, len(spatial_shape) + 1))

            return field.permute(*permute_order)

        raise ValueError(f"Unknown The Well field group '{group_name}'")

    def _channel_names_for_field(self, field, group_name, field_name, n_spatial_dims):
        component_shape = field.shape[1 + n_spatial_dims:]

        if group_name == "t0_fields":
            return [field_name]

        if group_name in ("t1_fields", "t2_fields"):
            n_components = 1
            for size in component_shape:
                n_components *= size

            names = []
            for flat_idx in range(n_components):
                names.append(f"{field_name}_{flat_idx}")

            return names

        raise ValueError(f"Unknown The Well field group '{group_name}'")

    def get_sample(self, idx):
        file = self._open()
        n_spatial_dims = int(file.attrs["n_spatial_dims"])
        field_groups = self._get_field_groups(file)

        tensors = []
        channel_names = []

        for group_name in ("t0_fields", "t1_fields", "t2_fields"):
            for field_name in field_groups.get(group_name, []):
                field = self._read_field(file, group_name, field_name, idx)

                field_tensor = self._field_to_channels(
                    field,
                    group_name,
                    field_name,
                    n_spatial_dims,
                )

                tensors.append(field_tensor)

                channel_names.extend(
                    self._channel_names_for_field(
                        field,
                        group_name,
                        field_name,
                        n_spatial_dims,
                    )
                )

        if not tensors:
            raise ValueError(f"No The Well fields selected from {self.path}")

        data = torch.cat(tensors, dim=1)

        self.channel_names = channel_names

        return {
            self.output_key: data,
            "channel_names": channel_names,
            "metadata": {
                "dataset_name": self._decode_name(file.attrs.get("dataset_name", "")),
                "grid_type": self._decode_name(file.attrs.get("grid_type", "")),
                "n_spatial_dims": n_spatial_dims,
                "path": self.path,
            },
        }

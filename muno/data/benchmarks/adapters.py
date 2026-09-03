import torch


SPATIAL_AXIS_NAMES = ("X", "Y", "Z", "H", "W")


def _as_list(indices):
    return None if indices is None else list(indices)


def _get_sample_value(sample, variable_name=None):
    if variable_name is None:
        return sample
    if not isinstance(sample, dict):
        raise TypeError("variable_name can be used only when raw sample is a dict")
    return sample[variable_name]


def _spatial_axes(order):
    return [axis for axis in SPATIAL_AXIS_NAMES if axis in order]


def _to_c_spatial(data, order):
    order = order.upper()
    if len(order) != data.ndim:
        raise ValueError(f"order='{order}' does not match tensor shape={tuple(data.shape)}")

    spatial_axes = _spatial_axes(order)
    if len(spatial_axes) not in (1, 2, 3):
        raise ValueError(f"Expected 1D/2D/3D spatial axes in order='{order}'")

    if "C" in order:
        permute_order = [order.index("C")] + [order.index(axis) for axis in spatial_axes]
        return data.permute(*permute_order)

    permute_order = [order.index(axis) for axis in spatial_axes]
    return data.permute(*permute_order).unsqueeze(0)


def _to_tc_spatial(data, order):
    order = order.upper()
    if len(order) != data.ndim:
        raise ValueError(f"order='{order}' does not match tensor shape={tuple(data.shape)}")
    if "T" not in order:
        raise ValueError(f"Temporal data order must contain 'T', got order='{order}'")

    spatial_axes = _spatial_axes(order)
    if len(spatial_axes) not in (1, 2, 3):
        raise ValueError(f"Expected 1D/2D/3D spatial axes in order='{order}'")

    if "C" in order:
        permute_order = [order.index("T"), order.index("C")]
        permute_order += [order.index(axis) for axis in spatial_axes]
        return data.permute(*permute_order)

    permute_order = [order.index("T")] + [order.index(axis) for axis in spatial_axes]
    return data.permute(*permute_order).unsqueeze(1)


def _ensure_2d(x, y, ensure_2d):
    if ensure_2d and x.ndim == 2:
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
    return x, y


class IndexAdapter:
    def __init__(
        self,
        input_indices,
        output_indices,
        variable_name=None,
        data_order="CHW",
        benchmark_name=None,
        physics_name=None,
        metadata=None,
        ensure_2d=False,
    ):
        self.input_indices = input_indices
        self.output_indices = output_indices
        self.variable_name = variable_name
        self.data_order = data_order
        self.benchmark_name = benchmark_name
        self.physics_name = physics_name
        self.metadata = {} if metadata is None else metadata
        self.ensure_2d = ensure_2d

    def canonize(self, sample):
        data = _get_sample_value(sample, self.variable_name)
        data = _to_c_spatial(data, self.data_order)

        x = data[_as_list(self.input_indices)]
        y = data[_as_list(self.output_indices)]
        x, y = _ensure_2d(x, y, self.ensure_2d)

        return {
            "x": x.float(),
            "y": y.float(),
            "benchmark_name": self.benchmark_name,
            "physics_name": self.physics_name,
            "metadata": self.metadata,
        }


class TemporalAdapter:
    def __init__(
        self,
        variable_name=None,
        variable_names=None,
        data_order="TCHW",
        temporal_mode="window",
        input_time_indices=None,
        output_time_indices=None,
        window_start_indices=None,
        input_time_index=0,
        input_channel_indices=None,
        output_channel_indices=None,
        static_inputs=None,
        benchmark_name=None,
        physics_name=None,
        metadata=None,
        ensure_2d=False,
        flatten_time_to_channels=True,
    ):
        self.variable_name = variable_name
        self.variable_names = variable_names
        self.data_order = data_order

        self.temporal_mode = temporal_mode
        self.input_time_indices = input_time_indices
        self.output_time_indices = output_time_indices
        self.window_start_indices = window_start_indices
        self.input_time_index = input_time_index

        self.input_channel_indices = input_channel_indices
        self.output_channel_indices = output_channel_indices
        self.static_inputs = [] if static_inputs is None else static_inputs
        self.benchmark_name = benchmark_name
        self.physics_name = physics_name
        self.metadata = {} if metadata is None else metadata
        self.ensure_2d = ensure_2d
        self.flatten_time_to_channels = flatten_time_to_channels

    def canonize(self, sample, window_start=None):
        data = self._read_temporal_data(sample)

        if self.temporal_mode == "window":
            x, y = self._canonize_window(data, window_start=window_start)
        elif self.temporal_mode == "initial_to_trajectory":
            x, y = self._canonize_initial_to_trajectory(data)
        else:
            raise ValueError(f"Unknown temporal_mode: {self.temporal_mode}")

        if self.input_channel_indices is not None:
            x = x[:, _as_list(self.input_channel_indices)]
        if self.output_channel_indices is not None:
            y = y[:, _as_list(self.output_channel_indices)]

        x, y = self._format_temporal_tensors(x, y)

        for static_config in self.static_inputs:
            x, y = self._append_static_input(sample, x, y, static_config)

        x, y = _ensure_2d(x, y, self.ensure_2d)

        return {
            "x": x.float(),
            "y": y.float(),
            "benchmark_name": self.benchmark_name,
            "physics_name": self.physics_name,
            "metadata": self.metadata,
        }

    def _read_temporal_data(self, sample):
        if self.variable_names is not None:
            fields = []
            for variable_name in self.variable_names:
                field = _get_sample_value(sample, variable_name)
                field = _to_tc_spatial(field, self.data_order)
                fields.append(field)
            return torch.cat(fields, dim=1)

        data = _get_sample_value(sample, self.variable_name)
        return _to_tc_spatial(data, self.data_order)

    def _canonize_window(self, data, window_start=None):
        if self.input_time_indices is None:
            raise ValueError("temporal_mode='window' requires input_time_indices")
        if self.output_time_indices is None:
            raise ValueError("temporal_mode='window' requires output_time_indices")

        input_time_indices = _as_list(self.input_time_indices)
        output_time_indices = _as_list(self.output_time_indices)

        if window_start is not None:
            window_start = int(window_start)
            input_time_indices = [window_start + idx for idx in input_time_indices]
            output_time_indices = [window_start + idx for idx in output_time_indices]

        max_time_index = max(input_time_indices + output_time_indices)
        if max_time_index >= data.shape[0]:
            raise IndexError(
                f"Temporal window reaches time index {max_time_index}, "
                f"but sample has only {data.shape[0]} time steps"
            )

        x = data[input_time_indices]
        y = data[output_time_indices]
        return x, y

    def _canonize_initial_to_trajectory(self, data):
        target_time_indices = self.output_time_indices

        if target_time_indices is None or target_time_indices == "all":
            y = data
        else:
            y = data[_as_list(target_time_indices)]

        x0 = data[self.input_time_index:self.input_time_index + 1]
        x = x0.expand(y.shape[0], -1, *([-1] * (y.ndim - 2)))

        return x, y

    def _format_temporal_tensors(self, x, y):
        if self.flatten_time_to_channels:
            x = x.flatten(0, 1)
            y = y.flatten(0, 1)
            return x, y

        x = x.permute(1, 0, *range(2, x.ndim))
        y = y.permute(1, 0, *range(2, y.ndim))
        return x, y

    def _append_static_input(self, sample, x, y, static_config):
        static_value = _get_sample_value(sample, static_config["variable_name"])
        static_value = _to_c_spatial(static_value, static_config.get("data_order", "CHW"))
        target = static_config.get("target", "x")

        if self.flatten_time_to_channels:
            static_x = static_value
            static_y = static_value
        else:
            static_x = static_value.unsqueeze(1).expand(-1, x.shape[1], *static_value.shape[1:])
            static_y = static_value.unsqueeze(1).expand(-1, y.shape[1], *static_value.shape[1:])

        if target == "x":
            x = torch.cat([x, static_x], dim=0)
        elif target == "y":
            y = torch.cat([y, static_y], dim=0)
        elif target == "both":
            x = torch.cat([x, static_x], dim=0)
            y = torch.cat([y, static_y], dim=0)
        else:
            raise ValueError(f"Unknown static input target: {target}")

        return x, y


class InputOutputAdapter:
    def __init__(
        self,
        input_variable_name,
        output_variable_name,
        input_order="CHW",
        output_order="CHW",
        benchmark_name=None,
        physics_name=None,
        metadata=None,
        ensure_2d=False,
    ):
        self.input_variable_name = input_variable_name
        self.output_variable_name = output_variable_name
        self.input_order = input_order
        self.output_order = output_order
        self.benchmark_name = benchmark_name
        self.physics_name = physics_name
        self.metadata = {} if metadata is None else metadata
        self.ensure_2d = ensure_2d

    def canonize(self, sample):
        x = _to_c_spatial(sample[self.input_variable_name], self.input_order)
        y = _to_c_spatial(sample[self.output_variable_name], self.output_order)
        x, y = _ensure_2d(x, y, self.ensure_2d)

        return {
            "x": x.float(),
            "y": y.float(),
            "benchmark_name": self.benchmark_name,
            "physics_name": self.physics_name,
            "metadata": self.metadata,
        }

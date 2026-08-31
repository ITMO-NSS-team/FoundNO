import torch

from muno.utils.metrics import check_same_shape, check_batched_channel_tensor


def kinetic_energy_2d(
        tensor,
        velocity_channels=(0, 1),
        density_channel=None,
        subtract_mean=False
):
    check_batched_channel_tensor(tensor)

    vx_idx, vy_idx = velocity_channels
    vx = tensor[:, vx_idx]
    vy = tensor[:, vy_idx]

    reduce_dims = tuple(range(1, vx.ndim))

    if subtract_mean:
        vx = vx - torch.mean(vx, dim=reduce_dims, keepdim=True)
        vy = vy - torch.mean(vy, dim=reduce_dims, keepdim=True)

    velocity_squared = vx ** 2 + vy ** 2

    if density_channel is not None:
        density = tensor[:, density_channel]
        energy_density = density * velocity_squared
    else:
        energy_density = velocity_squared

    return 0.5 * torch.mean(energy_density, dim=reduce_dims)


def kinetic_energy_error(
        pred,
        target,
        velocity_channels=(0, 1),
        density_channel=None,
        subtract_mean=False,
):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)

    pred_ke = kinetic_energy_2d(
        pred,
        velocity_channels=velocity_channels,
        density_channel=density_channel,
        subtract_mean=subtract_mean
    )
    target_ke = kinetic_energy_2d(
        target,
        velocity_channels=velocity_channels,
        density_channel=density_channel,
        subtract_mean=subtract_mean
    )

    return torch.mean(torch.abs(pred_ke - target_ke))


def relative_kinetic_energy_error(
        pred,
        target,
        velocity_channels=(0, 1),
        density_channel=None,
        subtract_mean=False,
        eps=1e-7
):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)

    pred_ke = kinetic_energy_2d(
        pred,
        velocity_channels=velocity_channels,
        density_channel=density_channel,
        subtract_mean=subtract_mean
    )
    target_ke = kinetic_energy_2d(
        target,
        velocity_channels=velocity_channels,
        density_channel=density_channel,
        subtract_mean=subtract_mean
    )

    error = torch.abs(pred_ke - target_ke)
    scale = torch.abs(target_ke).clamp_min(eps)

    return torch.mean(error / scale)


def spectral_mse(pred, target, spatial_dims=(-2, -1)):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)

    pred_fft = torch.fft.fftn(pred, dim=spatial_dims)
    target_fft = torch.fft.fftn(target, dim=spatial_dims)

    return torch.mean(torch.abs(pred_fft - target_fft) ** 2)


def spectral_nrmse(pred, target, spatial_dims=(-2, -1), eps=1e-7):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)

    pred_fft = torch.fft.fftn(pred, dim=spatial_dims)
    target_fft = torch.fft.fftn(target, dim=spatial_dims)

    error_power = torch.mean(torch.abs(pred_fft - target_fft) ** 2)
    target_power = torch.mean(torch.abs(target_fft) ** 2).clamp_min(eps)

    return torch.sqrt(error_power / target_power)


def _frequency_radius_2d(height, width, device):
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.fftfreq(width, device=device)

    ky, kx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.sqrt(kx ** 2 + ky ** 2)


def band_spectral_mse_2d(
        pred,
        target,
        num_bins=3,
        spatial_dims=(-2, -1),
        bin_names=None,
):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)
    check_batched_channel_tensor(target)

    if len(spatial_dims) != 2:
        raise ValueError("binned_spectral_mse_2d supports exactly 2 spatial dimensions")

    if bin_names is None:
        if num_bins == 3:
            bin_names = ["low", "mid", "high"]
        else:
            bin_names = [f"bin_{idx}" for idx in range(num_bins)]

    if len(bin_names) != num_bins:
        raise ValueError(
            f"bin_names length must match num_bins. "
            f"Got len(bin_names)={len(bin_names)}, num_bins={num_bins}"
        )

    height = pred.shape[spatial_dims[0]]
    width = pred.shape[spatial_dims[1]]

    pred_fft = torch.fft.fftn(pred, dim=spatial_dims)
    target_fft = torch.fft.fftn(target, dim=spatial_dims)

    radius = _frequency_radius_2d(height, width, pred.device)

    positive_radius = radius[radius > 0]
    if positive_radius.numel() == 0:
        raise ValueError("Cannot build spectral bins: no positive frequencies found")

    min_radius = torch.min(positive_radius)
    max_radius = torch.max(positive_radius)

    edges = torch.logspace(
        torch.log10(min_radius),
        torch.log10(max_radius),
        steps=num_bins + 1,
        device=pred.device,
    )

    spectral_error = torch.abs(pred_fft - target_fft) ** 2

    results = {}

    for bin_idx in range(num_bins):
        left = edges[bin_idx]
        right = edges[bin_idx + 1]

        if bin_idx == num_bins - 1:
            mask = (radius >= left) & (radius <= right)
        else:
            mask = (radius >= left) & (radius < right)

        if not torch.any(mask):
            results[f"spectral_mse_{bin_names[bin_idx]}"] = float("nan")
            continue

        selected_error = spectral_error[..., mask]
        error_power = torch.mean(selected_error)

        bin_name = bin_names[bin_idx]
        results[f"spectral_mse_{bin_name}"] = float(error_power.detach().cpu())

    return results


def band_spectral_nrmse_2d(
        pred,
        target,
        num_bins=3,
        spatial_dims=(-2, -1),
        bin_names=None,
        eps=1e-7,
):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)

    if len(spatial_dims) != 2:
        raise ValueError("binned_spectral_nrmse_2d supports exactly 2 spatial dimensions")

    if bin_names is None:
        if num_bins == 3:
            bin_names = ["low", "mid", "high"]
        else:
            bin_names = [f"bin_{idx}" for idx in range(num_bins)]

    if len(bin_names) != num_bins:
        raise ValueError(
            f"bin_names length must match num_bins. "
            f"Got len(bin_names)={len(bin_names)}, num_bins={num_bins}"
        )

    height = pred.shape[spatial_dims[0]]
    width = pred.shape[spatial_dims[1]]

    pred_fft = torch.fft.fftn(pred, dim=spatial_dims)
    target_fft = torch.fft.fftn(target, dim=spatial_dims)

    radius = _frequency_radius_2d(height, width, pred.device)

    positive_radius = radius[radius > 0]
    if positive_radius.numel() == 0:
        raise ValueError("Cannot build spectral bins: no positive frequencies found")

    min_radius = torch.min(positive_radius)
    max_radius = torch.max(positive_radius)

    edges = torch.logspace(
        torch.log10(min_radius),
        torch.log10(max_radius),
        steps=num_bins + 1,
        device=pred.device,
    )

    spectral_error = torch.abs(pred_fft - target_fft) ** 2
    target_power = torch.abs(target_fft) ** 2

    results = {}

    for bin_idx in range(num_bins):
        left = edges[bin_idx]
        right = edges[bin_idx + 1]

        if bin_idx == num_bins - 1:
            mask = (radius >= left) & (radius <= right)
        else:
            mask = (radius >= left) & (radius < right)

        if not torch.any(mask):
            results[f"spectral_nrmse_{bin_names[bin_idx]}"] = float("nan")
            continue

        selected_error = spectral_error[..., mask]
        selected_target_power = target_power[..., mask]

        error_power = torch.mean(selected_error)
        target_bin_power = torch.mean(selected_target_power).clamp_min(eps)

        bin_name = bin_names[bin_idx]
        results[f"spectral_nrmse_{bin_name}"] = float(
            torch.sqrt(error_power / target_bin_power).detach().cpu()
        )

    return results


PHYSICAL_METRIC_REGISTRY = {
    "kinetic_energy_error": kinetic_energy_error,
    "relative_kinetic_energy_error": relative_kinetic_energy_error,
    "spectral_mse": spectral_mse,
    "spectral_nrmse": spectral_nrmse,
    "band_spectral_mse_2d": band_spectral_mse_2d,
    "band_spectral_nrmse_2d": band_spectral_nrmse_2d,
}


def compute_physical_metrics(pred, target, metric_configs=None):
    if metric_configs is None:
        metric_configs = [
            {"name": "kinetic_energy_error"},
            {"name": "relative_kinetic_energy_error"},
            {"name": "spectral_nrmse"},
            {"name": "band_spectral_nrmse_2d"},
        ]

    results = {}

    for metric_config in metric_configs:
        name = metric_config["name"]
        kwargs = {
            key: value
            for key, value in metric_config.items()
            if key != "name"
        }

        if name not in PHYSICAL_METRIC_REGISTRY:
            raise ValueError(
                f"Unknown physical metric '{name}'. "
                f"Available physical metrics: {list(PHYSICAL_METRIC_REGISTRY.keys())}"
            )

        value = PHYSICAL_METRIC_REGISTRY[name](pred, target, **kwargs)

        if isinstance(value, dict):
            results.update(value)
        else:
            results[name] = float(value.detach().cpu())

    return results

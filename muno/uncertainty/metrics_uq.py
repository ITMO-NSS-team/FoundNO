import torch

from muno.utils.metrics import check_same_shape, check_batched_channel_tensor


def _validate_inputs(pred, band, target):
    check_same_shape(pred, band)
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)


def covered_mask(pred, band, target, k=1.0):
    _validate_inputs(pred, band, target)
    error = torch.abs(pred - target)
    return error <= k * band


def coverage_rate(pred, band, target, k=1.0):
    mask = covered_mask(pred, band, target, k=k)
    return torch.mean(mask.to(pred.dtype))


def normalized_avg_bandwidth(pred, band, target, k=1.0, eps=1e-7):
    _validate_inputs(pred, band, target)

    field_scale = torch.sqrt(torch.mean(target ** 2)).clamp_min(eps)

    mask = covered_mask(pred, band, target, k=k)

    mean_bw_all = torch.mean(band)

    if torch.any(mask):
        mean_bw_covered = torch.mean(band[mask])
    else:
        mean_bw_covered = torch.full((), float("nan"), dtype=band.dtype, device=band.device)

    if torch.any(~mask):
        mean_bw_missed = torch.mean(band[~mask])
    else:
        mean_bw_missed = torch.full((), float("nan"), dtype=band.dtype, device=band.device)

    return {
        "n_avg_bw_all": float((mean_bw_all / field_scale).detach().cpu()),
        "n_avg_bw_covered": float((mean_bw_covered / field_scale).detach().cpu()),
        "n_avg_bw_missed": float((mean_bw_missed / field_scale).detach().cpu()),
    }


UQ_METRIC_REGISTRY = {
    "coverage_rate": coverage_rate,
    "normalized_avg_bandwidth": normalized_avg_bandwidth,
}


def compute_uq_metrics(pred, band, target, metric_configs=None):
    if metric_configs is None:
        metric_configs = [
            {"name": "coverage_rate"},
            {"name": "normalized_avg_bandwidth"},
        ]

    results = {}

    for metric_config in metric_configs:
        name = metric_config["name"]
        kwargs = {
            key: value
            for key, value in metric_config.items()
            if key != "name"
        }

        if name not in UQ_METRIC_REGISTRY:
            raise ValueError(
                f"Unknown physical metric '{name}'. "
                f"Available physical metrics: {list(UQ_METRIC_REGISTRY.keys())}"
            )

        value = UQ_METRIC_REGISTRY[name](pred, band, target, **kwargs)

        if isinstance(value, dict):
            results.update(value)
        else:
            results[name] = float(value.detach().cpu())

    return results
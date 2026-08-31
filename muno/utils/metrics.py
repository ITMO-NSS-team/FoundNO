import torch


def check_same_shape(pred, target):
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes must match. "
            f"Got pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )


def check_batched_channel_tensor(tensor):
    if tensor.ndim < 3:
        raise ValueError(f"Expected tensor [B, C, ...], got shape {tuple(tensor.shape)}")


def flatten_per_sample(tensor):
    return tensor.reshape(tensor.shape[0], -1)


def mse(pred, target):
    check_same_shape(pred, target)
    return torch.mean((pred - target) ** 2)


def mae(pred, target):
    check_same_shape(pred, target)
    return torch.mean(torch.abs(pred - target))


def rmse(pred, target):
    return torch.sqrt(mse(pred, target))


def nrmse(pred, target, eps=1e-7):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)

    numerator = rmse(pred, target)
    denominator = torch.sqrt(torch.mean(target ** 2)).clamp_min(eps)

    return numerator / denominator


def vrmse(pred, target):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)

    reduce_dims = (0,) + tuple(range(2, pred.ndim))
    channel_mse = torch.mean((pred - target) ** 2, dim=reduce_dims)
    channel_rmse = torch.sqrt(channel_mse)

    return torch.mean(channel_rmse)


def relative_l2(pred, target, eps=1e-7):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)

    pred_flat = flatten_per_sample(pred)
    target_flat = flatten_per_sample(target)

    error_norm = torch.linalg.vector_norm(pred_flat - target_flat, ord=2, dim=1)
    target_norm = torch.linalg.vector_norm(target_flat, ord=2, dim=1).clamp_min(eps)

    return torch.mean(error_norm / target_norm)


def balanced_relative_l2(pred, target, eps=1e-7):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)

    channel_errors = []

    for channel_idx in range(pred.shape[1]):
        pred_channel = pred[:, channel_idx:channel_idx + 1]
        target_channel = target[:, channel_idx:channel_idx + 1]

        pred_flat = flatten_per_sample(pred_channel)
        target_flat = flatten_per_sample(target_channel)

        error_norm = torch.linalg.vector_norm(pred_flat - target_flat, ord=2, dim=1)
        target_norm = torch.linalg.vector_norm(target_flat, ord=2, dim=1).clamp_min(eps)

        channel_errors.append(error_norm / target_norm)

    channel_errors = torch.stack(channel_errors, dim=1)
    return torch.mean(channel_errors)


def nmae_range(pred, target, eps=1e-7):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)

    pred_flat = flatten_per_sample(pred)
    target_flat = flatten_per_sample(target)

    abs_error = torch.mean(torch.abs(pred_flat - target_flat), dim=1)
    target_range = (
            torch.amax(target_flat, dim=1) - torch.amin(target_flat, dim=1)
    ).clamp_min(eps)

    return torch.mean(abs_error / target_range)


def nmae_mean(pred, target, eps=1e-7):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)

    pred_flat = flatten_per_sample(pred)
    target_flat = flatten_per_sample(target)

    abs_error = torch.mean(torch.abs(pred_flat - target_flat), dim=1)
    target_scale = torch.mean(torch.abs(target_flat), dim=1).clamp_min(eps)

    return torch.mean(abs_error / target_scale)


def nmae_std(pred, target, eps=1e-7):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)

    pred_flat = flatten_per_sample(pred)
    target_flat = flatten_per_sample(target)

    abs_error = torch.mean(torch.abs(pred_flat - target_flat), dim=1)
    target_scale = torch.std(target_flat, dim=1, unbiased=False).clamp_min(eps)

    return torch.mean(abs_error / target_scale)


def r2_score(pred, target, eps=1e-7):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)

    pred_flat = flatten_per_sample(pred)
    target_flat = flatten_per_sample(target)

    target_mean = torch.mean(target_flat, dim=1, keepdim=True)

    ss_res = torch.sum((target_flat - pred_flat) ** 2, dim=1)
    ss_tot = torch.sum((target_flat - target_mean) ** 2, dim=1).clamp_min(eps)

    return torch.mean(1.0 - ss_res / ss_tot)


def mean_linf_error(pred, target):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)

    pred_flat = flatten_per_sample(pred)
    target_flat = flatten_per_sample(target)

    per_sample_max = torch.amax(torch.abs(pred_flat - target_flat), dim=1)

    return torch.mean(per_sample_max)


def max_linf_error(pred, target):
    check_same_shape(pred, target)
    check_batched_channel_tensor(pred)

    return torch.amax(torch.abs(pred - target))


METRIC_REGISTRY = {
    "mse": mse,
    "mae": mae,
    "rmse": rmse,
    "nrmse": nrmse,
    "vrmse": vrmse,
    "relative_l2": relative_l2,
    "balanced_relative_l2": balanced_relative_l2,
    "nmae_range": nmae_range,
    "nmae_mean": nmae_mean,
    "nmae_std": nmae_std,
    "r2_score": r2_score,
    "mean_linf_error": mean_linf_error,
    "max_linf_error": max_linf_error
}


def compute_metrics(pred, target, metric_names=None):
    if metric_names is None:
        metric_names = list(METRIC_REGISTRY.keys())

    results = {}
    for name in metric_names:
        if name not in METRIC_REGISTRY:
            raise ValueError(
                f"Unknown metric '{name}'. "
                f"Available metrics: {list(METRIC_REGISTRY.keys())}"
            )

        value = METRIC_REGISTRY[name](pred, target)
        results[name] = float(value.detach().cpu())

    return results

from __future__ import annotations

import torch

from .progress import tqdm


def nll_gaussian(mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor, scaled: bool = True) -> torch.Tensor:
    """Negative log-likelihood under a diagonal Gaussian predictive.

    Per-element:
        nll = 0.5 * ((target - mean) / std)^2 + log(std * sqrt(2 pi))
    ``scaled=True`` returns the mean over elements (laplax ``nll_gaussian`` default).
    """
    std_ext = std.reshape(-1)
    mean_ext = mean.reshape(-1)
    target_ext = target.reshape(-1)
    eps = torch.finfo(std_ext.dtype).eps
    std_safe = std_ext.clamp_min(eps)
    nll = (
        0.5 * ((target_ext - mean_ext) / std_safe) ** 2
        + std_safe.log()
        + 0.5 * torch.log(torch.tensor(2.0 * torch.pi, dtype=std_ext.dtype, device=std_ext.device))
    )
    if scaled:
        return nll.mean()
    return nll.sum()


def chi_squared(pred_mean, pred_std, target, *, averaged=True):
    """Эквивалент laplax ``chi_squared``: (y - mean)^2 / std^2.

    ``averaged=True`` -- среднее по элементам, иначе сумма.
    """
    pm = pred_mean.reshape(-1)
    ps = pred_std.reshape(-1)
    tg = target.reshape(-1)
    val = (pm - tg) ** 2 / ps**2
    return val.mean() if averaged else val.sum()


def chi_squared_zero(pred_mean, pred_std, target, *, averaged=True):
    """Калибровочный критерий ``|chi_squared - 1|`` (laplax ``chi_squared_zero``)."""
    return torch.abs(chi_squared(pred_mean, pred_std, target, averaged=averaged) - 1)


def evaluate_for_given_prior_arguments(
    prob_predictive: callable,
    prior_args: dict,
    batch: dict,
) -> torch.Tensor:
    """NLL of the calibrated predictive at a fixed prior argument on one batch.

    ``batch`` must contain ``input`` (model input) and ``target`` (supervised output),
    matching luno experiments' loader batches.
    """
    out = prob_predictive(batch["input"])
    return nll_gaussian(out["pred_mean"], out["pred_std"], batch["target"])


def grid_search(
    range_values,
    objective: callable,
    patience: int = 5,
    maximize: bool = False,
    progress: bool = False,
    desc: str = "grid_search",
) -> tuple:
    """Grid search with early stopping after ``patience`` non-improving steps.

    Returns ``(best_value, best_index)``.
    """
    best_value = -torch.inf if maximize else torch.inf
    best_idx: int = 0
    bad = 0
    bar = tqdm(
        total=len(range_values), desc=f"[{desc}]", unit="val", leave=False
    ) if progress else None
    try:
        for i, value in enumerate(range_values):
            val = float(objective(value))
            improved = val > best_value if maximize else val < best_value
            if improved:
                best_value = val
                best_idx = i
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    return best_value, best_idx
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()
    return best_value, best_idx


def optimize_prior_prec(
    prob_predictive: callable,
    data: dict,
    prior_prec_min: float = -3.0,
    prior_prec_max: float = 3.0,
    grid_size: int = 50,
    patience: int = 5,
    verbose: bool = True,
) -> dict:
    """Calibrate the scalar prior precision on a log10 grid of the NLL objective."""
    grid = torch.logspace(prior_prec_min, prior_prec_max, grid_size, base=10.0)

    def objective(prec):
        return evaluate_for_given_prior_arguments(
            prob_predictive, {"prior_prec": prec}, data
        )

    best_value, best_idx = grid_search(
        grid,
        objective,
        patience=patience,
        maximize=False,
    )
    if verbose:
        print(f"[calibrate] best prior_prec={grid[best_idx].item():.6e}, nll={best_value:.6e}")
    return {"prior_prec": grid[best_idx]}
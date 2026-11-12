from __future__ import annotations

import math

import pytest
import torch

from metrics_uq import log_rmse_over_rmv, reliability_bin_sums


def _calibrated(g=2, n=20000, seed=0):
    gen = torch.Generator().manual_seed(seed)
    # Ограниченный разброс std: у логнормального в верхнем бине тяжёлый хвост,
    # и RMSE по ~2000 точкам там шумит на десятки процентов.
    std = 0.2 + 4.8 * torch.rand(g, n, generator=gen, dtype=torch.float64)
    err = std * torch.randn(g, n, generator=gen, dtype=torch.float64)
    return std, err ** 2


def test_calibrated_model_is_flat_zero():
    std, err2 = _calibrated()
    s2, e2, cnt = reliability_bin_sums(std, err2, n_bins=10)
    assert torch.equal(cnt[0], torch.full((10,), 2000.0, dtype=torch.float64))
    lr = log_rmse_over_rmv(e2, s2)
    # ~2000 точек на бин: относительная ошибка RMSE ~ 1/sqrt(2*2000) ~ 1.6 %.
    assert lr.abs().max() < 0.06


def test_global_scale_shifts_curve_by_log():
    std, err2 = _calibrated()
    s2, e2, _ = reliability_bin_sums(2.0 * std, err2, n_bins=10)
    lr = log_rmse_over_rmv(e2, s2)
    assert torch.allclose(lr, torch.full_like(lr, -math.log(2.0)), atol=0.06)


def test_too_flat_std_shows_slope():
    """std = sqrt(истинного): малые std занижены относительно ошибки, большие -- завышены."""
    std, err2 = _calibrated()
    s2, e2, _ = reliability_bin_sums(std.sqrt(), err2, n_bins=10)
    lr = log_rmse_over_rmv(e2, s2)[0]
    # Ожидание: log(RMSE/RMV) = 0.5*log(std) -- около -0.4 в первом бине и +0.75 в последнем.
    assert lr[0] < -0.25 and lr[-1] > 0.5  # бины от уверенных к неуверенным
    assert bool((lr[1:] > lr[:-1]).all())


def test_zero_std_and_zero_over_zero():
    std, err2 = _calibrated(g=1, n=100)
    std[0, :10] = 0.0  # первый бин целиком с нулевым std
    s2, e2, _ = reliability_bin_sums(std, err2, n_bins=10)
    lr = log_rmse_over_rmv(e2, s2)
    assert lr[0, 0] == math.inf and torch.isfinite(lr[0, 1:]).all()
    err2[0, std[0] == 0.0] = 0.0  # 0/0 -> nan
    s2, e2, _ = reliability_bin_sums(std, err2, n_bins=10)
    assert math.isnan(log_rmse_over_rmv(e2, s2)[0, 0])


def test_nonfinite_group_is_nan():
    std, err2 = _calibrated()
    std[1, 3] = float("nan")
    s2, _, _ = reliability_bin_sums(std, err2)
    assert torch.isnan(s2[1]).all() and torch.isfinite(s2[0]).all()

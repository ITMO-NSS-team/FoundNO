from __future__ import annotations

import math

import pytest
import torch

run = pytest.importorskip("run_luno_ggn_calibrate")


def _fields(seed=0, shape=(1, 3, 5, 8, 8)):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(shape, generator=g, dtype=torch.float64)
    pred = target + 0.3 * torch.randn(shape, generator=g, dtype=torch.float64)
    std = 0.1 + torch.rand(shape, generator=g, dtype=torch.float64)
    return pred, std, target


def test_matches_direct_quantile_per_slice():
    pred, std, target = _fields()
    out = run.per_channel_time_sqrt_chi2(pred, std, target, pct=run.PER_CHANNEL_PCT)
    assert len(out) == len(run.PER_CHANNEL_PCT) and len(out[0]) == 3 and len(out[0][0]) == 5
    for li, q in enumerate(run.PER_CHANNEL_PCT):
        for c in range(3):
            for t in range(5):
                r = ((pred - target).abs() / std)[0, c, t].reshape(-1)
                ref = torch.quantile(r, q / 100.0).item()
                assert out[li][c][t] == pytest.approx(ref, rel=1e-12)


def test_per_channel_unchanged_by_refactor():
    """Поканальная версия = квантиль по всем t и пикселям канала."""
    pred, std, target = _fields(seed=1)
    out = run.per_channel_p99(pred, std, target, pct=(80, 99))
    for li, q in enumerate((80, 99)):
        for c in range(3):
            r = ((pred - target).abs() / std)[0, c].reshape(-1)
            assert out[li][c] == pytest.approx(torch.quantile(r, q / 100.0).item(), rel=1e-12)


def test_zero_std_gives_inf_and_zero_over_zero_is_dropped():
    pred, std, target = _fields(seed=2)
    # Всё поле (канал 1, t=2) с std = 0 и ненулевой ошибкой -> квантили inf.
    std[0, 1, 2] = 0.0
    out = run.per_channel_time_sqrt_chi2(pred, std, target, pct=(50, 99))
    assert out[0][1][2] == math.inf and out[1][1][2] == math.inf
    assert math.isfinite(out[0][1][3])
    # Малая доля inf (< 1 - q) не трогает квантиль: он равен квантилю по
    # данным, где эти пиксели заменены на любые большие числа.
    pred, std, target = _fields(seed=4)
    std[0, 0, 0, :2, :] = 0.0  # 16 из 64 пикселей поля (25 %)
    r = ((pred - target).abs() / std)[0, 0, 0].reshape(-1)
    ref50 = torch.quantile(torch.nan_to_num(r, posinf=1e30), 0.5).item()
    out = run.per_channel_time_sqrt_chi2(pred, std, target, pct=(50, 90))
    assert out[0][0][0] == pytest.approx(ref50, rel=1e-12)
    assert out[1][0][0] == math.inf  # p90 уже внутри inf-хвоста
    # 0/0 отбрасывается: квантиль считается по остальным пикселям.
    pred, std, target = _fields(seed=5)
    pred[0, 2, 1, 0, :] = target[0, 2, 1, 0, :]
    std[0, 2, 1, 0, :] = 0.0
    r = ((pred - target).abs() / std)[0, 2, 1].reshape(-1)
    ref = torch.nanquantile(r, 0.75).item()
    out = run.per_channel_time_sqrt_chi2(pred, std, target, pct=75)
    assert math.isfinite(ref) and out[0][2][1] == pytest.approx(ref, rel=1e-12)


def test_no_time_axis_matches_per_channel():
    p2, s2, t2 = (x[:, :, 0] for x in _fields(seed=3))
    flat = run.per_channel_time_sqrt_chi2(p2, s2, t2, pct=90)
    assert [row[0] for row in flat[0]] == run.per_channel_p99(p2, s2, t2, pct=90)[0]


def test_mean_non_nan_keeps_inf():
    assert run.mean_non_nan([1.0, float("nan"), 3.0]) == 2.0
    assert run.mean_non_nan([1.0, math.inf]) == math.inf
    assert math.isnan(run.mean_non_nan([float("nan")]))

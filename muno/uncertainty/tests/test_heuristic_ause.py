"""Эвристики ошибки (run_heuristic_ause): карты, доля top-10 %, AUSE и агрегация."""
from __future__ import annotations

import math

import pytest
import torch

heur = pytest.importorskip("run_heuristic_ause")


def _field(C=2, T=3, H=16, W=12):
    t = torch.arange(T, dtype=torch.float64).view(1, 1, T, 1, 1)
    x = torch.arange(H, dtype=torch.float64).view(1, 1, 1, H, 1)
    y = torch.arange(W, dtype=torch.float64).view(1, 1, 1, 1, W)
    return t, x, y


def test_grad_and_lap_on_polynomials():
    t, x, y = _field()
    lin = (3.0 * x + 4.0 * y + 0.0 * t).expand(1, 2, 3, 16, 12)
    maps = heur.heuristic_maps(lin)
    assert torch.allclose(maps["grad"], torch.full_like(maps["grad"], 5.0))
    # Лапласиан линейного поля -- 0 (включая края: повтор края даёт 0 только
    # внутри, поэтому проверяем внутренность).
    assert maps["lap"][..., 1:-1, 1:-1].abs().max() < 1e-12
    quad = (x ** 2 + 2.0 * y ** 2 + 0.0 * t).expand(1, 2, 3, 16, 12)
    lap = heur.heuristic_maps(quad)["lap"]
    assert torch.allclose(lap[..., 1:-1, 1:-1], torch.full_like(lap[..., 1:-1, 1:-1], 6.0))


def test_dt_uses_backward_difference_and_forward_at_t0():
    t, x, y = _field()
    f = (t ** 2 + 0.0 * x + 0.0 * y).expand(1, 1, 3, 16, 12)  # 0, 1, 4
    dt = heur.heuristic_maps(f)["dt"][0, 0, :, 0, 0]
    assert dt.tolist() == [1.0, 1.0, 3.0]
    # Без оси времени dt не считается.
    assert "dt" not in heur.heuristic_maps(torch.randn(1, 2, 8, 8))


def test_top_share():
    err2 = torch.tensor([[1.0] * 10, [0.0] * 9 + [5.0], [0.0] * 10], dtype=torch.float64)
    s = heur.top_share(err2, 0.1)
    assert s[0] == pytest.approx(0.1) and s[1] == pytest.approx(1.0)
    assert math.isnan(float(s[2]))


def test_ause_zero_when_error_follows_heuristic():
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(1, 2, 4, 16, 16, dtype=torch.float64, generator=g)
    grad = heur.heuristic_maps(pred)["grad"]
    target = pred + grad  # |ошибка| = grad -> порядок совпадает с оракулом
    m = heur.sample_metrics(pred, target)
    assert m["ause_grad"].shape == (2, 4)
    assert m["ause_grad"].abs().max() < 1e-9
    # Случайная ошибка, не связанная с полем: AUSE около 1.
    noise = torch.randn(pred.shape, dtype=torch.float64, generator=g)
    m2 = heur.sample_metrics(pred, pred + noise)
    assert 0.6 < float(m2["ause_grad"].mean()) < 1.4


def test_aggregate_keys_and_means():
    g = torch.Generator().manual_seed(1)
    samples = []
    for _ in range(3):
        pred = torch.randn(1, 2, 3, 8, 8, dtype=torch.float64, generator=g)
        target = pred + 0.1 * torch.randn(pred.shape, dtype=torch.float64, generator=g)
        samples.append(heur.sample_metrics(pred, target))
    out = heur.aggregate(samples, ["rho", "u"])
    assert out["samples"] == 3
    for h in ("grad", "lap", "dt"):
        assert f"ause_{h}_rho" in out and f"ause_{h}_u_t02" in out
    assert "nrmse_rho" in out and "nrmse_u_t01" in out and "err_top10_rho" in out
    expected = torch.stack([s["nrmse_ch"] for s in samples]).mean(0)
    assert out["nrmse_u"] == pytest.approx(float(expected[1]))
    per_t = [out[f"ause_grad_rho_t{t:02d}"] for t in range(3)]
    assert out["ause_grad_rho"] == pytest.approx(sum(per_t) / 3)

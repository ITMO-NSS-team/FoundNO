from __future__ import annotations

import math

import pytest
import torch

from metrics_uq import normalized_ause


def _errors(g=3, n=4000, seed=0):
    gen = torch.Generator().manual_seed(seed)
    scale = torch.rand(g, n, generator=gen, dtype=torch.float64) * 3 + 0.1
    err = scale * torch.randn(g, n, generator=gen, dtype=torch.float64)
    return scale, err ** 2, gen


def test_oracle_ordering_gives_zero():
    _, err2, _ = _errors()
    assert torch.allclose(normalized_ause(err2.sqrt(), err2), torch.zeros(3, dtype=torch.float64), atol=1e-12)


def test_random_ordering_is_about_one():
    _, err2, gen = _errors()
    std = torch.rand(err2.shape, generator=gen, dtype=torch.float64)
    r = normalized_ause(std, err2)
    assert bool(((r - 1.0).abs() < 0.15).all()), r


def test_informative_std_between_zero_and_one():
    """std = истинный масштаб шума: лучше случайного, но хуже оракула."""
    scale, err2, _ = _errors()
    r = normalized_ause(scale, err2)
    assert bool(((r > 0.05) & (r < 0.8)).all()), r


def test_reversed_ordering_is_worse_than_random():
    _, err2, _ = _errors()
    assert bool((normalized_ause(-err2, err2) > 1.5).all())


def test_scale_invariant_and_handles_zero_std():
    scale, err2, _ = _errors()
    base = normalized_ause(scale, err2)
    assert torch.allclose(normalized_ause(1e-12 * scale, err2), base)
    # Нулевой std у части пикселей -- просто последние в очереди на удаление.
    std = scale.clone()
    std[:, :100] = 0.0
    assert bool(torch.isfinite(normalized_ause(std, err2)).all())


def test_degenerate_groups_are_nan():
    _, err2, _ = _errors(g=2)
    std = torch.ones_like(err2)
    std[0, 5] = float("nan")
    r = normalized_ause(std, err2)
    assert math.isnan(r[0]) and math.isfinite(r[1])
    # Все ошибки одинаковы -- оракул совпадает со случайным, метрика не определена.
    assert math.isnan(normalized_ause(torch.rand(1, 50), torch.ones(1, 50))[0])


def test_evaluate_luno_reports_channel_time_table(tiny, monkeypatch):
    """evaluate_luno: по ключу на каждую пару (канал, t) плюс среднее по t."""
    run = pytest.importorskip("run_luno_ggn_calibrate")
    from .test_pipeline_smoke import _fake_pipeline
    import argparse
    import types

    monkeypatch.setattr(run, "DTYPE", torch.float64)
    norm = types.SimpleNamespace(
        mean=torch.zeros(1, 2, 1, 1, 1, dtype=torch.float64),
        std=torch.full((1, 2, 1, 1, 1), 2.0, dtype=torch.float64), eps=1e-6,
    )
    out_normalizer = types.SimpleNamespace(normalizers=[norm])
    dp = types.SimpleNamespace(
        preprocess=lambda sample, training=False, batched=True: sample,
        device=torch.device("cpu"), out_normalizer=out_normalizer,
    )
    args = argparse.Namespace(vjp_chunk=8)
    from luno_torch.factory import create_luno_cov
    from luno_torch.ggn import GGNMatvec, LowRankTerms, randomized_eigh

    xs = tiny.xs(2)
    U, S = randomized_eigh(GGNMatvec(tiny.model_fn, tiny.w0, xs, fwd_batch=4),
                           n=tiny.w0.numel(), rank=3, oversample=3)
    loader = _fake_pipeline(tiny, 2, seed=5)
    metrics = run.evaluate_luno(
        args, loader, tiny.model_fn, tiny.w0, LowRankTerms(U=U, S=S),
        {"prior_prec": torch.tensor(1.0)}, tiny.wrapper, dp, out_normalizer,
        "val", 2, affine=None, fwd_batch=8, save_images=False, diag_probes=16,
        channel_names=["rho", "p"],
    )
    n_t = loader[0][0]["y"].shape[2]
    for ch in ("rho", "p"):
        per_t = [metrics[f"luno_ause_{ch}_t{t:02d}"] for t in range(n_t)]
        assert all(math.isfinite(v) for v in per_t)
        assert metrics[f"luno_ause_{ch}"] == pytest.approx(sum(per_t) / n_t)
    assert sum(k.startswith("luno_ause_") for k in metrics) == 2 * n_t + 2
    # sqrt(chi2) по (канал, t): по ключу на каждую пару и уровень.
    for q in run.PER_CHANNEL_PCT:
        for ch in ("rho", "p"):
            assert all(f"luno_sqrt_chi2_{ch}_t{t:02d}_p{q}" in metrics for t in range(n_t))
    n_ct = sum(1 for k in metrics if k.startswith("luno_sqrt_chi2_") and "_t" in k[15:])
    assert n_ct == 2 * n_t * len(run.PER_CHANNEL_PCT)
    # Только поканальные метрики: ничего, что усредняет по каналам.
    for gone in ("luno_nll", "luno_rmse", "luno_chi2", "luno_spearman_std_resid",
                 "luno_valid_percent", "luno_dropped_elements", "luno_sqrt_chi2_mean"):
        assert gone not in metrics
    assert not any("chmean" in k for k in metrics)
    import re
    assert not any(re.fullmatch(r"luno_sqrt_chi2_p\d+", k) for k in metrics)
    for ch in ("rho", "p"):
        for base in ("nll", "rmse", "std_mean", "valid_percent", "dropped_elements"):
            assert f"luno_{base}_{ch}" in metrics
        assert math.isfinite(metrics[f"luno_nll_{ch}"])
        assert metrics[f"luno_std_mean_{ch}"] > 0
        for q in run.PER_CHANNEL_PCT:
            assert f"luno_sqrt_chi2_{ch}_p{q}" in metrics
    # ENCE по (канал, t) + среднее по t; кривая надёжности по бинам на канал.
    for ch in ("rho", "p"):
        per_t = [metrics[f"luno_ence_{ch}_t{t:02d}"] for t in range(n_t)]
        assert all(math.isfinite(v) and v >= 0 for v in per_t)
        assert metrics[f"luno_ence_{ch}"] == pytest.approx(sum(per_t) / n_t)
        for t in range(n_t):
            pre = f"luno_reliability_{ch}_t{t:02d}"
            for b in range(run.ENCE_BINS):
                for part in ("logratio", "rmv", "rmse"):
                    assert math.isfinite(metrics[f"{pre}_b{b:02d}_{part}"])
            rmv = [metrics[f"{pre}_b{b:02d}_rmv"] for b in range(run.ENCE_BINS)]
            assert rmv == sorted(rmv)  # бины от уверенных к неуверенным
            # ENCE этой пары -- средний модуль кривой.
            curve = [metrics[f"{pre}_b{b:02d}_logratio"] for b in range(run.ENCE_BINS)]
            assert metrics[f"luno_ence_{ch}_t{t:02d}"] == pytest.approx(
                sum(abs(v) for v in curve) / len(curve))
    n_rel = sum(1 for k in metrics if k.startswith("luno_reliability_"))
    assert n_rel == 2 * n_t * run.ENCE_BINS * 3

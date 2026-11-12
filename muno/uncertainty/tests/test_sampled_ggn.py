"""Полная GGN лифтинга сэмплированием выхода + калибровка с температурой."""
from __future__ import annotations

import argparse
import types

import pytest
import torch

from luno_torch.factory import create_luno_cov
from luno_torch.ggn import LowRankTerms, auto_ggn_mc, sampled_full_ggn
from luno_torch.jacobian import LiftingWeightJacobian, _var_of_congruence_hutchinson

from .conftest import close, dense_ggn, dense_jacobian


def test_auto_ggn_mc_gives_full_rank():
    assert auto_ggn_mc(29640, 1000, fwd_batch=2) == 34  # 1000 * 34 >= 1.1 d
    assert auto_ggn_mc(29400, 28, fwd_batch=2) * 28 >= 1.1 * 29400
    assert auto_ggn_mc(100, 1000, fwd_batch=4) == 8  # минимум


def test_sampled_ggn_matches_dense(tiny):
    lw = tiny.lift_wrapper
    xs = tiny.xs(3)
    G = dense_ggn(lw.model_fn, xs, lw.w0).detach()
    terms = sampled_full_ggn(lw.model_fn, lw.w0, xs, LiftingWeightJacobian,
                             n_mc=6000, fwd_batch=500, seed=0)
    d = lw.w0.numel()
    assert terms.U.shape == (d, d) and terms.S.shape == (d,)
    assert close(terms.U.T @ terms.U, torch.eye(d, dtype=G.dtype), tol=1e-6)
    G_hat = (terms.U * terms.S) @ terms.U.T
    # Ошибка Монте-Карло ~ 1/sqrt(3 * 6000) по Фробениусу.
    rel = torch.linalg.matrix_norm(G_hat - G) / torch.linalg.matrix_norm(G)
    assert rel < 0.05, rel
    ev = torch.linalg.eigvalsh(G).flip(0)
    assert close(terms.S[:3], ev[:3], tol=0.05)


def test_full_basis_cov_is_exact_and_temperature_scales():
    torch.manual_seed(0)
    d = 30
    X = torch.randn(d, d, dtype=torch.float64)
    G = X @ X.T
    S, U = torch.linalg.eigh(G)
    terms = LowRankTerms(U=U.flip(1), S=S.flip(0))
    exact = torch.linalg.inv(0.7 * torch.eye(d, dtype=G.dtype) + G)
    cov = create_luno_cov(terms, {"prior_prec": 0.7})
    assert close(cov.todense(), exact, tol=1e-9)
    cov_t = create_luno_cov(terms, {"prior_prec": 0.7, "temperature": 3.0})
    assert close(cov_t.todense(), 3.0 * exact, tol=1e-9)
    assert cov_t.U.data_ptr() == terms.U.data_ptr()  # U не копируется


def test_hutchinson_with_given_z_is_deterministic(tiny):
    lw = tiny.lift_wrapper
    x = tiny.xs(1)[0]
    terms = sampled_full_ggn(lw.model_fn, lw.w0, tiny.xs(2), LiftingWeightJacobian,
                             n_mc=200, fwd_batch=50)
    cov = create_luno_cov(terms, {"prior_prec": 1.0})
    jac = LiftingWeightJacobian(lw.model_fn, x, lw.w0, fwd_batch=4)
    Z = torch.empty(lw.w0.numel(), 6, dtype=lw.w0.dtype).bernoulli_(0.5).mul_(2).sub_(1)
    Z0 = Z.clone()
    a = _var_of_congruence_hutchinson(jac, cov, 6, Z=Z)
    b = _var_of_congruence_hutchinson(jac, cov, 6, Z=Z)
    assert torch.equal(a, b) and torch.equal(Z, Z0)  # Z не портится
    # и сходится к точной дисперсии
    J = dense_jacobian(lw.model_fn, x, lw.w0).detach()
    exact = ((J @ cov.todense()) * J).sum(dim=1)
    est = _var_of_congruence_hutchinson(jac, cov, 4000)
    assert ((est - exact).abs() / exact).median() < 0.05


def test_lifting_pipeline_end_to_end(tiny, monkeypatch, tmp_path):
    run = pytest.importorskip("run_luno_ggn_calibrate")
    from .test_pipeline_smoke import _fake_pipeline

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
    args = argparse.Namespace(
        ggn_target="lifting", jv_mode="forward", prefix_cache=True,
        ggn_method="randomized", max_rank=10, ggn_mc=0, max_num_samples=4,
        vjp_chunk=8, diag_hutchinson=32, calib_max_samples=0,
        calib_grid_min=-3.0, calib_grid_max=6.0, calib_grid_size=10, fwd_batch=8,
        ggn_checkpoint=None,
    )
    setup = run.build_ggn_setup(args, tiny.core, None, None, tiny.lifting, tiny.projection)
    assert setup.full_sampled
    pipeline = {"train_loader": _fake_pipeline(tiny, 4, seed=1)}
    terms, source, _ = run.resolve_low_rank(
        args, setup.w0, setup.model_fn, pipeline, dp, None, 0, tmp_path, "run", {},
        matvec_cls=setup.matvec_cls, artifact_prefix=setup.artifact_prefix,
        full_sampled=True, jacobian_cls=setup.jacobian_cls,
    )
    d = setup.w0.numel()
    assert source == "computed" and terms.U.shape == (d, d)
    assert run.method_from_terms(setup.target, terms) == "lifting"

    calib = _fake_pipeline(tiny, 3, seed=5)
    prior = run.calibrate_prior_prec(
        args, setup.model_fn, setup.w0, terms, setup.wrapper, dp, out_normalizer,
        calib, None, objective_name="nll", fwd_batch=args.fwd_batch,
        jacobian_cls=setup.jacobian_cls, temperature=True,
    )
    assert prior["temperature"] > 0
    # После температуры средний chi2 на calib ~ 1 (по построению T* = mean chi2;
    # здесь -- со свежим шумом Хатчинсона, поэтому допуск широкий).
    cov = create_luno_cov(terms, prior)
    chi2 = []
    for b in calib:
        x, y = b[0]["x"], b[0]["y"]
        with torch.no_grad():
            mean, std = run._batch_predictive_stats(
                setup.model_fn, setup.w0, cov, x, out_normalizer, 2,
                affine=None, fwd_batch=8, diag_probes=64,
                jacobian_cls=setup.jacobian_cls, out_shape=y.shape,
            )
        chi2.append((((mean - y.reshape(-1)) / std) ** 2).mean())
    assert 0.5 < float(torch.stack(chi2).mean()) < 2.0

    # chi2-objective с температурой -- понятная ошибка.
    with pytest.raises(SystemExit, match="nll"):
        run.calibrate_prior_prec(
            args, setup.model_fn, setup.w0, terms, setup.wrapper, dp, out_normalizer,
            calib, None, objective_name="chi2", fwd_batch=8,
            jacobian_cls=setup.jacobian_cls, temperature=True,
        )

    # Низкоранговый чекпойнт для лифтинга игнорируется, без права счёта -- None.
    import pickle
    ck = tmp_path / "old_low_rank.pkl"
    pickle.dump(LowRankTerms(U=terms.U[:, :3].clone(), S=terms.S[:3].clone()), open(ck, "wb"))
    args.ggn_checkpoint = str(ck)
    t2, src2, _ = run.resolve_low_rank(
        args, setup.w0, setup.model_fn, pipeline, dp, None, 0, tmp_path, "run2", {},
        artifact_prefix=setup.artifact_prefix, full_sampled=True,
        jacobian_cls=setup.jacobian_cls, allow_compute=False,
    )
    assert t2 is None and src2 == "missing"

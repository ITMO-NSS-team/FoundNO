"""Дымовой тест проводки run_luno_ggn_calibrate: GGN -> калибровка -> std.

Данных нет: синтетическая модель, фиктивные загрузчик / data_processor /
нормализатор. Проверяется, что три пути (вся сеть, prefix-cache, KFAC) проходят
через одни и те же функции скрипта и дают согласованный результат.
"""
from __future__ import annotations

import argparse
import types

import pytest
import torch

run = pytest.importorskip("run_luno_ggn_calibrate")


class _Loader(list):
    """Список батчей {task: {"x", "y"}} с .dataset, как у multidict-лоадера."""

    @property
    def dataset(self):
        return range(sum(b[0]["x"].shape[0] for b in self))


def _fake_pipeline(tiny, n, seed):
    xs = tiny.xs(n, seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    batches = _Loader()
    for x in xs:
        y = tiny.model_fn(x, tiny.w0).detach()
        batches.append({0: {"x": x, "y": y + 0.05 * torch.randn(y.shape, dtype=y.dtype, generator=g)}})
    return batches


@pytest.fixture()
def env(tiny, monkeypatch):
    monkeypatch.setattr(run, "DTYPE", torch.float64)
    norm = types.SimpleNamespace(
        mean=torch.zeros(1, 2, 1, 1, 1, dtype=torch.float64),
        std=torch.full((1, 2, 1, 1, 1), 2.0, dtype=torch.float64), eps=1e-6,
    )
    out_normalizer = types.SimpleNamespace(normalizers=[norm])
    data_processor = types.SimpleNamespace(
        preprocess=lambda sample, training=False, batched=True: sample,
        device=torch.device("cpu"),
        out_normalizer=out_normalizer,
    )
    args = argparse.Namespace(
        ggn_target="last-block", jv_mode="forward", prefix_cache=True,
        ggn_method="randomized", ggn_oversample=4, ggn_power_iter=2, max_rank=4,
        max_num_samples=3, ggn_sample_batch=1, vjp_chunk=8, kfac_mc=4,
        diag_hutchinson=64, calib_max_samples=0, calib_grid_min=-2.0,
        calib_grid_max=2.0, calib_grid_size=7, fwd_batch=16,
    )
    return args, data_processor, out_normalizer


def _run(tiny, env, **overrides):
    args, data_processor, out_normalizer = env
    args = argparse.Namespace(**{**vars(args), **overrides})
    setup = run.build_ggn_setup(
        args, tiny.core, tiny.wrapper, tiny.w0, tiny.lifting, tiny.projection
    )
    train, calib = _fake_pipeline(tiny, 3, seed=1), _fake_pipeline(tiny, 2, seed=5)
    if setup.kfac is not None:
        terms = run.compute_kfac_terms(args, setup.kfac, train, data_processor)
    else:
        terms = run.compute_low_rank_ggn(
            args, setup.model_fn, setup.w0, train, data_processor, setup.affine,
            fwd_batch=args.fwd_batch, matvec_cls=setup.matvec_cls, bind=setup.bind,
        )
    torch.manual_seed(0)
    prior = run.calibrate_prior_prec(
        args, setup.model_fn, setup.w0, terms, setup.wrapper, data_processor,
        out_normalizer, calib, setup.affine, objective_name="nll",
        fwd_batch=args.fwd_batch, jacobian_cls=setup.jacobian_cls,
        bind=setup.bind, kfac=setup.kfac,
    )
    x, y = calib[0][0]["x"], calib[0][0]["y"]
    cov = ((terms, prior["prior_prec"]) if setup.kfac is not None
           else run.create_luno_cov(terms, prior))
    torch.manual_seed(1)
    with torch.no_grad():
        mean, std = run._batch_predictive_stats(
            setup.model_fn, setup.w0, cov, x, out_normalizer,
            setup.wrapper.num_output_channels, affine=setup.affine,
            fwd_batch=args.fwd_batch, diag_probes=args.diag_hutchinson,
            jacobian_cls=setup.jacobian_cls, bind=setup.bind,
            out_shape=y.shape, kfac=setup.kfac,
        )
    return setup, terms, prior, mean, std


def test_prefix_cache_does_not_change_results(tiny, env):
    s_full, t_full, p_full, mean_full, std_full = _run(tiny, env, prefix_cache=False)
    s_pref, t_pref, p_pref, mean_pref, std_pref = _run(tiny, env, prefix_cache=True)
    assert s_full.bind is None and s_pref.bind is not None
    assert torch.allclose(t_full.S, t_pref.S, rtol=1e-4)
    assert float(p_full["prior_prec"]) == float(p_pref["prior_prec"])
    assert torch.allclose(mean_full, mean_pref, rtol=1e-5, atol=1e-7)
    # Один и тот же Z (тот же seed) -> та же оценка std.
    assert torch.allclose(std_full, std_pref, rtol=1e-3)


def test_kfac_path_runs_end_to_end(tiny, env):
    setup, terms, prior, mean, std = _run(tiny, env, ggn_method="kfac")
    assert setup.kfac is not None and isinstance(terms, run.KFACTerms)
    assert torch.isfinite(std).all() and bool((std > 0).all())
    _, _, _, mean_lr, std_lr = _run(tiny, env)
    assert torch.allclose(mean, mean_lr, rtol=1e-5, atol=1e-7)
    # Другая аппроксимация: совпадения нет, но после калибровки масштаб тот же.
    ratio = float(std.mean() / std_lr.mean())
    assert 0.2 < ratio < 5.0


def test_kfac_rejects_lifting_target(tiny, env):
    args = argparse.Namespace(**{**vars(env[0]), "ggn_method": "kfac", "ggn_target": "lifting"})
    with pytest.raises(ValueError, match="kfac"):
        run.build_ggn_setup(args, tiny.core, tiny.wrapper, tiny.w0, tiny.lifting, tiny.projection)


def test_method_labels():
    ns = lambda **kw: argparse.Namespace(**{"ggn_target": "lifting", "ggn_method": "randomized",
                                            "max_rank": 25, **kw})
    assert run.uq_method(ns()) == "lifting"
    assert run.uq_method(ns(max_rank=0)) == "lifting_iso"
    assert run.uq_method(ns(ggn_target="last-block")) == "lastblock_lowrank"
    assert run.uq_method(ns(ggn_target="last-block", ggn_method="kfac")) == "lastblock_kfac"

    # Метку задают in_distr-задачи; out_distr наследует их метод.
    plan = lambda in_distr, method: types.SimpleNamespace(in_distr=in_distr, method=method)
    assert run.run_method_label([plan(True, "lifting"), plan(False, "lifting")]) == "lifting"
    assert run.run_method_label(
        [plan(True, "lifting_iso"), plan(True, "lifting"), plan(False, "x")]
    ) == "lifting_iso+lifting"
    # Без in_distr -- по источникам out_distr.
    assert run.run_method_label([plan(False, "lastblock_kfac")]) == "lastblock_kfac"


def _plan_args(**kw):
    base = dict(ggn_target="lifting", jv_mode="forward", ggn_method="randomized",
                max_rank=10, ggn_checkpoint=None, prior_prec_source=None,
                diag_hutchinson=16, fwd_batch=2, vjp_chunk=16)
    return argparse.Namespace(**{**base, **kw})


def _results(path, **kw):
    import pickle
    path.mkdir(parents=True, exist_ok=True)
    res = {"task_idx": 0, "in_distr": True, "prior_prec": 2.0, "ggn_target": "lifting",
           "temperature": 1.5, "calib_objective": "nll", **kw}
    pickle.dump(res, open(path / "luno_results.pkl", "wb"))
    return path


def test_plan_tasks_config_errors():
    cfgs = [
        (0, {"name": "no_type", "task_idx": 0}),
        (1, {"name": "bad_key", "task_idx": 0, "task_type": "out_distr",
             "luno": {"ggn_target": "lifting", "calib_grid_size": 5}}),
        (2, {"name": "id_pps", "task_idx": 1, "task_type": "in_distr",
             "luno": {"prior_prec_source": "x"}}),
        (3, {"name": "orphan", "task_idx": 7, "task_type": "out_distr"}),
    ]
    with pytest.raises(SystemExit) as exc:
        run.plan_tasks(_plan_args(), cfgs)
    msg = str(exc.value)
    for needle in ("no_type", "luno.ggn_target", "id_pps", "orphan"):
        assert needle in msg


def test_plan_tasks_two_in_distr_same_idx():
    cfgs = [(0, {"name": "a", "task_idx": 0, "task_type": "in_distr"}),
            (1, {"name": "b", "task_idx": 0, "task_type": "in_distr"})]
    with pytest.raises(SystemExit, match="несколько in_distr"):
        run.plan_tasks(_plan_args(), cfgs)


def test_plan_tasks_out_distr_from_run():
    cfgs = [(0, {"name": "a", "task_idx": 0, "task_type": "in_distr",
                 "luno": {"ggn_target": "last-block", "ggn_method": "kfac"}}),
            (1, {"name": "b", "task_idx": 0, "task_type": "out_distr",
                 "luno": {"fwd_batch": 1}})]
    plans = run.plan_tasks(_plan_args(), cfgs)
    b = plans[1]
    assert b.source == "run" and b.peer == "a" and b.method == "lastblock_kfac"
    assert b.args.ggn_target == "last-block" and b.args.ggn_method == "kfac"
    assert b.args.fwd_batch == 1


def test_plan_tasks_out_distr_from_files(tmp_path):
    import pickle
    d = 5
    lr = run.LowRankTerms(U=torch.zeros(d, d), S=torch.ones(d), scalar=0.0)
    ck = tmp_path / "id" / "run_lifting_low_rank_terms.pkl"
    _results(tmp_path / "id", uq_method="lifting", ggn_rank=d)
    pickle.dump(lr, open(ck, "wb"))
    cfg = {"name": "b", "task_idx": 0, "task_type": "out_distr",
           "luno": {"prior_prec_source": str(tmp_path / "id"),
                    "ggn_checkpoint": str(tmp_path / "id")}}
    # Вместе с in_distr прогона: она приоритетнее явных путей.
    plans = run.plan_tasks(_plan_args(), [
        (0, {"name": "a", "task_idx": 0, "task_type": "in_distr"}), (1, cfg)])
    assert plans[1].source == "run" and plans[1].peer == "a"
    (b,) = run.plan_tasks(_plan_args(), [(1, cfg)])
    assert b.source == "files" and b.method == "lifting" and b.ggn_path == str(ck)
    assert b.args.ggn_target == "lifting" and b.calib["temperature"] == 1.5

    # Без ggn_checkpoint это Iso -- а калибровка сделана с GGN: ошибка.
    cfg_iso = {**cfg, "luno": {"prior_prec_source": str(tmp_path / "id")}}
    with pytest.raises(SystemExit, match="Iso"):
        run.plan_tasks(_plan_args(), [(0, cfg_iso)])
    _results(tmp_path / "iso", uq_method="lifting_iso", ggn_rank=0, temperature=None)
    cfg_iso["luno"] = {"prior_prec_source": str(tmp_path / "iso")}
    (p,) = run.plan_tasks(_plan_args(), [(0, cfg_iso)])
    assert p.method == "lifting_iso" and p.terms is None

    # Чекпойнт без калибровки -- ошибка; чужой ранг -- ошибка.
    with pytest.raises(SystemExit, match="prior_prec_source"):
        run.plan_tasks(_plan_args(), [(0, {**cfg, "luno": {"ggn_checkpoint": str(ck)}})])
    _results(tmp_path / "id", uq_method="lifting_lowrank", ggn_rank=3)
    with pytest.raises(SystemExit, match="одной in_distr"):
        run.plan_tasks(_plan_args(), [(0, cfg)])


def test_iso_runs_end_to_end(tiny, env, tmp_path):
    args, data_processor, out_normalizer = env
    args = argparse.Namespace(**{**vars(args), "max_rank": 0, "ggn_checkpoint": "missing.pkl"})
    setup = run.build_ggn_setup(
        args, tiny.core, tiny.wrapper, tiny.w0, tiny.lifting, tiny.projection
    )
    terms, source, ckpt = run.resolve_low_rank(
        args, setup.w0, setup.model_fn, {}, data_processor, setup.affine, 0,
        tmp_path, "run", {}, matvec_cls=setup.matvec_cls,
        artifact_prefix=setup.artifact_prefix, bind=setup.bind, iso=True,
    )
    assert source == "iso" and ckpt is None and terms.U.shape == (tiny.w0.numel(), 0)
    assert run.method_from_terms(setup.target, terms) == "lastblock_iso"

    calib = _fake_pipeline(tiny, 2, seed=5)
    prior = run.calibrate_prior_prec(
        args, setup.model_fn, setup.w0, terms, setup.wrapper, data_processor,
        out_normalizer, calib, setup.affine, objective_name="nll",
        fwd_batch=args.fwd_batch, jacobian_cls=setup.jacobian_cls, bind=setup.bind,
    )
    x, y = calib[0][0]["x"], calib[0][0]["y"]
    cov = run.create_luno_cov(terms, prior)
    with torch.no_grad():
        _, std = run._batch_predictive_stats(
            setup.model_fn, setup.w0, cov, x, out_normalizer,
            setup.wrapper.num_output_channels, affine=setup.affine,
            fwd_batch=args.fwd_batch, diag_probes=args.diag_hutchinson,
            jacobian_cls=setup.jacobian_cls, bind=setup.bind, out_shape=y.shape,
        )
    assert torch.isfinite(std).all() and bool((std > 0).all())

    # Артефакт Iso читается обратно тем же путём, что и low-rank чекпойнт.
    back = run.load_low_rank_ggn(tmp_path / "run_low_rank_terms.pkl", setup.w0)
    assert back.U.shape == (tiny.w0.numel(), 0)


def test_ggn_target_is_per_task_key():
    base = argparse.Namespace(ggn_target="last-block", ggn_method="randomized", max_rank=25)
    a = run.apply_task_overrides(base, {"name": "t", "luno": {"ggn_target": "lifting"}})
    assert a.ggn_target == "lifting" and base.ggn_target == "last-block"
    with pytest.raises(ValueError, match="ggn_target"):
        run.apply_task_overrides(base, {"name": "t", "luno": {"ggn_target": "core"}})


def test_shipped_config_is_valid():
    """luno_turbo_inf.yaml: все luno-ключи допустимы, метод прогона -- лифтинг low-rank."""
    import yaml
    from pathlib import Path

    cfg_path = Path(run.__file__).resolve().parents[2] / "experiments/scripts/configs/luno_turbo_inf.yaml"
    tasks = yaml.safe_load(cfg_path.read_text())["tasks"]
    args = argparse.Namespace(ggn_target="last-block", ggn_method="randomized", max_rank=10)
    for t in tasks:
        a = run.apply_task_overrides(args, t)
        assert a.ggn_target == "lifting", t["name"]
    assert run.run_method_label(args, list(enumerate(tasks))) == "lifting"

"""Замер стоимости GGN-матвека и предиктивной дисперсии на синтетическом Muno.

Данные и чекпойнты не нужны: сеть собирается со случайными весами, важны только
её размеры. Печатает время и число forward-проходов ядра -- по ним видно, куда
уходит работа (полный forward ядра против одного блока).

    python muno/uncertainty/bench_luno.py --hidden 120 --modes 16 32 32 \\
        --grid 16 128 128 --layers 4 --columns 8 --fwd-batch 2 --samples 2

Скрипт запускается и на прежних версиях luno_torch (без ``bind``), поэтому
сравнение "до/после" -- это два запуска на двух коммитах.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

UNCERTAINTY_DIR = Path(__file__).resolve().parent
for _p in (str(UNCERTAINTY_DIR.parents[1]), str(UNCERTAINTY_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from luno_torch import affine as affine_mod  # noqa: E402
from luno_torch.factory import create_luno_cov  # noqa: E402
from luno_torch.ggn import GGNMatvec, LiftingGGNMatvec, LowRankTerms  # noqa: E402
from luno_torch.jacobian import (  # noqa: E402
    LastFNOBlockWeightJacobian,
    LiftingWeightJacobian,
    var_of_congruence,
)
from luno_torch.progress import set_enabled  # noqa: E402
from luno_torch.synthetic import SyntheticMuno  # noqa: E402


def _timed(core, fn):
    """(секунды, число forward ядра) для ``fn()``; отладочные print'ы глушатся."""
    calls = [0]
    h = core.register_forward_hook(lambda *a: calls.__setitem__(0, calls[0] + 1))
    cuda = next(core.parameters()).is_cuda
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
    finally:
        h.remove()
    return dt, calls[0]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--modes", type=int, nargs="+", default=[8, 16, 16])
    p.add_argument("--grid", type=int, nargs="+", default=[8, 32, 32])
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--columns", type=int, default=8, help="q = rank + oversample")
    p.add_argument("--fwd-batch", type=int, default=2)
    p.add_argument("--probes", type=int, default=8, help="пробы Хатчинсона на eval")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--kfac-mc", type=int, default=8)
    p.add_argument("--repeat", type=int, default=2, help="берётся лучший из запусков")
    args = p.parse_args()
    set_enabled(False)

    with contextlib.redirect_stdout(io.StringIO()):
        net = SyntheticMuno(
            in_channels=args.channels, out_channels=args.channels,
            hidden_channels=args.hidden, n_modes=tuple(args.modes),
            n_layers=args.layers, grid=tuple(args.grid),
            adapter_hidden=args.hidden, dtype=torch.float32, device=args.device,
        )
    xs = net.xs(args.samples)
    lw = net.lift_wrapper
    print(f"device={args.device} torch={torch.__version__} grid={tuple(args.grid)} "
          f"hidden={args.hidden} modes={tuple(args.modes)} layers={args.layers}")
    print(f"d(last-block)={net.w0.numel():,}  d(lifting)={lw.w0.numel():,}  "
          f"samples={args.samples} columns={args.columns} fwd_batch={args.fwd_batch}")

    suffix_cls = getattr(affine_mod, "LastBlockSuffix", None)
    variants = [("last-block, вся сеть", GGNMatvec, net.model_fn, net.w0, None,
                 LastFNOBlockWeightJacobian)]
    if suffix_cls is not None:
        bind = suffix_cls(net.affine, net.model_fn, net.w0).bind
        variants.append(("last-block, prefix-cache", GGNMatvec, net.model_fn, net.w0,
                         bind, LastFNOBlockWeightJacobian))
    variants.append(("lifting", LiftingGGNMatvec, lw.model_fn, lw.w0, None,
                     LiftingWeightJacobian))

    def best(fn):
        runs = [_timed(net.core, fn) for _ in range(args.repeat)]
        return min(r[0] for r in runs), runs[-1][1]

    print(f"\n{'вариант':28s} {'GGN-матвек, с':>14s} {'forward ядра':>13s} "
          f"{'eval-сэмпл, с':>14s} {'forward ядра':>13s}")
    for name, mv_cls, fn, w0, bind, jac_cls in variants:
        kw = {} if bind is None else {"bind": bind}
        mv = mv_cls(fn, w0, xs, fwd_batch=args.fwd_batch, **kw)
        g = torch.Generator(device=w0.device).manual_seed(0)
        V = torch.randn(w0.numel(), args.columns, dtype=w0.dtype, device=w0.device,
                        generator=g)
        t_mv, n_mv = best(lambda: mv._matmul(V))

        # Предиктивная дисперсия одного eval-сэмпла (Хатчинсон, как на val/test).
        U = torch.linalg.qr(torch.randn(w0.numel(), args.rank, dtype=w0.dtype,
                                        device=w0.device, generator=g))[0]
        S = torch.logspace(2, 0, args.rank, dtype=w0.dtype, device=w0.device)
        cov = create_luno_cov(LowRankTerms(U=U, S=S), {"prior_prec": 1.0})
        x = xs[0].to(w0.device)

        def one_eval():
            f, state = bind(x) if bind is not None else (fn, x)
            jac = jac_cls(f, state, w0, fwd_batch=args.fwd_batch)
            _ = jac.fx
            var_of_congruence(jac, cov, diag_probes=args.probes)
            jac._release_vjp()

        t_ev, n_ev = best(one_eval)
        print(f"{name:28s} {t_mv:14.3f} {n_mv:13d} {t_ev:14.3f} {n_ev:13d}")
        del mv, V, U, cov

    # KFAC по модам: один проход вместо (power_iter + 1) матвеков на q колонок.
    try:
        from luno_torch.kfac import KFACLastBlock
    except ImportError:
        return
    ctx = KFACLastBlock(net.affine, net.model_fn, net.w0)
    terms = {}
    t_fit, n_fit = best(lambda: terms.__setitem__("t", ctx.fit(xs, n_mc=args.kfac_mc)))
    x = xs[0].to(net.w0.device)
    t_ev, n_ev = best(lambda: ctx.mean_and_var(
        x, terms["t"], [1.0], n_probe=args.probes, chunk=args.fwd_batch))
    print(f"{'last-block, KFAC (весь fit)':28s} {t_fit:14.3f} {n_fit:13d} "
          f"{t_ev:14.3f} {n_ev:13d}")
    print(f"\nKFAC fit -- это ВСЯ кривизна (n_mc={args.kfac_mc}); у low-rank пути "
          f"один матвек из таблицы умножается на (power_iter + 1) проходов.")


if __name__ == "__main__":
    main()

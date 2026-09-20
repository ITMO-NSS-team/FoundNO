# -*- coding: utf-8 -*-
"""
bench_adapter.py — почему эпоха с temporal-сканом стоит 571 с, а с flat 197 с.

Замеряет на РЕАЛЬНЫХ размерах (B, 60, 21, 134, 113) время forward+backward
каждого варианта адаптера и, главное, раскладывает его на две части:

  ssm      — сам селективный скан Mamba;
  обвязка  — permute/contiguous (две транспозиции тензора на 7.6М токенов)
             и LayerNorm вокруг него.

Гипотеза, которую скрипт проверяет: стоимость скана пропорциональна
d_inner * d_state на токен, токенов у обоих сканов поровну (B*T*H*W), а
d_state у temporal вчетверо меньше (16 против 60) — значит temporal-слой
обязан быть ДЕШЕВЛЕ. Если он не дешевле, время съедает обвязка, и тогда
дорого именно число слоёв, а не ось скана.

Запуск (в Colab, где стоит mamba_ssm):
  python bench_adapter.py --tnav-pkg /content/tnav_fno --foundno-repo /content/FoundNO
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


def timed(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / n


class Identity(nn.Module):
    def forward(self, x):
        return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tnav-pkg", default="/content/tnav_fno")
    p.add_argument("--foundno-repo", default="/content/FoundNO")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--H", type=int, default=134)
    p.add_argument("--W", type=int, default=113)
    p.add_argument("--width", type=int, default=60)
    p.add_argument("--steps-per-epoch", type=int, default=448,
                   help="для пересчёта в секунды/эпоху (1792 кейса / batch)")
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.tnav_pkg).resolve()))
    sys.path.insert(0, str(Path(args.foundno_repo).resolve()))
    sys.path.insert(0, str((Path(args.foundno_repo) / "experiments/scripts").resolve()))
    from finetune_tnav_adapters import _make_ssm_block

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, H, W, C = args.batch, args.T, args.H, args.W, args.width
    tokens = B * T * H * W
    print(f"Устройство: {dev} | тензор [{B}, {C}, {T}, {H}, {W}] = "
          f"{tokens / 1e6:.2f}М токенов\n")

    def bench_block(blk, label):
        blk = blk.to(dev)
        x = torch.randn(B, C, T, H, W, device=dev, requires_grad=True)

        def step():
            y = blk(x)
            y.float().pow(2).mean().backward()
            x.grad = None
            blk.zero_grad(set_to_none=True)

        try:
            t_full = timed(step)
        except RuntimeError as e:
            print(f"  {label:<34} OOM/ошибка: {str(e)[:60]}")
            return None
        # та же обвязка, но скан заменён на identity
        saved = blk.ssm
        blk.ssm = Identity()
        t_shell = timed(step)
        blk.ssm = saved
        print(f"  {label:<34} всего {1000 * t_full:7.1f} мс  "
              f"(скан {1000 * (t_full - t_shell):6.1f} + обвязка "
              f"{1000 * t_shell:6.1f})  => {t_full * args.steps_per_epoch:5.0f} с/эпоху")
        return t_full

    print("Один SSM-блок, forward+backward:")
    res = {}
    for scan, ds, dc, lab in [
        ("temporal", 16, 4, "temporal (d_state=16, как было)"),
        ("temporal", 60, 4, "temporal (d_state=60, честно)"),
        ("spatial", 16, 4, "spatial  (d_state=16)"),
        ("flat", 0, 0, "flat/repo (d_state=60, d_conv=2)"),
    ]:
        try:
            blk = _make_ssm_block(scan, C, False, ds, dc)
        except Exception as e:
            print(f"  {lab}: не создан ({e})")
            continue
        res[lab] = bench_block(blk, lab)

    vals = {k: v for k, v in res.items() if v}
    if len(vals) > 1:
        base = min(vals.values())
        print("\nОтносительно самого дешёвого варианта:")
        for k, v in sorted(vals.items(), key=lambda kv: kv[1]):
            print(f"  {k:<34} x{v / base:.2f}")
        print("\nЕсли 'обвязка' сопоставима со 'сканом' — дорого число слоёв,")
        print("а не ось; тогда 3 слоя temporal ≈ 3x, что и наблюдалось.")


if __name__ == "__main__":
    main()

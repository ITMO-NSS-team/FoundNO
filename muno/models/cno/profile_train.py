# -*- coding: utf-8 -*-
"""
profile_train.py — куда уходит время эпохи.

Оптимизировать вслепую бессмысленно: 613 с на эпоху могут быть и GPU, и
чтением h5. Скрипт разбирает время по слоям:

  1. ДАТАЛОАДЕР без модели — сколько стоит просто прочитать батчи.
     Если это близко к времени эпохи, узкое место в I/O, и ускорять
     архитектуру бесполезно.
  2. FORWARD и BACKWARD на GPU по отдельности.
  3. ВКЛАД ВЕТОК: одна и та же модель прогоняется с выключенными
     по очереди CNO-веткой и диф. ядрами (через обнуление alpha
     достаточно для скорости? нет — ветка всё равно считается, поэтому
     здесь модель пересобирается без неё).
  4. Проверка режимов ускорения: channels_last_3d и bf16-autocast.
     ВНИМАНИЕ: FFT в половинной точности может менять численику, так что
     bf16 — это замер потенциала, а не рекомендация включать вслепую.

Запуск:
  python profile_train.py --data $H5_V2 --tnav-pkg /content/tnav_fno \\
      --physics-channels --fault-channels \\
      --local-branch parallel --diff-kernels parallel --batch-size 4
"""

import argparse
import sys
import time
from pathlib import Path

import torch


def timed(fn, n=6, warmup=2):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--tnav-pkg', default='/content/tnav_fno')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--modes', default='12,48,48')
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--arch', default='spatial')
    p.add_argument('--local-branch', default='parallel')
    p.add_argument('--cno-levels', type=int, default=2)
    p.add_argument('--cno-blocks', type=int, default=2)
    p.add_argument('--diff-kernels', default='parallel')
    p.add_argument('--physics-channels', action='store_true')
    p.add_argument('--fault-channels', action='store_true')
    p.add_argument('--hfrac-channels', action='store_true')
    p.add_argument('--n-batches', type=int, default=12,
                   help='сколько батчей мерить в даталоадере')
    p.add_argument('--limit-cases', type=int, default=256)
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.tnav_pkg).resolve()))
    from torch.utils.data import DataLoader
    from tnav_data import (list_cases, fit_stats, TnavDataset, TnavCollate,
                           in_channels, C_OUT)
    from tnav_model import build_model

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    names = [n for _, n in list_cases(args.data)][:args.limit_cases]
    print(f"Кейсов для замера: {len(names)}, устройство: {dev}\n")
    norm = fit_stats(args.data, names, cache_path=None,
                     fault=args.fault_channels, hfrac=args.hfrac_channels,
                     physics=args.physics_channels)
    ds = TnavDataset(args.data, names, norm)
    T, H, W = ds.T, ds.H, ds.W
    N_IN = in_channels(False, args.fault_channels, args.hfrac_channels,
                       args.physics_channels)

    # ---------- 1. даталоадер ----------
    print(f"{'=' * 74}\n1. ДАТАЛОАДЕР (без модели)\n{'=' * 74}")
    for nw in (0, args.num_workers):
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=nw, pin_memory=True, drop_last=True,
                        persistent_workers=nw > 0,
                        collate_fn=TnavCollate(T, train=True))
        it = iter(dl)
        next(it)                                   # прогрев/старт воркеров
        t0 = time.time()
        k = 0
        for _ in range(args.n_batches):
            try:
                next(it)
                k += 1
            except StopIteration:
                break
        dt = (time.time() - t0) / max(k, 1)
        print(f"  num_workers={nw:<2} {dt*1000:8.1f} мс/батч  "
              f"=> {dt * (2256/args.batch_size):6.0f} с/эпоху на 2256 кейсах")
        del dl, it

    # ---------- 2. модель ----------
    def make(local_branch, diff_kernels):
        cfg = dict(arch=args.arch, modes=tuple(int(x) for x in args.modes.split(',')),
                   width=args.width, n_layers=args.n_layers, rank=0.10,
                   domain_padding=0.15, temporal_ssm='auto', d_state=16,
                   d_conv=4, dropout=0.1, residual_u0=True, global_token='off',
                   local_branch=local_branch, cno_levels=args.cno_levels,
                   cno_blocks=args.cno_blocks, cno_act_up=1,
                   diff_kernels=diff_kernels, diff_kernel_size=3,
                   diff_padding='replicate')
        return build_model(cfg, N_IN, C_OUT).to(dev)

    x = torch.randn(args.batch_size, N_IN, T, H, W, device=dev)
    x[:, 2] = 1.0

    print(f"\n{'=' * 74}\n2. МОДЕЛЬ: forward / backward, вклад веток\n{'=' * 74}")
    variants = [
        (args.local_branch, args.diff_kernels, 'полная конфигурация'),
        ('off', args.diff_kernels, 'без CNO-ветки'),
        (args.local_branch, 'off', 'без диф. ядер'),
        ('off', 'off', 'только FNO+SSM (база)'),
    ]
    steps = 2256 / args.batch_size
    for lb, dk, lab in variants:
        try:
            m = make(lb, dk)
        except Exception as e:
            print(f"  {lab:<24} не собрана: {e}")
            continue
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

        def fwd():
            with torch.no_grad():
                m(x)

        def step():
            loss = m(x).float().pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        try:
            tf = timed(fwd, n=4)
            ts = timed(step, n=4)
        except RuntimeError as e:
            print(f"  {lab:<24} OOM/ошибка: {str(e)[:60]}")
            del m, opt
            torch.cuda.empty_cache()
            continue
        mem = torch.cuda.max_memory_allocated() / 1e9 if dev.type == 'cuda' else 0
        print(f"  {lab:<24} fwd {tf*1000:7.0f} мс | fwd+bwd {ts*1000:7.0f} мс "
              f"=> {ts*steps:6.0f} с/эпоху | пик {mem:.1f} ГБ")
        del m, opt
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # ---------- 3. режимы ускорения ----------
    print(f"\n{'=' * 74}\n3. РЕЖИМЫ УСКОРЕНИЯ (полная конфигурация)\n{'=' * 74}")
    m = make(args.local_branch, args.diff_kernels)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    def step_plain():
        loss = m(x).float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    base = timed(step_plain, n=4)
    print(f"  fp32 (как сейчас)          {base*1000:7.0f} мс => {base*steps:6.0f} с/эпоху")

    try:
        if x.dim() != 5:
            raise RuntimeError('channels_last_3d требует 5D тензор')
        xc = x.contiguous(memory_format=torch.channels_last_3d)
        mc = m.to(memory_format=torch.channels_last_3d)

        def step_cl():
            loss = mc(xc).float().pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        t = timed(step_cl, n=4)
        print(f"  channels_last_3d           {t*1000:7.0f} мс => {t*steps:6.0f} с/эпоху "
              f"(x{base/t:.2f})")
        m = m.to(memory_format=torch.contiguous_format)
    except Exception as e:
        print(f"  channels_last_3d: {type(e).__name__}: {str(e)[:70]}")

    try:
        def step_amp():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = m(x)
            loss = out.float().pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        t = timed(step_amp, n=4)
        print(f"  bf16 autocast              {t*1000:7.0f} мс => {t*steps:6.0f} с/эпоху "
              f"(x{base/t:.2f})")
        print("    ВНИМАНИЕ: спектральная часть считает FFT — перед включением "
              "сверьте метрики на короткой прогонке")
    except Exception as e:
        print(f"  bf16 autocast: {type(e).__name__}: {str(e)[:70]}")

    print(f"\n{'=' * 74}\nКАК ЧИТАТЬ\n{'=' * 74}")
    print("  Если время даталоадера сопоставимо с fwd+bwd — узкое место I/O:")
    print("  поднимайте --num-workers, кэшируйте статику (уже сделано), либо")
    print("  держите датасет на локальном диске, а не на смонтированном Drive.")
    print("  Если доминирует fwd+bwd — смотрите вклад веток и режимы ускорения.")


if __name__ == '__main__':
    main()

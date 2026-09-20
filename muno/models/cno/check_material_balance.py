# -*- coding: utf-8 -*-
"""
check_material_balance.py — проверка признаков материального баланса ДО обучения.

Обе модели (supervised и FoundNO) систематически мажут АМПЛИТУДУ изменения
давления по кейсу. Гипотеза: эта амплитуда определяется накопленным отбором,
делённым на поровый объём, и её можно подать входным каналом. Скрипт проверяет
гипотезу количественно, не тратя прогон обучения.

Считает по каждому кейсу (только из inputs/* и targets/time_days — никаких
diagnostics, никакой утечки):
  cum_rate   = Σ_t (producer_orat - injector_rate) * dt   / суммарный поровый объём
  cum_drive  = Σ_t k*h*(p0 - bhp)*open * dt               / суммарный поровый объём
               (движущая сила по Дарси — работает там, где скважина на BHP,
                и канал дебита не заполнен)
и сравнивает их с ФАКТИЧЕСКОЙ амплитудой: mean(p0) - mean(p(T)) по маске.

Печатает корреляции (Пирсона и Спирмена), качество линейной подгонки R²,
и долю well-ячеек, у которых при открытой скважине дебит нулевой (признак
BHP-управления — там cum_rate слеп).

Решающий критерий: если |r| для cum_rate или cum_drive выше ~0.8, признак
объясняет ту самую величину, которую модели угадывают с ошибкой в 1.5-2 раза,
и его стоит добавлять. Если корреляция низкая — признак не поможет,
и прогон обучения сэкономлен.

Запуск:
  python check_material_balance.py --data $LOCAL_H5 --tnav-pkg /content/tnav_fno \\
      --limit 200
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def r2_linear(x, y):
    """R² простой линейной регрессии y ~ a*x + b."""
    A = np.stack([x, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / max(ss_tot, 1e-12)), coef


def case_features(f, name, read_static_raw, read_dynamic_raw):
    mask, static, p0, _ = read_static_raw(f, name)
    ctrl, y, tdays = read_dynamic_raw(f, name)      # ctrl [T,8,H,W], y [T,2,H,W]

    m = mask.astype(np.float32)
    permx = 10.0 ** static[1]                        # static[1] = log10(permx)
    thick = static[3]
    pv = static[5]
    pv_tot = float((pv * m).sum()) + 1e-8

    dt = np.diff(tdays, prepend=tdays[0] - (tdays[1] - tdays[0] if len(tdays) > 1 else 1.0))
    dt = np.maximum(dt, 0.0).astype(np.float32)      # [T]

    prod_open, inj_open = ctrl[:, 0], ctrl[:, 1]
    prod_bhp, prod_orat = ctrl[:, 2], ctrl[:, 3]
    inj_rate, inj_bhp = ctrl[:, 4], ctrl[:, 5]

    # --- накопленный чистый отбор (из расписания) ---
    net_rate = prod_orat - inj_rate                              # [T,H,W]
    cum_rate = float((net_rate * dt[:, None, None] * m).sum()) / pv_tot

    # --- накопленная движущая сила по Дарси (для BHP-режима) ---
    kh = permx * thick                                            # [H,W]
    drive = (prod_open * np.maximum(p0[None] - prod_bhp, 0.0)
             - inj_open * np.maximum(inj_bhp - p0[None], 0.0))    # [T,H,W]
    cum_drive = float((drive * kh[None] * dt[:, None, None] * m).sum()) / pv_tot

    # --- фактическая амплитуда: сколько пласт сработал ---
    mb = mask.astype(bool)
    amp = float(p0[mb].mean() - y[-1, 0][mb].mean())

    # --- доля открытых well-ячеек с нулевым дебитом (BHP-управление) ---
    open_any = (prod_open > 0.5) | (inj_open > 0.5)
    rate_any = (np.abs(prod_orat) > 1e-8) | (np.abs(inj_rate) > 1e-8)
    n_open = float(open_any.sum())
    frac_bhp = float((open_any & ~rate_any).sum()) / max(n_open, 1.0)

    return dict(case=name, cum_rate=cum_rate, cum_drive=cum_drive, amp=amp,
                frac_bhp=frac_bhp, n_open=n_open,
                p_final=float(y[-1, 0][mb].mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tnav-pkg", default="/content/tnav_fno")
    p.add_argument("--limit", type=int, default=200,
                   help="сколько кейсов проверить (0 = все)")
    p.add_argument("--cases", default="",
                   help="конкретные CaseID через запятую/диапазоны — например "
                        "проблемные: 531,533,554,564,566")
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.tnav_pkg).resolve()))
    from tnav_data import (list_cases, parse_ranges, read_static_raw,
                           read_dynamic_raw)

    cases = list_cases(args.data)
    if args.cases:
        want = parse_ranges(args.cases)
        cases = [(i, n) for i, n in cases if i in want]
    if args.limit:
        cases = cases[:args.limit]
    print(f"Проверяю {len(cases)} кейсов\n")

    rows = []
    with h5py.File(args.data, "r") as f:
        for k, (_, name) in enumerate(cases):
            rows.append(case_features(f, name, read_static_raw, read_dynamic_raw))
            if (k + 1) % 50 == 0 or k + 1 == len(cases):
                print(f"  ...{k + 1}/{len(cases)}", flush=True)

    amp = np.array([r["amp"] for r in rows])
    cr = np.array([r["cum_rate"] for r in rows])
    cd = np.array([r["cum_drive"] for r in rows])
    fb = np.array([r["frac_bhp"] for r in rows])

    print(f"\n{'=' * 78}\nФАКТИЧЕСКАЯ АМПЛИТУДА (p0 - p_final, среднее по маске)\n{'=' * 78}")
    print(f"  mean {amp.mean():7.2f} бар | med {np.median(amp):7.2f} | "
          f"диапазон [{amp.min():.1f}, {amp.max():.1f}] | std {amp.std():.2f}")
    print("  (именно эту величину обе модели угадывают с ошибкой в 1.5-2 раза)")

    print(f"\n{'=' * 78}\nИНФОРМАТИВНОСТЬ ПРИЗНАКОВ\n{'=' * 78}")
    for v, label in [(cr, "cum_rate  (накопленный отбор / поровый объём)"),
                     (cd, "cum_drive (Дарси-драйв k*h*dP*dt / поровый объём)")]:
        if v.std() < 1e-12:
            print(f"  {label}: КОНСТАНТА — канал не заполнен в датасете!")
            continue
        r_p = float(np.corrcoef(v, amp)[0, 1])
        r_s = spearman(v, amp)
        r2, coef = r2_linear(v, amp)
        print(f"  {label}")
        print(f"     Пирсон r = {r_p:+.3f} | Спирмен r = {r_s:+.3f} | "
              f"R² линейной подгонки = {r2:.3f}")
        print(f"     амплитуда ≈ {coef[0]:.4g} * признак + {coef[1]:.2f}")

    if cr.std() > 1e-12 and cd.std() > 1e-12:
        A = np.stack([cr, cd, np.ones_like(cr)], 1)
        coef, *_ = np.linalg.lstsq(A, amp, rcond=None)
        pred = A @ coef
        r2 = 1 - ((amp - pred) ** 2).sum() / max(((amp - amp.mean()) ** 2).sum(), 1e-12)
        print(f"\n  ОБА признака вместе: R² = {r2:.3f}  "
              f"(остаточная σ = {(amp - pred).std():.2f} бар)")

    print(f"\n{'=' * 78}\nРЕЖИМ УПРАВЛЕНИЯ\n{'=' * 78}")
    print(f"  доля открытых well-ячеек с нулевым дебитом в расписании "
          f"(= BHP-режим): mean {100 * fb.mean():.1f}%  med {100 * np.median(fb):.1f}%")
    if fb.mean() > 0.5:
        print("  ⚠ большинство скважин на BHP — cum_rate там слеп, основную "
              "нагрузку несёт cum_drive")
    hi = fb > np.percentile(fb, 75)
    if hi.any() and (~hi).any():
        print(f"  |амплитуда| у BHP-тяжёлых кейсов: {np.abs(amp[hi]).mean():.1f} бар, "
              f"у остальных: {np.abs(amp[~hi]).mean():.1f} бар")

    print(f"\n{'=' * 78}\nЭКСТРЕМАЛЬНЫЕ КЕЙСЫ (по |амплитуде|)\n{'=' * 78}")
    print(f"  {'case':<12}{'амплитуда':>11}{'p_final':>10}{'cum_rate':>12}"
          f"{'cum_drive':>12}{'BHP-доля':>10}")
    for r in sorted(rows, key=lambda r: -abs(r["amp"]))[:10]:
        print(f"  {r['case']:<12}{r['amp']:11.1f}{r['p_final']:10.1f}"
              f"{r['cum_rate']:12.4g}{r['cum_drive']:12.4g}{100*r['frac_bhp']:9.0f}%")


if __name__ == "__main__":
    main()

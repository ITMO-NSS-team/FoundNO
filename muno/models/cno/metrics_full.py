# -*- coding: utf-8 -*-
"""
metrics_full.py — полная оценка обученной модели по датасету v6.

Считает МЕТРИКИ ПО КАЖДОМУ КЕЙСУ трёх сплитов (train / geology_holdout /
controls_holdout) и агрегирует их по распределению (mean / std / median /
p10 / p90 / min / max). Два уровня метрик:

  FIELD (по активной маске пласта, все T):
    - pressure/swat  relL1  = mean_t mean_cells |err| / (|mean_cells target_t| + eps)
    - pressure/swat  MAE
    (relL1 совпадает по нормировке с train/eval — числа сопоставимы)

  WELLS (по ячейкам открытых completions, t >= t_start; как в газовом eval):
    - MAPE                     = mean 100 * |err| / (|target| + 1e-4)
    - L1 depression-normed     = mean |err| / (depression + 1e-4)
    - MAPE depression-normed   = mean 100 * |err| / (mean depression + 1e-4)
    depression(i,j) = среднее давление 4 соседей (как compute_depression).
    Well-метрики считаются ПО ДАВЛЕНИЮ (компонента 0).

Вывод — только в консоль. Per-case значения держатся в памяти на время
прохода (нужны для median/percentiles), на диск ничего не пишется.

Запуск (Colab):
  python metrics_full.py --data $LOCAL_H5 \\
      --ckpt /content/drive/MyDrive/tnav_ablation/spatial/best_field.pt \\
      --val-geology 512-639 --val-controls 1920-2047

  # быстрая проверка на подвыборке каждого сплита:
  python metrics_full.py --data $LOCAL_H5 --ckpt ... --limit 32

Модель (temporal/spatial/…) и нормализатор берутся из чекпойнта —
архитектуру указывать не нужно.
"""

import os
import time
import argparse

import numpy as np
import torch

from tnav_data import (list_cases, parse_ranges, predict_case, TARGET_NAMES,
                       C_OUT)
from eval_tnav import load_checkpoint


# ============================================================================
# МЕТРИКИ ПО ОДНОМУ КЕЙСУ
# ============================================================================

def _depression(pressure_thw, i, j):
    """pressure_thw: [t, H, W] тензор. Среднее 4 соседей — как compute_depression
    из газового eval-скрипта (клэмп индексов на границах)."""
    H, W = pressure_thw.shape[-2:]
    iu, idn = min(i + 1, H - 1), max(i - 1, 0)
    jr, jl = min(j + 1, W - 1), max(j - 1, 0)
    return 0.25 * (pressure_thw[:, i, jr] + pressure_thw[:, i, jl] +
                   pressure_thw[:, iu, j] + pressure_thw[:, idn, j])


def case_metrics_full(res):
    """
    res — выход predict_case: pred/target [T,2,H,W], mask [H,W], well [T,H,W].
    Возвращает:
      field: {pressure_relL1, swat_relL1, pressure_mae, swat_mae}
      wells: list[dict] по каждой открытой ячейке-скважине (МAPE и depression-normed)
    """
    pred = res['pred']            # [T,2,H,W] физ.
    tgt = res['target']
    mask = res['mask'].bool()
    well = res['well']            # [T,H,W] открытые completions
    T = pred.shape[0]

    # ---------- FIELD ----------
    field = {}
    for c, name in enumerate(TARGET_NAMES):
        dp = (pred[:, c] - tgt[:, c]).abs()               # [T,H,W]
        rel = []
        for t in range(T):
            mt = tgt[t, c][mask].mean().abs() + 1e-8
            rel.append(float(dp[t][mask].mean() / mt))
        field[f'{name}_relL1'] = float(np.mean(rel))
        field[f'{name}_mae'] = float(dp.permute(1, 2, 0)[mask].mean())

    # ---------- WELLS (по давлению, комп. 0) ----------
    pred_p = pred[:, 0]           # [T,H,W]
    tgt_p = tgt[:, 0]
    wells = []
    ever = well.any(dim=0)        # [H,W] — ячейка когда-либо открыта
    cells = torch.nonzero(ever, as_tuple=False)
    for idx in cells:
        i, j = int(idx[0]), int(idx[1])
        active_t = torch.nonzero(well[:, i, j], as_tuple=False)
        if len(active_t) == 0:
            continue
        t0 = int(active_t[0])
        pr = pred_p[t0:, i, j]
        tg = tgt_p[t0:, i, j]
        if len(pr) == 0:
            continue
        err = (pr - tg).abs()

        mape = float((100.0 * err / (tg.abs() + 1e-4)).mean())
        depr = _depression(tgt_p[t0:], i, j)
        l1_depr = float((err / (depr + 1e-4)).mean())
        mape_depr = float((100.0 * err / (depr.mean() + 1e-4)).mean())
        wells.append({'mape': mape, 'l1_depr': l1_depr, 'mape_depr': mape_depr,
                      't_start': t0, 'i': i, 'j': j})

    return field, wells


# ============================================================================
# АГРЕГАЦИЯ ПО РАСПРЕДЕЛЕНИЮ
# ============================================================================

def agg(vals):
    """mean/std/median/p10/p90/min/max по списку значений (с фильтром NaN)."""
    a = np.asarray([v for v in vals if v == v], dtype=np.float64)  # v==v отсеивает NaN
    if a.size == 0:
        return None
    return {
        'mean': float(a.mean()), 'std': float(a.std()),
        'median': float(np.median(a)),
        'p10': float(np.percentile(a, 10)), 'p90': float(np.percentile(a, 90)),
        'min': float(a.min()), 'max': float(a.max()), 'n': int(a.size),
    }


def _fmt(d, pct=False, prec=4):
    if d is None:
        return "  (нет данных)"
    u = '%' if pct else ''
    w = 8 if pct else prec + 4
    return (f"mean {d['mean']:.{prec}f}{u}  ±{d['std']:.{prec}f}  | "
            f"med {d['median']:.{prec}f}{u}  | "
            f"p10-p90 [{d['p10']:.{prec}f}, {d['p90']:.{prec}f}]{u}  | "
            f"max {d['max']:.{prec}f}{u}")


# ============================================================================
# ОЦЕНКА СПЛИТА
# ============================================================================

@torch.no_grad()
def evaluate_split(model, normalizer, h5_path, names, device, tag, limit=0):
    if limit:
        names = names[:limit]
    n = len(names)
    print(f"\n{'=' * 78}\n{tag}: {n} кейсов\n{'=' * 78}")
    if n == 0:
        print("  пусто — сплит не задан")
        return

    # per-case field-метрики
    pc = {f'{nm}_relL1': [] for nm in TARGET_NAMES}
    pc.update({f'{nm}_mae': [] for nm in TARGET_NAMES})
    # per-case средние well-метрики (усреднение по скважинам кейса)
    pc_well = {'mape': [], 'l1_depr': [], 'mape_depr': []}
    # per-well метрики (пул по всем скважинам всех кейсов сплита)
    pool_well = {'mape': [], 'l1_depr': [], 'mape_depr': []}
    n_wells_total = 0
    cases_with_wells = 0

    t0 = time.time()
    for k, name in enumerate(names):
        res = predict_case(model, normalizer, h5_path, name, device)
        field, wells = case_metrics_full(res)

        for key, v in field.items():
            pc[key].append(v)

        if wells:
            cases_with_wells += 1
            n_wells_total += len(wells)
            cm = {m: np.mean([w[m] for w in wells]) for m in pc_well}
            for m in pc_well:
                pc_well[m].append(float(cm[m]))
            for w in wells:
                for m in pool_well:
                    pool_well[m].append(w[m])

        if (k + 1) % 50 == 0 or k + 1 == n:
            el = time.time() - t0
            print(f"  ...{k + 1}/{n}  ({el:.0f}s, {el / (k + 1):.2f}s/case)")

    # ---------- вывод: FIELD ----------
    print(f"\n  ── FIELD (по маске пласта, распределение по {n} кейсам) ──")
    for nm in TARGET_NAMES:
        print(f"    {nm:>9s} relL1: {_fmt(agg(pc[f'{nm}_relL1']))}")
        print(f"    {nm:>9s}  MAE : {_fmt(agg(pc[f'{nm}_mae']))}")

    # ---------- вывод: WELLS ----------
    print(f"\n  ── WELLS (по давлению; {n_wells_total} скважин в "
          f"{cases_with_wells}/{n} кейсах) ──")
    print(f"    [per-case] усреднение по скважинам кейса, распределение по кейсам:")
    print(f"      MAPE              : {_fmt(agg(pc_well['mape']), pct=True)}")
    print(f"      L1 depression-norm: {_fmt(agg(pc_well['l1_depr']))}")
    print(f"      MAPE depr-norm    : {_fmt(agg(pc_well['mape_depr']), pct=True)}")
    print(f"    [per-well] распределение по всем {n_wells_total} скважинам:")
    print(f"      MAPE              : {_fmt(agg(pool_well['mape']), pct=True)}")
    print(f"      L1 depression-norm: {_fmt(agg(pool_well['l1_depr']))}")
    print(f"      MAPE depr-norm    : {_fmt(agg(pool_well['mape_depr']), pct=True)}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--val-geology', default='512-639')
    p.add_argument('--val-controls', default='1920-2047')
    p.add_argument('--limit', type=int, default=0,
                   help='ограничить каждый сплит N кейсами (быстрая проверка)')
    p.add_argument('--use-ema', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--splits', default='train,geology,controls',
                   help='какие сплиты считать (через запятую)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, normalizer, cfg = load_checkpoint(args.ckpt, device, args.use_ema)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Architecture: {cfg.get('arch', 'temporal')} | use_ema={args.use_ema} "
          f"| device={device}")

    cases = list_cases(args.data)
    geo_ids = parse_ranges(args.val_geology)
    ctrl_ids = parse_ranges(args.val_controls)
    val_ids = geo_ids | ctrl_ids

    split_names = {
        'train': [n for i, n in cases if i not in val_ids],
        'geology': [n for i, n in cases if i in geo_ids],
        'controls': [n for i, n in cases if i in ctrl_ids],
    }
    tags = {'train': 'TRAIN', 'geology': 'GEOLOGY_HOLDOUT (OOD)',
            'controls': 'CONTROLS_HOLDOUT (OOD)'}

    want = [s.strip() for s in args.splits.split(',')]
    for s in want:
        if s not in split_names:
            print(f"WARN: неизвестный сплит '{s}', пропуск"); continue
        evaluate_split(model, normalizer, args.data, split_names[s],
                       device, tags[s], limit=args.limit)

    print(f"\n{'=' * 78}\nГотово.\n{'=' * 78}")


if __name__ == '__main__':
    main()

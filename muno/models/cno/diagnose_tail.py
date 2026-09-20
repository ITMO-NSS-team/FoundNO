# -*- coding: utf-8 -*-
"""
diagnose_tail.py — разбор ХВОСТА ошибок по скважинам.

Средний MAPE ~12-14%, но распределение сильно скошено: медиана ~10%,
p90 ~24%, max 170-295%. Этот скрипт отвечает, ЧЕМ именно образован хвост,
до того как менять архитектуру:

  1) артефакт метрики — ячейка completion вне активной маски пласта
     (тогда target≈0 и MAPE = |pred|/1e-4 взлетает в сотни процентов);
  2) малый знаменатель — низкое |target| в конкретной ячейке;
  3) поздний запуск скважины (мало шагов, переходный режим);
  4) сильная депрессия / нагнетание — резкий прискважинный градиент,
     который спектральный оператор сглаживает;
  5) несколько «плохих» кейсов целиком (тогда дело в сценарии, не в скважинах).

Печатает: долю well-ячеек вне маски и их вклад, топ худших скважин,
разбивку producer/injector, вклад верхних процентилей в средний MAPE,
и связь MAPE с t_start / |target| / депрессией.

Запуск для FoundNO-модели:
  python diagnose_tail.py --data $LOCAL_H5 --split geology \\
      --ft-ckpt .../best_wells_adapters.pt --core-pt .../core.pt \\
      --foundno-repo /content/FoundNO --tnav-pkg /content/tnav_fno

Для supervised tnav-модели:
  python diagnose_tail.py --data $LOCAL_H5 --split geology \\
      --tnav-ckpt /content/drive/MyDrive/tnav_ablation/spatial/best_wells.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# ============================================================================
# СБОР PER-WELL ЗАПИСЕЙ
# ============================================================================

def collect_wells(model, norm_v6, h5_path, names, device, read_dynamic_raw,
                  read_static_raw, predict_case, eps=1e-4):
    import h5py
    recs = []
    per_case = []
    for k, name in enumerate(names):
        res = predict_case(model, norm_v6, h5_path, name, device)
        with h5py.File(h5_path, "r") as f:
            ctrl, _, _ = read_dynamic_raw(f, name)          # [T,8,H,W]
            _, _, p0_np, _ = read_static_raw(f, name)       # начальное давление
        ctrl = torch.from_numpy(ctrl)
        p0 = torch.from_numpy(p0_np)

        pred_p = res["pred"][:, 0]        # [T,H,W] давление
        tgt_p = res["target"][:, 0]
        mask = res["mask"].bool()
        well = res["well"]                # [T,H,W]
        H, W = mask.shape

        case_mapes = []
        for idx in torch.nonzero(well.any(dim=0), as_tuple=False):
            i, j = int(idx[0]), int(idx[1])
            act = torch.nonzero(well[:, i, j], as_tuple=False)
            if len(act) == 0:
                continue
            t0 = int(act[0])
            pr, tg = pred_p[t0:, i, j], tgt_p[t0:, i, j]
            err = (pr - tg).abs()
            mape = float((100.0 * err / (tg.abs() + eps)).mean())

            iu, idn = min(i + 1, H - 1), max(i - 1, 0)
            jr, jl = min(j + 1, W - 1), max(j - 1, 0)
            depr = 0.25 * (tgt_p[t0:, i, jr] + tgt_p[t0:, i, jl] +
                           tgt_p[t0:, iu, j] + tgt_p[t0:, idn, j])
            drawdown = float((depr - tg).mean())   # депрессия: соседи минус ячейка

            recs.append({
                "p0": float(p0[i, j]),
                "depl_true": float(p0[i, j] - tg.mean()),
                "depl_pred": float(p0[i, j] - pr.mean()),
                "case": name, "i": i, "j": j, "t_start": t0,
                "n_steps": int(len(pr)),
                "mape": mape,
                "in_mask": bool(mask[i, j]),
                "tgt_min": float(tg.abs().min()),
                "tgt_mean": float(tg.mean()),
                "pred_mean": float(pr.mean()),
                "drawdown": drawdown,
                "is_prod": bool((ctrl[t0:, 0, i, j] > 0.5).any()),
                "is_inj": bool((ctrl[t0:, 1, i, j] > 0.5).any()),
            })
            case_mapes.append(mape)
        if case_mapes:
            m_ = mask
            dp = (pred_p - tgt_p).abs()
            rel = [float(dp[t][m_].mean() / (tgt_p[t][m_].mean().abs() + 1e-8))
                   for t in range(dp.shape[0])]
            per_case.append((name, float(np.mean(case_mapes)), len(case_mapes),
                             float(np.mean(rel)),
                             float(tgt_p[:, m_].mean()),
                             float(p0[m_].mean())))
        if (k + 1) % 25 == 0 or k + 1 == len(names):
            print(f"  ...{k + 1}/{len(names)} кейсов, {len(recs)} скважин",
                  flush=True)
    return recs, per_case


# ============================================================================
# ОТЧЁТ
# ============================================================================

def report(recs, per_case, top_n=20):
    m = np.array([r["mape"] for r in recs])
    n = len(m)
    print(f"\n{'=' * 78}\nВСЕГО СКВАЖИН: {n}\n{'=' * 78}")
    print(f"MAPE: mean {m.mean():.2f}%  med {np.median(m):.2f}%  "
          f"p90 {np.percentile(m, 90):.2f}%  p99 {np.percentile(m, 99):.2f}%  "
          f"max {m.max():.2f}%")

    # --- 1. вне маски пласта ---
    out = np.array([not r["in_mask"] for r in recs])
    print(f"\n── 1. Completion вне активной маски ──")
    if out.any():
        print(f"  {out.sum()} из {n} ({100 * out.mean():.2f}%) — "
              f"их MAPE: mean {m[out].mean():.1f}%, max {m[out].max():.1f}%")
        print(f"  MAPE внутри маски: mean {m[~out].mean():.2f}%")
        print(f"  ⚠ well-маска не пересечена с маской пласта — это АРТЕФАКТ "
              f"метрики: target там ≈0, знаменатель 1e-4")
        delta = m.mean() - m[~out].mean()
        print(f"  вклад в средний MAPE: {delta:+.2f} п.п.")
    else:
        print("  нет — все completion внутри маски (артефакт исключён)")

    # --- 2. вклад хвоста ---
    print(f"\n── 2. Вклад хвоста в средний MAPE ──")
    s = np.sort(m)[::-1]
    for q in (1, 5, 10):
        k = max(1, int(n * q / 100))
        rest = s[k:].mean()
        print(f"  без топ-{q}% ({k} шт): mean {rest:.2f}%  "
              f"(было {m.mean():.2f}%, -{m.mean() - rest:.2f} п.п.); "
              f"эти {q}% дают {100 * s[:k].sum() / m.sum():.1f}% всей ошибки")

    # --- 3. связь с признаками ---
    print(f"\n── 3. С чем связан MAPE (корреляция Спирмена) ──")
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])
    for key, label in [("tgt_min", "|target| минимум в ячейке"),
                       ("tgt_mean", "target среднее"),
                       ("t_start", "шаг запуска скважины"),
                       ("n_steps", "длина активного окна"),
                       ("drawdown", "депрессия (соседи - ячейка)"),
                       ("p0", "начальное давление p0 в ячейке"),
                       ("depl_true", "истинная сработка p0 - target"),
                       ("depl_pred", "предсказанная сработка p0 - pred")]:
        v = np.array([r[key] for r in recs], dtype=float)
        if v.std() > 0:
            print(f"  {label:<32} r = {spearman(v, m):+.3f}")

    # --- 4. producer vs injector ---
    print(f"\n── 4. Тип скважины ──")
    for flag, label in [("is_prod", "producer"), ("is_inj", "injector")]:
        sel = np.array([r[flag] for r in recs])
        if sel.any():
            print(f"  {label:<10} {sel.sum():>6} шт | MAPE mean "
                  f"{m[sel].mean():6.2f}%  med {np.median(m[sel]):6.2f}%  "
                  f"max {m[sel].max():7.1f}%")

    # --- 5. концентрация по кейсам ---
    # --- 4b. сработка пласта: модель её недобирает? ---
    dt = np.array([r["depl_true"] for r in recs])
    dp_ = np.array([r["depl_pred"] for r in recs])
    print(f"\n── 4b. Сработка давления (p0 - p) в ячейках скважин ──")
    print(f"  истинная:      mean {dt.mean():8.2f} бар  med {np.median(dt):8.2f}")
    print(f"  предсказанная: mean {dp_.mean():8.2f} бар  med {np.median(dp_):8.2f}")
    frac = dp_.mean() / dt.mean() if abs(dt.mean()) > 1e-9 else float("nan")
    print(f"  модель воспроизводит {100 * frac:.1f}% сработки")
    worst = np.argsort(m)[::-1][:max(1, len(m) // 100)]
    print(f"  у топ-1% худших: истинная {dt[worst].mean():.1f}, "
          f"предсказанная {dp_[worst].mean():.1f} бар")

    print(f"\n── 5. Худшие кейсы ──")
    pc = sorted(per_case, key=lambda x: -x[1])
    print(f"  {'case':<12}{'MAPE%':>8}{'field relL1':>13}{'p_target':>10}"
          f"{'p0':>9}{'скважин':>9}")
    for name, val, cnt, fld, ptg, pini in pc[:8]:
        print(f"  {name:<12}{val:8.2f}{fld:13.4f}{ptg:10.1f}{pini:9.1f}{cnt:9d}")
    vals = np.array([p[1] for p in per_case])
    flds = np.array([p[3] for p in per_case])
    ptgs = np.array([p[4] for p in per_case])
    print(f"  медиана по кейсам: MAPE {np.median(vals):.2f}%  "
          f"field relL1 {np.median(flds):.4f}  p_target {np.median(ptgs):.1f}")
    if len(vals) > 2:
        print(f"  corr(well MAPE, field relL1) = {np.corrcoef(vals, flds)[0,1]:+.3f}"
              f"   corr(well MAPE, p_target) = {np.corrcoef(vals, ptgs)[0,1]:+.3f}")

    # --- 6. топ худших скважин ---
    print(f"\n── 6. Топ-{top_n} худших скважин ──")
    print(f"  {'MAPE%':>8} {'t0':>3} {'p0':>8} {'tgt_mean':>9} {'pred_mean':>10} "
          f"{'pred/tgt':>9} {'depl_true':>10} {'depl_pred':>10}  case")
    for r in sorted(recs, key=lambda r: -r["mape"])[:top_n]:
        ratio = r["pred_mean"] / r["tgt_mean"] if abs(r["tgt_mean"]) > 1e-9 else float("nan")
        print(f"  {r['mape']:8.1f} {r['t_start']:3d} {r['p0']:8.1f} "
              f"{r['tgt_mean']:9.2f} {r['pred_mean']:10.2f} {ratio:9.2f} "
              f"{r['depl_true']:10.1f} {r['depl_pred']:10.1f}  {r['case']}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="geology",
                   choices=["geology", "controls", "train", "all"])
    p.add_argument("--val-geology", default="512-639")
    p.add_argument("--val-controls", default="1920-2047")
    p.add_argument("--limit", type=int, default=64,
                   help="сколько кейсов разобрать (0 = все)")
    p.add_argument("--top-n", type=int, default=20)
    # источник модели: FoundNO или supervised tnav
    p.add_argument("--ft-ckpt", default="")
    p.add_argument("--core-pt", default="")
    p.add_argument("--foundno-repo", default="/content/FoundNO")
    p.add_argument("--tnav-pkg", default="/content/tnav_fno")
    p.add_argument("--norm-dir", default="")
    p.add_argument("--tnav-stats", default="")
    p.add_argument("--tnav-ckpt", default="")
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.tnav_pkg).resolve()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from tnav_data import (list_cases, parse_ranges, predict_case,
                           read_dynamic_raw, read_static_raw)

    if args.ft_ckpt:
        from eval_foundno import build
        model, norm_v6 = build(args, device)
    elif args.tnav_ckpt:
        from eval_tnav import load_checkpoint
        model, norm_v6, _ = load_checkpoint(args.tnav_ckpt, device)
    else:
        raise SystemExit("нужен --ft-ckpt (+ --core-pt) или --tnav-ckpt")

    cases = list_cases(args.data)
    geo, ctrl = parse_ranges(args.val_geology), parse_ranges(args.val_controls)
    pools = {
        "geology": [n for i, n in cases if i in geo],
        "controls": [n for i, n in cases if i in ctrl],
        "train": [n for i, n in cases if i not in (geo | ctrl)],
        "all": [n for _, n in cases],
    }
    names = pools[args.split]
    if args.limit:
        names = names[:args.limit]
    print(f"Сплит {args.split}: разбираю {len(names)} кейсов")

    recs, per_case = collect_wells(model, norm_v6, args.data, names, device,
                                   read_dynamic_raw, read_static_raw,
                                   predict_case)
    if not recs:
        raise SystemExit("скважин не найдено")
    report(recs, per_case, args.top_n)


if __name__ == "__main__":
    main()

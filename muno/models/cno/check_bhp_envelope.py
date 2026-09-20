# -*- coding: utf-8 -*-
"""
check_bhp_envelope.py — проверка гипотезы «огибающей забойных давлений».

Наблюдение, породившее гипотезу: в holdout у экстремальных кейсов p_final
принимает круглые значения (450.0, 450.0, 400.0, ~85, ~100), а признаки
материального баланса объясняют лишь R²=0.19 амплитуды. Похоже, что среднее
пластовое давление едет не к «объёму отобранного», а к взвешенному среднему
УСТАВОК активных скважин (producer_bhp / injector_bhp_limit) и упирается в них.

Скрипт считает несколько кандидатов-предикторов и ранжирует их по R² для
p_final (и для амплитуды p0 - p_final):

  env_kh        k*h-взвешенное среднее уставок всех открытых скважин
  env_flat      простое среднее уставок по открытым ячейкам (без k*h)
  env_kh_tavg   то же, что env_kh, но усреднённое по времени
  bhp_prod      k*h-взвешенное среднее только по добывающим
  bhp_inj       k*h-взвешенное среднее только по нагнетательным
  kh_frac_inj   доля k*h, приходящаяся на нагнетательные (баланс режимов)
  cum_rate      накопленный отбор / поровый объём      (базовая линия)
  cum_drive     накопленная дарси-сила / поровый объём (базовая линия)

Плюс проверка по времени: коррелирует ли p_mean(t) с env_kh(t) на всех
парах (кейс, шаг) — если да, огибающая описывает и динамику, а не только
финальную точку.

Использует только inputs/* (+ targets для оценки), никаких diagnostics.

Запуск:
  python check_bhp_envelope.py --data $LOCAL_H5 --cases 512-639
"""

import argparse
import re

import h5py
import numpy as np

_NUM = re.compile(r"(\d+)")


def parse_ranges(spec):
    ids = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            ids.update(range(int(a), int(b) + 1))
        else:
            ids.add(int(part))
    return ids


def list_cases(path):
    out = []
    with h5py.File(path, "r") as f:
        for name in f["cases"].keys():
            m = _NUM.search(name)
            if m:
                out.append((int(m.group(1)), name))
    out.sort(key=lambda x: x[0])
    return out


def case_features(f, name):
    g = f[f"cases/{name}"]
    mask = np.asarray(g["inputs/static/mask"][:]) > 0.5
    mf = mask.astype(np.float64)
    permx = np.asarray(g["inputs/static/permx"][:], np.float64) * mf
    thick = np.asarray(g["inputs/geometry/cell_thickness"][:], np.float64) * mf
    pv = np.asarray(g["inputs/geometry/pore_volume"][:], np.float64) * mf
    p0 = np.asarray(g["inputs/initial/pressure"][:], np.float64) * mf
    ctrl = np.asarray(g["inputs/controls/scheduled_grid_tensor"][:], np.float64)
    press = np.asarray(g["targets/pressure"][:], np.float64)
    tdays = np.asarray(g["targets/time_days"][:], np.float64)

    T = ctrl.shape[0]
    kh = (permx * thick)[None]                       # [1,H,W]
    op_p, op_i = ctrl[:, 0] > 0.5, ctrl[:, 1] > 0.5  # [T,H,W]
    bhp_p, bhp_i = ctrl[:, 2], ctrl[:, 5]

    w_p = kh * op_p * mf[None]
    w_i = kh * op_i * mf[None]
    sw_p = w_p.sum(axis=(1, 2))
    sw_i = w_i.sum(axis=(1, 2))
    sw = np.maximum(sw_p + sw_i, 1e-12)

    env_kh_t = ((w_p * bhp_p).sum(axis=(1, 2))
                + (w_i * bhp_i).sum(axis=(1, 2))) / sw          # [T]
    n_p = np.maximum(op_p.sum(axis=(1, 2)), 1e-12)
    n_i = np.maximum(op_i.sum(axis=(1, 2)), 1e-12)
    n_all = np.maximum(op_p.sum(axis=(1, 2)) + op_i.sum(axis=(1, 2)), 1e-12)
    env_flat_t = ((bhp_p * op_p).sum(axis=(1, 2))
                  + (bhp_i * op_i).sum(axis=(1, 2))) / n_all

    p_mean_t = (press * mf[None]).sum(axis=(1, 2)) / max(mf.sum(), 1e-12)
    p0_mean = float((p0 * mf).sum() / max(mf.sum(), 1e-12))

    # базовые линии — материальный баланс
    dt = np.zeros(T)
    if T > 1:
        dt[1:] = tdays[1:] - tdays[:-1]
        dt[0] = dt[1]
    else:
        dt[0] = 1.0
    dt = np.clip(dt, 0, None)[:, None, None]
    pv_tot = max(pv.sum(), 1e-12)
    net = (ctrl[:, 3] - ctrl[:, 4]) * dt * mf[None]
    cum_rate = np.cumsum(net.sum(axis=(1, 2))) / pv_tot
    drive = (op_p * np.clip(p0[None] - bhp_p, 0, None)
             - op_i * np.clip(bhp_i - p0[None], 0, None)) * kh * dt * mf[None]
    cum_drive = np.cumsum(drive.sum(axis=(1, 2))) / pv_tot

    active = (sw_p + sw_i) > 1e-9          # шаги с открытыми скважинами
    if active.any():
        env_last = float(env_kh_t[np.nonzero(active)[0][-1]])
        env_tavg = float(env_kh_t[active].mean())
        flat_last = float(env_flat_t[np.nonzero(active)[0][-1]])
    else:
        env_last = env_tavg = flat_last = np.nan

    return dict(
        name=name, p0=p0_mean, p_final=float(p_mean_t[-1]),
        env_kh_active=env_last, env_flat_active=flat_last,
        env_kh_tavg_active=env_tavg,
        d_to_inj=float(p_mean_t[-1]) - float((w_i * bhp_i).sum() / max(sw_i.sum(), 1e-12)),
        d_to_prod=float(p_mean_t[-1]) - float((w_p * bhp_p).sum() / max(sw_p.sum(), 1e-12)),
        amp=float(p0_mean - p_mean_t[-1]),
        env_kh=float(env_kh_t[-1]), env_flat=float(env_flat_t[-1]),
        env_kh_tavg=float(env_kh_t.mean()),
        bhp_prod=float((w_p * bhp_p).sum() / max(sw_p.sum(), 1e-12)),
        bhp_inj=float((w_i * bhp_i).sum() / max(sw_i.sum(), 1e-12)),
        kh_frac_inj=float(sw_i.sum() / max(sw_p.sum() + sw_i.sum(), 1e-12)),
        cum_rate=float(cum_rate[-1]), cum_drive=float(cum_drive[-1]),
        env_kh_t=env_kh_t, p_mean_t=p_mean_t,
    )


def r2_of(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) < 1e-12:
        return np.nan, np.nan, np.nan, np.nan
    r = float(np.corrcoef(x[ok], y[ok])[0, 1])
    a, b = np.polyfit(x[ok], y[ok], 1)
    resid = y[ok] - (a * x[ok] + b)
    return r, r ** 2, float(np.std(resid)), a


def multi_r2(cols, y):
    X = np.column_stack([np.asarray(c, float) for c in cols] + [np.ones(len(y))])
    y = np.asarray(y, float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if ok.sum() <= X.shape[1]:
        return np.nan, np.nan
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    resid = y[ok] - X[ok] @ beta
    ss_res, ss_tot = (resid ** 2).sum(), ((y[ok] - y[ok].mean()) ** 2).sum()
    return 1 - ss_res / max(ss_tot, 1e-12), float(np.std(resid))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--cases", default="", help="CaseID: диапазоны/через запятую")
    p.add_argument("--limit", type=int, default=200)
    args = p.parse_args()

    cases = list_cases(args.data)
    if args.cases:
        want = parse_ranges(args.cases)
        cases = [(i, n) for i, n in cases if i in want]
    elif args.limit:
        cases = cases[:args.limit]
    print(f"Проверяю {len(cases)} кейсов\n")

    rows = []
    with h5py.File(args.data, "r") as f:
        for k, (_, name) in enumerate(cases):
            rows.append(case_features(f, name))
            if (k + 1) % 50 == 0 or k + 1 == len(cases):
                print(f"  ...{k + 1}/{len(cases)}", flush=True)

    p_final = np.array([r["p_final"] for r in rows])
    amp = np.array([r["amp"] for r in rows])

    print(f"\n{'=' * 78}\nЦЕЛЕВЫЕ ВЕЛИЧИНЫ\n{'=' * 78}")
    print(f"  p_final: mean {p_final.mean():7.2f}  std {p_final.std():7.2f}  "
          f"диапазон [{p_final.min():.1f}, {p_final.max():.1f}]")
    print(f"  амплитуда p0-p_final: mean {amp.mean():7.2f}  std {amp.std():7.2f}")

    keys = ["env_kh_active", "env_flat_active", "env_kh_tavg_active",
            "env_kh_tavg", "bhp_prod", "bhp_inj",
            "kh_frac_inj", "cum_rate", "cum_drive"]
    print(f"\n{'=' * 78}\nПРЕДИКТОРЫ p_final (одиночные)\n{'=' * 78}")
    print(f"  {'признак':<14}{'r':>8}{'R²':>8}{'ост. σ, бар':>14}{'наклон':>12}")
    scored = []
    for key in keys:
        v = [r[key] for r in rows]
        r, r2, sd, slope = r2_of(v, p_final)
        scored.append((r2 if np.isfinite(r2) else -1, key, r, r2, sd, slope))
        print(f"  {key:<14}{r:>8.3f}{r2:>8.3f}{sd:>14.2f}{slope:>12.4g}")

    print(f"\n{'=' * 78}\nКОМБИНАЦИИ (p_final)\n{'=' * 78}")
    combos = [
        (["env_kh"], "огибающая"),
        (["env_kh", "kh_frac_inj"], "огибающая + доля k*h нагнетания"),
        (["bhp_prod", "bhp_inj", "kh_frac_inj"], "уставки раздельно + доля"),
        (["cum_rate", "cum_drive"], "матбаланс (базовая линия)"),
        (["env_kh", "kh_frac_inj", "cum_rate", "cum_drive"], "всё вместе"),
    ]
    for cols, lab in combos:
        r2, sd = multi_r2([[r[c] for r in rows] for c in cols], p_final)
        print(f"  {lab:<38} R² = {r2:6.3f}   ост. σ = {sd:6.2f} бар")

    # --- динамика: p_mean(t) против env(t) на всех парах (кейс, шаг) ---
    xs = np.concatenate([r["env_kh_t"] for r in rows])
    ys = np.concatenate([r["p_mean_t"] for r in rows])
    r, r2, sd, slope = r2_of(xs, ys)
    print(f"\n{'=' * 78}\nДИНАМИКА: p_mean(t) ~ env_kh(t), все пары (кейс, шаг)\n{'=' * 78}")
    print(f"  r = {r:+.3f}  R² = {r2:.3f}  ост. σ = {sd:.2f} бар  "
          f"({len(xs)} точек)")

    # --- НЕЛИНЕЙНАЯ информативность (кросс-валидация) ---
    print(f"\n{'=' * 78}\nНЕЛИНЕЙНАЯ ИНФОРМАТИВНОСТЬ (5-fold CV, градиентный бустинг)\n{'=' * 78}")
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_predict, KFold
        sets = [
            (["bhp_prod", "bhp_inj", "kh_frac_inj"], "уставки + доля k*h нагнетания"),
            (["cum_rate", "cum_drive"], "матбаланс (базовая линия)"),
            (["bhp_prod", "bhp_inj", "kh_frac_inj", "cum_rate", "cum_drive"], "всё вместе"),
        ]
        for cols, lab in sets:
            X = np.column_stack([[r[c] for r in rows] for c in cols])
            ok = np.all(np.isfinite(X), axis=1)
            if ok.sum() < 20:
                print(f"  {lab:<34} мало данных")
                continue
            pred = cross_val_predict(
                GradientBoostingRegressor(random_state=0, n_estimators=200,
                                          max_depth=3),
                X[ok], p_final[ok], cv=KFold(5, shuffle=True, random_state=0))
            resid = p_final[ok] - pred
            r2 = 1 - (resid ** 2).sum() / ((p_final[ok] - p_final[ok].mean()) ** 2).sum()
            print(f"  {lab:<34} R²(CV) = {r2:6.3f}   ост. σ = {resid.std():6.2f} бар")
    except ImportError:
        print("  sklearn недоступен — pip install scikit-learn")

    # --- ПЕРЕКЛЮЧЕНИЕ: к какой уставке приходит давление ---
    print(f"\n{'=' * 78}\nК КАКОЙ УСТАВКЕ ПРИХОДИТ ДАВЛЕНИЕ (по доле k*h нагнетания)\n{'=' * 78}")
    kf = np.array([r["kh_frac_inj"] for r in rows])
    di = np.abs([r["d_to_inj"] for r in rows])
    dp = np.abs([r["d_to_prod"] for r in rows])
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print(f"  {'доля k*h инж.':<16}{'кейсов':>8}{'|p-bhp_inj|':>14}{'|p-bhp_prod|':>15}  ближе к")
    for lo, hi in bins:
        sel = (kf >= lo) & (kf < hi)
        if sel.sum() == 0:
            continue
        mi, mp = np.nanmean(di[sel]), np.nanmean(dp[sel])
        who = "нагнетанию" if mi < mp else "добыче"
        print(f"  [{lo:.1f}, {hi:.1f})       {sel.sum():>8}{mi:>14.1f}{mp:>15.1f}  {who}")

    best = max(scored)[1]
    print(f"\n{'=' * 78}\nЭКСТРЕМАЛЬНЫЕ КЕЙСЫ (лучший одиночный признак: {best})\n{'=' * 78}")
    print(f"  {'case':<12}{'p_final':>10}{'env_kh':>10}{'bhp_prod':>10}"
          f"{'bhp_inj':>10}{'kh_inj':>9}{'cum_rate':>11}")
    for r_ in sorted(rows, key=lambda r: -abs(r["amp"]))[:10]:
        print(f"  {r_['name']:<12}{r_['p_final']:>10.1f}{r_['env_kh']:>10.1f}"
              f"{r_['bhp_prod']:>10.1f}{r_['bhp_inj']:>10.1f}"
              f"{r_['kh_frac_inj']:>9.2f}{r_['cum_rate']:>11.4g}")

    print(f"\n  Если env_kh близко к p_final на этих строках — гипотеза верна:")
    print(f"  давление едет к взвешенной уставке, и это и есть нужный канал.")


if __name__ == "__main__":
    main()

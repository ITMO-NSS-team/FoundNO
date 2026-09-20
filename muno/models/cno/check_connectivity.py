# -*- coding: utf-8 -*-
"""
check_connectivity.py — проверка гипотезы о гидродинамической связности.

Контекст: четыре гипотезы подряд не объяснили ошибку амплитуды давления
(каналы матбаланса, огибающая уставок, level-терм в лоссе, дисбаланс выборки).
Остаточное объяснение: амплитуда определяется не глобальными суммами, а тем,
КАКИЕ скважины гидродинамически сообщаются — пробивается ли нагнетание к
добыче по высокопроницаемому коридору или отсечено барьером. Это свойство
графа связности поля проницаемости, и никакой скаляр-агрегат его не выражает.

Скрипт считает по каждому кейсу признаки связности (только inputs/*, модель не
нужна) и отвечает на два вопроса:

  1. Объясняют ли они амплитуду ЛУЧШЕ глобальных агрегатов? Сравниваются
     нелинейные CV-R²: базовый набор, связность, оба вместе. Ключевая цифра —
     прирост от добавления связности к базовому набору.
  2. Отличаются ли ПРОВАЛЬНЫЕ кейсы от медианных по связности? Если да,
     механизм найден и лечится разрешением модели (моды, локальная ветка),
     а не входными каналами.

Признаки связности:
  lperm_std      разброс log10(permx) по пласту — контрастность геологии
  lperm_p90_p50  отношение перцентилей — насколько выражены каналы
  kh_total       суммарная проводимость (нормировка)
  dist_ip        расстояние между центроидами нагнетания и добычи (в ячейках)
  res_min        минимальное сопротивление пути нагнетание->добыча
                 (Дейкстра, вес ребра = 1/гармоническое среднее k*h)
  res_mean       среднее сопротивление до ближайшей добывающей по всем нагнет.
  perc_frac      доля ячеек верхнего дециля k*h в компоненте связности,
                 соединяющей нагнетание с добычей (перколяция каналов)
  n_prod, n_inj  число скважино-ячеек

Запуск:
  python check_connectivity.py --data $LOCAL_H5 --cases 512-639
"""

import argparse
import re

import h5py
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

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


def build_graph(kh, mask):
    """Граф соседей 4-связности; вес ребра = сопротивление = 1/гарм.среднее k*h."""
    H, W = kh.shape
    idx = -np.ones((H, W), np.int64)
    act = np.argwhere(mask)
    idx[mask] = np.arange(len(act))
    rows, cols, vals = [], [], []
    for di, dj in ((0, 1), (1, 0)):
        a = mask[: H - di, : W - dj] & mask[di:, dj:]
        if not a.any():
            continue
        i0 = idx[: H - di, : W - dj][a]
        i1 = idx[di:, dj:][a]
        k0 = kh[: H - di, : W - dj][a]
        k1 = kh[di:, dj:][a]
        harm = 2.0 * k0 * k1 / np.maximum(k0 + k1, 1e-12)
        w = 1.0 / np.maximum(harm, 1e-12)
        rows += [i0, i1]
        cols += [i1, i0]
        vals += [w, w]
    n = len(act)
    if not rows:
        return None, idx, act
    g = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n, n)).tocsr()
    return g, idx, act


def case_features(f, name):
    g = f[f"cases/{name}"]
    mask = np.asarray(g["inputs/static/mask"][:]) > 0.5
    permx = np.asarray(g["inputs/static/permx"][:], np.float64)
    thick = np.asarray(g["inputs/geometry/cell_thickness"][:], np.float64)
    p0 = np.asarray(g["inputs/initial/pressure"][:], np.float64)
    ctrl = np.asarray(g["inputs/controls/scheduled_grid_tensor"][:], np.float64)
    press = np.asarray(g["targets/pressure"][:], np.float64)
    tdays = np.asarray(g["targets/time_days"][:], np.float64)

    kh = np.where(mask, permx * thick, 0.0)
    lp = np.log10(np.clip(permx[mask], 1e-3, None))
    out = dict(name=name)
    out["lperm_std"] = float(lp.std())
    out["lperm_p90_p50"] = float(np.percentile(lp, 90) - np.percentile(lp, 50))
    out["kh_total"] = float(np.log10(max(kh[mask].sum(), 1e-9)))

    ever_p = (ctrl[:, 0] > 0.5).any(axis=0) & mask
    ever_i = (ctrl[:, 1] > 0.5).any(axis=0) & mask
    out["n_prod"] = int(ever_p.sum())
    out["n_inj"] = int(ever_i.sum())

    kh_p = kh[ever_p].sum()
    kh_i = kh[ever_i].sum()
    out["kh_frac_inj"] = float(kh_i / max(kh_p + kh_i, 1e-12))
    bp = ctrl[:, 2][(ctrl[:, 0] > 0.5)]
    bi = ctrl[:, 5][(ctrl[:, 1] > 0.5)]
    out["bhp_prod"] = float(bp.mean()) if bp.size else 250.0
    out["bhp_inj"] = float(bi.mean()) if bi.size else 250.0

    # --- расстояние между центроидами ---
    if ever_p.any() and ever_i.any():
        cp = np.argwhere(ever_p).mean(axis=0)
        ci = np.argwhere(ever_i).mean(axis=0)
        out["dist_ip"] = float(np.hypot(*(cp - ci)))
    else:
        out["dist_ip"] = np.nan

    # --- сопротивление пути нагнетание -> добыча ---
    graph, idx, act = build_graph(kh, mask)
    res_min = res_mean = np.nan
    perc_frac = np.nan
    if graph is not None and ever_p.any() and ever_i.any():
        src = idx[ever_i]
        dst = idx[ever_p]
        src = src[src >= 0]
        dst = dst[dst >= 0]
        if src.size and dst.size:
            d = dijkstra(graph, indices=src, min_only=False)
            dd = d[:, dst]
            dd = np.where(np.isfinite(dd), dd, np.nan)
            if np.isfinite(dd).any():
                res_min = float(np.nanmin(dd))
                res_mean = float(np.nanmean(np.nanmin(dd, axis=1)))

        # --- перколяция верхнего дециля k*h ---
        thr = np.percentile(kh[mask], 90)
        hi = mask & (kh >= thr)
        gh, idxh, acth = build_graph(kh, hi)
        if gh is not None and len(acth):
            ncomp, lab = connected_components(gh, directed=False)
            lp_ = np.full(mask.shape, -1)
            lp_[hi] = lab
            comp_p = set(lp_[ever_p & hi].tolist())
            comp_i = set(lp_[ever_i & hi].tolist())
            shared = (comp_p & comp_i) - {-1}
            if shared:
                sizes = np.bincount(lab, minlength=ncomp)
                perc_frac = float(sum(sizes[c] for c in shared) / max(len(acth), 1))
            else:
                perc_frac = 0.0
    out["res_min"] = res_min
    out["res_mean"] = res_mean
    out["perc_frac"] = perc_frac

    # --- базовые глобальные агрегаты (для честного сравнения) ---
    T = ctrl.shape[0]
    dt = np.zeros(T)
    if T > 1:
        dt[1:] = tdays[1:] - tdays[:-1]
        dt[0] = dt[1]
    else:
        dt[0] = 1.0
    dt = np.clip(dt, 0, None)[:, None, None]
    pv = np.asarray(g["inputs/geometry/pore_volume"][:], np.float64) * mask
    pv_tot = max(pv.sum(), 1e-12)
    net = (ctrl[:, 3] - ctrl[:, 4]) * dt * mask[None]
    out["cum_rate"] = float(np.cumsum(net.sum(axis=(1, 2)))[-1] / pv_tot)
    drive = ((ctrl[:, 0] > 0.5) * np.clip(p0[None] - ctrl[:, 2], 0, None)
             - (ctrl[:, 1] > 0.5) * np.clip(ctrl[:, 5] - p0[None], 0, None))
    drive = drive * kh[None] * dt * mask[None]
    out["cum_drive"] = float(np.cumsum(drive.sum(axis=(1, 2)))[-1] / pv_tot)

    p_mean = (press * mask[None]).sum(axis=(1, 2)) / max(mask.sum(), 1e-12)
    out["p_final"] = float(p_mean[-1])
    out["amp"] = float(p0[mask].mean() - p_mean[-1])
    return out


def cv_r2(rows, cols, target, seed=0):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_predict, KFold
    X = np.column_stack([[r[c] for r in rows] for c in cols])
    y = np.array([r[target] for r in rows], float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if ok.sum() < 25:
        return np.nan, np.nan, int(ok.sum())
    pred = cross_val_predict(
        GradientBoostingRegressor(random_state=seed, n_estimators=300, max_depth=3),
        X[ok], y[ok], cv=KFold(5, shuffle=True, random_state=seed))
    resid = y[ok] - pred
    r2 = 1 - (resid ** 2).sum() / max(((y[ok] - y[ok].mean()) ** 2).sum(), 1e-12)
    return float(r2), float(resid.std()), int(ok.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--cases", default="")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--worst", default="525,515,512,531,561,543,559,516",
                   help="CaseID провальных кейсов для сравнения с медианой")
    args = p.parse_args()

    cases = list_cases(args.data)
    if args.cases:
        want = parse_ranges(args.cases)
        cases = [(i, n) for i, n in cases if i in want]
    elif args.limit:
        cases = cases[:args.limit]
    print(f"Считаю связность по {len(cases)} кейсам\n")

    rows = []
    with h5py.File(args.data, "r") as f:
        for k, (_, name) in enumerate(cases):
            rows.append(case_features(f, name))
            if (k + 1) % 25 == 0 or k + 1 == len(cases):
                print(f"  ...{k + 1}/{len(cases)}", flush=True)

    GLOB = ["bhp_prod", "bhp_inj", "kh_frac_inj", "cum_rate", "cum_drive"]
    CONN = ["lperm_std", "lperm_p90_p50", "kh_total", "dist_ip",
            "res_min", "res_mean", "perc_frac", "n_prod", "n_inj"]

    print(f"\n{'=' * 78}\n1. ОБЪЯСНЯЮТ ЛИ СВЯЗНОСТЬ АМПЛИТУДУ (5-fold CV)\n{'=' * 78}")
    for cols, lab in [(GLOB, "глобальные агрегаты (базовая линия)"),
                      (CONN, "только связность"),
                      (GLOB + CONN, "агрегаты + связность")]:
        r2, sd, n = cv_r2(rows, cols, "amp")
        print(f"  {lab:<38} R² = {r2:6.3f}   ост. σ = {sd:6.2f} бар  (n={n})")
    print("  Ключевое: прирост R² от добавления связности к агрегатам.")
    print("  <0.05 — связность не при чём; >0.2 — механизм найден.")

    print(f"\n{'=' * 78}\n2. ОДИНОЧНАЯ СВЯЗЬ ПРИЗНАКОВ СВЯЗНОСТИ С АМПЛИТУДОЙ\n{'=' * 78}")
    amp = np.array([r["amp"] for r in rows], float)
    print(f"  {'признак':<16}{'Спирмен r':>12}{'Пирсон r':>11}")
    for c in CONN:
        v = np.array([r[c] for r in rows], float)
        ok = np.isfinite(v) & np.isfinite(amp)
        if ok.sum() < 5 or v[ok].std() < 1e-12:
            print(f"  {c:<16}{'н/д':>12}")
            continue
        rs = np.corrcoef(np.argsort(np.argsort(v[ok])),
                         np.argsort(np.argsort(amp[ok])))[0, 1]
        rp = np.corrcoef(v[ok], amp[ok])[0, 1]
        print(f"  {c:<16}{rs:>12.3f}{rp:>11.3f}")

    print(f"\n{'=' * 78}\n3. ПРОВАЛЬНЫЕ КЕЙСЫ ПРОТИВ ОСТАЛЬНЫХ\n{'=' * 78}")
    worst_ids = parse_ranges(args.worst)
    is_worst = np.array([bool(_NUM.search(r["name"])
                              and int(_NUM.search(r["name"]).group(1)) in worst_ids)
                         for r in rows])
    print(f"  провальных в выборке: {is_worst.sum()} из {len(rows)}")
    if is_worst.any() and (~is_worst).any():
        print(f"  {'признак':<16}{'провальные':>13}{'остальные':>12}{'отн.':>9}")
        for c in CONN + ["amp"]:
            v = np.array([r[c] for r in rows], float)
            a, b = np.nanmedian(v[is_worst]), np.nanmedian(v[~is_worst])
            rel = a / b if abs(b) > 1e-12 else np.nan
            print(f"  {c:<16}{a:>13.4g}{b:>12.4g}{rel:>9.2f}")
        print("  Отношение около 1 по всем строкам => по связности провальные")
        print("  кейсы не отличаются, и гипотеза не подтверждается.")


if __name__ == "__main__":
    main()

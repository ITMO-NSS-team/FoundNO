# -*- coding: utf-8 -*-
"""
inspect_dataset_v2.py — что реально варьируется в новом датасете
(tnav_operator_dataset_orthogonal_2512_bhp_rate_v1.h5).

Прежде чем расширять входные каналы, надо измерить, что меняется между
кейсами, а что константа. Скрипт отвечает на четыре вопроса:

  1. КАКИЕ ПОЛЯ ЕСТЬ. Перечисляет датасеты первого кейса — сразу видно,
     появились ли fault_mask / multx / multy / hfrac_*_mask и т.д.
  2. ЧТО ВАРЬИРУЕТСЯ. Для скалярных групп (PVT, ОФП, rock) и для карт
     считает разброс МЕЖДУ кейсами: если std/|mean| < 1e-6, признак
     константа и подавать его каналом бессмысленно.
  3. НАЧАЛЬНЫЕ УСЛОВИЯ. Раньше p0 был ровно 250 везде; теперь заявлены
     переменные уровень и градиент — проверяем распределение.
  4. ЛОВУШКА ИЗ РАЗДЕЛА 13. producer_orat сохраняется даже при BHP-
     контроле, то есть неактивные rate-поля становятся шумом. Считаем,
     в какой доле открытых скважино-ячеек rate задан при BHP-режиме.

Нужны только numpy и h5py.

Запуск:
  python inspect_dataset_v2.py --data /content/tnav_data_2/<файл>.h5 --limit 200
"""

import argparse
import json
import re

import h5py
import numpy as np

_NUM = re.compile(r"(\d+)")


def list_cases(path):
    out = []
    with h5py.File(path, "r") as f:
        for name in f["cases"].keys():
            m = _NUM.search(name)
            if m:
                out.append((int(m.group(1)), name))
    out.sort(key=lambda x: x[0])
    return out


def walk(g, prefix=""):
    items = []
    for k, v in g.items():
        if isinstance(v, h5py.Dataset):
            items.append((prefix + k, v.shape, str(v.dtype)))
        else:
            items.extend(walk(v, prefix + k + "/"))
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=200,
                   help="сколько кейсов просмотреть для статистики")
    args = p.parse_args()

    cases = list_cases(args.data)
    print(f"Кейсов в файле: {len(cases)} (id {cases[0][0]}..{cases[-1][0]})")
    names = [n for _, n in cases[:args.limit]]

    with h5py.File(args.data, "r") as f:
        # ---------- 1. структура ----------
        print(f"\n{'=' * 78}\n1. ПОЛЯ ПЕРВОГО КЕЙСА\n{'=' * 78}")
        for path, shape, dt in walk(f[f"cases/{names[0]}"]):
            print(f"  {path:<62} {str(shape):<18} {dt}")

        # ---------- 2. скалярные группы ----------
        print(f"\n{'=' * 78}\n2. СКАЛЯРНЫЕ ГРУППЫ: варьируются ли между кейсами\n{'=' * 78}")
        scalar_paths = [
            ("inputs/pvt/density", ["oil_density", "water_density"]),
            ("inputs/pvt/pvcdo", ["p", "oil_fvf", "oil_compr", "oil_visc", "oil_viscosib"]),
            ("inputs/pvt/pvtw", ["ref_p", "water_fvf", "water_compr", "water_visc"]),
            ("inputs/relperm/coreywo", ["swl", "swu", "swcr", "sowcr", "krolw",
                                        "krorw", "krwr", "krwu", "pcow", "now",
                                        "nw", "np", "spc0"]),
            ("inputs/rock/rock", ["ref_p", "rock_compr"]),
        ]
        for path, cols in scalar_paths:
            if path not in f[f"cases/{names[0]}"]:
                print(f"  {path}: НЕТ")
                continue
            vals = np.stack([np.asarray(f[f"cases/{n}/{path}"][:], np.float64).ravel()
                             for n in names])
            print(f"  {path}  {vals.shape[1]} значений:")
            for j in range(vals.shape[1]):
                v = vals[:, j]
                rel = v.std() / max(abs(v.mean()), 1e-12)
                tag = "варьируется" if rel > 1e-6 else "КОНСТАНТА"
                lab = cols[j] if j < len(cols) else f"col{j}"
                print(f"    [{j:2d}] {lab:<14} mean {v.mean():12.5g}  "
                      f"std {v.std():11.5g}  {tag}")

        # ---------- 3. карты и начальные условия ----------
        print(f"\n{'=' * 78}\n3. КАРТЫ: доля кейсов, где признак ненулевой / варьируется\n{'=' * 78}")
        map_paths = ["inputs/static/fault_mask", "inputs/static/multx",
                     "inputs/static/multy", "inputs/static/hfrac_core_mask",
                     "inputs/static/hfrac_proxy_mask", "inputs/static/hfrac_srv_mask",
                     "inputs/initial/pressure", "inputs/initial/swat"]
        for path in map_paths:
            if path not in f[f"cases/{names[0]}"]:
                print(f"  {path:<40} НЕТ")
                continue
            stats = []
            nonzero_cases = 0
            for n in names:
                a = np.asarray(f[f"cases/{n}/{path}"][:], np.float64)
                m = np.asarray(f[f"cases/{n}/inputs/static/mask"][:]) > 0.5
                a = a[m]
                stats.append((a.mean(), a.std(), a.min(), a.max()))
                if path.endswith("_mask"):
                    nonzero_cases += int((a > 0.5).any())
                elif "mult" in path:
                    nonzero_cases += int((a < 0.999).any())
            st = np.array(stats)
            between = st[:, 0].std() / max(abs(st[:, 0].mean()), 1e-12)
            extra = ""
            if path.endswith("_mask") or "mult" in path:
                extra = f" | активен в {100*nonzero_cases/len(names):.1f}% кейсов"
            print(f"  {path:<40} среднее по кейсам {st[:,0].mean():9.4g}  "
                  f"разброс между кейсами {between:8.4g}"
                  f"{'  ВАРЬИРУЕТСЯ' if between > 1e-6 else '  КОНСТАНТА'}{extra}")
            if "pressure" in path:
                print(f"      p0: диапазон средних по кейсам "
                      f"[{st[:,0].min():.1f}, {st[:,0].max():.1f}], "
                      f"градиент внутри кейса: std {st[:,1].mean():.2f} бар")

        # ---------- 4. ловушка из раздела 13 ----------
        print(f"\n{'=' * 78}\n4. CONTROL_TYPE_CODE и неактивные rate-поля\n{'=' * 78}")
        codes_p, codes_i = {}, {}
        orat_under_bhp = 0
        orat_total = 0
        for n in names[:min(len(names), 100)]:
            ctrl = np.asarray(f[f"cases/{n}/inputs/controls/scheduled_grid_tensor"][:],
                              np.float64)
            op_p = ctrl[:, 0] > 0.5
            op_i = ctrl[:, 1] > 0.5
            for c in np.unique(ctrl[:, 6][op_p]):
                codes_p[float(c)] = codes_p.get(float(c), 0) + int((ctrl[:, 6][op_p] == c).sum())
            for c in np.unique(ctrl[:, 7][op_i]):
                codes_i[float(c)] = codes_i.get(float(c), 0) + int((ctrl[:, 7][op_i] == c).sum())
            # rate задан, но режим не rate-контроль
            rate_set = (np.abs(ctrl[:, 3]) > 0) & op_p
            orat_total += int(rate_set.sum())
            # предполагаем, что код rate-контроля — самый частый среди ячеек с ненулевым rate
            orat_under_bhp += int((rate_set & (ctrl[:, 2] > 0)).sum())
        print(f"  producer_control_type_code (открытые ячейки): "
              f"{ {k: v for k, v in sorted(codes_p.items())} }")
        print(f"  injector_control_type_code (открытые ячейки): "
              f"{ {k: v for k, v in sorted(codes_i.items())} }")
        if orat_total:
            print(f"  ячеек с заданным orat, где одновременно задан bhp: "
                  f"{100*orat_under_bhp/orat_total:.1f}% "
                  f"— признак ловушки из раздела 13")

        # ---------- 5. метаданные ----------
        print(f"\n{'=' * 78}\n5. METADATA_JSON первого кейса (ключи)\n{'=' * 78}")
        try:
            raw = f[f"cases/{names[0]}/metadata_json"][()]
            md = json.loads(raw if isinstance(raw, str) else raw.decode())
            for k in sorted(md.keys()):
                v = md[k]
                s = str(v)
                print(f"  {k:<34} {s[:70]}")
        except Exception as e:
            print(f"  не прочитан: {e}")


if __name__ == "__main__":
    main()

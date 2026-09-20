# -*- coding: utf-8 -*-
"""
eval_foundno.py — гифки и метрики для ДООБУЧЕННОЙ FoundNO-модели.

Идея: FoundNO-модель ожидает свою цепочку нормализации
    x_tnav -> in_norm.transform -> model -> out_norm.inverse_transform -> физ.,
а весь готовый tnav-инструментарий (predict_case / eval_tnav / metrics_full)
ожидает модель, которая принимает tnav-стек и возвращает предсказание
в пространстве NormalizerV6. Обёртка ниже приводит одно к другому, поэтому
скрипты визуализации и метрик работают БЕЗ ПРАВОК.

Положить рядом с tnav-скриптами (или указать --tnav-pkg).

Запуск — метрики по всем сплитам:
  python eval_foundno.py --mode metrics \\
      --ft-ckpt /content/drive/MyDrive/foundno_tnav_mamba/best_wells_adapters.pt \\
      --core-pt /content/foundno_ckpts/checkpoints/core.pt \\
      --data $LOCAL_H5 --foundno-repo /content/FoundNO --tnav-pkg /content/tnav_fno

Гифки и метрики одного кейса:
  python eval_foundno.py --mode gifs --case-id 600 --out pics_foundno_600 ... (те же пути)

Нормализаторы и tnav-статистики берутся из папки чекпойнта
(normalizers/*.pkl, tnav_norm_stats.pt) — можно переопределить флагами.
"""

import argparse
import sys
from pathlib import Path

import torch


class FoundNOAsTnav(torch.nn.Module):
    """FoundNO-модель в интерфейсе tnav-модели.

    Вход:  tnav-стек x [B, 21, T, H, W] (как его строит predict_case).
    Выход: предсказание в пространстве NormalizerV6 — downstream-код применит
           norm_v6.denorm_targets и получит физические единицы.
    """

    def __init__(self, model, in_norm, out_norm, norm_v6):
        super().__init__()
        self.model = model
        self.in_norm = in_norm
        self.out_norm = out_norm
        self.norm_v6 = norm_v6

    def forward(self, x):
        z = self.in_norm.transform(x)
        pred_phys = self.out_norm.inverse_transform(self.model(z))
        return self.norm_v6.norm_targets(pred_phys, ch_dim=1)


def build(args, device):
    sys.path.insert(0, str(Path(args.tnav_pkg).resolve()))
    sys.path.insert(0, str(Path(args.foundno_repo).resolve()))
    sys.path.insert(0, str((Path(args.foundno_repo) / "experiments/scripts").resolve()))

    from tnav_data import NormalizerV6
    from fnofound.data import UnitGaussianNormalizer
    from finetune_tnav_adapters import load_frozen_core, load_finetuned

    ck_dir = Path(args.ft_ckpt).parent
    norm_dir = Path(args.norm_dir) if args.norm_dir else ck_dir / "normalizers"
    stats_path = Path(args.tnav_stats) if args.tnav_stats else ck_dir / "tnav_norm_stats.pt"

    # tnav-нормализатор (для сборки входа и денормализации выхода).
    # Обычно лежит рядом с чекпойнтом; если его нет — пересчитываем по train-
    # сплиту (детерминировано, зависит только от датасета и границ сплитов).
    if stats_path.exists():
        st = torch.load(stats_path, map_location="cpu", weights_only=False)
        norm_v6 = NormalizerV6.from_state(st["state"]).to(device)
    else:
        from tnav_data import list_cases, make_splits, fit_stats
        print(f"[!] {stats_path} не найден — пересчитываю tnav-статистики "
              f"(несколько минут)")
        print(f"    в папке чекпойнта: "
              f"{sorted(p.name for p in ck_dir.iterdir())}")
        cases = list_cases(args.data)
        train_names, _ = make_splits(cases, {"g": args.val_geology,
                                             "c": args.val_controls})
        norm_v6 = fit_stats(args.data, train_names,
                            cache_path=str(stats_path)).to(device)

    # FoundNO-нормализаторы
    if not (norm_dir / "input_normalizer_0.pkl").exists():
        raise SystemExit(
            f"Нет {norm_dir}/input_normalizer_0.pkl — укажите --norm-dir "
            f"или проверьте, что --ft-ckpt указывает на папку нужного прогона")
    in_norm = UnitGaussianNormalizer()
    in_norm.from_file(str(norm_dir / "input_normalizer_0.pkl"))
    out_norm = UnitGaussianNormalizer()
    out_norm.from_file(str(norm_dir / "output_normalizer_0.pkl"))
    in_norm.to(device)
    out_norm.to(device)

    # ядро + дообученные адаптеры (тип адаптера берётся из metadata чекпойнта)
    core = load_frozen_core(args.core_pt, device)
    model, md = load_finetuned(args.ft_ckpt, core, device)
    print(f"Checkpoint: {args.ft_ckpt}")
    print(f"  adapter={md.get('adapter')} scan={md.get('mamba_scan')} "
          f"core_mode={md.get('core_mode')} epoch={md.get('epoch', '?')}")

    wrapped = FoundNOAsTnav(model, in_norm, out_norm, norm_v6).to(device).eval()
    return wrapped, norm_v6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="metrics", choices=["metrics", "gifs", "both"])
    p.add_argument("--ft-ckpt", required=True,
                   help="best_wells_adapters.pt / best_adapters.pt / last_adapters.pt")
    p.add_argument("--core-pt", required=True, help="checkpoints/core.pt претрейна")
    p.add_argument("--data", required=True, help="tnav .h5")
    p.add_argument("--foundno-repo", default="/content/FoundNO")
    p.add_argument("--tnav-pkg", default="/content/tnav_fno")
    p.add_argument("--norm-dir", default="", help="по умолчанию <ckpt_dir>/normalizers")
    p.add_argument("--tnav-stats", default="", help="по умолчанию <ckpt_dir>/tnav_norm_stats.pt")
    p.add_argument("--val-geology", default="512-639")
    p.add_argument("--val-controls", default="1920-2047")
    p.add_argument("--splits", default="geology,controls",
                   help="для --mode metrics: train,geology,controls")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--case-id", type=int, default=-1, help="для --mode gifs")
    p.add_argument("--out", default="pics_foundno")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm_v6 = build(args, device)

    from tnav_data import list_cases, parse_ranges, predict_case

    if args.mode in ("metrics", "both"):
        from metrics_full import evaluate_split
        cases = list_cases(args.data)
        geo, ctrl = parse_ranges(args.val_geology), parse_ranges(args.val_controls)
        val_ids = geo | ctrl
        pools = {
            "train": [n for i, n in cases if i not in val_ids],
            "geology": [n for i, n in cases if i in geo],
            "controls": [n for i, n in cases if i in ctrl],
        }
        tags = {"train": "TRAIN", "geology": "GEOLOGY_HOLDOUT (OOD)",
                "controls": "CONTROLS_HOLDOUT (OOD)"}
        for s in [t.strip() for t in args.splits.split(",")]:
            if s not in pools:
                print(f"WARN: неизвестный сплит '{s}'")
                continue
            evaluate_split(model, norm_v6, args.data, pools[s], device,
                           tags[s], limit=args.limit)

    if args.mode in ("gifs", "both"):
        import json, os
        from eval_tnav import (case_metrics, animate_channel,
                               plot_well_pressure_curves)
        cases = dict(list_cases(args.data))
        name = (next(iter(cases.values())) if args.case_id < 0
                else cases[args.case_id])
        print(f"\nCase: {name}")
        res = predict_case(model, norm_v6, args.data, name, device)
        m = case_metrics(res)
        print(f"  pressure relL1 = {m['pressure_field_rel_l1_mean']:.5f} | "
              f"swat relL1 = {m['swat_field_rel_l1_mean']:.5f}")
        if m["well_pressure_mape"] is not None:
            print(f"  well P-MAPE = {m['well_pressure_mape']:.3f}%")
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "metrics.json"), "w") as jf:
            json.dump(m, jf, indent=2)
        animate_channel(res, 0, args.out, "Pressure")
        animate_channel(res, 1, args.out, "SWAT")
        plot_well_pressure_curves(res, args.out)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()

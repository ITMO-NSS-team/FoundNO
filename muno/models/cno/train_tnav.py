# -*- coding: utf-8 -*-
"""
train_tnav.py — обучение Temporal-Mamba + FNO3D на датасете tNavigator v6.

Запуск (Colab, A100 40GB):
  python train_tnav.py --data /content/tnav_operator_dataset_2048_v6.h5 \
      --ckpt-dir /content/drive/MyDrive/tnav_fno_ckpts --epochs 120

Смоук-проверка:
  python train_tnav.py --data ... --limit-cases 16 --epochs 2 --width 16

Сплиты (рекомендация из описания датасета — holdout по целым геологиям и
целым control-сценариям):
  --val-geology 512-639     ti_faithful_final_quality (Stage A)
  --val-controls 1920-2047  rare_multi_conversion (Stage B)
Остальные кейсы -> train. Бизнес-метрики печатаются отдельно по каждому
holdout: geology-OOD и controls-OOD.

Возобновление после обрыва Colab: --resume auto (подхватит last.pt).
"""

import os
import json
import math
import time
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tnav_data import (IN_CH, C_OUT, TARGET_NAMES, list_cases, make_splits,
                       fit_stats, TnavDataset, TnavCollate, NormalizerV6,
                       in_channels)
from tnav_model import build_model


# ============================================================================
# ЛОСС
# ============================================================================

class ReservoirLoss(nn.Module):
    """
    Полнополевой лосс на активной маске + явный терм на клетках скважин.
      relL2 — нормализованное пространство;
      relL1 — физическое, нормировка |mean_spatial(target_t)|;
      well  — |dp|/(|p|+1e-3*sigma_c) на открытых completions.
    Адаптивные веса каналов по EMA relL1 (pressure/swat балансируются сами).

    Хук для physics-loss: сюда можно добавить терм материального баланса на
    diagnostics/actual_connection_source_terms (q_o_surface, q_w_surface) —
    это НЕ вход модели, но легальный auxiliary target (раздел 1 схемы).
    """

    def __init__(self, normalizer, w_l2=1.0, w_l1=1.0, w_well=0.5,
                 w_level=0.0, adaptive=True, alpha=0.5, beta=0.99,
                 well_dilate=0, w_inj_mult=1.0, w_aux=0.0):
        super().__init__()
        self.norm = normalizer
        self.w_l2, self.w_l1, self.w_well = w_l2, w_l1, w_well
        self.w_level = w_level
        # well_dilate — радиус расширения well-маски (ячеек по H и W):
        #   depression-normalized метрика смотрит на соседей completion-ячейки,
        #   а лосс до сих пор штрафовал только саму ячейку;
        # w_inj_mult — множитель веса для нагнетательных: они стабильно вдвое
        #   хуже добывающих (уставка 750 атм на краю распределения);
        # w_aux — вес вспомогательного терма на фактические дебиты.
        self.well_dilate = int(well_dilate)
        self.w_inj_mult = float(w_inj_mult)
        self.w_aux = float(w_aux)
        self.adaptive, self.alpha, self.beta = adaptive, alpha, beta
        self.register_buffer('ema', torch.ones(C_OUT))
        self.register_buffer('ema_init', torch.tensor(False))

    def _weights(self, well_mask, inj_mask, res_mask):
        """Веса well-терма [B,1,T,H,W]: 1 у добывающих, w_inj_mult у
        нагнетательных, дилатация на радиус well_dilate внутри маски пласта."""
        w = well_mask.float()
        if inj_mask is not None and self.w_inj_mult != 1.0:
            w = torch.where(inj_mask, w * self.w_inj_mult, w)
        if self.well_dilate > 0:
            k = 2 * self.well_dilate + 1
            B, T, H, W = w.shape
            w = F.max_pool2d(w.reshape(B * T, 1, H, W), k, stride=1,
                             padding=self.well_dilate).reshape(B, T, H, W)
            w = w * res_mask.float()[:, None]
        return w[:, None]

    def forward(self, pred_n, tgt_n, res_mask, well_mask, inj_mask=None,
                aux_pred=None, aux_tgt=None):
        B, C, T, H, W = pred_n.shape
        m = res_mask.float()[:, None, None]                  # [B,1,1,H,W]
        eps3 = (1e-3 * self.norm.tgt_std).view(1, C, 1)
        eps5 = eps3.view(1, C, 1, 1, 1)

        diff = pred_n - tgt_n
        num = ((diff ** 2) * m).sum(dim=(0, 2, 3, 4)).clamp_min(1e-12).sqrt()
        den = ((tgt_n ** 2) * m).sum(dim=(0, 2, 3, 4)).clamp_min(1e-12).sqrt()
        rel_l2 = num / den.clamp_min(1e-3)

        pred_p = self.norm.denorm_targets(pred_n, ch_dim=1)
        tgt_p = self.norm.denorm_targets(tgt_n, ch_dim=1)
        dp = (pred_p - tgt_p).abs()
        area = m.sum(dim=(3, 4)).clamp_min(1.0)
        mae_t = (dp * m).sum(dim=(3, 4)) / area              # [B,C,T]
        mtgt = (tgt_p * m).sum(dim=(3, 4)) / area
        rel_l1 = (mae_t / (mtgt.abs() + eps3)).mean(dim=(0, 2))

        wm = self._weights(well_mask, inj_mask, res_mask)   # [B,1,T,H,W]
        wcnt = wm.sum().clamp_min(1.0)
        well = ((dp / (tgt_p.abs() + eps5)) * wm).sum(dim=(0, 2, 3, 4)) / wcnt

        # --- AUX: фактические дебиты в completion-ячейках (multi-task) ---
        # L1 в нормализованном (signed-log) пространстве, только на ячейках
        # скважин: вне их таргет нулевой и не несёт информации.
        aux = pred_n.new_zeros(())
        if self.w_aux > 0 and aux_pred is not None and aux_tgt is not None:
            wa = well_mask.float()[:, None]
            aux = ((aux_pred - aux_tgt).abs() * wa).sum() / wa.sum().clamp_min(1.0) / aux_pred.shape[1]

        # --- LEVEL: систематический сдвиг среднего давления по пласту ---
        # rel_l1 выше берёт mean(|err|) — величину ошибки независимо от знака.
        # Здесь наоборот |mean(err)|: знаковое усреднение по ячейкам, поэтому
        # шум сокращается и остаётся именно СМЕЩЕНИЕ уровня. Диагностика
        # показала, что в провальных кейсах ошибка почти постоянна по полю
        # (pred/target ~ 1.5-2 во всех ячейках) — этот терм там равен полной
        # ошибке, а на хорошо предсказанных кейсах близок к нулю.
        bias_t = ((pred_p - tgt_p) * m).sum(dim=(3, 4)) / area   # [B,C,T] знаковое
        level = (bias_t.abs() / (mtgt.abs() + eps3)).mean(dim=(0, 2))

        per_c = (self.w_l2 * rel_l2 + self.w_l1 * rel_l1
                 + self.w_well * well + self.w_level * level)

        if self.adaptive and self.training:
            with torch.no_grad():
                r = rel_l1.detach()
                if not bool(self.ema_init):
                    self.ema.copy_(r)
                    self.ema_init.fill_(True)
                else:
                    self.ema.mul_(self.beta).add_(r, alpha=1 - self.beta)
        if self.adaptive:
            lam = (self.ema / self.ema.mean().clamp_min(1e-12)).pow(self.alpha)
            lam = lam * C / lam.sum().clamp_min(1e-12)
        else:
            lam = torch.ones_like(per_c)

        return (lam * per_c).mean() + self.w_aux * aux, {
            'rel_l2': rel_l2.detach(), 'rel_l1': rel_l1.detach(),
            'well': well.detach(), 'level': level.detach(),
            'aux': aux.detach(),
        }


# ============================================================================
# БИЗНЕС-МЕТРИКИ (на denorm-значениях; well MAPE — по давлению)
# ============================================================================

@torch.no_grad()
def evaluate_business(model, loader, normalizer, device, max_samples=128):
    model.eval()
    field_sum = torch.zeros(C_OUT, device=device)
    field_cnt = 0.0
    wp_sum = torch.zeros((), device=device)   # pressure MAPE на скважинах
    wp_cnt = 0.0
    sw_mae_sum = torch.zeros((), device=device)
    sw_cnt = 0.0
    seen = 0

    for batch in loader:
        x = batch['x'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)
        res = batch['res'].to(device, non_blocking=True)
        well = batch['well'].to(device, non_blocking=True)

        pred_p = normalizer.denorm_targets(model(x), ch_dim=1)
        tgt_p = normalizer.denorm_targets(y, ch_dim=1)
        B, C, T, H, W = pred_p.shape

        m = res.float()[:, None, None]
        dp = (pred_p - tgt_p).abs()
        area = m.sum(dim=(3, 4)).clamp_min(1.0)
        mae_t = (dp * m).sum(dim=(3, 4)) / area
        mtgt = (tgt_p * m).sum(dim=(3, 4)) / area
        field_sum += (mae_t / (mtgt.abs() + 1e-8)).sum(dim=(0, 2))
        field_cnt += B * T

        wm = well.float()                                        # [B,T,H,W]
        wp_sum += (100.0 * dp[:, 0] / (tgt_p[:, 0].abs() + 1e-4) * wm).sum()
        wp_cnt += wm.sum().item()

        sw_mae_sum += (dp[:, 1] * m[:, 0]).sum()
        sw_cnt += (m[:, 0].sum() * T).item()

        seen += B
        if seen >= max_samples:
            break

    return {
        'field_rel_l1': (field_sum / max(field_cnt, 1.0)).cpu(),
        'well_pressure_mape': float(wp_sum.item() / max(wp_cnt, 1.0)),
        'swat_mae': float(sw_mae_sum.item() / max(sw_cnt, 1.0)),
        'n_samples': seen,
    }


def fmt_pc(t):
    return " ".join(f"{TARGET_NAMES[i]}:{v:.4f}" for i, v in enumerate(t.tolist()))


# ============================================================================
# EMA ВЕСОВ
# ============================================================================

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.n_updates = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # Разгон decay: min(decay, (1+n)/(10+n)) — ранняя EMA плотно следует
        # за моделью, иначе первые ~15 эпох она содержит случайную
        # инициализацию и бизнес-метрики по ней бессмысленны.
        self.n_updates += 1
        d = min(self.decay, (1 + self.n_updates) / (10 + self.n_updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k]
            if not torch.is_tensor(v) or not torch.is_tensor(src):
                continue
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src.detach(), alpha=1 - d)
            else:
                v.copy_(src)


# ============================================================================
# ЧЕКПОЙНТЫ
# ============================================================================

def save_ckpt(path, model, ema, opt, sched, normalizer, cfg, epoch,
              best_field, best_wells, metrics=None):
    torch.save({
        'model': model.state_dict(),
        'ema': ema.module.state_dict(),
        'ema_n': ema.n_updates,
        'opt': opt.state_dict(),
        'sched': sched.state_dict(),
        'normalizer': normalizer.state(),
        'config': cfg,
        'epoch': epoch,
        'best_field': best_field,
        'best_wells': best_wells,
        'val_business': metrics,
    }, path)


def metrics_to_jsonable(d):
    out = {}
    for k, v in d.items():
        out[k] = v.tolist() if torch.is_tensor(v) else v
    return out


# ============================================================================
# TRAIN
# ============================================================================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='путь к tnav_operator_dataset_*.h5')
    p.add_argument('--ckpt-dir', default='./tnav_ckpts')
    p.add_argument('--val-geology', default='512-639',
                   help='CaseID holdout по геологии (пусто = выключить)')
    p.add_argument('--val-controls', default='1920-2047',
                   help='CaseID holdout по control-сценарию (пусто = выключить)')
    p.add_argument('--limit-cases', type=int, default=0,
                   help='ограничить train N кейсами (смоук-режим)')
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--min-lr-ratio', type=float, default=0.03)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--ema-decay', type=float, default=0.999)
    p.add_argument('--modes', default='12,32,32')
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--rank', type=float, default=0.10)
    p.add_argument('--domain-padding', type=float, default=0.15)
    p.add_argument('--arch', default='temporal',
                   choices=['fno', 'temporal', 'spatial', 'spatial-temporal'],
                   help='fno: чистый FNO3D; temporal: Mamba/conv по времени; '
                        'spatial: PostLift-скан по растру H*W (исходный '
                        'PostLiftMambaFNO, исправленные оси); spatial-temporal: оба')
    p.add_argument('--temporal-ssm', default='auto', choices=['auto', 'mamba', 'conv'])
    p.add_argument('--split-mode', default='caseid',
                   choices=['caseid', 'fault-ood', 'random'],
                   help='caseid — диапазоны --val-geology/--val-controls '
                        '(осмысленно только для старого датасета); '
                        'fault-ood — отложить ВСЕ кейсы с разломами как '
                        'структурный OOD плюс случайный in-distribution '
                        'holdout; random — только случайный holdout')
    p.add_argument('--val-frac', type=float, default=0.1,
                   help='доля случайного holdout для split-mode random/fault-ood')
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--fault-channels', action='store_true',
                   help='+3 канала разломов (fault_mask, log multx, log multy) '
                        '— только для датасета orthogonal_2512')
    p.add_argument('--hfrac-channels', action='store_true',
                   help='+1 канал маски ГРП. ВНИМАНИЕ: маски не согласованы с '
                        'метаданными (ненулевые при hfrac_mode=none), '
                        'семантика неизвестна — включать осознанно')
    p.add_argument('--physics-channels', action='store_true',
                   help='+10 каналов физических скаляров (вязкости, '
                        'сжимаемости, параметры Кори), broadcast по полю. '
                        'Сжимаемости прямо задают амплитуду давления')
    p.add_argument('--diff-kernels', default='off',
                   choices=['off', 'parallel'],
                   help='дифференциальные ядра LocalNO параллельно каждому '
                        'спектральному блоку (конечно-разностный шаблон с '
                        'нормировкой на шаг сетки). Дёшево: ~0.4M при width=64. '
                        'alpha=0 на старте, поэтому off и parallel стартуют '
                        'одинаково. Комбинируется с --local-branch (CNO)')
    p.add_argument('--diff-kernel-size', type=int, default=3)
    p.add_argument('--diff-padding', default='replicate',
                   choices=['periodic', 'replicate', 'reflect', 'zeros'],
                   help='пласт непериодичен, поэтому по умолчанию replicate, '
                        'а не periodic как в оригинале')
    p.add_argument('--local-branch', default='off',
                   choices=['off', 'parallel', 'only'],
                   help='CNO-ветка (свёрточный нейрооператор, FFT-free): '
                        'parallel — параллельно спектральному стеку с '
                        'обучаемым alpha (нулевая инициализация, старт как у '
                        'базовой модели); only — вместо спектрального стека')
    p.add_argument('--cno-levels', type=int, default=2)
    p.add_argument('--cno-blocks', type=int, default=2)
    p.add_argument('--cno-act-up', type=int, default=1,
                   help='1 — обычная GELU (легко); >=2 — точная alias-free '
                        'активация CNO, в 3D стоит act_up^3 объёма активаций')
    p.add_argument('--global-token', default='off',
                   choices=['off', 'post', 'pre', 'both'],
                   help='глобальный агрегат по пласту: маскированное среднее '
                        'скрытых признаков -> MLP -> сдвиг всех позиций. '
                        'post — перед projection, pre — перед спектральными '
                        'блоками, both — оба. Модуль инициализирован нулями, '
                        'поэтому off и любой другой режим стартуют одинаково')
    p.add_argument('--min-window', type=int, default=0,
                   help='0 = авто: max(12, 0.6*T)')
    p.add_argument('--w-l2', type=float, default=1.0)
    p.add_argument('--w-l1', type=float, default=1.0)
    p.add_argument('--w-well', type=float, default=0.5)
    p.add_argument('--well-dilate', type=int, default=0,
                   help='радиус расширения well-маски в лоссе (ячеек); 1-2 '
                        'согласует терм с depression-normalized метрикой')
    p.add_argument('--w-inj-mult', type=float, default=1.0,
                   help='множитель веса нагнетательных скважин в well-терме '
                        '(они стабильно вдвое хуже добывающих)')
    p.add_argument('--aux-rates', action='store_true',
                   help='вспомогательная голова на фактические поточечные '
                        'дебиты (diagnostics/actual_connection_source_terms), '
                        'multi-task; на инференс не влияет')
    p.add_argument('--w-aux', type=float, default=0.5,
                   help='вес aux-терма (при --aux-rates)')
    p.add_argument('--no-adaptive-weights', action='store_true',
                   help='фиксированные веса каналов вместо адаптивных '
                        '(swat стабильно деградировал, пока давление улучшалось)')
    p.add_argument('--w-level', type=float, default=0.0,
                   help='вес терма на систематический сдвиг среднего по маске '
                        'давления (|mean(err)| / |mean(target)|). 0 = выключен, '
                        'прежние прогоны воспроизводятся точно')
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--eval-max-samples', type=int, default=128)
    p.add_argument('--resume', default='', help="'auto' или путь к чекпойнту")
    p.add_argument('--recompute-stats', action='store_true')
    p.add_argument('--material-balance', action='store_true',
                   help='добавить 4 канала материального баланса (накопленный '
                        'отбор и Дарси-драйв, локально и глобально; IN_CH 21 -> 25). '
                        'Считаются только из inputs/* и targets/time_days')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    cfg = dict(
        arch=args.arch,
        material_balance=bool(args.material_balance),
        split_mode=args.split_mode,
        split_seed=int(args.split_seed),
        fault_channels=bool(args.fault_channels),
        hfrac_channels=bool(args.hfrac_channels),
        physics_channels=bool(args.physics_channels),
        global_token=args.global_token,
        local_branch=args.local_branch,
        diff_kernels=args.diff_kernels,
        diff_kernel_size=args.diff_kernel_size,
        diff_padding=args.diff_padding,
        cno_levels=args.cno_levels,
        cno_blocks=args.cno_blocks,
        cno_act_up=args.cno_act_up,
        modes=tuple(int(x) for x in args.modes.split(',')),
        width=args.width, n_layers=args.n_layers, rank=args.rank,
        domain_padding=args.domain_padding, temporal_ssm=args.temporal_ssm,
        d_state=16, d_conv=4, dropout=0.1, residual_u0=True,
    )
    print(f"Architecture: {args.arch}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    # A100: TF32 даёт заметное ускорение matmul/conv без потери качества
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"Device: {device}")

    # ---------- сплиты ----------
    cases = list_cases(args.data)
    if args.split_mode == 'caseid':
        val_sets = {}
        if args.val_geology:
            val_sets['geology_holdout'] = args.val_geology
        if args.val_controls:
            val_sets['controls_holdout'] = args.val_controls
        train_names, val_names = make_splits(cases, val_sets, args.limit_cases)
    else:
        from tnav_data import cases_with_faults, split_random
        all_names = [n for _, n in cases]
        val_names = {}
        if args.split_mode == 'fault-ood':
            faulted = set(cases_with_faults(args.data, all_names))
            rest = [n for n in all_names if n not in faulted]
            if faulted:
                val_names['fault_ood'] = sorted(faulted)
            print(f"Кейсов с разломами (структурный OOD): {len(faulted)}")
        else:
            rest = all_names
        train_names, rnd = split_random(rest, args.val_frac, args.split_seed)
        val_names['random_holdout'] = rnd
        if args.limit_cases:
            train_names = train_names[:args.limit_cases]
            val_names = {k: v[:max(2, args.limit_cases // 4)]
                         for k, v in val_names.items()}
    print(f"Train cases: {len(train_names)}")
    for k, v in val_names.items():
        print(f"Val [{k}]: {len(v)} cases")

    # ---------- нормализация ----------
    stats_cache = os.path.join(args.ckpt_dir, 'norm_stats.pt')
    normalizer = fit_stats(args.data, train_names, cache_path=stats_cache,
                           force=args.recompute_stats,
                           material_balance=args.material_balance,
                        fault=args.fault_channels,
                        hfrac=args.hfrac_channels,
                        physics=args.physics_channels,
                        aux_rates=args.aux_rates)
    N_IN = in_channels(args.material_balance, args.fault_channels,
                        args.hfrac_channels, args.physics_channels)
    cfg['in_channels'] = int(N_IN)
    cfg['aux_out'] = 2 if args.aux_rates else 0
    cfg['material_balance'] = bool(args.material_balance)
    norm_gpu = copy.deepcopy(normalizer).to(device)

    # ---------- датасеты ----------
    train_ds = TnavDataset(args.data, train_names, normalizer)
    T_full, H, W = train_ds.T, train_ds.H, train_ds.W
    print(f"Grid: T={T_full}, H(NY)={H}, W(NX)={W}, IN_CH={N_IN}"
          f"{' (+material balance)' if args.material_balance else ''}")

    min_window = args.min_window or max(12, int(0.6 * T_full))
    min_window = min(min_window, T_full)
    if cfg['modes'][0] > min_window:
        cfg['modes'] = (min_window, cfg['modes'][1], cfg['modes'][2])
        print(f"WARN: modes_t урезаны до окна: modes={cfg['modes']}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
        collate_fn=TnavCollate(min_window, train=True))
    val_loaders = {}
    for k, names in val_names.items():
        if not names:
            continue
        ds = TnavDataset(args.data, names, normalizer)
        val_loaders[k] = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=TnavCollate(min_window, train=False))

    # ---------- модель / лосс / оптимизатор ----------
    model = build_model(cfg, N_IN, C_OUT).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params / 1e6:.2f}M")

    loss_fn = ReservoirLoss(norm_gpu, args.w_l2, args.w_l1, args.w_well,
                            w_level=args.w_level,
                            adaptive=not args.no_adaptive_weights,
                            well_dilate=args.well_dilate,
                            w_inj_mult=args.w_inj_mult,
                            w_aux=args.w_aux if args.aux_rates else 0.0).to(device)
    if args.well_dilate or args.w_inj_mult != 1.0 or args.aux_rates:
        print(f"Well-терм: w_well={args.w_well} dilate={args.well_dilate} "
              f"inj_mult={args.w_inj_mult} | aux={args.aux_rates} w_aux={args.w_aux}"
              f" | adaptive={not args.no_adaptive_weights}")
    if args.w_level > 0:
        print(f'Level-терм включён: w_level={args.w_level}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        p = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        r = args.min_lr_ratio
        return r + 0.5 * (1 - r) * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = ModelEMA(model, args.ema_decay)

    start_epoch = 0
    best_field, best_wells = float('inf'), float('inf')

    # ---------- resume ----------
    resume_path = args.resume
    if resume_path == 'auto':
        cand = os.path.join(args.ckpt_dir, 'last.pt')
        resume_path = cand if os.path.exists(cand) else ''
    if resume_path:
        print(f"Resume from {resume_path}")
        ck = torch.load(resume_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        ema.module.load_state_dict(ck['ema'])
        ema.n_updates = ck.get('ema_n', 100_000)
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        normalizer = NormalizerV6.from_state(ck['normalizer'])
        norm_gpu = copy.deepcopy(normalizer).to(device)
        loss_fn.norm = norm_gpu
        start_epoch = ck['epoch'] + 1
        best_field = ck.get('best_field', best_field)
        best_wells = ck.get('best_wells', best_wells)

    jsonl_path = os.path.join(args.ckpt_dir, 'metrics.jsonl')

    # ---------- цикл ----------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_fn.train()
        t_ep = time.time()
        tr_loss, n_b, skipped = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            x = batch['x'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)
            res = batch['res'].to(device, non_blocking=True)
            well = batch['well'].to(device, non_blocking=True)
            inj = batch['well_inj'].to(device, non_blocking=True)

            if args.aux_rates:
                out, aux_pred = model.forward_with_aux(x)
                aux_tgt = batch['aux'].to(device, non_blocking=True)
                loss, _ = loss_fn(out, y, res, well, inj, aux_pred, aux_tgt)
            else:
                loss, _ = loss_fn(model(x), y, res, well, inj)
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                continue

            (loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema.update(model)

            tr_loss += loss.item()
            n_b += 1

        # ---------- val loss (по всем holdout вместе) ----------
        model.eval()
        loss_fn.eval()
        va_loss, va_n = 0.0, 0
        with torch.no_grad():
            for loader in val_loaders.values():
                for batch in loader:
                    x = batch['x'].to(device, non_blocking=True)
                    y = batch['y'].to(device, non_blocking=True)
                    res = batch['res'].to(device, non_blocking=True)
                    well = batch['well'].to(device, non_blocking=True)
                    loss, _ = loss_fn(model(x), y, res, well)
                    va_loss += loss.item()
                    va_n += 1

        sched.step()
        msg = (f"Epoch {epoch + 1}/{args.epochs} | "
               f"Train: {tr_loss / max(n_b, 1):.4f} | "
               f"Val: {va_loss / max(va_n, 1):.4f} | "
               f"lr: {sched.get_last_lr()[0]:.2e} | {time.time() - t_ep:.0f}s")
        if skipped:
            msg += f" | skipped(NaN): {skipped}"
        print(msg, flush=True)

        # ---------- бизнес-метрики по каждому holdout (EMA-веса) ----------
        all_metrics = None
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            all_metrics = {}
            f_means, w_means = [], []
            for k, loader in val_loaders.items():
                bm = evaluate_business(ema.module, loader, norm_gpu, device,
                                       args.eval_max_samples)
                all_metrics[k] = metrics_to_jsonable(bm)
                f_mean = float(bm['field_rel_l1'].mean())
                f_means.append(f_mean)
                w_means.append(bm['well_pressure_mape'])
                print(f"  [{k}] field relL1: {fmt_pc(bm['field_rel_l1'])} "
                      f"(mean {f_mean:.4f}) | well P-MAPE: "
                      f"{bm['well_pressure_mape']:.3f}% | "
                      f"swat MAE: {bm['swat_mae']:.4f}", flush=True)

            f_score = float(np.mean(f_means)) if f_means else float('inf')
            w_score = float(np.mean(w_means)) if w_means else float('inf')
            with open(jsonl_path, 'a') as jf:
                jf.write(json.dumps({'epoch': epoch + 1,
                                     'train_loss': tr_loss / max(n_b, 1),
                                     'val_loss': va_loss / max(va_n, 1),
                                     **all_metrics}) + '\n')

            if f_score < best_field:
                best_field = f_score
                save_ckpt(os.path.join(args.ckpt_dir, 'best_field.pt'),
                          model, ema, opt, sched, normalizer, cfg, epoch,
                          best_field, best_wells, all_metrics)
                print(f"  ✓ new best FIELD ({best_field:.4f}) saved")
            if w_score < best_wells:
                best_wells = w_score
                save_ckpt(os.path.join(args.ckpt_dir, 'best_wells.pt'),
                          model, ema, opt, sched, normalizer, cfg, epoch,
                          best_field, best_wells, all_metrics)
                print(f"  ✓ new best WELLS ({best_wells:.3f}%) saved")

        save_ckpt(os.path.join(args.ckpt_dir, 'last.pt'),
                  model, ema, opt, sched, normalizer, cfg, epoch,
                  best_field, best_wells, all_metrics)

    print(f"\nDone. best field relL1={best_field:.4f}, "
          f"best well P-MAPE={best_wells:.3f}%")


if __name__ == '__main__':
    main()

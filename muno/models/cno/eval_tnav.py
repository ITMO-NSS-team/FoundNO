# -*- coding: utf-8 -*-
"""
eval_tnav.py — оценка и визуализация одного кейса из датасета v6.

Запуск:
  python eval_tnav.py --data /content/tnav_operator_dataset_2048_v6.h5 \
      --ckpt /content/drive/MyDrive/tnav_fno_ckpts/best_wells.pt \
      --case-id 600 --out pics_case600

Выход в --out:
  pressure.gif / swat.gif      — Prediction | Target | |Error| по времени
  well_pressure_curves.png     — давление pred vs target на скважинных ячейках
  metrics.json                 — метрики кейса
"""

import os
import json
import argparse

import numpy as np
import numpy.ma as ma
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tnav_data import (list_cases, NormalizerV6, predict_case, C_OUT,
                       TARGET_NAMES)
from tnav_model import build_model


def load_checkpoint(path, device, use_ema=True):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    normalizer = NormalizerV6.from_state(ck['normalizer']).to(device)
    from tnav_data import in_channels
    cfg = ck['config']
    n_in = cfg.get('in_channels') or in_channels(
        cfg.get('material_balance', False), cfg.get('fault_channels', False),
        cfg.get('hfrac_channels', False), cfg.get('physics_channels', False))
    model = build_model(ck['config'], n_in, C_OUT).to(device)
    model.load_state_dict(ck['ema'] if (use_ema and 'ema' in ck) else ck['model'])
    model.eval()
    return model, normalizer, ck['config']


# ============================================================================
# МЕТРИКИ КЕЙСА
# ============================================================================

def case_metrics(res):
    pred = res['pred']            # [T,2,H,W] физ.
    tgt = res['target']
    mask = res['mask'].bool()
    well = res['well']            # [T,H,W] открытые completions
    T = pred.shape[0]

    out = {}
    for c, name in enumerate(TARGET_NAMES):
        dp = (pred[:, c] - tgt[:, c]).abs()
        rel = []
        for t in range(T):
            mt = tgt[t, c][mask].mean().abs() + 1e-8
            rel.append(float(dp[t][mask].mean() / mt))
        out[f'{name}_field_rel_l1_mean'] = float(np.mean(rel))
        out[f'{name}_field_rel_l1_per_t'] = rel
        out[f'{name}_mae'] = float(dp.permute(1, 2, 0)[mask].mean())

    dp0 = (pred[:, 0] - tgt[:, 0]).abs()
    wm = well.float()
    if wm.sum() > 0:
        out['well_pressure_mape'] = float(
            (100.0 * dp0 / (tgt[:, 0].abs() + 1e-4) * wm).sum() / wm.sum())
    else:
        out['well_pressure_mape'] = None
    out['n_well_cell_steps'] = int(wm.sum().item())
    return out


# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================

def animate_channel(res, comp, location, name, mode='gif',
                    background_color='#800080'):
    pred_np = res['pred'][:, comp].numpy()
    ref_np = res['target'][:, comp].numpy()
    mask = res['mask'].numpy().astype(bool)
    T = pred_np.shape[0]

    mask_3d = np.broadcast_to(mask[None], pred_np.shape)
    err = np.abs(pred_np - ref_np)

    pred_m = ma.masked_where(~mask_3d, pred_np)
    ref_m = ma.masked_where(~mask_3d, ref_np)
    err_m = ma.masked_where(~mask_3d, err)

    vmin = min(pred_np[mask_3d].min(), ref_np[mask_3d].min())
    vmax = max(pred_np[mask_3d].max(), ref_np[mask_3d].max())
    emax = max(np.percentile(err[mask_3d], 99), 1e-9)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color=background_color)

    from matplotlib import gridspec
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, wspace=0.15)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    for ax, title in zip(axes, ['Prediction', 'Target', '|Error|']):
        ax.set_title(title)
    axes[1].set_yticks([])

    ims = [
        axes[0].imshow(pred_m[0], cmap=cmap, vmin=vmin, vmax=vmax),
        axes[1].imshow(ref_m[0], cmap=cmap, vmin=vmin, vmax=vmax),
        axes[2].imshow(err_m[0], cmap=cmap, vmin=0, vmax=emax),
    ]
    for ax, im in ((axes[1], ims[1]), (axes[2], ims[2])):
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

    tdays = res['time_days'].numpy()
    fig.suptitle(f'{name}, t = {tdays[0]:.0f} d', fontsize=15, fontweight='bold')

    def update(t):
        ims[0].set_array(pred_m[t])
        ims[1].set_array(ref_m[t])
        ims[2].set_array(err_m[t])
        fig.suptitle(f'{name}, t = {tdays[t]:.0f} d',
                     fontsize=15, fontweight='bold')
        return ims

    ani = animation.FuncAnimation(fig, update, frames=T, interval=150)
    os.makedirs(location, exist_ok=True)
    fname = f'{name.lower()}.{ "gif" if mode == "gif" else "mp4"}'.replace(' ', '_')
    if mode == 'gif':
        ani.save(os.path.join(location, fname),
                 writer=animation.PillowWriter(fps=2), dpi=150)
    else:
        ani.save(os.path.join(location, fname), dpi=150)
    plt.close(fig)
    print(f"Saved: {os.path.join(location, fname)}")


def plot_well_pressure_curves(res, location, max_wells=8):
    """Давление pred vs target на ячейках когда-либо открытых completions."""
    pred = res['pred'][:, 0].numpy()
    tgt = res['target'][:, 0].numpy()
    well = res['well'].numpy()
    tdays = res['time_days'].numpy()

    ever = well.any(axis=0)
    cells = np.argwhere(ever)
    if len(cells) == 0:
        print("Нет открытых completions — пропускаю well-кривые")
        return
    if len(cells) > max_wells:
        sel = np.linspace(0, len(cells) - 1, max_wells).astype(int)
        cells = cells[sel]

    n = len(cells)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)
    for k, (i, j) in enumerate(cells):
        ax = axes[k // ncols][k % ncols]
        ax.plot(tdays, tgt[:, i, j], 'k-', lw=2, label='target')
        ax.plot(tdays, pred[:, i, j], 'r--', lw=2, label='pred')
        open_t = well[:, i, j].astype(bool)
        if open_t.any():
            ax.axvspan(tdays[open_t.argmax()], tdays[-1], alpha=0.08, color='g')
        ax.set_title(f'cell ({i},{j})', fontsize=10)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=9)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis('off')
    fig.suptitle('Pressure at well completion cells', fontsize=14)
    fig.tight_layout()
    os.makedirs(location, exist_ok=True)
    path = os.path.join(location, 'well_pressure_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--case-id', type=int, default=-1,
                   help='-1 = первый кейс датасета')
    p.add_argument('--out', default='pics_eval')
    p.add_argument('--use-ema', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, normalizer, cfg = load_checkpoint(args.ckpt, device, args.use_ema)

    cases = dict(list_cases(args.data))            # id -> name
    if args.case_id < 0:
        case_name = next(iter(cases.values()))
    else:
        if args.case_id not in cases:
            raise SystemExit(f"CaseID {args.case_id} нет в датасете")
        case_name = cases[args.case_id]
    print(f"Case: {case_name}")

    res = predict_case(model, normalizer, args.data, case_name, device)
    m = case_metrics(res)

    print("\n===== METRICS =====")
    for c, name in enumerate(TARGET_NAMES):
        print(f"{name:>9s}: field relL1 = {m[f'{name}_field_rel_l1_mean']:.5f}"
              f" | MAE = {m[f'{name}_mae']:.5f}")
    if m['well_pressure_mape'] is not None:
        print(f"well P-MAPE = {m['well_pressure_mape']:.3f}% "
              f"({m['n_well_cell_steps']} cell-steps)")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'metrics.json'), 'w') as jf:
        json.dump(m, jf, indent=2)

    animate_channel(res, 0, args.out, 'Pressure')
    animate_channel(res, 1, args.out, 'SWAT')
    plot_well_pressure_curves(res, args.out)
    print('✅ Done!')


if __name__ == '__main__':
    main()

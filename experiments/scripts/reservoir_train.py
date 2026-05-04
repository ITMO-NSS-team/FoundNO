"""

Seq2Seq forecast: given initial state + injection history, predicts
evolution of pressure and saturation 36 steps ahead.

Configuration via fnofound.data.config.dino_config.DnoDefault:
    python experiments/scripts/reservoir_train.py \\
        --data_path ./data/fno_ready_128x128.hdf5 \\
        --model.in_channels 39 --model.out_channels 72 \\
        --model.n_modes 24 24 --model.hidden_channels 48 \\
        --opt.learning_rate 1e-3 --opt.n_epochs 200 \\
        --vis_every 20

HDF5 format:
    dataset_128 [Time(37), Scenario(N), H(128), W(128), Params(3)]
        Params: 0=Pressure, 1=Saturation, 2=Source
    x_map [N, H, W]
    y_map [N, H, W]

INPUT: 39 channels [X, Y, P0, S0, Src0..Src34]
OUTPUT: 72 channels [P1, S1, P2, S2, ..., P36, S36] (interleaved)
"""

import os
import sys
import json
import argparse
from timeit import default_timer
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import importlib
import importlib.util

sys.path.append('.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# DNO — direct import
_spec = importlib.util.spec_from_file_location(
    'fnofound.models.dno', 'fnofound/models/dno.py',
    submodule_search_locations=[],
)
_dno_mod = importlib.util.module_from_spec(_spec)
sys.modules['fnofound.models.dno'] = _dno_mod
_spec.loader.exec_module(_dno_mod)
DNO = _dno_mod.DNO


# CONFIG
from fnofound.data.config.dno_config import DnoDefault


# LpLoss
class LpLoss:
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        assert p > 0
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
    def rel(self, x, y):
        num = x.size()[0]
        dn = torch.norm(x.reshape(num, -1) - y.reshape(num, -1), self.p, 1)
        yn = torch.norm(y.reshape(num, -1), self.p, 1)
        if self.reduction:
            return torch.mean(dn / yn) if self.size_average else torch.sum(dn / yn)
        return dn / yn
    def __call__(self, x, y, type=False):
        return self.rel(x, y)


# ReservoirDataset 
class ReservoirDataset(Dataset):
    """Seq2Seq dataset from HDF5.

    INPUT (39 channels): [X_map, Y_map, P0, S0, Src0...Src34]
    OUTPUT (72 channels): [P1, S1, P2, S2, ..., P36, S36]
    """
    def __init__(self, filepath, indices, p_scale=100.0, s_scale=1000.0,
                 x_map_key='x_map', y_map_key='y_map',
                 dataset_key='dataset_128'):
        self.filepath = filepath
        self.indices = indices
        self.p_scale = p_scale
        self.s_scale = s_scale
        self.x_map_key = x_map_key
        self.y_map_key = y_map_key
        self.dataset_key = dataset_key
        self._h5_file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        scenario_idx = self.indices[idx]
        if self._h5_file is None:
            self._h5_file = h5py.File(self.filepath, 'r', swmr=True)
        f = self._h5_file

        # [Time(37), H, W, Params(3)]
        data = f[self.dataset_key][:, scenario_idx, :, :, :]
        x_map = f[self.x_map_key][scenario_idx]
        y_map = f[self.y_map_key][scenario_idx]

        # Input: 39 channels
        p0 = data[0, :, :, 0] / self.p_scale
        s0 = data[0, :, :, 1]
        sources = data[:, :, :, 2] / self.s_scale

        x_input = np.concatenate([
            x_map[..., None], y_map[..., None],
            p0[..., None], s0[..., None],
            np.transpose(sources[:35], (1, 2, 0)),
        ], axis=-1).astype(np.float32)

        # Output: 72 channels (P/S interleaved)
        p_seq = data[1:, :, :, 0] / self.p_scale   # [36, H, W]
        s_seq = data[1:, :, :, 1]                    # [36, H, W]
        y_out = np.zeros((128, 128, 72), dtype=np.float32)
        y_out[..., 0::2] = np.transpose(p_seq, (1, 2, 0))
        y_out[..., 1::2] = np.transpose(s_seq, (1, 2, 0))

        return {'x': torch.from_numpy(x_input),
                'y': torch.from_numpy(y_out)}

# Visualization
def visualize_predictions(model, dataset, indices, time_steps, epoch,
                          output_dir, device):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    for s_idx in indices:
        sample = dataset[dataset.indices.index(s_idx)] if hasattr(dataset, 'indices') else dataset
        x = sample['x'].unsqueeze(0).to(device)
        with torch.no_grad():
            y_pred = model(x)

        fig, axes = plt.subplots(2, len(time_steps), figsize=(5*len(time_steps), 8))
        for t_idx, t in enumerate(time_steps):
            # Pressure: even channels
            gt = dataset[dataset.indices.index(s_idx)]['y'].numpy()
            ax = axes[0, t_idx]
            ax.imshow(gt[:, :, 2*t], cmap='viridis')
            ax.set_title(f'P true t={t}')
            ax.axis('off')
            ax = axes[1, t_idx]
            ax.imshow(y_pred[0, :, :, 2*t].cpu().numpy(), cmap='viridis')
            ax.set_title(f'P pred t={t}')
            ax.axis('off')

        plt.suptitle(f'Scenario #{s_idx}, epoch {epoch}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'vis_ep{epoch:03d}_s{s_idx}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

# Weighted loss for Reservoir
def reservoir_loss(pred, target, p_weight=2.0, s_weight=1.0):
    """Weighted relative L2 loss.
    Even channels (0,2,4...) = P, odd channels (1,3,5...) = S.
    """
    loss_fn = LpLoss(p=2, size_average=True)
    total = 0.0
    for c in range(pred.shape[-1]):
        w = p_weight if c % 2 == 0 else s_weight
        total += w * loss_fn(pred[..., c], target[..., c])
    return total


if __name__ == '__main__':
    # ── 1. Base config ──
    cfg = DnoDefault()

    # ── 2. Override defaults for Reservoir ──
    cfg.model.in_channels = 39
    cfg.model.out_channels = 72
    cfg.model.n_modes = [24, 24]
    cfg.model.hidden_channels = 48
    cfg.model.n_layers = 6
    cfg.opt.n_epochs = 200
    cfg.opt.learning_rate = 1e-3

    # ── 3. CLI ──
    parser = argparse.ArgumentParser(description='Reservoir DNO training')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to HDF5 file')
    parser.add_argument('--output_dir', type=str, default='./runs_reservoir')
    parser.add_argument('--vis_every', type=int, default=0)

    parser.add_argument('--model.in_channels', type=int, default=None)
    parser.add_argument('--model.out_channels', type=int, default=None)
    parser.add_argument('--model.n_modes', type=int, nargs=2, default=None)
    parser.add_argument('--model.hidden_channels', type=int, default=None)
    parser.add_argument('--model.n_layers', type=int, default=None)

    parser.add_argument('--data.batch_size', type=int, default=None)
    parser.add_argument('--data.val_split', type=float, default=None)
    parser.add_argument('--opt.n_epochs', type=int, default=None)
    parser.add_argument('--opt.learning_rate', type=float, default=None)
    parser.add_argument('--opt.weight_decay', type=float, default=None)
    parser.add_argument('--opt.step_size', type=int, default=None)
    parser.add_argument('--opt.gamma', type=float, default=None)

    # Reservoir-specific params (not in DnoDefault)
    parser.add_argument('--p_scale', type=float, default=100.0)
    parser.add_argument('--s_scale', type=float, default=1000.0)
    parser.add_argument('--p_weight', type=float, default=2.0)
    parser.add_argument('--s_weight', type=float, default=1.0)
    parser.add_argument('--dataset_key', type=str, default='dataset_128')
    parser.add_argument('--x_map_key', type=str, default='x_map')
    parser.add_argument('--y_map_key', type=str, default='y_map')
    parser.add_argument('--resume', type=str, default='')

    def _apply(cfg, key, val):
        if val is not None:
            parts = key.split('.')
            setattr(getattr(cfg, parts[0]), parts[1], val)

    args = parser.parse_args()
    for key in ['model.in_channels', 'model.out_channels', 'model.n_modes',
                'model.hidden_channels', 'model.n_layers',
                'data.batch_size', 'data.val_split',
                'opt.n_epochs', 'opt.learning_rate', 'opt.weight_decay',
                'opt.step_size', 'opt.gamma']:
        _apply(cfg, key, getattr(args, key.replace('.', '_'), None))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data: {args.data_path}")

    # ── 4. Data ──
    with h5py.File(args.data_path, 'r') as f:
        n_total = f[args.dataset_key].shape[1]
    print(f"Total scenarios: {n_total}")

    n_val = int(n_total * cfg.data.val_split)
    indices = list(range(n_total))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    train_ds = ReservoirDataset(args.data_path, train_idx,
                                p_scale=args.p_scale, s_scale=args.s_scale,
                                dataset_key=args.dataset_key,
                                x_map_key=args.x_map_key,
                                y_map_key=args.y_map_key)
    val_ds = ReservoirDataset(args.data_path, val_idx,
                              p_scale=args.p_scale, s_scale=args.s_scale,
                              dataset_key=args.dataset_key,
                              x_map_key=args.x_map_key,
                              y_map_key=args.y_map_key)

    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # ── 5. Model ──
    model_params = {
        'n_modes': cfg.model.n_modes,
        'hidden_channels': cfg.model.hidden_channels,
        'in_channels': cfg.model.in_channels,
        'out_channels': cfg.model.out_channels,
        'n_layers': cfg.model.n_layers,
    }
    model = DNO(**model_params).to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg.opt.learning_rate,
                            weight_decay=cfg.opt.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt,
                                            step_size=cfg.opt.step_size,
                                            gamma=cfg.opt.gamma)

    ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
    ensure_dir(args.output_dir)
    for d in ['models', 'logs', 'vis']:
        ensure_dir(os.path.join(args.output_dir, d))

    best_val = float('inf')
    path_best = os.path.join(args.output_dir, 'models', 'model_best.pth')
    path_last = os.path.join(args.output_dir, 'models', 'model_last.pth')
    train_losses, val_losses = [], []

    # ── 6. Training loop ──
    t0 = default_timer()
    for ep in range(cfg.opt.n_epochs):
        # Train
        model.train()
        sum_tr = 0.0
        n_tr = 0
        for batch in train_loader:
            x, y = batch['x'].to(device), batch['y'].to(device)
            pred = model(x)
            loss = reservoir_loss(pred, y,
                                  p_weight=args.p_weight,
                                  s_weight=args.s_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            sum_tr += float(loss); n_tr += 1
        train_losses.append(sum_tr / max(1, n_tr))

        # Val
        model.eval()
        sum_val = 0.0
        n_val_b = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch['x'].to(device), batch['y'].to(device)
                pred = model(x)
                loss = reservoir_loss(pred, y,
                                      p_weight=args.p_weight,
                                      s_weight=args.s_weight)
                sum_val += float(loss); n_val_b += 1
        val_losses.append(sum_val / max(1, n_val_b))

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            torch.save(model.state_dict(), path_best)
        torch.save(model.state_dict(), path_last)
        sched.step()

        lr = opt.param_groups[0]['lr']
        print(f"ep={ep:03d} lr={lr:.6f} train={train_losses[-1]:.6f} "
              f"val={val_losses[-1]:.6f} best_val={best_val:.6f}")

        if args.vis_every > 0 and (ep + 1) % args.vis_every == 0:
            visualize_predictions(model, val_ds, val_idx[:3],
                                  time_steps=[10, 20, 36],
                                  epoch=ep + 1,
                                  output_dir=os.path.join(args.output_dir, 'vis'),
                                  device=device)

    dt = default_timer() - t0
    print(f"\nDone in {dt:.1f}s. Best val loss: {best_val:.6f}")

    meta = {
        'model': 'dno',
        'epochs': cfg.opt.n_epochs,
        'batch_size': cfg.data.batch_size,
        'lr': cfg.opt.learning_rate,
        'best_val_loss': float(best_val),
        'final_val_loss': float(val_losses[-1]),
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'time_sec': round(dt, 2),
        'params_count': int(sum(p.numel() for p in model.parameters())),
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    np.savetxt(os.path.join(args.output_dir, 'logs', 'train_loss.csv'),
               np.array(train_losses), delimiter=',')
    np.savetxt(os.path.join(args.output_dir, 'logs', 'val_loss.csv'),
               np.array(val_losses), delimiter=',')
    print(f"Results saved to {args.output_dir}/")

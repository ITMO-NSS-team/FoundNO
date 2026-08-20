"""
airfoil_trainer.py - lightweight training loop for the AirfRANS experiments.

Shared by all four models (FNO2d, RNO2d, DNOAirfoil, FNO2d+padding).

The trainer is case-agnostic: batches come from airfoil_datasets.collate_fn
as dicts {x [B,H,W,C], y [B,H,W,4], mask [B,H,W]|None, grid_mesh [B,H,W,2]|None}.
All models expose a uniform forward(x, mask=None, grid_mesh=None).

Loss: FieldLpLoss (per-field rel-L2 over all four fields incl. nu_t - the
fl4 setup), masked when the case provides a mask, unmasked otherwise.
Reporting: PerFieldLoss (all 4 fields) -> summary.json / leaderboard.

Output layout per run:
    <run_dir>/models/model_best.pth, model_last.pth
    <run_dir>/logs/train_loss.csv, val_loss.csv, summary.json
    <run_dir>/plots/loss.png
"""

import os
import json
import time
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from fnofound.utils.losses import FieldLpLoss, PerFieldLoss


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_csv(path: str, arr):
    np.savetxt(path, np.array(arr, dtype=np.float64), delimiter=',')


def plot_losses(train_l, val_l, path: str):
    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(train_l, label='train')
    plt.plot(val_l, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


class AirfoilTrainer:
    """
    Training loop for the AirfRANS models.

    Parameters
    ----------
    model : nn.Module
        Model with forward(x, mask=None, grid_mesh=None).
    criterion : nn.Module
        Training loss (FieldLpLoss by default), forward(pred, target, mask).
    device : torch.device
    """

    def __init__(self, model, criterion: Optional[torch.nn.Module] = None,
                 device: Optional[torch.device] = None,
                 field_names=('vel_x', 'vel_y', 'pressure', 'nu_t')):
        self.model = model
        self.criterion = criterion if criterion is not None else FieldLpLoss()
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.field_names = field_names
        self.model.to(self.device)

    # helpers

    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _mask_for_loss(self, batch: dict) -> Optional[torch.Tensor]:
        """Loss mask [B, H, W, 1] or None."""
        mask = batch.get('mask')
        if mask is None:
            return None
        return mask.unsqueeze(-1)

    def _forward(self, batch: dict):
        x = batch['x']
        mask = batch.get('mask')
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [B, H, W] -> [B, H, W, 1]
        grid_mesh = batch.get('grid_mesh')
        return self.model(x, mask=mask, grid_mesh=grid_mesh)

    # single pass

    def run_epoch(self, loader, train: bool) -> float:
        """One epoch over the loader. Returns mean loss."""
        self.model.train(train)
        total, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = self._to_device(batch)
                pred = self._forward(batch)
                loss = self.criterion(pred, batch['y'], self._mask_for_loss(batch))
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total += float(loss)
                n += 1
        return total / max(1, n)

    def evaluate(self, loader) -> dict:
        """Validation metrics: {'loss': float, 'per_field': dict}."""
        self.model.eval()
        total, n = 0.0, 0
        per_field = PerFieldLoss(field_names=self.field_names)
        pf_acc = None
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                pred = self._forward(batch)
                loss = self.criterion(pred, batch['y'], self._mask_for_loss(batch))
                total += float(loss)
                n += 1
                pf = per_field(pred, batch['y'], self._mask_for_loss(batch))
                pf_acc = pf if pf_acc is None else {
                    k: pf_acc[k] + v for k, v in pf.items()
                }
        if pf_acc is not None:
            pf_acc = {k: v / n for k, v in pf_acc.items()}
        return {'loss': total / max(1, n), 'per_field': pf_acc}

    # full training

    def fit(self, train_loader, val_loader, epochs: int, lr: float = 1e-3,
            weight_decay: float = 1e-4, step_size: int = 50, gamma: float = 0.5,
            run_dir: str = './runs', log_every: int = 10, seed: int = 42) -> dict:
        """
        Train with AdamW + StepLR, saving best/last checkpoints and logs.

        Returns summary dict {best_val_loss, last_val_loss, epochs, time_sec,
                              params_count, per_field}.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        ensure_dir(run_dir)
        for d in ('models', 'logs', 'plots'):
            ensure_dir(os.path.join(run_dir, d))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)

        path_best = os.path.join(run_dir, 'models', 'model_best.pth')
        path_last = os.path.join(run_dir, 'models', 'model_last.pth')

        train_losses, val_losses = [], []
        best_val = float('inf')
        best_per_field = None
        t0 = time.time()

        for ep in range(epochs):
            tr = self.run_epoch(train_loader, train=True)
            va = self.evaluate(val_loader)
            train_losses.append(tr)
            val_losses.append(va['loss'])
            self.scheduler.step()

            if va['loss'] < best_val:
                best_val = va['loss']
                best_per_field = va['per_field']
                torch.save(self.model.state_dict(), path_best)
            torch.save(self.model.state_dict(), path_last)

            if ep % log_every == 0 or ep == epochs - 1:
                lr_now = self.optimizer.param_groups[0]['lr']
                print(f'  ep={ep:03d} lr={lr_now:.6f} train={tr:.6f} val={va["loss"]:.6f}')

        dt = time.time() - t0
        save_csv(os.path.join(run_dir, 'logs', 'train_loss.csv'), train_losses)
        save_csv(os.path.join(run_dir, 'logs', 'val_loss.csv'), val_losses)
        plot_losses(train_losses, val_losses, os.path.join(run_dir, 'plots', 'loss.png'))

        summary = {
            'best_val_loss': float(best_val),
            'last_val_loss': float(val_losses[-1]),
            'epochs': epochs,
            'time_sec': round(dt, 2),
            'params_count': int(sum(p.numel() for p in self.model.parameters())),
            'per_field': best_per_field,
        }
        with open(os.path.join(run_dir, 'logs', 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        return summary

    # checkpoint helpers

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str, strict: bool = True):
        """Load a state dict, optionally remapping Geo-FNO keys."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if any(k.startswith('blocks.') for k in state):
            from fnofound.models.fno2d import remap_geofno_keys
            state = remap_geofno_keys(state)
        self.model.load_state_dict(state, strict=strict)

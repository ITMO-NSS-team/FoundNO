"""
diff_train.py — DNO training for Darcy flow with arbitrary geometries.

Covers geometries from ~/diff/: heptagon, sq_w_hole, pentagon, sq_art, etc.

Data format (CSV):
    train_C.csv      — conductivity coefficient       [N, 128²]
    train_x_data.csv — X coordinates after diffeomorphism
    train_y_data.csv — Y coordinates
    train_U.csv      — Darcy solution (pressure)

Each sample is a flattened 128×128 vector. NaN = outside domain (hole).
Input:  4 channels [C, X, Y, Mask]
Output: 1 channel  [U * u_scale]

Usage:
    python experiments/scripts/diff_train.py \\
        --data_dir ./data/dno_tasks/diff_heptagon \\
        --model.in_channels 4 --model.out_channels 1 \\
        --data.mask_loss True --data.u_scale 10.0 \\
        --opt.n_epochs 300 --sweep
"""

import os
import sys
import json
import argparse
import itertools
from timeit import default_timer
from datetime import datetime

import numpy as np
import torch
import importlib
import importlib.util

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('.')

# ── DNO — direct load ──
_spec = importlib.util.spec_from_file_location(
    'fnofound.models.dno', 'fnofound/models/dno.py',
    submodule_search_locations=[],
)
_dno_mod = importlib.util.module_from_spec(_spec)
sys.modules['fnofound.models.dno'] = _dno_mod
_spec.loader.exec_module(_dno_mod)
DNO = _dno_mod.DNO

# ── Config ──
from fnofound.data.config.dno_config import DnoDefault


# ── Loss functions ──
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


class MaskedLpLoss:
    """LpLoss with mask — ignores holes (mask=0 = outside domain)."""
    def __init__(self, d=2, p=2):
        self.d = d
        self.p = p
    def abs(self, x, y, mask=None):
        if mask is not None:
            x, y = x * mask, y * mask
        num = torch.sum(mask) if mask is not None else x.numel()
        return torch.norm(x.view(x.shape[0], -1) - y.view(y.shape[0], -1),
                          p=self.p) / num
    def rel(self, x, y, mask=None):
        if mask is not None:
            x, y = x * mask, y * mask
        diff = torch.norm(x.view(x.shape[0], -1) - y.view(y.shape[0], -1), p=self.p)
        norm_y = torch.norm(y.view(y.shape[0], -1), p=self.p)
        return diff / (norm_y + 1e-8)


# ── Data loading ──
def load_darcy_data(data_dir, resolution=128, train_ratio=0.9,
                    u_scale=10.0, mask_loss=True):
    """Load Darcy CSV data. NaN → mask (0=hole). U *= u_scale."""
    S = resolution
    print(f"  Loading Darcy data from {data_dir}...")

    raw_c = np.loadtxt(os.path.join(data_dir, 'train_C.csv'), delimiter=',')
    raw_x = np.loadtxt(os.path.join(data_dir, 'train_x_data.csv'), delimiter=',')
    raw_y = np.loadtxt(os.path.join(data_dir, 'train_y_data.csv'), delimiter=',')
    raw_u = np.loadtxt(os.path.join(data_dir, 'train_U.csv'), delimiter=',')

    mask = (~np.isnan(raw_c)).astype(np.float32)
    c = np.nan_to_num(raw_c, nan=0.0).reshape(-1, S, S)
    x = np.nan_to_num(raw_x, nan=0.0).reshape(-1, S, S)
    y = np.nan_to_num(raw_y, nan=0.0).reshape(-1, S, S)
    m = mask.reshape(-1, S, S)
    u = np.nan_to_num(raw_u, nan=0.0).reshape(-1, S, S)

    num = c.shape[0]
    inputs  = np.stack([c, x, y, m], axis=-1)
    targets = (u * u_scale)[..., None]

    ntrain = int(num * train_ratio)
    return (inputs[:ntrain], targets[:ntrain],
            inputs[ntrain:], targets[ntrain:], mask)


# ── Utilities ──
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_csv(path, arr):
    np.savetxt(path, np.array(arr, dtype=np.float64), delimiter=',')

def plot_losses(train_l, test_l, path):
    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(train_l, label='train')
    plt.plot(test_l, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()


# ── Training loop ──
def train_one_run(run_dir, device, train_a, train_u, test_a, test_u,
                  model_class, model_params, cfg, seed=42, log_every=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ensure_dir(run_dir)
    for d in ['models', 'plots', 'logs']:
        ensure_dir(os.path.join(run_dir, d))

    bs = cfg.data.batch_size
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.FloatTensor(train_a), torch.FloatTensor(train_u)),
        batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.FloatTensor(test_a), torch.FloatTensor(test_u)),
        batch_size=bs, shuffle=False)

    model = model_class(**model_params).to(device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg.opt.learning_rate,
                            weight_decay=cfg.opt.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt,
                                            step_size=cfg.opt.step_size,
                                            gamma=cfg.opt.gamma)

    loss_fn = MaskedLpLoss() if cfg.data.mask_loss else LpLoss(size_average=True)

    train_losses, test_losses = [], []
    best_test = float('inf')
    path_best = os.path.join(run_dir, 'models', 'model_best.pth')
    path_last = os.path.join(run_dir, 'models', 'model_last.pth')

    t0 = default_timer()
    for ep in range(cfg.opt.n_epochs):
        model.train()
        sum_tr, n_tr = 0.0, 0
        for xx, yy in train_loader:
            xx, yy = xx.to(device), yy.to(device)
            pred = model(xx)
            if cfg.data.mask_loss:
                mask = xx[..., 3:4].to(device)
                loss = loss_fn.rel(pred, yy, mask)
            else:
                loss = loss_fn(pred, yy)
            opt.zero_grad(); loss.backward(); opt.step()
            sum_tr += float(loss); n_tr += 1
        train_losses.append(sum_tr / max(1, n_tr))

        model.eval()
        sum_te, n_te = 0.0, 0
        with torch.no_grad():
            for xx, yy in test_loader:
                xx, yy = xx.to(device), yy.to(device)
                pred = model(xx)
                if cfg.data.mask_loss:
                    mask = xx[..., 3:4].to(device)
                    loss = loss_fn.rel(pred, yy, mask)
                else:
                    loss = loss_fn(pred, yy)
                sum_te += float(loss); n_te += 1
        test_losses.append(sum_te / max(1, n_te))

        if test_losses[-1] < best_test:
            best_test = test_losses[-1]
            torch.save(model.state_dict(), path_best)
        torch.save(model.state_dict(), path_last)
        sched.step()

        if ep % log_every == 0:
            lr = opt.param_groups[0]['lr']
            print(f"  ep={ep:03d} lr={lr:.6f} train={train_losses[-1]:.6f} test={test_losses[-1]:.6f}")

    dt = default_timer() - t0
    save_csv(os.path.join(run_dir, 'logs', 'train_loss.csv'), train_losses)
    save_csv(os.path.join(run_dir, 'logs', 'test_loss.csv'), test_losses)
    plot_losses(train_losses, test_losses, os.path.join(run_dir, 'plots', 'loss.png'))

    summary = {
        'best_test_loss': float(best_test),
        'last_test_loss': float(test_losses[-1]),
        'epochs': cfg.opt.n_epochs,
        'time_sec': round(dt, 2),
        'params_count': int(sum(p.numel() for p in model.parameters())),
    }
    with open(os.path.join(run_dir, 'logs', 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


# ── Main ──
if __name__ == '__main__':
    cfg = DnoDefault()
    # Defaults are already for Darcy: in=4, out=1, mask_loss=True, u_scale=10.0

    parser = argparse.ArgumentParser(description='Darcy DNO training')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./runs_diff')

    # Model params
    parser.add_argument('--model.in_channels', type=int, default=None)
    parser.add_argument('--model.out_channels', type=int, default=None)
    parser.add_argument('--model.n_modes', type=int, nargs=2, default=None)
    parser.add_argument('--model.hidden_channels', type=int, default=None)
    parser.add_argument('--model.n_layers', type=int, default=None)
    # Data params
    parser.add_argument('--data.batch_size', type=int, default=None)
    parser.add_argument('--data.u_scale', type=float, default=None)
    parser.add_argument('--data.mask_loss', type=lambda x: x.lower() == 'true',
                        default=None)
    parser.add_argument('--data.train_resolution', type=int, default=None)
    # Opt params
    parser.add_argument('--opt.n_epochs', type=int, default=None)
    parser.add_argument('--opt.learning_rate', type=float, default=None)
    parser.add_argument('--opt.weight_decay', type=float, default=None)
    parser.add_argument('--opt.step_size', type=int, default=None)
    parser.add_argument('--opt.gamma', type=float, default=None)

    def _apply_override(cfg, key, val):
        if val is not None:
            parts = key.split('.')
            setattr(getattr(cfg, parts[0]), parts[1], val)

    args = parser.parse_args()
    for key in ['model.in_channels', 'model.out_channels', 'model.n_modes',
                'model.hidden_channels', 'model.n_layers',
                'data.batch_size', 'data.u_scale', 'data.mask_loss',
                'data.train_resolution',
                'opt.n_epochs', 'opt.learning_rate', 'opt.weight_decay',
                'opt.step_size', 'opt.gamma']:
        _apply_override(cfg, key, getattr(args, key.replace('.', '_'), None))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")

    train_a, train_u, test_a, test_u, _ = load_darcy_data(
        data_dir=args.data_dir,
        resolution=cfg.data.train_resolution,
        train_ratio=1.0 - cfg.data.val_split,
        u_scale=cfg.data.u_scale,
        mask_loss=cfg.data.mask_loss,
    )

    ensure_dir(args.output_dir)
    meta = {
        'u_scale': cfg.data.u_scale,
        'mask_loss': cfg.data.mask_loss,
        'resolution': cfg.data.train_resolution,
        'n_train': train_a.shape[0],
        'n_test': test_a.shape[0],
    }
    with open(os.path.join(args.output_dir, 'data_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    model_params = {
        'n_modes': cfg.model.n_modes,
        'hidden_channels': cfg.model.hidden_channels,
        'in_channels': cfg.model.in_channels,
        'out_channels': cfg.model.out_channels,
        'n_layers': cfg.model.n_layers,
        # Darcy layout: [C, X, Y, Mask] → geometry channels are 1 and 2
        'geometry_channels': (1, 2),
    }

    if args.sweep:
        sweep_grid = {
            'batch_size': [8, 16, 32],
            'learning_rate': [1e-3, 5e-4],
            'modes': [12, 16, 24],
            'width': [16, 32, 48],
        }
        existing = {d.name for d in os.scandir(args.output_dir) if d.is_dir()}
        results = []
        for combo in itertools.product(*sweep_grid.values()):
            d = dict(zip(sweep_grid.keys(), combo))
            lr_str = f"{d['learning_rate']:.10f}".rstrip('0').rstrip('.')
            name = f"bs{d['batch_size']}_m{d['modes']}_w{d['width']}_lr{lr_str}"
            if name in existing:
                print(f"SKIP: {name}")
                continue
            mp = model_params.copy()
            mp['n_modes'] = [d['modes'], d['modes']]
            mp['hidden_channels'] = d['width']
            cfg.data.batch_size = d['batch_size']
            cfg.opt.learning_rate = d['learning_rate']

            print(f"\n=== {name} ===")
            s = train_one_run(
                run_dir=os.path.join(args.output_dir, name),
                device=device, cfg=cfg,
                train_a=train_a, train_u=train_u,
                test_a=test_a, test_u=test_u,
                model_class=DNO, model_params=mp)
            results.append({'run': name, **d, **s})
        results.sort(key=lambda x: x['best_test_loss'])
        with open(os.path.join(args.output_dir, 'leaderboard.json'), 'w') as f:
            json.dump(results, f, indent=2)
        if results:
            print(f"\nBest: {results[0]['run']} loss={results[0]['best_test_loss']:.6f}")
    else:
        now = datetime.now()
        name = f"{cfg.arch}_{now.day}_{now.hour}_{now.minute}"
        print(f"\n=== {name} ===")
        s = train_one_run(
            run_dir=os.path.join(args.output_dir, name),
            device=device, cfg=cfg,
            train_a=train_a, train_u=train_u,
            test_a=test_a, test_u=test_u,
            model_class=DNO, model_params=model_params)
        print(json.dumps(s, indent=2))

"""
Obstacle training — Flow over a Obstacle (step) with DNO using Navier-Stokes eq.

Loads CSV data (x_data, y_data, u_data, v_data, p_data, re_data),
normalises with StandardScaler, and trains a DNO model.

Configuration via fnofound.data.config.dino_config.DnoDefault:
    python experiments/scripts/obstacle_train.py \\
        --data_dir ./data/obstacle \\
        --model.in_channels 3 --model.out_channels 3 \\
        --model.n_modes 24 24 --model.hidden_channels 32 \\
        --opt.learning_rate 1e-3 --opt.n_epochs 200 \\
        --sweep 

Pipeline:
    mesh.msh → diffeomorphism maps (x_data.csv, y_data.csv)
             → Firedrake NS solver (u, v, p, re CSVs)
             → this script → trained DNO model.pth
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
import torch.nn as nn
import importlib
import importlib.util

sys.path.append('.')

# ── DNO — direct load (bypasses models/__init__.py) ──
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


# ── LpLoss (from obstacle/Loss_function.py) ──
class LpLoss:
    """Relative/Absolute Lp loss."""
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


# ── Data utilities ──
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_csv(path, arr):
    np.savetxt(path, np.array(arr, dtype=np.float64), delimiter=',')


class StandardScaler:
    """Z-score normaliser (ignores zeros = former NaN)."""
    def __init__(self, data):
        valid = data[data != 0]
        if len(valid) == 0:
            valid = data
        self.mean = float(np.mean(valid))
        self.std = float(np.std(valid))
        if self.std == 0:
            self.std = 1.0

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


def load_fluid_data(data_dir, resolution=128, train_ratio=0.9):
    """Load Navier-Stokes CSV data and apply Z-score normalisation."""
    def _load(name):
        print(f"  Loading {name}...")
        return np.loadtxt(os.path.join(data_dir, name), delimiter=',')

    raw_x, raw_y = _load('x_data.csv'), _load('y_data.csv')
    raw_u, raw_v = _load('u_data.csv'), _load('v_data.csv')
    raw_p, raw_re = _load('p_data.csv'), _load('re_data.csv')

    num = raw_u.shape[0]
    S = resolution
    raw_x, raw_y = raw_x[:num], raw_y[:num]

    X  = np.nan_to_num(raw_x, nan=0.0)
    Y  = np.nan_to_num(raw_y, nan=0.0)
    Re = np.nan_to_num(raw_re, nan=0.0)
    U  = np.nan_to_num(raw_u, nan=0.0)
    V  = np.nan_to_num(raw_v, nan=0.0)
    P  = np.nan_to_num(raw_p, nan=0.0)
    if Re.ndim == 1:
        Re = Re.reshape(-1, 1)
    Re = np.repeat(Re, S * S, axis=1)

    scalers = {k: StandardScaler(v) for k, v in
               zip(['x','y','re','u','v','p'], [X, Y, Re, U, V, P])}
    X  = scalers['x'].encode(X).reshape(num, S, S)
    Y  = scalers['y'].encode(Y).reshape(num, S, S)
    Re = scalers['re'].encode(Re).reshape(num, S, S)
    U  = scalers['u'].encode(U).reshape(num, S, S)
    V  = scalers['v'].encode(V).reshape(num, S, S)
    P  = scalers['p'].encode(P).reshape(num, S, S)

    inputs  = np.stack([X, Y, Re], axis=-1)
    targets = np.stack([U, V, P], axis=-1)

    ntrain = int(num * train_ratio)
    return (inputs[:ntrain], targets[:ntrain],
            inputs[ntrain:], targets[ntrain:], scalers)


def plot_losses(train_l, test_l, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(train_l, label='train')
    plt.plot(test_l, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


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
    loss_fn = LpLoss(size_average=True)

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
            loss = (loss_fn(pred[..., 0], yy[..., 0]) +
                    loss_fn(pred[..., 1], yy[..., 1]) +
                    loss_fn(pred[..., 2], yy[..., 2]))
            opt.zero_grad(); loss.backward(); opt.step()
            sum_tr += float(loss); n_tr += 1
        train_losses.append(sum_tr / max(1, n_tr))

        model.eval()
        sum_te, n_te = 0.0, 0
        with torch.no_grad():
            for xx, yy in test_loader:
                xx, yy = xx.to(device), yy.to(device)
                pred = model(xx)
                loss = (loss_fn(pred[..., 0], yy[..., 0]) +
                        loss_fn(pred[..., 1], yy[..., 1]) +
                        loss_fn(pred[..., 2], yy[..., 2]))
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
    # Override defaults for fluid (obstacle) case
    cfg.model.in_channels = 3
    cfg.model.out_channels = 3
    cfg.model.n_modes = [24, 24]
    cfg.model.hidden_channels = 32
    cfg.model.n_layers = 6
    cfg.data.case_type = 'fluid'
    cfg.data.mask_loss = False

    parser = argparse.ArgumentParser(description='Obstacle DNO training')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./runs_obstacle')

    parser.add_argument('--model.in_channels', type=int, default=None)
    parser.add_argument('--model.out_channels', type=int, default=None)
    parser.add_argument('--model.n_modes', type=int, nargs=2, default=None)
    parser.add_argument('--model.hidden_channels', type=int, default=None)
    parser.add_argument('--model.n_layers', type=int, default=None)
    parser.add_argument('--data.batch_size', type=int, default=None)
    parser.add_argument('--data.train_resolution', type=int, default=None)
    parser.add_argument('--opt.n_epochs', type=int, default=None)
    parser.add_argument('--opt.learning_rate', type=float, default=None)
    parser.add_argument('--opt.weight_decay', type=float, default=None)
    parser.add_argument('--opt.step_size', type=int, default=None)
    parser.add_argument('--opt.gamma', type=float, default=None)

    def _apply(cfg, key, val):
        if val is not None:
            parts = key.split('.')
            setattr(getattr(cfg, parts[0]), parts[1], val)

    args = parser.parse_args()
    for key in ['model.in_channels', 'model.out_channels', 'model.n_modes',
                'model.hidden_channels', 'model.n_layers',
                'data.batch_size', 'data.train_resolution',
                'opt.n_epochs', 'opt.learning_rate', 'opt.weight_decay',
                'opt.step_size', 'opt.gamma']:
        _apply(cfg, key, getattr(args, key.replace('.', '_'), None))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")

    train_a, train_u, test_a, test_u, scalers = load_fluid_data(
        args.data_dir, resolution=cfg.data.train_resolution, train_ratio=0.9)

    ensure_dir(args.output_dir)
    scalers_dict = {k: {'mean': float(v.mean), 'std': float(v.std)}
                    for k, v in scalers.items()}
    with open(os.path.join(args.output_dir, 'scalers_info.json'), 'w') as f:
        json.dump(scalers_dict, f, indent=2)

    model_params = {
        'n_modes': cfg.model.n_modes,
        'hidden_channels': cfg.model.hidden_channels,
        'in_channels': cfg.model.in_channels,
        'out_channels': cfg.model.out_channels,
        'n_layers': cfg.model.n_layers,
    }

    if args.sweep:
        sweep_grid = {
            'batch_size': [16, 32],
            'learning_rate': [1e-3, 5e-4],
            'modes': [24, 32],
            'width': [32, 48],
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

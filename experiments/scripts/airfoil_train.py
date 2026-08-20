#!/usr/bin/env python3
"""
airfoil_train.py - unified training for FNO / RNO / DNO / Geo-FNO on AirfRANS.

One entry point, four architectures: the --model argument selects the model
class, the dataset case and the per-case defaults via MODEL_REGISTRY.

    python experiments/scripts/airfoil_train.py --model fno
    python experiments/scripts/airfoil_train.py --model dno --opt.n_epochs 200
    python experiments/scripts/airfoil_train.py --model geofno --model.padding 8
    python experiments/scripts/airfoil_train.py --model rno --sweep
    python experiments/scripts/airfoil_train.py --model dno \\
        --checkpoint /media/.../runs_dno_small/dno_m16_w64_l4_b4/model_best.pth

Data (see fnofound.data.data.datasets.airfoil_datasets):
    fno/rno:   {data.root_dir}/fno_dataset          (fno_data_*.npz, 128x256)
    dno:       {data.root_dir}/DNO_data/dno_small   (batch_N/*.npz, 256x256)
    geofno:    {data.root_dir}/Geo-FNO_data         (batch_N/*.pth, C-grid)

Loss: FieldLpLoss over all four fields (vel_x, vel_y, pressure, nu_t) -
the fl4 setup matching the airrans runs_fl4, masked where available.
Output: runs_airfoils/<model>_<ts>/{models, logs, plots}
"""

import argparse
import itertools
import json
import os
import sys
from datetime import datetime

sys.path.append('.')

import torch
from torch.utils.data import DataLoader

from fnofound.data.config.airfoil_config import AirfoilDefault
from fnofound.data.data.datasets.airfoil_datasets import (
    get_airfoil_dataset,
    collate_fn,
)
from fnofound.models import FNO2d, RNO2d, DNOAirfoil
from fnofound.utils.airfoil_trainer import AirfoilTrainer
from fnofound.utils.losses import FieldLpLoss


# Per-model defaults: model class, dataset case, input channels, padding.
# NOTE: n_modes/hidden_channels/n_layers come from config (default [16,16]/32/4).
MODEL_REGISTRY = {
    'fno':    {'cls': FNO2d,      'dataset': 'fno',    'in_channels': 3, 'padding': 0},
    'rno':    {'cls': RNO2d,      'dataset': 'fno',    'in_channels': 3, 'padding': 0},
    'dno':    {'cls': DNOAirfoil, 'dataset': 'dno',    'in_channels': 3, 'padding': 0},
    'geofno': {'cls': FNO2d,      'dataset': 'geofno', 'in_channels': 2, 'padding': 8},
}

CASE_DIRS = {
    'fno':    'fno_dataset',
    'dno':    'DNO_data/dno_small',
    'geofno': 'Geo-FNO_data',
}


def build_model(cfg: AirfoilDefault, entry: dict, modes: int) -> torch.nn.Module:
    cls = entry['cls']
    if cls is DNOAirfoil:
        return cls(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            modes=modes,
            width=cfg.model.hidden_channels,
            n_layers=cfg.model.n_layers,
            use_grid=cfg.model.use_grid,
            use_geom=True,
            padding=cfg.model.padding,
        )
    return cls(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        modes=modes,
        width=cfg.model.hidden_channels,
        n_layers=cfg.model.n_layers,
        use_grid=cfg.model.use_grid,
        padding=cfg.model.padding,
    )


def main():
    parser = argparse.ArgumentParser(description='AirfRANS unified training.')
    parser.add_argument('--model', type=str, required=True,
                        choices=sorted(MODEL_REGISTRY),
                        help='Architecture to train')
    parser.add_argument('--sweep', action='store_true',
                        help='Grid search over modes/width/layers/batch_size')
    parser.add_argument('--output_dir', type=str, default='./runs_airfoils')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Pretrained model_best.pth to load (Geo-FNO keys auto-remapped)')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--limit', type=int, default=0,
                        help='Use only the first N train samples (for quick tests)')

    # dotted config overrides, e.g. --data.batch_size 4 --opt.n_epochs 200
    parser.add_argument('--data.root_dir', type=str, default=None)
    parser.add_argument('--data.batch_size', type=int, default=None)
    parser.add_argument('--data.num_workers', type=int, default=None)
    parser.add_argument('--model.n_modes', type=int, nargs='+', default=None)
    parser.add_argument('--model.hidden_channels', type=int, default=None)
    parser.add_argument('--model.n_layers', type=int, default=None)
    parser.add_argument('--model.in_channels', type=int, default=None)
    parser.add_argument('--model.padding', type=int, default=None)
    parser.add_argument('--opt.n_epochs', type=int, default=None)
    parser.add_argument('--opt.learning_rate', type=float, default=None)
    parser.add_argument('--opt.weight_decay', type=float, default=None)
    parser.add_argument('--opt.step_size', type=int, default=None)
    parser.add_argument('--opt.gamma', type=float, default=None)

    args = parser.parse_args()

    # config + per-model defaults
    entry = MODEL_REGISTRY[args.model]
    cfg = AirfoilDefault()
    cfg.data.case_type = entry['dataset']
    cfg.model.in_channels = entry['in_channels']
    cfg.model.padding = entry['padding']

    overrides = {
        'data.root_dir': getattr(args, 'data.root_dir', None),
        'data.batch_size': getattr(args, 'data.batch_size', None),
        'data.num_workers': getattr(args, 'data.num_workers', None),
        'model.hidden_channels': getattr(args, 'model.hidden_channels', None),
        'model.n_layers': getattr(args, 'model.n_layers', None),
        'model.in_channels': getattr(args, 'model.in_channels', None),
        'model.padding': getattr(args, 'model.padding', None),
        'opt.n_epochs': getattr(args, 'opt.n_epochs', None),
        'opt.learning_rate': getattr(args, 'opt.learning_rate', None),
        'opt.weight_decay': getattr(args, 'opt.weight_decay', None),
        'opt.step_size': getattr(args, 'opt.step_size', None),
        'opt.gamma': getattr(args, 'opt.gamma', None),
    }
    for key, val in overrides.items():
        if val is not None:
            parts = key.split('.')
            setattr(getattr(cfg, parts[0]), parts[1], val)

    n_modes = getattr(args, 'model.n_modes', None) or cfg.model.n_modes
    if len(n_modes) == 1:
        n_modes = [n_modes[0], n_modes[0]]

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Model: {args.model} | device: {device} | case: {cfg.data.case_type}')

    data_root = os.path.join(cfg.data.root_dir, CASE_DIRS[cfg.data.case_type])

    # data
    train_ds = get_airfoil_dataset(cfg.data.case_type, data_root, split='train')
    val_ds = get_airfoil_dataset(cfg.data.case_type, data_root, split='val')
    test_ds = get_airfoil_dataset(cfg.data.case_type, data_root, split='test')
    if args.limit:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(args.limit, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(args.limit // 4, len(val_ds)))))
        print(f'[limit] train={len(train_ds)}, val={len(val_ds)}')

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=shuffle,
                          num_workers=cfg.data.num_workers,
                          collate_fn=collate_fn)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)
    print(f'Data: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} '
          f'| bs={cfg.data.batch_size}')

    # sweep grid
    if args.sweep:
        sweep_grid = {
            'modes': [8, 16],
            'width': [32, 64],
            'layers': [4],
            'batch_size': [4, 8],
        }
        os.makedirs(args.output_dir, exist_ok=True)
        existing = {d.name for d in os.scandir(args.output_dir) if d.is_dir()}
        results = []
        for combo in itertools.product(*sweep_grid.values()):
            d = dict(zip(sweep_grid.keys(), combo))
            name = f"{args.model}_m{d['modes']}_w{d['width']}_l{d['layers']}_b{d['batch_size']}"
            if name in existing:
                print(f'SKIP: {name}')
                continue
            run_dir = os.path.join(args.output_dir, name)
            model = build_model(cfg, entry, d['modes'])
            model._hparams = d  # for summary
            trainer = AirfoilTrainer(model, device=device)
            cfg.data.batch_size = d['batch_size']
            s = trainer.fit(train_loader, val_loader,
                            epochs=cfg.opt.n_epochs,
                            lr=cfg.opt.learning_rate,
                            weight_decay=cfg.opt.weight_decay,
                            step_size=cfg.opt.step_size,
                            gamma=cfg.opt.gamma,
                            run_dir=run_dir)
            test_metrics = trainer.evaluate(test_loader)
            s['test_loss'] = test_metrics['loss']
            s['test_per_field'] = test_metrics['per_field']
            results.append({'run': name, **d, **s})
        results.sort(key=lambda x: x['best_val_loss'])
        with open(os.path.join(args.output_dir, 'leaderboard.json'), 'w') as f:
            json.dump(results, f, indent=2)
        if results:
            print(f"\nBest: {results[0]['run']} val={results[0]['best_val_loss']:.6f}")
    else:
        run_name = f"{args.model}_{datetime.now().strftime('%d_%H_%M')}"
        run_dir = os.path.join(args.output_dir, run_name)
        model = build_model(cfg, entry, n_modes[0])
        trainer = AirfoilTrainer(model, device=device)

        if args.checkpoint:
            trainer.load(args.checkpoint)
            print(f'Loaded checkpoint: {args.checkpoint}')

        print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
        s = trainer.fit(train_loader, val_loader,
                        epochs=cfg.opt.n_epochs,
                        lr=cfg.opt.learning_rate,
                        weight_decay=cfg.opt.weight_decay,
                        step_size=cfg.opt.step_size,
                        gamma=cfg.opt.gamma,
                        run_dir=run_dir)
        test_metrics = trainer.evaluate(test_loader)
        s['test_loss'] = test_metrics['loss']
        s['test_per_field'] = test_metrics['per_field']
        with open(os.path.join(run_dir, 'logs', 'summary.json'), 'w') as f:
            json.dump(s, f, indent=2)
        print(f'\nDone: {run_dir}')
        print(f'val={s["best_val_loss"]:.6f} | test={s["test_loss"]:.6f}')
        print(f'test per-field: {s["test_per_field"]}')


if __name__ == '__main__':
    main()

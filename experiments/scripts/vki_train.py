#!/usr/bin/env python3
"""
vki_train.py - unified training for FNO / RNO / DNO / RNO_D on VKI-LS59.

One entry point, four architectures: the --model argument selects the model
class, the dataset case and the per-case defaults via MODEL_REGISTRY.

    python experiments/scripts/vki_train.py --model fno
    python experiments/scripts/vki_train.py --model dno --opt.n_epochs 200
    python experiments/scripts/vki_train.py --model rno_d --model.n_modes 24
    python experiments/scripts/vki_train.py --model fno --sweep

Data (see fnofound.data.data.datasets.vki_datasets):
    fno/rno:   raw C-grid 301x121 (plaid)        -> {data.plaid_dir}
    dno/rno_d: universal square 128x128 (npy)    -> {data.root_dir}/DNO_dataset

Input per sample: [H, W, 5] = (x, y, sdf) + (angle_in, mach_out) broadcast
as constant channels; output [H, W, 6] = (mach, nut, ro, roe, rou, rov).
DNO additionally receives grid_mesh (physical x, y) for the geometry terms.

Loss: FieldLpLoss over all six fields (nut included - same decision as the
airfoils fl4 setup), computed in PHYSICAL space (no normalization), with
optional per-field weights (--loss.weights, e.g. mach weight 1.5).
Output: runs_vki/<model>_<ts>/{models, logs, plots}

Splits: train = train_500 (500), val = 171. The test split (168) has NO
outputs in the dataset (as published), so it is not used here - see
vki_eval.py for val metrics + test inference.
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

from fnofound.data.config.vki_config import VkiDefault
from fnofound.data.data.datasets.vki_datasets import (
    get_vki_dataset,
    collate_fn,
    OUT_FIELDS,
)
from fnofound.models import FNO2d, RNO2d, DNOAirfoil
from fnofound.utils.airfoil_trainer import AirfoilTrainer
from fnofound.utils.losses import FieldLpLoss


# Per-model defaults: model class, dataset case, input channels, padding,
# DNO geometry flag. n_modes/hidden_channels/n_layers come from config.
MODEL_REGISTRY = {
    'fno':   {'cls': FNO2d,      'dataset': 'raw',    'in_channels': 5, 'padding': 8,  'use_geom': False},
    'rno':   {'cls': RNO2d,      'dataset': 'raw',    'in_channels': 5, 'padding': 8,  'use_geom': False},
    'dno':   {'cls': DNOAirfoil, 'dataset': 'square', 'in_channels': 5, 'padding': 0,  'use_geom': True},
    'rno_d': {'cls': RNO2d,      'dataset': 'square', 'in_channels': 5, 'padding': 8,  'use_geom': False},
}

CASE_DIRS = {
    'raw':    None,            # data root = plaid_dir
    'square': 'DNO_dataset',
}


def build_model(cfg: VkiDefault, entry: dict, modes: int) -> torch.nn.Module:
    cls = entry['cls']
    if cls is DNOAirfoil:
        return cls(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            modes=modes,
            width=cfg.model.hidden_channels,
            n_layers=cfg.model.n_layers,
            use_grid=cfg.model.use_grid,
            use_geom=entry['use_geom'],
            padding=entry['padding'],
        )
    return cls(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        modes=modes,
        width=cfg.model.hidden_channels,
        n_layers=cfg.model.n_layers,
        use_grid=cfg.model.use_grid,
        padding=entry['padding'],
    )


def main():
    parser = argparse.ArgumentParser(description='VKI-LS59 unified training.')
    parser.add_argument('--model', type=str, required=True,
                        choices=sorted(MODEL_REGISTRY),
                        help='Architecture to train')
    parser.add_argument('--sweep', action='store_true',
                        help='Grid search over modes/width/layers/batch_size')
    parser.add_argument('--output_dir', type=str, default='./runs_vki')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Pretrained model_best.pth to load')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--limit', type=int, default=0,
                        help='Use only the first N train samples (quick tests)')

    # dotted config overrides, e.g. --data.batch_size 4 --opt.n_epochs 200
    parser.add_argument('--data.root_dir', type=str, default=None)
    parser.add_argument('--data.plaid_dir', type=str, default=None)
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
    parser.add_argument('--loss.field_indices', type=int, nargs='+', default=None)
    parser.add_argument('--loss.weights', type=float, nargs='+', default=None)

    args = parser.parse_args()

    # config + per-model defaults
    entry = MODEL_REGISTRY[args.model]
    cfg = VkiDefault()
    cfg.data.case_type = entry['dataset']
    cfg.model.in_channels = entry['in_channels']
    cfg.model.padding = entry['padding']

    overrides = {
        'data.root_dir': getattr(args, 'data.root_dir', None),
        'data.plaid_dir': getattr(args, 'data.plaid_dir', None),
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
        'loss.field_indices': getattr(args, 'loss.field_indices', None),
        'loss.weights': getattr(args, 'loss.weights', None),
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

    # data
    train_ds = get_vki_dataset(cfg.data.case_type, cfg.data.root_dir,
                               plaid_dir=cfg.data.plaid_dir, split='train')
    val_ds = get_vki_dataset(cfg.data.case_type, cfg.data.root_dir,
                             plaid_dir=cfg.data.plaid_dir, split='val')
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
    print(f'Data: train={len(train_ds)} val={len(val_ds)} | bs={cfg.data.batch_size}')

    # loss: per-field rel-L2 over the selected channels (nut included),
    # physical space, optional per-channel weights
    criterion = FieldLpLoss(field_indices=tuple(cfg.loss.field_indices),
                            weights=tuple(cfg.loss.weights))

    # sweep grid
    if args.sweep:
        sweep_grid = {
            'modes': [8, 24],
            'width': [64],
            'layers': [5],
            'batch_size': [4],
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
            trainer = AirfoilTrainer(model, criterion=criterion, device=device,
                                     field_names=OUT_FIELDS)
            cfg.data.batch_size = d['batch_size']
            s = trainer.fit(train_loader, val_loader,
                            epochs=cfg.opt.n_epochs,
                            lr=cfg.opt.learning_rate,
                            weight_decay=cfg.opt.weight_decay,
                            step_size=cfg.opt.step_size,
                            gamma=cfg.opt.gamma,
                            run_dir=run_dir)
            s['val_per_field'] = s.get('per_field')
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
        trainer = AirfoilTrainer(model, criterion=criterion, device=device,
                                 field_names=OUT_FIELDS)

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
        with open(os.path.join(run_dir, 'logs', 'summary.json'), 'w') as f:
            json.dump(s, f, indent=2)
        print(f'\nDone: {run_dir}')
        print(f'val={s["best_val_loss"]:.6f}')
        print(f'val per-field: {s["per_field"]}')


if __name__ == '__main__':
    main()

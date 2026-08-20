#!/usr/bin/env python3
"""
vki_eval.py - evaluate trained FNO / RNO / DNO / RNO_D models on VKI-LS59.

Two parts:

  val metrics  - per-field rel-L2 (epsilon_s) on the val split (171 samples,
                 the only split WITH ground truth; the published test split
                 has NO outputs - see below). Written to leaderboard.json.
  test inference - predict on the 168-sample test split (inputs only, as
                 published on HuggingFace) and save the raw predictions to
                 runs_vki/test_pred/<model>_pred.npy [N, H, W, 6].

Checkpoints: standard layout runs_vki/<model>_*/models/model_best.pth; the
architecture (kind/hyperparameters) is inferred from each state dict, with
per-model padding/use_geom defaults from VKI_MODEL_DEFAULTS (padding is not
visible in a state dict).

Usage:
    python experiments/scripts/vki_eval.py \\
        --data.root_dir /media/.../VKI-LS59 \\
        --out runs_vki/leaderboard.json

    python experiments/scripts/vki_eval.py --models fno dno \\
        --checkpoint fno=/path/to/model_best.pth --no-test
"""

import argparse
import glob
import json
import os
import sys

sys.path.append('.')

import numpy as np
import torch
from torch.utils.data import DataLoader

from airfoil_eval import infer_arch        # same directory

from fnofound.data.data.datasets.vki_datasets import (
    get_vki_dataset,
    collate_fn,
    OUT_FIELDS,
)
from fnofound.models import FNO2d, RNO2d, DNOAirfoil
from fnofound.utils.airfoil_trainer import AirfoilTrainer
from fnofound.utils.losses import FieldLpLoss

DATA_ROOT_DEFAULT = '/media/expiwt/f2d8c27a-bdac-4400-b1be-cdb75b6b5a40/3sem/hugging_face/VKI-LS59'
PLAID_DIR_DEFAULT = '/media/expiwt/f2d8c27a-bdac-4400-b1be-cdb75b6b5a40/hugging_face/VKI-LS59/plaid_dataset'

# per-model dataset case + eval defaults (padding/use_geom are NOT in the
# state dict, so they must be supplied here to match training)
VKI_MODELS = {
    'fno':   {'case': 'raw',    'padding': 8,  'use_geom': False},
    'rno':   {'case': 'raw',    'padding': 8,  'use_geom': False},
    'dno':   {'case': 'square', 'padding': 0,  'use_geom': True},
    'rno_d': {'case': 'square', 'padding': 8,  'use_geom': False},
}


def build_model(arch: dict, out_channels: int = 6) -> torch.nn.Module:
    kind = arch['kind']
    if kind == 'rno':
        return RNO2d(in_channels=arch['in_channels'], out_channels=out_channels,
                     modes=arch['modes'], width=arch['width'],
                     n_layers=arch['n_layers'], use_grid=arch['use_grid'],
                     padding=arch['padding'])
    if kind == 'dno':
        return DNOAirfoil(in_channels=arch['in_channels'], out_channels=out_channels,
                          modes=arch['modes'], width=arch['width'],
                          n_layers=arch['n_layers'], use_grid=arch['use_grid'],
                          use_geom=arch['use_geom'], padding=arch['padding'])
    return FNO2d(in_channels=arch['in_channels'], out_channels=out_channels,
                 modes=arch['modes'], width=arch['width'],
                 n_layers=arch['n_layers'], use_grid=arch['use_grid'],
                 padding=arch['padding'])


def find_checkpoint(model: str, output_dir: str) -> str:
    """Best checkpoint in runs_vki/<model>_*/ (lowest best_val_loss)."""
    runs = sorted(glob.glob(os.path.join(output_dir, f'{model}_*')))
    best_path, best_val = None, float('inf')
    for run in runs:
        p = os.path.join(run, 'models', 'model_best.pth')
        if not os.path.exists(p):
            continue
        s = os.path.join(run, 'logs', 'summary.json')
        val = float('inf')
        if os.path.exists(s):
            val = json.load(open(s)).get('best_val_loss', float('inf'))
        if val < best_val:
            best_val, best_path = val, p
    return best_path


def main():
    parser = argparse.ArgumentParser(description='VKI-LS59 evaluation.')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['fno', 'rno', 'dno', 'rno_d'],
                        choices=sorted(VKI_MODELS))
    parser.add_argument('--checkpoint', action='append', default=[],
                        metavar='MODEL=PATH')
    parser.add_argument('--data.root_dir', type=str, default=DATA_ROOT_DEFAULT)
    parser.add_argument('--data.plaid_dir', type=str, default=PLAID_DIR_DEFAULT)
    parser.add_argument('--output_dir', type=str, default='./runs_vki',
                        help='run outputs (default checkpoint location)')
    parser.add_argument('--out', type=str, default='./runs_vki/leaderboard.json')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--no-test', action='store_true',
                        help='skip test-split inference')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    root = getattr(args, 'data.root_dir', DATA_ROOT_DEFAULT)
    plaid = getattr(args, 'data.plaid_dir', PLAID_DIR_DEFAULT)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'device: {device}')

    overrides = dict(kv.split('=', 1) for kv in args.checkpoint)

    results = {}
    for model_name in args.models:
        entry = VKI_MODELS[model_name]
        ckpt_path = overrides.get(model_name) or find_checkpoint(model_name, args.output_dir)
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f'[skip] {model_name}: no checkpoint in {args.output_dir} '
                  f'(use --checkpoint {model_name}=/path/to/model_best.pth)')
            continue

        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        arch = infer_arch(state)
        arch['padding'] = entry['padding']
        arch['use_geom'] = entry['use_geom']
        model = build_model(arch)
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f'[{model_name}] {arch} | params={params:,} | {ckpt_path}')

        # ── val metrics (the only split with ground truth) ──
        ds = get_vki_dataset(entry['case'], root, plaid_dir=plaid, split='val')
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)
        criterion = FieldLpLoss(field_indices=tuple(range(len(OUT_FIELDS))))
        trainer = AirfoilTrainer(model, criterion=criterion, device=device,
                                 field_names=OUT_FIELDS)
        val = trainer.evaluate(loader)

        results[model_name] = {
            'checkpoint': ckpt_path,
            'arch': arch,
            'params_count': int(params),
            'val_loss': val['loss'],
            'val_per_field': val['per_field'],
        }
        pf = ' | '.join(f'{f}={val["per_field"][f]:.4f}' for f in OUT_FIELDS)
        print(f'  val: {pf}')

        # ── test inference (no answers in the dataset) ──
        if not args.no_test:
            ds_test = get_vki_dataset(entry['case'], root, plaid_dir=plaid,
                                      split='test', want_y=False)
            loader_test = DataLoader(ds_test, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers,
                                     collate_fn=collate_fn)
            preds = []
            with torch.no_grad():
                for batch in loader_test:
                    x = batch['x'].to(device)
                    grid_mesh = batch['grid_mesh']
                    if grid_mesh is not None:
                        grid_mesh = grid_mesh.to(device)
                    preds.append(model(x, mask=None, grid_mesh=grid_mesh).cpu().numpy())
            preds = np.concatenate(preds, axis=0)          # [N, H, W, 6]
            out_dir = os.path.join(args.output_dir, 'test_pred')
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, f'{model_name}_pred.npy'), preds)
            print(f'  test: predictions saved -> {out_dir}/{model_name}_pred.npy '
                  f'{preds.shape}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {args.out}')

    if results:
        print(f'\n=== Val leaderboard (epsilon_s per field) ===')
        print(f'{"model":<8} {"loss":<8} ' + ' '.join(f'{f:<8}' for f in OUT_FIELDS))
        for name in sorted(results, key=lambda n: results[n]['val_loss']):
            r = results[name]
            pf = ' '.join(f'{r["val_per_field"][f]:<8.4f}' for f in OUT_FIELDS)
            print(f'{name:<8} {r["val_loss"]:<8.4f} {pf}')


if __name__ == '__main__':
    main()

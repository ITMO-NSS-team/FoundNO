# VKI-LS59 Experiments (Turbine Cascade)

Unified pipeline for training and comparing four neural operators on the
[VKI-LS59](https://huggingface.co/datasets/PLAID-datasets/VKI-LS59) dataset
(Safran, CC-BY-SA-4.0): **FNO**, **RNO**, **DNO**, **RNO_D**.

2D internal aerodynamics, a VKI-LS59 turbine blade cascade. Steady
compressible RANS, Spalart-Allmaras, Broadcast solver. All quantities are
dimensionless.

## Layout

```
experiments/vki/
├── README.md
└── data_prep/
    └── prep_square.py        # offline prep (run once): mesh -> square 128x128
                              #   (harmonic map + P2 biquadratic) -> DNO_dataset

fnofound/
├── models/                   # reused as-is: fno2d.py, rno.py, dno_airfoil.py
│                             #   (scalars are broadcast by the dataset, so no
│                             #    scalar head is needed in the models)
├── utils/
│   ├── square_dev.py         # mesh -> square mapping (boundary loop, sides,
│   │                         #   harmonic map, structured (i,j)) - ported
│   ├── losses.py             # FieldLpLoss (train), PerFieldLoss (report)
│   └── airfoil_trainer.py    # shared training loop (standard checkpoint layout)
├── data/config/vki_config.py              # VkiDefault (zencfg)
└── data/data/datasets/vki_datasets.py     # RawGridDataset + SquareDataset

experiments/scripts/
├── vki_train.py             # unified training: --model fno|rno|dno|rno_d
└── vki_eval.py              # val metrics (leaderboard) + test inference
```

## Models and grids

| Model  | Class        | Dataset case | Grid        | padding | Notes |
|--------|--------------|--------------|-------------|---------|-------|
| fno    | FNO2d        | raw          | 301x121     | 8       | structured C-grid, u non-periodic |
| rno    | RNO2d        | raw          | 301x121     | 8       | Riesz Neural Operator |
| dno    | DNOAirfoil   | square       | 128x128     | 0       | + c(grid_mesh) geometry terms |
| rno_d  | RNO2d        | square       | 128x128     | 8       | RNO on the universal square |

Geo-FNO is not a separate model: the deformation is already baked into the
data (harmonic map) and the v axis of the square is periodic, so Geo-FNO
degenerates into FNO (same conclusion as the VKI workspace README).

## Data

Dataset source: <https://huggingface.co/datasets/PLAID-datasets/VKI-LS59>
(owner Safran, license CC-BY-SA-4.0).

Raw dataset and the prepared square live on the external drive (not committed):

| Data | Path |
|------|------|
| PLAID raw (Base_2_2 meshes) | `/media/expiwt/f2d8c27a-bdac-4400-b1be-cdb75b6b5a40/hugging_face/VKI-LS59/plaid_dataset` |
| Square (DNO_dataset, npy)   | `/media/expiwt/f2d8c27a-bdac-4400-b1be-cdb75b6b5a40/3sem/hugging_face/VKI-LS59/DNO_dataset` |

Input per sample (raw and square): `[H, W, 5]` = (x, y, sdf) + the scalar
conditions (angle_in, mach_out) broadcast by the dataset as constant channels.
Output: `[H, W, 6]` = (mach, nut, ro, roe, rou, rov).

Splits (defined by the dataset itself, see `meta.json` / `split.json`):

| Split | Samples | Outputs |
|-------|---------|---------|
| train (train_500) | 500 | yes |
| val (rest of train) | 171 | yes |
| test | 168 | **no** (as published - "outputs are not provided") |

The test split is prediction-only; the val split is used for the leaderboard.

## Data preparation (once)

```bash
python -m experiments.vki.data_prep.prep_square \
    --plaid_dir /media/.../VKI-LS59/plaid_dataset \
    --out /media/.../VKI-LS59/DNO_dataset \
    --grid 128 128 --jobs 1
```

~10 min sequentially (`--jobs 1`, no OOM; `--jobs > 1` loads the dataset once
per worker). Debug a few samples without saving: `--ids 0 1 2`. The pipeline
(harmonic map + P2 biquadratic interpolation, Newton inversion) was verified
in the VKI workspace (`verify_interp.py`): reproduction of quadratic fields
~1e-11, inversion residual ~4e-7, no overshoots on transonic samples.

## Training

```bash
python experiments/scripts/vki_train.py --model fno      # raw 301x121
python experiments/scripts/vki_train.py --model rno
python experiments/scripts/vki_train.py --model dno      # square 128x128
python experiments/scripts/vki_train.py --model rno_d
python experiments/scripts/vki_train.py --model fno --sweep
```

Per-model defaults come from the registry in `vki_train.py`. Optional dotted
overrides: `--data.root_dir`, `--data.plaid_dir`, `--model.n_modes`,
`--opt.n_epochs`, `--loss.weights` (e.g. `--loss.weights 1.5 1 1 1 1 1` for a
larger mach weight, as in the m24 draft runs), etc.

Loss: `FieldLpLoss` over all six fields (nut included - same decision as the
airfoils fl4 setup), computed in PHYSICAL space (no normalization, no
norm.json - per-field relative L2 is scale-invariant). Note: the draft VKI
runs used z-score + log1p(nut) + Charbonnier; switching to FieldLpLoss in
physical space changes training dynamics, so the numbers may differ from the
draft `runs/summary`.

Output per run: `runs_vki/<model>_<ts>/{models, logs, plots}` (standard
checkpoint layout: `models/model_best.pth`, `logs/summary.json`, `loss.png`).

## Evaluation

```bash
python experiments/scripts/vki_eval.py \
    --data.root_dir /media/.../VKI-LS59 \
    --out runs_vki/leaderboard.json
```

- **val metrics**: per-field rel-L2 (epsilon_s) on the 171-sample val split
  (the only split with ground truth) -> `runs_vki/leaderboard.json`;
- **test inference**: predictions for the 168-sample test split (no answers)
  saved to `runs_vki/test_pred/<model>_pred.npy` [N, H, W, 6];
- architecture is inferred from each state dict (`infer_arch`), with
  per-model padding/use_geom defaults from the eval registry;
- checkpoints are picked as the best `runs_vki/<model>_*/models/model_best.pth`
  or given explicitly: `--checkpoint fno=/path/to/model_best.pth`.

## Loading pre-trained models (draft VKI runs)

The draft runs in the VKI workspace (`VKI-LS59/runs/*/best.pt`) are loadable:
the state dict layout matches (fc0 in = 7 = 3 fields + 2 scalars + 2 grid in
the draft, = 5 channels + 2 grid here), so `best.pt` can be used with
`vki_eval.py` after extracting the `model` key:

```python
sd = torch.load('runs/fno_m8_w64_l5_b4/best.pt', weights_only=False)['model']
torch.save(sd, 'model_best.pth')
```

## Running on your own machine

1. **Dependencies** - the same set the airfoils experiments already need:
   torch, numpy, scipy, pandas, matplotlib, zencfg, einops, transformers,
   neuralop. Plus `plaid` (pip install plaid-lib) - ONLY for the `raw` case
   and for `prep_square.py`; the `square` case works without it.
2. **Data** - `DNO_dataset` (square) and/or `plaid_dataset` (raw) on your own
   disk (see the table above; `prep_square.py` generates the square from the
   raw dataset).
3. **Paths** - two places, both overridable from the CLI:
   - `fnofound/data/config/vki_config.py`: `root_dir` (square) and `plaid_dir`
     (raw) - or `--data.root_dir` / `--data.plaid_dir` at launch;
   - `prep_square.py`: `--plaid_dir` / `--out` (only if you generate the
     dataset yourself).
4. **Run:**
   ```bash
   python experiments/scripts/vki_train.py --model fno|rno|dno|rno_d
   # -> runs_vki/<model>_<ts>/{models, logs, plots}
   python experiments/scripts/vki_eval.py
   # -> runs_vki/leaderboard.json + runs_vki/test_pred/<model>_pred.npy
   ```

## Environment

Python 3.10 venv at `~/my_jupyter_project/venv` (torch 2.10+cu128, numpy,
scipy, pandas, zencfg). The `plaid` package (read the raw PLAID dataset) is
needed ONLY for the `raw` case and for `prep_square.py` - install with
`pip install plaid-lib` (already present for python3.10 at
`~/.local/lib/python3.10/site-packages`). GPU: RTX 2070 SUPER 8 GB.

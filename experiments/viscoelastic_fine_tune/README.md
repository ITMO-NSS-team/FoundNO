# Full FoundNO fine-tuning on viscoelastic_instability_v2

This directory contains the training code used in the November 2026 local experiment. The model starts from the original pretrained `cfno_9_11/core.pt`, with every network parameter unfrozen. A checkpoint previously fine-tuned on Euler is not used for initialization.

## Data and prediction task

- Preserve the official The Well `train`, `valid`, and `test` directories: **210 / 22 / 22 trajectories** in the downloaded v2 release. No trajectories are re-split or filtered.
- Take the first **20 consecutive stored frames**, indices `0..19`, from every trajectory. The input is frame 0 repeated across the time axis; the output contains all 20 frames, including reconstruction of frame 0 and prediction of 19 future frames.
- Area-average native `512x512` fields to `128x128` in a separate cache. Original HDF5 files remain unchanged.
- Physical channels: `c_zz`, `pressure`, `velocity_x`, `velocity_y`, `C_xx`, `C_xy`, `C_yx`, `C_yy`. Both stored off-diagonal tensor components are retained.
- Fit channel mean/std on train only, using the same normalization for training and validation.
- Add actual elapsed physical time as an input channel, scaled by the maximum train horizon. Stored time intervals are approximately 1 in AH/CAR/EIT and 4 in the transition regimes; frame number alone does not encode this difference.

Only the split follows The Well here. This is an initial-condition-to-space-time-cube task, not the four-input-frames-to-one-output-frame autoregressive benchmark protocol.

## Model and training

The pretrained core is loaded intact. The compatible Mamba lifting tensors, its grid-coordinate input columns, and the hidden projection layer are transferred from the original adapters. Input weights for the new physical fields and the final eight-channel output layer are newly initialized. All **193,030,123 parameters** have `requires_grad=True`; the original architecture also contains unused channel-MLP skip parameters that do not participate in its forward pass.

The core uses the pinned source modules under `cfno_upstream/`; their provenance is recorded in `cfno_upstream/SOURCE.txt`. Activation checkpointing recomputes intermediate activations without changing the forward operations. `ChunkedAdamW` applies ordinary FP32 AdamW to storage-sharing parameter chunks, including real/imaginary views of Fourier weights.

The experiment uses batch size 1, 200 epochs, core LR `1e-4`, adapter LR `1e-3`, weight decay `1e-4`, and gradient clipping at 1. The loss is normalized MSE over all 20 frames. `ReduceLROnPlateau` monitors validation MSE, with patience 12 and factor 0.5; early stopping is disabled. The scheduler has minimum learning rates `1e-7` for the core and `1e-6` for the adapters.

VRMSE is computed per trajectory, channel, and time as

```text
sqrt(mean_xy((prediction - truth)^2) / (var_xy(truth, unbiased=True) + 1e-7))
```

Reported future VRMSE averages these values over channels, trajectories, and frames `1..19`. `best.pt` is selected by validation MSE; `best_vrmse.pt` is selected by validation future VRMSE. Test is evaluated after training. GIF generation can also evaluate the validation-selected checkpoint on the complete held-out test split after an interrupted run.

## Environment

The verified local runtime is Ubuntu 24.04 under WSL2, Python 3.10, PyTorch `2.4.0+cu121`, `mamba-ssm==2.2.2`, and `causal-conv1d==1.4.0`, with an RTX 4080 SUPER (16 GiB). This code requires real CUDA Mamba; there is no SimpleSSM fallback. The remaining package versions are in `requirements.txt`.

Use a matching CUDA/PyTorch/Mamba environment, then install the Python dependencies:

```bash
cd experiments/viscoelastic_fine_tune
python -m pip install -r requirements.txt
```

## Files to provide

Place the dataset under `data/viscoelastic_instability_v2/`, containing `data/train`, `data/valid`, `data/test`, and the original HDF5 files. The dataset is available from [The Well](https://polymathic-ai.org/the_well/datasets/viscoelastic_instability_v2/).

Place the original pretrained `core.pt`, `lift_0.pt`, and `proj_0.pt` under `pretrained/cfno_9_11/`. Model weights and datasets are not committed to this repository.

Extract the compatible lifting tensor state once:

```bash
python extract_cfno_lifting_tensors.py \
  --source pretrained/cfno_9_11/lift_0.pt \
  --output pretrained/cfno_9_11/lifting_tensors.pth
```

The extractor handles legacy dill references in an isolated process and saves tensor state only. It never executes the unavailable legacy kernel helpers. Only load the trusted original checkpoint files.

## Run and resume

The portable configuration is `cfgs/foundno_viscoelastic_v2_first20_full.yaml`. Paths are relative to this experiment directory; edit them to use existing datasets and checkpoints. The `.wsl.yaml` configuration and `.wsl.sh` launcher retain the exact paths from the original local experiment for provenance. That launcher targets the original working directory on Nikita's machine; use the portable launcher below from a fresh clone.

```bash
# Materialize caches and fit train-only normalization, without loading the model.
python train_foundno_viscoelastic.py \
  --config cfgs/foundno_viscoelastic_v2_first20_full.yaml --prepare-only

# Three optimizer steps to verify gradients, core updates, memory, and timing.
# These updated weights are discarded; subsequent training starts from pretraining.
bash run_foundno_viscoelastic_training.sh --smoke-only

# Start full fine-tuning.
bash run_foundno_viscoelastic_training.sh

# Resume from the last completed epoch, including optimizer and scheduler state.
bash run_foundno_viscoelastic_training.sh \
  --resume runs/viscoelastic_v2_first20_full/latest.pt
```

Set `PYTHON=/path/to/venv/bin/python` when using a different interpreter. An existing training history requires `--resume` or a new output directory. Resume repeats an unfinished epoch from its last saved boundary; RNG states are not restored, so bitwise identity with an uninterrupted run is not promised.

Outputs include `latest.pt`, `best.pt`, `best_vrmse.pt`, `history.jsonl`, `metrics.csv`, `training_curves.png`, per-epoch validation metrics, split manifests, and normalization/model provenance. Each full epoch took approximately 4 to 4.5 minutes on the local GPU.

## True / Predict / error GIFs

```bash
python make_viscoelastic_gifs.py \
  --checkpoint runs/viscoelastic_v2_first20_full/best_vrmse.pt \
  --output gifs/viscoelastic_v2_best
```

This selects one trajectory per regime in each split with seed 42, without filtering by prediction error. It creates 21 GIFs with all eight physical fields, three overview GIFs for `velocity_y`, previews, an HTML gallery, and cached true/prediction arrays. Every GIF has 20 frames with fixed True/Predict color scales. Validation predictions are checked against the stored per-channel, per-time metrics for that checkpoint.

The image percentage is `100*abs(pred-true)/max(abs(true),epsilon)`, where `epsilon=max(1% P99(abs(true)),1e-7)` per field and trajectory. Error colors saturate at 100%; numerical metrics are not clipped. This pointwise percentage is different from VRMSE.

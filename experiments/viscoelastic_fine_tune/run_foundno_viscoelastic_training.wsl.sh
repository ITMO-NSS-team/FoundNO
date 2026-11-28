#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Nikita/Documents/GitHub/NeuralOil-FoundNO-euler-test/FoundNO
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7
exec /home/nikita/venvs/cfno_euler/bin/python -u train_foundno_viscoelastic.py \
  --config cfgs/foundno_viscoelastic_v2_first20_full.wsl.yaml "$@"

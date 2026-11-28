#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.7}"
exec "${PYTHON:-python}" -u train_foundno_viscoelastic.py \
  --config cfgs/foundno_viscoelastic_v2_first20_full.yaml "$@"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
  echoheart/train/train.py \
  --config-file "${CONFIG_FILE:-echoheart/configs/train/structure_aware_vits_pretrain.yaml}" \
  --output-dir "${OUTPUT_DIR:-./checkpoints/echoheart_stage1}" \
  "$@"

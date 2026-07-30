#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONFIG=${CONFIG:-configs/training/demo_full_finetune.yaml}
CHECKPOINT=${CHECKPOINT:-checkpoints/rhinovla_pretrain.ckpt}

if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "missing RhinoVLA pretrained checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${TRACKERS:-}" ]]; then
  EXTRA_ARGS+=("trackers=${TRACKERS}")
fi
if [[ -n "${RUN_ID:-}" ]]; then
  EXTRA_ARGS+=("run_id=${RUN_ID}")
fi
if [[ -n "${RUN_ROOT:-}" ]]; then
  EXTRA_ARGS+=("run_root_dir=${RUN_ROOT}")
fi

exec "${PYTHON:-python}" -m rhinovla.training.train \
  --config_yaml "${CONFIG}" \
  trainer.pretrained_checkpoint="${CHECKPOINT}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

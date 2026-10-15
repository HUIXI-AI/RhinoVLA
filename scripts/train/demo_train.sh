#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONFIG=${CONFIG:-configs/training/demo_ae_finetune.yaml}
CHECKPOINT=${CHECKPOINT:-checkpoints/rhinovla_pretrain.ckpt}

# Optional SwanLab reporting:
#   SWANLAB_API_KEY=... SWANLAB_PROJECT=rhinovla-finetune SWANLAB_WEB_HOST=http://your-swanlab-host:8000 TRACKERS='[jsonl,swanlab]' ./scripts/train/demo_train.sh
#   SWANLAB_API_KEY=... SWANLAB_PROJECT=rhinovla-finetune TRACKERS='[jsonl,swanlab]' ./scripts/train/demo_train.sh   # SwanLab online

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "missing ${CHECKPOINT}" >&2
  echo "download the V3 weights from https://huggingface.co/HuixiAI/RhinoVLA-V3 and set CHECKPOINT to the downloaded file" >&2
  exit 1
fi

MIN_CHECKPOINT_BYTES=${MIN_CHECKPOINT_BYTES:-1048576}
CHECKPOINT_BYTES=$(wc -c < "${CHECKPOINT}")
if (( CHECKPOINT_BYTES < MIN_CHECKPOINT_BYTES )); then
  echo "${CHECKPOINT} is too small to be a full checkpoint" >&2
  echo "download the V3 weights from https://huggingface.co/HuixiAI/RhinoVLA-V3 and set CHECKPOINT to the downloaded file" >&2
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
  "${EXTRA_ARGS[@]}" "$@"

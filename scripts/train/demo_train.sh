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
  echo "download it from Hugging Face with: hf download HuixiAI/RhinoVLA --include rhinovla_pretrain.ckpt --local-dir checkpoints" >&2
  exit 1
fi

MIN_CHECKPOINT_BYTES=${MIN_CHECKPOINT_BYTES:-1048576}
CHECKPOINT_BYTES=$(wc -c < "${CHECKPOINT}")
if (( CHECKPOINT_BYTES < MIN_CHECKPOINT_BYTES )); then
  echo "${CHECKPOINT} is too small to be a full checkpoint" >&2
  echo "download it from Hugging Face with: hf download HuixiAI/RhinoVLA --include rhinovla_pretrain.ckpt --local-dir checkpoints" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${TRACKERS:-}" ]]; then
  EXTRA_ARGS+=("trackers=${TRACKERS}")
fi

exec "${PYTHON:-python}" -m rhinovla.training.train --config_yaml "${CONFIG}" "${EXTRA_ARGS[@]}" "$@"

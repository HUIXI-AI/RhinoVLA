#!/usr/bin/env bash
# Single-frame RhinoVLA inference through rpu_backend public API.
#
# Required environment:
#   PREPARE_ARTIFACT=/path/to/rhinovla_prepare_artifact.pt
#   CONFIG=/path/to/training_config.yaml
#   CHECKPOINT=/path/to/steps_<n>_pytorch_model.pt
#   NORM_STATS=/path/to/norm.json
#   INSTRUCTION='pick up the object'
#   STATE='[raw,state,values]'   # JSON list, comma list, or path to JSON array
#   IMAGES='/path/head.png /path/left.png /path/right.png'
#
# Optional:
#   MAPPING=/path/to/native72_mapping.yaml
#   MAPPING_DATASET_ID=dataset_id
#   RPU_ENV_FILE=/path/to/rhinovla_runtime_env.toml
#   RPU_ENV='KEY=VALUE KEY2=VALUE2'
#   RPU_RHINO_REPO=/path/to/rhinovla

set -eu

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-python}

PREPARE_ARTIFACT=${PREPARE_ARTIFACT:?set PREPARE_ARTIFACT}
CONFIG=${CONFIG:?set CONFIG}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT}
NORM_STATS=${NORM_STATS:?set NORM_STATS}
INSTRUCTION=${INSTRUCTION:?set INSTRUCTION}
STATE=${STATE:?set STATE}
IMAGES=${IMAGES:?set IMAGES}
OUT=${OUT:-./rpu_backend_infer_output.json}
NUM_STEPS=${NUM_STEPS:-5}
ACTION_HZ=${ACTION_HZ:-30}
NOISE_SEED=${NOISE_SEED:-0}
VIEW_ROLES=${VIEW_ROLES:-top_head,hand_left,hand_right}
VIEW_MODALITIES=${VIEW_MODALITIES:-rgb,rgb,rgb}

cmd=(
  "$PY" -m rhinovla.inference.rpu_backend.cli
  --prepare-artifact "$PREPARE_ARTIFACT"
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --norm-stats "$NORM_STATS"
  --instruction "$INSTRUCTION"
  --state "$STATE"
  --output "$OUT"
  --num-steps "$NUM_STEPS"
  --action-hz "$ACTION_HZ"
  --noise-seed "$NOISE_SEED"
  --view-roles "$VIEW_ROLES"
  --view-modalities "$VIEW_MODALITIES"
)

for image in $IMAGES; do
  cmd+=(--image "$image")
done

if [[ -n "${ACTIVE_SLOTS:-}" ]]; then
  cmd+=(--active-slots "$ACTIVE_SLOTS")
fi
if [[ -n "${MAPPING:-}" ]]; then
  cmd+=(--mapping "$MAPPING")
fi
if [[ -n "${MAPPING_DATASET_ID:-}" ]]; then
  cmd+=(--mapping-dataset-id "$MAPPING_DATASET_ID")
fi
if [[ -n "${RPU_RHINO_REPO:-}" ]]; then
  cmd+=(--rpu-rhino-repo "$RPU_RHINO_REPO")
fi
if [[ -n "${RPU_ENV_FILE:-}" ]]; then
  cmd+=(--rpu-env-file "$RPU_ENV_FILE")
fi
if [[ -n "${RPU_ENV:-}" ]]; then
  for kv in $RPU_ENV; do
    cmd+=(--rpu-env "$kv")
  done
fi
if [[ "${RPU_ARTIFACT_STRICT:-0}" == "1" ]]; then
  cmd+=(--rpu-artifact-strict)
fi

PYTHONPATH="$ROOT:${PYTHONPATH:-}" exec "${cmd[@]}"


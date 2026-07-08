#!/usr/bin/env bash
# RhinoVLA environment setup via uv (https://github.com/astral-ng/uv).
#
# Creates a local .venv and installs the CUDA-matched PyTorch stack plus the
# RhinoVLA package and its dependencies.
#
# Usage:
#   bash scripts/setup_env.sh                 # default: py3.10 + torch 2.10 cu128
#   CUDA=cu128 bash scripts/setup_env.sh      # explicit CUDA 12.8 wheel set
#   WITH_FLASH=1 WITH_LOGGING=1 bash scripts/setup_env.sh   # optional extras
#
# After it finishes:  source .venv/bin/activate

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO}"

PYTHON_VERSION=${PYTHON_VERSION:-3.10}
TORCH_VERSION=${TORCH_VERSION:-2.10.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.25.0}
TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-5.3.0}
CUDA=${CUDA:-cu128}
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA}"
WITH_FLASH=${WITH_FLASH:-0}
WITH_LOGGING=${WITH_LOGGING:-0}

CUDA_NUM=${CUDA#cu}
if [[ ! "${CUDA_NUM}" =~ ^[0-9]+$ ]] || (( CUDA_NUM < 128 )); then
  echo "[setup] CUDA must be cu128 or newer; got ${CUDA}" >&2
  exit 1
fi

# 1. Install uv if it is not already on PATH.
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found; installing to ~/.local/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
echo "[setup] uv: $(uv --version)"

# 2. Create the virtual environment.
echo "[setup] creating .venv (python ${PYTHON_VERSION})"
uv venv --python "${PYTHON_VERSION}" .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Install the CUDA-matched PyTorch + torchvision FIRST, from the PyTorch wheel
#    index, so the cu* builds are used instead of the default CPU wheels from PyPI.
#    (uv >= 0.5 users may alternatively run: uv pip install --torch-backend=auto ...)
echo "[setup] installing torch==${TORCH_VERSION} + torchvision==${TORCHVISION_VERSION} (${CUDA}) from ${TORCH_INDEX}"
uv pip install --index-url "${TORCH_INDEX}" "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# 4. Pin the Qwen3-VL-compatible transformers, then install the project + deps.
#    torch is already satisfied, so it is not re-resolved from PyPI.
echo "[setup] installing transformers==${TRANSFORMERS_VERSION} + RhinoVLA (editable)"
uv pip install "transformers==${TRANSFORMERS_VERSION}"
uv pip install -e .

# 5. Optional extras.
if [[ "${WITH_LOGGING}" == "1" ]]; then
  echo "[setup] installing logging extra (swanlab + matplotlib)"
  uv pip install -e ".[logging]"
fi
if [[ "${WITH_FLASH}" == "1" ]]; then
  echo "[setup] installing flash-attn (requires a build matching your torch/CUDA)"
  uv pip install -e ".[flash]" || echo "[setup] flash-attn install failed; train with attn_implementation=sdpa instead"
fi

# 6. Smoke check.
echo "[setup] verifying install ..."
python - <<'PY'
import torch, transformers
print("torch       ", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
print("arch_list   ", torch.cuda.get_arch_list())
print("transformers", transformers.__version__)
import importlib.util as u
print("qwen3_vl    ", bool(u.find_spec("transformers.models.qwen3_vl")))
import rhinovla.model.framework as fw
print("framework   ", fw.RhinoVLA.__name__)
PY

echo "[setup] done. Activate with: source ${REPO}/.venv/bin/activate"

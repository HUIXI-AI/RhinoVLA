#!/usr/bin/env bash
# RhinoVLA environment setup via uv (https://github.com/astral-ng/uv).
#
# Creates a local .venv and installs the CUDA-matched PyTorch stack plus the
# RhinoVLA package and its dependencies.
#
# Usage:
#   bash scripts/setup_env.sh                 # default: py3.13 + torch 2.10 cu128
#   FLASH_ATTN_WHEEL=/path/to/cp313.whl bash scripts/setup_env.sh
#
# After it finishes:  source .venv/bin/activate

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO}"

readonly PYTHON_VERSION=3.13
readonly TORCH_VERSION=2.10.0
readonly TORCHVISION_VERSION=0.25.0
readonly TRANSFORMERS_VERSION=5.3.0
readonly CUDA=cu128
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA}"
PYPI_INDEX_URL=${PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}
UV_LINK_MODE=${UV_LINK_MODE:-hardlink}
UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}
TORCH_OFFLINE=${TORCH_OFFLINE:-0}
FLASH_ATTN_WHEEL=${FLASH_ATTN_WHEEL:-}
readonly FLASH_ATTN_FILENAME=flash_attn-2.8.3+cu128torch2.10-cp313-cp313-linux_x86_64.whl
readonly FLASH_ATTN_SHA256=eebd92a7ba3e97b4e4ad542d4f33f2a018451b92ffe87ccf6a38d00e8e3269e8
readonly ENV_ID=py313-torch2.10.0-cu128-fa2.2.8.3
ENV_DIR="${REPO}/.venvs/${ENV_ID}"
ENV_PYTHON="${ENV_DIR}/bin/python"

export UV_LINK_MODE UV_HTTP_TIMEOUT

# 1. Install uv if it is not already on PATH.
if ! command -v uv >/dev/null 2>&1; then
	echo "[setup] uv not found; installing to ~/.local/bin ..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="${HOME}/.local/bin:${PATH}"
fi
echo "[setup] uv: $(uv --version)"
echo "[setup] cache: $(uv cache dir) (link mode: ${UV_LINK_MODE})"

# 2. Validate all fixed binary inputs before touching the canonical environment.
BASE_PYTHON=$(uv python find "${PYTHON_VERSION}")
if [[ "$(${BASE_PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "${PYTHON_VERSION}" ]]; then
	echo "[setup] uv did not resolve Python ${PYTHON_VERSION}: ${BASE_PYTHON}" >&2
	exit 1
fi
if [[ -z "${FLASH_ATTN_WHEEL}" || ! -f "${FLASH_ATTN_WHEEL}" ]]; then
	echo "[setup] FLASH_ATTN_WHEEL must name an existing wheel" >&2
	exit 1
fi
if [[ "$(basename "${FLASH_ATTN_WHEEL}")" != "${FLASH_ATTN_FILENAME}" ]]; then
	echo "[setup] flash-attn wheel must be ${FLASH_ATTN_FILENAME}: ${FLASH_ATTN_WHEEL}" >&2
	exit 1
fi
actual_flash_sha=$(sha256sum "${FLASH_ATTN_WHEEL}" | awk '{print $1}')
if [[ "${actual_flash_sha}" != "${FLASH_ATTN_SHA256}" ]]; then
	echo "[setup] flash-attn SHA256 mismatch: expected ${FLASH_ATTN_SHA256}, got ${actual_flash_sha}" >&2
	exit 1
fi

TORCH_INSTALL_ARGS=(--index-url "${TORCH_INDEX}")
if [[ "${TORCH_OFFLINE}" == "1" ]]; then
	TORCH_INSTALL_ARGS+=(--offline)
	uv pip install --dry-run --python "${BASE_PYTHON}" "${TORCH_INSTALL_ARGS[@]}" \
		"torch==${TORCH_VERSION}+${CUDA}" \
		"torchvision==${TORCHVISION_VERSION}+${CUDA}"
fi

# 3. Build at a stable project-local path so console-script shebangs remain valid.
# A completed versioned environment is immutable and reused after verification;
# never clear the currently published training environment in place.
mkdir -p "${REPO}/.venvs"
REUSE_COMPLETE_ENV=0
if [[ -f "${ENV_DIR}/.rhinovla-env-complete.json" ]]; then
	REUSE_COMPLETE_ENV=1
	echo "[setup] reusing completed environment at ${ENV_DIR}"
elif [[ -e "${ENV_DIR}" ]]; then
	failed_env="${ENV_DIR}.incomplete.$(date +%Y%m%d-%H%M%S)"
	mv "${ENV_DIR}" "${failed_env}"
	echo "[setup] preserved incomplete environment at ${failed_env}"
fi
if [[ "${REUSE_COMPLETE_ENV}" == "0" ]]; then
	echo "[setup] creating ${ENV_DIR} (python ${PYTHON_VERSION})"
	uv venv --clear --python "${BASE_PYTHON}" "${ENV_DIR}"

	# Resolve generic runtime wheels from the fast PyPI mirror before switching
	# to the CUDA-only PyTorch index.
	echo "[setup] installing numpy + pillow from ${PYPI_INDEX_URL}"
	uv pip install --python "${ENV_PYTHON}" --index-url "${PYPI_INDEX_URL}" numpy pillow

	# 4. Install the CUDA-matched PyTorch + torchvision FIRST, from the PyTorch
	#    wheel index, so the cu* builds are used instead of CPU PyPI wheels.
	echo "[setup] installing torch==${TORCH_VERSION} + torchvision==${TORCHVISION_VERSION} (${CUDA}) from ${TORCH_INDEX}"
	uv pip install --python "${ENV_PYTHON}" "${TORCH_INSTALL_ARGS[@]}" \
		"torch==${TORCH_VERSION}+${CUDA}" \
		"torchvision==${TORCHVISION_VERSION}+${CUDA}"

	# 5. Pin Qwen3-VL transformers, then install the project with reporting deps.
	echo "[setup] installing transformers==${TRANSFORMERS_VERSION} + RhinoVLA (editable)"
	uv pip install --python "${ENV_PYTHON}" --index-url "${PYPI_INDEX_URL}" \
		"transformers==${TRANSFORMERS_VERSION}"
	uv pip install --python "${ENV_PYTHON}" --index-url "${PYPI_INDEX_URL}" -e .

	echo "[setup] installing logging extra (swanlab + matplotlib)"
	uv pip install --python "${ENV_PYTHON}" --index-url "${PYPI_INDEX_URL}" \
		-e ".[logging]"
	echo "[setup] installing verified third-party flash-attn binary wheel"
	uv pip install --python "${ENV_PYTHON}" --no-deps "${FLASH_ATTN_WHEEL}"
fi

# 7. Fail closed on version, CUDA, ABI, dependency, and project import drift.
echo "[setup] verifying install ..."
uv pip check --python "${ENV_PYTHON}"
"${ENV_PYTHON}" - <<'PY'
import sys
import flash_attn
import torch
import torchvision
import transformers

assert sys.version_info[:2] == (3, 13), sys.version
assert torch.__version__ == "2.10.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torchvision.__version__ == "0.25.0+cu128", torchvision.__version__
assert transformers.__version__ == "5.3.0", transformers.__version__
assert flash_attn.__version__.startswith("2.8.3"), flash_attn.__version__
print("torch       ", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
print("arch_list   ", torch.cuda.get_arch_list())
print("transformers", transformers.__version__)
import importlib.util as u
assert u.find_spec("transformers.models.qwen3_vl") is not None
print("qwen3_vl    ", True)
print("flash_attn  ", flash_attn.__version__)
import rhinovla.model.framework as fw
print("framework   ", fw.RhinoVLA.__name__)
PY

# 8. Write a durable completion receipt, then publish the canonical symlink.
"${ENV_PYTHON}" - "${ENV_DIR}/.rhinovla-env-complete.json" "${actual_flash_sha}" <<'PY'
import json
from pathlib import Path
import platform
import sys
import torch
import torchvision
import flash_attn
import transformers

receipt = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda": torch.version.cuda,
    "flash_attn": flash_attn.__version__,
    "flash_attn_sha256": sys.argv[2],
    "transformers": transformers.__version__,
}
Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
if [[ -e "${REPO}/.venv" && ! -L "${REPO}/.venv" ]]; then
	previous_env="${REPO}/.venv.previous.$(date +%Y%m%d-%H%M%S)"
	mv "${REPO}/.venv" "${previous_env}"
	echo "[setup] preserved previous .venv at ${previous_env}"
fi
ln -sfn ".venvs/${ENV_ID}" "${REPO}/.venv"

echo "[setup] done. Activate with: source ${REPO}/.venv/bin/activate"

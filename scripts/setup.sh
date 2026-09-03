#!/usr/bin/env bash
# Reproducible installation for a fresh CUDA 12.8 Python 3.11+ environment.
#
# Override PYTORCH_INDEX_URL for another compatible wheel source. Override
# MAX_JOBS to control CUDA extension compilation parallelism. The default CUDA
# arch list produces kernels for the local RTX 4090 and remote H200 workers.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
MAX_JOBS="${MAX_JOBS:-4}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;9.0}"
export MAX_JOBS TORCH_CUDA_ARCH_LIST
export PIP_PROGRESS_BAR=off
# Some cluster images inject an unreachable NVIDIA extra index, making every
# ordinary PyPI lookup wait through TLS retries. Opt back in explicitly with
# PIP_EXTRA_INDEX_URL_OVERRIDE when that mirror is actually needed.
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL_OVERRIDE:-}"
if [ -z "${PIP_EXTRA_INDEX_URL_OVERRIDE:-}" ]; then
    export PIP_CONFIG_FILE=/dev/null
fi

"$PYTHON_BIN" -c '
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ is required, got {sys.version.split()[0]}")
'

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel packaging ninja

# Install ABI-coupled PyTorch packages together from one CUDA wheel index.
"$PYTHON_BIN" -m pip install \
    --index-url "$PYTORCH_INDEX_URL" \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchcodec==0.5.0

# causal-conv1d imports torch from its build backend. Keep torch visible rather
# than creating an isolated build environment that cannot see it.
"$PYTHON_BIN" -m pip install --no-build-isolation -r requirements.txt

# LeRobot's wheel has stale upper bounds for transformers/huggingface-hub.
# requirements.txt explicitly installs its concrete runtime dependencies; only
# dependency resolution for this one wheel is bypassed.
"$PYTHON_BIN" -m pip install --no-deps lerobot==0.4.1

# flash-attn's official installation path expects torch to be visible in the
# build environment. MAX_JOBS=4 avoids exhausting RAM on many-core hosts.
"$PYTHON_BIN" -m pip install \
    flash-attn==2.7.3 \
    --no-build-isolation

"$PYTHON_BIN" -m pip install --no-deps -e .

"$PYTHON_BIN" -c '
import importlib.metadata as metadata
import torch

for package in (
    "torch",
    "torchvision",
    "torchcodec",
    "transformers",
    "huggingface-hub",
    "lerobot",
    "flash-attn",
    "flash-linear-attention",
    "causal-conv1d",
    "deepspeed",
    "tau0_vla",
):
    print(f"{package}=={metadata.version(package)}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"torch_cuda={torch.version.cuda}")
print(f"gpu_count={torch.cuda.device_count()}")
print(f"cxx11_abi={int(torch.compiled_with_cxx11_abi())}")
'

"$PYTHON_BIN" scripts/validate_training_stack.py

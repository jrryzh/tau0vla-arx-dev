#!/usr/bin/env python3
"""Validate the ABI-coupled CUDA training stack, including causal-conv1d."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata

import torch
import torch.nn.functional as F
from packaging.version import Version


EXPECTED = {
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "torchcodec": "0.5.0",
    "flash-attn": "2.7.3",
    "flash-linear-attention": "0.5.0",
    "causal-conv1d": "1.7.0",
    "deepspeed": "0.18.3",
}


def _versions() -> None:
    for package, expected in EXPECTED.items():
        actual = metadata.version(package)
        if Version(actual.split("+")[0]) != Version(expected):
            raise RuntimeError(f"{package} must be {expected}, got {actual}")
        print(f"{package}={actual}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"PyTorch must target CUDA 12.8, got {torch.version.cuda}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cxx11_abi={int(torch.compiled_with_cxx11_abi())}")


def _imports() -> None:
    for module in ("flash_attn", "fla", "causal_conv1d", "deepspeed"):
        importlib.import_module(module)
        print(f"import:{module}=ok")


def _causal_conv_cuda_test() -> None:
    if not torch.cuda.is_available():
        print("causal_conv_cuda=skipped(no CUDA device)")
        return

    from causal_conv1d import causal_conv1d_fn

    torch.manual_seed(7)
    device = torch.device("cuda")
    batch, channels, length, width = 2, 8, 32, 4
    x_base = torch.randn(batch, channels, length, device=device, dtype=torch.bfloat16)
    w_base = torch.randn(channels, width, device=device, dtype=torch.bfloat16)
    b_base = torch.randn(channels, device=device, dtype=torch.bfloat16)
    grad = torch.randn(batch, channels, length, device=device, dtype=torch.bfloat16)

    x_fast = x_base.detach().clone().requires_grad_(True)
    w_fast = w_base.detach().clone().requires_grad_(True)
    b_fast = b_base.detach().clone().requires_grad_(True)
    y_fast = causal_conv1d_fn(x_fast, w_fast, b_fast, activation=None)
    y_fast.backward(grad)

    x_ref = x_base.detach().clone().requires_grad_(True)
    w_ref = w_base.detach().clone().requires_grad_(True)
    b_ref = b_base.detach().clone().requires_grad_(True)
    y_ref = F.conv1d(
        x_ref,
        w_ref.unsqueeze(1),
        b_ref,
        padding=width - 1,
        groups=channels,
    )[..., :length]
    y_ref.backward(grad)

    checks = {
        "output": (y_fast, y_ref),
        "input_grad": (x_fast.grad, x_ref.grad),
        "weight_grad": (w_fast.grad, w_ref.grad),
        "bias_grad": (b_fast.grad, b_ref.grad),
    }
    for name, (actual, expected) in checks.items():
        torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-2, atol=3e-2)
        print(f"causal_conv_{name}=ok")
    print(f"causal_conv_gpu={torch.cuda.get_device_name(0)}")
    print("causal_conv_cuda=ok")


def main() -> None:
    _versions()
    _imports()
    _causal_conv_cuda_test()
    print("training_stack=ok")


if __name__ == "__main__":
    main()

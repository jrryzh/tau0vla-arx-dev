#!/usr/bin/env python3
"""Per-worker guard used before a two-node H200 torchrun launch."""

from __future__ import annotations

import os
import socket

import torch


def main() -> None:
    expected_nodes = int(os.environ.get("EXPECTED_NNODES", "2"))
    expected_gpus = int(os.environ.get("EXPECTED_GPUS_PER_NODE", "8"))
    nnodes = int(os.environ.get("PET_NNODES", "0"))
    node_rank = int(os.environ.get("PET_NODE_RANK", "-1"))
    if nnodes != expected_nodes:
        raise RuntimeError(f"PET_NNODES must be {expected_nodes}, got {nnodes}")
    if node_rank not in range(expected_nodes):
        raise RuntimeError(f"PET_NODE_RANK must be in [0, {expected_nodes}), got {node_rank}")
    if not os.environ.get("PET_MASTER_ADDR"):
        raise RuntimeError("PET_MASTER_ADDR is not set")
    if not os.environ.get("PET_MASTER_PORT"):
        raise RuntimeError("PET_MASTER_PORT is not set")

    count = torch.cuda.device_count()
    if count != expected_gpus:
        raise RuntimeError(f"Expected {expected_gpus} GPUs on this worker, found {count}")
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    if any("H200" not in name.upper() for name in names):
        raise RuntimeError(f"All GPUs must be H200s, found {names}")
    if not all(torch.cuda.get_device_properties(index).major == 9 for index in range(count)):
        raise RuntimeError("All H200 GPUs must report compute capability 9.x")

    print(
        "[H200_PREFLIGHT] "
        f"host={socket.gethostname()} node_rank={node_rank} nnodes={nnodes} "
        f"gpus={count} gpu={names[0]} master={os.environ['PET_MASTER_ADDR']}:{os.environ['PET_MASTER_PORT']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

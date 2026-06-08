# SPDX-FileCopyrightText: hzwer / Practical-RIFE contributors
# SPDX-License-Identifier: MIT AND AGPL-3.0
# Code vendored from: https://github.com/hzwer/Practical-RIFE
from __future__ import annotations

import torch
import torch.nn.functional as F

# Cache of base sampling grids, keyed by (device, dtype, shape). Stays on-device.
_grid_cache: dict[tuple[str, torch.dtype, tuple[int, ...]], torch.Tensor] = {}


def warp(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    """Backward-warp ``tenInput`` by optical flow ``tenFlow`` (RIFE convention).

    All tensors stay on their input device (CUDA in the pipeline). The grid is
    built in ``tenFlow``'s dtype: grid_sample requires grid and input to share
    a dtype, and a float32 grid would silently promote the half flow sum.
    """
    key = (str(tenFlow.device), tenFlow.dtype, tuple(tenFlow.shape))
    grid = _grid_cache.get(key)
    if grid is None:
        h, w = tenFlow.shape[2], tenFlow.shape[3]
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, w, device=tenFlow.device, dtype=tenFlow.dtype)
            .view(1, 1, 1, w)
            .expand(tenFlow.shape[0], -1, h, -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, h, device=tenFlow.device, dtype=tenFlow.dtype)
            .view(1, 1, h, 1)
            .expand(tenFlow.shape[0], -1, -1, w)
        )
        grid = torch.cat([tenHorizontal, tenVertical], 1)
        _grid_cache[key] = grid

    flow_norm = torch.cat(
        [
            tenFlow[:, 0:1] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )
    g = (grid + flow_norm).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=tenInput, grid=g, mode="bilinear", padding_mode="border", align_corners=True
    )

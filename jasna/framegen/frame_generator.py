from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class FrameGenerator(Protocol):
    """Pluggable frame-interpolation backend.

    Given two consecutive full-resolution frames, produce one interpolated frame
    per requested temporal position. Positions are in the open interval (0, 1),
    where 0 is ``frame_a`` and 1 is ``frame_b``; e.g. 2x uses ``[0.5]`` and 4x
    uses ``[0.25, 0.5, 0.75]``.

    Contract:
    - ``frame_a`` / ``frame_b``: CHW RGB ``uint8`` tensors on CUDA (the exact
      format the encoder consumes via ``chw_rgb_to_surface``).
    - Returns a list with one CHW RGB ``uint8`` CUDA tensor per position, in the
      same order. The returned frames must match ``frame_a``'s shape.
    """

    name: str

    def interpolate(
        self, frame_a: torch.Tensor, frame_b: torch.Tensor, positions: list[float]
    ) -> list[torch.Tensor]:
        ...

    def close(self) -> None:
        ...

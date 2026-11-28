"""+modi: whole-frame upscaling by a model that works on runs of frames (``jasna-postprocess``).

``FrameUpscaleWriter`` hands its upscaler one frame at a time. A video model restores a run of
frames per call and keeps no state between calls, so the first frames of a run are made without
temporal context and do not match the end of the run before. ``ClipUpscaleWriter`` cuts the video
into windows that overlap, has both windows restore the overlap, and crossfades between the two
results there."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from jasna.ltx.restore import FrameWriter

DEFAULT_WINDOW = 89
"""Frames per window: ``4k + 1``, the length SwiftVR restores without padding."""
DEFAULT_OVERLAP = 16
"""Frames two neighbouring windows share (the clip overlap of the restoration pipeline)."""


class ClipUpscaler(Protocol):
    def upscale(self, frames: torch.Tensor) -> Iterator[torch.Tensor]:
        """uint8 ``[n, 3, H, W]`` CPU frames to the ``n`` uint8 ``[3, H', W']`` output frames on
        the GPU, in order. The writer consumes every frame before it calls again."""

    def close(self) -> None: ...


def overlap_weights(overlap: int) -> list[float]:
    """The share of the later window at each of the ``overlap`` frames two windows share: 0 over
    the first third (the later window has no context yet), 1 over the last third, a linear ramp
    in between."""
    edge = overlap // 3
    ramp = overlap - 2 * edge
    return [min(1.0, max(0.0, (index - edge + 1) / (ramp + 1))) for index in range(overlap)]


class ClipUpscaleWriter:
    """Upscales windows of ``window`` frames, ``overlap`` of them shared with the next window,
    before ``writer``; owns ``upscaler`` and closes it with ``writer``. Frames wait on the CPU
    until their window is full, so ``writer`` receives them up to a window late."""

    def __init__(
        self,
        writer: FrameWriter,
        upscaler: ClipUpscaler,
        *,
        window: int = DEFAULT_WINDOW,
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        if overlap < 0 or window < 2 * overlap + 1:
            raise ValueError(f"window ({window}) must hold at least 2 * overlap + 1 frames (overlap {overlap})")
        self._writer = writer
        self._upscaler = upscaler
        self._window = int(window)
        self._overlap = int(overlap)
        self._weights = overlap_weights(self._overlap)
        self._frames: list[torch.Tensor] = []
        self._pts: list[int] = []
        # The previous window's output for the first ``overlap`` buffered frames.
        self._tail: list[torch.Tensor] = []
        self._device: torch.device | None = None
        self.gpu_turn = threading.Lock()
        """Held while the upscaler works on a window. A caller whose own GPU work must not
        overlap the upscaler's peak (the LTX window decode) takes it for that work."""

    def write(self, frame: torch.Tensor, pts: int) -> None:
        self._frames.append(frame.cpu())
        self._pts.append(int(pts))
        if len(self._frames) == self._window:
            self._run(last=False)

    def _run(self, *, last: bool) -> None:
        with self.gpu_turn:
            self._run_window(last=last)

    def _run_window(self, *, last: bool) -> None:
        count = len(self._frames)
        keep = 0 if last else self._overlap
        tail, self._tail = self._tail, []
        for index, upscaled in enumerate(self._upscaler.upscale(torch.stack(self._frames))):
            self._device = upscaled.device
            if index < len(tail):
                weight = self._weights[index]
                if weight < 1.0:
                    earlier = tail[index].to(upscaled.device)
                    if weight > 0.0:
                        mixed = torch.lerp(earlier.float(), upscaled.float(), weight)
                        upscaled = mixed.round_().clamp_(0.0, 255.0).to(torch.uint8)
                    else:
                        upscaled = earlier
            if index < count - keep:
                self._writer.write(upscaled, self._pts[index])
            else:
                self._tail.append(upscaled.cpu())
        self._frames = self._frames[count - keep:]
        self._pts = self._pts[count - keep:]

    def after_write(self, frames_written: int, stop_event) -> None:
        after_write = getattr(self._writer, "after_write", None)
        if after_write is not None:
            after_write(frames_written, stop_event)

    def finish(self) -> None:
        if len(self._frames) > len(self._tail):
            self._run(last=True)
        else:
            # Nothing came after the last window: its output for these frames is final.
            for upscaled, pts in zip(self._tail, self._pts):
                self._writer.write(upscaled.to(self._device), pts)
            self._frames, self._pts, self._tail = [], [], []
        self._writer.finish()

    def close(self) -> None:
        self._frames, self._pts, self._tail = [], [], []
        try:
            self._writer.close()
        finally:
            self._upscaler.close()

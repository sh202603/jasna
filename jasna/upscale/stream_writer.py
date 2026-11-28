"""+modi: whole-frame upscaling by a model that keeps its state over the whole video
(``jasna-postprocess``).

``ClipUpscaleWriter`` serves models without state: it cuts the video into overlapping windows and
crossfades them. A streaming model needs neither, cutting the video would throw its context away.
What it needs instead is room for its delay: it emits a frame only when it has seen enough of the
frames after it, so one input frame is answered by zero or more output frames.
``StreamUpscaleWriter`` hands the frames through one by one and keeps the timestamps of the
frames that have not come out yet."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from jasna.ltx.restore import FrameWriter


class StreamUpscaler(Protocol):
    def push(self, frame: torch.Tensor) -> Iterator[torch.Tensor]:
        """One uint8 ``[3, H, W]`` frame in; the uint8 ``[3, H', W']`` output frames that are
        final by now (none, one or several), in the order of the input. The writer consumes
        them all before it calls again."""

    def finish(self) -> Iterator[torch.Tensor]:
        """The input has ended: the output frames still owed."""

    def close(self) -> None: ...


class StreamUpscaleWriter:
    """Upscales the frames through ``upscaler`` before ``writer``; owns ``upscaler`` and closes it
    with ``writer``. Every input frame comes out once and in order, so the timestamps wait in a
    queue; ``writer`` receives the frames as late as the model emits them."""

    def __init__(self, writer: FrameWriter, upscaler: StreamUpscaler) -> None:
        self._writer = writer
        self._upscaler = upscaler
        self._pts: deque[int] = deque()
        self.gpu_turn = threading.Lock()
        """Held while the upscaler takes a frame (``ClipUpscaleWriter.gpu_turn``)."""

    def write(self, frame: torch.Tensor, pts: int) -> None:
        self._pts.append(int(pts))
        with self.gpu_turn:
            for upscaled in self._upscaler.push(frame):
                self._writer.write(upscaled, self._pts.popleft())

    def after_write(self, frames_written: int, stop_event) -> None:
        after_write = getattr(self._writer, "after_write", None)
        if after_write is not None:
            after_write(frames_written, stop_event)

    def finish(self) -> None:
        with self.gpu_turn:
            for upscaled in self._upscaler.finish():
                self._writer.write(upscaled, self._pts.popleft())
        if self._pts:
            raise RuntimeError(f"the upscaler kept {len(self._pts)} frame(s) at the end of the video")
        self._writer.finish()

    def close(self) -> None:
        self._pts.clear()
        try:
            self._writer.close()
        finally:
            self._upscaler.close()

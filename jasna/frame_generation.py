"""Frame generation: frames made between each pair of finished frames, right before the encoder.

Upstream (v0.12.2) generates one frame at each midpoint with NVIDIA RTX Video Frame Generation
(2x). modi keeps that plumbing (``Pipeline._frame_writer``, the encoder's time base) and adds its
own policy on it: a 4x factor, the RIFE backend (``jasna.framegen``, every GPU) next to RTX, and
the RTX quality mode. The writer and the encoder share one PTS scheme: the source time base is
divided by the multiplier, so every generated PTS stays whole.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from jasna.ltx.restore import FrameWriter

MULTIPLIERS = {"none": 1, "2x": 2, "4x": 4}
"""``--frame-generation`` values to the output frame multiplier."""


class FrameGenerator(Protocol):
    """A backend that makes the frames between two source frames (``jasna.framegen``)."""

    def interpolate(self, frame_a: torch.Tensor, frame_b: torch.Tensor, positions: list[float]) -> list[torch.Tensor]: ...

    def close(self) -> None: ...


def multiplier_of(frame_generation: str) -> int:
    """The output frame multiplier of a ``--frame-generation`` value."""
    try:
        return MULTIPLIERS[str(frame_generation).lower()]
    except KeyError:
        raise ValueError(f"Unsupported frame generation {frame_generation!r} (choices: {', '.join(MULTIPLIERS)})") from None


class FrameGenerationWriter:
    """Writes each frame and, before the next one, the ``multiplier - 1`` frames generated
    between them. PTS are multiplied by ``multiplier`` so the generated ones stay whole; the
    encoder runs with ``frame_generation=multiplier`` and the source time base divided by it. The
    last frame is written ``multiplier`` times, the copies spaced by its share of the last
    interval, so ``N`` frames become ``N * multiplier`` and the output keeps the source duration.

    ``generate`` is either a ``FrameGenerator`` (``interpolate`` at the positions ``k / multiplier``)
    or, for 2x, a plain callable ``(previous, current) -> midpoint frame`` (upstream's contract)."""

    def __init__(self, writer: FrameWriter, generate, multiplier: int = 2, *, owns_generator: bool = True) -> None:
        """Closes ``generate`` with ``writer`` unless ``owns_generator`` is False (a folder batch
        that shares one generator across its videos)."""
        self._writer = writer
        self._generate = generate
        self._multiplier = int(multiplier)
        if self._multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {self._multiplier}")
        if self._multiplier != 2 and not hasattr(generate, "interpolate"):
            raise ValueError("a plain midpoint callable can only generate 2x; give a FrameGenerator")
        self._positions = [k / self._multiplier for k in range(1, self._multiplier)]
        self._owns_generator = owns_generator
        self._previous: tuple[torch.Tensor, int] | None = None
        self._interval = 0

    @property
    def multiplier(self) -> int:
        return self._multiplier

    def _between(self, previous: torch.Tensor, current: torch.Tensor) -> list[torch.Tensor]:
        if hasattr(self._generate, "interpolate"):
            return list(self._generate.interpolate(previous, current, self._positions))
        return [self._generate(previous, current)]

    def write(self, frame: torch.Tensor, pts: int) -> None:
        m = self._multiplier
        if m == 1:
            self._writer.write(frame, pts)
            return
        if self._previous is not None:
            previous, previous_pts = self._previous
            self._writer.write(previous, m * previous_pts)
            span = pts - previous_pts
            for k, generated in enumerate(self._between(previous, frame), start=1):
                self._writer.write(generated, m * previous_pts + k * span)
            self._interval = span
        self._previous = (frame, pts)

    def after_write(self, frames_written: int, stop_event) -> None:
        after_write = getattr(self._writer, "after_write", None)
        if after_write is not None:
            after_write(frames_written, stop_event)

    def finish(self) -> None:
        if self._previous is not None:
            frame, pts = self._previous
            m = self._multiplier
            self._writer.write(frame, m * pts)
            if self._interval > 0:
                for k in range(1, m):
                    self._writer.write(frame, m * pts + k * self._interval)
            self._previous = None
        self._writer.finish()

    def close(self) -> None:
        self._previous = None
        try:
            self._writer.close()
        finally:
            if self._owns_generator:
                self._generate.close()


class RtxFrameGeneration:
    """nvvfx ``VideoFrameGeneration`` between two uint8 ``[3, height, width]`` GPU frames: the
    midpoint when called, any positions through ``interpolate``. Its automatic shot-change
    detection stays on, so a hard cut is not blended across. ``mode`` is the SDK's quality mode
    (+modi: low / medium / high)."""

    def __init__(self, device: torch.device, width: int, height: int, *, mode: str = "medium") -> None:
        from jasna.framegen.rtx_frame_generator import RtxFrameGenerator

        self._device = torch.device(device)
        self._generator = RtxFrameGenerator(device=self._device, mode=mode)
        self._generator.load(int(height), int(width))

    def __call__(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        return self.interpolate(previous, current, [0.5])[0]

    def interpolate(self, previous: torch.Tensor, current: torch.Tensor, positions: list[float]) -> list[torch.Tensor]:
        if self._device.type != "cuda":
            return self._generator.interpolate(previous, current, positions)
        # nvvfx effects fail on any stream but the default one they load with (see ltx/super_res.py):
        # run on it, ordered after and before the caller's stream.
        stream = torch.cuda.current_stream(self._device)
        default = torch.cuda.default_stream(self._device)
        default.wait_stream(stream)
        with torch.cuda.stream(default):
            generated = self._generator.interpolate(previous, current, positions)
        stream.wait_stream(default)
        for frame in generated:
            frame.record_stream(stream)
        return generated

    def close(self) -> None:
        """Frees the effect's GPU memory; safe to call again."""
        self._generator.close()


def build_generator(
    backend: str,
    *,
    device: torch.device,
    width: int,
    height: int,
    rtx_mode: str = "medium",
    model_path: str | None = None,
    fp16: bool = True,
):
    """The frame generator of ``backend``: ``rtx`` (RTX Video Frame Generation, Ada and newer)
    or ``rife`` (RIFE in PyTorch, every GPU, ``model_path`` or ``model_weights/rife.pth``)."""
    backend = str(backend).lower()
    if backend == "rtx":
        # The default mode is left out, so the call keeps upstream's shape (its tests assert it).
        return RtxFrameGeneration(device, width, height, **({} if rtx_mode == "medium" else {"mode": rtx_mode}))
    if backend == "rife":
        from jasna.framegen.rife_frame_generator import RifeFrameGenerator

        return RifeFrameGenerator(device=device, model_path=model_path, fp16=fp16)
    raise ValueError(f"Unsupported frame generation backend {backend!r} (choices: rife, rtx)")

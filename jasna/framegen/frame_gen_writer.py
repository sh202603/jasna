from __future__ import annotations

import logging

import torch

from jasna.framegen.frame_generator import FrameGenerator

log = logging.getLogger(__name__)


class FrameGenWriter:
    """A ``FrameWriter`` decorator that multiplies the output frame rate.

    It wraps the real frame writer (the encoder adapter) and, for every pair of
    consecutive source frames, asks the backend to synthesize ``multiplier - 1``
    intermediate frames and emits them with interpolated PTS between the two
    source PTS. The rest of the pipeline and the encoder are unchanged: they
    keep calling ``write(frame, pts)`` once per *source* frame.

    PTS for the k-th intermediate of a [prev, curr] interval:
        pts_k = prev_pts + round((curr_pts - prev_pts) * k / M),  k = 1 .. M-1

    Output timing is fully PTS-driven (mkvmerge explicit timecodes), so inserting
    these timestamps is what produces the 2x / 4x frame rate. Interpolation only
    runs when ``span >= M`` (``span = curr_pts - prev_pts``): that guarantees all
    intermediate PTS land strictly between prev/curr and are mutually distinct.
    Besides non-monotonic / zero-length intervals, this skips coarse time_base
    cases (e.g. tbn == fps) where a rounded intermediate PTS would collide with a
    real frame's PTS and the encoder's duplicate-PTS handling could silently drop
    the real frame.

    Total frames emitted for N source frames: ``(N - 1) * M + 1``.
    """

    def __init__(self, real_writer, generator: FrameGenerator, multiplier: int) -> None:
        self._writer = real_writer
        self._generator = generator
        self._multiplier = int(multiplier)
        if self._multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {self._multiplier}")
        self._positions = [k / self._multiplier for k in range(1, self._multiplier)]
        self._prev: tuple[torch.Tensor, int, bool] | None = None

    def write(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True) -> None:
        if self._multiplier == 1:
            self._writer.write(frame, pts, apply_lut=apply_lut)
            return

        if self._prev is None:
            self._prev = (frame, int(pts), apply_lut)
            return

        prev_frame, prev_pts, prev_lut = self._prev
        curr_pts = int(pts)

        # Emit the previous (real) frame first, then the interpolated frames.
        self._writer.write(prev_frame, prev_pts, apply_lut=prev_lut)

        span = curr_pts - prev_pts
        if span >= self._multiplier:
            inters = self._generator.interpolate(prev_frame, frame, self._positions)
            for k, inter in enumerate(inters, start=1):
                pts_k = prev_pts + round(span * k / self._multiplier)
                # An intermediate gets the LUT only when both of its source frames do.
                self._writer.write(inter, pts_k, apply_lut=prev_lut and apply_lut)
        else:
            # span < M: non-monotonic/duplicate PTS, or a time_base too coarse to
            # place M-1 distinct intermediates without colliding with a real frame.
            log.debug(
                "[frame-gen] skipping interpolation for PTS span %d < multiplier %d (%d -> %d)",
                span, self._multiplier, prev_pts, curr_pts,
            )

        self._prev = (frame, curr_pts, apply_lut)

    def after_write(self, frames_written: int) -> None:
        self._writer.after_write(frames_written)

    def close(self) -> None:
        # Flush the final held source frame, then close the real writer.
        #
        # The generator is *borrowed*, not owned: whoever built it (the CLI
        # folder loop in main.py / the GUI job runner) is responsible for
        # closing it. A folder batch builds one generator and reuses it across
        # every video, so closing it here would free the model (sets _model to
        # None) and crash the next video's interpolation. This mirrors how the
        # pipeline treats the borrowed restoration_pipeline (it drops the
        # reference but does not close it).
        if self._prev is not None:
            prev_frame, prev_pts, prev_lut = self._prev
            self._writer.write(prev_frame, prev_pts, apply_lut=prev_lut)
            self._prev = None
        self._writer.close()

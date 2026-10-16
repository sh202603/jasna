"""Sharpen the mosaic areas of an LTX-restored video with SwiftVR (``--ltx-crop-vsr``).

The release build restores mosaics with LTX; the public source cannot run the LTX model
but has LTX's scan (track cameras and windows) and compose (the feathered composite).
This module reruns the scan on the source video (the mosaics are gone from the LTX
output), crops the LTX output with the same cameras, upscales every window x2 with
SwiftVR, and composites the result back onto the LTX output with LTX's own composite.
The output keeps the LTX output's resolution, frame count and timestamps.

Two readers run in lockstep over the LTX output and the source. The source only tells
the windows LTX restored from those it did not (a mosaic LTX left alone must not be
sharpened): a window whose track mask hardly differs between the two is left as is.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
from itertools import zip_longest
from pathlib import Path

import av
import torch
import torch.nn.functional as F
from tqdm import tqdm

from jasna.ltx.camera import CropFrame
from jasna.ltx.compose import Candidate, _area_weights, _reflect101_index, composite_frame, polygon_mask
from jasna.ltx.plan import TrackPlan, Window, crossfade_weights, feather_pixels
from jasna.ltx.restore import WHOLE_VIDEO_PTS, Cancelled, FrameSource, FrameWriter, LtxSegment, scan_span, video_frames
from jasna.media.probe import UnsupportedVideoError
from jasna.progressbar import ProgressCallback, Progressbar

logger = logging.getLogger(__name__)

SKIP_MEAN_LUMA_DIFF = 3.0
"""Provisional (design §3.4, settled in L0): a window whose track mask differs from the
source by fewer grey levels than this on average was not restored by LTX."""

_LUMA_WEIGHTS = (0.299, 0.587, 0.114)


# -- alignment of the LTX output with its source ------------------------------------------


@dataclass(frozen=True)
class StreamTimes:
    """A video stream as demuxed, without decoding: its frame size and the presentation
    timestamps of its frames in ``time_base`` units, sorted."""

    width: int
    height: int
    time_base: Fraction
    pts: tuple[int, ...]

    def seconds(self) -> list[Fraction]:
        """Each frame's time from the first frame."""
        start = self.pts[0] if self.pts else 0
        return [(p - start) * self.time_base for p in self.pts]


def read_stream_times(path: Path) -> StreamTimes:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # The demuxer ends with an empty flush packet; a real packet without a timestamp
        # cannot be aligned.
        pts = [packet.pts for packet in container.demux(stream) if packet.size > 0]
        if any(p is None for p in pts):
            raise UnsupportedVideoError(f"{path.name}: video packets without timestamps cannot be aligned")
        return StreamTimes(
            width=int(stream.codec_context.width),
            height=int(stream.codec_context.height),
            time_base=Fraction(stream.time_base),
            pts=tuple(sorted(int(p) for p in pts)),
        )


def check_alignment(ltx: StreamTimes, source: StreamTimes, *, fps: Fraction, ltx_name: str, source_name: str) -> None:
    """Every LTX frame needs a source frame at the same time from its stream's start
    (within half a frame), so the source's plan applies to the LTX output frame by frame.
    Times, not raw timestamps: the encoder drops the source's start offset and a
    container can change the time base."""
    if (ltx.width, ltx.height) != (source.width, source.height):
        raise UnsupportedVideoError(
            f"{ltx_name} is {ltx.width}x{ltx.height} but --ltx-crop-vsr-source {source_name} is "
            f"{source.width}x{source.height}; the LTX output and its source must have the same frame size"
        )
    if len(ltx.pts) != len(source.pts):
        raise UnsupportedVideoError(
            f"{ltx_name} has {len(ltx.pts)} frames but --ltx-crop-vsr-source {source_name} has "
            f"{len(source.pts)}; the source must be the video the LTX output was restored from, "
            "restored without --retarget-high-fps"
        )
    tolerance = Fraction(1, 2) / fps
    for index, (a, b) in enumerate(zip(ltx.seconds(), source.seconds())):
        if abs(a - b) > tolerance:
            raise UnsupportedVideoError(
                f"frame {index} is at {float(a):.4f} s in {ltx_name} but at {float(b):.4f} s in "
                f"--ltx-crop-vsr-source {source_name}; the two videos' frames do not line up"
            )


# -- canvases ------------------------------------------------------------------------------


def crop_to_canvas_area(frame: torch.Tensor, crop: CropFrame) -> torch.Tensor:
    """``crop_to_canvas`` shrinking by pixel area and enlarging bicubically: the inverse
    of ``canvas_to_source``, so an LTX output crop comes back close to the canvas LTX
    rendered."""
    x1, y1, x2, y2 = crop.crop_box
    content = frame[:, y1:y2, x1:x2]
    height, width = content.shape[1:]
    resize_h, resize_w = crop.resize_hw
    if (height, width) != (resize_h, resize_w):
        if max(height, width) > max(resize_h, resize_w):
            rows = _area_weights(resize_h, height, frame.device)
            cols = _area_weights(resize_w, width, frame.device)
            resized = torch.einsum("hy,cyx,wx->chw", rows, content.float(), cols)
        else:
            resized = F.interpolate(content[None].float(), size=(resize_h, resize_w), mode="bicubic", align_corners=False)[0]
        content = resized.round_().clamp_(0, 255).to(torch.uint8)
    top, bottom, left, right = crop.pad
    if top or bottom or left or right:
        rows = _reflect101_index(content.shape[1], top, bottom, frame.device)
        cols = _reflect101_index(content.shape[2], left, right, frame.device)
        content = content[:, rows][:, :, cols]
    return content


def doubled(crop: CropFrame) -> CropFrame:
    """The same source box on the x2 canvas: twice the resize and the padding."""
    resize_h, resize_w = crop.resize_hw
    top, bottom, left, right = crop.pad
    return CropFrame(
        crop_box=crop.crop_box,
        resize_hw=(2 * resize_h, 2 * resize_w),
        pad=(2 * top, 2 * bottom, 2 * left, 2 * right),
    )


def _luma(region: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor(_LUMA_WEIGHTS, dtype=torch.float32, device=region.device)[:, None, None]
    return (region.float() * weights).sum(0)


class _PinnedFrames:
    """Reused page-locked host buffers for the frames held until their windows are done."""

    def __init__(self) -> None:
        self._free: list[torch.Tensor] = []

    def take(self, frame: torch.Tensor) -> torch.Tensor:
        for index, buffer in enumerate(self._free):
            if buffer.shape == frame.shape and buffer.dtype == frame.dtype:
                del self._free[index]
                break
        else:
            buffer = torch.empty(frame.shape, dtype=frame.dtype, device="cpu", pin_memory=frame.is_cuda)
        buffer.copy_(frame)
        return buffer

    def give(self, buffer: torch.Tensor) -> None:
        self._free.append(buffer)


@dataclass
class _Held:
    index: int
    pts: int
    frame: torch.Tensor  # host copy
    windows: tuple[int, ...]  # the windows covering it, all of which must finish first


class WindowRefiner:
    """Frame-by-frame driver of the compose pass: builds every window's canvases from
    the LTX frames as they arrive, runs the x2 restorer once a window is complete, and
    releases the frames in order once every window covering them has finished (or was
    skipped). A frame waits for at most ``WINDOW_FRAMES`` later frames, whatever the
    number of tracks: every window covering it ends within that many frames."""

    def __init__(
        self,
        plans: Sequence[TrackPlan],
        *,
        restorer,
        device: torch.device,
        feather: int,
        skip_threshold: float = SKIP_MEAN_LUMA_DIFF,
    ) -> None:
        windows = sorted((w for plan in plans for w in plan.windows), key=lambda w: w.start)
        self._weights = {
            w.index: weight for plan in plans for w, weight in zip(plan.windows, crossfade_weights(plan.windows))
        }
        self._tracks = {w.index: plan for plan in plans for w in plan.windows}
        self._windows = {w.index: w for w in windows}
        self._pending = iter(windows)
        self._upcoming = next(self._pending, None)
        self._covering: list[Window] = []
        self._canvases: dict[int, list[torch.Tensor]] = {}
        # Per window: (sum, count) of |luma difference| inside the track mask, and the
        # same outside it within the crop (the re-encode noise, logged for reference).
        self._difference: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._results: dict[int, list[torch.Tensor] | None] = {}
        self._held: deque[_Held] = deque()
        self._pinned = _PinnedFrames()
        self._restorer = restorer
        self._device = device
        self._feather = feather
        self._skip_threshold = skip_threshold
        self._frame_idx = 0
        self.restored: list[int] = []
        self.skipped: list[int] = []

    def frame(self, ltx: torch.Tensor, source: torch.Tensor, pts: int) -> list[tuple[torch.Tensor, int, bool]]:
        """Take the next LTX frame (``[3, H, W]`` uint8 on the device) with its source
        frame; return the frames now ready to write, in order, as
        ``(frame, pts, composited)``."""
        frame_idx = self._frame_idx
        while self._upcoming is not None and self._upcoming.start == frame_idx:
            self._covering.append(self._upcoming)
            self._canvases[self._upcoming.index] = []
            zero = torch.zeros((), dtype=torch.float64, device=ltx.device)
            self._difference[self._upcoming.index] = (zero, zero.clone(), zero.clone(), zero.clone())
            self._upcoming = next(self._pending, None)
        covering = tuple(window.index for window in self._covering)
        for window in list(self._covering):
            local = frame_idx - window.start
            self._canvases[window.index].append(crop_to_canvas_area(ltx, window.crops[local]).cpu())
            self._accumulate_difference(window, frame_idx, ltx, source)
            if local == window.real_frames - 1:
                self._covering.remove(window)
                self._finish(window)
        self._frame_idx += 1
        if not covering and not self._held:
            return [(ltx, pts, False)]
        self._held.append(_Held(frame_idx, pts, self._pinned.take(ltx), covering))
        return self._drain()

    def finish(self) -> list[tuple[torch.Tensor, int, bool]]:
        """The frames still held once the video ends (none when every window was planned
        from this video)."""
        if self._covering or self._upcoming is not None:
            raise RuntimeError("the video ended before every planned window was filled")
        out = self._drain()
        if self._held:
            raise RuntimeError("frames left unwritten after the last window")
        return out

    def _accumulate_difference(self, window: Window, frame_idx: int, ltx: torch.Tensor, source: torch.Tensor) -> None:
        track = self._tracks[window.index]
        x1, y1, x2, y2 = window.crops[frame_idx - window.start].crop_box
        mask = polygon_mask(track.polygons[frame_idx - track.start], (y2 - y1, x2 - x1), (x1, y1), ltx.device)
        difference = (_luma(ltx[:, y1:y2, x1:x2]) - _luma(source[:, y1:y2, x1:x2])).abs_()
        inside, count, outside, outside_count = self._difference[window.index]
        self._difference[window.index] = (
            inside + difference[mask].sum().double(),
            count + mask.sum().double(),
            outside + difference[~mask].sum().double(),
            outside_count + (~mask).sum().double(),
        )

    def _finish(self, window: Window) -> None:
        canvases = torch.stack(self._canvases.pop(window.index))
        inside, count, outside, outside_count = self._difference.pop(window.index)
        mean = float(inside.item()) / max(float(count.item()), 1.0)
        baseline = float(outside.item()) / max(float(outside_count.item()), 1.0)
        track = self._tracks[window.index]
        if mean < self._skip_threshold:
            self._results[window.index] = None
            self.skipped.append(window.index)
            logger.warning(
                "LTX refine: window %d (track %d, frames %d-%d) left as is: the LTX output matches the "
                "source inside the mask (mean luma difference %.2f, below %.2f; %.2f outside the mask)",
                window.index, track.track_id, window.start, window.stop - 1, mean, self._skip_threshold, baseline,
            )
            return
        self._results[window.index] = self._restore(canvases)
        self.restored.append(window.index)
        logger.debug(
            "LTX refine: window %d (track %d, frames %d-%d) restored: mean luma difference %.2f inside the "
            "mask, %.2f outside",
            window.index, track.track_id, window.start, window.stop - 1, mean, baseline,
        )

    def _restore(self, canvases: torch.Tensor) -> list[torch.Tensor]:
        """x2 of a window's canvases (``[T, 3, S, S]`` uint8, host), as host uint8 frames."""
        count = int(canvases.shape[0])
        restored = self._restorer.restore(canvases.float().div_(255.0), keep_start=0, keep_end=count)
        if len(restored) != count:
            raise RuntimeError(f"the restorer returned {len(restored)} frames for a {count}-frame window")
        return [frame.to("cpu") for frame in restored]

    def _drain(self) -> list[tuple[torch.Tensor, int, bool]]:
        out = []
        while self._held and all(index in self._results for index in self._held[0].windows):
            held = self._held.popleft()
            frame = held.frame.to(self._device, copy=True)
            self._pinned.give(held.frame)
            candidates = self._candidates(held)
            composited = composite_frame(frame, candidates, feather=self._feather) if candidates else frame
            out.append((composited, held.pts, bool(candidates)))
            for index in held.windows:
                if self._windows[index].stop - 1 == held.index:
                    del self._results[index]
        return out

    def _candidates(self, held: _Held) -> list[Candidate]:
        candidates = []
        for index in held.windows:
            restored = self._results[index]
            if restored is None:
                continue
            window, track = self._windows[index], self._tracks[index]
            local = held.index - window.start
            candidates.append(
                Candidate(
                    canvas=restored[local].to(self._device),
                    crop=doubled(window.crops[local]),
                    weight=float(self._weights[index][local]),
                    track=track.track_id,
                    polygons=track.polygons[held.index - track.start],
                    presence=track.presence[held.index - track.start],
                )
            )
        return candidates


# -- the two passes ------------------------------------------------------------------------


def scan_source(
    source: Path,
    metadata,
    detector,
    *,
    batch_size: int,
    device: torch.device,
    large_canvas: bool,
    disable_progress: bool,
    cancel: threading.Event,
) -> list[TrackPlan]:
    """LTX's own scan of the source video as one segment. ``large_canvas`` is taken as
    given: whether LTX rendered at 768 px was decided on the machine that ran it."""
    frames = video_frames(
        source, metadata, batch_size=batch_size, device=device, frame_stride=1, seek_ts=None, end_pts=None
    )
    with tqdm(total=int(metadata.num_frames), desc="LTX refine scan", unit="frame", disable=disable_progress) as bar:
        plans = scan_span(
            frames,
            (LtxSegment(*WHOLE_VIDEO_PTS, 0),),
            detector,
            batch_size=batch_size,
            frame_h=int(metadata.video_height),
            frame_w=int(metadata.video_width),
            large_canvas=large_canvas,
            bar=bar,
            cancel=cancel,
        )[0]
    canvases = Counter(window.canvas for plan in plans for window in plan.windows)
    logger.info(
        "LTX refine: windows per canvas: %s",
        ", ".join(f"{count} at {canvas} px" for canvas, count in sorted(canvases.items())) or "none",
    )
    return plans


def lockstep_frames(
    ltx_frames: FrameSource, source_frames: FrameSource
) -> Iterator[tuple[torch.Tensor, torch.Tensor, list[int]]]:
    """Batches of the LTX output and of the source side by side, with the LTX pts."""
    missing = object()
    with closing(ltx_frames()) as ltx_batches, closing(source_frames()) as source_batches:
        for ltx_item, source_item in zip_longest(ltx_batches, source_batches, fillvalue=missing):
            if ltx_item is missing or source_item is missing:
                raise UnsupportedVideoError("the LTX output and --ltx-crop-vsr-source decoded to different frame counts")
            (ltx, pts), (source, _) = ltx_item, source_item
            if len(ltx) != len(source):
                raise UnsupportedVideoError("the LTX output and --ltx-crop-vsr-source decoded to different frame counts")
            yield ltx, source, pts


@dataclass(frozen=True)
class RefineSummary:
    frames: int
    windows: int
    skipped: int
    scan_seconds: float
    compose_seconds: float


def run_refine(
    *,
    ltx_video: Path,
    ltx_metadata,
    source_video: Path,
    source_metadata,
    writer: FrameWriter,
    detector,
    release_detector: Callable[[], None],
    make_restorer: Callable[[], object],
    batch_size: int,
    device: torch.device,
    large_canvas: bool,
    disable_progress: bool,
    progress_callback: ProgressCallback | None,
    cancel: threading.Event,
    skip_threshold: float = SKIP_MEAN_LUMA_DIFF,
) -> RefineSummary:
    """Write the sharpened LTX output to ``writer`` (left open for the caller to close).
    The alignment check comes first and needs no model; the detector is released after
    the scan, before the restorer loads."""
    check_alignment(
        read_stream_times(ltx_video),
        read_stream_times(source_video),
        fps=Fraction(ltx_metadata.video_fps_exact),
        ltx_name=ltx_video.name,
        source_name=source_video.name,
    )
    started = time.monotonic()
    plans = scan_source(
        source_video,
        source_metadata,
        detector,
        batch_size=batch_size,
        device=device,
        large_canvas=large_canvas,
        disable_progress=disable_progress,
        cancel=cancel,
    )
    release_detector()
    scan_seconds = time.monotonic() - started
    windows = sum(len(plan.windows) for plan in plans)
    if not windows:
        logger.warning(
            "LTX refine: no mosaic found in %s; the output is the LTX output re-encoded unchanged", source_video.name
        )
    restorer = make_restorer() if windows else None
    refiner = WindowRefiner(
        plans,
        restorer=restorer,
        device=device,
        feather=feather_pixels(int(ltx_metadata.video_height)),
        skip_threshold=skip_threshold,
    )
    progress = Progressbar(
        total_frames=int(ltx_metadata.num_frames),
        video_fps=float(ltx_metadata.video_fps),
        disable=disable_progress,
        callback=progress_callback,
    )
    started = time.monotonic()
    written = 0

    def write(ready: list[tuple[torch.Tensor, int, bool]]) -> None:
        nonlocal written
        for frame, pts, composited in ready:
            writer.write(frame, pts, apply_lut=composited)
            written += 1
        progress.update(len(ready))

    try:
        ltx_frames = video_frames(
            ltx_video, ltx_metadata, batch_size=batch_size, device=device, frame_stride=1, seek_ts=None, end_pts=None
        )
        source_frames = video_frames(
            source_video, source_metadata, batch_size=batch_size, device=device, frame_stride=1, seek_ts=None, end_pts=None
        )
        for ltx_batch, source_batch, pts in lockstep_frames(ltx_frames, source_frames):
            if cancel.is_set():
                raise Cancelled()
            for ltx, source, frame_pts in zip(ltx_batch, source_batch, pts):
                write(refiner.frame(ltx, source, int(frame_pts)))
        write(refiner.finish())
    finally:
        progress.close(ensure_completed_bar=True)
        if restorer is not None:
            restorer.close()
    compose_seconds = time.monotonic() - started
    logger.info(
        "LTX refine: %d frames, %d windows (%d restored, %d left as is); scan %.0f s, sharpen and compose %.0f s",
        written, windows, len(refiner.restored), len(refiner.skipped), scan_seconds, compose_seconds,
    )
    return RefineSummary(written, windows, len(refiner.skipped), scan_seconds, compose_seconds)

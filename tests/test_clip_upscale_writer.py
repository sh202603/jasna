"""ClipUpscaleWriter: the windows it cuts, the crossfade where two windows overlap, and that
every source frame comes out once, in order, with its own pts (CPU)."""
from __future__ import annotations

import pytest
import torch

from jasna.upscale.clip_writer import ClipUpscaleWriter, overlap_weights


class RecordingWriter:
    def __init__(self):
        self.frames: list[tuple[torch.Tensor, int]] = []
        self.finished = 0
        self.closed = 0
        self.after: list[int] = []

    def write(self, frame, pts):
        self.frames.append((frame, pts))

    def after_write(self, frames_written, stop_event):
        self.after.append(frames_written)

    def finish(self):
        self.finished += 1

    def close(self):
        self.closed += 1


class FakeUpscaler:
    """Returns each frame unchanged, or filled with ``100 * call index`` when ``mark_calls``."""

    def __init__(self, *, mark_calls: bool = False):
        self.windows: list[int] = []
        self.mark_calls = mark_calls
        self.closed = 0

    def upscale(self, frames):
        call = len(self.windows)
        self.windows.append(int(frames.shape[0]))
        for frame in frames:
            yield torch.full_like(frame, 100 * call) if self.mark_calls else frame.clone()

    def close(self):
        self.closed += 1


def _frame(value: int) -> torch.Tensor:
    return torch.full((3, 2, 2), value, dtype=torch.uint8)


def _run(count: int, upscaler: FakeUpscaler, *, window: int = 9, overlap: int = 4) -> RecordingWriter:
    sink = RecordingWriter()
    writer = ClipUpscaleWriter(sink, upscaler, window=window, overlap=overlap)
    for index in range(count):
        writer.write(_frame(index), 10 * index)
    writer.finish()
    writer.close()
    return sink


def test_overlap_weights_ramp_between_the_thirds():
    assert overlap_weights(16) == pytest.approx([0.0] * 5 + [k / 7 for k in range(1, 7)] + [1.0] * 5)
    assert overlap_weights(4) == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert overlap_weights(0) == []


@pytest.mark.parametrize("count", [0, 1, 4, 5, 8, 9, 10, 13, 14, 15, 23, 24, 50])
def test_every_frame_once_in_order_with_its_pts(count):
    upscaler = FakeUpscaler()
    sink = _run(count, upscaler)

    assert [pts for _, pts in sink.frames] == [10 * index for index in range(count)]
    assert [int(frame[0, 0, 0]) for frame, _ in sink.frames] == list(range(count))
    assert sink.finished == 1 and sink.closed == 1 and upscaler.closed == 1
    assert sum(upscaler.windows) >= count
    assert all(size <= 9 for size in upscaler.windows)


def test_windows_advance_by_window_minus_overlap():
    upscaler = FakeUpscaler()
    _run(20, upscaler)

    # 0-8, 5-13, 10-18, then the last window from frame 15: its 4 shared frames and frame 19.
    assert upscaler.windows == [9, 9, 9, 5]


def test_no_extra_window_when_the_video_ends_with_a_window():
    upscaler = FakeUpscaler()
    sink = _run(14, upscaler)

    # 0-8 and 5-13: the frames kept from the second window are its final output.
    assert upscaler.windows == [9, 9]
    assert len(sink.frames) == 14


def test_overlap_is_crossfaded_from_the_earlier_window_to_the_later():
    sink = _run(14, FakeUpscaler(mark_calls=True))

    values = [int(frame[0, 0, 0]) for frame, _ in sink.frames]
    # Frames 5-8 are shared by window 0 (value 0) and window 1 (value 100).
    assert values[:5] == [0] * 5
    assert values[5:9] == [0, 33, 67, 100]
    assert values[9:] == [100] * 5


def test_frames_wait_until_their_window_is_full():
    sink = RecordingWriter()
    writer = ClipUpscaleWriter(sink, FakeUpscaler(), window=9, overlap=4)
    for index in range(8):
        writer.write(_frame(index), index)
    assert sink.frames == []
    writer.write(_frame(8), 8)
    # The window's last 4 frames wait for the next window's version of them.
    assert [pts for _, pts in sink.frames] == [0, 1, 2, 3, 4]


def test_window_must_hold_two_overlaps():
    with pytest.raises(ValueError):
        ClipUpscaleWriter(RecordingWriter(), FakeUpscaler(), window=8, overlap=4)


def test_close_closes_the_upscaler_when_the_writer_fails():
    class FailingWriter(RecordingWriter):
        def close(self):
            raise RuntimeError("mux failed")

    upscaler = FakeUpscaler()
    writer = ClipUpscaleWriter(FailingWriter(), upscaler, window=9, overlap=4)
    with pytest.raises(RuntimeError):
        writer.close()
    assert upscaler.closed == 1


def test_after_write_reaches_the_inner_writer():
    sink = RecordingWriter()
    ClipUpscaleWriter(sink, FakeUpscaler(), window=9, overlap=4).after_write(7, None)
    assert sink.after == [7]

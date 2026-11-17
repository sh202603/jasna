"""StreamUpscaleWriter: frames that come out late keep their order and their timestamps."""
from __future__ import annotations

import pytest

from jasna.upscale.stream_writer import StreamUpscaleWriter


class Sink:
    def __init__(self):
        self.written = []
        self.finished = 0
        self.closed = 0
        self.after = []

    def write(self, frame, pts):
        self.written.append((frame, pts))

    def after_write(self, frames_written, stop_event):
        self.after.append(frames_written)

    def finish(self):
        self.finished += 1

    def close(self):
        self.closed += 1


class Delayed:
    """Emits like RelayVSR: the first ``window`` frames together, then ``window - 1`` at a time."""

    def __init__(self, window, *, keep=0):
        self.window = window
        self.keep = keep
        self.held = []
        self.first = True
        self.closed = 0

    def push(self, frame):
        self.held.append(frame)
        if len(self.held) == (self.window if self.first else self.window - 1):
            self.first = False
            yield from self._flush()

    def _flush(self):
        held, self.held = self.held, []
        yield from (("up", frame) for frame in held)

    def finish(self):
        del self.held[len(self.held) - self.keep:len(self.held)]
        yield from self._flush()

    def close(self):
        self.closed += 1


@pytest.mark.parametrize("count", [0, 1, 3, 4, 5, 7, 8, 11])
def test_every_frame_comes_out_once_with_its_pts(count):
    sink, upscaler = Sink(), Delayed(4)
    writer = StreamUpscaleWriter(sink, upscaler)

    for index in range(count):
        writer.write(f"f{index}", index * 100)
    writer.finish()
    writer.close()

    assert sink.written == [(("up", f"f{index}"), index * 100) for index in range(count)]
    assert sink.finished == 1 and sink.closed == 1 and upscaler.closed == 1


def test_frames_reach_the_writer_as_the_model_emits_them():
    sink = Sink()
    writer = StreamUpscaleWriter(sink, Delayed(4))
    seen = []

    for index in range(8):
        writer.write(index, index)
        seen.append(len(sink.written))

    assert seen == [0, 0, 0, 4, 4, 4, 7, 7]


def test_a_frame_the_upscaler_never_returns_is_an_error():
    sink = Sink()
    writer = StreamUpscaleWriter(sink, Delayed(4, keep=1))
    for index in range(3):
        writer.write(index, index)

    with pytest.raises(RuntimeError, match="kept 1 frame"):
        writer.finish()
    assert sink.finished == 0


def test_upscaler_is_closed_when_the_writer_fails_to_close():
    class Failing(Sink):
        def close(self):
            raise OSError("mux failed")

    upscaler = Delayed(4)
    with pytest.raises(OSError):
        StreamUpscaleWriter(Failing(), upscaler).close()
    assert upscaler.closed == 1


def test_after_write_reaches_the_inner_writer():
    sink = Sink()
    StreamUpscaleWriter(sink, Delayed(4)).after_write(5, None)
    assert sink.after == [5]

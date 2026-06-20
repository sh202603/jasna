"""Unit tests for the standalone frame-gen CLI driver loop (run_framegen).

GPU-free: run_framegen does no tensor ops itself (FrameGenWriter holds frames as
opaque values), so frames are plain sentinels and the reader/encoder/generator are
fakes. This exercises the decode -> FrameGenWriter -> encoder adapter wiring and
the (N-1)*M+1 frame-count / interpolated-PTS contract.
"""
from __future__ import annotations

import jasna.framegen_cli as cli


class FakeReader:
    """Yields (batch, pts_list) like NvidiaVideoReader.frames().

    batch[i] indexes a sentinel frame; pts_list may be shorter than the batch
    (the real decoder yields a partial final batch at EOF).
    """

    def __init__(self, batches):
        self._batches = batches

    def frames(self):
        for batch, pts in self._batches:
            yield batch, pts


class FakeGenerator:
    name = "fake"

    def __init__(self):
        self.calls: list[tuple[object, object, list[float]]] = []
        self.closed = False

    def interpolate(self, frame_a, frame_b, positions):
        self.calls.append((frame_a, frame_b, list(positions)))
        return [("interp", frame_a, frame_b, p) for p in positions]

    def close(self):
        self.closed = True


class FakeEncoderCtx:
    def __init__(self):
        self.encoded: list[tuple[object, int]] = []
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.exited += 1
        return False

    def encode(self, frame, pts):
        self.encoded.append((frame, int(pts)))


def test_2x_count_pts_and_lifecycle():
    reader = FakeReader([(["f0", "f1"], [0, 100]), (["f2"], [200])])
    enc, gen = FakeEncoderCtx(), FakeGenerator()

    source_frames = cli.run_framegen(reader, enc, gen, 2)

    assert source_frames == 3
    pts = [p for _, p in enc.encoded]
    # (N-1)*M + 1 = (3-1)*2 + 1 = 5 emitted, strictly increasing.
    assert pts == [0, 50, 100, 150, 200]
    assert pts == sorted(pts) and len(set(pts)) == len(pts)
    # Encoder entered/exited exactly once (lazy enter on first frame, mux on close).
    assert enc.entered == 1 and enc.exited == 1
    # Generator is borrowed: run_framegen / FrameGenWriter must not close it.
    assert not gen.closed


def test_4x_count_and_pts():
    reader = FakeReader([(["f0", "f1"], [0, 100])])
    enc, gen = FakeEncoderCtx(), FakeGenerator()

    source_frames = cli.run_framegen(reader, enc, gen, 4)

    assert source_frames == 2
    assert [p for _, p in enc.encoded] == [0, 25, 50, 75, 100]
    assert enc.entered == 1 and enc.exited == 1


def test_empty_input_never_enters_encoder():
    # No frames -> encoder context never entered -> close() must not mux an empty
    # stream (the lazy-enter guard in _EncoderWriter).
    reader = FakeReader([])
    enc, gen = FakeEncoderCtx(), FakeGenerator()

    source_frames = cli.run_framegen(reader, enc, gen, 2)

    assert source_frames == 0
    assert enc.entered == 0 and enc.exited == 0
    assert enc.encoded == []


def test_empty_trailing_batch_is_tolerated():
    # The decoder yields a partial/empty final batch at EOF: (tensor, []).
    reader = FakeReader([(["f0", "f1"], [0, 100]), (["unused"], [])])
    enc, gen = FakeEncoderCtx(), FakeGenerator()

    source_frames = cli.run_framegen(reader, enc, gen, 2)

    assert source_frames == 2
    assert [p for _, p in enc.encoded] == [0, 50, 100]


def test_single_frame_no_interpolation():
    reader = FakeReader([(["only"], [42])])
    enc, gen = FakeEncoderCtx(), FakeGenerator()

    source_frames = cli.run_framegen(reader, enc, gen, 2)

    assert source_frames == 1
    assert enc.encoded == [("only", 42)]
    assert gen.calls == []
    assert enc.entered == 1 and enc.exited == 1


def test_progress_counts_source_frames_only():
    class P:
        def __init__(self):
            self.n = 0

        def update(self, k):
            self.n += k

    reader = FakeReader([(["f0", "f1"], [0, 100]), (["f2"], [200])])
    enc, gen, prog = FakeEncoderCtx(), FakeGenerator(), P()

    cli.run_framegen(reader, enc, gen, 2, progress=prog)

    # One update per SOURCE frame (3), not per emitted frame (5).
    assert prog.n == 3

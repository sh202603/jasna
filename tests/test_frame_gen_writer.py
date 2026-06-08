"""Unit tests for FrameGenWriter PTS math and frame counting.

These are intentionally GPU-free: FrameGenWriter performs no tensor ops itself,
so frames are represented by plain sentinels and the generator/writer are fakes.
"""
from __future__ import annotations

from jasna.framegen.frame_gen_writer import FrameGenWriter


class FakeWriter:
    def __init__(self):
        self.written: list[tuple[object, int]] = []
        self.closed = False

    def write(self, frame, pts):
        self.written.append((frame, pts))

    def after_write(self, frames_written):
        pass

    def close(self):
        self.closed = True


class FakeGenerator:
    """Returns interpolation markers: (a, b, position) per requested position."""

    name = "fake"

    def __init__(self):
        self.calls: list[tuple[object, object, list[float]]] = []
        self.closed = False

    def interpolate(self, frame_a, frame_b, positions):
        self.calls.append((frame_a, frame_b, list(positions)))
        return [("interp", frame_a, frame_b, p) for p in positions]

    def close(self):
        self.closed = True


def _run(multiplier, frames_pts):
    writer = FakeWriter()
    gen = FakeGenerator()
    fgw = FrameGenWriter(writer, gen, multiplier)
    for frame, pts in frames_pts:
        fgw.write(frame, pts)
    fgw.close()
    return writer, gen


def test_multiplier_one_passthrough():
    writer, gen = _run(1, [("f0", 0), ("f1", 100), ("f2", 200)])
    assert writer.written == [("f0", 0), ("f1", 100), ("f2", 200)]
    assert gen.calls == []  # no interpolation requested
    assert writer.closed


def test_2x_pts_and_count():
    src = [("f0", 0), ("f1", 100), ("f2", 200)]
    writer, gen = _run(2, src)
    # (N-1)*M + 1 = 2*2 + 1 = 5
    assert len(writer.written) == 5
    pts = [p for _, p in writer.written]
    assert pts == [0, 50, 100, 150, 200]
    # Real source frames land on their original PTS, interps in between.
    assert writer.written[0] == ("f0", 0)
    assert writer.written[2] == ("f1", 100)
    assert writer.written[4] == ("f2", 200)
    assert writer.written[1][0][0] == "interp" and writer.written[1][0][3] == 0.5
    assert writer.closed


def test_4x_pts_and_count():
    src = [("f0", 0), ("f1", 100)]
    writer, gen = _run(4, src)
    # (2-1)*4 + 1 = 5
    assert len(writer.written) == 5
    pts = [p for _, p in writer.written]
    assert pts == [0, 25, 50, 75, 100]
    assert writer.written[0] == ("f0", 0)
    assert writer.written[4] == ("f1", 100)
    positions = [w[0][3] for w in writer.written[1:4]]
    assert positions == [0.25, 0.5, 0.75]


def test_single_frame_flush():
    writer, gen = _run(2, [("only", 42)])
    assert writer.written == [("only", 42)]
    assert gen.calls == []
    assert writer.closed


def test_non_monotonic_pts_skips_interpolation():
    # Second frame has a non-increasing PTS: emit prev only, no interps.
    src = [("f0", 100), ("f1", 100), ("f2", 200)]
    writer, gen = _run(2, src)
    pts = [p for _, p in writer.written]
    # f0@100 (no interp, span=0), then f1@100 + interp@150, then f2@200
    assert pts == [100, 100, 150, 200]
    assert len(gen.calls) == 1  # only the f1->f2 interval interpolated


def test_span_below_multiplier_skips_interpolation():
    # Coarse time_base: span=2 < M=4 cannot hold 3 distinct intermediates without
    # rounding onto a real frame's PTS. Real frames must still all be written.
    src = [("f0", 0), ("f1", 2), ("f2", 6)]
    writer, gen = _run(4, src)
    assert [p for _, p in writer.written][:2] == [0, 2]  # f0->f1 skipped
    assert ("f1", 2) in writer.written and ("f2", 6) in writer.written
    assert len(gen.calls) == 1  # only f1->f2 (span=4) interpolated
    assert [p for _, p in writer.written] == [0, 2, 3, 4, 5, 6]


def test_span_equal_to_multiplier_interpolates():
    # Boundary: span == M yields strictly-between, mutually distinct PTS.
    writer, gen = _run(4, [("f0", 0), ("f1", 4)])
    assert [p for _, p in writer.written] == [0, 1, 2, 3, 4]
    assert len(gen.calls) == 1


def test_non_uniform_pts_proportional():
    # Uneven spacing: intermediates scale with each interval's span.
    src = [("f0", 0), ("f1", 90), ("f2", 300)]
    writer, gen = _run(2, src)
    pts = [p for _, p in writer.written]
    assert pts == [0, 45, 90, 195, 300]


def test_close_closes_writer_but_not_borrowed_generator():
    # The generator is borrowed (owned by the caller that built it, e.g. the CLI
    # folder loop that shares one generator across every video in a batch), so
    # close() must close the real writer but leave the generator alone.
    writer = FakeWriter()
    gen = FakeGenerator()
    fgw = FrameGenWriter(writer, gen, 2)
    fgw.write("f0", 0)
    fgw.write("f1", 100)
    fgw.close()
    assert not gen.closed
    assert writer.closed

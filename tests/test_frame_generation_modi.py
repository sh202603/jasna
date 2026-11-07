"""modi-only frame generation tests: the writer generalised to a multiplier and a generator with
``interpolate``, the borrowed-generator mode of ``jasna-framegen``, and the backend routing.

modi 独自。上流の tests/test_frame_generation.py には追記しない。GPU 不要（フレームは番兵）。
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from jasna.frame_generation import FrameGenerationWriter, build_generator, multiplier_of


class FakeWriter:
    def __init__(self):
        self.written: list[tuple[object, int]] = []
        self.finished = False
        self.closed = False

    def write(self, frame, pts):
        self.written.append((frame, pts))

    def after_write(self, frames_written, stop_event):
        pass

    def finish(self):
        self.finished = True

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


def _run(multiplier, frames_pts, **kwargs):
    writer, gen = FakeWriter(), FakeGenerator()
    fgw = FrameGenerationWriter(writer, gen, multiplier, **kwargs)
    for frame, pts in frames_pts:
        fgw.write(frame, pts)
    fgw.finish()
    fgw.close()
    return writer, gen


def test_multiplier_one_passes_frames_through():
    writer, gen = _run(1, [("f0", 0), ("f1", 100), ("f2", 200)])
    assert writer.written == [("f0", 0), ("f1", 100), ("f2", 200)]
    assert gen.calls == [] and writer.finished and writer.closed


def test_4x_pts_are_scaled_by_the_multiplier_and_the_last_frame_fills_its_interval():
    writer, gen = _run(4, [("f0", 0), ("f1", 100)])
    # Every PTS is in the source time base divided by 4: the generated frames sit at the
    # source interval's quarters, and the last frame is repeated to keep the duration.
    assert [p for _, p in writer.written] == [0, 100, 200, 300, 400, 500, 600, 700]
    assert writer.written[0] == ("f0", 0) and writer.written[4] == ("f1", 400)
    assert [w[0][3] for w in writer.written[1:4]] == [0.25, 0.5, 0.75]
    assert [w[0] for w in writer.written[5:]] == ["f1", "f1", "f1"]
    assert gen.calls == [("f0", "f1", [0.25, 0.5, 0.75])]


def test_2x_with_uneven_intervals_keeps_the_midpoints():
    writer, _ = _run(2, [("f0", 0), ("f1", 90), ("f2", 300)])
    assert [p for _, p in writer.written] == [0, 90, 180, 390, 600, 810]


def test_a_single_frame_is_written_once():
    writer, gen = _run(2, [("only", 42)])
    assert writer.written == [("only", 84)]
    assert gen.calls == []


def test_a_plain_callable_generates_2x_only():
    class Midpoint:
        """Upstream's contract: a callable with close(), no interpolate()."""

        def __call__(self, a, b):
            return ("mid", a, b)

        def close(self):
            pass

    writer = FakeWriter()
    midpoint = Midpoint()
    fgw = FrameGenerationWriter(writer, midpoint, 2)
    fgw.write("a", 0)
    fgw.write("b", 10)
    fgw.finish()
    assert writer.written == [("a", 0), (("mid", "a", "b"), 10), ("b", 20), ("b", 30)]
    with pytest.raises(ValueError, match="2x"):
        FrameGenerationWriter(FakeWriter(), midpoint, 4)


def test_close_owns_the_generator_unless_told_otherwise():
    writer, gen = _run(2, [("f0", 0), ("f1", 100)])
    assert gen.closed and writer.closed
    writer, gen = _run(2, [("f0", 0), ("f1", 100)], owns_generator=False)
    assert not gen.closed and writer.closed


def test_a_stopped_job_closes_without_the_held_frame():
    writer, gen = FakeWriter(), FakeGenerator()
    fgw = FrameGenerationWriter(writer, gen, 2)
    fgw.write("a", 0)
    fgw.write("b", 1)
    fgw.close()
    assert writer.written == [("a", 0), (("interp", "a", "b", 0.5), 1)]
    assert not writer.finished and writer.closed and gen.closed


def test_multiplier_of_the_cli_values():
    assert [multiplier_of(v) for v in ("none", "2x", "4x")] == [1, 2, 4]
    with pytest.raises(ValueError):
        multiplier_of("3x")


def test_build_generator_routes_the_backends():
    with patch("jasna.frame_generation.RtxFrameGeneration") as rtx:
        build_generator("rtx", device="cuda:0", width=8, height=6)
        build_generator("rtx", device="cuda:0", width=8, height=6, rtx_mode="high")
    assert rtx.call_args_list[0].args == ("cuda:0", 8, 6) and rtx.call_args_list[0].kwargs == {}
    assert rtx.call_args_list[1].kwargs == {"mode": "high"}
    with patch("jasna.framegen.rife_frame_generator.RifeFrameGenerator") as rife:
        build_generator("rife", device="cuda:0", width=8, height=6, model_path="/w/rife.pth", fp16=False)
    assert rife.call_args.kwargs == {"device": "cuda:0", "model_path": "/w/rife.pth", "fp16": False}
    with pytest.raises(ValueError, match="backend"):
        build_generator("dain", device="cuda:0", width=8, height=6)

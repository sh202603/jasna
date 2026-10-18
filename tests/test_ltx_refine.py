"""``--ltx-crop-vsr`` (jasna.ltx.refine): alignment of the LTX output with its source,
the canvases, and the window driver with a fake x2 restorer."""

from __future__ import annotations

from fractions import Fraction

import av
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from jasna.ltx import camera, plan as plan_module, refine
from jasna.ltx.compose import Candidate, composite_frame, crop_to_canvas
from jasna.ltx.plan import WINDOW_HOP, Region, feather_pixels, plan_video
from jasna.media.probe import UnsupportedVideoError

FPS = Fraction(30000, 1001)


def _times(pts, time_base=Fraction(1, 30000), size=(64, 48)):
    return refine.StreamTimes(width=size[0], height=size[1], time_base=time_base, pts=tuple(pts))


def _check(ltx, source):
    refine.check_alignment(ltx, source, fps=FPS, ltx_name="ltx.mp4", source_name="src.mp4")


class TestAlignment:
    def test_ignores_the_start_offset_and_the_time_base(self):
        ltx = _times([i * 1001 for i in range(10)])
        _check(ltx, _times([990 + i * 1001 for i in range(10)]))  # start_pts 990 (test2)
        _check(ltx, _times([round(i * 1001 / 30) for i in range(10)], time_base=Fraction(1, 1000)))  # MKV

    def test_rejects_a_different_frame_count(self):
        with pytest.raises(UnsupportedVideoError, match="10 frames.*9"):
            _check(_times([i * 1001 for i in range(10)]), _times([i * 1001 for i in range(9)]))

    def test_rejects_a_frame_off_by_more_than_half_a_frame(self):
        source = [i * 1001 for i in range(10)]
        source[5] += 600
        with pytest.raises(UnsupportedVideoError, match="frame 5"):
            _check(_times([i * 1001 for i in range(10)]), _times(source))

    def test_rejects_a_different_frame_size(self):
        with pytest.raises(UnsupportedVideoError, match="frame size"):
            _check(_times([0, 1001]), _times([0, 1001], size=(32, 48)))


def _write_video(path, frames: int, *, rate: int = 30) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 20, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_stream_times_align_across_containers(tmp_path):
    _write_video(tmp_path / "a.mp4", 12)
    _write_video(tmp_path / "b.mkv", 12)
    _write_video(tmp_path / "c.mp4", 11)
    a, b, c = (refine.read_stream_times(tmp_path / name) for name in ("a.mp4", "b.mkv", "c.mp4"))
    assert len(a.pts) == len(b.pts) == 12 and a.time_base != b.time_base
    refine.check_alignment(a, b, fps=Fraction(30), ltx_name="a", source_name="b")
    with pytest.raises(UnsupportedVideoError, match="12 frames"):
        refine.check_alignment(a, c, fps=Fraction(30), ltx_name="a", source_name="c")


# -- canvases ------------------------------------------------------------------------------


def test_doubled_crop_keeps_the_box_and_doubles_the_canvas():
    crop = camera.letterbox((10, 20, 110, 70), 128)
    twice = refine.doubled(crop)
    assert twice.crop_box == crop.crop_box
    assert twice.resize_hw == (2 * crop.resize_hw[0], 2 * crop.resize_hw[1])
    assert twice.pad == tuple(2 * p for p in crop.pad)
    assert twice.canvas == 2 * crop.canvas


def test_crop_to_canvas_area_matches_crop_to_canvas_at_unit_scale_and_pads_alike():
    frame = torch.randint(0, 256, (3, 120, 160), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))
    crop = camera.letterbox((20, 10, 84, 110), 100)  # 64x100 content letterboxed into 100: no resize, side pads
    assert torch.equal(refine.crop_to_canvas_area(frame, crop), crop_to_canvas(frame, crop))
    shrunk = refine.crop_to_canvas_area(frame, camera.letterbox((0, 0, 160, 120), 64))
    assert shrunk.shape == (3, 64, 64)
    enlarged = refine.crop_to_canvas_area(frame, camera.letterbox((0, 0, 40, 30), 64))
    assert enlarged.shape == (3, 64, 64)


# -- the window driver ---------------------------------------------------------------------


class _FakeRestorer:
    """x2 by interpolation; records the window lengths it saw."""

    name = "fake"
    prefers_cpu_input = True

    def __init__(self, mode: str = "bicubic"):
        self.mode = mode
        self.calls: list[int] = []
        self.closed = False

    def restore(self, frames, *, keep_start, keep_end):
        self.calls.append(int(frames.shape[0]))
        if self.mode == "nearest":
            up = F.interpolate(frames, scale_factor=2, mode="nearest")
        else:
            up = F.interpolate(frames, scale_factor=2, mode=self.mode, align_corners=False)
        return list(up.clamp(0, 1).mul(255).round().to(torch.uint8).unbind(0))[keep_start:keep_end]

    def close(self):
        self.closed = True


def _square(x1, y1, x2, y2) -> Region:
    return Region(box=(float(x1), float(y1), float(x2), float(y2)), polygon=np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32))


def _plans(frames: int, mosaics: list[tuple[Region, range]], *, cuts=(), size=(64, 64)):
    regions = [[region for region, span in mosaics if i in span] for i in range(frames)]
    return plan_video(regions, set(cuts), frame_w=size[1], frame_h=size[0], large_canvas=False, window_hop=WINDOW_HOP)


def _frames(count: int, *, seed: int, size=(64, 64)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 256, (3, *size), dtype=torch.uint8, generator=generator) for _ in range(count)]


def _run(refiner, ltx: list[torch.Tensor], source: list[torch.Tensor]):
    out = []
    for index, (a, b) in enumerate(zip(ltx, source)):
        out.extend(refiner.frame(a, b, 10 * index))
    out.extend(refiner.finish())
    return out


@pytest.fixture
def small_canvas(monkeypatch):
    """A 64 px plan canvas: a 64x64 frame needs no resize, so the fake x2 is cheap."""
    monkeypatch.setattr(camera, "PLAN_CANVAS", 64)
    monkeypatch.setattr(plan_module, "PLAN_CANVAS", 64)


def test_frames_come_out_in_order_and_complete(small_canvas):
    plans = _plans(150, [(_square(10, 10, 40, 40), range(3, 140))])
    assert sum(len(plan.windows) for plan in plans) == 2  # 137 frames: windows at 0 and 88 of the track
    restorer = _FakeRestorer("nearest")
    ltx, source = _frames(150, seed=1), _frames(150, seed=2)
    refiner = refine.WindowRefiner(plans, restorer=restorer, device=torch.device("cpu"), feather=feather_pixels(64))
    out = _run(refiner, ltx, source)
    assert [pts for _, pts, _ in out] == [10 * i for i in range(150)]
    assert restorer.calls == [window.real_frames for plan in plans for window in plan.windows]
    assert refiner.restored == [0, 1] and refiner.skipped == []
    # Frames no window covers pass through untouched and are not marked composited.
    assert all(not composited and torch.equal(frame, ltx[i]) for i, (frame, _, composited) in enumerate(out) if i < 3 or i >= 140)
    assert all(composited for _, _, composited in out[3:140])


def test_two_tracks_and_a_scene_cut_keep_every_frame(small_canvas):
    plans = _plans(
        150,
        [(_square(4, 4, 24, 24), range(0, 100)), (_square(40, 40, 60, 60), range(60, 150))],
        cuts=(45,),
    )
    assert len(plans) >= 3  # the first track is cut at 45; the second overlaps both halves
    restorer = _FakeRestorer("nearest")
    refiner = refine.WindowRefiner(plans, restorer=restorer, device=torch.device("cpu"), feather=feather_pixels(64))
    out = _run(refiner, _frames(150, seed=3), _frames(150, seed=4))
    assert [pts for _, pts, _ in out] == [10 * i for i in range(150)]
    assert sorted(refiner.restored) == sorted(w.index for plan in plans for w in plan.windows)


def test_a_window_ltx_left_alone_is_skipped(small_canvas, caplog):
    plans = _plans(30, [(_square(10, 10, 40, 40), range(0, 30))])
    ltx = _frames(30, seed=5)
    restorer = _FakeRestorer("nearest")
    refiner = refine.WindowRefiner(plans, restorer=restorer, device=torch.device("cpu"), feather=feather_pixels(64))
    with caplog.at_level("WARNING", logger="jasna.ltx.refine"):
        out = _run(refiner, ltx, [frame.clone() for frame in ltx])  # the source is the LTX output: nothing restored
    assert refiner.skipped == [0] and restorer.calls == []
    assert "left as is" in caplog.text
    assert all(torch.equal(frame, ltx[i]) and not composited for i, (frame, _, composited) in enumerate(out))


def test_a_window_ltx_restored_is_sharpened(small_canvas):
    plans = _plans(30, [(_square(10, 10, 40, 40), range(0, 30))])
    source = _frames(30, seed=6)
    ltx = [frame.clone() for frame in source]
    for frame in ltx:
        frame[:, 10:40, 10:40] = 128  # LTX repainted the mosaic
    restorer = _FakeRestorer("nearest")
    refiner = refine.WindowRefiner(plans, restorer=restorer, device=torch.device("cpu"), feather=feather_pixels(64))
    out = _run(refiner, ltx, source)
    assert refiner.restored == [0] and restorer.calls == [30]
    assert all(composited for _, _, composited in out)


def test_finish_rejects_a_video_shorter_than_its_plan(small_canvas):
    plans = _plans(30, [(_square(10, 10, 40, 40), range(0, 30))])
    refiner = refine.WindowRefiner(plans, restorer=_FakeRestorer("nearest"), device=torch.device("cpu"), feather=2)
    for index, (a, b) in enumerate(zip(_frames(20, seed=7), _frames(20, seed=8))):
        refiner.frame(a, b, index)
    with pytest.raises(RuntimeError, match="ended before"):
        refiner.finish()


def test_composite_matches_the_plumbing_free_reference():
    """Outside the mask the LTX frame is bit-exact; inside, the driver's result equals the
    same canvases pushed through the public functions by hand (real 512 px canvas)."""
    frames = 8
    plans = _plans(frames, [(_square(10, 10, 40, 40), range(0, frames))])
    (plan,) = plans
    (window,) = plan.windows
    restorer = _FakeRestorer("bicubic")
    ltx = _frames(frames, seed=9)
    refiner = refine.WindowRefiner(plans, restorer=restorer, device=torch.device("cpu"), feather=feather_pixels(64))
    out = _run(refiner, ltx, [torch.zeros_like(frame) for frame in ltx])
    assert restorer.calls == [frames]
    canvases = torch.stack([refine.crop_to_canvas_area(ltx[i], window.crops[i]) for i in range(frames)])
    sharpened = _FakeRestorer("bicubic").restore(canvases.float() / 255, keep_start=0, keep_end=frames)
    feather = feather_pixels(64)
    for i, (frame, _, composited) in enumerate(out):
        assert composited
        candidate = Candidate(
            canvas=sharpened[i], crop=refine.doubled(window.crops[i]), weight=1.0, track=plan.track_id,
            polygons=plan.polygons[i], presence=plan.presence[i],
        )
        expected = composite_frame(ltx[i], [candidate], feather=feather)
        assert torch.equal(frame, expected)
        assert not torch.equal(frame, ltx[i])  # the x2 round trip does change the mosaic area
        # Beyond the feather of the mask the LTX frame is untouched.
        far = ltx[i].clone()
        far[:, : 40 + 2 * feather + 2, : 40 + 2 * feather + 2] = frame[:, : 40 + 2 * feather + 2, : 40 + 2 * feather + 2]
        assert torch.equal(frame, far)


def test_scan_source_hands_the_window_hop_to_ltx_scan(monkeypatch):
    """The refine scan must plan with the hop LTX used (v0.11.2 made it a required
    argument of ``scan_span``); the plans come back as the first segment's list."""
    import threading
    from pathlib import Path
    from types import SimpleNamespace

    seen = {}
    plan_a = SimpleNamespace(windows=(SimpleNamespace(canvas=512),))
    plan_b = SimpleNamespace(windows=(SimpleNamespace(canvas=512), SimpleNamespace(canvas=768)))

    def fake_scan_span(frames, segments, detector, **kwargs):
        seen.update(kwargs)
        seen["segments"] = segments
        return [[plan_a, plan_b]]

    monkeypatch.setattr(refine, "scan_span", fake_scan_span)
    monkeypatch.setattr(refine, "video_frames", lambda *args, **kwargs: iter(()))
    metadata = SimpleNamespace(num_frames=3, video_height=48, video_width=64)
    plans = refine.scan_source(
        Path("source.mp4"), metadata, detector=None, batch_size=2, device=torch.device("cpu"),
        large_canvas=False, window_hop=104, disable_progress=True, cancel=threading.Event(),
    )
    assert plans == [plan_a, plan_b]
    assert seen["window_hop"] == 104
    assert seen["large_canvas"] is False and seen["frame_h"] == 48 and seen["frame_w"] == 64
    assert len(seen["segments"]) == 1

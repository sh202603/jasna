"""modi's policy on upstream's LTX super resolution (``jasna/ltx/super_res.py``): with a deblur
pass every window is upscaled 2x, not only the ones compose enlarges; the upscalers of the
run load before the first window decodes and are freed after the compose pass; such runs
keep a speed profile of their own (``+rtx``). Without a deblur pass upstream's choice holds.

modi 独自。上流の tests/test_ltx_restore.py の fixture（``fakes``）と helper、
tests/test_ltx_compose.py の ``_frame`` に依存する。上流のファイルには追記しない。
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from jasna.ltx import restore
from jasna.ltx.camera import CropFrame, letterbox
from jasna.ltx.compose import Candidate, canvas_to_source, composite_frame, crop_to_canvas
from jasna.ltx.plan import Window
from jasna.ltx.super_res import LtxSuperRes, enlarged_windows, upscaled_crop
from test_ltx_compose import _frame
from test_ltx_restore import FILES, MAX_GAP, _Bar, _Detector, _render, _source, _video, _Writer, fakes  # noqa: F401


class _FakeSuperRes:
    """An ``LtxSuperRes`` that doubles every pixel and records its life."""

    instances: list["_FakeSuperRes"] = []

    def __init__(self, *, sharpens: bool = True):
        self.sharpens = sharpens
        self.prepared: list[list[int]] = []
        self.upscaled: list[tuple[int, ...]] = []
        self.released = 0
        _FakeSuperRes.instances.append(self)

    windows_to_upscale = LtxSuperRes.windows_to_upscale

    def prepare(self, sizes):
        self.prepared.append(list(sizes))

    def upscale(self, canvas):
        self.upscaled.append(tuple(canvas.shape))
        return canvas.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)

    def release(self):
        self.released += 1


def _compose_one_window(monkeypatch, crop: CropFrame, canvas: int, super_res: _FakeSuperRes) -> list:
    candidates_seen = []

    def decode(decoder, latent, seed, canvas_side, budget, device):
        return torch.full((5, 3, canvas, canvas), 100, dtype=torch.uint8)

    def composite(source, candidates, *, feather):
        candidates_seen.append([(tuple(c.canvas.shape), c.crop) for c in candidates])
        return source

    windows = (Window(0, canvas, 0, 5, (crop,) * 5),)
    plan = restore.TrackPlan(0, 0, ((),) * 5, (1.0,) * 5, windows, 3)
    monkeypatch.setattr(restore, "_decode_window", decode)
    monkeypatch.setattr(restore, "composite_frame", composite)
    decodes = ThreadPoolExecutor(max_workers=1)
    composer = restore._SegmentComposer(
        [plan], None, SimpleNamespace(get=lambda index: index), seed=0, feather=1, budget=lambda side: 0,
        device=torch.device("cpu"), bar=_Bar(), decodes=decodes, stream=None, super_res=super_res,
    )
    try:
        for _ in range(5):
            composer.frame(torch.zeros(3, 4, 4, dtype=torch.uint8))
    finally:
        decodes.shutdown()
    return candidates_seen


def test_with_a_deblur_pass_a_window_compose_shrinks_is_upscaled_too(monkeypatch):
    crop = CropFrame((0, 0, 4, 4), (4, 4), (0, 0, 0, 0))  # crop side 4 <= canvas 4: compose never enlarges it
    super_res = _FakeSuperRes(sharpens=True)
    seen = _compose_one_window(monkeypatch, crop, 4, super_res)
    assert seen == [[((3, 8, 8), upscaled_crop(crop))]] * 5
    assert super_res.upscaled == [(3, 4, 4)] * 5  # one frame at a time, never the window


def test_without_a_deblur_pass_only_the_windows_compose_enlarges_are_upscaled(monkeypatch):
    shrunk = CropFrame((0, 0, 4, 4), (4, 4), (0, 0, 0, 0))
    super_res = _FakeSuperRes(sharpens=False)
    assert _compose_one_window(monkeypatch, shrunk, 4, super_res) == [[((3, 4, 4), shrunk)]] * 5
    assert super_res.upscaled == []

    enlarged = CropFrame((0, 0, 8, 8), (4, 4), (0, 0, 0, 0))  # crop side 8 > canvas 4
    super_res = _FakeSuperRes(sharpens=False)
    assert _compose_one_window(monkeypatch, enlarged, 4, super_res) == [[((3, 8, 8), upscaled_crop(enlarged))]] * 5


def _render_with_super_res(tmp_path, super_res):
    for budget in ("segment+rtx", "free VRAM+rtx"):
        restore.speed.save_rates(
            tmp_path / "speed.json", restore.speed.profile_key("Test GPU", "t", budget), {"denoise/window512": 4.0}
        )
    return replace(_render(tmp_path), super_res=super_res)


def _restore_spans(tmp_path, spans, render):
    restore.restore_spans(
        spans, render, work_dir=tmp_path, resume_dir=tmp_path / "saved",
        progress=restore.Progress(0, disable=True, report=None), cancel=threading.Event(),
    )


def test_compose_spans_prepares_every_canvas_of_the_run_once_and_releases_it(fakes, monkeypatch, tmp_path):
    video = _video(30, range(3, 28))
    writer = _Writer()
    segments = (restore.LtxSegment(50, 150, seed=100), restore.LtxSegment(200, 260, seed=7))
    super_res = _FakeSuperRes()
    render = _render_with_super_res(tmp_path, super_res)
    saved = []
    monkeypatch.setattr(restore.speed, "save_rates", lambda path, key, rates: saved.append(key))
    _restore_spans(tmp_path, [restore.LtxSpan(_source(video), segments, lambda: writer)], render)
    assert super_res.prepared == [[restore.PLAN_CANVAS]]
    assert super_res.released == 1 and writer.closed
    # The run's profile is kept apart from runs without super resolution (its compose rate holds the x2).
    assert saved == [restore.speed.profile_key("Test GPU", "t", "segment+rtx")]
    assert restore._profile_key(_render(tmp_path)).endswith("decode segment")


def test_compose_spans_releases_the_upscalers_when_a_window_decode_fails(fakes, monkeypatch, tmp_path):
    def decode(decoder, latent, free_bytes, generator):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(restore, "decode_latent", decode)
    super_res = _FakeSuperRes()
    video = _video(12, range(2, 10))
    with pytest.raises(RuntimeError, match="decode failed"):
        _restore_spans(
            tmp_path,
            [restore.LtxSpan(_source(video), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 0),), _Writer)],
            _render_with_super_res(tmp_path, super_res),
        )
    assert super_res.prepared == [[restore.PLAN_CANVAS]] and super_res.released == 1


def test_a_run_without_windows_prepares_nothing(fakes, tmp_path):
    super_res = _FakeSuperRes()
    _restore_spans(
        tmp_path,
        [restore.LtxSpan(_source(_video(12, range(0))), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 0),), _Writer)],
        _render_with_super_res(tmp_path, super_res),
    )
    assert super_res.prepared == [] and super_res.upscaled == []


def test_restore_video_hands_super_res_to_the_compose_pass(fakes, tmp_path):
    restore.speed.save_rates(
        tmp_path / "speed.json", restore.speed.profile_key("Test GPU", "t", "free VRAM+rtx"), {"denoise/window512": 4.0}
    )
    super_res = _FakeSuperRes()
    restore.restore_video(
        _source(_video(12, range(2, 10))), _Writer(), detector=_Detector(), files=FILES, frame_h=64, frame_w=64,
        batch_size=4, large_canvas=False, window_hop=restore.WINDOW_HOP, seed=5, device=torch.device("cpu"),
        work_dir=tmp_path, speed_profile=tmp_path / "speed.json", progress=restore.Progress(12, disable=True, report=None),
        resume_dir=tmp_path / "saved", cancel=threading.Event(), max_gap=MAX_GAP, min_duration=0, super_res=super_res,
    )
    assert super_res.prepared == [[restore.PLAN_CANVAS]] and super_res.released == 1
    assert super_res.upscaled


def test_ltx_super_res_loads_one_upscaler_per_size_with_the_rtx_options_and_releases_them():
    windows = [Window(0, 512, 0, 1, (CropFrame((0, 0, 400, 400), (512, 512), (0, 0, 0, 0)),))]
    with (
        patch("jasna.restorer.rtx_superres_secondary_restorer.RtxSuperresSecondaryRestorer") as rtx_cls,
        patch("torch.cuda.default_stream"),
        patch("torch.cuda.stream", return_value=nullcontext()),
    ):
        rtx_cls.side_effect = lambda **kwargs: MagicMock(kwargs=kwargs)
        super_res = LtxSuperRes(
            device=torch.device("cuda:0"), quality="high", denoise="none", deblur="low", strength=0.75, highbitrate=True
        )
        assert super_res.sharpens and super_res.windows_to_upscale(windows) == {0}
        super_res.prepare([512, 768, 512])
        upscalers = dict(super_res._upscalers)
        assert sorted(upscalers) == [512, 768]
        kwargs = upscalers[768].kwargs
        assert kwargs["scale"] == 2 and kwargs["input_size"] == 768 and kwargs["denoise"] is None
        assert kwargs["deblur"] == "low" and kwargs["strength"] == 0.75 and kwargs["highbitrate"] is True
        super_res.release()
        assert super_res._upscalers == {} and all(u.close.called for u in upscalers.values())
        super_res.prepare([512])  # made again for the next compose pass
        assert sorted(super_res._upscalers) == [512]

    plain = LtxSuperRes(device=torch.device("cuda:0"), quality="high", denoise="medium", deblur="none")
    assert not plain.sharpens and plain.windows_to_upscale(windows) == enlarged_windows(windows) == set()


def test_an_upscaled_canvas_maps_back_to_the_same_crop():
    """An x2 canvas (here the exact pixel doubling) with the upscaled crop lands on the same
    source box with the same pixels: the x2 padding is skipped and the area shrink undoes
    the doubling."""
    frame = _frame()
    crop = letterbox((0, 0, 200, 100), 200)  # letterboxed: pads above and below
    canvas = crop_to_canvas(frame, crop)
    twice = canvas.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    assert upscaled_crop(crop).pad == tuple(2 * p for p in crop.pad) and upscaled_crop(crop).canvas == 400
    assert torch.equal(canvas_to_source(twice, upscaled_crop(crop)), canvas_to_source(canvas, crop))
    assert torch.equal(canvas_to_source(twice, upscaled_crop(crop)).to(torch.uint8), frame[:, 0:100, 0:200])


def test_an_upscaled_canvas_composites_only_the_feathered_mask():
    source = _frame(1)
    crop = letterbox((200, 100, 456, 356), 256)
    restored = torch.full((3, 512, 512), 128, dtype=torch.uint8)
    polygon = [[300.0, 200.0], [360.0, 200.0], [360.0, 260.0], [300.0, 260.0]]
    out = composite_frame(source, [Candidate(restored, upscaled_crop(crop), 1.0, 0, [polygon], 1.0)], feather=8)
    changed = (out != source).any(dim=0)
    ys, xs = torch.nonzero(changed, as_tuple=True)
    assert changed[230, 330] and int(out[0, 230, 330]) == 128
    assert xs.min() >= 300 - 2 * 8 - 1 and xs.max() <= 360 + 2 * 8 + 1
    assert ys.min() >= 200 - 2 * 8 - 1 and ys.max() <= 260 + 2 * 8 + 1

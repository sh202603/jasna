"""``--secondary-restoration swiftvr-inline`` / ``swiftvr-distill`` with an LTX model
(``jasna/ltx/window_restorer.py``): the compose pass hands each decoded window, whole and
once, to the x2 restorer ``--ltx-crop-vsr`` uses, without the frames that never reach the
output; the restorer lives from the start to the end of a compose pass; the CLI and the
session wire it without auto split only.

modi 独自。factories.py と tests/test_ltx_restore.py、test_ltx_sharpen_cli.py の helper に依存する。
上流のファイルには追記しない。
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from factories import NullProgressBar, cli_args, cli_environment, session_config, track_plan
from jasna.ltx import compose_pass, restore
from jasna.ltx.camera import CropFrame
from jasna.ltx.super_res import upscaled_crop
from jasna.ltx.window_restorer import LtxWindowRestorer, kept_frames
from jasna.ltx.windows import UNFUSED_LEAD_FRAMES, Window, crossfade_weights
from test_ltx_restore import _render, fakes  # noqa: F401

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


class _FakeRestorer:
    """Doubles every pixel and adds 1, so a restored frame differs from a merely doubled one."""

    def __init__(self):
        self.clips: list[torch.Tensor] = []
        self.closed = 0

    def restore(self, frames, *, keep_start, keep_end):
        assert frames.dtype == torch.float32 and float(frames.max()) <= 1.0
        self.clips.append(frames.clone())
        u8 = frames.mul(255.0).round().to(torch.uint8) + 1
        return list(u8.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)[keep_start:keep_end].unbind(0))

    def close(self):
        self.closed += 1


def _window_restorer():
    made: list[_FakeRestorer] = []

    def make():
        made.append(_FakeRestorer())
        return made[-1]

    return LtxWindowRestorer("swiftvr-distill", make), made


def test_kept_frames_span_the_frames_with_a_compose_weight():
    assert kept_frames(np.ones(5, dtype=np.float32)) == range(0, 5)
    assert kept_frames(np.array([0, 0, 0.5, 1, 0.5, 0], dtype=np.float32)) == range(2, 5)
    assert kept_frames(np.zeros(3, dtype=np.float32)) == range(0)


def test_a_later_windows_unfused_lead_frames_are_not_kept():
    crop = CropFrame((0, 0, 4, 4), (4, 4), (0, 0, 0, 0))
    windows = [Window(0, 0, 0, 40, (crop,) * 40, 0), Window(1, 0, 20, 40, (crop,) * 40, 1)]
    left, right = crossfade_weights(windows)
    assert kept_frames(right) == range(UNFUSED_LEAD_FRAMES, 40)
    assert kept_frames(left) == range(0, 39)  # its last frame has faded out completely


def test_a_window_is_restored_from_its_kept_frames_only():
    window_restorer, made = _window_restorer()
    video = torch.arange(6, dtype=torch.uint8).mul(10).reshape(6, 1, 1, 1).expand(6, 3, 4, 4).contiguous()
    out = window_restorer.upscale_window(video, range(2, 5))
    assert out.shape == (6, 3, 8, 8) and out.dtype == torch.uint8 and out.device.type == "cpu"
    (clip,) = made[0].clips
    assert clip.shape == (3, 3, 4, 4) and torch.equal(clip[:, 0, 0, 0].mul(255).round(), torch.tensor([20.0, 30.0, 40.0]))
    assert out[:, 0, 0, 0].tolist() == [0, 10, 21, 31, 41, 50]  # kept frames restored, the others doubled


def test_the_restorer_lives_from_prepare_to_release():
    window_restorer, made = _window_restorer()
    assert window_restorer.profile_tag == "+swiftvr-distill" and window_restorer.sharpens
    window_restorer.prepare([512])
    window_restorer.prepare([512, 768])
    assert len(made) == 1
    window_restorer.release()
    window_restorer.release()
    assert made[0].closed == 1
    window_restorer.upscale_window(torch.zeros(2, 3, 4, 4, dtype=torch.uint8), range(0, 2))
    assert len(made) == 2  # made again for the next compose pass


def test_a_restorer_returning_the_wrong_frames_is_an_error():
    class Short(_FakeRestorer):
        def restore(self, frames, *, keep_start, keep_end):
            return super().restore(frames, keep_start=keep_start, keep_end=keep_end)[:-1]

    with pytest.raises(RuntimeError, match="returned 1 frames for a 2-frame window"):
        LtxWindowRestorer("swiftvr-inline", Short).upscale_window(torch.zeros(2, 3, 4, 4, dtype=torch.uint8), range(0, 2))

    class SameSize(_FakeRestorer):
        def restore(self, frames, *, keep_start, keep_end):
            return list(frames.mul(255).to(torch.uint8).unbind(0))

    with pytest.raises(RuntimeError, match="expected uint8"):
        LtxWindowRestorer("swiftvr-inline", SameSize).upscale_window(
            torch.zeros(2, 3, 4, 4, dtype=torch.uint8), range(0, 2)
        )


def test_compose_restores_each_window_whole_and_once(monkeypatch):
    crop = CropFrame((0, 0, 4, 4), (4, 4), (0, 0, 0, 0))
    windows = [Window(0, 0, 0, 30, (crop,) * 30, 0), Window(1, 0, 15, 30, (crop,) * 30, 1)]
    window_restorer, made = _window_restorer()
    seen = []

    def decode(decoder, latent, seed, canvas_side, budget, device):
        return torch.full((32, 3, 4, 4), 100, dtype=torch.uint8)  # the model's padded tail included

    def composite(source, candidates, *, feather, projector=None):
        seen.append([(tuple(c.canvas.shape), c.crop, int(c.canvas[0, 0, 0]), c.weight > 0) for c in candidates])
        return source

    monkeypatch.setattr(compose_pass, "_decode_window", decode)
    monkeypatch.setattr(compose_pass, "composite_frame", composite)
    pool = ThreadPoolExecutor(max_workers=1)
    composer = compose_pass._SegmentComposer(
        [track_plan(windows)], None, SimpleNamespace(get=lambda index: index), seed=0, feather=1,
        budget=lambda side: 0, device=torch.device("cpu"), bar=NullProgressBar(), decodes=pool, receives=pool,
        latent_reads=pool, stream=None, super_res=window_restorer, projector=None,
    )
    try:
        for _ in range(45):
            composer.frame(torch.zeros(3, 4, 4, dtype=torch.uint8))
    finally:
        pool.shutdown()
    # One clip per window: the first without its faded-out last frame, the second without its lead.
    assert [len(clip) for clip in made[0].clips] == [29, 30 - UNFUSED_LEAD_FRAMES]
    assert all(shape == (3, 8, 8) and c == upscaled_crop(crop) for frame in seen for shape, c, _, _ in frame)
    # Every candidate with a weight carries restored pixels (101), never the plain doubling (100).
    assert all(value == 101 for frame in seen for _, _, value, live in frame if live)
    assert [len(frame) for frame in seen] == [1] * 15 + [2] * 15 + [1] * 15


def test_decoding_and_restoring_take_turns_on_the_gpu(monkeypatch):
    """The next window never decodes while one is restored: their peaks would add up."""
    crop = CropFrame((0, 0, 4, 4), (4, 4), (0, 0, 0, 0))
    windows = [Window(index, 0, 15 * index, 30, (crop,) * 30, index) for index in range(4)]
    window_restorer, _ = _window_restorer()
    busy, overlaps = [], []

    def decode(decoder, latent, seed, canvas_side, budget, device):
        overlaps.append(bool(busy))
        return torch.zeros((30, 3, 4, 4), dtype=torch.uint8)

    upscale_window = window_restorer.upscale_window

    def restoring(video, kept):
        import time

        busy.append(1)
        time.sleep(0.05)  # long enough for the decode thread to start the next window if it could
        busy.pop()
        return upscale_window(video, kept)

    monkeypatch.setattr(compose_pass, "_decode_window", decode)
    monkeypatch.setattr(compose_pass, "composite_frame", lambda source, candidates, *, feather, projector=None: source)
    monkeypatch.setattr(window_restorer, "upscale_window", restoring)
    pool, reads = ThreadPoolExecutor(max_workers=1), ThreadPoolExecutor(max_workers=1)
    composer = compose_pass._SegmentComposer(
        [track_plan(windows)], None, SimpleNamespace(get=lambda index: index), seed=0, feather=1,
        budget=lambda side: 0, device=torch.device("cpu"), bar=NullProgressBar(), decodes=pool, receives=pool,
        latent_reads=reads, stream=None, super_res=window_restorer, projector=None,
    )
    try:
        for _ in range(75):
            composer.frame(torch.zeros(3, 4, 4, dtype=torch.uint8))
    finally:
        pool.shutdown()
        reads.shutdown()
    assert len(overlaps) == 4 and not any(overlaps)


def test_the_speed_profile_is_keyed_by_the_restorer(fakes, tmp_path):
    window_restorer, _ = _window_restorer()
    key = restore._profile_key(replace(_render(tmp_path), super_res=window_restorer))
    assert key.endswith("+swiftvr-distill") and not key.endswith("+rtx")


def _run_main(tmp_path, extra_args):
    argv = cli_args(tmp_path, "--restoration-model-name", "ltx-8step", "--ltx-trial", *extra_args)
    with (
        cli_environment() as pipeline_cls,
        patch("jasna.accelerator.is_nvidia_device", return_value=True),
        patch("jasna.ltx.transformer.require_kernels"),
        patch("jasna.restorer.swiftvr_distill_model.resolve_swiftvr_distill_model_path", return_value=tmp_path / "d.pt"),
        patch("jasna.restorer.swiftvr_common.resolve_swiftvr_paths", return_value=(tmp_path,) * 3),
        patch.object(sys, "argv", ["jasna", *argv]),
    ):
        from jasna.main import main

        main()
    return pipeline_cls


class TestCli:
    @pytest.mark.parametrize("secondary", ["swiftvr-inline", "swiftvr-distill"])
    def test_the_swiftvr_restorers_reach_the_pipeline_with_an_ltx_model(self, tmp_path, secondary):
        config = _run_main(tmp_path, ["--secondary-restoration", secondary]).call_args.kwargs["config"]
        assert config.restoration_model_name == "ltx" and config.secondary_restoration == secondary

    def test_swiftvr_scale_4_is_refused_when_spelled_out(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main(tmp_path, ["--secondary-restoration", "swiftvr-inline", "--swiftvr-scale", "4"])
        assert "runs SwiftVR at scale 2; --swiftvr-scale 4 is not supported" in capsys.readouterr().err

    def test_distill_stabilize_is_refused_on_the_large_canvas(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main(
                tmp_path,
                ["--secondary-restoration", "swiftvr-distill", "--swiftvr-distill-stabilize", "2", "--ltx-large-canvas"],
            )
        assert "cannot combine --ltx-large-canvas and --swiftvr-distill-stabilize" in capsys.readouterr().err


class TestSession:
    def _session(self, **overrides):
        from jasna import session_factory

        config = session_config(restoration_model_name="ltx", ltx_trial=True, ltx_large_canvas=False, **overrides)
        with (
            patch.object(session_factory, "_validate_window_restorer") as validate,
            patch.object(session_factory, "_build_secondary_restorer") as build,
            patch("jasna.accelerator.is_nvidia_device", return_value=True),
            patch("jasna.ltx.transformer.require_kernels"),
            patch("jasna.engine_compiler.ensure_engines_compiled"),
        ):
            session = session_factory.build_restoration_session(config, log_callback=None)
            if session.ltx_super_res is not None:
                session.ltx_super_res.prepare([512])
        return session, validate, build

    def test_swiftvr_runs_at_scale_2_without_view_smoothing_sized_from_the_canvases(self):
        session, validate, build = self._session(
            secondary_restoration="swiftvr-inline", swiftvr_scale=4, swiftvr_view_window=15
        )
        assert isinstance(session.ltx_super_res, LtxWindowRestorer) and session.ltx_super_res.name == "swiftvr-inline"
        refined = build.call_args.args[0]
        assert refined.swiftvr_scale == 2 and refined.swiftvr_view_window == 0
        assert build.call_args.kwargs == {"size_from_frames": True}
        assert validate.call_args.args[0].swiftvr_scale == 2  # paths checked before anything loads

    def test_distill_runs_without_view_smoothing(self):
        session, _, build = self._session(secondary_restoration="swiftvr-distill", swiftvr_distill_view_window=15)
        assert session.ltx_super_res.name == "swiftvr-distill"
        assert build.call_args.args[0].swiftvr_distill_view_window == 0

    def test_nothing_is_built_before_a_compose_pass(self):
        from jasna import session_factory

        config = session_config(restoration_model_name="ltx", ltx_trial=True, secondary_restoration="swiftvr-distill")
        with (
            patch.object(session_factory, "_validate_window_restorer"),
            patch.object(session_factory, "_build_secondary_restorer") as build,
            patch("jasna.accelerator.is_nvidia_device", return_value=True),
            patch("jasna.ltx.transformer.require_kernels"),
            patch("jasna.engine_compiler.ensure_engines_compiled"),
        ):
            session_factory.build_restoration_session(config, log_callback=None)
        assert not build.called

    def test_a_worker_restorer_on_the_large_canvas_needs_a_card_above_16_gb(self):
        from types import SimpleNamespace as NS

        from jasna import session_factory

        config = session_config(
            restoration_model_name="ltx", secondary_restoration="swiftvr-inline", ltx_large_canvas=True
        )
        device = torch.device("cuda:0")
        with patch("torch.cuda.get_device_properties", return_value=NS(total_memory=16 << 30)):
            with pytest.raises(ValueError, match="needs a GPU with 20 GB or more"):
                session_factory._check_worker_beside_large_canvas(config, device)
            with pytest.raises(ValueError, match="relayvsr with an LTX model and --ltx-large-canvas"):
                session_factory._check_worker_beside_large_canvas(
                    replace(config, secondary_restoration="relayvsr"), device
                )
            session_factory._check_worker_beside_large_canvas(replace(config, ltx_large_canvas=False), device)
            session_factory._check_worker_beside_large_canvas(
                replace(config, secondary_restoration="swiftvr-distill"), device
            )
        with patch("torch.cuda.get_device_properties", return_value=NS(total_memory=24 << 30)):
            session_factory._check_worker_beside_large_canvas(config, device)

    def test_other_jobs_get_no_window_restorer(self):
        from jasna.session_factory import restores_ltx_windows

        ltx = session_config(restoration_model_name="ltx", secondary_restoration="swiftvr-distill")
        assert restores_ltx_windows(ltx)
        assert not restores_ltx_windows(replace(ltx, ltx_auto_split_percent=12))  # stays with the BasicVSR++ parts
        assert not restores_ltx_windows(replace(ltx, secondary_restoration="rtx-super-res"))
        assert not restores_ltx_windows(replace(ltx, restoration_model_name="basicvsrpp"))


def test_the_pipeline_hands_the_sessions_window_restorer_to_ltx():
    from pathlib import Path
    from unittest.mock import MagicMock

    from factories import FixedDetectorSession
    from jasna.pipeline import Pipeline

    window_restorer, _ = _window_restorer()

    def pipeline(**config):
        session = FixedDetectorSession(
            device=torch.device("cpu"), restoration_pipeline=None, log_callback=None, ltx_files=MagicMock(),
            ltx_super_res=window_restorer, detector=MagicMock(),
        )
        return Pipeline(
            config=session_config(restoration_model_name="ltx", ltx_trial=True, **config), session=session,
            input_video=Path("in.mp4"), output_video=Path("out.mkv"), progress_callback=None, segments=None,
            splice_plan=None,
        )

    assert pipeline(secondary_restoration="swiftvr-distill").ltx_super_res is window_restorer
    assert pipeline(secondary_restoration="none").ltx_super_res is None
    # With auto split the restorer belongs to the BasicVSR++ parts' frame pipeline.
    assert pipeline(secondary_restoration="swiftvr-distill", ltx_auto_split_percent=12).ltx_super_res is None

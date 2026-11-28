"""+modi: upstream's whole-video upscaling (``--upscale``, v0.12.3) next to modi's own paths."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from factories import make_pipeline, video_metadata
from jasna import update_check


class _Writer:
    def write(self, frame, pts):
        pass

    def after_write(self, frames_written, stop_event):
        pass

    def finish(self):
        pass

    def close(self):
        pass


@pytest.mark.usefixtures("nvidia_vendor")
@pytest.mark.parametrize("backend", ["rtx", "rife"])
def test_frame_generation_after_upscaling_is_built_at_the_upscaled_size(backend):
    pipeline = make_pipeline(
        device=torch.device("cuda", 0), upscale_model="span", upscale_height=1080,
        frame_generation="2x", frame_generation_backend=backend,
    )
    metadata = video_metadata(video_width=960, video_height=540)

    with patch("jasna.pipeline.build_frame_upscaler"), \
            patch("jasna.frame_generation.build_generator") as build_generator:
        pipeline._frame_writer(_Writer(), metadata)

    assert build_generator.call_args.args == (backend,)
    assert (build_generator.call_args.kwargs["width"], build_generator.call_args.kwargs["height"]) == (1920, 1080)


@pytest.mark.usefixtures("nvidia_vendor")
def test_frame_generation_without_upscaling_keeps_the_source_size():
    pipeline = make_pipeline(device=torch.device("cuda", 0), frame_generation="4x", frame_generation_backend="rife")
    metadata = video_metadata(video_width=960, video_height=540)

    with patch("jasna.frame_generation.build_generator") as build_generator:
        pipeline._frame_writer(_Writer(), metadata)

    assert (build_generator.call_args.kwargs["width"], build_generator.call_args.kwargs["height"]) == (960, 540)


@pytest.mark.parametrize(
    ("version", "current", "newer"),
    [("0.12.4", "0.12.3+modi", True), ("0.12.3", "0.12.3+modi", False), ("0.12.2", "0.12.3+modi", False)],
)
def test_the_update_check_compares_the_release_part_of_the_fork_version(version, current, newer):
    assert update_check.is_newer(version, current) is newer


# --- --upscale swiftvr / relayvsr in the restoration pipeline, and the 720 px height ---


class _Worker:
    alive = True
    closed = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def close(self):
        self.closed = True


def _worker_pipeline(model, **overrides):
    return make_pipeline(
        device=torch.device("cuda", 0), upscale_model=model, upscale_height=1080,
        swiftvr_repo="sv", relayvsr_repo="rv", **overrides,
    )


def test_720_is_an_upscale_height():
    from jasna.cli_parser import build_parser
    from jasna.session_config import UPSCALE_HEIGHTS
    from jasna.upscale import upscaled_size

    assert UPSCALE_HEIGHTS[0] == 720
    assert build_parser().parse_args(["--upscale", "span", "--upscale-height", "720"]).upscale_height == 720
    assert upscaled_size(640, 480, 720) == (960, 720)
    assert upscaled_size(1280, 720, 720) is None


def test_every_locale_names_every_upscale_height():
    from jasna.gui.locales import en, ja, ko, th, zh
    from jasna.session_config import UPSCALE_HEIGHTS

    for module in (en, ja, ko, th, zh):
        text = open(module.__file__, encoding="utf-8").read()
        for height in UPSCALE_HEIGHTS:
            assert f'"upscale_height_{height}"' in text, (module.__name__, height)


@pytest.mark.parametrize("model", ["swiftvr", "relayvsr"])
def test_the_cli_offers_the_worker_upscalers(model):
    from jasna.cli_parser import build_parser

    args = build_parser().parse_args(["--upscale", model, "--upscale-swiftvr-tile", "0.8", "--upscale-swiftvr-window", "49"])
    assert (args.upscale, args.upscale_swiftvr_tile, args.upscale_swiftvr_window) == (model, 0.8, 49)


@pytest.mark.parametrize("model", ["swiftvr", "relayvsr"])
@pytest.mark.parametrize("secondary", ["unet-4x", "tvai", "rtx-super-res", "swiftvr-inline", "swiftvr-distill"])
def test_a_worker_upscaler_refuses_every_secondary_restoration(model, secondary):
    from factories import session_config

    with pytest.raises(ValueError, match=f"--upscale {model} cannot be combined with --secondary-restoration"):
        session_config(upscale_model=model, secondary_restoration=secondary)


@pytest.mark.parametrize("model", ["swiftvr", "relayvsr"])
def test_a_worker_upscaler_refuses_frame_generation(model):
    from factories import session_config

    with pytest.raises(ValueError, match="cannot be combined with --frame-generation"):
        session_config(upscale_model=model, frame_generation="2x")


def _job_args(*argv):
    from jasna.cli_parser import build_parser

    return build_parser().parse_args(list(argv))


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--upscale", "swiftvr", "--secondary-restoration", "unet-4x"], "cannot be combined with --secondary-restoration unet-4x"),
        (["--upscale", "relayvsr", "--secondary-restoration", "swiftvr-inline"], "cannot be combined with --secondary-restoration"),
        (["--upscale", "relayvsr", "--frame-generation", "2x"], "cannot be combined with --frame-generation"),
        (["--upscale", "swiftvr"], "--swiftvr-repo is required for --upscale swiftvr"),
        (["--upscale", "relayvsr"], "--relayvsr-repo is required for --upscale relayvsr"),
        (["--upscale", "swiftvr", "--swiftvr-repo", "x", "--upscale-swiftvr-window", "32"], "at least 33"),
        (["--upscale", "swiftvr", "--swiftvr-repo", "x", "--upscale-swiftvr-tile", "0"], "must be positive"),
    ],
)
def test_the_job_checks_refuse_what_a_worker_upscaler_cannot_run_beside(argv, message):
    from jasna import cli_modi

    with pytest.raises(ValueError, match=message):
        cli_modi.check_job_options(_job_args(*argv), restoration_backend="basicvsrpp", is_streaming=False, videos=[])


def test_swiftvr_upscaling_writes_through_overlapping_windows_of_one_worker_per_job(tmp_path):
    from jasna.upscale.clip_writer import ClipUpscaleWriter

    pipeline = _worker_pipeline("swiftvr", upscale_swiftvr_tile=0.8, upscale_swiftvr_window=49)
    metadata = video_metadata(video_width=640, video_height=480)
    paths = (tmp_path, tmp_path / "py", tmp_path / "ckpt")

    with patch("jasna.restorer.swiftvr_common.resolve_swiftvr_paths", return_value=paths), \
            patch("jasna.upscale.swiftvr.SwiftvrWorker", _Worker), \
            patch("jasna.pipeline.build_frame_upscaler") as build_frame_upscaler:
        first = pipeline._frame_writer(_Writer(), metadata)
        worker = pipeline._session.upscale_worker
        second = pipeline._frame_writer(_Writer(), metadata)

        assert isinstance(first, ClipUpscaleWriter) and isinstance(second, ClipUpscaleWriter)
        assert (first._window, first._overlap) == (49, 16)
        assert pipeline._session.upscale_worker is worker  # shared by the job's videos
        assert worker.kwargs["tile_megapixels"] == 0.8
        assert (worker.kwargs["warmup_size"], worker.kwargs["warmup_scale"]) == ((640, 480), 2)
        build_frame_upscaler.assert_not_called()

        worker.alive = False  # a video failed in the middle of a window
        pipeline._frame_writer(_Writer(), metadata)
        assert worker.closed and pipeline._session.upscale_worker is not worker

    replacement = pipeline._session.upscale_worker
    pipeline._session.close()
    assert replacement.closed and pipeline._session.upscale_worker is None


def test_relayvsr_upscaling_writes_through_a_stream(tmp_path):
    from jasna.upscale.stream_writer import StreamUpscaleWriter

    pipeline = _worker_pipeline("relayvsr", relayvsr_window_size=12)
    metadata = video_metadata(video_width=640, video_height=480)
    paths = (tmp_path, tmp_path / "py", tmp_path / "weights")

    with patch("jasna.restorer.relayvsr_common.resolve_relayvsr_paths", return_value=paths), \
            patch("jasna.upscale.relayvsr.RelayvsrWorker", _Worker):
        writer = pipeline._frame_writer(_Writer(), metadata)

    assert isinstance(writer, StreamUpscaleWriter)
    worker = pipeline._session.upscale_worker
    assert (worker.kwargs["scale"], worker.kwargs["window_size"]) == (2, 12)


def test_a_worker_upscaler_leaves_a_video_that_is_tall_enough_alone():
    pipeline = _worker_pipeline("swiftvr")

    writer = _Writer()
    assert pipeline._frame_writer(writer, video_metadata(video_width=1920, video_height=1080)) is writer
    assert pipeline._session.upscale_worker is None


def test_the_settings_tag_records_the_worker_upscaler():
    from factories import session_config
    from jasna.settings_tag import _upscale_settings

    swiftvr = _upscale_settings(session_config(upscale_model="swiftvr", upscale_height=720, upscale_swiftvr_tile=0.8))
    assert (swiftvr["model"], swiftvr["height"], swiftvr["tile"], swiftvr["window"]) == ("swiftvr", 720, 0.8, 89)
    relayvsr = _upscale_settings(session_config(upscale_model="relayvsr"))
    assert (relayvsr["window_size"], relayvsr["wan_weights"]) == (16, "fp8")


def test_a_worker_upscale_writer_offers_its_gpu_turn_and_holds_it_while_the_worker_runs():
    from jasna.upscale.clip_writer import ClipUpscaleWriter
    from jasna.upscale.stream_writer import StreamUpscaleWriter

    held = []

    class _Clip:
        def upscale(self, frames):
            held.append(clip.gpu_turn.locked())
            yield from frames

        def close(self):
            pass

    class _Stream:
        def push(self, frame):
            held.append(stream.gpu_turn.locked())
            yield frame

        def finish(self):
            held.append(stream.gpu_turn.locked())
            return iter(())

        def close(self):
            pass

    clip = ClipUpscaleWriter(_Writer(), _Clip(), window=3, overlap=1)
    stream = StreamUpscaleWriter(_Writer(), _Stream())
    for writer in (clip, stream):
        for pts in range(4):
            writer.write(torch.zeros(3, 2, 2, dtype=torch.uint8), pts)
        writer.finish()
        assert not writer.gpu_turn.locked()

    assert held and all(held)


def test_the_ltx_window_decoder_takes_the_writers_gpu_turn():
    from jasna.ltx.compose_pass import _SegmentComposer

    composer = _SegmentComposer.__new__(_SegmentComposer)
    composer._device = torch.device("cuda", 0)
    turn = object()
    composer.share_gpu_turn(turn)
    assert composer._gpu_turn is turn and composer._frees_after_decode


def _gpu_bytes(gib):
    from types import SimpleNamespace

    return patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=gib << 30))


@pytest.mark.parametrize(
    ("model", "kv_cache", "gib", "expected"),
    [("ltx", "auto", 16, "cpu"), ("ltx", "auto", 24, "auto"), ("ltx", "gpu", 16, "gpu"), ("basicvsrpp", "auto", 16, "auto")],
)
def test_relayvsr_beside_ltx_keeps_its_kv_cache_in_host_memory_on_a_small_card(tmp_path, model, kv_cache, gib, expected):
    pipeline = _worker_pipeline("relayvsr", restoration_model_name=model, relayvsr_kv_cache=kv_cache)
    paths = (tmp_path, tmp_path / "py", tmp_path / "weights")

    with patch("jasna.restorer.relayvsr_common.resolve_relayvsr_paths", return_value=paths), \
            patch("jasna.upscale.relayvsr.RelayvsrWorker", _Worker), _gpu_bytes(gib):
        pipeline._frame_writer(_Writer(), video_metadata(video_width=640, video_height=480))

    assert pipeline._session.upscale_worker.kwargs["kv_cache"] == expected


@pytest.mark.parametrize(
    ("model", "restoration", "height", "gib", "warns"),
    [("relayvsr", "ltx", 2160, 16, True), ("relayvsr", "ltx", 1080, 16, False), ("relayvsr", "ltx", 2160, 24, False),
     ("relayvsr", "basicvsrpp", 2160, 16, False), ("swiftvr", "ltx", 2160, 16, False)],
)
def test_an_ltx_job_warns_before_a_relayvsr_upscale_known_not_to_fit(model, restoration, height, gib, warns):
    from factories import session_config
    from jasna.session_factory import worker_upscale_ltx_warning
    from jasna.upscale import upscaled_size

    config = session_config(upscale_model=model, upscale_height=height, restoration_model_name=restoration)
    with _gpu_bytes(gib):
        warning = worker_upscale_ltx_warning(config, torch.device("cuda", 0), (640, 480), upscaled_size(640, 480, height))

    assert (warning is not None) == warns

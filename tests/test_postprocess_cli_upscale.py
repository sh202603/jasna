"""jasna-postprocess --upscale: the argument checks, the writer order, the encoder's upscaled
metadata, and that ``UpscaleSettings`` is all session_factory's upscale functions read.

GPU-free and native-lib-free like test_postprocess_cli_device.py."""
from __future__ import annotations

import sys
import types
from typing import get_args
from unittest.mock import MagicMock, patch

import pytest
import torch

import jasna.postprocess_cli as cli
from jasna.session_config import UpscaleModelName


class FakeReader:
    def __init__(self, batches):
        self._batches = batches

    def frames(self):
        yield from self._batches


class FakeEncoderCtx:
    def __init__(self):
        self.encoded = []
        self.exited = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exited += 1
        return False

    def encode(self, frame, pts):
        self.encoded.append((frame, int(pts)))


class FakeGenerator:
    def interpolate(self, frame_a, frame_b, positions):
        return [("mid", frame_a, frame_b) for _ in positions]

    def close(self):
        raise AssertionError("the generator is borrowed")


class FakeUpscaler:
    def __init__(self, fail_at=None):
        self.closed = 0
        self.fail_at = fail_at

    def __call__(self, frame):
        if frame == self.fail_at:
            raise RuntimeError("upscale failed")
        return ("up", frame)

    def close(self):
        self.closed += 1


def _settings(model: str, **overrides) -> cli.UpscaleSettings:
    values = dict(
        device="cuda:0", fp16=True, upscale_model=model, upscale_height=2160, upscale_strength=0.8,
        upscale_rtx_quality="high", upscale_rtx_denoise="low", upscale_rtx_deblur="none",
        upscale_dlss5_preset="m", upscale_dlss5_style="natural", upscale_dlss5_intensity=1.5,
    )
    values.update(overrides)
    return cli.UpscaleSettings(**values)


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #

def _usage_error(argv, capsys) -> str:
    with patch.object(sys, "argv", ["jasna-postprocess", "--input", "in.mkv", "--output", "out.mkv", *argv]):
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
    assert exit_info.value.code == 2
    return capsys.readouterr().err


def test_no_step_is_a_usage_error(capsys):
    assert "nothing to do" in _usage_error([], capsys)


def test_swiftvr_refuses_frame_generation(capsys):
    err = _usage_error(["--upscale", "swiftvr", "--swiftvr-repo", "x", "--frame-generation", "2x"], capsys)
    assert "cannot be combined with --frame-generation" in err


def test_swiftvr_needs_its_repo(capsys):
    assert "--swiftvr-repo is required" in _usage_error(["--upscale", "swiftvr"], capsys)


def test_swiftvr_repo_is_checked_before_the_gpu(tmp_path, capsys):
    err = _usage_error(["--upscale", "swiftvr", "--swiftvr-repo", str(tmp_path / "absent")], capsys)
    assert "--swiftvr-repo not found" in err


def test_swiftvr_window_must_hold_two_overlaps(capsys):
    err = _usage_error(["--upscale", "swiftvr", "--swiftvr-repo", "x", "--upscale-swiftvr-window", "32"], capsys)
    assert "--upscale-swiftvr-window must be at least 33" in err


def test_strength_out_of_range_is_a_usage_error(capsys):
    assert "--upscale-strength" in _usage_error(["--upscale", "rtx-super-res", "--upscale-strength", "1.5"], capsys)


def test_parser_offers_jasnas_upscalers_and_swiftvr():
    action = next(a for a in cli.build_parser()._actions if a.dest == "upscale")
    assert action.choices == [*get_args(UpscaleModelName), "swiftvr"]
    assert action.default == "none"


def test_parser_defaults_match_the_modules_they_are_copied_from():
    from jasna.restorer import swiftvr_frame_worker
    from jasna.upscale import clip_writer

    args = cli.build_parser().parse_args(["--input", "a", "--output", "b"])
    assert args.upscale_swiftvr_window == clip_writer.DEFAULT_WINDOW
    assert cli._SWIFTVR_WINDOW_OVERLAP == clip_writer.DEFAULT_OVERLAP
    assert args.upscale_swiftvr_tile == swiftvr_frame_worker.DEFAULT_TILE_MEGAPIXELS
    assert args.frame_generation == "none"


# --------------------------------------------------------------------------- #
# Writer chain
# --------------------------------------------------------------------------- #

def test_frames_are_upscaled_before_they_are_generated():
    reader = FakeReader([(["f0", "f1"], [0, 100])])
    enc, upscaler = FakeEncoderCtx(), FakeUpscaler()

    count = cli.run_postprocess(reader, enc, generator=FakeGenerator(), multiplier=2, upscaler=upscaler)

    assert count == 2
    assert enc.encoded == [
        (("up", "f0"), 0),
        (("mid", ("up", "f0"), ("up", "f1")), 100),
        (("up", "f1"), 200),
        (("up", "f1"), 300),
    ]
    assert upscaler.closed == 1 and enc.exited == 1


def test_upscale_alone_keeps_the_frames_and_pts():
    reader = FakeReader([(["f0", "f1"], [0, 100]), (["f2"], [200])])
    enc, upscaler = FakeEncoderCtx(), FakeUpscaler()

    cli.run_postprocess(reader, enc, upscaler=upscaler)

    assert enc.encoded == [(("up", "f0"), 0), (("up", "f1"), 100), (("up", "f2"), 200)]


def test_upscaler_and_encoder_are_closed_when_a_frame_fails():
    reader = FakeReader([(["f0", "f1"], [0, 100])])
    enc, upscaler = FakeEncoderCtx(), FakeUpscaler(fail_at="f1")

    with pytest.raises(RuntimeError, match="upscale failed"):
        cli.run_postprocess(reader, enc, upscaler=upscaler)

    assert upscaler.closed == 1 and enc.exited == 1


def test_clip_upscaler_gets_windows_and_every_frame_comes_out():
    class Clip:
        def __init__(self):
            self.windows = []
            self.closed = 0

        def upscale(self, frames):
            self.windows.append(int(frames.shape[0]))
            yield from frames

        def close(self):
            self.closed += 1

    frames = torch.arange(40, dtype=torch.uint8).view(40, 1, 1, 1).expand(40, 3, 2, 2)
    reader = FakeReader([(frames[:25], list(range(25))), (frames[25:], list(range(25, 40)))])
    enc, clip = FakeEncoderCtx(), Clip()

    count = cli.run_postprocess(reader, enc, clip_upscaler=clip, window=33)

    assert count == 40
    assert clip.windows == [33, 23]  # the second starts 16 frames back, at frame 17
    assert [pts for _, pts in enc.encoded] == list(range(40))
    assert [int(frame[0, 0, 0]) for frame, _ in enc.encoded] == list(range(40))
    assert clip.closed == 1


# --------------------------------------------------------------------------- #
# UpscaleSettings in place of a SessionConfig
# --------------------------------------------------------------------------- #

def test_settings_give_the_output_size():
    from jasna.session_factory import frame_upscale_size

    assert frame_upscale_size(_settings("rtx-super-res"), 1920, 1080) == (3840, 2160)
    assert frame_upscale_size(_settings("swiftvr", upscale_height=1440), 1920, 1080) == (2560, 1440)
    assert frame_upscale_size(_settings("span"), 3840, 2160) is None


def test_settings_build_every_upscaler():
    from jasna.session_factory import build_frame_upscaler

    device, source, output = torch.device("cpu"), (1920, 1080), (3840, 2160)
    with (
        patch("jasna.upscale.rtx.RtxFrameUpscaler") as rtx,
        patch("jasna.upscale.dlss5.Dlss5FrameUpscaler") as dlss5,
        patch("jasna.upscale.network.NetworkFrameUpscaler") as network,
    ):
        for model in get_args(UpscaleModelName):
            if model != "none":
                build_frame_upscaler(_settings(model), device, source, output)

    assert rtx.call_args.kwargs == dict(
        source_size=source, output_size=output, quality="high", denoise="low", deblur="none", strength=0.8
    )
    assert dlss5.call_args.kwargs["preset"] == "m" and dlss5.call_args.kwargs["intensity"] == 1.5
    assert network.call_count == 2
    assert network.call_args.kwargs["fp16"] is True and network.call_args.kwargs["strength"] == 0.8


def test_settings_request_the_network_engines():
    from jasna.session_factory import ensure_frame_upscale_engines

    with patch("jasna.engine_compiler.ensure_engines_compiled") as compile_engines:
        ensure_frame_upscale_engines(_settings("span"), (1920, 1080), (3840, 2160), log_callback=None)
        ensure_frame_upscale_engines(_settings("rtx-super-res"), (1920, 1080), (3840, 2160), log_callback=None)
        ensure_frame_upscale_engines(_settings("swiftvr"), (1920, 1080), (3840, 2160), log_callback=None)

    request = compile_engines.call_args.args[0]
    assert compile_engines.call_count == 1
    assert request.device == "cuda:0" and request.fp16 is True and request.upscale_model == "span"
    assert request.upscale_input_sizes == [(1920, 1080)]


def test_rtx_upscale_is_refused_on_amd():
    with patch("jasna.accelerator.is_amd_device", return_value=True):
        assert "NVIDIA" in cli._upscale_unavailable_reason(_settings("rtx-super-res"))
    with patch("jasna.accelerator.is_amd_device", return_value=False):
        assert cli._upscale_unavailable_reason(_settings("rtx-super-res")) is None
    assert cli._upscale_unavailable_reason(_settings("span")) is None


def test_dlss5_reason_comes_from_its_module():
    with patch("jasna.upscale.dlss5.dlss5_unavailable_reason", return_value="no runtime") as reason:
        assert cli._upscale_unavailable_reason(_settings("dlss5")) == "no runtime"
    assert reason.call_args.args[0] == torch.device("cuda:0")


# --------------------------------------------------------------------------- #
# main(): one video end to end with stubbed media
# --------------------------------------------------------------------------- #

class _NoOpDeviceContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _run_main(tmp_path, argv, *, width=1920, height=1080, upscaler=None, is_dir=False, frame="frame"):
    captured: dict = {"encoders": []}

    class Reader:
        def __init__(self, file, batch_size, device, metadata, frame_stride=1):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def frames(self):
            yield [frame], [0]

    class Encoder:
        def __init__(self, file, device, metadata, **kwargs):
            captured["encoders"].append({"file": file, "metadata": metadata, **kwargs})

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def encode(self, frame, pts):
            captured.setdefault("encoded", []).append((frame, pts))

    decoder_mod = types.ModuleType("jasna.media.video_decoder")
    decoder_mod.VideoReader = Reader
    encoder_mod = types.ModuleType("jasna.media.video_encoder")
    encoder_mod.VideoEncoder = Encoder

    meta = MagicMock()
    meta.num_frames = 1
    meta.video_width, meta.video_height = width, height

    if is_dir:
        input_path = tmp_path / "in"
        input_path.mkdir()
        (input_path / "a.mp4").touch()
        output = tmp_path / "out"
    else:
        input_path = tmp_path / "in.mkv"
        input_path.touch()
        output = tmp_path / "out.mkv"

    with (
        patch("jasna.postprocess_cli.check_ascii_install_path", return_value=(True, "/fake")),
        patch("jasna.postprocess_cli.check_supported_gpu", return_value=(True, "Fake GPU")),
        patch("jasna.postprocess_cli.check_gpu_driver_version", return_value=(True, "590.18")),
        patch("jasna.postprocess_cli.check_required_executables"),
        patch("jasna.postprocess_cli.check_windows_nvidia_sysmem_fallback_policy", return_value=(True, "OK")),
        patch("jasna.postprocess_cli._upscale_unavailable_reason", return_value=None),
        patch("jasna.media.probe.get_video_meta_data", return_value=meta),
        patch("jasna.framegen.build_frame_generator") as build_generator,
        patch("jasna.session_factory.build_frame_upscaler", return_value=upscaler or FakeUpscaler()) as build_upscaler,
        patch("jasna.session_factory.ensure_frame_upscale_engines") as ensure_engines,
        patch("jasna.upscale.upscaled_metadata", side_effect=lambda metadata, size: ("upscaled", size)),
        patch.dict(sys.modules, {
            "jasna.media.video_decoder": decoder_mod,
            "jasna.media.video_encoder": encoder_mod,
        }),
        patch("torch.cuda.device", side_effect=lambda d: _NoOpDeviceContext()),
        patch.object(sys, "argv", [
            "jasna-postprocess", "--input", str(input_path), "--output", str(output), "--no-progress", *argv,
        ]),
    ):
        cli.main()

    captured.update(build_generator=build_generator, build_upscaler=build_upscaler, ensure_engines=ensure_engines)
    return captured


def test_upscale_only_run_hands_the_encoder_the_upscaled_metadata(tmp_path):
    upscaler = FakeUpscaler()
    captured = _run_main(tmp_path, ["--upscale", "rtx-super-res", "--upscale-height", "1440"], upscaler=upscaler)

    (encoder,) = captured["encoders"]
    assert encoder["metadata"] == ("upscaled", (2560, 1440))
    assert encoder["frame_generation"] == 1
    assert encoder["settings"] is None  # the output carries no settings tag
    assert captured["encoded"] == [(("up", "frame"), 0)]
    assert upscaler.closed == 1

    settings, device, source, output = captured["build_upscaler"].call_args.args
    assert isinstance(settings, cli.UpscaleSettings) and settings.upscale_model == "rtx-super-res"
    assert (source, output) == ((1920, 1080), (2560, 1440))
    assert captured["ensure_engines"].call_args.args[:3] == (settings, (1920, 1080), (2560, 1440))
    captured["build_generator"].assert_not_called()


def test_video_already_tall_enough_is_not_reencoded(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        _run_main(tmp_path, ["--upscale", "rtx-super-res", "--upscale-height", "1080"])

    assert exit_info.value.code == 1
    assert "already 1080 px tall" in capsys.readouterr().out


def test_folder_batch_skips_a_video_that_is_tall_enough(tmp_path, capsys):
    captured = _run_main(tmp_path, ["--upscale", "rtx-super-res", "--upscale-height", "1080"], is_dir=True)

    assert captured["encoders"] == []
    out = capsys.readouterr().out
    assert "Skipping a.mp4" in out and "0 processed, 1 failed/skipped" in out


def test_tall_video_still_gets_its_frames_generated(tmp_path):
    captured = _run_main(
        tmp_path, ["--upscale", "rtx-super-res", "--upscale-height", "1080", "--frame-generation", "2x"]
    )

    (encoder,) = captured["encoders"]
    assert encoder["frame_generation"] == 2
    assert not isinstance(encoder["metadata"], tuple)  # the source metadata, not an upscaled one
    captured["build_upscaler"].assert_not_called()
    captured["build_generator"].assert_called_once()


def test_swiftvr_run_uses_one_shared_worker_and_the_clip_writer(tmp_path):
    repo = tmp_path / "SwiftVR"
    created = []

    class Clip:
        def upscale(self, frames):
            yield from (frame + 1 for frame in frames)

        def close(self):
            pass

    class Session:
        def __init__(self, **kwargs):
            created.append(self)
            self.kwargs = kwargs
            self.requests = []
            self.closed = 0

        def upscaler(self, device, source_size, output_size, strength):
            self.requests.append((source_size, output_size, strength))
            return Clip()

        def close(self):
            self.closed += 1

    with (
        patch("jasna.restorer.swiftvr_common.resolve_swiftvr_paths", return_value=(repo, repo / "py", repo / "ckpt")),
        patch("jasna.postprocess_cli._SwiftvrSession", Session),
    ):
        captured = _run_main(
            tmp_path,
            ["--upscale", "swiftvr", "--swiftvr-repo", str(repo), "--upscale-strength", "0.5",
             "--upscale-swiftvr-tile", "1.64", "--no-swiftvr-accel"],
            frame=torch.full((3, 2, 2), 7, dtype=torch.uint8),
        )

    (session,) = created
    assert session.kwargs["repo"] == repo and session.kwargs["sv_python"] == repo / "py"
    assert session.kwargs["accel"] is False and session.kwargs["tile_megapixels"] == 1.64
    assert session.requests == [((1920, 1080), (3840, 2160), 0.5)]
    assert session.closed == 1
    ((encoded, pts),) = captured["encoded"]
    assert pts == 0 and int(encoded.max()) == 8
    captured["ensure_engines"].assert_not_called()
    captured["build_upscaler"].assert_not_called()

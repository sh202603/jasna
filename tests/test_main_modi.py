"""modi-only CLI tests (FlashVSR, SwiftVR, SwiftVR distill, frame generation, SeedVR2).

modi 独自。上流の tests/test_main.py が使う factories.py の ``cli_args`` / ``cli_environment``
と conftest の ``nvidia_vendor`` fixture に依存する。上流のファイルには追記しない。
"""
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from factories import cli_args, cli_environment
from jasna.main import build_parser

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


@pytest.fixture(autouse=True)
def _mock_device_context():
    """CLI orchestration tests use mocked models and must not initialize CUDA."""
    with patch("jasna.accelerator.device_context", side_effect=lambda _device: nullcontext()):
        yield


def _make_model_files(tmp_path):
    """The input, output and model paths ``cli_args`` creates in ``tmp_path``."""
    args = cli_args(tmp_path)
    paths = dict(zip(args[::2], args[1::2]))
    return tuple(
        Path(paths[flag])
        for flag in ("--input", "--output", "--restoration-model-path", "--detection-model-path")
    )


def _base_argv(input_path, output_path, restoration_path, detection_path, extra=None):
    return [
        "jasna",
        "--input", str(input_path),
        "--output", str(output_path),
        "--restoration-model-path", str(restoration_path),
        "--detection-model-path", str(detection_path),
        *(extra or []),
    ]


_main_patches = cli_environment


def _run_main(argv):
    with cli_environment() as pipeline_cls, patch.object(sys, "argv", argv):
        from jasna.main import main

        main()
    return pipeline_cls


class TestBuildParserModi:
    def test_frame_generation_defaults_and_rtx_mode(self):
        args = build_parser().parse_args(["--input", "a.mp4", "--output", "b.mp4"])
        assert args.frame_generation == "none"
        assert args.frame_generation_backend == "rtx"
        assert args.frame_generation_rtx_mode == "medium"
        assert args.frame_generation_model_path == ""
        args = build_parser().parse_args([
            "--input", "a.mp4", "--output", "b.mp4",
            "--frame-generation", "4x", "--frame-generation-backend", "rife", "--frame-generation-rtx-mode", "high",
            "--frame-generation-model-path", "/w/rife.pth", "--rtx-strength", "0.5", "--rtx-highbitrate",
        ])
        assert args.frame_generation == "4x"
        assert args.frame_generation_backend == "rife"
        assert args.frame_generation_rtx_mode == "high"
        assert args.frame_generation_model_path == "/w/rife.pth"
        assert args.rtx_strength == 0.5
        assert args.rtx_highbitrate is True

    def test_flashvsr_choice_and_defaults(self):
        args = build_parser().parse_args([
            "--input", "a.mp4", "--output", "b.mp4",
            "--secondary-restoration", "flashvsr",
            "--flashvsr-repo", "/opt/FlashVSR_plus",
        ])
        assert args.secondary_restoration == "flashvsr"
        assert args.flashvsr_repo == "/opt/FlashVSR_plus"
        assert args.flashvsr_version == "11"
        assert args.flashvsr_dtype == "bf16"
        assert args.flashvsr_max_clip_frames == 90
        assert args.flashvsr_unload_dit is True
        assert args.flashvsr_tiled_vae is True
        assert args.flashvsr_keep_bundle is False
        assert args.flashvsr_scale == 4

    def test_flashvsr_scale_choices(self):
        args = build_parser().parse_args([
            "--input", "a.mp4", "--output", "b.mp4",
            "--secondary-restoration", "flashvsr-inline",
            "--flashvsr-repo", "/opt/FlashVSR_plus",
            "--flashvsr-scale", "2",
        ])
        assert args.flashvsr_scale == 2
        with pytest.raises(SystemExit):
            build_parser().parse_args([
                "--input", "a.mp4", "--output", "b.mp4",
                "--flashvsr-scale", "3",
            ])

    def test_swiftvr_choice_and_defaults(self):
        args = build_parser().parse_args([
            "--input", "a.mp4", "--output", "b.mp4",
            "--secondary-restoration", "swiftvr-inline",
            "--swiftvr-repo", "/opt/SwiftVR",
        ])
        assert args.secondary_restoration == "swiftvr-inline"
        assert args.swiftvr_repo == "/opt/SwiftVR"
        assert args.swiftvr_python == ""
        assert args.swiftvr_model_dir == ""
        assert args.swiftvr_scale == 4
        assert args.swiftvr_accel is True  # acceleration is the default
        assert args.swiftvr_view_window == 15  # the crop view is smoothed by default
        assert args.swiftvr_bundle_dir == ""
        assert args.swiftvr_keep_bundle is False

    def test_swiftvr_offline_choice_parses(self):
        args = build_parser().parse_args([
            "--input", "a.mp4", "--output", "b.mp4",
            "--secondary-restoration", "swiftvr",
            "--swiftvr-repo", "/opt/SwiftVR",
            "--swiftvr-bundle-dir", "/data/bundle", "--swiftvr-keep-bundle",
        ])
        assert args.secondary_restoration == "swiftvr"
        assert args.swiftvr_bundle_dir == "/data/bundle"
        assert args.swiftvr_keep_bundle is True

    def test_swiftvr_scale_and_accel_flags(self):
        args = build_parser().parse_args([
            "--input", "a.mp4", "--output", "b.mp4",
            "--secondary-restoration", "swiftvr-inline",
            "--swiftvr-repo", "/opt/SwiftVR",
            "--swiftvr-scale", "2", "--no-swiftvr-accel",
        ])
        assert args.swiftvr_scale == 2
        assert args.swiftvr_accel is False
        with pytest.raises(SystemExit):
            build_parser().parse_args([
                "--input", "a.mp4", "--output", "b.mp4",
                "--swiftvr-scale", "3",
            ])


class TestSecondaryRestorersModi:
    def test_flashvsr_secondary_dispatches_offline(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        with patch("jasna.restorer.flashvsr_offline.run_flashvsr_offline") as mock_offline:
            pipeline_cls = _run_main(_base_argv(inp, out, rest, det, [
                "--secondary-restoration", "flashvsr",
                "--flashvsr-repo", str(tmp_path / "repo"),
            ]))
        # The offline orchestrator runs instead of the normal pipeline.
        mock_offline.assert_called_once()
        assert mock_offline.call_args.kwargs["engine"] == "flashvsr"
        pipeline_cls.assert_not_called()

    def test_swiftvr_secondary_dispatches_offline(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        with patch("jasna.restorer.flashvsr_offline.run_flashvsr_offline") as mock_offline:
            pipeline_cls = _run_main(_base_argv(inp, out, rest, det, [
                "--secondary-restoration", "swiftvr",
                "--swiftvr-repo", str(tmp_path / "repo"),
            ]))
        # Same orchestrator as flashvsr, with the SwiftVR engine; no pipeline here.
        mock_offline.assert_called_once()
        assert mock_offline.call_args.kwargs["engine"] == "swiftvr"
        pipeline_cls.assert_not_called()

    def test_swiftvr_distill_secondary(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        model = tmp_path / "distill.pt"
        model.touch()
        mock_distill = MagicMock()
        with patch(
            "jasna.restorer.swiftvr_distill_secondary_restorer.SwiftvrDistillSecondaryRestorer", mock_distill
        ):
            _run_main(_base_argv(inp, out, rest, det, [
                "--secondary-restoration", "swiftvr-distill",
                "--swiftvr-distill-model", str(model),
                "--swiftvr-distill-view-window", "15",
                "--swiftvr-distill-strength", "0.5",
                "--swiftvr-distill-stabilize", "2",
            ]))
        mock_distill.assert_called_once()
        kw = mock_distill.call_args.kwargs
        assert kw["model_path"] == model
        assert kw["view_window"] == 15
        assert kw["strength"] == 0.5
        assert kw["stabilize_radius"] == 2

    def test_swiftvr_distill_defaults_to_model_weights(self, tmp_path, monkeypatch):
        inp, out, rest, det = _make_model_files(tmp_path)
        (tmp_path / "model_weights").mkdir(exist_ok=True)
        model = tmp_path / "model_weights" / "swiftvr-distill.pt"
        model.touch()
        monkeypatch.chdir(tmp_path)
        mock_distill = MagicMock()
        with patch(
            "jasna.restorer.swiftvr_distill_secondary_restorer.SwiftvrDistillSecondaryRestorer", mock_distill
        ):
            _run_main(_base_argv(inp, out, rest, det, ["--secondary-restoration", "swiftvr-distill"]))
        assert mock_distill.call_args.kwargs["model_path"].resolve() == model.resolve()

    def test_swiftvr_distill_needs_an_existing_model(self, tmp_path, monkeypatch):
        inp, out, rest, det = _make_model_files(tmp_path)
        (tmp_path / "model_weights").mkdir(exist_ok=True)  # no swiftvr-distill.pt: the default file is missing
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="SwiftVR distill checkpoint not found") as e:
            _run_main(_base_argv(inp, out, rest, det, ["--secondary-restoration", "swiftvr-distill"]))
        assert "huggingface.co/okatti/swiftvr-distill" in str(e.value)
        with pytest.raises(FileNotFoundError, match="--swiftvr-distill-model not found"):
            _run_main(_base_argv(inp, out, rest, det, [
                "--secondary-restoration", "swiftvr-distill",
                "--swiftvr-distill-model", str(tmp_path / "missing.pt"),
            ]))

    def test_swiftvr_distill_rejects_a_negative_view_window(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        model = tmp_path / "distill.pt"
        model.touch()
        with pytest.raises(ValueError, match="--swiftvr-distill-view-window must be >= 0"):
            _run_main(_base_argv(inp, out, rest, det, [
                "--secondary-restoration", "swiftvr-distill",
                "--swiftvr-distill-model", str(model),
                "--swiftvr-distill-view-window", "-1",
            ]))

    @pytest.mark.parametrize("value", ["-0.1", "2.5"])
    def test_swiftvr_distill_rejects_an_out_of_range_strength(self, tmp_path, value):
        inp, out, rest, det = _make_model_files(tmp_path)
        model = tmp_path / "distill.pt"
        model.touch()
        with pytest.raises(ValueError, match="--swiftvr-distill-strength must be in"):
            _run_main(_base_argv(inp, out, rest, det, [
                "--secondary-restoration", "swiftvr-distill",
                "--swiftvr-distill-model", str(model),
                "--swiftvr-distill-strength", value,
            ]))

    @pytest.mark.parametrize("value", ["-1", "9"])
    def test_swiftvr_distill_rejects_an_out_of_range_stabilize(self, tmp_path, value):
        inp, out, rest, det = _make_model_files(tmp_path)
        model = tmp_path / "distill.pt"
        model.touch()
        with pytest.raises(ValueError, match="--swiftvr-distill-stabilize must be in"):
            _run_main(_base_argv(inp, out, rest, det, [
                "--secondary-restoration", "swiftvr-distill",
                "--swiftvr-distill-model", str(model),
                "--swiftvr-distill-stabilize", value,
            ]))

    def test_flashvsr_inline_choice_parses(self):
        args = build_parser().parse_args([
            "--input", "a.mp4", "--output", "b.mp4",
            "--secondary-restoration", "flashvsr-inline",
            "--flashvsr-repo", "/opt/FlashVSR_plus",
        ])
        assert args.secondary_restoration == "flashvsr-inline"

    def test_flashvsr_inline_dispatches_normal_pipeline_and_autoenables_fp8(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        repo = tmp_path / "repo"
        # default_flashvsr_python resolves per-platform; create both layouts.
        (repo / ".venv" / "bin").mkdir(parents=True)
        (repo / ".venv" / "bin" / "python").touch()
        (repo / ".venv" / "Scripts").mkdir(parents=True)
        (repo / ".venv" / "Scripts" / "python.exe").touch()
        (repo / "models" / "FlashVSR-v1.1").mkdir(parents=True)
        argv = _base_argv(inp, out, rest, det, [
            "--secondary-restoration", "flashvsr-inline",
            "--flashvsr-repo", str(repo),
            "--max-clip-size", "180",
        ])
        with _main_patches() as pipeline_cls, \
                patch("jasna.restorer.flashvsr_inline_secondary_restorer.FlashvsrInlineSecondaryRestorer") as mock_r, \
                patch("jasna.restorer.flashvsr_offline.run_flashvsr_offline") as mock_offline, \
                patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JASNA_FP8_RECON", None)
            with patch.object(sys, "argv", argv):
                from jasna.main import main
                main()
            # capture inside the patch.dict block (it restores os.environ on exit)
            fp8_env = os.environ.get("JASNA_FP8_RECON")
        # Inline goes through the normal streaming pipeline, NOT the offline early-return.
        pipeline_cls.assert_called_once()
        mock_offline.assert_not_called()
        mock_r.assert_called_once()
        assert mock_r.call_args.kwargs["scale"] == 4  # default reaches the restorer
        # fp8-recon is auto-enabled for inline (co-residence headroom).
        assert fp8_env == "1"
        # No clip-length cap for inline (tiny-long is flat in the clip length):
        # the user's --max-clip-size reaches the pipeline unchanged.
        assert pipeline_cls.call_args.kwargs["config"].max_clip_size == 180

    def test_flashvsr_inline_requires_repo(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        with pytest.raises(ValueError, match="flashvsr-repo is required"):
            _run_main(_base_argv(inp, out, rest, det, [
                "--secondary-restoration", "flashvsr-inline",
            ]))


def _make_seedvr2_env(tmp_path):
    """Fake seedvr2 checkout (marker + venv pythons) and a LoRA file."""
    repo = tmp_path / "seedvr2_repo"
    (repo / "src" / "core").mkdir(parents=True)
    (repo / "src" / "core" / "generation_utils.py").touch()
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").touch()
    (repo / ".venv" / "Scripts").mkdir(parents=True)
    (repo / ".venv" / "Scripts" / "python.exe").touch()
    (repo / "models" / "SEEDVR2").mkdir(parents=True)
    lora = tmp_path / "lora.pt"
    lora.touch()
    return repo, lora


def _seedvr2_argv(inp, out, rest, det, repo, lora, extra=None):
    argv = _base_argv(inp, out, rest, det, [
        "--restoration-model-name", "seedvr2",
        "--seedvr2-repo", str(repo),
        "--seedvr2-lora", str(lora),
        "--vr-mode", "off",
    ])
    if extra:
        argv.extend(extra)
    return argv


class TestSeedvr2Main:
    def test_flag_defaults(self):
        args = build_parser().parse_args(["--input", "a.mp4", "--output", "b.mp4"])
        assert args.restoration_model_name == "basicvsrpp"
        assert args.seedvr2_repo == ""
        assert args.seedvr2_lora == ""
        assert args.seedvr2_lora_rank == 16
        assert args.seedvr2_window == 33
        assert args.seedvr2_overlap == 9
        assert args.seedvr2_color_fix == "lab"
        assert args.seedvr2_empty_cache == "auto"
        assert args.seedvr2_accel is False
        assert args.seedvr2_dit == "seedvr2_ema_3b_fp16.safetensors"

    def test_accel_flag_parses_both_ways(self):
        base = ["--input", "a.mp4", "--output", "b.mp4"]
        assert build_parser().parse_args(base + ["--seedvr2-accel"]).seedvr2_accel is True
        assert build_parser().parse_args(base + ["--no-seedvr2-accel"]).seedvr2_accel is False

    def _run_expect_error(self, argv, match):
        with _main_patches():
            with patch.object(sys, "argv", argv):
                from jasna.main import main
                with pytest.raises(ValueError, match=match):
                    main()

    def test_requires_repo(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        _, lora = _make_seedvr2_env(tmp_path)
        argv = _base_argv(inp, out, rest, det, [
            "--restoration-model-name", "seedvr2",
            "--seedvr2-lora", str(lora),
            "--vr-mode", "off",
        ])
        self._run_expect_error(argv, "requires --seedvr2-repo")

    def test_missing_lora_names_download(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        repo, _ = _make_seedvr2_env(tmp_path)
        argv = _base_argv(inp, out, rest, det, [
            "--restoration-model-name", "seedvr2",
            "--seedvr2-repo", str(repo),
            "--seedvr2-lora", str(tmp_path / "missing.pt"),
            "--vr-mode", "off",
        ])
        with _main_patches():
            with patch.object(sys, "argv", argv):
                from jasna.main import main
                with pytest.raises(FileNotFoundError, match="huggingface"):
                    main()

    def test_window_must_be_4n1(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        repo, lora = _make_seedvr2_env(tmp_path)
        argv = _seedvr2_argv(inp, out, rest, det, repo, lora, ["--seedvr2-window", "32"])
        self._run_expect_error(argv, "4n\\+1")

    def test_overlap_must_be_below_window(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        repo, lora = _make_seedvr2_env(tmp_path)
        argv = _seedvr2_argv(inp, out, rest, det, repo, lora, ["--seedvr2-overlap", "33"])
        self._run_expect_error(argv, "seedvr2-overlap")

    def test_rejects_frame_generation(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        repo, lora = _make_seedvr2_env(tmp_path)
        argv = _seedvr2_argv(inp, out, rest, det, repo, lora, ["--frame-generation", "2x"])
        self._run_expect_error(argv, "frame-gen")

    def test_rejects_flashvsr_inline_combo(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        repo, lora = _make_seedvr2_env(tmp_path)
        argv = _seedvr2_argv(inp, out, rest, det, repo, lora, [
            "--secondary-restoration", "flashvsr-inline",
            "--flashvsr-repo", str(tmp_path / "fv"),
        ])
        self._run_expect_error(argv, "flashvsr-inline")

    def test_rejects_vr_sbs(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        repo, lora = _make_seedvr2_env(tmp_path)
        argv = _base_argv(inp, out, rest, det, [
            "--restoration-model-name", "seedvr2",
            "--seedvr2-repo", str(repo),
            "--seedvr2-lora", str(lora),
            "--vr-mode", "sbs",
        ])
        self._run_expect_error(argv, "VR processing")

    def test_rejects_vr_auto_when_detected(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        repo, lora = _make_seedvr2_env(tmp_path)
        argv = _base_argv(inp, out, rest, det, [
            "--restoration-model-name", "seedvr2",
            "--seedvr2-repo", str(repo),
            "--seedvr2-lora", str(lora),
            "--vr-mode", "auto",
        ])
        with _main_patches(), \
                patch("jasna.media.probe.get_video_meta_data", return_value=MagicMock()), \
                patch("jasna.vr180.resolve_vr_mode",
                      return_value=MagicMock(resolved="sbs-fisheye", reason="8k aspect")):
            with patch.object(sys, "argv", argv):
                from jasna.main import main
                with pytest.raises(ValueError, match="detected VR content"):
                    main()

    def test_dispatch_builds_seedvr2_and_skips_basicvsrpp_engines(self, tmp_path):
        inp, out, rest, det = _make_model_files(tmp_path)
        repo, lora = _make_seedvr2_env(tmp_path)
        argv = _seedvr2_argv(inp, out, rest, det, repo, lora)
        mock_basicvsrpp = MagicMock()
        with (
            patch("jasna.main.check_ascii_install_path", return_value=(True, "C:\\fake")),
            patch("jasna.main.check_supported_gpu", return_value=(True, "Fake GPU")),
            patch("jasna.main.check_gpu_driver_version", return_value=(True, "610.18")),
            patch("jasna.main.check_required_executables"),
            patch("jasna.main.check_windows_nvidia_sysmem_fallback_policy", return_value=(True, "OK")),
            patch("jasna.engine_compiler.ensure_engines_compiled",
                  return_value=MagicMock(use_basicvsrpp_tensorrt=False)) as mock_compile,
            patch("jasna.pipeline.Pipeline", return_value=MagicMock()) as pipeline_cls,
            patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer", mock_basicvsrpp),
            patch("jasna.restorer.seedvr2_lora_restorer.Seedvr2LoraRestorer") as mock_seedvr2,
        ):
            with patch.object(sys, "argv", argv):
                from jasna.main import main
                main()

        mock_seedvr2.assert_called_once()
        kw = mock_seedvr2.call_args.kwargs
        assert kw["lora_path"] == str(lora)
        assert kw["window"] == 33
        assert kw["overlap"] == 9
        assert kw["color_fix"] == "lab"
        assert kw["empty_cache"] == "auto"
        assert kw["accel"] is False
        mock_basicvsrpp.assert_not_called()
        req = mock_compile.call_args[0][0]
        assert req.basicvsrpp is False
        pipeline_cls.assert_called_once()

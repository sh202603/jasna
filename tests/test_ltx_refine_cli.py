"""``--ltx-crop-vsr`` at the CLI and in the session factory: what it refuses, what it
reaches the pipeline with, and that no primary restorer loads or compiles."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import session_config


def _run_main(tmp_path, extra_args, *, create_restoration=False):
    """Like test_main_validation's helper, with the LTX output as --input and a source video."""
    ltx_output = tmp_path / "restored-ltx.mp4"
    ltx_output.touch()
    source = tmp_path / "source.mp4"
    source.touch()
    output_path = tmp_path / "out.mkv"
    model_weights = tmp_path / "model_weights"
    model_weights.mkdir(exist_ok=True)
    restoration_path = model_weights / "restore.pth"
    if create_restoration:
        restoration_path.touch()
    detection_path = model_weights / "det.onnx"
    detection_path.touch()
    distill = model_weights / "swiftvr-distill.pt"
    distill.touch()

    base_args = [
        "jasna",
        "--input", str(ltx_output),
        "--output", str(output_path),
        "--restoration-model-path", str(restoration_path),
        "--detection-model-path", str(detection_path),
        "--ltx-crop-vsr-source", str(source),
        "--swiftvr-distill-model", str(distill),
    ]
    with (
        patch("jasna.main.check_ascii_install_path", return_value=(True, "C:\\fake")),
        patch("jasna.main.check_supported_gpu", return_value=(True, "Fake GPU")),
        patch("jasna.main.check_gpu_driver_version", return_value=(True, "610.18")),
        patch("jasna.main.check_required_executables"),
        patch("jasna.main.check_windows_nvidia_sysmem_fallback_policy", return_value=(True, "OK")),
        patch("jasna.engine_compiler.ensure_engines_compiled", return_value=MagicMock(use_basicvsrpp_tensorrt=False)) as compiled,
        patch("jasna.session_factory._build_basicvsrpp_pipeline", side_effect=AssertionError("BasicVSR++ must not load")),
        patch("jasna.pipeline.Pipeline", return_value=MagicMock()) as pipeline_cls,
    ):
        with patch.object(sys, "argv", base_args + extra_args):
            from jasna.main import main
            main()
    return pipeline_cls, compiled


class TestCli:
    def test_reaches_the_pipeline_without_the_primary_weights(self, tmp_path):
        pipeline_cls, compiled = _run_main(tmp_path, ["--ltx-crop-vsr", "swiftvr-distill", "--ltx-large-canvas"])
        config = pipeline_cls.call_args.kwargs["config"]
        assert config.ltx_crop_vsr == "swiftvr-distill"
        assert config.ltx_crop_vsr_source == tmp_path / "source.mp4"
        assert config.ltx_large_canvas is True
        assert config.restoration_model_name == "basicvsrpp"
        request = compiled.call_args.args[0]
        assert request.detection is True and request.basicvsrpp is False

    def test_needs_the_source(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main(tmp_path, ["--ltx-crop-vsr", "swiftvr-distill", "--ltx-crop-vsr-source", ""])
        assert "needs --ltx-crop-vsr-source" in capsys.readouterr().err

    def test_rejects_a_missing_source(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main(tmp_path, ["--ltx-crop-vsr", "swiftvr-distill", "--ltx-crop-vsr-source", str(tmp_path / "no.mp4")])
        assert "--ltx-crop-vsr-source not found" in capsys.readouterr().err

    def test_rejects_the_input_as_its_own_source(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main(
                tmp_path,
                ["--ltx-crop-vsr", "swiftvr-distill", "--ltx-crop-vsr-source", str(tmp_path / "restored-ltx.mp4")],
            )
        assert "not the LTX output" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "args, message",
        [
            (["--restoration-model-name", "ltx"], "needs --restoration-model-name basicvsrpp"),
            (["--restoration-model-name", "seedvr2"], "needs --restoration-model-name basicvsrpp"),
            (["--secondary-restoration", "unet-4x"], "--secondary-restoration"),
            (["--denoise", "low"], "--denoise"),
            (["--segments", "1-2"], "--segments"),
            (["--ltx-auto-split", "12"], "--ltx-auto-split"),
            (["--retarget-high-fps"], "--retarget-high-fps"),
            (["--fmp4"], "--fmp4"),
            (["--frame-gen", "2x"], "--frame-gen"),
            (["--vr-mode", "sbs"], "--vr-mode sbs"),
            (["--lut", __file__], "--lut"),
            (["--sharpen", "0.5"], "--sharpen"),
            (["--swiftvr-view-window", "3"], "--swiftvr-view-window"),
            (["--swiftvr-distill-view-window", "3"], "--swiftvr-distill-view-window"),
            (["--swiftvr-scale", "4"], "--swiftvr-scale 4"),
            (["--ltx-large-canvas", "--swiftvr-distill-stabilize", "2"], "--swiftvr-distill-stabilize"),
            (["--ltx-seed", "7"], "needs --restoration-model-name ltx"),
            (["--ltx-trial"], "needs --restoration-model-name ltx"),
        ],
    )
    def test_rejected_combinations(self, tmp_path, capsys, args, message):
        with pytest.raises(SystemExit):
            _run_main(tmp_path, ["--ltx-crop-vsr", "swiftvr-distill", *args])
        assert message in capsys.readouterr().err

    def test_scale_2_is_accepted(self, tmp_path):
        pipeline_cls, _ = _run_main(tmp_path, ["--ltx-crop-vsr", "swiftvr-distill", "--swiftvr-scale", "2"])
        assert pipeline_cls.call_args.kwargs["config"].ltx_crop_vsr == "swiftvr-distill"


class TestSession:
    def test_the_refine_session_loads_no_primary(self, tmp_path):
        from jasna import session_factory

        distill = tmp_path / "swiftvr-distill.pt"
        distill.touch()
        config = session_config(
            ltx_crop_vsr="swiftvr-distill",
            ltx_crop_vsr_source=tmp_path / "source.mp4",
            swiftvr_distill_model=str(distill),
            device="cpu",
        )
        with (
            patch("jasna.engine_compiler.ensure_engines_compiled") as compiled,
            patch.object(session_factory, "_build_basicvsrpp_pipeline", side_effect=AssertionError("must not load")),
        ):
            session = session_factory.build_restoration_session(config, log_callback=None)
        assert session.restoration_pipeline is None and session.ltx_files is None
        request = compiled.call_args.args[0]
        assert request.detection is True and request.basicvsrpp is False and request.detection_batch_size == 4

    def test_the_refine_restorer_runs_swiftvr_at_scale_2_without_view_smoothing(self, tmp_path):
        from jasna import session_factory

        config = session_config(ltx_crop_vsr="swiftvr", ltx_crop_vsr_source=tmp_path / "s.mp4", swiftvr_view_window=15)
        refined = session_factory.ltx_crop_vsr_config(config)
        assert refined.secondary_restoration == "swiftvr-inline"
        assert refined.swiftvr_scale == 2 and refined.swiftvr_view_window == 0
        distill = session_factory.ltx_crop_vsr_config(config.__class__(**{**config.__dict__, "ltx_crop_vsr": "swiftvr-distill"}))
        assert distill.secondary_restoration == "swiftvr-distill" and distill.swiftvr_distill_view_window == 0

    def test_build_pipeline_skips_the_primary(self, tmp_path):
        from jasna import session_factory

        config = session_config(ltx_crop_vsr="swiftvr-distill", ltx_crop_vsr_source=tmp_path / "s.mp4", device="cpu")
        session = MagicMock()
        with (
            patch.object(session_factory, "provide_restoration_models", side_effect=AssertionError("must not load")),
            patch("jasna.pipeline.Pipeline", return_value=MagicMock()) as pipeline_cls,
        ):
            session_factory.build_pipeline(config, session, tmp_path / "ltx.mp4", tmp_path / "out.mkv")
        assert pipeline_cls.call_args.kwargs["config"] is config

    def test_config_needs_a_source(self, tmp_path):
        with pytest.raises(ValueError, match="needs --ltx-crop-vsr-source"):
            session_config(ltx_crop_vsr="swiftvr")
        with pytest.raises(ValueError, match="must be one of"):
            session_config(ltx_crop_vsr="flashvsr", ltx_crop_vsr_source=tmp_path / "s.mp4")

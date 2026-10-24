"""``--secondary-restoration rtx-super-res`` with an LTX model (modi): the CLI accepts it
and only it, the session builds the compose pass's restorer at scale 2 sized from the
canvases, and the BasicVSR++ segments of an auto split get no secondary restoration."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.factories import make_pipeline, session_config


def _run_main(tmp_path, extra_args, *, nvvfx=True):
    """An LTX trial job (no model files), with the system checks and the heavy models patched."""
    input_path = tmp_path / "in.mp4"
    input_path.touch()
    model_weights = tmp_path / "model_weights"
    model_weights.mkdir(exist_ok=True)
    detection_path = model_weights / "det.onnx"
    detection_path.touch()
    base_args = [
        "jasna",
        "--input", str(input_path),
        "--output", str(tmp_path / "out.mkv"),
        "--restoration-model-path", str(model_weights / "restore.pth"),
        "--detection-model-path", str(detection_path),
        "--restoration-model-name", "ltx-8step", "--ltx-trial",
    ]
    with (
        patch("jasna.main.check_ascii_install_path", return_value=(True, "C:\\fake")),
        patch("jasna.main.check_supported_gpu", return_value=(True, "Fake GPU")),
        patch("jasna.main.check_gpu_driver_version", return_value=(True, "610.18")),
        patch("jasna.main.check_required_executables"),
        patch("jasna.main.check_windows_nvidia_sysmem_fallback_policy", return_value=(True, "OK")),
        patch("jasna.accelerator.is_nvidia_device", return_value=True),
        patch("jasna.ltx.transformer.require_kernels"),
        patch("jasna.engine_compiler.ensure_engines_compiled", return_value=MagicMock(use_basicvsrpp_tensorrt=False)),
        patch("jasna.session_factory.nvvfx_installed", return_value=nvvfx),
        patch("jasna.pipeline.Pipeline", return_value=MagicMock()) as pipeline_cls,
    ):
        with patch.object(sys, "argv", base_args + extra_args):
            from jasna.main import main
            main()
    return pipeline_cls


class TestCli:
    def test_rtx_super_res_reaches_the_pipeline_with_the_deblur_default(self, tmp_path):
        pipeline_cls = _run_main(tmp_path, ["--secondary-restoration", "rtx-super-res", "--rtx-strength", "0.75"])
        config = pipeline_cls.call_args.kwargs["config"]
        assert config.restoration_model_name == "ltx" and config.secondary_restoration == "rtx-super-res"
        assert config.rtx_deblur == "medium" and config.rtx_strength == 0.75
        assert config.rtx_quality == "high" and config.rtx_denoise == "medium"

    def test_deblur_can_be_turned_off(self, tmp_path):
        pipeline_cls = _run_main(tmp_path, ["--secondary-restoration", "rtx-super-res", "--rtx-deblur", "none"])
        assert pipeline_cls.call_args.kwargs["config"].rtx_deblur == "none"

    def test_without_the_secondary_the_deblur_default_stays_off(self, tmp_path):
        pipeline_cls = _run_main(tmp_path, [])
        config = pipeline_cls.call_args.kwargs["config"]
        assert config.secondary_restoration == "none" and config.rtx_deblur == "none"

    def test_scale_2_is_accepted_and_4_refused(self, tmp_path, capsys):
        pipeline_cls = _run_main(tmp_path, ["--secondary-restoration", "rtx-super-res", "--rtx-scale", "2"])
        assert pipeline_cls.call_args.kwargs["config"].secondary_restoration == "rtx-super-res"
        with pytest.raises(SystemExit):
            _run_main(tmp_path, ["--secondary-restoration", "rtx-super-res", "--rtx-scale", "4"])
        assert "runs RTX Super Res at scale 2; --rtx-scale 4 is not supported" in capsys.readouterr().err

    def test_the_default_scale_4_is_fine_when_not_spelled_out(self, tmp_path):
        pipeline_cls = _run_main(tmp_path, ["--secondary-restoration", "rtx-super-res"])
        assert pipeline_cls.call_args.kwargs["config"].rtx_scale == 4  # resolved to 2 by the session

    @pytest.mark.parametrize("secondary", ["unet-4x", "tvai", "swiftvr-distill", "flashvsr-inline"])
    def test_other_secondaries_stay_refused(self, tmp_path, capsys, secondary):
        with pytest.raises(SystemExit):
            _run_main(tmp_path, ["--secondary-restoration", secondary])
        assert f"does not support --secondary-restoration {secondary}" in capsys.readouterr().err

    def test_denoise_stays_refused(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main(tmp_path, ["--secondary-restoration", "rtx-super-res", "--denoise", "low"])
        assert "does not support --denoise" in capsys.readouterr().err

    def test_needs_nvvfx_before_anything_loads(self, tmp_path):
        with (
            patch("jasna.session_factory.provide_restoration_models", side_effect=AssertionError("loaded a model")),
            pytest.raises(ValueError, match="nvidia-vfx"),
        ):
            _run_main(tmp_path, ["--secondary-restoration", "rtx-super-res"], nvvfx=False)


class TestSession:
    def test_the_sharpen_restorer_runs_at_scale_2_sized_from_the_canvases(self):
        from jasna import session_factory

        config = session_config(
            restoration_model_name="ltx", secondary_restoration="rtx-super-res", rtx_scale=4, rtx_deblur="medium",
            rtx_strength=0.75,
        )
        assert session_factory.uses_ltx_sharpen(config)
        assert not session_factory.uses_ltx_sharpen(session_config(secondary_restoration="rtx-super-res"))
        assert not session_factory.uses_ltx_sharpen(session_config(restoration_model_name="ltx"))
        with patch("jasna.restorer.rtx_superres_secondary_restorer.RtxSuperresSecondaryRestorer") as rtx_cls:
            restorer = session_factory.build_ltx_sharpen_restorer(config, "cpu")
        assert restorer is rtx_cls.return_value
        kwargs = rtx_cls.call_args.kwargs
        assert kwargs["scale"] == 2 and kwargs["deblur"] == "medium" and kwargs["strength"] == 0.75
        assert kwargs["input_size"] is None

    def test_the_session_refuses_other_secondaries_and_a_missing_nvvfx(self):
        from jasna import session_factory

        with pytest.raises(ValueError, match="does not support secondary restoration 'unet-4x'"):
            session_factory.build_restoration_session(
                session_config(restoration_model_name="ltx", secondary_restoration="unet-4x"), log_callback=None
            )
        with pytest.raises(ValueError, match="does not support denoise"):
            session_factory.build_restoration_session(
                session_config(restoration_model_name="ltx", denoise_strength="low"), log_callback=None
            )
        with (
            patch.object(session_factory, "nvvfx_installed", return_value=False),
            pytest.raises(ValueError, match="nvidia-vfx"),
        ):
            session_factory.build_restoration_session(
                session_config(restoration_model_name="ltx", secondary_restoration="rtx-super-res"), log_callback=None
            )

    def test_the_basicvsrpp_segments_of_an_ltx_job_get_no_secondary(self):
        from jasna import session_factory

        config = session_config(restoration_model_name="ltx", secondary_restoration="rtx-super-res", device="cpu")
        with (
            patch("jasna.engine_compiler.ensure_engines_compiled", return_value=MagicMock(use_basicvsrpp_tensorrt=False)),
            patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer"),
            patch("jasna.restorer.restoration_pipeline.RestorationPipeline") as pipeline_cls,
            patch.object(session_factory, "_build_secondary_restorer", side_effect=AssertionError("built the rtx restorer")),
        ):
            session_factory._build_basicvsrpp_pipeline(config, torch.device("cpu"), log_callback=None)
        assert pipeline_cls.call_args.kwargs["secondary_restorer"] is None

    def test_a_basicvsrpp_job_still_gets_its_secondary(self):
        from jasna import session_factory

        config = session_config(secondary_restoration="rtx-super-res", device="cpu")
        with (
            patch("jasna.engine_compiler.ensure_engines_compiled", return_value=MagicMock(use_basicvsrpp_tensorrt=False)),
            patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer"),
            patch("jasna.restorer.restoration_pipeline.RestorationPipeline") as pipeline_cls,
            patch.object(session_factory, "_build_secondary_restorer", return_value="rtx") as built,
        ):
            session_factory._build_basicvsrpp_pipeline(config, torch.device("cpu"), log_callback=None)
        assert pipeline_cls.call_args.kwargs["secondary_restorer"] == "rtx"
        assert built.call_args.args[0] is config


class TestPipeline:
    def test_the_render_carries_the_sharpener_factory(self):
        pipeline = make_pipeline(restoration_model_name="ltx", secondary_restoration="rtx-super-res", rtx_deblur="low")
        assert pipeline.ltx_sharpen is not None
        with patch("jasna.restorer.rtx_superres_secondary_restorer.RtxSuperresSecondaryRestorer") as rtx_cls:
            assert pipeline.ltx_sharpen() is rtx_cls.return_value
        kwargs = rtx_cls.call_args.kwargs
        assert kwargs["device"] == torch.device("cpu") and kwargs["scale"] == 2 and kwargs["deblur"] == "low"
        assert kwargs["input_size"] is None
        pipeline.vr_resolution = SimpleNamespace(is_sbs=False)
        metadata = SimpleNamespace(video_height=480, video_width=720)
        with patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=16 << 30)):
            render = pipeline.ltx_segment_render(metadata)
        assert render.sharpen is pipeline.ltx_sharpen and render.budget_name == "segment"

    def test_without_the_secondary_the_render_has_no_sharpener(self):
        pipeline = make_pipeline(restoration_model_name="ltx")
        assert pipeline.ltx_sharpen is None
        pipeline.vr_resolution = SimpleNamespace(is_sbs=False)
        with patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=16 << 30)):
            assert pipeline.ltx_segment_render(SimpleNamespace(video_height=480, video_width=720)).sharpen is None

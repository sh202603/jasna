"""``--secondary-restoration rtx-super-res`` with an LTX model, modi's policy on upstream's
plumbing: the CLI defaults the deblur pass to low there and refuses an explicit
``--rtx-scale 4`` without auto split, the session refuses a missing nvvfx before anything
loads and hands the ``--rtx-*`` options (strength, high bitrate included) to ``LtxSuperRes``,
and the GUI maps a ``none`` deblur setting to the same low default.

modi 独自。factories.py の ``cli_args`` / ``cli_environment`` / ``session_config`` と conftest の
``nvidia_vendor`` fixture に依存する。上流のファイルには追記しない。
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from factories import cli_args, cli_environment, session_config

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


def _run_main(tmp_path, extra_args, *, nvvfx=True):
    """An LTX trial job (no model files), with the system checks and the heavy models patched."""
    argv = cli_args(tmp_path, "--restoration-model-name", "ltx-8step", "--ltx-trial", *extra_args)
    with (
        cli_environment() as pipeline_cls,
        patch("jasna.accelerator.is_nvidia_device", return_value=True),
        patch("jasna.ltx.transformer.require_kernels"),
        patch("jasna.session_factory.nvvfx_installed", return_value=nvvfx),
        patch("jasna.ltx.super_res.LtxSuperRes"),
        patch.object(sys, "argv", ["jasna", *argv]),
    ):
        from jasna.main import main

        main()
    return pipeline_cls


class TestCli:
    def test_rtx_super_res_reaches_the_pipeline_with_the_deblur_default(self, tmp_path):
        pipeline_cls = _run_main(tmp_path, ["--secondary-restoration", "rtx-super-res", "--rtx-strength", "0.75"])
        config = pipeline_cls.call_args.kwargs["config"]
        assert config.restoration_model_name == "ltx" and config.secondary_restoration == "rtx-super-res"
        assert config.rtx_deblur == "low" and config.rtx_strength == 0.75
        assert config.rtx_quality == "high" and config.rtx_denoise == "medium"

    def test_deblur_can_be_set_or_turned_off(self, tmp_path):
        pipeline_cls = _run_main(tmp_path, ["--secondary-restoration", "rtx-super-res", "--rtx-deblur", "medium"])
        assert pipeline_cls.call_args.kwargs["config"].rtx_deblur == "medium"
        pipeline_cls = _run_main(tmp_path, ["--secondary-restoration", "rtx-super-res", "--rtx-deblur", "none"])
        assert pipeline_cls.call_args.kwargs["config"].rtx_deblur == "none"

    def test_without_the_secondary_the_deblur_default_stays_off(self, tmp_path):
        config = _run_main(tmp_path, []).call_args.kwargs["config"]
        assert config.secondary_restoration == "none" and config.rtx_deblur == "none"

    def test_scale_2_is_accepted_and_4_refused(self, tmp_path, capsys):
        pipeline_cls = _run_main(tmp_path, ["--secondary-restoration", "rtx-super-res", "--rtx-scale", "2"])
        assert pipeline_cls.call_args.kwargs["config"].secondary_restoration == "rtx-super-res"
        with pytest.raises(SystemExit):
            _run_main(tmp_path, ["--secondary-restoration", "rtx-super-res", "--rtx-scale", "4"])
        assert "runs RTX Super Res at scale 2; --rtx-scale 4 is not supported" in capsys.readouterr().err

    def test_scale_4_is_fine_with_auto_split_where_the_basicvsrpp_parts_use_it(self, tmp_path):
        with patch("jasna.session_factory.ltx_auto_split", return_value=None):
            pipeline_cls = _run_main(
                tmp_path, ["--secondary-restoration", "rtx-super-res", "--rtx-scale", "4", "--ltx-auto-split", "12"]
            )
        assert pipeline_cls.call_args.kwargs["config"].rtx_scale == 4

    def test_the_default_scale_4_is_fine_when_not_spelled_out(self, tmp_path):
        pipeline_cls = _run_main(tmp_path, ["--secondary-restoration", "rtx-super-res"])
        assert pipeline_cls.call_args.kwargs["config"].rtx_scale == 4  # LTX windows always upscale 2x

    @pytest.mark.parametrize("secondary", ["unet-4x", "tvai", "swiftvr-distill", "flashvsr-inline"])
    def test_other_secondaries_stay_refused(self, tmp_path, capsys, secondary):
        with pytest.raises(SystemExit):
            _run_main(tmp_path, ["--secondary-restoration", secondary])
        assert f"supports --secondary-restoration {secondary} only with --ltx-auto-split" in capsys.readouterr().err

    def test_needs_nvvfx_before_anything_loads(self, tmp_path):
        with (
            patch("jasna.session_factory.provide_restoration_models", side_effect=AssertionError("loaded a model")),
            pytest.raises(ValueError, match="nvidia-vfx"),
        ):
            _run_main(tmp_path, ["--secondary-restoration", "rtx-super-res"], nvvfx=False)


class TestSession:
    def test_ltx_super_res_gets_every_rtx_option(self):
        from jasna import session_factory

        config = session_config(
            restoration_model_name="ltx", ltx_trial=True, secondary_restoration="rtx-super-res", rtx_scale=4,
            rtx_deblur="low", rtx_strength=0.75, rtx_highbitrate=True,
        )
        with (
            patch("jasna.ltx.super_res.LtxSuperRes") as super_res_cls,
            patch.object(session_factory, "nvvfx_installed", return_value=True),
            patch("jasna.accelerator.is_nvidia_device", return_value=True),
            patch("jasna.ltx.transformer.require_kernels"),
            patch("jasna.engine_compiler.ensure_engines_compiled"),
        ):
            session = session_factory.build_restoration_session(config, log_callback=None)
        assert session.ltx_super_res is super_res_cls.return_value
        kwargs = super_res_cls.call_args.kwargs
        assert kwargs["deblur"] == "low" and kwargs["strength"] == 0.75 and kwargs["highbitrate"] is True
        assert kwargs["quality"] == "high" and kwargs["denoise"] == "medium"

    def test_the_session_refuses_a_missing_nvvfx(self):
        from jasna import session_factory

        with (
            patch.object(session_factory, "nvvfx_installed", return_value=False),
            patch.object(session_factory, "provide_restoration_models", side_effect=AssertionError("loaded a model")),
            pytest.raises(ValueError, match="nvidia-vfx"),
        ):
            session_factory.build_restoration_session(
                session_config(restoration_model_name="ltx", ltx_trial=True, secondary_restoration="rtx-super-res"),
                log_callback=None,
            )


class TestGui:
    @pytest.mark.parametrize(
        ("model", "secondary", "setting", "expected"),
        [
            ("ltx", "rtx-super-res", "none", "low"),
            ("ltx", "rtx-super-res", "medium", "medium"),
            ("ltx", "none", "none", "none"),
            ("basicvsrpp", "rtx-super-res", "none", "none"),
        ],
    )
    def test_an_ltx_job_maps_a_none_deblur_setting_to_low(self, model, secondary, setting, expected):
        from jasna.gui.models import AppSettings
        from jasna.gui.video_session import _rtx_deblur

        assert _rtx_deblur(AppSettings(rtx_deblur=setting), model, secondary) == expected

"""+modi: the fork's command-line options reach ``SessionConfig`` (``jasna.cli_modi``)."""

from __future__ import annotations

import pytest

from factories import GUI_WEIGHTS_DIR
from jasna.cli_parser import build_parser
from jasna.engine_paths import default_restoration_model_path
from jasna.main import _session_config_from_args
from jasna.mosaic.detection_registry import recommended_score_threshold
from jasna.session_config import LTX_CROP_VSR_RTX_DEBLUR, LTX_SHARPEN_RTX_DEBLUR

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


def _cli_config(extra_args: list[str] | None = None):
    args = build_parser().parse_args(["--input", "a.mp4", "--output", "b.mp4", *(extra_args or [])])
    threshold = args.detection_score_threshold
    return _session_config_from_args(
        args,
        codec="hevc",
        encoder_settings={},
        detection_model_name="rfdetr-v6",
        detection_model_path=GUI_WEIGHTS_DIR / "rfdetr-v6.onnx",
        detection_score_threshold=recommended_score_threshold("rfdetr-v6") if threshold is None else threshold,
        restoration_model_path=GUI_WEIGHTS_DIR / default_restoration_model_path("basicvsrpp").name,
        lut_path=None,
    )


def test_cli_rtx_defaults() -> None:
    config = _cli_config()

    assert config.rtx_deblur == "none"
    assert config.rtx_strength == 1.0
    assert config.rtx_highbitrate is False


def test_cli_rtx_options_are_mapped() -> None:
    config = _cli_config(["--rtx-deblur", "high", "--rtx-strength", "0.5", "--rtx-highbitrate"])

    assert config.rtx_deblur == "high"
    assert config.rtx_strength == 0.5
    assert config.rtx_highbitrate is True


@pytest.mark.parametrize(
    "extra_args, expected",
    [
        (["--restoration-model-name", "ltx", "--secondary-restoration", "rtx-super-res"], LTX_SHARPEN_RTX_DEBLUR),
        (["--ltx-crop-vsr", "rtx-super-res", "--ltx-crop-vsr-source", "s.mp4"], LTX_CROP_VSR_RTX_DEBLUR),
        (["--secondary-restoration", "rtx-super-res"], "none"),
        (
            ["--restoration-model-name", "ltx", "--secondary-restoration", "rtx-super-res", "--rtx-deblur", "none"],
            "none",
        ),
    ],
)
def test_cli_rtx_deblur_default_follows_where_the_pass_sharpens(extra_args, expected) -> None:
    assert _cli_config(extra_args).rtx_deblur == expected


def test_cli_frame_generation_options_are_mapped() -> None:
    defaults = _cli_config()
    assert defaults.frame_generation_backend == "rtx"
    assert defaults.frame_generation_rtx_mode == "medium"
    assert defaults.frame_generation_model_path == ""

    config = _cli_config(
        [
            "--frame-generation", "4x",
            "--frame-generation-backend", "rife",
            "--frame-generation-rtx-mode", "high",
            "--frame-generation-model-path", " /weights/rife.pth ",
        ]
    )

    assert config.frame_generation == "4x"
    assert config.frame_generation_backend == "rife"
    assert config.frame_generation_rtx_mode == "high"
    assert config.frame_generation_model_path == "/weights/rife.pth"


def test_cli_swiftvr_distill_args_are_mapped() -> None:
    defaults = _cli_config()
    assert defaults.swiftvr_distill_model == ""
    assert defaults.swiftvr_distill_view_window == 15
    assert defaults.swiftvr_distill_strength == 0.75
    assert defaults.swiftvr_distill_stabilize == 0

    config = _cli_config(
        [
            "--secondary-restoration", "swiftvr-distill",
            "--swiftvr-distill-model", "/weights/best.pt",
            "--swiftvr-distill-view-window", "15",
            "--swiftvr-distill-strength", "0.75",
            "--swiftvr-distill-stabilize", "2",
        ]
    )

    assert config.secondary_restoration == "swiftvr-distill"
    assert config.swiftvr_distill_model == "/weights/best.pt"
    assert config.swiftvr_distill_view_window == 15
    assert config.swiftvr_distill_strength == 0.75
    assert config.swiftvr_distill_stabilize == 2


def test_cli_seedvr2_is_a_restoration_model() -> None:
    config = _cli_config(["--restoration-model-name", "seedvr2", "--seedvr2-repo", "/repo", "--seedvr2-window", "17"])

    assert config.restoration_model_name == "seedvr2"
    assert config.seedvr2_repo == "/repo"
    assert config.seedvr2_window == 17

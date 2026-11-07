"""modi-only CLI validation tests (frame generation, the torchcodec video backend).

modi 独自。上流の tests/test_main_validation.py と同じく factories.py の ``cli_args`` / ``run_cli``
と conftest の ``nvidia_vendor`` fixture に依存する。上流のファイルには追記しない。
"""
import pytest

from factories import cli_args, run_cli

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


def _run_main_with_args(tmp_path, extra_args):
    return run_cli(cli_args(tmp_path, *extra_args))


class TestMainValidationModi:
    def test_frame_generation_4x_reaches_the_config(self, tmp_path):
        from unittest.mock import patch

        with patch("jasna.accelerator.supports_frame_generation", return_value=False):
            config = _run_main_with_args(
                tmp_path, ["--frame-generation", "4x", "--frame-generation-backend", "rife"]
            ).call_args.kwargs["config"]
        assert (config.frame_generation, config.frame_generation_backend) == ("4x", "rife")

    def test_frame_generation_rife_runs_on_any_gpu(self, tmp_path, capsys):
        # The RTX 40-series gate is the rtx backend's; RIFE needs no such GPU.
        from unittest.mock import patch

        with patch("jasna.accelerator.supports_frame_generation", return_value=False):
            _run_main_with_args(tmp_path, ["--frame-generation", "2x", "--frame-generation-backend", "rife"])
        assert "RTX 40-series" not in capsys.readouterr().err

    def test_torchcodec_streaming_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="streaming"):
            _run_main_with_args(tmp_path, ["--video-backend", "torchcodec", "--stream"])

    def test_torchcodec_encode_segments_rejected(self, tmp_path):
        # Smart-render fragments always use the native encoder, so forcing
        # torchcodec encode alongside --segments is rejected up front (before
        # the input is probed).
        with pytest.raises(ValueError, match="torchcodec encode"):
            _run_main_with_args(
                tmp_path,
                ["--video-encode-backend", "torchcodec", "--segments", "1-2"],
            )

    def test_torchcodec_decode_retarget_rejected(self, tmp_path):
        # The fps-retarget frame stride is only implemented by the native reader.
        with pytest.raises(ValueError, match="retarget"):
            _run_main_with_args(
                tmp_path,
                ["--video-decode-backend", "torchcodec", "--retarget-high-fps"],
            )

    def test_torchcodec_av1_allowed(self, tmp_path):
        # torchcodec supports av1_nvenc, so --video-backend torchcodec --codec av1
        # must NOT raise (Pipeline is mocked, so this reaches construction cleanly).
        _run_main_with_args(tmp_path, ["--video-backend", "torchcodec", "--codec", "av1"])

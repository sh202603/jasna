"""modi-only TensorRT builder tests under the TensorRT-RTX flavor (mocked builder).

modi 独自。上流の tests/test_trt_utils.py は実エンジンを組む形に書き直されたので、
modi の RTX 分岐（BuilderFlag.FP16 を立てない）は mock の builder で別に確かめる。
"""
from unittest.mock import MagicMock, patch

import pytest
import torch

from jasna.trt import compile_onnx_to_tensorrt_engine, get_onnx_tensorrt_engine_path
from jasna.trt._backend import trt


def _setup_trt_mocks(tmp_path, *, num_inputs=1, num_outputs=1, dynamic_shape=False, parse_ok=True, build_result=b"engine_bytes"):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx")

    engine_path = get_onnx_tensorrt_engine_path(onnx_path, fp16=True)
    if engine_path.exists():
        engine_path.unlink()

    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input.shape = (-1, 3, 64, 64) if dynamic_shape else (1, 3, 64, 64)
    mock_input.dtype = trt.DataType.FLOAT

    mock_output = MagicMock()
    mock_output.name = "output"
    mock_output.dtype = trt.DataType.FLOAT

    mock_network = MagicMock()
    mock_network.num_inputs = num_inputs
    mock_network.num_outputs = num_outputs
    mock_network.get_input = MagicMock(return_value=mock_input)
    mock_network.get_output = MagicMock(return_value=mock_output)

    mock_parser = MagicMock()
    mock_parser.parse.return_value = parse_ok
    mock_parser.num_errors = 0 if parse_ok else 1
    mock_parser.get_error = lambda i: "parse error"

    mock_config = MagicMock()
    mock_builder = MagicMock()
    mock_builder.create_network.return_value = mock_network
    mock_builder.create_builder_config.return_value = mock_config
    mock_builder.build_serialized_network.return_value = build_result

    return onnx_path, engine_path, mock_builder, mock_parser, mock_config, mock_network


class TestCompileOnnxToTensorrtEngineModi:
    @pytest.fixture(autouse=True)
    def _standard_flavor(self, monkeypatch):
        """These tests exercise the builder plumbing with fake ONNX bytes; pin
        the standard flavor so the RTX fp16 pre-conversion (which needs a real
        ONNX graph) stays out of the way in either venv."""
        monkeypatch.setattr("jasna.trt.TRT_FLAVOR", "standard")

    def test_fp16_flag_skipped_under_rtx(self, tmp_path):
        # TensorRT-RTX is strongly typed: BuilderFlag.FP16 must not be set,
        # while the I/O HALF forcing (which carries the FP16 intent) stays.
        # The fp16 pre-conversion needs a real ONNX graph — stub it out.
        onnx_path, engine_path, mock_builder, mock_parser, mock_config, network = _setup_trt_mocks(tmp_path)

        with (
            patch("jasna.trt.TRT_FLAVOR", "rtx"),
            patch("jasna.trt._convert_onnx_bytes_to_fp16", side_effect=lambda b: b),
            patch("jasna.trt.trt.Logger"),
            patch("jasna.trt.trt.Builder", return_value=mock_builder),
            patch("jasna.trt.trt.OnnxParser", return_value=mock_parser),
            patch("jasna.trt.torch.cuda.device", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ):
            result = compile_onnx_to_tensorrt_engine(onnx_path, torch.device("cuda:0"), fp16=True, workspace_gb=20)

        assert result == engine_path
        mock_config.set_flag.assert_not_called()
        assert network.get_input(0).dtype == trt.DataType.HALF
        assert network.get_output(0).dtype == trt.DataType.HALF

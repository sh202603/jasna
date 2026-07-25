from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import tensorrt as trt
import torch

from jasna.trt.trt_runner import TrtRunner


def _build_runner(tmp_path, *, num_outputs=1):
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"fake")

    mock_engine = MagicMock()
    mock_engine.num_io_tensors = 1 + num_outputs
    names = ["input"] + [f"output_{i}" for i in range(num_outputs)]
    mock_engine.get_tensor_name = lambda i: names[i]
    mock_engine.get_tensor_mode = lambda name: (
        trt.TensorIOMode.INPUT if name == "input" else trt.TensorIOMode.OUTPUT
    )
    mock_engine.get_tensor_dtype = lambda name: trt.DataType.FLOAT
    mock_engine.get_tensor_shape = lambda name: (1, 3, 64, 64)

    mock_context = MagicMock()
    mock_context.get_tensor_shape = lambda name: (1, 3, 64, 64)
    mock_context.set_input_shape = MagicMock()
    mock_context.set_tensor_address = MagicMock()
    mock_context.execute_async_v3 = MagicMock()
    mock_engine.create_execution_context.return_value = mock_context

    mock_runtime = MagicMock()
    mock_runtime.deserialize_cuda_engine.return_value = mock_engine

    with (
        patch("jasna.trt.trt_runner.trt.Logger"),
        patch("jasna.trt.trt_runner.trt.Runtime", return_value=mock_runtime),
    ):
        runner = TrtRunner(
            engine_path=engine_path,
            input_shapes={"input": (1, 3, 64, 64)},
            device=torch.device("cuda:0"),
        )
    return runner, mock_context


class TestTrtRunnerInit:
    def test_basic_init(self, tmp_path):
        runner, ctx = _build_runner(tmp_path)
        assert runner.input_names == ["input"]
        assert len(runner.output_names) == 1
        assert "output_0" in runner.outputs
        assert runner.input_dtypes["input"] == torch.float32

    def test_multiple_outputs(self, tmp_path):
        runner, ctx = _build_runner(tmp_path, num_outputs=3)
        assert len(runner.output_names) == 3
        assert len(runner.outputs) == 3

    def test_deserialization_failure_raises(self, tmp_path):
        engine_path = tmp_path / "bad.engine"
        engine_path.write_bytes(b"bad")

        mock_runtime = MagicMock()
        mock_runtime.deserialize_cuda_engine.return_value = None

        with (
            patch("jasna.trt.trt_runner.trt.Logger"),
            patch("jasna.trt.trt_runner.trt.Runtime", return_value=mock_runtime),
        ):
            with pytest.raises(RuntimeError, match="Failed to deserialize"):
                TrtRunner(engine_path=engine_path, input_shapes={"input": (1, 3, 64, 64)}, device=torch.device("cuda:0"))

    def test_context_creation_failure_raises(self, tmp_path):
        engine_path = tmp_path / "bad.engine"
        engine_path.write_bytes(b"bad")

        mock_engine = MagicMock()
        mock_engine.create_execution_context.return_value = None

        mock_runtime = MagicMock()
        mock_runtime.deserialize_cuda_engine.return_value = mock_engine

        with (
            patch("jasna.trt.trt_runner.trt.Logger"),
            patch("jasna.trt.trt_runner.trt.Runtime", return_value=mock_runtime),
        ):
            with pytest.raises(RuntimeError, match="Failed to create TensorRT execution context"):
                TrtRunner(engine_path=engine_path, input_shapes={"input": (1, 3, 64, 64)}, device=torch.device("cuda:0"))


class TestTrtRunnerInfer:
    def test_infer_sets_address_and_executes(self, tmp_path):
        runner, ctx = _build_runner(tmp_path)
        x = torch.randn(1, 3, 64, 64, device="cuda:0")
        result = runner.infer({"input": x})
        ctx.set_tensor_address.assert_called()
        ctx.execute_async_v3.assert_called_once()
        assert result is runner.outputs

    def test_infer_returns_output_dict(self, tmp_path):
        runner, ctx = _build_runner(tmp_path, num_outputs=2)
        x = torch.randn(1, 3, 64, 64, device="cuda:0")
        result = runner.infer({"input": x})
        assert "output_0" in result
        assert "output_1" in result


class TestTrtRunnerStaticShapeBinding:
    def _build(self, tmp_path, engine_shape, requested, context_shape=None):
        engine_path = tmp_path / "model.engine"
        engine_path.write_bytes(b"fake")

        mock_engine = MagicMock()
        mock_engine.num_io_tensors = 2
        names = ["input", "output_0"]
        mock_engine.get_tensor_name = lambda i: names[i]
        mock_engine.get_tensor_mode = lambda name: (
            trt.TensorIOMode.INPUT if name == "input" else trt.TensorIOMode.OUTPUT
        )
        mock_engine.get_tensor_dtype = lambda name: trt.DataType.FLOAT
        mock_engine.get_tensor_shape = lambda name: engine_shape

        bound = context_shape if context_shape is not None else engine_shape
        mock_context = MagicMock()
        mock_context.get_tensor_shape = lambda name: bound
        mock_engine.create_execution_context.return_value = mock_context

        mock_runtime = MagicMock()
        mock_runtime.deserialize_cuda_engine.return_value = mock_engine

        with (
            patch("jasna.trt.trt_runner.trt.Logger"),
            patch("jasna.trt.trt_runner.trt.Runtime", return_value=mock_runtime),
        ):
            return TrtRunner(
                engine_path=engine_path,
                input_shapes={"input": requested},
                device=torch.device("cuda:0"),
            ), mock_context

    def test_static_engine_binds_engine_shape_over_requested(self):
        runner, ctx = self._build(
            self.tmp_path, engine_shape=(4, 3, 64, 64), requested=(1, 3, 64, 64)
        )
        assert runner.input_shapes["input"] == (4, 3, 64, 64)
        ctx.set_input_shape.assert_called_once_with("input", (4, 3, 64, 64))

    def test_dynamic_engine_keeps_requested_shape(self):
        runner, ctx = self._build(
            self.tmp_path,
            engine_shape=(-1, 3, 64, 64),
            requested=(2, 3, 64, 64),
            context_shape=(2, 3, 64, 64),
        )
        assert runner.input_shapes["input"] == (2, 3, 64, 64)

    def test_rejected_shape_raises(self):
        with pytest.raises(RuntimeError, match="rejected input shape"):
            self._build(
                self.tmp_path,
                engine_shape=(-1, 3, 64, 64),
                requested=(2, 3, 64, 64),
                context_shape=(4, 3, 64, 64),
            )

    def test_infer_rejects_mismatched_input_shape(self):
        runner, ctx = self._build(
            self.tmp_path, engine_shape=(4, 3, 64, 64), requested=(4, 3, 64, 64)
        )
        with pytest.raises(ValueError, match="does not match"):
            runner.infer({"input": torch.zeros(2, 3, 64, 64)})

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp_path = tmp_path

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from jasna.trt._backend import trt
from jasna.trt.trt_runner import TrtRunner


def _build_engine_mocks(
    *,
    num_outputs=1,
    engine_batch=1,
    initial_batch=1,
    min_batch=1,
):
    mock_engine = MagicMock()
    mock_engine.num_io_tensors = 1 + num_outputs
    names = ["input"] + [f"output_{i}" for i in range(num_outputs)]
    mock_engine.get_tensor_name = lambda i: names[i]
    mock_engine.get_tensor_mode = lambda name: (
        trt.TensorIOMode.INPUT if name == "input" else trt.TensorIOMode.OUTPUT
    )
    mock_engine.get_tensor_dtype = lambda name: trt.DataType.FLOAT
    mock_engine.get_tensor_shape = lambda name: (engine_batch, 3, 64, 64)
    mock_engine.get_tensor_profile_shape = lambda name, profile: (
        (min_batch, 3, 64, 64),
        (initial_batch, 3, 64, 64),
        (initial_batch, 3, 64, 64),
    )

    mock_context = MagicMock()
    current_batch = initial_batch

    def set_input_shape(_name, shape):
        nonlocal current_batch
        if engine_batch > 0 and shape[0] != engine_batch:
            return False
        current_batch = shape[0]
        return True

    mock_context.get_tensor_shape = lambda name: (
        current_batch,
        3,
        64,
        64,
    )
    mock_context.set_input_shape = MagicMock(side_effect=set_input_shape)
    mock_context.set_tensor_address = MagicMock()
    mock_context.execute_async_v3 = MagicMock(return_value=True)
    mock_engine.create_execution_context.return_value = mock_context

    mock_runtime = MagicMock()
    mock_runtime.deserialize_cuda_engine.return_value = mock_engine
    return mock_engine, mock_context, mock_runtime


def _build_runner(
    tmp_path,
    *,
    num_outputs=1,
    engine_batch=1,
    initial_batch=1,
    min_batch=1,
    use_cuda_graphs=False,
):
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"fake")

    mock_engine, mock_context, mock_runtime = _build_engine_mocks(
        num_outputs=num_outputs,
        engine_batch=engine_batch,
        initial_batch=initial_batch,
        min_batch=min_batch,
    )

    with (
        patch("jasna.trt.trt_runner.trt.Logger"),
        patch("jasna.trt.trt_runner.trt.Runtime", return_value=mock_runtime),
    ):
        runner = TrtRunner(
            engine_path=engine_path,
            input_shapes={"input": (initial_batch, 3, 64, 64)},
            device=torch.device("cuda:0"),
            use_cuda_graphs=use_cuda_graphs,
        )
    return runner, mock_context


def _build_rtx_runner(
    tmp_path,
    monkeypatch,
    *,
    engine_batch=-1,
    initial_batch=4,
    min_batch=1,
    use_cuda_graphs=True,
    execute_side_effect=None,
    cache_deserialize=True,
    trt_version="1.4.0.76",
    env=None,
):
    """Build a TrtRunner with TRT_FLAVOR patched to "rtx" and the module's trt
    reference replaced by a MagicMock (the standard tensorrt module lacks the
    RTX-only APIs). The patches stay active for the whole test via monkeypatch
    so post-construction calls (infer fallback, close) still see them.

    Graph capture ships opt-in (env default off), so the helper enables the
    env unless the test overrides it."""
    monkeypatch.delenv("JASNA_TRT_RTX_SPECIALIZATION", raising=False)
    monkeypatch.setenv("JASNA_TRT_RUNNER_CUDAGRAPHS", "1")
    for key, value in (env or {}).items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"fake")

    mock_engine, mock_context, mock_runtime = _build_engine_mocks(
        engine_batch=engine_batch,
        initial_batch=initial_batch,
        min_batch=min_batch,
    )
    if execute_side_effect is not None:
        mock_context.execute_async_v3 = MagicMock(side_effect=execute_side_effect)

    mock_config = MagicMock()
    mock_cache = MagicMock()
    mock_cache.serialize.return_value = b"CACHE"
    mock_cache.deserialize.return_value = cache_deserialize
    mock_config.create_runtime_cache.return_value = mock_cache
    mock_engine.create_runtime_config.return_value = mock_config
    mock_engine.create_execution_context = MagicMock(return_value=mock_context)

    mock_trt = MagicMock()
    mock_trt.__version__ = trt_version
    mock_trt.Runtime.return_value = mock_runtime

    monkeypatch.setattr("jasna.trt.trt_runner.trt", mock_trt)
    monkeypatch.setattr("jasna.trt.trt_runner.TRT_FLAVOR", "rtx")

    runner = TrtRunner(
        engine_path=engine_path,
        input_shapes={"input": (initial_batch, 3, 64, 64)},
        device=torch.device("cuda:0"),
        use_cuda_graphs=use_cuda_graphs,
    )
    mocks = {
        "engine": mock_engine,
        "context": mock_context,
        "config": mock_config,
        "cache": mock_cache,
        "trt": mock_trt,
        "engine_path": engine_path,
    }
    return runner, mocks


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

    def test_dynamic_engine_rebinds_without_padding(self, tmp_path):
        runner, ctx = _build_runner(
            tmp_path,
            engine_batch=-1,
            initial_batch=4,
        )
        x = torch.randn(1, 3, 64, 64, device="cuda:0")

        result = runner.infer({"input": x})

        assert result["output_0"].shape[0] == 1
        assert ctx.set_input_shape.call_args.args[1][0] == 1

    def test_fixed_engine_pads_partial_batch_and_trims_outputs(self, tmp_path):
        runner, ctx = _build_runner(
            tmp_path,
            engine_batch=4,
            initial_batch=4,
        )
        x = torch.randn(1, 3, 64, 64, device="cuda:0")

        result = runner.infer({"input": x})

        assert result["output_0"].shape[0] == 1
        assert ctx.set_input_shape.call_args.args[1][0] == 4

    def test_fixed_profile_pads_partial_batch_and_trims_outputs(self, tmp_path):
        runner, _ = _build_runner(
            tmp_path,
            engine_batch=-1,
            initial_batch=4,
            min_batch=4,
        )
        x = torch.randn(1, 3, 64, 64, device="cuda:0")

        result = runner.infer({"input": x})

        assert runner.dynamic_batch is False
        assert result["output_0"].shape[0] == 1

    def test_fixed_engine_rejects_oversized_batch(self, tmp_path):
        runner, _ = _build_runner(
            tmp_path,
            engine_batch=4,
            initial_batch=4,
        )
        x = torch.randn(5, 3, 64, 64, device="cuda:0")

        with pytest.raises(ValueError, match="exceeds fixed TensorRT batch 4"):
            runner.infer({"input": x})


class TestTrtRunnerRtxCudaGraphs:
    def test_warmup_runs_three_iterations_on_private_stream(
        self, tmp_path, monkeypatch
    ):
        runner, mocks = _build_rtx_runner(tmp_path, monkeypatch)

        assert runner._graphs_enabled is True
        assert runner._stream is not None
        ctx = mocks["context"]
        assert ctx.execute_async_v3.call_count == 3
        default_stream = torch.cuda.current_stream(runner.device).cuda_stream
        for call in ctx.execute_async_v3.call_args_list:
            assert call.args[0] == runner._stream.cuda_stream
            assert call.args[0] != default_stream
        assert (
            mocks["config"].cuda_graph_strategy
            is mocks["trt"].CudaGraphStrategy.WHOLE_GRAPH_CAPTURE
        )

    def test_input_address_set_only_at_bind_time(self, tmp_path, monkeypatch):
        runner, mocks = _build_rtx_runner(tmp_path, monkeypatch)
        ctx = mocks["context"]
        input_addr_calls = [
            c for c in ctx.set_tensor_address.call_args_list if c.args[0] == "input"
        ]
        assert len(input_addr_calls) == 1
        staging_ptr = runner._staging["input"].data_ptr()
        assert input_addr_calls[0].args[1] == staging_ptr

        before = len(ctx.set_tensor_address.call_args_list)
        x = torch.randn(4, 3, 64, 64, device="cuda:0")
        runner.infer({"input": x})
        runner.infer({"input": x})
        assert len(ctx.set_tensor_address.call_args_list) == before
        assert runner._staging["input"].data_ptr() == staging_ptr

    def test_dynamic_engine_pads_to_engine_batch_without_rebinding(
        self, tmp_path, monkeypatch
    ):
        runner, mocks = _build_rtx_runner(tmp_path, monkeypatch)
        assert runner.dynamic_batch is True
        ctx = mocks["context"]
        binds_before = ctx.set_input_shape.call_count

        x = torch.randn(1, 3, 64, 64, device="cuda:0")
        result = runner.infer({"input": x})

        assert result["output_0"].shape[0] == 1
        assert ctx.set_input_shape.call_count == binds_before

    def test_execute_failure_falls_back_without_graphs(self, tmp_path, monkeypatch):
        runner, mocks = _build_rtx_runner(
            tmp_path,
            monkeypatch,
            execute_side_effect=[False] + [True] * 10,
        )

        assert runner._graphs_enabled is False
        assert runner._staging == {}
        assert mocks["engine"].create_execution_context.call_count >= 2

        ctx = mocks["context"]
        x = torch.randn(1, 3, 64, 64, device="cuda:0")
        result = runner.infer({"input": x})
        assert result["output_0"].shape[0] == 1
        assert (
            ctx.execute_async_v3.call_args.args[0]
            == torch.cuda.current_stream(runner.device).cuda_stream
        )

    def test_graphs_default_off_without_env(self, tmp_path, monkeypatch):
        runner, mocks = _build_rtx_runner(
            tmp_path,
            monkeypatch,
            env={"JASNA_TRT_RUNNER_CUDAGRAPHS": None},
        )

        assert runner._graphs_enabled is False
        assert runner._staging == {}
        assert mocks["context"].execute_async_v3.call_count == 0  # no warmup

    def test_env_gate_disables_graphs(self, tmp_path, monkeypatch):
        runner, mocks = _build_rtx_runner(
            tmp_path,
            monkeypatch,
            env={"JASNA_TRT_RUNNER_CUDAGRAPHS": "0"},
        )

        assert runner._graphs_enabled is False
        assert runner._staging == {}
        ctx = mocks["context"]
        assert ctx.execute_async_v3.call_count == 0  # no warmup

        x = torch.randn(4, 3, 64, 64, device="cuda:0")
        before = ctx.set_tensor_address.call_count
        runner.infer({"input": x})
        assert ctx.set_tensor_address.call_count == before + 1  # per-call input
        assert (
            ctx.execute_async_v3.call_args.args[0]
            == torch.cuda.current_stream(runner.device).cuda_stream
        )
        # The runtime cache is independent of the graphs gate.
        mocks["engine"].create_runtime_config.assert_called_once()

    def test_standard_flavor_never_touches_runtime_config(self, tmp_path):
        runner, ctx = _build_runner(tmp_path, use_cuda_graphs=True)

        assert runner._graphs_enabled is False
        assert runner._staging == {}
        runner.engine.create_runtime_config.assert_not_called()

        x = torch.randn(1, 3, 64, 64, device="cuda:0")
        runner.infer({"input": x})
        assert (
            ctx.execute_async_v3.call_args.args[0]
            == torch.cuda.current_stream(runner.device).cuda_stream
        )

    def test_close_persists_versioned_cache_and_removes_legacy(
        self, tmp_path, monkeypatch
    ):
        legacy = tmp_path / "model.engine.jitcache"
        legacy.write_bytes(b"OLD")
        runner, mocks = _build_rtx_runner(
            tmp_path,
            monkeypatch,
            use_cuda_graphs=False,
        )

        x = torch.randn(4, 3, 64, 64, device="cuda:0")
        runner.infer({"input": x})
        runner.close()

        tagged = tmp_path / "model.engine.trtrtx1.4.jitcache"
        assert tagged.read_bytes() == b"CACHE"
        assert not legacy.exists()

    def test_legacy_cache_seeds_deserialize(self, tmp_path, monkeypatch):
        legacy = tmp_path / "model.engine.jitcache"
        legacy.write_bytes(b"OLD")
        runner, mocks = _build_rtx_runner(
            tmp_path,
            monkeypatch,
            use_cuda_graphs=False,
        )

        mocks["cache"].deserialize.assert_called_once_with(b"OLD")

    def test_stale_cache_is_tolerated(self, tmp_path, monkeypatch):
        tagged = tmp_path / "model.engine.trtrtx1.4.jitcache"
        tagged.write_bytes(b"STALE")
        runner, mocks = _build_rtx_runner(
            tmp_path,
            monkeypatch,
            use_cuda_graphs=False,
            cache_deserialize=False,
        )

        mocks["cache"].deserialize.assert_called_once_with(b"STALE")
        assert runner.context is not None

    def test_specialization_env_sets_strategy(self, tmp_path, monkeypatch):
        runner, mocks = _build_rtx_runner(
            tmp_path,
            monkeypatch,
            use_cuda_graphs=False,
            env={"JASNA_TRT_RTX_SPECIALIZATION": "eager"},
        )

        assert (
            mocks["config"].dynamic_shapes_kernel_specialization_strategy
            is mocks["trt"].DynamicShapesKernelSpecializationStrategy.EAGER
        )

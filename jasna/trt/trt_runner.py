from __future__ import annotations

import os

from pathlib import Path

import logging

import torch

from jasna.accelerator import current_stream, new_stream, stream_context
from jasna.trt import _engine_io_names, _trt_dtype_to_torch, get_trt_logger
from jasna.trt._backend import TRT_FLAVOR, trt
from jasna.tensor_utils import pad_batch_with_last

logger = logging.getLogger(__name__)

_CUDAGRAPHS_ENV = "JASNA_TRT_RUNNER_CUDAGRAPHS"
_SPECIALIZATION_ENV = "JASNA_TRT_RTX_SPECIALIZATION"


def _specialization_strategy():
    """Optional TRT-RTX kernel-specialization override; None keeps the default."""
    value = os.environ.get(_SPECIALIZATION_ENV, "").strip().lower()
    mapping = {"lazy": "LAZY", "eager": "EAGER", "none": "NONE"}
    if value in mapping:
        return getattr(trt.DynamicShapesKernelSpecializationStrategy, mapping[value])
    return None


class TrtRunner:
    def __init__(
        self,
        engine_path: Path,
        input_shapes: dict[str, tuple[int, ...]] | list[tuple[int, ...]],
        device: torch.device,
        *,
        use_cuda_graphs: bool = False,
    ) -> None:
        self.engine_path = engine_path
        self._setup(
            engine_path.read_bytes(),
            input_shapes,
            device,
            str(engine_path),
            use_cuda_graphs=use_cuda_graphs,
        )

    @classmethod
    def from_engine_bytes(
        cls,
        engine_bytes: bytes,
        input_shapes: dict[str, tuple[int, ...]] | list[tuple[int, ...]],
        device: torch.device,
        source: str = "<memory>",
        *,
        use_cuda_graphs: bool = False,
    ) -> "TrtRunner":
        self = cls.__new__(cls)
        self.engine_path = None
        self._setup(
            engine_bytes,
            input_shapes,
            device,
            source,
            use_cuda_graphs=use_cuda_graphs,
        )
        return self

    def _setup(
        self,
        engine_bytes: bytes,
        input_shapes: dict[str, tuple[int, ...]] | list[tuple[int, ...]],
        device: torch.device,
        source: str,
        use_cuda_graphs: bool = False,
    ) -> None:
        self.device = device
        # Whole-graph capture measured neutral-to-negative on Linux (the RTX
        # deficit is JIT kernel quality, not launch overhead; the staging copy
        # and cross-stream sync cost ~1% at the stage level), so it ships
        # opt-in. WDDM launch overhead may still make it pay off on Windows.
        self._graphs_enabled = (
            TRT_FLAVOR == "rtx"
            and use_cuda_graphs
            and os.environ.get(_CUDAGRAPHS_ENV, "0") != "0"
        )
        self._runtime_cache = None
        self._runtime_cache_path: Path | None = None
        self._staging: dict[str, torch.Tensor] = {}
        self._stream = None

        self.runtime = trt.Runtime(get_trt_logger())
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {source}")
        self.context = self._create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self.input_names, self.output_names = _engine_io_names(self.engine)

        if isinstance(input_shapes, list):
            input_shapes = dict(zip(self.input_names, input_shapes))

        self.input_dtypes: dict[str, torch.dtype] = {
            name: _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            for name in self.input_names
        }
        input_name = self.input_names[0]
        engine_batch = int(self.engine.get_tensor_shape(input_name)[0])
        if engine_batch < 0:
            min_shape, _, max_shape = self.engine.get_tensor_profile_shape(
                input_name,
                0,
            )
            min_batch = int(min_shape[0])
            max_batch = int(max_shape[0])
            self.dynamic_batch = min_batch != max_batch
            self._engine_batch = max_batch
        else:
            self.dynamic_batch = False
            self._engine_batch = engine_batch
        self.outputs: dict[str, torch.Tensor] = {}
        self._cur_shapes: dict[str, tuple[int, ...]] = {}
        if self._graphs_enabled:
            self._stream = new_stream(device)
        self._bind({name: tuple(int(d) for d in input_shapes[name]) for name in self.input_names})
        if self._graphs_enabled:
            self._warmup_graph_capture()

    def _legacy_cache_path(self) -> Path | None:
        if self.engine_path is None:
            return None
        engine_path = Path(self.engine_path)
        return engine_path.with_name(engine_path.name + ".jitcache")

    def _runtime_cache_file(self) -> Path | None:
        """Version-tagged JIT cache path (e.g. `<engine>.trtrtx1.4.jitcache`).

        The engine filename does not encode the TensorRT-RTX version, so an
        untagged cache would be silently invalidated (and re-JITed every
        process start) after a tensorrt-rtx upgrade.
        """
        if self.engine_path is None:
            return None
        engine_path = Path(self.engine_path)
        parts = str(getattr(trt, "__version__", "")).split(".")
        if len(parts) < 2:
            return self._legacy_cache_path()
        return engine_path.with_name(
            f"{engine_path.name}.trtrtx{parts[0]}.{parts[1]}.jitcache"
        )

    def _create_execution_context(self):
        """Create the execution context; under TensorRT-RTX, with a JIT kernel
        disk cache next to the engine file and (opt-in) whole-graph CUDA graph
        capture.

        TensorRT-RTX defers kernel generation to context creation (JIT), which
        costs seconds per engine on every process start. The runtime cache
        makes later loads near-instant. Failure-tolerant: any cache/config
        problem falls back to a plain context (and disables graph capture,
        which needs the runtime config to be applied). In-memory engines
        (encrypted unet-4x) have no path to cache next to, so they skip the
        cache.
        """
        if TRT_FLAVOR == "rtx":
            cache_path = self._runtime_cache_file()
            try:
                config = self.engine.create_runtime_config()
                if self._graphs_enabled:
                    config.cuda_graph_strategy = (
                        trt.CudaGraphStrategy.WHOLE_GRAPH_CAPTURE
                    )
                strategy = _specialization_strategy()
                if strategy is not None:
                    config.dynamic_shapes_kernel_specialization_strategy = strategy
                cache = None
                if cache_path is not None:
                    cache = config.create_runtime_cache()
                    seed_path = cache_path
                    if not seed_path.exists():
                        # A cache written by an older jasna (untagged name) is
                        # safe to seed from: a version mismatch just makes
                        # deserialize() return False below.
                        legacy = self._legacy_cache_path()
                        if legacy is not None and legacy.exists():
                            seed_path = legacy
                    if seed_path.exists():
                        if not cache.deserialize(seed_path.read_bytes()):
                            logger.debug(
                                "Stale TRT-RTX runtime cache ignored: %s", seed_path
                            )
                    config.set_runtime_cache(cache)
                context = self.engine.create_execution_context(runtime_config=config)
                if context is not None:
                    self._runtime_cache = cache
                    self._runtime_cache_path = cache_path
                    self._persist_runtime_cache()
                    return context
            except Exception:
                logger.debug("TRT-RTX runtime config unavailable", exc_info=True)
            self._graphs_enabled = False
        return self.engine.create_execution_context()

    def _persist_runtime_cache(self) -> None:
        """Write the runtime cache to disk (atomic, failure-tolerant).

        Called again after warmup and on close(): lazily shape-specialized
        kernels are JITed only once inference runs, so a single write at
        context creation would lose them and every process start would
        re-specialize.
        """
        if self._runtime_cache is None or self._runtime_cache_path is None:
            return
        try:
            data = bytes(self._runtime_cache.serialize())
            tmp_path = self._runtime_cache_path.with_name(
                f"{self._runtime_cache_path.name}.tmp{os.getpid()}"
            )
            tmp_path.write_bytes(data)
            os.replace(tmp_path, self._runtime_cache_path)
            legacy = self._legacy_cache_path()
            if legacy is not None and legacy != self._runtime_cache_path:
                legacy.unlink(missing_ok=True)
        except Exception:
            logger.debug("Could not persist TRT-RTX runtime cache", exc_info=True)

    def _bind(self, input_shapes: dict[str, tuple[int, ...]]) -> None:
        for name in self.input_names:
            accepted = self.context.set_input_shape(name, input_shapes[name])
            if accepted is False:
                raise ValueError(
                    f"TensorRT engine rejected input shape for {name}: "
                    f"{input_shapes[name]}"
                )
        dev = torch.device(self.device)
        if self._graphs_enabled:
            # CUDA graphs record tensor addresses, so inputs go through
            # runner-owned staging buffers whose addresses never change.
            self._staging = {
                name: torch.empty(
                    size=input_shapes[name],
                    dtype=self.input_dtypes[name],
                    device=dev,
                )
                for name in self.input_names
            }
            for name, tensor in self._staging.items():
                self.context.set_tensor_address(name, int(tensor.data_ptr()))
        self.outputs = {}
        for name in self.output_names:
            shape = tuple(int(d) for d in self.context.get_tensor_shape(name))
            if any(d <= 0 for d in shape):
                raise RuntimeError(
                    f"TensorRT output shape for {name} is unresolved: {shape}"
                )
            dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            t = torch.empty(size=shape, dtype=dtype, device=dev)
            self.outputs[name] = t
            self.context.set_tensor_address(name, int(t.data_ptr()))
        self._cur_shapes = dict(input_shapes)

    def _warmup_graph_capture(self) -> None:
        """Run dummy inferences so graph capture completes before pipeline
        threads start.

        TRT-RTX skips capture on the first execution and captures on the
        second; three iterations guarantee replay. Capture must not overlap
        concurrent CUDA work from other threads, hence warmup at construction
        time. The cache re-persist afterwards saves the kernels specialized
        for the engine-batch shape.
        """
        dummy = {
            name: torch.zeros(
                size=(self._engine_batch, *self._cur_shapes[name][1:]),
                dtype=self.input_dtypes[name],
                device=torch.device(self.device),
            )
            for name in self.input_names
        }
        for _ in range(3):
            self.infer(dummy)
        torch.cuda.synchronize(self.device)
        self._persist_runtime_cache()

    def close(self) -> None:
        self._persist_runtime_cache()
        self._runtime_cache = None
        self._runtime_cache_path = None
        self.outputs.clear()
        self._staging = {}
        self._stream = None
        self.context = None
        self.engine = None
        self.runtime = None

    def _execute_on_private_stream(self, inputs: dict[str, torch.Tensor]) -> bool:
        """Execute on the runner-owned stream (graph capture is not allowed on
        the default stream), ordered against the caller's stream both ways."""
        caller = current_stream(self.device)
        self._stream.wait_stream(caller)
        with stream_context(self._stream):
            for name, tensor in inputs.items():
                self._staging[name].copy_(tensor, non_blocking=True)
            executed = self.context.execute_async_v3(self._stream.cuda_stream)
        if executed is False:
            return False
        caller.wait_stream(self._stream)
        return True

    def _fallback_without_graphs(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        logger.warning(
            "TensorRT-RTX CUDA graph execution failed; retrying without graph capture"
        )
        self._graphs_enabled = False
        self._staging = {}
        self.context = self._create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self._cur_shapes = {}
        return self.infer(inputs)

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if set(inputs) != set(self.input_names):
            raise ValueError(
                f"TensorRT inputs must be {self.input_names}, got {list(inputs)}"
            )
        batch_sizes = {
            int(inputs[name].shape[0])
            for name in self.input_names
        }
        if len(batch_sizes) != 1:
            raise ValueError("TensorRT inputs must use the same batch size")
        trim = None
        n = batch_sizes.pop()
        if n <= 0:
            raise ValueError("TensorRT inference requires a non-empty batch")
        orig_inputs = inputs
        # Under graph capture, dynamic engines also run at the single
        # engine-batch shape: one shape means one graph and zero recaptures.
        if not self.dynamic_batch or self._graphs_enabled:
            if n > self._engine_batch:
                raise ValueError(
                    f"input batch {n} exceeds fixed TensorRT batch "
                    f"{self._engine_batch}"
                )
            if n < self._engine_batch:
                inputs = {
                    name: pad_batch_with_last(
                        tensor,
                        batch_size=self._engine_batch,
                    )
                    for name, tensor in inputs.items()
                }
                trim = n
        shapes = {name: tuple(inputs[name].shape) for name in self.input_names}
        if shapes != self._cur_shapes:
            self._bind(shapes)
        if self._graphs_enabled:
            if not self._execute_on_private_stream(inputs):
                return self._fallback_without_graphs(orig_inputs)
        else:
            for name, tensor in inputs.items():
                self.context.set_tensor_address(name, int(tensor.data_ptr()))
            executed = self.context.execute_async_v3(
                torch.cuda.current_stream(self.device).cuda_stream
            )
            if executed is False:
                raise RuntimeError("TensorRT inference execution failed")
        if trim is not None:
            return {name: out[:trim] for name, out in self.outputs.items()}
        return self.outputs

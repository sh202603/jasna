from __future__ import annotations

import torch
from pathlib import Path
import tensorrt as trt
from jasna.trt import _engine_io_names, _trt_dtype_to_torch, get_trt_logger


class TrtRunner:
    def __init__(
        self,
        engine_path: Path,
        input_shapes: dict[str, tuple[int, ...]] | list[tuple[int, ...]],
        device: torch.device,
    ) -> None:
        self.engine_path = engine_path
        self._setup(engine_path.read_bytes(), input_shapes, device, str(engine_path))

    @classmethod
    def from_engine_bytes(
        cls,
        engine_bytes: bytes,
        input_shapes: dict[str, tuple[int, ...]] | list[tuple[int, ...]],
        device: torch.device,
        source: str = "<memory>",
    ) -> "TrtRunner":
        self = cls.__new__(cls)
        self.engine_path = None
        self._setup(engine_bytes, input_shapes, device, source)
        return self

    def _setup(
        self,
        engine_bytes: bytes,
        input_shapes: dict[str, tuple[int, ...]] | list[tuple[int, ...]],
        device: torch.device,
        source: str,
    ) -> None:
        self.device = device

        self.runtime = trt.Runtime(get_trt_logger())
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {source}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self.input_names, self.output_names = _engine_io_names(self.engine)

        if isinstance(input_shapes, list):
            input_shapes = dict(zip(self.input_names, input_shapes))

        self.input_dtypes: dict[str, torch.dtype] = {}
        self.input_shapes: dict[str, tuple[int, ...]] = {}
        for name in self.input_names:
            requested = tuple(int(d) for d in input_shapes[name])
            engine_shape = tuple(int(d) for d in self.engine.get_tensor_shape(name))
            # Static engines only accept their build-time shape; silently
            # executing with a smaller caller buffer reads out of bounds
            # (Xid 31), so bind the engine shape and let callers pad.
            is_static = all(d >= 0 for d in engine_shape)
            shape = engine_shape if is_static and engine_shape != requested else requested
            self.context.set_input_shape(name, shape)
            if tuple(int(d) for d in self.context.get_tensor_shape(name)) != shape:
                raise RuntimeError(
                    f"TensorRT rejected input shape {shape} for '{name}' "
                    f"(engine shape {engine_shape}, requested {requested}): {source}"
                )
            self.input_shapes[name] = shape
            self.input_dtypes[name] = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))

        dev = torch.device(self.device)
        self.outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(int(d) for d in self.context.get_tensor_shape(name))
            dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            t = torch.empty(size=shape, dtype=dtype, device=dev)
            self.outputs[name] = t
            self.context.set_tensor_address(name, int(t.data_ptr()))

    def close(self) -> None:
        self.outputs.clear()
        self.context = None
        self.engine = None
        self.runtime = None

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for name, tensor in inputs.items():
            expected = self.input_shapes.get(name)
            if expected is not None and tuple(tensor.shape) != expected:
                raise ValueError(
                    f"Input '{name}' shape {tuple(tensor.shape)} does not match "
                    f"the bound engine shape {expected}"
                )
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        self.context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream)
        return self.outputs


"""ONNX Runtime runner (CPU EP) as a drop-in replacement for TrtRunner.

Used on platforms where TensorRT engine building is unavailable (e.g. Windows
on ARM: NVIDIA ships no Myelin resource DLLs for win_arm64, so every engine
build fails — NVIDIA/TensorRT#4904). Activated by setting the environment
variable ``JASNA_RFDETR_BACKEND=onnxruntime``.

Implements the same runner contract as ``TrtRunner`` and
``RfDetrTorchRunner``: ``input_names`` / ``input_dtypes`` / ``output_names`` /
``outputs`` / ``infer`` / ``close``.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import torch

log = logging.getLogger(__name__)


def _dtype_from_ort(ort_dtype: np.dtype):
    mapping = {
        np.dtype("float32"): torch.float32,
        np.dtype("float16"): torch.float16,
        np.dtype("int64"): torch.int64,
        np.dtype("int32"): torch.int32,
    }
    return mapping.get(ort_dtype, torch.float32)


class OnnxRuntimeRunner:
    def __init__(self, model_path: str, *, intra_threads: int | None = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if intra_threads:
            opts.intra_op_num_threads = int(intra_threads)
            opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._lock = threading.Lock()
        self.input_names = [i.name for i in self._session.get_inputs()]
        self.input_dtypes = {
            i.name: _dtype_from_ort(i.type) for i in self._session.get_inputs()
        }
        self.output_names = [o.name for o in self._session.get_outputs()]
        # Shape-compatible stand-ins so consumers can read .ndim/.shape/.dtype.
        self.outputs = {
            o.name: torch.empty(tuple(0 for _ in o.shape), dtype=_dtype_from_ort(o.type))
            if any(isinstance(d, str) for d in o.shape)
            else torch.empty(tuple(o.shape), dtype=_dtype_from_ort(o.type))
            for o in self._session.get_outputs()
        }
        # Real per-run shapes are reported lazily via _infer; most consumers
        # only need ndim and the last axis, so stash first-call shapes too.
        self._shape_cache: dict[str, torch.Size] = {}
        log.info(
            "OnnxRuntimeRunner loaded %s (inputs=%s outputs=%s)",
            model_path,
            self.input_names,
            self.output_names,
        )

    def _report_shape(self, name: str, shape: tuple[int, ...]) -> None:
        self._shape_cache[name] = torch.Size(shape)

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = next(iter(inputs.values())).device
        feed: dict[str, np.ndarray] = {}
        for name in self.input_names:
            tensor = inputs[name]
            want = self.input_dtypes.get(name, torch.float32)
            feed[name] = tensor.detach().to("cpu", dtype=want).contiguous().numpy()
        with self._lock:
            out_arrays = self._session.run(self.output_names, feed)
        result: dict[str, torch.Tensor] = {}
        for name, arr in zip(self.output_names, out_arrays):
            self._report_shape(name, tuple(arr.shape))
            result[name] = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        return result

    def close(self) -> None:
        self._session = None

"""Streaming safetensors writer: the tensor plan (names, dtypes, shapes) is fixed up front so the
header can be written first and each tensor appended as it is produced. A 46 GB input converts
without holding the output in RAM."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

DTYPES = {
    torch.bool: "BOOL", torch.uint8: "U8", torch.int8: "I8", torch.int16: "I16", torch.int32: "I32",
    torch.int64: "I64", torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32",
    torch.float64: "F64", torch.float8_e4m3fn: "F8_E4M3", torch.float8_e5m2: "F8_E5M2",
}


def _nbytes(dtype: torch.dtype, shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n * dtype.itemsize


class StreamingSafetensorsWriter:
    """``plan`` is an ordered list of ``(name, dtype, shape)``; ``add`` must follow that order."""

    def __init__(self, path: Path, plan: list[tuple[str, torch.dtype, tuple[int, ...]]], metadata: dict[str, str] | None = None) -> None:
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.plan = list(plan)
        self.index = 0
        header: dict = {"__metadata__": dict(metadata or {})}
        offset = 0
        for name, dtype, shape in self.plan:
            if name in header:
                raise KeyError(f"duplicate tensor {name}")
            size = _nbytes(dtype, tuple(shape))
            header[name] = {"dtype": DTYPES[dtype], "shape": list(shape), "data_offsets": [offset, offset + size]}
            offset += size
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        self.total = offset
        self.out = self.tmp.open("wb")
        self.out.write(struct.pack("<Q", len(encoded)))
        self.out.write(encoded)

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if self.index >= len(self.plan):
            raise IndexError("more tensors than planned")
        pname, pdtype, pshape = self.plan[self.index]
        if name != pname or tensor.dtype != pdtype or tuple(tensor.shape) != tuple(pshape):
            raise ValueError(f"plan mismatch at {self.index}: planned {pname} {pdtype} {tuple(pshape)}, got {name} {tensor.dtype} {tuple(tensor.shape)}")
        tensor = tensor.detach().contiguous().cpu()
        if tensor.numel():
            self.out.write(tensor.view(-1).view(torch.uint8).numpy().tobytes())
        self.index += 1

    def close(self) -> None:
        self.out.close()
        if self.index != len(self.plan):
            raise RuntimeError(f"only {self.index} of {len(self.plan)} planned tensors were written")
        self.tmp.replace(self.path)

"""Pure read()/seek() safetensors reader — no mmap.

The Rust mmap path in every published safetensors wheel SIGSEGVs on Windows
ARM64 as soon as one memory-maps a file larger than 2**35 bytes, which the
stock LTX checkpoints exceed. The header and each tensor are small; sequential
reads never touch the failing mapping. Supports the small slice of the
``safe_open`` API these scripts use: keys(), metadata(), get_tensor(), and
use as a context manager.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

_DTYPE = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
_ITEMSIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}


class IoSafeOpen:
    def __init__(self, path) -> None:
        self.path = Path(path)
        with open(self.path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        # data_offsets in the header count from the first byte after the header,
        # not from the start of the file.
        self._data_start = 8 + header_len
        self._metadata = header.pop("__metadata__", None)
        self._tensors = header

    def keys(self):
        return list(self._tensors)

    def metadata(self):
        return self._metadata

    def get_tensor(self, key: str) -> torch.Tensor:
        info = self._tensors[key]
        dtype = _DTYPE[info["dtype"]]
        start, end = info["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self._data_start + start)
            buf = bytearray(f.read(end - start))
        # Build via a byte tensor so BF16 (no numpy equivalent) works too.
        flat = torch.frombuffer(buf, dtype=torch.uint8)
        tensor = flat.view(dtype).reshape(info["shape"])
        return tensor

    def get_slice(self, key: str):
        """Lazy slice view (shape/dtype only) — mirrors the safe_open API."""
        return _IoSlice(self, key)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _IoSlice:
    """Shape/dtype view over one header entry; no data is read."""

    def __init__(self, owner: "IoSafeOpen", key: str) -> None:
        self._info = owner._tensors[key]

    def get_shape(self):
        return list(self._info["shape"])

    def get_dtype(self):
        return self._info["dtype"]

    def __getitem__(self, index):
        # Full-range reads only; the scripts use shape inspection on slices.
        raise NotImplementedError("IoSafeOpen slices support shape/dtype only")


def io_safe_open(path, framework: str = "pt", device: str = "cpu") -> IoSafeOpen:
    """Drop-in for ``safetensors.safe_open`` on files the mmap reader cannot open.

    ``framework`` and ``device`` are accepted for signature compatibility only;
    tensors are always returned on the CPU.
    """
    return IoSafeOpen(path)

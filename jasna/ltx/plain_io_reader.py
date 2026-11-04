"""Pure read()/seek() safetensors reader — no mmap.

The Rust mmap path in every published ``safetensors`` wheel memory-maps the whole file. On
Windows on ARM64 (Windows 11 on the NVIDIA RTX Spark / GB10) that commits the full file size
against the system's page-file limit: a 12+ GiB LTX checkpoint then fails with
``os error 1455`` (page file too small) at ``safe_open``, or SIGSEGVs later on a page fault —
also in non-main threads, where the fault cannot be reported. Plain ``seek``/``read`` of the
header and each tensor commits only one tensor at a time and works regardless of file size.

The reader mirrors the slice of the ``safe_open`` API that :class:`jasna.ltx.model_files
<._PlaintextHandle>` uses: ``keys()``, ``offset_keys()``, ``metadata()``, ``get_tensor()``,
``get_slice()`` (shape/dtype only) and context-manager use. Tensors are always returned on
the CPU.
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
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


class PlainIoSource:
    """``safe_open``-compatible reader of one safetensors file using only read/seek."""

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

    def keys(self) -> list[str]:
        return list(self._tensors)

    def offset_keys(self) -> list[str]:
        """Keys ordered by their offset in the file, as ``safe_open.offset_keys()`` returns."""
        return [key for key, _ in sorted(self._tensors.items(), key=lambda kv: kv[1]["data_offsets"][0])]

    def metadata(self) -> dict | None:
        return self._metadata

    def get_tensor(self, key: str) -> torch.Tensor:
        info = self._tensors[key]
        dtype = _DTYPE[info["dtype"]]
        start, end = info["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self._data_start + start)
            buf = bytearray(f.read(end - start))
        # Build via a byte tensor so BF16 / float8 (no numpy equivalents) work too.
        flat = torch.frombuffer(buf, dtype=torch.uint8)
        return flat.view(dtype).reshape(info["shape"])

    def get_slice(self, key: str):
        """Lazy slice view (shape/dtype only) — mirrors the safe_open API."""
        return _PlainIoSlice(self, key)

    def close(self) -> None:
        # Nothing held open between reads; present for API symmetry.
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _PlainIoSlice:
    """Shape/dtype view over one header entry; no data is read."""

    def __init__(self, owner: PlainIoSource, key: str) -> None:
        self._info = owner._tensors[key]

    def get_shape(self):
        return list(self._info["shape"])

    def get_dtype(self):
        return self._info["dtype"]

    def __getitem__(self, index):
        raise NotImplementedError("PlainIoSource slices support shape/dtype only")

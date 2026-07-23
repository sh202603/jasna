"""torchcodec-based video reader, mirroring ``NvidiaVideoReader``.

Decodes via torchcodec's CUDA/NVDEC backend straight into GPU ``uint8`` RGB NCHW
tensors — exactly what jasna's detection/restoration models consume. The contract
matches the native reader: a context manager whose ``frames(seek_ts=None)`` yields
``(uint8 [B,3,H,W] CUDA, list[int] PTS)``.

Differences from native (``media/video_decoder.py``), see ``docs/{en,ja}/torchcodec_backend.md``:

- torchcodec always outputs 8-bit RGB; there is no 10-bit dither step (the models
  are 8-bit RGB anyway).
- torchcodec manages its own internal CUDA stream. We ``torch.cuda.synchronize``
  the device before each yield so frames are fully materialized for the caller's
  default-stream model inference (same per-batch cadence as the native reader's
  ``stream.synchronize()``).
- PTS comes as float ``pts_seconds``; we convert to integer ticks in the stream
  ``time_base`` and enforce strict monotonicity (the native reader's ``pkt_data.pts``
  is already strictly increasing, and ``tests/test_video_decoder_seek.py`` asserts it).
- ``__enter__`` decodes frame 0 as a probe. This forces NVDEC initialization (so an
  unsupported codec or a silent CPU fallback surfaces here, where the AUTO factory
  can fall back to native) and confirms the decoded tensor is on CUDA.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterator

if sys.platform == "win32":
    _tc_spec = importlib.util.find_spec("torchcodec")
    if _tc_spec and _tc_spec.origin:
        os.add_dll_directory(str(Path(_tc_spec.origin).parent))
    _cuda_path = os.environ.get("CUDA_PATH")
    if _cuda_path:
        _cuda_bin = os.path.join(_cuda_path, "bin")
        if os.path.isdir(_cuda_bin):
            os.add_dll_directory(_cuda_bin)

import torch
from torchcodec.decoders import VideoDecoder

from jasna.media import VideoMetadata


class TorchcodecUnavailable(RuntimeError):
    """torchcodec could not decode on the GPU (CPU fallback / NVDEC failure).

    Raised from ``__enter__`` so the AUTO factory can fall back to the native
    (GPU-only) reader.
    """


class TorchcodecVideoReader:
    backend = "torchcodec"

    def __init__(self, file: str, batch_size: int, device: torch.device, metadata: VideoMetadata):
        self.file = file
        self.batch_size = int(batch_size)
        self.device = device
        self.metadata = metadata
        self._decoder: VideoDecoder | None = None
        self._num_frames = 0

    def __enter__(self):
        self._decoder = VideoDecoder(
            self.file,
            device=str(self.device),
            dimension_order="NCHW",
            seek_mode="exact",
        )
        num = self._decoder.metadata.num_frames
        self._num_frames = int(num) if num is not None else int(self.metadata.num_frames)

        # Probe frame 0: forces NVDEC init and surfaces a silent CPU fallback.
        if self._num_frames > 0:
            probe = self._decoder.get_frames_in_range(0, 1).data
            if probe.numel() > 0 and probe.device.type != "cuda":
                raise TorchcodecUnavailable(
                    f"torchcodec decoded {self.file!r} to {probe.device} (CPU fallback); "
                    "native GPU decoder required"
                )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._decoder = None

    def frames(self, seek_ts: float | None = None) -> Iterator[tuple[torch.Tensor, list[int]]]:
        assert self._decoder is not None, "TorchcodecVideoReader used outside its context"
        time_base = float(self.metadata.time_base)
        next_index = 0
        if seek_ts is not None:
            # Match the native reader's frame estimate (_estimate_start_frame).
            next_index = max(0, int(seek_ts * self.metadata.video_fps))
        last_pts: int | None = None

        while next_index < self._num_frames:
            stop = min(next_index + self.batch_size, self._num_frames)
            fb = self._decoder.get_frames_in_range(next_index, stop)
            # torchcodec returns NCHW as a non-contiguous view of its internal HWC
            # buffer (stride for C/W is interleaved). Consumers that read raw memory
            # assuming contiguous NCHW — notably the detection TensorRT engine — get
            # garbage from it, silently producing zero detections. Force contiguity.
            data = fb.data.contiguous()
            if data.shape[0] == 0:
                break

            pts_list: list[int] = []
            for t in fb.pts_seconds.tolist():
                p = int(round(t / time_base)) if time_base > 0 else (last_pts + 1 if last_pts is not None else 0)
                if last_pts is not None and p <= last_pts:
                    p = last_pts + 1
                last_pts = p
                pts_list.append(p)

            torch.cuda.synchronize(self.device)
            yield data, pts_list
            next_index = stop

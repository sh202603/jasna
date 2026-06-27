"""torchcodec-based video encoder, mirroring ``NvidiaVideoEncoder``.

Encodes RGB frames on the GPU via torchcodec's incremental multi-stream
``Encoder`` (NVENC), then reuses the native module's remux helper to add audio
and the source colorspace metadata. The contract matches the native encoder: a
context manager plus ``encode(frame, pts)`` (the ``FrameWriter`` surface the
pipeline drives through ``_OfflineFrameWriter``).

Scope (see ``docs/TORCHCODEC_BACKEND_*.md`` and ``torchcodec_encoder_eligibility``):
HEVC or AV1, 8-bit, limited range, no frame-gen, non-streaming, default settings.
10-bit, custom ``--encoder-settings``, frame generation, and streaming stay on the
native encoder.

Notable differences from native (``media/video_encoder.py``):

- GPU encode is always 8-bit ``nv12``; there is no P010 path here.
- torchcodec encodes at a constant ``frame_rate`` (``video_fps_exact``); unlike the
  native path (raw elementary + mkvmerge explicit per-frame timecodes) it does not
  carry arbitrary per-frame PTS. This is correct for CFR sources (jasna's case);
  VFR timing would drift, so VFR is left to the native encoder.
- Color metadata (BT.601/709/2020 tags + the HEVC VUI rewrite) and audio are applied
  by ``remux_with_audio_and_metadata`` — the same helper the native encoder uses —
  so the colorspace result is identical to native.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import queue
import sys
import threading
from pathlib import Path

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
from torchcodec.encoders import Encoder

from jasna.media import VideoMetadata
from jasna.media.lut import GpuLutApplier, parse_cube_file
from jasna.media.video_encoder import remux_with_audio_and_metadata

log = logging.getLogger(__name__)


class TorchcodecEncodeUnsupported(ValueError):
    """The request cannot be encoded by torchcodec (e.g. 10-bit / custom settings)."""


class TorchcodecVideoEncoder:
    backend = "torchcodec"

    def __init__(
        self,
        file: str,
        device: torch.device,
        metadata: VideoMetadata,
        *,
        codec: str,
        encoder_settings: dict[str, object],
        stream_mode: bool = False,
        working_directory: Path | None = None,
        bit_depth: int | None = None,
        lut_path: str | Path | None = None,
        output_fps_multiplier: int = 1,
    ):
        from jasna.media.backend import torchcodec_encoder_eligibility

        eligible, reason = torchcodec_encoder_eligibility(
            codec=codec,
            bit_depth=bit_depth,
            metadata=metadata,
            encoder_settings=encoder_settings,
            output_fps_multiplier=output_fps_multiplier,
            stream_mode=stream_mode,
        )
        if not eligible:
            raise TorchcodecEncodeUnsupported(reason)

        self.metadata = metadata
        self.device = device
        self.output_path = Path(file)
        self.codec = codec.lower()
        self._nvenc_codec = "av1_nvenc" if self.codec == "av1" else "hevc_nvenc"
        self.output_fps_multiplier = max(1, int(output_fps_multiplier))
        self._frame_rate = float(metadata.video_fps_exact) * self.output_fps_multiplier
        self._preset, self._extra_options = self._build_encode_params(dict(encoder_settings))

        self._lut_applier: GpuLutApplier | None = None
        if lut_path:
            lut = parse_cube_file(lut_path)
            self._lut_applier = GpuLutApplier(lut, device)

        temp_dir = Path(working_directory) if working_directory is not None else self.output_path.parent
        if working_directory is not None:
            temp_dir.mkdir(parents=True, exist_ok=True)
        # Intermediate container torchcodec writes to; mkv carries both HEVC and AV1.
        self._temp_path = temp_dir / (self.output_path.stem + ".tcraw.mkv")

        self._encoder: Encoder | None = None
        self._vstream = None

        # Encoding runs on a dedicated worker thread so the synchronous
        # torchcodec ``add_frames`` does not block the BlendEncode thread (see
        # docs/TORCHCODEC_BACKEND_ja.md 付録A). The whole torchcodec ``Encoder``
        # lifecycle (add_video / open_file / add_frames / close) lives on that
        # worker to avoid cross-thread use of its internal CUDA stream.
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._stop_sentinel = object()
        self._worker: threading.Thread | None = None
        self._ready = threading.Event()      # worker signals after open_file (or init failure)
        self._failed = threading.Event()     # worker signals a runtime encode failure
        self._init_error: BaseException | None = None
        self._worker_error: BaseException | None = None

    def _build_encode_params(self, settings: dict[str, object]) -> tuple[str, dict[str, str]]:
        """Build the (preset, extra_options) for ``add_video`` from jasna's tuning.

        Starts from jasna's default NVENC quality settings (the subset torchcodec
        accepts) so torchcodec output ~matches native, then overlays the
        ``--encoder-settings`` (already gated to mappable keys by eligibility).
        rc/tuning_info/b_ref_mode and the PyNvVideoCodec-only knobs are not
        accepted by torchcodec and are simply omitted (such user settings route to
        native instead).
        """
        from jasna.media.backend import NVENC_EXTRA_KEYMAP

        gop = 250 * self.output_fps_multiplier
        extra: dict[str, str] = {
            "cq": "25", "qmin": "17", "qmax": "34", "nonref_p": "1",
            "g": str(gop), "temporal-aq": "1", "rc-lookahead": "32",
            "spatial-aq": "1", "aq-strength": "8",
        }
        if self.codec == "hevc":
            extra["bf"] = "4"  # av1_nvenc has no B-frames
        preset = "p5"

        for key, value in settings.items():
            if key == "preset":
                preset = str(value).lower()
            elif key == "aq":
                level = int(value)
                if level > 0:
                    extra["spatial-aq"], extra["aq-strength"] = "1", str(level)
                else:
                    extra["spatial-aq"] = "0"
                    extra.pop("aq-strength", None)
            elif key in NVENC_EXTRA_KEYMAP:
                extra[NVENC_EXTRA_KEYMAP[key]] = str(value)
        return preset, extra

    def __enter__(self):
        self._worker = threading.Thread(
            target=self._encode_worker, name="TorchcodecEncoderWorker", daemon=True
        )
        self._worker.start()
        self._ready.wait()
        if self._init_error is not None:
            # Propagate on the caller's thread, matching the previous synchronous
            # __enter__ (BlendEncode thread -> error_holder via blend_encode_loop).
            raise self._init_error
        return self

    def _encode_worker(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        try:
            self._encoder = Encoder()
            self._vstream = self._encoder.add_video(
                height=self.metadata.video_height,
                width=self.metadata.video_width,
                frame_rate=self._frame_rate,
                device=str(self.device),
                codec=self._nvenc_codec,
                preset=self._preset,
                extra_options=self._extra_options,
            )
            self._encoder.open_file(str(self._temp_path))
        except BaseException as e:  # surface to __enter__
            self._init_error = e
            self._ready.set()
            return
        self._ready.set()

        try:
            while True:
                item = self._queue.get()
                if item is self._stop_sentinel:
                    break
                frame, ready_event = item
                # Wait (host-side) for the producer's blend kernels to finish:
                # torchcodec manages its own internal CUDA stream, so we cannot
                # inject a stream-wait the way the native encoder does.
                ready_event.synchronize()
                if self._lut_applier is not None:
                    frame = self._lut_applier.apply(frame)
                if frame.ndim == 3:
                    frame = frame.unsqueeze(0)
                self._vstream.add_frames(frame)
                del frame
        except BaseException as e:
            self._worker_error = e
            self._failed.set()
            log.exception("[torchcodec-encoder-worker] crashed")
            # Drain anything still queued so a blocked producer's put() unblocks.
            self._drain_queue()
            return

        self._encoder.close()
        self._encoder = None

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def encode(self, frame: torch.Tensor, pts: int) -> None:
        # pts is unused: torchcodec encodes CFR at frame_rate (eligibility gates VFR
        # out by requiring the native path for anything PTS-sensitive). Frames are
        # added in queue order, so a single FIFO worker preserves frame order.
        if self._failed.is_set():
            raise self._worker_error  # type: ignore[misc]
        ready_event = torch.cuda.Event()
        torch.cuda.current_stream(self.device).record_event(ready_event)
        self._queue.put((frame, ready_event))

    def __exit__(self, exc_type, exc_value, traceback):
        if self._worker is not None:
            self._queue.put(self._stop_sentinel)
            self._worker.join()
            self._worker = None
        if self._worker_error is not None and exc_type is None:
            raise self._worker_error
        try:
            if exc_type is None and self._worker_error is None:
                remux_with_audio_and_metadata(
                    self._temp_path, self.output_path, self.metadata, self.codec
                )
        finally:
            if self._temp_path.exists():
                self._temp_path.unlink()

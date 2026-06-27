"""torchcodec-based video encoder, mirroring ``NvidiaVideoEncoder``.

Encodes RGB frames on the GPU via torchcodec's incremental multi-stream
``Encoder`` (NVENC), then remuxes to add audio and the source colorspace
metadata. The contract matches the native encoder: a context manager plus
``encode(frame, pts)`` (the ``FrameWriter`` surface the pipeline drives through
``_OfflineFrameWriter``).

Scope (see ``docs/TORCHCODEC_BACKEND_*.md`` and ``torchcodec_encoder_eligibility``):
HEVC or AV1, 8-bit sources, limited range, source frame rate, mappable settings,
non-streaming, no smart-render fragments. Everything else stays on the native
encoder. Note that since upstream v0.8.0 the native encoder always outputs
10-bit for HEVC/AV1; torchcodec output is 8-bit ``nv12`` and is therefore only
used when forced via ``--encode-backend torchcodec``.

Notable differences from native (``media/video_encoder.py``):

- GPU encode is always 8-bit ``nv12``; there is no P010 path here.
- torchcodec encodes at a constant ``frame_rate`` (``video_fps_exact``); unlike
  the native path it does not carry arbitrary per-frame PTS. This is correct for
  CFR sources (jasna's case); VFR timing would drift, so VFR is left to the
  native encoder.
- Color metadata (BT.601/709/2020 tags + the HEVC VUI rewrite) and audio are
  applied by the ffmpeg remux in ``__exit__`` using the same H.273 code points
  as the native encoder, so the tagged colorspace matches native output.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import queue
import subprocess
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
from jasna.media.audio_utils import needs_audio_reencode
from jasna.media.lut import GpuLutApplier, parse_cube_file
from jasna.os_utils import resolve_executable, subprocess_no_window_kwargs

logger = logging.getLogger(__name__)
log = logger


class TorchcodecEncodeUnsupported(ValueError):
    """The request cannot be encoded by torchcodec (e.g. 10-bit / custom settings)."""


def _remux_with_audio_and_metadata(
    video_input: Path, output_path: Path, metadata: VideoMetadata, video_codec: str
) -> None:
    """Copy-remux the torchcodec video with source audio and color tags.

    Uses the same ITU-T H.273 code points as the native encoder
    (``video_encoder._COLOR_TAGS`` refined by the probed primaries/transfer), so
    the tagged colorspace matches native output. NVENC bakes a BT.709 VUI into
    HEVC bitstreams regardless of the source, so for HEVC the VUI is rewritten
    in place (no re-encode) to agree with the container tags; the hevc_metadata
    BSF does not apply to AV1 streams.
    """

    import av
    from av.video.reformatter import ColorRange as AvColorRange

    from jasna.media.video_encoder import _COLOR_PRIMARIES, _COLOR_TAGS, _COLOR_TRANSFERS

    matrix, primaries, transfer = _COLOR_TAGS.get(metadata.color_space, (1, 1, 1))
    primaries = _COLOR_PRIMARIES.get(str(metadata.color_primaries).lower(), primaries)
    transfer = _COLOR_TRANSFERS.get(str(metadata.color_transfer).lower(), transfer)
    full_range_flag = 1 if metadata.color_range == AvColorRange.JPEG else 0

    cmd = [
        resolve_executable("ffmpeg"),
        "-y",
        "-i", str(video_input),
        "-i", str(metadata.video_file),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-map_metadata", "1",
        "-c:v", "copy",
    ]
    if str(video_codec).lower() == "hevc":
        cmd += [
            "-bsf:v",
            (
                f"hevc_metadata=colour_primaries={primaries}"
                f":transfer_characteristics={transfer}"
                f":matrix_coefficients={matrix}"
                f":video_full_range_flag={full_range_flag}"
            ),
        ]
    with av.open(str(metadata.video_file)) as container:
        for output_index, stream in enumerate(container.streams.audio):
            name = stream.codec_context.name
            if needs_audio_reencode(name, output_path.suffix):
                cmd += [f"-c:a:{output_index}", "aac", f"-b:a:{output_index}", "256k"]
            else:
                cmd += [f"-c:a:{output_index}", "copy"]
    cmd += [
        "-color_primaries", str(primaries),
        "-color_trc", str(transfer),
        "-colorspace", str(matrix),
        "-color_range", "pc" if full_range_flag else "tv",
    ]
    if output_path.suffix.lower() in {".mp4", ".mov"}:
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(output_path))
    logger.debug("[remux] cmd: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, **subprocess_no_window_kwargs())
    if result.returncode != 0:
        stdout_text = (result.stdout or b"").decode(errors="replace")
        stderr_text = (result.stderr or b"").decode(errors="replace")
        logger.error(
            "ffmpeg failed (exit code %s). stdout:\n%s\nstderr:\n%s",
            result.returncode, stdout_text, stderr_text,
        )
        raise RuntimeError(
            f"ffmpeg failed with code {result.returncode}: {' '.join(cmd)}\n{stderr_text}"
        )


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
        lut_path: str | Path | None = None,
        output_fps=None,
        mux_audio: bool = True,
    ):
        from jasna.media.backend import torchcodec_encoder_eligibility

        eligible, reason = torchcodec_encoder_eligibility(
            codec=codec,
            metadata=metadata,
            encoder_settings=encoder_settings,
            output_fps=output_fps,
            mux_audio=mux_audio,
        )
        if not eligible:
            raise TorchcodecEncodeUnsupported(reason)

        self.metadata = metadata
        self.device = device
        self.output_path = Path(file)
        self.codec = codec.lower()
        self._nvenc_codec = "av1_nvenc" if self.codec == "av1" else "hevc_nvenc"
        self._frame_rate = float(output_fps if output_fps is not None else metadata.video_fps_exact)
        self._preset, self._extra_options = self._build_encode_params(dict(encoder_settings))

        self._lut_applier: GpuLutApplier | None = None
        if lut_path:
            lut = parse_cube_file(lut_path)
            self._lut_applier = GpuLutApplier(lut, device)

        # Intermediate container torchcodec writes to; mkv carries both HEVC and AV1.
        self._temp_path = self.output_path.parent / (self.output_path.stem + ".tcraw.mkv")

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
        rc/tuning_info/b_ref_mode and the other unmapped knobs are not accepted
        by torchcodec and are simply omitted (such user settings route to native
        instead).
        """
        from jasna.media.backend import NVENC_EXTRA_KEYMAP

        extra: dict[str, str] = {
            "cq": "25", "qmin": "17", "qmax": "34", "nonref_p": "1",
            "g": "250", "temporal-aq": "1", "rc-lookahead": "32",
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
                frame, ready_event, apply_lut = item
                # Wait (host-side) for the producer's blend kernels to finish:
                # torchcodec manages its own internal CUDA stream, so we cannot
                # inject a stream-wait the way the native encoder does.
                ready_event.synchronize()
                if apply_lut and self._lut_applier is not None:
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

    def encode(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True) -> None:
        # pts is unused: torchcodec encodes CFR at frame_rate (eligibility gates VFR
        # out by requiring the native path for anything PTS-sensitive). Frames are
        # added in queue order, so a single FIFO worker preserves frame order.
        if self._failed.is_set():
            raise self._worker_error  # type: ignore[misc]
        ready_event = torch.cuda.Event()
        torch.cuda.current_stream(self.device).record_event(ready_event)
        self._queue.put((frame, ready_event, apply_lut))

    def __exit__(self, exc_type, exc_value, traceback):
        if self._worker is not None:
            self._queue.put(self._stop_sentinel)
            self._worker.join()
            self._worker = None
        if self._worker_error is not None and exc_type is None:
            raise self._worker_error
        try:
            if exc_type is None and self._worker_error is None:
                _remux_with_audio_and_metadata(
                    self._temp_path, self.output_path, self.metadata, self.codec
                )
        finally:
            if self._temp_path.exists():
                self._temp_path.unlink()

from __future__ import annotations

import torch
import logging

import PyNvVideoCodec as nvc
from pathlib import Path
from jasna.media import VideoMetadata, Colorspace, UnsupportedColorspaceError
from jasna.media.lut import GpuLutApplier, parse_cube_file
from jasna.media.rgb_to_p010 import chw_rgb_to_surface
from jasna.os_utils import subprocess_no_window_kwargs
from jasna.media.audio_utils import audio_codec_args
import av
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange
import heapq
from collections  import deque
import subprocess
import threading
import queue
av.logging.set_level(logging.ERROR)

from jasna.os_utils import resolve_executable

logger = logging.getLogger(__name__)

def _parse_hevc_nal_units(data: bytes):
    """Parse HEVC NAL units from Annex B bitstream. Returns list of (nal_type, start, end)."""
    nal_units = []
    i = 0
    n = len(data)
    
    while i < n - 3:
        # Find start code (0x000001 or 0x00000001)
        if data[i:i+3] == b'\x00\x00\x01':
            start = i + 3
            sc_len = 3
        elif i < n - 4 and data[i:i+4] == b'\x00\x00\x00\x01':
            start = i + 4
            sc_len = 4
        else:
            i += 1
            continue
        
        # Find next start code
        end = start
        while end < n - 3:
            if data[end:end+3] == b'\x00\x00\x01' or (end < n - 4 and data[end:end+4] == b'\x00\x00\x00\x01'):
                break
            end += 1
        if end >= n - 3:
            end = n
        
        if start < n:
            # HEVC NAL unit type is bits 1-6 of first byte
            nal_type = (data[start] >> 1) & 0x3F
            nal_units.append((nal_type, i, end))
        
        i = end
    
    return nal_units


def _is_hevc_keyframe(data: bytes) -> bool:
    """Check if HEVC bitstream contains an IDR or CRA frame."""
    # HEVC NAL types for keyframes: IDR_W_RADL=19, IDR_N_LP=20, CRA_NUT=21, BLA types=16-18
    keyframe_types = {16, 17, 18, 19, 20, 21}
    for nal_type, _, _ in _parse_hevc_nal_units(data):
        if nal_type in keyframe_types:
            return True
    return False


def _extract_hevc_extradata(data: bytes) -> bytes:
    """Extract VPS, SPS, PPS NAL units for codec extradata."""
    # VPS=32, SPS=33, PPS=34
    param_types = {32, 33, 34}
    extradata_parts = []
    
    for nal_type, start, end in _parse_hevc_nal_units(data):
        if nal_type in param_types:
            # Include the start code
            extradata_parts.append(data[start-4:end] if data[start-4:start] == b'\x00\x00\x00\x01' else b'\x00\x00\x00\x01' + data[start:end])
    
    return b''.join(extradata_parts)


def mux_elementary_to_mkv(raw_path: Path, output_path: Path, pts_list, time_base):
    """Mux a raw elementary video stream (HEVC/AV1) into MKV with explicit timecodes.

    Codec-agnostic: mkvmerge auto-detects the codec from the elementary stream.
    """
    hevc_path = raw_path
    timecodes_path = output_path.with_suffix('.txt')
    with open(timecodes_path, 'w') as f:
        f.write("# timestamp format v4\n")
        for pts in pts_list:
            timestamp_ms = float(pts * time_base * 1000)
            f.write(f"{timestamp_ms:.6f}\n")
    
    cmd = [
        resolve_executable("mkvmerge"),
        "-o",
        str(output_path),
        "--timestamps",
        f"0:{timecodes_path}",
        str(hevc_path),
    ]
    result = subprocess.run(cmd, capture_output=True, **subprocess_no_window_kwargs())
    if result.returncode != 0:
        stdout_text = (result.stdout or b"").decode(errors="replace")
        stderr_text = (result.stderr or b"").decode(errors="replace")
        logger.error("mkvmerge failed (exit code %s). stdout:\n%s\nstderr:\n%s", result.returncode, stdout_text, stderr_text)
        raise RuntimeError(f"mkvmerge failed with code {result.returncode}: {' '.join(cmd)}\n{stderr_text}")
    timecodes_path.unlink()


# Backward-compatible alias (codec-agnostic implementation).
mux_hevc_to_mkv = mux_elementary_to_mkv


def remux_with_audio_and_metadata(video_input: Path, output_path: Path, metadata: VideoMetadata, video_codec: str = "hevc"):
    # Map the YUV matrix family to ffmpeg color tags: (matrix, primaries, transfer).
    # HDR transfer (PQ/HLG) is out of scope; for BT.2020 we tag the SDR-wide
    # transfer bt2020-10 (a valid color_trc) — the matrix name (bt2020nc) is NOT
    # a valid color_trc value, so matrix and transfer must be kept distinct.
    colorspace_map = {
        Colorspace.BT709: ('bt709', 'bt709', 'bt709'),
        Colorspace.BT601: ('smpte170m', 'smpte170m', 'smpte170m'),
        Colorspace.BT2020: ('bt2020nc', 'bt2020', 'bt2020-10'),
    }
    color_range_map = {
        AvColorRange.MPEG: 'tv',
        AvColorRange.JPEG: 'pc',
    }
    # ITU-T H.273 / ISO 23091-2 code points (colour_primaries, transfer, matrix).
    vui_code_map = {
        Colorspace.BT709: (1, 1, 1),
        Colorspace.BT601: (6, 6, 6),    # SMPTE 170M
        Colorspace.BT2020: (9, 14, 9),  # BT.2020 primaries/matrix, BT.2020-10 transfer
    }
    ffmpeg_matrix, ffmpeg_primaries, ffmpeg_trc = colorspace_map.get(metadata.yuv_colorspace, ('bt709', 'bt709', 'bt709'))
    ffmpeg_color_range = color_range_map.get(metadata.color_range, 'tv')
    prim_code, trc_code, mat_code = vui_code_map.get(metadata.yuv_colorspace, (1, 1, 1))
    full_range_flag = 1 if metadata.color_range == AvColorRange.JPEG else 0

    cmd = [
        resolve_executable("ffmpeg"),
        "-y",
        "-i",
        str(video_input),
        "-i",
        metadata.video_file,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map_metadata",
        "1",
        "-c:v",
        "copy",
    ]
    # NVENC bakes a BT.709 description into the HEVC bitstream VUI regardless of
    # the source. Rewrite the VUI in-place (no re-encode) so the bitstream agrees
    # with the container color tags set below. HEVC-only: the hevc_metadata BSF
    # does not apply to AV1 streams.
    if str(video_codec).lower() == "hevc":
        cmd += [
            "-bsf:v",
            (
                f"hevc_metadata=colour_primaries={prim_code}"
                f":transfer_characteristics={trc_code}"
                f":matrix_coefficients={mat_code}"
                f":video_full_range_flag={full_range_flag}"
            ),
        ]
    cmd += [
        "-c:a",
        *audio_codec_args(metadata.video_file, output_path),
        "-color_primaries",
        ffmpeg_primaries,
        "-color_trc",
        ffmpeg_trc,
        "-colorspace",
        ffmpeg_matrix,
        "-color_range",
        ffmpeg_color_range,
    ]
    if output_path.suffix.lower() in {'.mp4', '.mov'}:
        cmd += ['-movflags', '+faststart']
    cmd.append(str(output_path))
    logger.debug("[remux] cmd: %s", ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, **subprocess_no_window_kwargs())
    if result.returncode != 0:
        stdout_text = (result.stdout or b"").decode(errors="replace")
        stderr_text = (result.stderr or b"").decode(errors="replace")
        logger.error("ffmpeg failed (exit code %s). stdout:\n%s\nstderr:\n%s", result.returncode, stdout_text, stderr_text)
        raise RuntimeError(f"ffmpeg failed with code {result.returncode}: {' '.join(cmd)}\n{stderr_text}")


class NvidiaVideoEncoder:
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
        self.metadata = metadata
        self.device = device
        self.file = file
        self.output_path = Path(file)
        self.stream_mode = stream_mode
        self.codec = codec.lower()
        # Frame generation multiplies the output frame rate. Timing itself is
        # PTS-driven (mkvmerge timecodes), but NVENC needs the true output fps
        # for correct GOP/rate-control, and the keyframe interval (gop, in
        # frames) is scaled to keep roughly the same wall-clock spacing.
        self.output_fps_multiplier = max(1, int(output_fps_multiplier))

        self._lut_applier: GpuLutApplier | None = None
        if lut_path:
            lut = parse_cube_file(lut_path)
            self._lut_applier = GpuLutApplier(lut, device)

        # Bit depth defaults to the source: 10-bit sources → P010, otherwise NV12.
        # (The restoration pipeline is internally 8-bit RGB, so 10-bit output only
        # re-packs into a wider container — no information is gained.)
        self.bit_depth = bit_depth if bit_depth is not None else (10 if metadata.is_10bit else 8)
        if self.bit_depth not in (8, 10):
            raise ValueError(f"Unsupported bit depth: {self.bit_depth}")
        self.colorspace = metadata.yuv_colorspace.value
        nvenc_fmt = "P010" if self.bit_depth == 10 else "NV12"

        if self.codec == "hevc":
            profile = "main10" if self.bit_depth == 10 else "main"
        elif self.codec == "av1":
            profile = "main"
        else:
            raise ValueError(f"Unsupported codec: {self.codec}")

        temp_dir = Path(working_directory) if working_directory is not None else self.output_path.parent
        if working_directory is not None:
            temp_dir.mkdir(parents=True, exist_ok=True)
        # AV1 NVENC does not support B-frames; HEVC uses 1 (stream) / 4 (file).
        if self.codec == "av1":
            bf = 0
        else:
            bf = 1 if stream_mode else 4 # 1 or 2?

        #todo for streaming mode enable tuning low latency, disable qpass
        encoder_options = {
            'codec': self.codec,
            'preset': 'P5',
            'tuning_info': 'high_quality',
            'profile': profile,
            'rc': 'vbr',
            "cq": 25,
            "qmin": 17,
            "qmax": 34,
            # 'rc': 'constqp',
            # 'constqp': 21,
            'nonrefp': 1,
            # 'multipass': 'qres', # lower psnr
            'gop': 250 * self.output_fps_multiplier,
            'fps': float(metadata.video_fps_exact) * self.output_fps_multiplier,
            "maxbitrate": 0,
            # "maxbitrate": 153600,
            "vbvinit": 0,
            "vbvbufsize": 0,
            'temporalaq': 1,
            'lookahead': 32,
            'lookahead_level': 1,
            'aq': 8,
            "initqp": 17,
            'bf': bf,
            'tflevel': 0,
            "bref": 0 if (self.codec == "av1" or stream_mode) else 2,
        }

        if encoder_settings:
            encoder_options.update(encoder_settings)

        gpu_id = self.device.index if self.device.index is not None else 0
        self.stream = torch.cuda.Stream(device)
        self.encoder = nvc.CreateEncoder(
            width=metadata.video_width,
            height=metadata.video_height,
            gpu_id=gpu_id,
            cudastream=self.stream.cuda_stream,
            fmt=nvenc_fmt,
            usecpuinputbuffer=False,
            **encoder_options
        )

        self.BUFFER_MAX_SIZE = 8
        self.pts_heap = []
        self.frame_buffer = deque()
        self.pts_set = set()
        self.reordered_pts_queue = deque()

        self._stop_sentinel = object()
        self._encode_queue: queue.Queue = queue.Queue(maxsize=self.BUFFER_MAX_SIZE)
        self._encode_thread = threading.Thread(target=self._encode_worker, name="NvidiaVideoEncoderWorker", daemon=True)
        self._encode_thread.start()

        # The RGB→YUV conversion emits limited (MPEG) range only; full-range
        # (JPEG) output is not supported. BT.601/709/2020 matrices are all handled.
        if metadata.color_range == AvColorRange.JPEG:
            raise UnsupportedColorspaceError(
                f"Unsupported color range for encoding: {metadata.color_range} (only limited/MPEG range supported)"
            )

        self.temp_video_path = temp_dir / (self.output_path.stem + '_temp_video' + self.output_path.suffix)

        if self.stream_mode:
            dst_file = av.open(str(self.temp_video_path), 'w')
            out_stream = dst_file.add_stream(self.codec, rate=metadata.video_fps_exact)
            out_stream.width = metadata.video_width
            out_stream.height = metadata.video_height
            out_stream.time_base = metadata.time_base
            out_stream.color_range = metadata.color_range
            out_stream.colorspace = metadata.color_space
            out_stream.codec_context.width = metadata.video_width
            out_stream.codec_context.height = metadata.video_height
            out_stream.codec_context.time_base = metadata.time_base
            out_stream.codec_context.color_range = metadata.color_range
            out_stream.codec_context.colorspace = metadata.color_space
            out_stream.options.update({'x265-params': 'log_level=error'})
            self.dst_file = dst_file
            self.out_stream = out_stream
            self.extradata_set = False
        else:
            raw_ext = '.obu' if self.codec == 'av1' else '.hevc'
            self.hevc_path = temp_dir / (self.output_path.stem + raw_ext)
            self.raw_hevc = open(self.hevc_path, "wb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        while self.frame_buffer:
            self._process_buffer(flush_all=True)

        self._encode_queue.join()
        self._encode_queue.put(self._stop_sentinel)
        self._encode_thread.join()

        while True:
            with torch.cuda.stream(self.stream):
                bitstream = self.encoder.EndEncode()
            if len(bitstream) == 0:
                break
            data = bytearray(bitstream)
            if self.stream_mode:
                pts = self.reordered_pts_queue.popleft()
                self._mux_packet_pyav(data, pts)
            else:
                self.raw_hevc.write(data)

        if self.stream_mode:
            self.dst_file.close()
            remux_with_audio_and_metadata(self.temp_video_path, self.output_path, self.metadata, self.codec)
            self.temp_video_path.unlink()
        else:
            self.raw_hevc.close()
            mux_hevc_to_mkv(self.hevc_path, self.temp_video_path, self.reordered_pts_queue, self.metadata.time_base)
            self.hevc_path.unlink()
            remux_with_audio_and_metadata(self.temp_video_path, self.output_path, self.metadata, self.codec)
            self.temp_video_path.unlink()

        del self.encoder

    def _encode_worker(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        while True:
            item = self._encode_queue.get()
            try:
                if item is self._stop_sentinel:
                    return
                frame, pts, ready_event = item
                self._handle_encode_item(frame, pts, ready_event)
            except Exception:
                logger.exception("[encoder-worker] crashed")
                raise
            finally:
                self._encode_queue.task_done()

    def _build_encode_item(self, frame: torch.Tensor, pts: int) -> tuple[torch.Tensor, int, torch.cuda.Event]:
        producer_stream = torch.cuda.current_stream(self.device)
        ready_event = torch.cuda.Event()
        producer_stream.record_event(ready_event)
        return frame, pts, ready_event

    def _handle_encode_item(self, frame: torch.Tensor, pts: int, ready_event: torch.cuda.Event) -> None:
        self.stream.wait_event(ready_event)
        frame.record_stream(self.stream)
        self._encode_frame(frame, pts)

    def _mux_packet_pyav(self, data: bytearray, pts: int):
        data_bytes = bytes(data)
        
        if not self.extradata_set:
            extradata = _extract_hevc_extradata(data_bytes)
            if extradata:
                self.out_stream.codec_context.extradata = extradata
                self.extradata_set = True
        
        pkt = av.packet.Packet(data_bytes)
        pkt.stream = self.out_stream
        pkt.time_base = self.out_stream.time_base
        pkt.pts = pts
        
        if _is_hevc_keyframe(data_bytes):
            pkt.is_keyframe = True
        
        self.dst_file.mux(pkt)

    def _process_buffer(self, flush_all=False):
        if len(self.frame_buffer) > (self.BUFFER_MAX_SIZE // 2) or (flush_all and self.frame_buffer):
            frame_to_encode = self.frame_buffer.popleft()
            pts_to_assign = heapq.heappop(self.pts_heap)
            self.pts_set.remove(pts_to_assign)
            self._encode_queue.put(self._build_encode_item(frame_to_encode, pts_to_assign))

    def _encode_frame(self, frame: torch.Tensor, pts: int):
        self.reordered_pts_queue.append(pts)

        with torch.cuda.stream(self.stream):
            if self._lut_applier is not None:
                frame = self._lut_applier.apply(frame)
            surface = chw_rgb_to_surface(frame, self.colorspace, self.bit_depth)
            bitstream = self.encoder.Encode(surface)

        if len(bitstream) > 0:
            data = bytearray(bitstream)
            if self.stream_mode:
                pts = self.reordered_pts_queue.popleft()
                self._mux_packet_pyav(data, pts)
            else:
                self.raw_hevc.write(data)

    def encode(self, frame: torch.Tensor, pts: int):
        while pts in self.pts_set:
            pts += 1
        heapq.heappush(self.pts_heap, pts)
        self.frame_buffer.append(frame)
        self.pts_set.add(pts)
        self._process_buffer()

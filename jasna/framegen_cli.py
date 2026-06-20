"""Standalone frame-generation (RIFE frame interpolation) CLI: ``jasna-framegen``.

Pass 2 of a two-pass workflow. Pass 1 (the official jasna binary) removes mosaics
and optionally upscales, producing a near-lossless intermediate video. This CLI
then applies *only* RIFE frame-rate up-conversion (2x/4x) to that already-restored
video: it runs no mosaic detection and no BasicVSR++ restoration.

Why a separate entry point instead of the existing ``--frame-gen`` flag: the main
pipeline always runs detection + restoration, so feeding it an already-restored
video wastes the whole restore pass and risks re-mangling clean frames on a false
detection. This CLI is a thin driver over jasna's protection-free media + framegen
modules (NVDEC decode, NVENC encode + mkvmerge/ffmpeg mux, ``FrameGenWriter``, the
RIFE backend). It imports nothing from ``jasna.pipeline`` or ``jasna.protection``.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

from jasna import startup_timing  # noqa: F401  captures PROCESS_START near process start

# The ``jasna-framegen`` console_script bypasses jasna/__main__.py, so the same
# startup that __main__.py performs must run here, before any torch / native
# extension import: CUDA env, Windows DLL search paths, multiprocessing guards,
# and dev-mode sys.path sanitization.
if sys.platform == "win32":
    os.environ.setdefault("OMP_WAIT_POLICY", "passive")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from jasna.packaging.windows_dll_paths import configure_windows_dll_search_paths

configure_windows_dll_search_paths()

_JASNA_MAIN_PID = os.environ.get("JASNA_MAIN_PID")
if _JASNA_MAIN_PID and str(os.getpid()) != _JASNA_MAIN_PID:
    if len(sys.argv) < 2 or sys.argv[1] != "--multiprocessing-fork":
        sys.exit(0)
if multiprocessing.parent_process() is not None:
    sys.exit(0)
os.environ["JASNA_MAIN_PID"] = str(os.getpid())

from jasna._frozen import is_frozen
from jasna.bootstrap import sanitize_sys_path_for_local_dev

if not is_frozen():
    sanitize_sys_path_for_local_dev(Path(__file__).resolve().parent)

import argparse
import logging

from jasna.os_utils import (
    check_ascii_install_path,
    check_gpu_driver_version,
    check_nvidia_gpu,
    check_required_executables,
    check_windows_nvidia_sysmem_fallback_policy,
)

log = logging.getLogger("jasna.framegen_cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jasna-framegen",
        description=(
            "Pass 2: RIFE frame-rate up-conversion (2x/4x) on an already-restored video. "
            "No mosaic detection, no restoration. Reuses jasna's NVDEC/NVENC + mkvmerge "
            "pipeline; audio and color metadata are carried over from the input."
        ),
    )
    parser.add_argument("--input", required=True, type=str, help="Path to the input (already-restored) video.")
    parser.add_argument("--output", required=True, type=str, help="Path to the output video (.mkv recommended).")
    parser.add_argument(
        "--factor",
        type=str,
        default="2x",
        choices=["2x", "4x"],
        help="Frame-rate up-conversion factor (default: %(default)s).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="rife",
        choices=["rife", "rtx"],
        help='Frame-gen backend: "rife" (neural, works now) or "rtx" (NVIDIA RTX Video '
             "Frame Generation, pending an nvidia-vfx release that ships the effect) "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help='Optional path to RIFE weights. If not set, uses "<model_weights>/rife.pth".',
    )
    parser.add_argument(
        "--fp16",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run the frame-gen model in FP16 (auto-falls back to FP32 if unsupported) (default: %(default)s).",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (default: %(default)s).")
    parser.add_argument(
        "--codec",
        type=str,
        default="hevc",
        choices=["hevc", "av1"],
        help="Output video codec (default: %(default)s).",
    )
    parser.add_argument(
        "--bit-depth",
        type=str,
        default="auto",
        choices=["auto", "8", "10"],
        help='Output bit depth. "auto" matches the source (default: %(default)s).',
    )
    parser.add_argument(
        "--encoder-settings",
        type=str,
        default="",
        help="Optional NVENC overrides (key=value,... or JSON), same format as jasna. "
             "Empty inherits jasna's high-quality defaults (cq=25, preset P5, ...).",
    )
    parser.add_argument(
        "--working-directory",
        type=str,
        default="",
        help="Directory for temporary encode files (raw elementary stream + timecodes). "
             "Defaults to the output directory.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Decode batch size (default: %(default)s).")
    parser.add_argument(
        "--log-level",
        type=str,
        default="error",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: %(default)s).",
    )
    parser.add_argument(
        "--disable-ffmpeg-check",
        action="store_true",
        help="Skip checking for ffmpeg/ffprobe in PATH and their version.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar.")
    return parser


class _EncoderWriter:
    """Minimal ``FrameWriter`` over ``NvidiaVideoEncoder``.

    Lazily enters the encoder context on the first frame, encodes each frame, and
    exits on ``close()`` (the exit performs raw-stream -> mkvmerge timecodes ->
    ffmpeg remux). This mirrors ``pipeline._OfflineFrameWriter`` minus the
    VramOffloader heartbeat, so the CLI stays decoupled from ``jasna.pipeline``
    (which would pull in the restorer/detection/tracking stack).

    The lazy-enter guard matters for empty input: if no frame is ever written the
    encoder context is never entered, so ``close()`` will not try to mux an empty
    stream.
    """

    def __init__(self, encoder_ctx) -> None:
        self._encoder_ctx = encoder_ctx
        self._entered = False

    def write(self, frame, pts: int) -> None:
        if not self._entered:
            self._encoder_ctx.__enter__()
            self._entered = True
        self._encoder_ctx.encode(frame, int(pts))

    def after_write(self, frames_written: int) -> None:
        pass

    def close(self) -> None:
        if self._entered:
            self._encoder_ctx.__exit__(None, None, None)
            self._entered = False


def run_framegen(reader, encoder_ctx, generator, multiplier, *, progress=None) -> int:
    """Decode every frame, push it through ``FrameGenWriter``, return the source-frame count.

    ``reader`` is an already-entered ``NvidiaVideoReader``. The ``generator`` is
    *borrowed*: this function never closes it (``FrameGenWriter.close()`` does not
    either) — the caller owns its lifetime. The encoder is closed via the writer.
    """
    from jasna.framegen import FrameGenWriter

    writer = FrameGenWriter(_EncoderWriter(encoder_ctx), generator, multiplier)

    source_frames = 0
    for batch, pts_list in reader.frames():
        # The decoder allocates a fresh batch tensor per iteration, and
        # FrameGenWriter holds the last frame as a view, which keeps that batch's
        # storage alive until the next frame overwrites the view. So no clone is
        # needed here (and the encoder's record_stream covers cross-stream safety).
        # Do NOT "optimize" the decoder to reuse one buffer across batches, or the
        # held previous frame would be corrupted.
        for i in range(len(pts_list)):
            writer.write(batch[i], int(pts_list[i]))
            source_frames += 1
            if progress is not None:
                progress.update(1)
    writer.close()
    return source_frames


def main() -> None:
    multiprocessing.freeze_support()
    parser = build_parser()
    args = parser.parse_args()

    path_ok, path_info = check_ascii_install_path()
    if not path_ok:
        print("Error: Jasna must be installed in a path with ASCII characters only.")
        print(f"Current path: {path_info}")
        sys.exit(1)

    check_required_executables(disable_ffmpeg_check=args.disable_ffmpeg_check)

    gpu_ok, gpu_result = check_nvidia_gpu()
    if not gpu_ok:
        if gpu_result == "no_cuda":
            print("Error: No CUDA device. An NVIDIA GPU with compute capability 7.5+ is required.")
        else:
            _, major, minor = gpu_result
            print(f"Error: Compute capability 7.5+ required (GPU: {major}.{minor}).")
        sys.exit(1)

    driver_ok, driver_info = check_gpu_driver_version()
    if not driver_ok:
        print(f"Error: GPU driver version check failed: {driver_info}")
        print("Please update your NVIDIA driver to version 580 or newer.")
        sys.exit(1)

    if sys.platform == "win32":
        sysmem_ok, sysmem_info = check_windows_nvidia_sysmem_fallback_policy()
        if not sysmem_ok:
            print(f"Warning: CUDA Sysmem Fallback Policy: {sysmem_info}")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from jasna._suppress_noise import install as _install_noise_filters
    _install_noise_filters()
    import torch

    # Deferred heavy imports. Crucially NOT jasna.pipeline (keeps detection /
    # restoration out of this tool's import surface).
    from jasna.framegen import MULTIPLIER_CHOICES, build_frame_generator
    from jasna.media import get_video_meta_data, parse_encoder_settings, validate_encoder_settings
    from jasna.media.video_decoder import NvidiaVideoReader
    from jasna.media.video_encoder import NvidiaVideoEncoder

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    multiplier = MULTIPLIER_CHOICES[str(args.factor).lower()]
    if multiplier <= 1:
        parser.error("--factor must be 2x or 4x")

    codec = str(args.codec).lower()
    bit_depth = None if str(args.bit_depth).lower() == "auto" else int(args.bit_depth)
    encoder_settings = validate_encoder_settings(parse_encoder_settings(str(args.encoder_settings)))
    working_directory = Path(args.working_directory) if args.working_directory else None
    device = torch.device(str(args.device))
    fp16 = bool(args.fp16)
    fg_model_path = str(args.model_path).strip() or None
    batch_size = max(1, int(args.batch_size))

    metadata = get_video_meta_data(str(input_path))

    log.info(
        "frame-gen: %s -> %s | backend=%s factor=%dx codec=%s bit_depth=%s device=%s",
        input_path, output_path, args.backend, multiplier, codec, args.bit_depth, device,
    )

    source_frames = 0
    with torch.cuda.device(device):
        # The generator is built (and closed) here — the caller owns it; the writer
        # only borrows it. build_frame_generator may raise for an unavailable
        # backend (e.g. "rtx"); that propagates before the try below, so generator
        # is only closed when it was actually created.
        generator = build_frame_generator(
            args.backend, device=device, model_path=fg_model_path, fp16=fp16,
        )
        try:
            encoder_ctx = NvidiaVideoEncoder(
                str(output_path),
                device=device,
                metadata=metadata,
                codec=codec,
                encoder_settings=encoder_settings,
                stream_mode=False,
                working_directory=working_directory,
                bit_depth=bit_depth,
                lut_path=None,
                output_fps_multiplier=multiplier,
            )
            progress = None
            if not args.no_progress:
                from tqdm import tqdm
                total = metadata.num_frames if metadata.num_frames else None
                progress = tqdm(total=total, unit="frame", desc="frame-gen")
            try:
                with (
                    NvidiaVideoReader(
                        str(input_path), batch_size=batch_size, device=device, metadata=metadata,
                    ) as reader,
                    torch.inference_mode(),
                ):
                    source_frames = run_framegen(
                        reader, encoder_ctx, generator, multiplier, progress=progress,
                    )
            finally:
                if progress is not None:
                    progress.close()
        finally:
            generator.close()

    if source_frames == 0:
        print("Error: no video frames were decoded from the input.")
        sys.exit(1)

    log.info(
        "frame-gen done: %d source frames -> %d output frames",
        source_frames, (source_frames - 1) * multiplier + 1,
    )


if __name__ == "__main__":
    main()

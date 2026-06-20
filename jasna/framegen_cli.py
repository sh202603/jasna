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
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to the input (already-restored) video, or a folder of videos.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path to the output video (.mkv recommended), or an output folder when --input is a folder.",
    )
    parser.add_argument(
        "--output-pattern",
        type=str,
        default=None,
        help="Filename template for folder input. Use {original} for the input stem. "
             "Default: {original}_out with each input's extension. Folder mode only.",
    )
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


def _path_collision_key(path: Path) -> str:
    # Same normalization jasna/main.py uses for collision detection: case-fold on
    # Windows (case-insensitive filesystem), exact otherwise.
    absolute = path.resolve(strict=False)
    key = str(absolute)
    return key.casefold() if sys.platform == "win32" else key


def _plan_folder_jobs(input_dir: Path, output_dir: Path, output_pattern: str | None):
    """Plan ``(input_video, output_path)`` jobs for a folder batch.

    Returns ``(jobs, skipped_images)``. Videos only — frame generation is
    video-only, so images in the folder are reported separately (skipped by the
    caller). Mirrors the folder logic in ``jasna/main.py``: reuses
    ``classify_folder`` / ``folder_output_path`` for discovery and naming, and
    raises ``ValueError`` on no videos, a non-folder ``--output``, or
    ``--output-pattern`` collisions (two inputs -> one output, or an output that
    would overwrite an input).
    """
    from jasna.media.media_files import classify_folder, folder_output_path, is_media

    images, videos = classify_folder(input_dir)
    if not videos:
        if images:
            raise ValueError(
                f"No video files found in folder (only {len(images)} image(s)); "
                "frame generation is video-only."
            )
        raise ValueError(f"No video files found in folder: {input_dir}")

    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(
            f"--output must be a folder when --input is a folder (got existing file: {output_dir})"
        )
    if not output_dir.exists() and is_media(output_dir):
        raise ValueError(
            f"--output must be a folder when --input is a folder; got a media filename: {output_dir}"
        )

    jobs: list[tuple[Path, Path]] = []
    planned: dict[str, tuple[Path, Path]] = {}
    input_keys = {_path_collision_key(p) for p in videos}
    for vid in videos:
        out_path = folder_output_path(output_dir, vid, output_pattern)
        key = _path_collision_key(out_path)
        if key in planned:
            other_in, other_out = planned[key]
            raise ValueError(
                "--output-pattern maps multiple inputs to the same output: "
                f"{other_in.name} and {vid.name} -> {other_out}"
            )
        if key in input_keys:
            raise ValueError(f"--output-pattern would overwrite an input file: {out_path}")
        planned[key] = (vid, out_path)
        jobs.append((vid, out_path))
    return jobs, images


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


def _process_video(
    input_path: Path,
    output_path: Path,
    *,
    generator,
    multiplier: int,
    codec: str,
    encoder_settings: dict,
    bit_depth: int | None,
    working_directory: Path | None,
    device,
    batch_size: int,
    show_progress: bool,
) -> int:
    """Frame-generate a single video; return its source-frame count.

    The ``generator`` is borrowed (built/closed once by the caller so a folder
    batch reuses one model across all videos). Everything else — metadata,
    encoder, reader — is created fresh per video. The encoder is closed by
    ``run_framegen`` (via the writer) which performs the mkvmerge/ffmpeg mux.
    """
    import torch

    from jasna.media import get_video_meta_data
    from jasna.media.video_decoder import NvidiaVideoReader
    from jasna.media.video_encoder import NvidiaVideoEncoder

    metadata = get_video_meta_data(str(input_path))
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
    if show_progress:
        from tqdm import tqdm
        total = metadata.num_frames if metadata.num_frames else None
        progress = tqdm(total=total, unit="frame", desc=input_path.name)
    try:
        with (
            NvidiaVideoReader(
                str(input_path), batch_size=batch_size, device=device, metadata=metadata,
            ) as reader,
            torch.inference_mode(),
        ):
            return run_framegen(reader, encoder_ctx, generator, multiplier, progress=progress)
    finally:
        if progress is not None:
            progress.close()


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
    from jasna.media import UnsupportedColorspaceError, parse_encoder_settings, validate_encoder_settings

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

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

    # Single file vs folder batch. Folder mode mirrors jasna/main.py: discover
    # videos, apply the naming pattern, and run them through one shared generator.
    input_is_dir = input_path.is_dir()
    if input_is_dir:
        output_dir = Path(args.output)
        try:
            jobs, skipped_images = _plan_folder_jobs(input_path, output_dir, args.output_pattern)
        except ValueError as e:
            parser.error(str(e))
        if skipped_images:
            print(f"Skipping {len(skipped_images)} image(s): frame generation is video-only.")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        jobs = [(input_path, output_path)]

    log.info(
        "frame-gen: %d job(s) | backend=%s factor=%dx codec=%s bit_depth=%s device=%s",
        len(jobs), args.backend, multiplier, codec, args.bit_depth, device,
    )

    total = len(jobs)
    processed = 0
    failed = 0
    with torch.cuda.device(device):
        # Build the generator ONCE and reuse it across every job (a folder batch
        # shares one model). It is borrowed by each FrameGenWriter and closed here
        # exactly once. build_frame_generator may raise for an unavailable backend
        # (e.g. "rtx"); that propagates before the try below, so the generator is
        # only closed when it was actually created.
        generator = build_frame_generator(
            args.backend, device=device, model_path=fg_model_path, fp16=fp16,
        )
        try:
            for i, (vid, out_path) in enumerate(jobs, start=1):
                if input_is_dir:
                    print(f"[{i}/{total}] Processing {vid.name} -> {out_path.name}")
                try:
                    n = _process_video(
                        vid, out_path,
                        generator=generator, multiplier=multiplier, codec=codec,
                        encoder_settings=encoder_settings, bit_depth=bit_depth,
                        working_directory=working_directory, device=device,
                        batch_size=batch_size, show_progress=not args.no_progress,
                    )
                except UnsupportedColorspaceError as e:
                    # Parity with jasna/main.py: in a folder batch, skip the bad
                    # file and keep going; for a single file, fail.
                    failed += 1
                    print(f"Error processing {vid.name}: {e}")
                    if not input_is_dir:
                        sys.exit(1)
                    continue
                if n == 0:
                    failed += 1
                    print(f"Error: no video frames were decoded from {vid.name}.")
                    if not input_is_dir:
                        sys.exit(1)
                    continue
                processed += 1
                log.info(
                    "frame-gen done: %s | %d source frames -> %d output frames",
                    out_path.name, n, (n - 1) * multiplier + 1,
                )
        finally:
            generator.close()

    if input_is_dir:
        print(f"Done: {processed} processed, {failed} failed/skipped (of {total}).")


if __name__ == "__main__":
    main()

"""Standalone post-processing CLI: ``jasna-postprocess`` (before 0.12.3+modi: ``jasna-framegen``).

Pass 2 of a two-pass workflow. Pass 1 (jasna) removes the mosaics and writes a near-lossless
intermediate video. This CLI then applies only whole-frame post-processing to that finished
video: upscaling (``--upscale``: the whole-video upscalers of v0.12.3, and SwiftVR and RelayVSR on
whole frames) and frame-rate up-conversion (``--frame-generation`` 2x/4x, RIFE or RTX Video Frame
Generation). It runs no mosaic detection and no restoration.

Why a separate entry point instead of the main CLI's ``--upscale`` / ``--frame-generation``: the
main pipeline always runs detection + restoration, so feeding it an already-restored video
wastes the whole restore pass and risks re-mangling clean frames on a false detection, and it
refuses ``--upscale`` next to several restorers. This CLI is a thin driver over jasna's
protection-free media, upscale and framegen modules (NVDEC decode, NVENC encode + mkvmerge/ffmpeg
mux, ``FrameUpscaleWriter`` / ``ClipUpscaleWriter`` / ``StreamUpscaleWriter``,
``FrameGenerationWriter``). It imports nothing from ``jasna.pipeline`` or ``jasna.protection``.

The writers wrap each other in the main pipeline's order: frames are upscaled first, so the
generator works on upscaled frames.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

from jasna import startup_timing  # noqa: F401  captures PROCESS_START near process start

# The ``jasna-postprocess`` console_script bypasses jasna/__main__.py, so the same
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
from dataclasses import dataclass
from typing import get_args

from jasna.system_requirements import (
    check_ascii_install_path,
    check_gpu_driver_version,
    check_required_executables,
    check_supported_gpu,
    check_windows_nvidia_sysmem_fallback_policy,
    gpu_check_error,
)
from jasna.session_config import (
    DEFAULT_UPSCALE_HEIGHT,
    DEFAULT_UPSCALE_STRENGTH,
    UPSCALE_DLSS5_DEFAULT_INTENSITY,
    UPSCALE_DLSS5_DEFAULT_PRESET,
    UPSCALE_DLSS5_DEFAULT_STYLE,
    UPSCALE_HEIGHTS,
    UPSCALE_RTX_DEFAULT_DEBLUR,
    UPSCALE_RTX_DEFAULT_DENOISE,
    UPSCALE_RTX_DEFAULT_QUALITY,
    Dlss5PresetName,
    Dlss5StyleName,
    RtxLevelName,
    RtxQualityName,
    UpscaleModelName,
)

log = logging.getLogger("jasna.postprocess_cli")

# Kept as literals (as in jasna.upscale.clip_writer / swiftvr_frame_worker / relayvsr_frame_worker)
# so that building the parser imports neither torch nor the worker scripts.
_SWIFTVR_DEFAULT_WINDOW = 89
_SWIFTVR_WINDOW_OVERLAP = 16
_SWIFTVR_DEFAULT_TILE_MEGAPIXELS = 1.25
_RELAYVSR_DEFAULT_WINDOW_SIZE = 16
# The upscalers that run in a worker of their own venv, not through session_factory.
_EXTERNAL_UPSCALERS = ("swiftvr", "relayvsr")


@dataclass(frozen=True)
class UpscaleSettings:
    """What ``session_factory``'s whole-frame upscale functions (``frame_upscale_size``,
    ``ensure_frame_upscale_engines``, ``build_frame_upscaler``) read from a ``SessionConfig``.
    They take this in its place, so the CLI does not have to make up a restoration config."""

    device: str
    fp16: bool
    upscale_model: str
    upscale_height: int
    upscale_strength: float
    upscale_rtx_quality: str
    upscale_rtx_denoise: str
    upscale_rtx_deblur: str
    upscale_dlss5_preset: str
    upscale_dlss5_style: str
    upscale_dlss5_intensity: float


class NothingToDo(Exception):
    """A video none of the requested steps changes (already tall enough, no frame generation)."""


def build_parser() -> argparse.ArgumentParser:
    from jasna.cli_help import CLI_HELP

    parser = argparse.ArgumentParser(
        prog="jasna-postprocess",
        description=(
            "Pass 2: whole-frame post-processing of an already-restored video: upscaling (--upscale) "
            "and/or frame-rate up-conversion (--frame-generation). No mosaic detection, no restoration. "
            "Reuses jasna's NVDEC/NVENC + mkvmerge pipeline; audio and color metadata are carried over "
            "from the input."
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
        "--fp16",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run the models in FP16 where they support it (RIFE falls back to FP32 if unsupported) "
             "(default: %(default)s).",
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
        "--encoder-settings",
        type=str,
        default="",
        help="Optional NVENC overrides (key=value,... or JSON), same format as jasna. "
             "Empty inherits jasna's high-quality defaults (cq=25, preset P5, ...).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Decode batch size (default: %(default)s).")
    parser.add_argument(
        "--video-backend",
        type=str,
        default="native",
        choices=["native", "auto", "torchcodec"],
        help='Decode/encode backend: "native" (vali + PyNvVideoCodec, default), "auto" (torchcodec where '
             'available/eligible, else native), or "torchcodec" (force torchcodec). Note: frame generation '
             'always encodes natively (CFR output is required), so torchcodec then only affects decoding.',
    )
    parser.add_argument(
        "--video-decode-backend", type=str, default="inherit",
        choices=["inherit", "native", "auto", "torchcodec"],
        help='Override the decode backend only (default "inherit" = --video-backend).',
    )
    parser.add_argument(
        "--video-encode-backend", type=str, default="inherit",
        choices=["inherit", "native", "auto", "torchcodec"],
        help='Override the encode backend only (default "inherit" = --video-backend).',
    )
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

    upscale = parser.add_argument_group("Upscale whole video")
    upscale.add_argument(
        "--upscale",
        type=str,
        default="none",
        choices=[*get_args(UpscaleModelName), *_EXTERNAL_UPSCALERS],
        help="Upscale every frame to --upscale-height. span, realplksr, rtx-super-res and dlss5 are the "
             "upscalers of jasna's own --upscale (span and realplksr need their weights in model_weights/ "
             "and compile a TensorRT engine per input size). swiftvr runs the SwiftVR diffusion restorer "
             "on whole frames, in tiles and overlapping windows: far slower than the others (a few frames "
             "per second for 1080p to 2160p) and needs --swiftvr-repo. relayvsr runs the RelayVSR streaming "
             "super-resolution model on whole frames, 2x or 4x, with its state kept over the whole video; "
             "it needs --relayvsr-repo. Neither of the two can be combined with --frame-generation. "
             "(default: %(default)s)",
    )
    upscale.add_argument(
        "--upscale-height", type=int, default=DEFAULT_UPSCALE_HEIGHT, choices=list(UPSCALE_HEIGHTS),
        help=CLI_HELP["upscale_height"],
    )
    upscale.add_argument(
        "--upscale-strength", type=float, default=DEFAULT_UPSCALE_STRENGTH, help=CLI_HELP["upscale_strength"],
    )
    upscale.add_argument(
        "--upscale-rtx-quality", type=str, default=UPSCALE_RTX_DEFAULT_QUALITY,
        choices=list(get_args(RtxQualityName)), help=CLI_HELP["upscale_rtx_quality"],
    )
    upscale.add_argument(
        "--upscale-rtx-denoise", type=str, default=UPSCALE_RTX_DEFAULT_DENOISE,
        choices=list(get_args(RtxLevelName)), help=CLI_HELP["upscale_rtx_denoise"],
    )
    upscale.add_argument(
        "--upscale-rtx-deblur", type=str, default=UPSCALE_RTX_DEFAULT_DEBLUR,
        choices=list(get_args(RtxLevelName)), help=CLI_HELP["upscale_rtx_deblur"],
    )
    upscale.add_argument(
        "--upscale-dlss5-preset", type=str, default=UPSCALE_DLSS5_DEFAULT_PRESET,
        choices=list(get_args(Dlss5PresetName)), help=CLI_HELP["upscale_dlss5_preset"],
    )
    upscale.add_argument(
        "--upscale-dlss5-style", type=str, default=UPSCALE_DLSS5_DEFAULT_STYLE,
        choices=list(get_args(Dlss5StyleName)), help=CLI_HELP["upscale_dlss5_style"],
    )
    upscale.add_argument(
        "--upscale-dlss5-intensity", type=float, default=UPSCALE_DLSS5_DEFAULT_INTENSITY,
        help=CLI_HELP["upscale_dlss5_intensity"],
    )

    swiftvr = parser.add_argument_group("SwiftVR (--upscale swiftvr)")
    swiftvr.add_argument(
        "--swiftvr-repo", type=str, default="",
        help="Path to the SwiftVR checkout (fork sh202603/SwiftVR, branch modi). Required for "
             "--upscale swiftvr.",
    )
    swiftvr.add_argument(
        "--swiftvr-python", type=str, default="",
        help="Python of the SwiftVR venv (default: <repo>/.venv/bin/python; on Windows "
             "<repo>/.venv/Scripts/python.exe).",
    )
    swiftvr.add_argument(
        "--swiftvr-model-dir", type=str, default="",
        help="SwiftVR checkpoint directory (default: <repo>/checkpoints).",
    )
    swiftvr.add_argument(
        "--swiftvr-accel", default=True, action=argparse.BooleanOptionalAction,
        help="Run the SwiftVR DiT with FP8 linears and torch.compile (default: %(default)s). Needs an "
             "RTX 40 series or newer GPU and a working Triton; the worker drops whatever is unavailable "
             "(with a warning) and then runs in bf16, which needs about 12 GiB for a 1 megapixel tile.",
    )
    swiftvr.add_argument(
        "--upscale-swiftvr-tile", type=float, default=_SWIFTVR_DEFAULT_TILE_MEGAPIXELS, metavar="MP",
        help="Largest tile SwiftVR restores in one call, in megapixels after upscaling (default: "
             "%(default)s). The frame is divided evenly into the fewest tiles that fit. Larger tiles "
             "need more VRAM (FP8: about 7.5 GiB at 1.05, 9.0 GiB at 1.64).",
    )
    swiftvr.add_argument(
        "--upscale-swiftvr-window", type=int, default=_SWIFTVR_DEFAULT_WINDOW, metavar="N",
        help="Frames SwiftVR restores per call (default: %(default)s). SwiftVR keeps no state between "
             f"calls, so neighbouring windows share {_SWIFTVR_WINDOW_OVERLAP} frames and are crossfaded "
             "there. 4k+1 values avoid padding.",
    )

    relayvsr = parser.add_argument_group("RelayVSR (--upscale relayvsr)")
    relayvsr.add_argument(
        "--relayvsr-repo", type=str, default="",
        help="Path to the RelayVSR checkout (fork sh202603/RelayVSR, branch windows-support). Required "
             "for --upscale relayvsr.",
    )
    relayvsr.add_argument(
        "--relayvsr-python", type=str, default="",
        help="Python of the RelayVSR venv (default: <repo>/.venv/bin/python; on Windows "
             "<repo>/.venv/Scripts/python.exe).",
    )
    relayvsr.add_argument(
        "--relayvsr-weights-dir", type=str, default="",
        help="Directory of the RelayVSR checkpoints (default: <repo>/weights). 2x needs wan_x2.safetensors "
             "and flash_x2.safetensors, 4x wan.safetensors and flash.safetensors; the factor follows the "
             "input and --upscale-height.",
    )
    relayvsr.add_argument(
        "--relayvsr-window-size", type=int, default=_RELAYVSR_DEFAULT_WINDOW_SIZE, metavar="N",
        help="Keyframe interval + 1 (default: %(default)s): the large model restores every (N-1)th frame "
             "and the small one the frames between. Larger is faster; the output arrives up to N frames late.",
    )
    relayvsr.add_argument(
        "--relayvsr-wan-weights", type=str, default="fp8", choices=["fp8", "bf16"],
        help="fp8 halves the Wan weights on the GPU and speeds up the keyframes; it needs an RTX 40 series "
             "or newer GPU and falls back to bf16 (with a warning) elsewhere. bf16 gives RelayVSR's reference "
             "output (default: %(default)s).",
    )
    relayvsr.add_argument(
        "--relayvsr-kv-cache", type=str, default="auto", choices=["auto", "gpu", "cpu"],
        help="Where the Wan K/V cache lives (about 12 GB for 2160p output). cpu keeps it in host memory: "
             "the same output, slower keyframes. auto uses the GPU when the cache fits at the start of the "
             "video. gpu forces it there: when the VRAM overflows, everything gets slower "
             "(default: %(default)s).",
    )
    relayvsr.add_argument(
        "--relayvsr-kv-cache-dtype", type=str, default="native", choices=["native", "int8"],
        help="int8 halves the K/V cache and changes the output slightly. The smaller cache can make auto "
             "choose the GPU where it barely fits: on a 16 GB GPU with a desktop on it, 2160p output then "
             "overflowed the VRAM and ran slower than the defaults (default: %(default)s).",
    )

    framegen = parser.add_argument_group("Frame generation")
    framegen.add_argument(
        "--frame-generation",
        type=str,
        default="none",
        choices=["none", "2x", "4x"],
        help="Frame-rate up-conversion factor (default: %(default)s).",
    )
    framegen.add_argument(
        "--frame-generation-backend",
        type=str,
        default="rife",
        choices=["rife", "rtx"],
        help='Frame-gen backend: "rife" (RIFE in PyTorch, any supported GPU, needs rife.pth) or "rtx" '
             "(NVIDIA RTX Video Frame Generation via nvidia-vfx 0.2.0.0+, RTX 40 series or newer, no "
             "weights, about 10x faster than RIFE) (default: %(default)s).",
    )
    framegen.add_argument(
        "--frame-generation-rtx-mode",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help='Quality mode of RTX Video Frame Generation ("rtx" backend only). "high" is about 6x slower '
             'than "medium" for a marginal gain (default: %(default)s).',
    )
    framegen.add_argument(
        "--frame-generation-model-path",
        type=str,
        default="",
        help='Optional path to RIFE weights. If not set, uses "<model_weights>/rife.pth". RIFE backend only.',
    )
    return parser


def _path_collision_key(path: Path) -> str:
    # Same normalization jasna/main.py uses for collision detection: case-fold on
    # Windows (case-insensitive filesystem), exact otherwise.
    absolute = path.resolve(strict=False)
    key = str(absolute)
    return key.casefold() if sys.platform == "win32" else key


def _plan_folder_jobs(input_dir: Path, output_dir: Path, output_pattern: str | None):
    """Plan ``(input_video, output_path)`` jobs for a folder batch.

    Returns ``(jobs, skipped_images)``. Videos only — post-processing is
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
                "post-processing is video-only."
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
    """Minimal ``FrameWriter`` over ``VideoEncoder``.

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

    def after_write(self, frames_written: int, stop_event) -> None:
        pass

    def finish(self) -> None:
        self.close()

    def close(self) -> None:
        if self._entered:
            self._encoder_ctx.__exit__(None, None, None)
            self._entered = False


def run_postprocess(
    reader,
    encoder_ctx,
    *,
    generator=None,
    multiplier: int = 1,
    upscaler=None,
    clip_upscaler=None,
    stream_upscaler=None,
    window: int = _SWIFTVR_DEFAULT_WINDOW,
    progress=None,
) -> int:
    """Decode every frame, push it through the writers, return the source-frame count.

    ``reader`` is an already-entered ``VideoReader``. The writers wrap the encoder in the main
    pipeline's order: ``upscaler`` (a ``FrameUpscaler``), ``clip_upscaler`` (a ``ClipUpscaler``,
    fed windows of ``window`` frames) or ``stream_upscaler`` (a ``StreamUpscaler``, fed frame by
    frame) first, then ``generator`` when ``multiplier`` is above 1.
    The upscaler is owned and closed here with the encoder. The ``generator`` is *borrowed*:
    this function never closes it (the writer is told not to own it); the caller owns its
    lifetime.
    """
    writer = _EncoderWriter(encoder_ctx)
    if multiplier > 1:
        from jasna.frame_generation import FrameGenerationWriter

        writer = FrameGenerationWriter(writer, generator, multiplier, owns_generator=False)
    if upscaler is not None:
        from jasna.upscale import FrameUpscaleWriter

        writer = FrameUpscaleWriter(writer, upscaler)
    elif clip_upscaler is not None:
        from jasna.upscale.clip_writer import ClipUpscaleWriter

        writer = ClipUpscaleWriter(writer, clip_upscaler, window=window, overlap=_SWIFTVR_WINDOW_OVERLAP)
    elif stream_upscaler is not None:
        from jasna.upscale.stream_writer import StreamUpscaleWriter

        writer = StreamUpscaleWriter(writer, stream_upscaler)

    source_frames = 0
    try:
        for batch, pts_list in reader.frames():
            # The decoder allocates a fresh batch tensor per iteration, and the frame
            # generation writer holds the last frame as a view, which keeps that batch's
            # storage alive until the next frame overwrites the view. So no clone is
            # needed here (and the encoder's record_stream covers cross-stream safety).
            # Do NOT "optimize" the decoder to reuse one buffer across batches, or the
            # held previous frame would be corrupted.
            for i in range(len(pts_list)):
                writer.write(batch[i], int(pts_list[i]))
                source_frames += 1
                if progress is not None:
                    progress.update(1)
        writer.finish()
    finally:
        writer.close()
    return source_frames


class _SwiftvrSession:
    """The SwiftVR worker of a run, started for the first video that is upscaled (its frame size
    is the one the worker warms up for) and shared by the videos after it."""

    def __init__(self, *, repo: Path, sv_python: Path, model_dir: Path, accel: bool,
                 tile_megapixels: float, verbose: bool) -> None:
        self._options = dict(
            repo=repo, sv_python=sv_python, model_dir=model_dir, accel=accel,
            tile_megapixels=tile_megapixels, verbose=verbose,
        )
        self._worker = None

    def upscaler(self, device, source_size, output_size, strength: float):
        from jasna.upscale.swiftvr import SwiftvrClipUpscaler, SwiftvrWorker, swiftvr_scale

        if self._worker is None:
            self._worker = SwiftvrWorker(
                device=str(device), warmup_size=source_size,
                warmup_scale=swiftvr_scale(source_size[1], output_size[1]), **self._options,
            )
        return SwiftvrClipUpscaler(
            self._worker, device, source_size=source_size, output_size=output_size, strength=strength
        )

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None


class _RelayvsrSession:
    """The RelayVSR worker of a run, started for the first video that is upscaled (it loads the
    checkpoints of that video's factor) and shared by the videos after it."""

    def __init__(self, *, rv_python: Path, weights_dir: Path, window_size: int, wan_weights: str,
                 kv_cache: str, kv_cache_dtype: str, verbose: bool) -> None:
        self._options = dict(
            rv_python=rv_python, weights_dir=weights_dir, window_size=window_size, wan_weights=wan_weights,
            kv_cache=kv_cache, kv_cache_dtype=kv_cache_dtype, verbose=verbose,
        )
        self._worker = None

    def upscaler(self, device, source_size, output_size, strength: float):
        from jasna.upscale.relayvsr import RelayvsrStreamUpscaler, RelayvsrWorker, relayvsr_scale

        if self._worker is None:
            self._worker = RelayvsrWorker(
                device=str(device), scale=relayvsr_scale(source_size[1], output_size[1]), **self._options
            )
        return RelayvsrStreamUpscaler(
            self._worker, device, source_size=source_size, output_size=output_size, strength=strength
        )

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None


def _warn_heavy_relayvsr(name: str, source_size, output_size) -> None:
    """RelayVSR's memory and time follow what the model emits, not ``output_size``: tell the user
    when the factor lands above 2160p (the K/V cache then moves to host memory on most GPUs)."""
    from jasna.upscale.relayvsr import HEAVY_MODEL_HEIGHT, relayvsr_scale

    scale = relayvsr_scale(source_size[1], output_size[1])
    if source_size[1] * scale > HEAVY_MODEL_HEIGHT:
        print(
            f"Warning: {name}: RelayVSR generates {source_size[0] * scale}x{source_size[1] * scale} "
            f"({scale}x) and the result is resized to {output_size[0]}x{output_size[1]}. This needs far "
            f"more memory and time than {HEAVY_MODEL_HEIGHT}p output."
        )


def _process_video(
    input_path: Path,
    output_path: Path,
    *,
    generator,
    multiplier: int,
    codec: str,
    encoder_settings: dict,
    device,
    batch_size: int,
    show_progress: bool,
    video_decode_backend="native",
    video_encode_backend="native",
    upscale: UpscaleSettings | None = None,
    swiftvr: _SwiftvrSession | None = None,
    relayvsr: _RelayvsrSession | None = None,
    window: int = _SWIFTVR_DEFAULT_WINDOW,
) -> int:
    """Post-process a single video; return its source-frame count.

    The ``generator`` and the worker of ``swiftvr`` or ``relayvsr`` are borrowed (built/closed
    once by the caller so a folder batch reuses them across all videos). Everything else — metadata,
    encoder, reader, the upscaler of ``upscale`` (it depends on the frame size) — is created
    fresh per video. The encoder is closed by ``run_postprocess`` (via the writer) which
    finalizes the encode and muxes audio. Raises ``NothingToDo`` for a video that is already
    ``upscale.upscale_height`` tall when no frames are generated either.
    """
    import torch

    from jasna.media.probe import get_video_meta_data
    from jasna.media.backend import make_video_encoder, make_video_reader

    metadata = get_video_meta_data(str(input_path))
    encoder_metadata = metadata
    source_size = output_size = None
    if upscale is not None:
        from jasna.session_factory import ensure_frame_upscale_engines, frame_upscale_size
        from jasna.upscale import upscaled_metadata

        source_size = (int(metadata.video_width), int(metadata.video_height))
        output_size = frame_upscale_size(upscale, *source_size)
        if output_size is None:
            reason = f"already {source_size[1]} px tall (target {upscale.upscale_height} px)"
            if multiplier == 1:
                raise NothingToDo(reason)
            log.info("Not upscaling %s: %s", input_path.name, reason)
        else:
            log.info(
                "Upscaling %s %dx%d to %dx%d with %s",
                input_path.name, *source_size, *output_size, upscale.upscale_model,
            )
            if relayvsr is not None:
                _warn_heavy_relayvsr(input_path.name, source_size, output_size)
            elif swiftvr is None:
                ensure_frame_upscale_engines(upscale, source_size, output_size, log_callback=log.info)
            encoder_metadata = upscaled_metadata(metadata, output_size)

    encoder_ctx = make_video_encoder(
        file=str(output_path),
        device=device,
        metadata=encoder_metadata,
        codec=codec,
        encoder_settings=encoder_settings,
        lut_path=None,
        frame_generation=multiplier,
        backend=video_encode_backend,
    )
    progress = None
    if show_progress:
        from tqdm import tqdm
        total = metadata.num_frames if metadata.num_frames else None
        progress = tqdm(total=total, unit="frame", desc=input_path.name)
    try:
        with (
            make_video_reader(
                file=str(input_path), batch_size=batch_size, device=device, metadata=metadata,
                backend=video_decode_backend,
            ) as reader,
            torch.inference_mode(),
        ):
            upscaler = clip_upscaler = stream_upscaler = None
            if output_size is not None and swiftvr is not None:
                clip_upscaler = swiftvr.upscaler(device, source_size, output_size, upscale.upscale_strength)
            elif output_size is not None and relayvsr is not None:
                stream_upscaler = relayvsr.upscaler(device, source_size, output_size, upscale.upscale_strength)
            elif output_size is not None:
                from jasna.session_factory import build_frame_upscaler

                upscaler = build_frame_upscaler(upscale, device, source_size, output_size)
            return run_postprocess(
                reader, encoder_ctx, generator=generator, multiplier=multiplier,
                upscaler=upscaler, clip_upscaler=clip_upscaler, stream_upscaler=stream_upscaler,
                window=window, progress=progress,
            )
    finally:
        if progress is not None:
            progress.close()


def _upscale_unavailable_reason(settings: UpscaleSettings) -> str | None:
    """Why this GPU or installation cannot run ``settings.upscale_model`` (the checks
    ``build_restoration_session`` makes for the main CLI)."""
    import torch

    device = torch.device(settings.device)
    if settings.upscale_model == "rtx-super-res":
        from jasna.accelerator import is_amd_device
        from jasna.session_factory import UPSCALE_RTX_NEEDS_NVIDIA

        return UPSCALE_RTX_NEEDS_NVIDIA if is_amd_device(device) else None
    if settings.upscale_model == "dlss5":
        from jasna.upscale.dlss5 import dlss5_runtime_dir, dlss5_unavailable_reason

        return dlss5_unavailable_reason(device, dlss5_runtime_dir())
    return None


def main() -> None:
    multiprocessing.freeze_support()
    parser = build_parser()
    args = parser.parse_args()

    frame_generation = str(args.frame_generation).lower()
    upscale_model = str(args.upscale).lower()
    if frame_generation == "none" and upscale_model == "none":
        parser.error("nothing to do: pass --upscale and/or --frame-generation")
    if not 0.0 <= float(args.upscale_strength) <= 1.0:
        parser.error("--upscale-strength must be between 0 and 1")
    window = int(args.upscale_swiftvr_window)
    swiftvr_paths = relayvsr_paths = None
    if upscale_model in _EXTERNAL_UPSCALERS and frame_generation != "none":
        parser.error(
            f"--upscale {upscale_model} cannot be combined with --frame-generation "
            "(run frame generation as a separate pass)"
        )
    if upscale_model == "relayvsr":
        if int(args.relayvsr_window_size) < 2:
            parser.error("--relayvsr-window-size must be at least 2")
        from jasna.restorer.relayvsr_common import resolve_relayvsr_paths

        try:
            relayvsr_paths = resolve_relayvsr_paths(
                args.relayvsr_repo, args.relayvsr_python, args.relayvsr_weights_dir
            )
        except (ValueError, OSError, RuntimeError) as e:
            parser.error(str(e))
    if upscale_model == "swiftvr":
        if not str(args.swiftvr_repo).strip():
            parser.error("--swiftvr-repo is required for --upscale swiftvr")
        if window < 2 * _SWIFTVR_WINDOW_OVERLAP + 1:
            parser.error(f"--upscale-swiftvr-window must be at least {2 * _SWIFTVR_WINDOW_OVERLAP + 1}")
        if float(args.upscale_swiftvr_tile) <= 0.0:
            parser.error("--upscale-swiftvr-tile must be positive")
        from jasna.restorer.swiftvr_common import resolve_swiftvr_paths

        try:
            swiftvr_paths = resolve_swiftvr_paths(
                args.swiftvr_repo, args.swiftvr_python, args.swiftvr_model_dir, mode="swiftvr"
            )
        except (ValueError, OSError, RuntimeError) as e:
            parser.error(str(e))

    path_ok, path_info = check_ascii_install_path()
    if not path_ok:
        print("Error: Jasna must be installed in a path with ASCII characters only.")
        print(f"Current path: {path_info}")
        sys.exit(1)

    if not args.disable_ffmpeg_check:
        check_required_executables()

    gpu_ok, gpu_result = check_supported_gpu(str(args.device))
    if not gpu_ok:
        print(f"Error: {gpu_check_error(gpu_result)}")
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
    from jasna.media.probe import UnsupportedVideoError

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    multiplier = MULTIPLIER_CHOICES[frame_generation]

    upscale = None
    if upscale_model != "none":
        upscale = UpscaleSettings(
            device=str(args.device),
            fp16=bool(args.fp16),
            upscale_model=upscale_model,
            upscale_height=int(args.upscale_height),
            upscale_strength=float(args.upscale_strength),
            upscale_rtx_quality=args.upscale_rtx_quality,
            upscale_rtx_denoise=args.upscale_rtx_denoise,
            upscale_rtx_deblur=args.upscale_rtx_deblur,
            upscale_dlss5_preset=args.upscale_dlss5_preset,
            upscale_dlss5_style=args.upscale_dlss5_style,
            upscale_dlss5_intensity=float(args.upscale_dlss5_intensity),
        )
        if upscale_model == "dlss5":
            from jasna.cli_downloads import ensure_dlss5_runtime

            ensure_dlss5_runtime(upscale)
        reason = _upscale_unavailable_reason(upscale)
        if reason:
            print(f"Error: {reason}")
            sys.exit(1)

    codec = str(args.codec).lower()
    from jasna.accelerator import vendor_for_device
    from jasna.media.encoder_settings import resolve_custom_encoder_settings
    encoder_settings = resolve_custom_encoder_settings(
        str(args.encoder_settings), cq=None, codec=codec, vendor=vendor_for_device(str(args.device))
    )

    from jasna.media.backend import VideoBackend
    video_backend = VideoBackend(str(args.video_backend).lower())

    def _resolve_backend(override: str) -> VideoBackend:
        return video_backend if override == "inherit" else VideoBackend(override.lower())

    video_decode_backend = _resolve_backend(str(args.video_decode_backend).lower())
    video_encode_backend = _resolve_backend(str(args.video_encode_backend).lower())
    device = torch.device(str(args.device))
    fp16 = bool(args.fp16)
    fg_model_path = str(args.frame_generation_model_path).strip() or None
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
            print(f"Skipping {len(skipped_images)} image(s): post-processing is video-only.")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        jobs = [(input_path, output_path)]

    log.info(
        "postprocess: %d job(s) | upscale=%s frame-generation=%s (%s) codec=%s device=%s",
        len(jobs), upscale_model, frame_generation, args.frame_generation_backend, codec, device,
    )

    total = len(jobs)
    processed = 0
    failed = 0
    with torch.cuda.device(device):
        # Build the generator and the worker session ONCE and reuse them across every job (a
        # folder batch shares one model). They are borrowed by each video's writers and closed
        # here exactly once. build_frame_generator may raise for an unavailable backend
        # (e.g. "rtx"); that propagates before the try below, so the generator is only closed
        # when it was actually created.
        generator = None
        if multiplier > 1:
            generator = build_frame_generator(
                args.frame_generation_backend, device=device, model_path=fg_model_path, fp16=fp16,
                rtx_mode=str(args.frame_generation_rtx_mode).lower(),
            )
        swiftvr = None
        if swiftvr_paths is not None:
            repo, sv_python, model_dir = swiftvr_paths
            swiftvr = _SwiftvrSession(
                repo=repo, sv_python=sv_python, model_dir=model_dir, accel=bool(args.swiftvr_accel),
                tile_megapixels=float(args.upscale_swiftvr_tile),
                verbose=str(args.log_level).lower() == "debug",
            )
        relayvsr = None
        if relayvsr_paths is not None:
            _, rv_python, weights_dir = relayvsr_paths
            relayvsr = _RelayvsrSession(
                rv_python=rv_python, weights_dir=weights_dir, window_size=int(args.relayvsr_window_size),
                wan_weights=str(args.relayvsr_wan_weights), kv_cache=str(args.relayvsr_kv_cache),
                kv_cache_dtype=str(args.relayvsr_kv_cache_dtype),
                verbose=str(args.log_level).lower() == "debug",
            )
        try:
            for i, (vid, out_path) in enumerate(jobs, start=1):
                if input_is_dir:
                    print(f"[{i}/{total}] Processing {vid.name} -> {out_path.name}")
                try:
                    n = _process_video(
                        vid, out_path,
                        generator=generator, multiplier=multiplier, codec=codec,
                        encoder_settings=encoder_settings, device=device,
                        batch_size=batch_size, show_progress=not args.no_progress,
                        video_decode_backend=video_decode_backend, video_encode_backend=video_encode_backend,
                        upscale=upscale, swiftvr=swiftvr, relayvsr=relayvsr, window=window,
                    )
                except NothingToDo as e:
                    # A folder batch skips the video; for a single file there is no output to write.
                    failed += 1
                    print(f"Skipping {vid.name}: {e}.")
                    if not input_is_dir:
                        sys.exit(1)
                    continue
                except UnsupportedVideoError as e:
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
                    "postprocess done: %s | %d source frames -> %d output frames",
                    out_path.name, n, n * multiplier,
                )
        finally:
            if generator is not None:
                generator.close()
            if swiftvr is not None:
                swiftvr.close()
            if relayvsr is not None:
                relayvsr.close()

    if input_is_dir:
        print(f"Done: {processed} processed, {failed} failed/skipped (of {total}).")


if __name__ == "__main__":
    main()

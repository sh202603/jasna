"""The modi fork's part of the ``jasna`` command line: its options, their checks and the
``SessionConfig`` fields they fill.

Upstream keeps the parser in ``jasna.cli_parser`` and the run in ``jasna.main``; both call into
this module at a few marked places (``+modi``), so that an upstream change to those files meets
as little fork code as possible.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from jasna.cli_help import CLI_HELP
from jasna.session_config import (
    LTX_CROP_VSR_BACKENDS,
    LTX_CROP_VSR_RTX_DEBLUR,
    LTX_DEFAULT_MODEL,
    LTX_SHARPEN_RTX_DEBLUR,
)

# Added to upstream's ``CLI_RESTORATION_MODELS``.
RESTORATION_MODELS = {"seedvr2": ("seedvr2", LTX_DEFAULT_MODEL)}

# Added to upstream's ``--secondary-restoration`` choices.
SECONDARY_RESTORATION_CHOICES = ["flashvsr", "flashvsr-inline", "swiftvr", "swiftvr-inline", "swiftvr-distill"]

SECONDARY_RESTORATION_HELP = (
    ' "flashvsr" runs an offline 3-phase pass; "flashvsr-inline" runs FlashVSR '
    'inline in the streaming pipeline (no intermediate files, needs a patched '
    'FlashVSR repo). Both need --flashvsr-repo. See the "FlashVSR" group for '
    'its options. "swiftvr" runs the same offline 3-phase pass with SwiftVR '
    '(for GPUs where it cannot co-reside with the primary: 12 GB cards, no FP8, '
    'the SeedVR2 primary); "swiftvr-inline" runs SwiftVR inline (faster than '
    'FlashVSR, FP8 by default). Both need --swiftvr-repo, see the "SwiftVR" group. '
    '"swiftvr-distill" (experimental) runs a small 2x model distilled from SwiftVR '
    'inside the jasna process; it needs swiftvr-distill.pt in model_weights/ (or '
    '--swiftvr-distill-model).'
)

FRAME_GENERATION_CHOICES = ["4x"]

DETECTION_MODEL_PATH_HELP = (
    " model_weights/ is searched in: $JASNA_MODEL_WEIGHTS_DIR, next to the executable (frozen build), "
    "CWD, then next to the jasna package."
)

RTX_DEBLUR_HELP = (
    "RTX Super Res deblur level, none to disable (default: none; on LTX windows, where the deblur "
    "pass is what sharpens, medium under --ltx-crop-vsr rtx-super-res and low in an LTX job with "
    "--secondary-restoration rtx-super-res)"
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def add_restoration_arguments(restoration) -> None:
    restoration.add_argument(
        "--fp8-recon",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Experimental: run the BasicVSR++ upsample stage as cuDNN FP8 convolutions "
             "(~1.6x faster stage, skips the TRT upsample engine's VRAM arena). Requires an "
             "FP8-capable GPU (sm89+) and --fp16; falls back to the TensorRT engine on any "
             "failure. (default: %(default)s)",
    )


def add_seedvr2_group(parser: argparse.ArgumentParser) -> None:
    seedvr2 = parser.add_argument_group("SeedVR2 (primary restoration, experimental)")
    seedvr2.add_argument(
        "--seedvr2-repo",
        type=str,
        default="",
        help="Path to a ComfyUI-SeedVR2_VideoUpscaler checkout with its own venv "
             "(required for --restoration-model-name seedvr2).",
    )
    seedvr2.add_argument(
        "--seedvr2-python",
        type=str,
        default="",
        help="Python interpreter of the SeedVR2 venv (default: <repo>/.venv/bin/python, "
             "Scripts\\python.exe on Windows).",
    )
    seedvr2.add_argument(
        "--seedvr2-model-dir",
        type=str,
        default="",
        help="SeedVR2 base weights directory (default: <repo>/models/SEEDVR2; "
             "auto-downloaded there on first load).",
    )
    seedvr2.add_argument(
        "--seedvr2-dit",
        type=str,
        default="seedvr2_ema_3b_fp16.safetensors",
        help="DiT weights filename inside the model dir (default: %(default)s)",
    )
    seedvr2.add_argument(
        "--seedvr2-lora",
        type=str,
        default="",
        help='LoRA checkpoint path (default: "<model_weights>/lada_seedvr2_lora_v6.pt").',
    )
    seedvr2.add_argument(
        "--seedvr2-lora-rank",
        type=int,
        default=16,
        help="LoRA rank (must match the checkpoint; default: %(default)s)",
    )
    seedvr2.add_argument(
        "--seedvr2-window",
        type=int,
        default=33,
        help="Sliding-window length in frames, must be 4n+1 (default: %(default)s)",
    )
    seedvr2.add_argument(
        "--seedvr2-overlap",
        type=int,
        default=9,
        help="Cross-fade overlap between windows in frames (default: %(default)s)",
    )
    seedvr2.add_argument(
        "--seedvr2-color-fix",
        type=str,
        default="lab",
        choices=["none", "lab", "wavelet"],
        help="Per-clip color correction against the input crops. 'wavelet' can re-introduce "
             "mosaic grid structure from the reference (default: %(default)s)",
    )
    seedvr2.add_argument(
        "--seedvr2-empty-cache",
        type=str,
        default="auto",
        choices=["auto", "always", "never"],
        help="Per-clip VRAM release in the worker, returning cached VRAM to the co-resident "
             "detection/decode process. 'auto' resolves to 'always' below 20GiB total VRAM "
             "(an OOM safeguard there) and 'never' above (pure overhead) (default: %(default)s)",
    )
    seedvr2.add_argument(
        "--seedvr2-accel",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Run the SeedVR2 VAE through fused fp16 kernels and the DiT block linears as FP8 "
             "GEMM with the LoRA merged into the weights (default: %(default)s). Needs a "
             "SeedVR2 checkout that provides these paths (the modi branch of the fork "
             "sh202603/ComfyUI-SeedVR2_VideoUpscaler; not in the upstream repository), "
             "comfy-kitchen in its venv, and for FP8 an RTX 40 series or newer GPU with a "
             "working Triton; the worker drops whatever is unavailable (with a warning) and "
             "runs the standard path for it. About 1.85x the crop throughput on an RTX 5080; "
             "the output differs slightly (about 46 dB PSNR).",
    )


def add_rtx_arguments(rtx) -> None:
    rtx.add_argument(
        "--rtx-strength",
        type=float,
        default=1.0,
        help="Strength of every RTX Super Res pass (upscale, denoise, deblur), 0.0-1.0; lower values blend "
             "toward the unprocessed picture. Needs nvidia-vfx 0.2.0.0+ (default: %(default)s)",
    )
    rtx.add_argument(
        "--rtx-highbitrate",
        default=False,
        action="store_true",
        help="Use the RTX Super Res upscale models tuned for clean, high-bitrate sources (HIGHBITRATE_*). "
             "They skip compression-artifact suppression, which may keep more detail on the restored frames. "
             "Experimental (default: off)",
    )


def add_secondary_groups(parser: argparse.ArgumentParser) -> None:
    """The FlashVSR, SwiftVR, SwiftVR distill and LTX output sharpening groups."""
    flashvsr = parser.add_argument_group("FlashVSR")
    from jasna.restorer.flashvsr_offline import add_flashvsr_arguments
    add_flashvsr_arguments(flashvsr)

    swiftvr = parser.add_argument_group("SwiftVR")
    from jasna.restorer.swiftvr_common import add_swiftvr_arguments
    add_swiftvr_arguments(swiftvr)

    swiftvr_distill = parser.add_argument_group("SwiftVR distill (experimental)")
    swiftvr_distill.add_argument(
        "--swiftvr-distill-model",
        type=str,
        default="",
        help='Path to the SwiftVR distillation checkpoint for --secondary-restoration '
             'swiftvr-distill. If not set, uses "<model_weights>/swiftvr-distill.pt"; a bare '
             'file name is looked up in model_weights/. The weights are not bundled: download '
             'swiftvr-distill.pt from https://huggingface.co/okatti/swiftvr-distill (AGPL-3.0).',
    )
    swiftvr_distill.add_argument(
        "--swiftvr-distill-view-window",
        type=int,
        default=15,
        metavar="N",
        help="Smooth the placement of the crops the model sees over N frames, like "
             "--swiftvr-view-window (default: %(default)s; 0 disables). On real footage it "
             "lowers the frame-to-frame flicker without costing detail.",
    )
    swiftvr_distill.add_argument(
        "--swiftvr-distill-strength",
        type=float,
        default=0.75,
        metavar="S",
        help="Scale the detail the model adds to its upscaled input (0-2; default: "
             "%(default)s). 0 is the plain bilinear upscale, 1 the model's output; lower "
             "values trade detail for less frame-to-frame flicker. The default keeps the "
             "flicker below that of swiftvr-inline at scale 2.",
    )
    swiftvr_distill.add_argument(
        "--swiftvr-distill-stabilize",
        type=int,
        default=0,
        metavar="N",
        help="Experimental: blend the detail the model adds over N frames on each side "
             "(0-8; default: %(default)s, disabled). A neighbouring frame only contributes "
             "where its input is close to that of the current frame, so moving content is "
             "left alone. On the test footage it traded detail for flicker like a lower "
             "--swiftvr-distill-strength; not measured on real footage.",
    )

    ltx_refine = parser.add_argument_group("LTX output sharpening (modi)")
    ltx_refine.add_argument(
        "--ltx-crop-vsr",
        type=str,
        default=None,
        choices=list(LTX_CROP_VSR_BACKENDS),
        metavar="BACKEND",
        help="Sharpen the mosaic areas of a video restored with LTX by the release build. --input is "
             "the LTX output and --ltx-crop-vsr-source the video it was restored from: the source is "
             "scanned for the mosaics (the LTX output has none left), the LTX output is cropped with "
             "LTX's own cameras, each crop is upscaled x2 (swiftvr: the full model through the SwiftVR "
             "worker; swiftvr-distill: the in-process student; rtx-super-res: RTX Video Super Resolution "
             "frame by frame with the --rtx-* options except --rtx-scale, its deblur pass on by default since "
             "that is what sharpens) and composited back with LTX's own "
             "blending. The output keeps the LTX output's resolution, frame count and timestamps; the "
             "rest of the picture is re-encoded unchanged, so keep both the LTX output and this output "
             "at a high quality (low --cq). One video at a time; BasicVSR++ is not loaded. Pass "
             "--ltx-large-canvas only if the LTX run restored at 768 px (no '512 px' warning in its log).",
    )
    ltx_refine.add_argument(
        "--ltx-crop-vsr-source",
        type=str,
        default="",
        metavar="VIDEO",
        help="The video the LTX output was restored from (the one with the mosaics). Required with "
             "--ltx-crop-vsr; must have the same frame size, frame count and timing as --input.",
    )


def add_video_backend_arguments(encoding) -> None:
    encoding.add_argument(
        "--video-backend",
        type=str,
        default="native",
        choices=["native", "auto", "torchcodec"],
        help='Decode/encode backend: "native" (PyAV NVDEC/NVENC, default), "auto" (use torchcodec for decode where '
             'available, else native), or "torchcodec" (force torchcodec; errors when it cannot satisfy '
             'the request, e.g. 10-bit source / streaming).',
    )
    encoding.add_argument(
        "--video-decode-backend",
        type=str,
        default="inherit",
        choices=["inherit", "native", "auto", "torchcodec"],
        help='Override the video decode backend (native / torchcodec) only (default "inherit" = --video-backend).',
    )
    encoding.add_argument(
        "--video-encode-backend",
        type=str,
        default="inherit",
        choices=["inherit", "native", "auto", "torchcodec"],
        help='Override the video encode backend (native / torchcodec) only (default "inherit" = --video-backend). '
             "torchcodec encode runs only when forced and outputs 8-bit HEVC/AV1.",
    )


def add_frame_generation_arguments(encoding) -> None:
    encoding.add_argument(
        "--frame-generation-backend",
        type=str,
        default="rtx",
        choices=["rife", "rtx"],
        help=CLI_HELP["frame_generation_backend"],
    )
    encoding.add_argument(
        "--frame-generation-rtx-mode",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help=CLI_HELP["frame_generation_rtx_mode"],
    )
    encoding.add_argument(
        "--frame-generation-model-path",
        type=str,
        default="",
        help=CLI_HELP["frame_generation_model_path"],
    )


# ---------------------------------------------------------------------------
# SessionConfig
# ---------------------------------------------------------------------------


def rtx_deblur(args: argparse.Namespace, restoration_models: dict) -> str:
    """``--rtx-deblur`` resolved: off unless given, except where RTX Super Res sharpens LTX
    canvases, where the deblur pass is the one that sharpens: medium under ``--ltx-crop-vsr
    rtx-super-res``, low in an LTX job's compose pass (``--secondary-restoration rtx-super-res``)."""
    if args.rtx_deblur is not None:
        return str(args.rtx_deblur).lower()
    if getattr(args, "ltx_crop_vsr", None) == "rtx-super-res":
        return LTX_CROP_VSR_RTX_DEBLUR
    return LTX_SHARPEN_RTX_DEBLUR if ltx_sharpens(args, restoration_models) else "none"


def ltx_sharpens(args: argparse.Namespace, restoration_models: dict) -> bool:
    """Whether the job is an LTX model with ``--secondary-restoration rtx-super-res``."""
    model = restoration_models.get(getattr(args, "restoration_model_name", None))
    return model is not None and model[0] == "ltx" and str(args.secondary_restoration).lower() == "rtx-super-res"


def session_config_fields(args: argparse.Namespace, restoration_models: dict) -> dict[str, object]:
    """The ``SessionConfig`` fields the fork's options fill (``rtx_deblur`` replaces upstream's)."""
    return dict(
        rtx_deblur=rtx_deblur(args, restoration_models),
        rtx_strength=float(args.rtx_strength),
        rtx_highbitrate=bool(args.rtx_highbitrate),
        frame_generation_backend=str(args.frame_generation_backend).lower(),
        frame_generation_rtx_mode=str(args.frame_generation_rtx_mode).lower(),
        frame_generation_model_path=str(args.frame_generation_model_path).strip(),
        flashvsr_repo=str(getattr(args, "flashvsr_repo", "") or ""),
        flashvsr_python=str(getattr(args, "flashvsr_python", "") or ""),
        flashvsr_model_dir=str(getattr(args, "flashvsr_model_dir", "") or ""),
        flashvsr_version=str(getattr(args, "flashvsr_version", "11")),
        flashvsr_dtype=str(getattr(args, "flashvsr_dtype", "bf16")),
        flashvsr_tiles=int(getattr(args, "flashvsr_tiles", 1)),
        flashvsr_scale=int(getattr(args, "flashvsr_scale", 4)),
        flashvsr_accel=bool(getattr(args, "flashvsr_accel", False)),
        flashvsr_lora=str(getattr(args, "flashvsr_lora", "") or ""),
        flashvsr_log_level=str(getattr(args, "log_level", "error")),
        swiftvr_repo=str(getattr(args, "swiftvr_repo", "") or ""),
        swiftvr_python=str(getattr(args, "swiftvr_python", "") or ""),
        swiftvr_model_dir=str(getattr(args, "swiftvr_model_dir", "") or ""),
        swiftvr_scale=int(getattr(args, "swiftvr_scale", 4)),
        swiftvr_accel=bool(getattr(args, "swiftvr_accel", True)),
        swiftvr_view_window=int(getattr(args, "swiftvr_view_window", 15)),
        swiftvr_log_level=str(getattr(args, "log_level", "error")),
        ltx_crop_vsr=getattr(args, "ltx_crop_vsr", None),
        ltx_crop_vsr_source=(
            Path(str(args.ltx_crop_vsr_source)) if str(getattr(args, "ltx_crop_vsr_source", "") or "").strip() else None
        ),
        swiftvr_distill_model=str(getattr(args, "swiftvr_distill_model", "") or ""),
        swiftvr_distill_view_window=int(getattr(args, "swiftvr_distill_view_window", 15)),
        swiftvr_distill_strength=float(getattr(args, "swiftvr_distill_strength", 0.75)),
        swiftvr_distill_stabilize=int(getattr(args, "swiftvr_distill_stabilize", 0)),
        seedvr2_repo=str(getattr(args, "seedvr2_repo", "") or ""),
        seedvr2_python=str(getattr(args, "seedvr2_python", "") or ""),
        seedvr2_model_dir=str(getattr(args, "seedvr2_model_dir", "") or ""),
        seedvr2_dit=str(getattr(args, "seedvr2_dit", "seedvr2_ema_3b_fp16.safetensors")),
        seedvr2_lora_rank=int(getattr(args, "seedvr2_lora_rank", 16)),
        seedvr2_window=int(getattr(args, "seedvr2_window", 33)),
        seedvr2_overlap=int(getattr(args, "seedvr2_overlap", 9)),
        seedvr2_color_fix=str(getattr(args, "seedvr2_color_fix", "lab")),
        seedvr2_empty_cache=str(getattr(args, "seedvr2_empty_cache", "auto")),
        seedvr2_accel=bool(getattr(args, "seedvr2_accel", False)),
    )


# ---------------------------------------------------------------------------
# Checks (no GPU needed)
# ---------------------------------------------------------------------------


def _explicit(option: str) -> bool:
    return any(value == option or value.startswith(option + "=") for value in sys.argv[1:])


def explicit_rtx_scale() -> bool:
    return _explicit("--rtx-scale")


def check_ltx_rtx_scale(
    parser: argparse.ArgumentParser, args: argparse.Namespace, restoration_models: dict, model_option: str
) -> None:
    """Without auto split RTX Super Res only runs on LTX windows, at 2x."""
    if (
        args.ltx_auto_split is None
        and ltx_sharpens(args, restoration_models)
        and explicit_rtx_scale()
        and int(args.rtx_scale) != 2
    ):
        parser.error(f"{model_option} runs RTX Super Res at scale 2; --rtx-scale {args.rtx_scale} is not supported")


def check_ltx_crop_vsr_options(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    input_video: Path | None,
    is_streaming: bool,
    input_is_image: bool,
    input_is_dir: bool,
    segments_spec: str,
) -> None:
    """``--ltx-crop-vsr`` runs its own pass (LTX_REFINE_DESIGN §2.3): the input is an LTX
    output, so the primary restorer, its options and everything that changes the frame
    sequence or the whole picture are refused; the SwiftVR options that only apply to
    the frame pipeline's crops are refused too."""
    option = f"--ltx-crop-vsr {args.ltx_crop_vsr}"
    source = str(args.ltx_crop_vsr_source).strip()
    if not source:
        parser.error(f"{option} needs --ltx-crop-vsr-source (the video the LTX output was restored from)")
    source_path = Path(source)
    if not source_path.is_file():
        parser.error(f"--ltx-crop-vsr-source not found: {source_path}")
    if input_video is not None and input_video.is_file() and source_path.resolve() == input_video.resolve():
        parser.error("--ltx-crop-vsr-source must be the source video, not the LTX output given to --input")
    if args.restoration_model_name != "basicvsrpp":
        parser.error(f"{option} needs --restoration-model-name basicvsrpp (no primary restorer loads)")
    for dest in ("ltx_fast", "ltx_seed", "ltx_trial"):
        if getattr(args, dest) != parser.get_default(dest):
            parser.error(f"--{dest.replace('_', '-')} needs --restoration-model-name ltx or ltx-undistilled")
    if _explicit("--swiftvr-scale") and int(args.swiftvr_scale) != 2:
        parser.error(f"{option} runs SwiftVR at scale 2; --swiftvr-scale {args.swiftvr_scale} is not supported")
    if args.ltx_crop_vsr == "rtx-super-res" and explicit_rtx_scale() and int(args.rtx_scale) != 2:
        parser.error(f"{option} runs RTX Super Res at scale 2; --rtx-scale {args.rtx_scale} is not supported")
    rejected = (
        (str(args.secondary_restoration).lower() != "none", "--secondary-restoration"),
        (str(args.denoise).lower() != "none", "--denoise"),
        (bool(segments_spec), "--segments"),
        (args.ltx_auto_split is not None, "--ltx-auto-split"),
        (is_streaming, "--stream"),
        (bool(args.retarget_high_fps), "--retarget-high-fps"),
        (bool(args.fmp4), "--fmp4"),
        (str(args.frame_generation).lower() != "none", "--frame-generation"),
        (str(args.upscale).lower() != "none", "--upscale (upscale in the LTX run instead)"),
        (str(args.vr_mode).lower() in ("sbs", "sbs-fisheye"), f"--vr-mode {args.vr_mode}"),
        (input_is_image, "an image input"),
        (input_is_dir, "a folder input"),
        (bool(str(args.lut).strip()), "--lut (apply it in the LTX run instead)"),
        (float(args.sharpen) != 0.0, "--sharpen (apply it in the LTX run instead)"),
        (
            int(args.swiftvr_view_window) != parser.get_default("swiftvr_view_window"),
            "--swiftvr-view-window (the LTX camera is smooth already)",
        ),
        (
            int(args.swiftvr_distill_view_window) != parser.get_default("swiftvr_distill_view_window"),
            "--swiftvr-distill-view-window (the LTX camera is smooth already)",
        ),
        (
            bool(args.ltx_large_canvas) and int(args.swiftvr_distill_stabilize) > 0,
            "--ltx-large-canvas and --swiftvr-distill-stabilize together (about 10 GB of VRAM at 768 px)",
        ),
    )
    for condition, what in rejected:
        if condition:
            parser.error(f"{option} cannot be combined with {what}")


# ---------------------------------------------------------------------------
# Run (called from ``jasna.main._main`` in this order)
# ---------------------------------------------------------------------------


def apply_fp8_recon(args: argparse.Namespace) -> None:
    """``--fp8-recon``, switched on by the inline secondary restorers, reaches every execution path
    (pipeline, benchmarks, GUI-launched runs) through the environment."""
    if str(getattr(args, "secondary_restoration", "")).lower() in ("flashvsr-inline", "swiftvr-inline") \
            and not getattr(args, "fp8_recon", False):
        # fp8-recon shrinks the primary's peak by ~0.9-1.7GB, which the inline
        # co-residence budget assumes (FLASHVSR_INLINE_FEASIBILITY §12; the same
        # budget for SwiftVR). Auto-enable it; the restorer falls back to TRT if
        # the GPU can't do fp8 (sm89+/--fp16).
        args.fp8_recon = True
    if getattr(args, "fp8_recon", False):
        os.environ["JASNA_FP8_RECON"] = "1"


def run_offline_secondary(args: argparse.Namespace) -> bool:
    """Run the offline FlashVSR / SwiftVR 3-phase pass when the job asks for one; True when it ran.

    Each phase is its own subprocess so their peak VRAM never overlaps. Dispatched before the
    heavy torch / pipeline imports: the orchestrator only spawns subprocesses."""
    if str(getattr(args, "flashvsr_lora", "") or "").strip() \
            and str(args.secondary_restoration).lower() != "flashvsr-inline":
        # Checked before the offline dispatch: its phases re-run jasna with the same argv,
        # where a later check would surface as a phase failure.
        raise ValueError("--flashvsr-lora is supported with --secondary-restoration flashvsr-inline only")
    engine = str(args.secondary_restoration).lower()
    if engine not in ("flashvsr", "swiftvr"):
        return False
    from jasna.restorer.flashvsr_offline import run_flashvsr_offline

    run_flashvsr_offline(args, engine=engine)
    return True


def video_backends(args: argparse.Namespace, *, is_streaming: bool, segments_spec: str):
    """``(decode, encode)`` torchcodec / native backends of the job, with the combinations
    torchcodec cannot serve refused."""
    from jasna.media.backend import VideoBackend

    video_backend = VideoBackend(str(args.video_backend).lower())

    def resolve(override: str) -> VideoBackend:
        return video_backend if override == "inherit" else VideoBackend(override.lower())

    decode = resolve(str(args.video_decode_backend).lower())
    encode = resolve(str(args.video_encode_backend).lower())
    if VideoBackend.TORCHCODEC in (decode, encode) and is_streaming:
        raise ValueError("torchcodec backend is not supported in streaming mode (use native or auto)")
    if encode is VideoBackend.TORCHCODEC and segments_spec:
        raise ValueError(
            "torchcodec encode is not supported with --segments smart rendering "
            "(fragments require the native encoder)"
        )
    if decode is VideoBackend.TORCHCODEC and bool(args.retarget_high_fps):
        raise ValueError(
            "torchcodec decode does not support --retarget-high-fps (frame stride); "
            "use the native decode backend"
        )
    return decode, encode


def seedvr2_lora_path(args: argparse.Namespace) -> Path:
    """The LoRA checkpoint ``restoration_model_path`` carries for the SeedVR2 primary."""
    from jasna.engine_paths import default_restoration_model_path

    if args.restoration_model_path:
        print("Warning: --restoration-model-path is BasicVSR++-only and ignored with "
              "--restoration-model-name seedvr2 (use --seedvr2-lora)")
    lora_arg = str(args.seedvr2_lora).strip()
    path = Path(lora_arg).expanduser() if lora_arg else default_restoration_model_path("seedvr2")
    if not path.exists():
        raise FileNotFoundError(
            f"SeedVR2 LoRA checkpoint not found: {path}. Download it with:\n"
            "  wget -O model_weights/lada_seedvr2_lora_v6.pt "
            "https://huggingface.co/sh202603/lada-seedvr2-lora/resolve/main/lada_seedvr2_lora_v6.pt\n"
            "or pass --seedvr2-lora."
        )
    return path


def _check_seedvr2(args: argparse.Namespace, *, is_streaming: bool, videos: list[Path]) -> None:
    log = logging.getLogger("jasna.main")
    secondary_name = str(args.secondary_restoration).lower()
    if not str(args.seedvr2_repo).strip():
        raise ValueError(
            "--restoration-model-name seedvr2 requires --seedvr2-repo "
            "(path to a ComfyUI-SeedVR2_VideoUpscaler checkout)"
        )
    if int(args.seedvr2_window) < 1 or int(args.seedvr2_window) % 4 != 1:
        raise ValueError("--seedvr2-window must be 4n+1 (e.g. 33)")
    if not 0 <= int(args.seedvr2_overlap) < int(args.seedvr2_window):
        raise ValueError("--seedvr2-overlap must be >= 0 and smaller than --seedvr2-window")
    if args.frame_generation != "none":
        raise ValueError(
            "--restoration-model-name seedvr2 does not support --frame-generation "
            "(no VRAM budget next to the resident worker; run frame generation as a separate pass)"
        )
    if args.upscale != "none":
        # Untested next to the resident worker (same VRAM budget concern as frame generation).
        raise ValueError("--restoration-model-name seedvr2 does not support --upscale")
    if secondary_name in ("flashvsr-inline", "swiftvr-inline"):
        raise ValueError(
            "--restoration-model-name seedvr2 cannot be combined with --secondary-restoration "
            f"{secondary_name} (the two resident workers exceed a 16 GB card; use the offline "
            "'flashvsr' or 'swiftvr' mode, whose phases never co-reside)"
        )
    if secondary_name == "tvai":
        log.warning("[seedvr2] TVAI sharpening on top of a diffusion primary risks over-enhancement")
    if getattr(args, "fp8_recon", False):
        log.warning("[seedvr2] --fp8-recon is BasicVSR++-only and has no effect with seedvr2")
    if is_streaming:
        log.warning(
            "[seedvr2] streaming: mosaic-dense stretches are limited to the SeedVR2 "
            "rate (~17-21 crop-fps); expect player stalls on such content"
        )
    # jasna's VR crops are projection-conditioned: out of distribution for the LoRA and
    # unverified (same rejection style as flashvsr offline).
    vr_mode = str(args.vr_mode).strip().lower()
    if vr_mode in ("sbs", "sbs-fisheye"):
        raise ValueError(f"--restoration-model-name seedvr2 does not support VR processing (--vr-mode {vr_mode})")
    if vr_mode == "auto":
        from jasna.media.probe import get_video_meta_data
        from jasna.vr180 import resolve_vr_mode

        for probe_path in videos:
            vr_resolution = resolve_vr_mode("auto", get_video_meta_data(str(probe_path)), probe_path)
            if vr_resolution.resolved != "off":
                raise ValueError(
                    "--restoration-model-name seedvr2 does not support VR processing, but "
                    f"--vr-mode auto detected VR content in {probe_path.name} "
                    f"({vr_resolution.resolved}: {vr_resolution.reason}). "
                    "Pass --vr-mode off to force flat processing."
                )


def check_job_options(
    args: argparse.Namespace, *, restoration_backend: str, is_streaming: bool, videos: list[Path]
) -> None:
    """The fork's checks that run after the system checks and before the session is built.
    ``args.flashvsr_lora`` is made absolute (the worker changes into its checkout)."""
    secondary_name = str(args.secondary_restoration).lower()
    if restoration_backend == "seedvr2":
        _check_seedvr2(args, is_streaming=is_streaming, videos=videos)

    lora_arg = str(getattr(args, "flashvsr_lora", "") or "").strip()
    if lora_arg:
        from jasna.model_weights_resolver import resolve_model_weights_file

        lora_file = Path(lora_arg).expanduser()
        if not lora_file.is_file() and lora_file.parent == Path("."):
            lora_file = resolve_model_weights_file(lora_arg)  # bare file name -> model_weights/
        if not lora_file.is_file():
            raise FileNotFoundError(f"--flashvsr-lora not found: {lora_arg} (also looked in {lora_file})")
        args.flashvsr_lora = str(lora_file.resolve())

    if secondary_name in ("flashvsr-inline", "swiftvr-inline"):
        # No clip cap (both are flat in VRAM vs clip length); frame-gen is a separate pass.
        if args.frame_generation != "none":
            raise ValueError(
                f"--secondary-restoration {secondary_name} does not support --frame-generation "
                "(run frame generation as a separate pass)"
            )
        if args.upscale != "none":
            # Untested next to the resident worker (same VRAM budget concern as frame generation).
            raise ValueError(f"--secondary-restoration {secondary_name} does not support --upscale")

    if args.ltx_crop_vsr == "rtx-super-res":
        # Checked here as well as in the session factory, which runs after engine compilation.
        from jasna.session_factory import nvvfx_installed

        if not nvvfx_installed():
            raise ValueError(
                "--ltx-crop-vsr rtx-super-res needs the RTX Video Effects runtime (the nvidia-vfx package "
                "of the nvidia extra), which is not installed"
            )

    if secondary_name == "swiftvr-distill" or args.ltx_crop_vsr == "swiftvr-distill":
        # Checked here as well as in the session factory: the factory runs after engine
        # compilation, which can take the better part of an hour.
        from jasna.restorer.swiftvr_distill_model import resolve_swiftvr_distill_model_path

        resolve_swiftvr_distill_model_path(str(args.swiftvr_distill_model))
        if int(args.swiftvr_distill_view_window) < 0:
            raise ValueError("--swiftvr-distill-view-window must be >= 0")
        if not 0.0 <= float(args.swiftvr_distill_strength) <= 2.0:
            raise ValueError("--swiftvr-distill-strength must be in [0, 2]")
        if not 0 <= int(args.swiftvr_distill_stabilize) <= 8:
            raise ValueError("--swiftvr-distill-stabilize must be in [0, 8]")

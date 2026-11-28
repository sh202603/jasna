"""Shared composition root for the video restoration pipeline.

Builds the heavy restoration session (engine compilation, primary and
secondary restorers, and a detector cached across videos) and per-video
``Pipeline`` instances from one ``SessionConfig``. Consumed by both the CLI (``jasna.main``) and the GUI
(``jasna.gui.video_session`` / ``jasna.gui.processor``).

All heavy imports (torch, restorers, pipeline) stay inside the functions.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, get_args

from jasna.session_config import (
    LTX_WINDOW_RESTORERS,
    RestorationModelName,
    SecondaryRestorationName,
    SessionConfig,
    UpscaleModelName,
    WORKER_UPSCALE_MODELS,
    ltx_unsupported_options,
)

if TYPE_CHECKING:
    import torch

    from jasna.accelerator import AcceleratorVendor

    from jasna.ltx.model_files import LtxModelFiles
    from jasna.ltx.super_res import LtxSuperRes
    from jasna.media.probe import VideoMetadata
    from jasna.media.splice import KeyframeIndex, SplicePlan
    from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel
    from jasna.mosaic.yolo import YoloMosaicDetectionModel
    from jasna.pipeline import Pipeline
    from jasna.progressbar import ProgressCallback
    from jasna.restorer.restoration_pipeline import RestorationPipeline
    from jasna.restorer.sd15_inpaint_restorer import Sd15InpaintRestorer
    from jasna.segments import SegmentRange
    from jasna.upscale import FrameUpscaler, Size

    DetectionModel = RfDetrMosaicDetectionModel | YoloMosaicDetectionModel


logger = logging.getLogger(__name__)


@dataclass
class RestorationSession:
    device: "torch.device"
    restoration_pipeline: "RestorationPipeline | None"
    log_callback: Callable[[str], None] | None
    ltx_files: "LtxModelFiles | None" = None
    ltx_super_res: "LtxSuperRes | None" = None
    # +modi: the worker of ``--upscale swiftvr`` / ``relayvsr``, kept for the videos of the job.
    upscale_worker: object | None = None
    _detection_key: tuple | None = None
    _detection_model: "DetectionModel | None" = None

    def detection_model_for(self, config: SessionConfig) -> "DetectionModel":
        """The detector for ``config``, its engine compiled when missing, reused across videos
        until its settings change."""
        key = (
            config.detection_model_name,
            config.detection_model_path,
            config.detection_score_threshold,
            config.batch_size,
            config.fp16,
        )
        if key != self._detection_key:
            if self._detection_model is not None:
                self._detection_model.close()
                self._detection_model, self._detection_key = None, None
            self._detection_model = build_compiled_detection_model(
                config.detection_model_name,
                config.detection_model_path,
                device=self.device,
                batch_size=config.batch_size,
                fp16=config.fp16,
                score_threshold=config.detection_score_threshold,
                log_callback=self.log_callback,
            )
            self._detection_key = key
        return self._detection_model

    def release_detection_model(self) -> None:
        """Free the detector (its VRAM goes to the next model); the next
        ``detection_model_for`` builds it again."""
        if self._detection_model is not None:
            self._detection_model.close()
            self._detection_model = None
        self._detection_key = None

    def close_upscale_worker(self) -> None:
        if self.upscale_worker is not None:
            self.upscale_worker.close()
            self.upscale_worker = None

    def close(self) -> None:
        self.release_detection_model()
        self.close_upscale_worker()
        if self.ltx_super_res is not None:
            self.ltx_super_res.close()
            self.ltx_super_res = None
        if self.restoration_pipeline is not None:
            self.restoration_pipeline.restorer.close()
            if self.restoration_pipeline.secondary_restorer is not None:
                self.restoration_pipeline.secondary_restorer.close()


def build_compiled_detection_model(
    detection_model_name: str,
    detection_model_path: Path,
    *,
    device: "torch.device",
    batch_size: int,
    fp16: bool,
    score_threshold: float,
    log_callback: Callable[[str], None] | None,
) -> "DetectionModel":
    """Compile the detector's TensorRT engine if it is missing, then load the detector."""
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.mosaic.detection_registry import build_detection_model

    ensure_engines_compiled(
        EngineCompilationRequest(
            device=str(device),
            fp16=fp16,
            detection=True,
            detection_model_name=detection_model_name,
            detection_model_path=str(detection_model_path),
            detection_batch_size=batch_size,
        ),
        log_callback=log_callback,
    )
    return build_detection_model(
        detection_model_name,
        detection_model_path,
        batch_size=batch_size,
        device=device,
        score_threshold=score_threshold,
        fp16=fp16,
    )


@dataclass
class ImageSession:
    """The mosaic detector and SD 1.5 inpaint restorer that restore still images."""

    detector: "DetectionModel"
    restorer: "Sd15InpaintRestorer"
    device: "torch.device"
    fp16: bool

    def close(self) -> None:
        self.detector.close()
        self.restorer.close()


def build_image_session(
    detection_model_name: str,
    detection_model_path: Path,
    *,
    device: "torch.device",
    batch_size: int,
    fp16: bool,
    score_threshold: float,
    log_callback: Callable[[str], None] | None,
) -> ImageSession:
    from jasna.engine_paths import SD15_DIR
    from jasna.restorer.sd15_inpaint_restorer import Sd15InpaintRestorer

    detector = build_compiled_detection_model(
        detection_model_name,
        detection_model_path,
        device=device,
        batch_size=batch_size,
        fp16=fp16,
        score_threshold=score_threshold,
        log_callback=log_callback,
    )
    return ImageSession(detector, Sd15InpaintRestorer(SD15_DIR, device, fp16), device, fp16)


def nvvfx_installed() -> bool:
    """Whether the RTX Video Effects runtime (the ``nvidia-vfx`` package) is importable."""
    from importlib.util import find_spec

    try:
        return find_spec("nvvfx") is not None
    except ValueError:  # a module object without a spec (the test suite's fake) is installed
        return True


def _build_secondary_restorer(config: SessionConfig, device: "torch.device", *, size_from_frames: bool = False):
    """``size_from_frames``: the restorer takes its input side from the frames it is given
    instead of the frame pipeline's 256 px crops (RTX Super Res binds its passes to a size)."""
    if config.secondary_restoration == "none":
        return None
    if config.secondary_restoration == "tvai":
        from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer

        tvai_args = f"model={config.tvai_model}:scale={config.tvai_scale}:{config.tvai_args}"
        return TvaiSecondaryRestorer(
            ffmpeg_path=config.tvai_ffmpeg_path,
            tvai_args=tvai_args,
            tvai_denoise=bool(config.tvai_denoise),
            scale=int(config.tvai_scale),
            num_workers=int(config.tvai_workers),
        )
    if config.secondary_restoration == "unet-4x":
        from jasna.restorer.unet4x_secondary_restorer import Unet4xSecondaryRestorer

        return Unet4xSecondaryRestorer(device=device, fp16=bool(config.fp16))
    if config.secondary_restoration == "rtx-super-res":
        from jasna.restorer.rtx_superres_secondary_restorer import RtxSuperresSecondaryRestorer

        return RtxSuperresSecondaryRestorer(
            device=device,
            scale=int(config.rtx_scale),
            quality=config.rtx_quality,
            denoise=config.rtx_denoise,
            deblur=config.rtx_deblur,
            strength=float(config.rtx_strength),
            highbitrate=bool(config.rtx_highbitrate),
            **({"input_size": None} if size_from_frames else {}),
        )
    if config.secondary_restoration == "flashvsr-inline":
        from jasna.restorer.flashvsr_inline_secondary_restorer import FlashvsrInlineSecondaryRestorer

        if not str(config.flashvsr_repo).strip():
            raise ValueError("--flashvsr-repo is required for --secondary-restoration flashvsr-inline")
        fv_repo = Path(str(config.flashvsr_repo)).expanduser()
        if not fv_repo.is_dir():
            raise FileNotFoundError(f"--flashvsr-repo not found: {fv_repo}")
        from jasna.restorer.flashvsr_offline import default_flashvsr_python

        py_arg = str(config.flashvsr_python).strip()
        fv_py = Path(py_arg).expanduser() if py_arg else default_flashvsr_python(fv_repo)
        if not fv_py.exists():
            raise FileNotFoundError(
                f"FlashVSR Python not found: {fv_py}. Pass --flashvsr-python. Its base "
                "Python must ship the dev headers the Triton JIT needs (a uv-managed "
                "standalone Python, or a system Python with its -dev package)."
            )
        md_arg = str(config.flashvsr_model_dir).strip()
        fv_model = Path(md_arg).expanduser() if md_arg else fv_repo / "models" / "FlashVSR-v1.1"
        if not fv_model.is_dir():
            raise FileNotFoundError(f"--flashvsr-model-dir not found: {fv_model}")
        return FlashvsrInlineSecondaryRestorer(
            repo=fv_repo,
            model_dir=fv_model,
            fv_python=fv_py,
            version=str(config.flashvsr_version),
            dtype=str(config.flashvsr_dtype),
            device=config.device,
            tiles=int(config.flashvsr_tiles),
            scale=int(config.flashvsr_scale),
            accel=bool(config.flashvsr_accel),
            lora_path=str(config.flashvsr_lora) or None,
            log_level=str(config.flashvsr_log_level),
        )
    if config.secondary_restoration == "swiftvr-inline":
        from jasna.restorer.swiftvr_common import resolve_swiftvr_paths
        from jasna.restorer.swiftvr_inline_secondary_restorer import SwiftvrInlineSecondaryRestorer

        sv_repo, sv_py, sv_model = resolve_swiftvr_paths(
            config.swiftvr_repo, config.swiftvr_python, config.swiftvr_model_dir,
            mode="swiftvr-inline",
        )
        return SwiftvrInlineSecondaryRestorer(
            repo=sv_repo,
            model_dir=sv_model,
            sv_python=sv_py,
            device=config.device,
            scale=int(config.swiftvr_scale),
            accel=bool(config.swiftvr_accel),
            view_window=int(config.swiftvr_view_window),
            log_level=str(config.swiftvr_log_level),
        )
    if config.secondary_restoration == "swiftvr-distill":
        from jasna.restorer.swiftvr_distill_model import resolve_swiftvr_distill_model_path
        from jasna.restorer.swiftvr_distill_secondary_restorer import SwiftvrDistillSecondaryRestorer

        return SwiftvrDistillSecondaryRestorer(
            model_path=resolve_swiftvr_distill_model_path(config.swiftvr_distill_model),
            device=device,
            view_window=int(config.swiftvr_distill_view_window),
            strength=float(config.swiftvr_distill_strength),
            stabilize_radius=int(config.swiftvr_distill_stabilize),
        )
    if config.secondary_restoration == "relayvsr":
        from jasna.restorer.relayvsr_window_restorer import RelayvsrWindowRestorer

        _, rv_python, weights_dir = _relayvsr_paths(config)
        return RelayvsrWindowRestorer(
            rv_python=rv_python,
            weights_dir=weights_dir,
            device=config.device,
            window_size=int(config.relayvsr_window_size),
            wan_weights=config.relayvsr_wan_weights,
            kv_cache=config.relayvsr_kv_cache,
            kv_cache_dtype=config.relayvsr_kv_cache_dtype,
        )
    raise ValueError(f"Unsupported secondary restoration: {config.secondary_restoration}")


def _relayvsr_paths(config: SessionConfig) -> tuple[Path, Path, Path]:
    """``(repo, python, weights_dir)`` of ``--secondary-restoration relayvsr``, with the 2x pair
    of checkpoints an LTX window needs in place."""
    from jasna.restorer.relayvsr_common import relayvsr_checkpoints, resolve_relayvsr_paths
    from jasna.restorer.relayvsr_window_restorer import RELAYVSR_WINDOW_SCALE

    paths = resolve_relayvsr_paths(
        config.relayvsr_repo, config.relayvsr_python, config.relayvsr_weights_dir,
        option="--secondary-restoration relayvsr",
    )
    relayvsr_checkpoints(paths[2], RELAYVSR_WINDOW_SCALE)
    return paths


def frame_upscale_size(config: SessionConfig, width: int, height: int) -> "Size | None":
    """The size offline exports of a ``width`` x ``height`` video are upscaled to, or ``None``."""
    from jasna.upscale import upscaled_size

    if config.upscale_model == "none":
        return None
    return upscaled_size(width, height, config.upscale_height)


def ensure_frame_upscale_engines(
    config: SessionConfig,
    source_size: "Size",
    output_size: "Size",
    *,
    log_callback: Callable[[str], None] | None,
) -> None:
    """Compile the TensorRT engines a network upscaler needs for this source size (NVIDIA)."""
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.upscale.network import UPSCALE_NETWORKS, network_input_sizes

    network = UPSCALE_NETWORKS.get(config.upscale_model)
    if network is None:
        return
    ensure_engines_compiled(
        EngineCompilationRequest(
            device=config.device,
            fp16=bool(config.fp16),
            upscale_model=config.upscale_model,
            upscale_input_sizes=list(network_input_sizes(network, source_size, output_size)),
        ),
        log_callback=log_callback,
    )


def build_frame_upscaler(
    config: SessionConfig, device: "torch.device", source_size: "Size", output_size: "Size"
) -> "FrameUpscaler":
    if config.upscale_model == "dlss5":
        from jasna.upscale.dlss5 import Dlss5FrameUpscaler, dlss5_runtime_dir

        return Dlss5FrameUpscaler(
            device,
            source_size=source_size,
            output_size=output_size,
            preset=config.upscale_dlss5_preset,
            style=config.upscale_dlss5_style,
            intensity=config.upscale_dlss5_intensity,
            strength=config.upscale_strength,
            runtime_dir=dlss5_runtime_dir(),
        )
    if config.upscale_model == "rtx-super-res":
        from jasna.upscale.rtx import RtxFrameUpscaler

        return RtxFrameUpscaler(
            device,
            source_size=source_size,
            output_size=output_size,
            quality=config.upscale_rtx_quality,
            denoise=config.upscale_rtx_denoise,
            deblur=config.upscale_rtx_deblur,
            strength=config.upscale_strength,
        )
    from jasna.upscale.network import UPSCALE_NETWORKS, NetworkFrameUpscaler

    if config.upscale_model in UPSCALE_NETWORKS:
        return NetworkFrameUpscaler(
            UPSCALE_NETWORKS[config.upscale_model],
            device,
            source_size=source_size,
            output_size=output_size,
            fp16=bool(config.fp16),
            strength=config.upscale_strength,
        )
    raise ValueError(f"Unsupported upscale model: {config.upscale_model}")


def worker_upscale_writer(
    config: SessionConfig, session: RestorationSession, writer, source_size: "Size", output_size: "Size"
):
    """+modi: ``writer`` behind ``--upscale swiftvr`` (overlapping windows, crossfaded) or
    ``relayvsr`` (a stream with a delay). The worker starts for the job's first upscaled video and
    stays in ``session`` for the ones after it; one a failed video left unusable is replaced."""
    from jasna.session_config import UPSCALE_SWIFTVR_WINDOW_OVERLAP

    device, strength = session.device, config.upscale_strength
    if session.upscale_worker is not None and not session.upscale_worker.alive:
        session.close_upscale_worker()
    verbose = str(config.swiftvr_log_level).lower() == "debug"
    if config.upscale_model == "swiftvr":
        from jasna.restorer.swiftvr_common import resolve_swiftvr_paths
        from jasna.upscale.clip_writer import ClipUpscaleWriter
        from jasna.upscale.swiftvr import SwiftvrClipUpscaler, SwiftvrWorker, swiftvr_scale

        if session.upscale_worker is None:
            repo, sv_python, model_dir = resolve_swiftvr_paths(
                config.swiftvr_repo, config.swiftvr_python, config.swiftvr_model_dir, option="--upscale swiftvr"
            )
            session.upscale_worker = SwiftvrWorker(
                repo=repo, sv_python=sv_python, model_dir=model_dir, device=str(device),
                warmup_size=source_size, warmup_scale=swiftvr_scale(source_size[1], output_size[1]),
                accel=bool(config.swiftvr_accel), tile_megapixels=float(config.upscale_swiftvr_tile),
                verbose=verbose,
            )
        upscaler = SwiftvrClipUpscaler(
            session.upscale_worker, device, source_size=source_size, output_size=output_size, strength=strength
        )
        return ClipUpscaleWriter(
            writer, upscaler, window=int(config.upscale_swiftvr_window), overlap=UPSCALE_SWIFTVR_WINDOW_OVERLAP
        )
    from jasna.restorer.relayvsr_common import resolve_relayvsr_paths
    from jasna.upscale.relayvsr import HEAVY_MODEL_HEIGHT, RelayvsrStreamUpscaler, RelayvsrWorker, relayvsr_scale
    from jasna.upscale.stream_writer import StreamUpscaleWriter

    scale = relayvsr_scale(source_size[1], output_size[1])
    if source_size[1] * scale > HEAVY_MODEL_HEIGHT:
        logger.warning(
            "RelayVSR generates %dx%d (%dx) and the result is resized to %dx%d: far more memory and time "
            "than %dp output", source_size[0] * scale, source_size[1] * scale, scale, *output_size, HEAVY_MODEL_HEIGHT,
        )
    kv_cache = str(config.relayvsr_kv_cache)
    if kv_cache == "auto" and config.restoration_model_name == "ltx" and device.type == "cuda":
        import torch

        if torch.cuda.get_device_properties(device).total_memory < WORKER_LARGE_CANVAS_MIN_BYTES:
            # The worker decides when the video begins, where it cannot see the VRAM the LTX window
            # decodes take later: a cache placed on the GPU then ran the card out of memory (16 GB).
            kv_cache = "cpu"
            logger.info("RelayVSR K/V cache in host memory beside LTX (--relayvsr-kv-cache gpu forces the GPU)")
    if session.upscale_worker is None:
        _, rv_python, weights_dir = resolve_relayvsr_paths(
            config.relayvsr_repo, config.relayvsr_python, config.relayvsr_weights_dir
        )
        session.upscale_worker = RelayvsrWorker(
            rv_python=rv_python, weights_dir=weights_dir, device=str(device), scale=scale,
            window_size=int(config.relayvsr_window_size), wan_weights=str(config.relayvsr_wan_weights),
            kv_cache=kv_cache, kv_cache_dtype=str(config.relayvsr_kv_cache_dtype),
            verbose=verbose,
        )
    upscaler = RelayvsrStreamUpscaler(
        session.upscale_worker, device, source_size=source_size, output_size=output_size, strength=strength
    )
    return StreamUpscaleWriter(writer, upscaler)


def worker_upscale_ltx_warning(config: SessionConfig, device: "torch.device", source_size: "Size", output_size: "Size") -> str | None:
    """+modi: what to tell the user before an LTX job whose RelayVSR upscale is known not to fit:
    on a 16 GB card the 4x pair (2560x1920 from 480p) ran out of memory in the compose pass,
    after the whole denoise."""
    if config.upscale_model != "relayvsr" or config.restoration_model_name != "ltx" or device.type != "cuda":
        return None
    import torch

    from jasna.upscale.relayvsr import relayvsr_scale

    scale = relayvsr_scale(source_size[1], output_size[1])
    if scale < 4 or torch.cuda.get_device_properties(device).total_memory >= WORKER_LARGE_CANVAS_MIN_BYTES:
        return None
    return (
        f"--upscale relayvsr generates {source_size[0] * scale}x{source_size[1] * scale} (4x) beside LTX: on a "
        "16 GB card this ran out of memory in the compose pass. A lower --upscale-height (2x), --upscale swiftvr "
        "or a second pass with jasna-postprocess fits."
    )


def secondary_restorations_for(vendor: "AcceleratorVendor") -> tuple[SecondaryRestorationName, ...]:
    """The secondary restorers a GPU of ``vendor`` can run."""
    from jasna.accelerator import AcceleratorVendor

    if vendor is AcceleratorVendor.NVIDIA:
        return get_args(SecondaryRestorationName)
    if vendor is AcceleratorVendor.AMD:
        return ("none", "unet-4x")
    return ("none",)


def upscale_models_for(vendor: "AcceleratorVendor", platform: str) -> tuple[UpscaleModelName, ...]:
    """The whole-video upscalers a GPU of ``vendor`` can run on ``platform``: the AI networks on
    NVIDIA and AMD, RTX Super Res only on NVIDIA, DLSS 5 only on NVIDIA under Windows."""
    from jasna.accelerator import AcceleratorVendor

    if vendor is AcceleratorVendor.NVIDIA:
        return ("none", "span", "realplksr", "rtx-super-res", *(("dlss5",) if platform == "win32" else ()))
    if vendor is AcceleratorVendor.AMD:
        return ("none", "span", "realplksr")
    return ("none",)


def _build_seedvr2_restorer(config: SessionConfig, device: "torch.device"):
    from jasna.restorer.seedvr2_lora_restorer import Seedvr2LoraRestorer, default_seedvr2_python

    if not str(config.seedvr2_repo).strip():
        raise ValueError("--seedvr2-repo is required for --restoration-model-name seedvr2")
    repo = Path(str(config.seedvr2_repo)).expanduser()
    if not repo.is_dir():
        raise FileNotFoundError(f"--seedvr2-repo not found: {repo}")

    py_arg = str(config.seedvr2_python).strip()
    py = Path(py_arg).expanduser() if py_arg else default_seedvr2_python(repo)
    if not py.exists():
        raise FileNotFoundError(f"SeedVR2 Python not found: {py}. Pass --seedvr2-python.")

    return Seedvr2LoraRestorer(
        repo_path=str(repo),
        lora_path=str(config.restoration_model_path),
        device=device,
        python_path=str(py),
        model_dir=str(config.seedvr2_model_dir).strip() or None,
        dit_model=str(config.seedvr2_dit),
        lora_rank=int(config.seedvr2_lora_rank),
        window=int(config.seedvr2_window),
        overlap=int(config.seedvr2_overlap),
        color_fix=str(config.seedvr2_color_fix),
        empty_cache=str(config.seedvr2_empty_cache),
        accel=bool(config.seedvr2_accel),
    )


def build_restoration_session(
    config: SessionConfig,
    *,
    log_callback: Callable[[str], None] | None,
) -> RestorationSession:
    import torch

    from jasna.accelerator import is_amd_device, log_amd_backend, supports_frame_generation, vendor_for_device

    device = torch.device(config.device)
    if is_amd_device(device):
        log_amd_backend(device)
    if config.frame_generation != "none" and config.frame_generation_backend == "rtx" and not supports_frame_generation(device):
        raise ValueError(FRAME_GENERATION_NEEDS_RTX_40)
    if config.tvai_denoise and config.secondary_restoration != "tvai":
        raise ValueError("TVAI Denoise requires secondary restoration 'tvai'")
    vendor = vendor_for_device(device)
    if config.secondary_restoration not in secondary_restorations_for(vendor):
        raise ValueError(
            f"Secondary restoration '{config.secondary_restoration}' is not available in the AMD build yet"
        )
    if config.upscale_model == "dlss5":
        from jasna.upscale.dlss5 import dlss5_runtime_dir, dlss5_unavailable_reason

        reason = dlss5_unavailable_reason(device, dlss5_runtime_dir())
        if reason:
            raise ValueError(reason)
    elif config.upscale_model in WORKER_UPSCALE_MODELS:  # +modi
        from jasna.accelerator import AcceleratorVendor

        if vendor is not AcceleratorVendor.NVIDIA:
            raise ValueError(f"--upscale {config.upscale_model} needs an NVIDIA GPU")
    elif config.upscale_model not in upscale_models_for(vendor, sys.platform):
        raise ValueError(UPSCALE_RTX_NEEDS_NVIDIA)
    if config.restoration_model_name == "ltx" and ltx_unsupported_options(
        auto_split=config.ltx_auto_split_percent is not None,
        secondary_restoration=config.secondary_restoration,
        denoise_strength=config.denoise_strength,
    ):
        raise ValueError(
            "LTX restoration supports Unet 4x, TVAI or denoise only with LTX auto split, "
            "where they apply to the BasicVSR++ parts"
        )
    if (
        config.restoration_model_name == "ltx"
        and config.secondary_restoration == "rtx-super-res"
        and not nvvfx_installed()
    ):
        # +modi: refused before anything loads, not when compose first upscales.
        raise ValueError(
            "--secondary-restoration rtx-super-res with LTX needs the RTX Video Effects runtime (the "
            "nvidia-vfx package of the nvidia extra), which is not installed"
        )
    if restores_ltx_windows(config):
        # +modi: a wrong path fails here, not when the compose pass starts.
        _validate_window_restorer(ltx_window_restorer_config(config))
        _check_worker_beside_large_canvas(config, device)
    session = RestorationSession(device=device, restoration_pipeline=None, log_callback=log_callback)
    if config.ltx_crop_vsr is not None:
        # The input is an LTX output: only the detector (for the source's scan) and,
        # after the scan, the SwiftVR restorer run. BasicVSR++ neither loads nor compiles.
        from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled

        validate_ltx_crop_vsr(config)
        ensure_engines_compiled(
            EngineCompilationRequest(
                device=str(device),
                fp16=bool(config.fp16),
                detection=True,
                detection_model_name=config.detection_model_name,
                detection_model_path=str(config.detection_model_path),
                detection_batch_size=int(config.batch_size),
            ),
            log_callback=log_callback,
        )
        return session
    provide_restoration_models(
        config, session, frozenset({config.restoration_model_name}), log_callback=log_callback
    )
    return session


def ltx_crop_vsr_config(config: SessionConfig) -> SessionConfig:
    """The secondary-restorer settings ``--ltx-crop-vsr`` runs with: SwiftVR at scale 2
    through the inline worker, or the distill student, both without the crop-view
    smoothing (the LTX camera is smooth already, and the crops never go through the
    frame pipeline), or RTX Super Res at scale 2 with the other ``--rtx-*`` options."""
    if config.ltx_crop_vsr == "swiftvr":
        return replace(config, secondary_restoration="swiftvr-inline", swiftvr_scale=2, swiftvr_view_window=0)
    if config.ltx_crop_vsr == "swiftvr-distill":
        return replace(config, secondary_restoration="swiftvr-distill", swiftvr_distill_view_window=0)
    if config.ltx_crop_vsr == "rtx-super-res":
        return replace(config, secondary_restoration="rtx-super-res", rtx_scale=2)
    raise ValueError(f"Unsupported --ltx-crop-vsr backend: {config.ltx_crop_vsr}")


def ltx_window_restorer_config(config: SessionConfig) -> SessionConfig:
    """The settings a temporal secondary restorer runs with on the windows of an LTX job's
    compose pass: scale 2 and no crop-view smoothing, as under ``--ltx-crop-vsr``."""
    if config.secondary_restoration == "swiftvr-inline":
        return replace(config, swiftvr_scale=2, swiftvr_view_window=0)
    if config.secondary_restoration == "swiftvr-distill":
        return replace(config, swiftvr_distill_view_window=0)
    if config.secondary_restoration == "relayvsr":
        return config
    raise ValueError(f"Unsupported LTX window restorer: {config.secondary_restoration}")


def restores_ltx_windows(config: SessionConfig) -> bool:
    """Whether the job's secondary restoration runs on LTX windows in the compose pass. With
    auto split it stays with the BasicVSR++ parts, whose frame pipeline holds the restorer."""
    return (
        config.restoration_model_name == "ltx"
        and config.ltx_auto_split_percent is None
        and config.ltx_crop_vsr is None
        and config.secondary_restoration in LTX_WINDOW_RESTORERS
    )


WORKER_LARGE_CANVAS_MIN_BYTES = 20 << 30
"""GPU memory a restorer in a worker of its own (SwiftVR, RelayVSR) needs beside the decode of a
768 px LTX window: on a 16 GB card the decode or the worker ran out of memory (conv VAE, RTX 5080;
the SwiftVR worker holds 5.5 GB between windows, the RelayVSR worker 5.0 GB)."""


def _check_worker_beside_large_canvas(config: SessionConfig, device: "torch.device") -> None:
    worker_restorers = ("swiftvr-inline", "relayvsr")
    if config.secondary_restoration not in worker_restorers or not config.ltx_large_canvas or device.type != "cuda":
        return
    import torch

    total = torch.cuda.get_device_properties(device).total_memory
    if total < WORKER_LARGE_CANVAS_MIN_BYTES:
        raise ValueError(
            f"--secondary-restoration {config.secondary_restoration} with an LTX model and --ltx-large-canvas "
            f"needs a GPU with {WORKER_LARGE_CANVAS_MIN_BYTES >> 30} GB or more ({total / 2**30:.0f} GB here): "
            "its worker does not fit beside the decode of a 768 px window. Drop --ltx-large-canvas or use "
            "swiftvr-distill."
        )


def validate_ltx_crop_vsr(config: SessionConfig) -> None:
    """Resolve the restorer's files now, before the scan, so a wrong path (or a missing
    runtime) fails early."""
    _validate_window_restorer(ltx_crop_vsr_config(config))


def _validate_window_restorer(refined: SessionConfig) -> None:
    if refined.secondary_restoration == "swiftvr-inline":
        from jasna.restorer.swiftvr_common import resolve_swiftvr_paths

        resolve_swiftvr_paths(
            refined.swiftvr_repo, refined.swiftvr_python, refined.swiftvr_model_dir, mode="swiftvr-inline"
        )
    elif refined.secondary_restoration == "relayvsr":
        _relayvsr_paths(refined)
    elif refined.secondary_restoration == "rtx-super-res":
        if not nvvfx_installed():
            raise ValueError(
                "--ltx-crop-vsr rtx-super-res needs the RTX Video Effects runtime (the nvidia-vfx package "
                "of the nvidia extra), which is not installed"
            )
    else:
        from jasna.restorer.swiftvr_distill_model import resolve_swiftvr_distill_model_path

        resolve_swiftvr_distill_model_path(refined.swiftvr_distill_model)


def build_ltx_crop_vsr_restorer(config: SessionConfig, device: "torch.device"):
    """The x2 restorer of ``--ltx-crop-vsr``; built after the scan so it never shares
    the GPU with the detector. It is sized from the canvases it is given (512 or 768 px,
    both in one video when ``--ltx-large-canvas`` is on)."""
    return _build_secondary_restorer(ltx_crop_vsr_config(config), device, size_from_frames=True)


def provide_restoration_models(
    config: SessionConfig,
    session: RestorationSession,
    models: frozenset[RestorationModelName],
    *,
    log_callback: Callable[[str], None] | None,
) -> None:
    """Load each of ``models`` the session does not hold yet."""
    if "basicvsrpp" in models and session.restoration_pipeline is None:
        session.restoration_pipeline = _build_basicvsrpp_pipeline(config, session.device, log_callback=log_callback)
    if "seedvr2" in models and session.restoration_pipeline is None:
        session.restoration_pipeline = _build_basicvsrpp_pipeline(
            config, session.device, log_callback=log_callback, use_seedvr2=True
        )
    if "ltx" in models and session.ltx_files is None:
        session.ltx_files = _ltx_model_files(config, session.device)
    if "ltx" in models and config.secondary_restoration == "rtx-super-res" and session.ltx_super_res is None:
        from jasna.ltx.super_res import LtxSuperRes

        session.ltx_super_res = LtxSuperRes(
            device=session.device,
            quality=config.rtx_quality,
            denoise=config.rtx_denoise,
            deblur=config.rtx_deblur,
            strength=config.rtx_strength,
            highbitrate=config.rtx_highbitrate,
        )


    if "ltx" in models and restores_ltx_windows(config) and session.ltx_super_res is None:
        from jasna.ltx.window_restorer import LtxWindowRestorer

        refined, device = ltx_window_restorer_config(config), session.device
        session.ltx_super_res = LtxWindowRestorer(
            config.secondary_restoration,
            lambda: _build_secondary_restorer(refined, device, size_from_frames=True),
        )


def _build_basicvsrpp_pipeline(
    config: SessionConfig,
    device: "torch.device",
    *,
    log_callback: Callable[[str], None] | None,
    use_seedvr2: bool = False,
) -> "RestorationPipeline":
    """The frame-pipeline restorer: BasicVSR++, or SeedVR2 in its place (``use_seedvr2``)."""
    from jasna.accelerator import is_amd_device
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.restorer.denoise import DenoiseStrength
    from jasna.restorer.restoration_pipeline import RestorationPipeline

    model_path = config.model_path("seedvr2" if use_seedvr2 else "basicvsrpp")
    compile_result = ensure_engines_compiled(
        EngineCompilationRequest(
            device=str(device),
            fp16=bool(config.fp16),
            # For seedvr2, the model path is the LoRA checkpoint and BasicVSR++
            # never loads, so its engine compilation is skipped entirely.
            basicvsrpp=config.compile_basicvsrpp and not is_amd_device(device) and not use_seedvr2,
            basicvsrpp_model_path=str(model_path),
            detection=True,
            detection_model_name=config.detection_model_name,
            detection_model_path=str(config.detection_model_path),
            detection_batch_size=int(config.batch_size),
            unet4x=(config.secondary_restoration == "unet-4x"),
        ),
        log_callback=log_callback,
    )
    if use_seedvr2:
        primary_restorer = _build_seedvr2_restorer(config, device)
    else:
        primary_restorer = BasicvsrppMosaicRestorer(
            checkpoint_path=str(model_path),
            device=device,
            max_clip_size=int(config.max_clip_size),
            use_tensorrt=compile_result.use_basicvsrpp_tensorrt,
            fp16=bool(config.fp16),
        )
    return RestorationPipeline(
        restorer=primary_restorer,
        secondary_restorer=_build_secondary_restorer(config, device),
        denoise_strength=DenoiseStrength(config.denoise_strength),
    )


LTX_FAST_NEEDS_BLACKWELL = "The fast LTX model needs an RTX 50-series (Blackwell) GPU"
FRAME_GENERATION_NEEDS_RTX_40 = (
    "Frame generation needs an NVIDIA RTX 40-series or newer GPU for the rtx backend; "
    "--frame-generation-backend rife runs on every GPU"
)
UPSCALE_RTX_NEEDS_NVIDIA = "RTX Super Res upscaling needs an NVIDIA GPU"


def _ltx_model_files(config: SessionConfig, device: "torch.device") -> "LtxModelFiles":
    from jasna.ltx.model_files import LtxModelFiles

    from jasna.accelerator import TRITON_HIP_ARCHITECTURES, amd_gpu_architecture, is_amd_device, supports_nvfp4
    from jasna.ltx.transformer import require_kernels

    if config.ltx_fast and not supports_nvfp4(device):
        raise ValueError(LTX_FAST_NEEDS_BLACKWELL)
    if is_amd_device(device) and (architecture := amd_gpu_architecture(device)) not in TRITON_HIP_ARCHITECTURES:
        raise ValueError(
            f"LTX can't run on this AMD GPU ({architecture}): it needs a Radeon RX 7000 or 9000 series GPU. "
            "Use BasicVSR++ instead."
        )
    require_kernels(device)
    if config.ltx_trial:
        from jasna.ltx.model_files import LTX_TRIAL_NOTICE

        logger.warning(LTX_TRIAL_NOTICE)
        return LtxModelFiles.placeholder(
            config.ltx_model, fast=config.ltx_fast, encoder=config.ltx_vae_encoder, decoder=config.ltx_vae_decoder
        )
    return LtxModelFiles.from_dir(
        config.model_path("ltx"),
        config.ltx_model,
        fast=config.ltx_fast,
        encoder=config.ltx_vae_encoder,
        decoder=config.ltx_vae_decoder,
    )


def segment_job_config(config: SessionConfig, source_codec: str, input_video: Path) -> SessionConfig:
    """A segment job copies the parts it leaves alone, so its output keeps the source codec.
    The CQ and custom encoder settings were chosen for ``config.codec`` (``tune=uhq`` of
    av1_nvenc fails on h264_nvenc), so another codec gets only its own default CQ."""
    if source_codec == config.codec:
        return config
    from jasna.accelerator import vendor_for_device
    from jasna.media.encoder_settings import encoder_cq_spec

    cq = encoder_cq_spec(source_codec, vendor_for_device(config.device)).default
    logger.warning(
        "%s: the output keeps the source codec %s so the parts without changes can be copied; "
        "the CQ and custom encoder settings are for %s and are not used (%s CQ %d)",
        input_video.name,
        source_codec.upper(),
        config.codec.upper(),
        source_codec.upper(),
        cq,
    )
    return replace(config, codec=source_codec, encoder_settings={"cq": cq})


@dataclass(frozen=True)
class SpliceSource:
    """A video that can be restored in parts: its metadata, codec and keyframes."""

    metadata: "VideoMetadata"
    codec: str
    index: "KeyframeIndex"


def splice_source(config: SessionConfig, source: Path, output: Path) -> SpliceSource:
    """Raises ``SmartRenderCompatibilityError`` when ``source`` cannot be restored in parts
    into ``output``."""
    from jasna.media.probe import get_video_meta_data
    from jasna.media.splice import (
        SmartRenderCompatibilityError,
        canonical_codec,
        probe_keyframes,
        validate_smart_render,
    )

    if config.frame_generation != "none":
        # Copied parts keep the source frame rate, so they cannot sit next to doubled ones.
        raise SmartRenderCompatibilityError("Smart rendering cannot be combined with frame generation")
    if config.upscale_model != "none":
        # Copied parts keep the source size, so they cannot sit next to upscaled ones.
        raise SmartRenderCompatibilityError("Smart rendering cannot be combined with upscaling")
    metadata = get_video_meta_data(str(source))
    codec = canonical_codec(metadata.codec_name)
    validate_smart_render(metadata, output_path=output, codec=codec, retarget_high_fps=config.retarget_high_fps)
    return SpliceSource(metadata, codec, probe_keyframes(source, metadata))


@dataclass(frozen=True)
class SegmentJob:
    """A job that restores ranges and copies the rest: its config in the source codec and
    the plan of its ranges."""

    config: SessionConfig
    splice_plan: "SplicePlan"

    @property
    def reencodes_whole_video(self) -> bool:
        """Past the encoder's B-frame cap the ranges cannot be cut out, so the pipeline
        re-encodes the whole video with BasicVSR++ and keeps no finished parts."""
        from jasna.accelerator import vendor_for_device
        from jasna.media.splice import h264_b_frames_over_cap

        return h264_b_frames_over_cap(self.config.codec, self.splice_plan.index, vendor_for_device(self.config.device))


def _require_ltx_cuts(config: SessionConfig, spliced: SpliceSource) -> None:
    """LTX restores ranges only as cut-out parts, which the encoder cannot match past its B-frame cap."""
    from jasna.accelerator import vendor_for_device
    from jasna.media.splice import SmartRenderCompatibilityError, h264_b_frames_over_cap

    if h264_b_frames_over_cap(spliced.codec, spliced.index, vendor_for_device(config.device)):
        raise SmartRenderCompatibilityError(
            f"This H.264 video uses {spliced.index.max_b_frames} consecutive B-frames, more than the GPU encoder "
            "can match, so LTX ranges cannot be cut out of it; restore the ranges with BasicVSR++, "
            "or clear them to restore the whole video with LTX"
        )


def plan_segment_job(
    config: SessionConfig, source: Path, spliced: SpliceSource, segments: "tuple[SegmentRange, ...]"
) -> SegmentJob:
    from jasna.media.splice import build_splice_plan
    from jasna.segments import job_models

    if "ltx" in job_models(config.restoration_model_name, segments):
        _require_ltx_cuts(config, spliced)
    plan = build_splice_plan(segments, spliced.index, duration=spliced.metadata.duration)
    if config.lut_path is not None or config.sharpen_strength > 0:
        logger.info("%s: a color LUT or sharpening is set, so the parts without ranges are encoded again", source.name)
        plan = plan.reencoding_copies()
    job_config = segment_job_config(config, spliced.codec, source)
    if job_config.fmp4:
        logger.warning(
            "Fragmented MP4 is not available with segment processing; the output is assembled after processing finishes"
        )
        job_config = replace(job_config, fmp4=False)
    return SegmentJob(job_config, plan)


def ltx_auto_split(
    config: SessionConfig,
    session: RestorationSession,
    input_video: Path,
    output_video: Path,
    *,
    progress_callback: "ProgressCallback | None",
) -> SegmentJob | None:
    """For an LTX job with ``ltx_auto_split_percent`` and no ranges of its own: scan the
    video, give big mosaics LTX and the rest BasicVSR++, copy the parts without mosaics.
    None when the job does not split, including a video segment processing cannot handle
    or with no mosaics found; those are logged and restored whole with LTX."""
    from jasna.mosaic_scan import ltx_auto_segments, scan_mosaic_sizes
    from jasna.progressbar import StageProgress
    from jasna.vr180 import resolve_vr_mode

    def on_progress(fraction: float) -> None:
        if progress_callback is not None:
            progress_callback(0.0, 0.0, 0.0, 0, 0, StageProgress("split", 1, 1, fraction, 0.0))

    def scan(spliced: SpliceSource) -> "tuple[SegmentRange, ...]":
        stereo = resolve_vr_mode(config.vr_mode, spliced.metadata, input_video).layout(spliced.metadata)
        samples = scan_mosaic_sizes(
            session.detection_model_for(config),
            input_video,
            spliced.metadata,
            stereo=stereo,
            batch_size=config.batch_size,
            device=session.device,
            decode_backend=config.decode_backend,
            disable_progress=config.disable_progress,
            on_progress=on_progress,
        )
        return ltx_auto_segments(
            samples.times,
            samples.scores,
            samples.sizes,
            threshold=config.detection_score_threshold,
            ltx_min_size=config.ltx_auto_split_percent / 100,
            stride=samples.stride,
            duration=spliced.metadata.duration,
            keyframes=spliced.index.keyframe_seconds(),
            ltx_seed=config.ltx_seed,
        )

    return _auto_split_job(config, input_video, output_video, scan)


def plan_job(
    config: SessionConfig,
    session: RestorationSession,
    input_video: Path,
    output_video: Path,
    ranges_job: SegmentJob | None,
    *,
    progress_callback: "ProgressCallback | None",
) -> SegmentJob | None:
    """The job's ranges: the ones the user set win over the LTX auto split."""
    if ranges_job is not None:
        return ranges_job
    return ltx_auto_split(config, session, input_video, output_video, progress_callback=progress_callback)


def _auto_split_job(
    config: SessionConfig,
    input_video: Path,
    output_video: Path,
    scan: "Callable[[SpliceSource], tuple[SegmentRange, ...]] | None",
) -> SegmentJob | None:
    """The auto split job from the saved split, else from ``scan``; with ``scan`` None only
    a saved split counts."""
    from jasna.job_resume import ResumeJob
    from jasna.media.splice import SmartRenderCompatibilityError

    if config.ltx_auto_split_percent is None or config.restoration_model_name != "ltx":
        return None
    try:
        spliced = splice_source(config, input_video, output_video)
        _require_ltx_cuts(config, spliced)
    except SmartRenderCompatibilityError as exc:
        logger.warning("%s: LTX auto split skipped (%s); restoring the whole video with LTX", input_video.name, exc)
        return None
    checkpoint = ResumeJob(config, input_video, output_video, spliced.metadata, None)
    segments = checkpoint.saved_split()
    if segments is not None:
        logger.info("Reusing saved LTX auto split for %s", input_video.name)
    elif scan is None:
        return None
    else:
        segments = scan(spliced)
        checkpoint.save_split(segments)
    if not segments:
        logger.warning("%s: LTX auto split found no mosaics; restoring the whole video with LTX", input_video.name)
        return None
    try:
        job = plan_segment_job(config, input_video, spliced, segments)
    except SmartRenderCompatibilityError as exc:
        logger.warning("%s: LTX auto split skipped (%s); restoring the whole video with LTX", input_video.name, exc)
        return None
    mosaic_seconds = sum(segment.duration for segment in segments)
    ltx_seconds = sum(segment.duration for segment in segments if segment.restoration.model == "ltx")
    logger.info(
        "%s: LTX auto split: %.0f s with mosaics, %.0f s (%.0f%%) of it with LTX",
        input_video.name,
        mosaic_seconds,
        ltx_seconds,
        ltx_seconds / mosaic_seconds * 100,
    )
    return job


def find_resume_output(
    config: SessionConfig, source: Path, output: Path, segments: "tuple[SegmentRange, ...]"
) -> Path | None:
    """The output whose saved work continues this job, so it wins over GUI file conflicts
    (idea from #415), including an earlier auto-rename. The job is derived as a run derives
    it; ranges that cannot be restored in parts raise as the run would."""
    from jasna.job_resume import ResumeJob, ResumeMismatchError, saved_outputs
    from jasna.media.probe import get_video_meta_data

    candidates = saved_outputs(config, output)
    if not candidates:
        return None
    metadata = get_video_meta_data(str(source))
    for saved in candidates:
        try:
            job = (
                plan_segment_job(config, source, splice_source(config, source, saved), segments) if segments
                else _auto_split_job(config, source, saved, None)
            )
        except ResumeMismatchError:
            continue
        checkpoint = (
            ResumeJob(config, source, saved, metadata, None) if job is None
            else ResumeJob(job.config, source, saved, metadata, job.splice_plan)
        )
        if checkpoint.matches():
            return saved
    return None


def build_pipeline(
    config: SessionConfig,
    session: RestorationSession,
    input_video: Path,
    output_video: Path,
    *,
    progress_callback: Callable | None = None,
    segments: "tuple[SegmentRange, ...] | None" = None,
    splice_plan: "SplicePlan | None" = None,
    video_decode_backend=None,
    video_encode_backend=None,
) -> "Pipeline":
    """A per-video ``Pipeline``; the session first loads any model the segments ask for."""
    from jasna.media.backend import VideoBackend
    from jasna.pipeline import Pipeline
    from jasna.segments import job_models

    if config.ltx_crop_vsr is None:
        provide_restoration_models(
            config, session, job_models(config.restoration_model_name, segments), log_callback=None
        )

    if video_decode_backend is None:
        video_decode_backend = VideoBackend.NATIVE
    if video_encode_backend is None:
        video_encode_backend = VideoBackend.NATIVE

    return Pipeline(
        config=config,
        session=session,
        input_video=input_video,
        output_video=output_video,
        progress_callback=progress_callback,
        segments=segments,
        splice_plan=splice_plan,
        video_decode_backend=video_decode_backend,
        video_encode_backend=video_encode_backend,
    )


def build_job_pipeline(
    config: SessionConfig,
    session: RestorationSession,
    input_video: Path,
    output_video: Path,
    segment_job: SegmentJob | None,
    *,
    progress_callback: Callable | None,
    video_decode_backend=None,
    video_encode_backend=None,
) -> "Pipeline":
    """The pipeline of a job: ``segment_job`` when it restores ranges, else the whole video."""
    if segment_job is None:
        return build_pipeline(
            config,
            session,
            input_video,
            output_video,
            progress_callback=progress_callback,
            video_decode_backend=video_decode_backend,
            video_encode_backend=video_encode_backend,
        )
    return build_pipeline(
        segment_job.config,
        session,
        input_video,
        output_video,
        progress_callback=progress_callback,
        segments=segment_job.splice_plan.segments,
        splice_plan=segment_job.splice_plan,
        video_decode_backend=video_decode_backend,
        video_encode_backend=video_encode_backend,
    )

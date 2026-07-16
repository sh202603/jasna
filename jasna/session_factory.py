"""Shared composition root for the video restoration pipeline.

Builds the heavy restoration session (engine compilation, primary and
secondary restorers) and per-video ``Pipeline`` instances from one
``SessionConfig``. Consumed by both the CLI (``jasna.main``) and the GUI
(``jasna.gui.video_session`` / ``jasna.gui.processor``).

All heavy imports (torch, restorers, pipeline) stay inside the functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from jasna.session_config import SessionConfig

if TYPE_CHECKING:
    import torch

    from jasna.media.splice import SplicePlan
    from jasna.pipeline import Pipeline
    from jasna.restorer.restoration_pipeline import RestorationPipeline
    from jasna.restorer.secondary_restorer import SecondaryRestorer
    from jasna.segments import SegmentRange


@dataclass
class RestorationSession:
    device: "torch.device"
    detection_model_name: str
    detection_model_path: Path
    restoration_pipeline: "RestorationPipeline"
    secondary_restorer: "SecondaryRestorer | None"

    def close(self) -> None:
        self.restoration_pipeline.restorer.close()
        if self.secondary_restorer is not None and hasattr(self.secondary_restorer, "close"):
            self.secondary_restorer.close()


def _build_secondary_restorer(config: SessionConfig, device: "torch.device"):
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
            denoise=None if config.rtx_denoise == "none" else config.rtx_denoise,
            deblur=None if config.rtx_deblur == "none" else config.rtx_deblur,
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
                f"FlashVSR Python not found: {fv_py}. Pass --flashvsr-python. It must be a "
                "uv-managed standalone Python venv (system Python lacks the dev headers Triton "
                "JIT needs)."
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
            log_level=str(config.flashvsr_log_level),
        )
    raise ValueError(f"Unsupported secondary restoration: {config.secondary_restoration}")


def build_restoration_session(
    config: SessionConfig,
    *,
    disable_basicvsrpp_tensorrt: bool,
    log_callback: Callable[[str], None] | None,
) -> RestorationSession:
    import torch

    from jasna.accelerator import is_amd_device
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.restorer.denoise import DenoiseStep, DenoiseStrength
    from jasna.restorer.restoration_pipeline import RestorationPipeline

    device = torch.device(config.device)
    amd = is_amd_device(device)
    if config.tvai_denoise and config.secondary_restoration != "tvai":
        raise ValueError("TVAI Denoise requires secondary restoration 'tvai'")
    if amd and config.secondary_restoration != "none":
        raise ValueError(
            f"Secondary restoration '{config.secondary_restoration}' is not available in the AMD build yet"
        )

    compile_result = ensure_engines_compiled(
        EngineCompilationRequest(
            device=str(device),
            fp16=bool(config.fp16),
            basicvsrpp=bool(config.compile_basicvsrpp) and not disable_basicvsrpp_tensorrt and not amd,
            basicvsrpp_model_path=str(config.restoration_model_path),
            detection=True,
            detection_model_name=config.detection_model_name,
            detection_model_path=str(config.detection_model_path),
            detection_batch_size=int(config.batch_size),
            unet4x=(config.secondary_restoration == "unet-4x"),
        ),
        log_callback=log_callback,
    )

    secondary_restorer = _build_secondary_restorer(config, device)
    restoration_pipeline = RestorationPipeline(
        restorer=BasicvsrppMosaicRestorer(
            checkpoint_path=str(config.restoration_model_path),
            device=device,
            max_clip_size=int(config.max_clip_size),
            use_tensorrt=compile_result.use_basicvsrpp_tensorrt,
            fp16=bool(config.fp16),
        ),
        secondary_restorer=secondary_restorer,
        denoise_strength=DenoiseStrength(config.denoise_strength),
        denoise_step=DenoiseStep(config.denoise_step),
    )

    return RestorationSession(
        device=device,
        detection_model_name=config.detection_model_name,
        detection_model_path=Path(config.detection_model_path),
        restoration_pipeline=restoration_pipeline,
        secondary_restorer=secondary_restorer,
    )


def build_pipeline(
    config: SessionConfig,
    session: RestorationSession,
    input_video: Path,
    output_video: Path,
    *,
    progress_callback: Callable | None = None,
    segments: "tuple[SegmentRange, ...] | None" = None,
    splice_plan: "SplicePlan | None" = None,
    frame_gen_multiplier: int = 1,
    frame_generator=None,
    decode_backend=None,
    encode_backend=None,
) -> "Pipeline":
    from jasna.pipeline import Pipeline
    from jasna.media.backend import VideoBackend

    if decode_backend is None:
        decode_backend = VideoBackend.NATIVE
    if encode_backend is None:
        encode_backend = VideoBackend.NATIVE
    return Pipeline(
        input_video=input_video,
        output_video=output_video,
        detection_model_name=config.detection_model_name,
        detection_model_path=config.detection_model_path,
        detection_score_threshold=float(config.detection_score_threshold),
        restoration_pipeline=session.restoration_pipeline,
        codec=config.codec,
        encoder_settings=dict(config.encoder_settings),
        batch_size=int(config.batch_size),
        device=session.device,
        max_clip_size=int(config.max_clip_size),
        temporal_overlap=int(config.temporal_overlap),
        max_detection_gap=int(config.max_detection_gap),
        min_detection_duration=int(config.min_detection_duration),
        enable_crossfade=bool(config.enable_crossfade),
        scene_detection=bool(config.scene_detection),
        vr_mode=config.vr_mode,
        vr_projection=config.vr_projection,
        fp16=bool(config.fp16),
        disable_progress=bool(config.disable_progress),
        progress_callback=progress_callback,
        lut_path=config.lut_path,
        sharpen_strength=config.sharpen_strength,
        retarget_high_fps=bool(config.retarget_high_fps),
        fmp4=bool(config.fmp4),
        segments=segments,
        splice_plan=splice_plan,
        working_dir=config.working_dir,
        frame_gen_multiplier=frame_gen_multiplier,
        frame_generator=frame_generator,
        decode_backend=decode_backend,
        encode_backend=encode_backend,
    )

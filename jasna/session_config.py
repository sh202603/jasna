"""Core-owned typed settings model for pipeline composition.

Holds every parameter the shared composition root (``jasna.session_factory``)
needs. CLI (``jasna.main``) and GUI (``jasna.gui.video_session``) each map
their own settings representation into this one model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

RestorationModelName = Literal["basicvsrpp", "seedvr2"]
SecondaryRestorationName = Literal["none", "unet-4x", "tvai", "rtx-super-res", "flashvsr-inline"]
DenoiseStrengthName = Literal["none", "low", "medium", "high"]
DenoiseStepName = Literal["after_primary", "after_secondary"]
VrModeName = Literal["auto", "off", "sbs", "sbs-fisheye"]
VrProjectionName = Literal["auto", "raw", "fisheye", "gnomonic"]
RtxQualityName = Literal["low", "medium", "high", "ultra"]
RtxLevelName = Literal["none", "low", "medium", "high", "ultra"]
CodecName = Literal["hevc", "h264", "av1"]


@dataclass(frozen=True)
class SessionConfig:
    device: str
    fp16: bool
    batch_size: int
    detection_model_name: str
    detection_model_path: Path
    detection_score_threshold: float
    max_detection_gap: int
    min_detection_duration: int
    scene_detection: bool
    restoration_model_path: Path
    compile_basicvsrpp: bool
    max_clip_size: int
    temporal_overlap: int
    enable_crossfade: bool
    denoise_strength: DenoiseStrengthName
    denoise_step: DenoiseStepName
    secondary_restoration: SecondaryRestorationName
    tvai_ffmpeg_path: str
    tvai_model: str
    tvai_scale: int
    tvai_args: str
    tvai_workers: int
    rtx_scale: int
    rtx_quality: RtxQualityName
    rtx_denoise: RtxLevelName
    rtx_deblur: RtxLevelName
    vr_mode: VrModeName
    codec: CodecName
    encoder_settings: Mapping[str, object]
    lut_path: str | None
    retarget_high_fps: bool
    disable_progress: bool
    working_dir: Path | None
    vr_projection: VrProjectionName = "auto"
    fmp4: bool = False
    sharpen_strength: float = 0.0
    tvai_denoise: bool = False
    flashvsr_repo: str = ""
    flashvsr_python: str = ""
    flashvsr_model_dir: str = ""
    flashvsr_version: str = "11"
    flashvsr_dtype: str = "bf16"
    flashvsr_tiles: int = 1
    flashvsr_log_level: str = "error"
    # Primary restoration model. For "seedvr2", ``restoration_model_path``
    # carries the LoRA checkpoint instead of the BasicVSR++ checkpoint.
    restoration_model_name: RestorationModelName = "basicvsrpp"
    seedvr2_repo: str = ""
    seedvr2_python: str = ""
    seedvr2_model_dir: str = ""
    seedvr2_dit: str = "seedvr2_ema_3b_fp16.safetensors"
    seedvr2_lora_rank: int = 16
    seedvr2_window: int = 33
    seedvr2_overlap: int = 9
    seedvr2_color_fix: str = "lab"
    seedvr2_empty_cache: str = "auto"

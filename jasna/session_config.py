"""Core-owned typed settings model for pipeline composition.

Holds every parameter the shared composition root (``jasna.session_factory``)
needs. CLI (``jasna.main``) and GUI (``jasna.gui.video_session``) each map
their own settings representation into this one model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

RestorationModelName = Literal["basicvsrpp", "ltx", "seedvr2"]
LtxModelName = Literal["distilled1", "distilled2", "distilled4", "distilled", "undistilled"]
LTX_DEFAULT_MODEL: LtxModelName = "distilled2"

LTX_DEFAULT_SEED = 20260923
LTX_AUTO_DEFAULT_PERCENT = 12
# Frames two neighbouring LTX windows share: the hop stays whole latents and leaves room for
# the crossfade after a window's unfused lead frames.
LTX_OVERLAP_CHOICES = (17, 25, 33, 41, 49)
LTX_DEFAULT_OVERLAP = 17
SecondaryRestorationName = Literal[
    "none", "unet-4x", "tvai", "rtx-super-res", "flashvsr-inline", "swiftvr-inline",
    "swiftvr-distill",
]
LTX_SECONDARY_RESTORATIONS: tuple[SecondaryRestorationName, ...] = ("none", "rtx-super-res")
"""What LTX restoration supports without auto split; the others need BasicVSR++ parts."""

DenoiseStrengthName = Literal["none", "low", "medium", "high"]
VrModeName = Literal["auto", "off", "sbs", "sbs-fisheye", "tb"]
VrProjectionName = Literal["auto", "raw", "fisheye", "gnomonic"]
RtxQualityName = Literal["low", "medium", "high", "ultra"]
RtxLevelName = Literal["none", "low", "medium", "high", "ultra"]
CodecName = Literal["hevc", "h264", "av1"]
DEFAULT_CODEC: CodecName = "hevc"
LtxVaeName = Literal["auto", "full", "half", "max"]
"""An LTX video encoder or decoder size; "auto" is the model's own default."""
DecodeBackendName = Literal["auto", "vali", "pyav-hw", "pyav-sw"]
FrameGenerationName = Literal["none", "2x"]

# Option defaults shared by the CLI (``jasna.main.build_parser``) and the GUI (``AppSettings``).
DEFAULT_DETECTION_MODEL_NAME = "rfdetr-v6"
DEFAULT_MAX_CLIP_SIZE = 90
DEFAULT_TEMPORAL_OVERLAP = 8
DEFAULT_MAX_DETECTION_GAP = 2
DEFAULT_MIN_DETECTION_DURATION = 2
TVAI_DEFAULT_FFMPEG_PATH = r"C:\Program Files\Topaz Labs LLC\Topaz Video\ffmpeg.exe"
TVAI_DEFAULT_MODEL = "iris-2"
TVAI_DEFAULT_SCALE = 4
TVAI_DEFAULT_ARGS = (
    "preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=-2:vram=1:instances=1"
)
TVAI_DEFAULT_WORKERS = 2
RTX_DEFAULT_SCALE = 4
RTX_DEFAULT_QUALITY: RtxQualityName = "high"
RTX_DEFAULT_DENOISE: RtxLevelName = "medium"
RTX_DEFAULT_DEBLUR: RtxLevelName = "none"
SD15_DEFAULT_STEPS = 25
SD15_DEFAULT_STRENGTH = 0.6
SD15_DEFAULT_SEED = 0
SD15_DEFAULT_VARIANTS = 1
LTX_CROP_VSR_BACKENDS = ("swiftvr", "swiftvr-distill")


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
    restoration_model_name: RestorationModelName
    restoration_model_path: Path
    ltx_large_canvas: bool
    ltx_window_hop: int
    ltx_seed: int
    ltx_fast: bool
    ltx_model: LtxModelName
    ltx_vae_encoder: LtxVaeName
    ltx_vae_decoder: LtxVaeName
    ltx_trial: bool
    ltx_auto_split_percent: float | None
    compile_basicvsrpp: bool
    max_clip_size: int
    temporal_overlap: int
    denoise_strength: DenoiseStrengthName
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
    frame_generation: FrameGenerationName
    disable_progress: bool
    working_dir: Path | None
    vr_projection: VrProjectionName
    fmp4: bool
    sharpen_strength: float
    tvai_denoise: bool
    decode_backend: DecodeBackendName
    # RTX Super Res extras (nvidia-vfx >= 0.2.0.0): effect strength for every
    # RTX pass and the HIGHBITRATE_* (clean-source) upscale model family.
    rtx_strength: float = 1.0
    rtx_highbitrate: bool = False
    flashvsr_repo: str = ""
    flashvsr_python: str = ""
    flashvsr_model_dir: str = ""
    flashvsr_version: str = "11"
    flashvsr_dtype: str = "bf16"
    flashvsr_tiles: int = 1
    flashvsr_scale: int = 4
    flashvsr_accel: bool = False
    flashvsr_lora: str = ""
    flashvsr_log_level: str = "error"
    swiftvr_repo: str = ""
    swiftvr_python: str = ""
    swiftvr_model_dir: str = ""
    swiftvr_scale: int = 4
    swiftvr_accel: bool = True
    swiftvr_view_window: int = 15
    swiftvr_log_level: str = "error"
    # SwiftVR distillation student (experimental): an external checkpoint path.
    swiftvr_distill_model: str = ""
    swiftvr_distill_view_window: int = 15
    swiftvr_distill_strength: float = 0.75
    swiftvr_distill_stabilize: int = 0
    # SeedVR2 primary (restoration_model_name "seedvr2"): ``restoration_model_path``
    # carries the LoRA checkpoint instead of the BasicVSR++ checkpoint.
    seedvr2_repo: str = ""
    seedvr2_python: str = ""
    seedvr2_model_dir: str = ""
    seedvr2_dit: str = "seedvr2_ema_3b_fp16.safetensors"
    seedvr2_lora_rank: int = 16
    seedvr2_window: int = 33
    seedvr2_overlap: int = 9
    seedvr2_color_fix: str = "lab"
    seedvr2_empty_cache: str = "auto"
    seedvr2_accel: bool = False
    # ``--ltx-crop-vsr`` (modi): sharpen the mosaic areas of an LTX output (the input)
    # with SwiftVR; ``ltx_crop_vsr_source`` is the video LTX restored, scanned for the
    # mosaics. No primary restorer loads.
    ltx_crop_vsr: str | None = None
    ltx_crop_vsr_source: Path | None = None

    def model_path(self, name: RestorationModelName) -> Path:
        """``restoration_model_path`` belongs to the job's model; another model loads from its
        default path."""
        from jasna.engine_paths import default_restoration_model_path

        if name == self.restoration_model_name:
            return self.restoration_model_path
        return default_restoration_model_path(name)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("Batch size must be > 0")
        if self.max_clip_size <= 0:
            raise ValueError("Max clip size must be > 0")
        if self.temporal_overlap < 0:
            raise ValueError("Temporal overlap must be >= 0")
        if self.temporal_overlap > 0 and 2 * self.temporal_overlap >= self.max_clip_size:
            raise ValueError("Temporal overlap must satisfy 2 * temporal overlap < max clip size")
        if not 0 <= self.max_detection_gap < self.max_clip_size:
            raise ValueError("Max detection gap must be >= 0 and < max clip size")
        if not 0 <= self.min_detection_duration < self.max_clip_size:
            raise ValueError("Min detection duration must be >= 0 and < max clip size")
        if not 0.0 <= self.detection_score_threshold <= 1.0:
            raise ValueError("Detection score threshold must be in [0, 1]")
        if not 0.0 <= self.sharpen_strength <= 1.0:
            raise ValueError("Sharpening must be in [0, 1]")
        if self.ltx_crop_vsr is not None:
            if self.ltx_crop_vsr not in LTX_CROP_VSR_BACKENDS:
                raise ValueError(f"--ltx-crop-vsr must be one of {', '.join(LTX_CROP_VSR_BACKENDS)}")
            if self.ltx_crop_vsr_source is None:
                raise ValueError("--ltx-crop-vsr needs --ltx-crop-vsr-source")


def ltx_unsupported_options(*, auto_split: bool, secondary_restoration: str, denoise_strength: str) -> tuple[str, ...]:
    """The chosen options, as ``SessionConfig`` field names, that an LTX job cannot use: without
    auto split it has no BasicVSR++ parts for denoise or a secondary restoration other than
    ``LTX_SECONDARY_RESTORATIONS``."""
    if auto_split:
        return ()
    return tuple(
        name for name, unsupported in (
            ("secondary_restoration", secondary_restoration not in LTX_SECONDARY_RESTORATIONS),
            ("denoise_strength", denoise_strength != "none"),
        ) if unsupported
    )

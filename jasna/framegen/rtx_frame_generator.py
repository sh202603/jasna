from __future__ import annotations

import logging
from typing import Optional

import torch

from jasna.framegen.frame_generator import FrameGenerator

logger = logging.getLogger(__name__)

# NVIDIA Video Frame Generation (VFX SDK 1.3.0, shipped as nvidia-vfx >= 0.2.0.0)
# runs on Ada (sm 8.9), Hopper (Linux) and Blackwell only. Turing/Ampere GPUs
# must keep using the RIFE backend.
MIN_COMPUTE_CAPABILITY = (8, 9)
MODE_CHOICES = ["low", "medium", "high"]
DEFAULT_MODE = "medium"
MIN_NVVFX_VERSION = "0.2.0.0"


def _import_effect():
    """Return ``nvvfx.VideoFrameGeneration`` or raise a clear, actionable error."""
    try:
        import nvvfx
    except ImportError as e:
        raise RuntimeError(
            "RTX Video Frame Generation needs the nvidia-vfx package "
            f"(>= {MIN_NVVFX_VERSION}), which is not installed. Install it or use "
            "'--frame-gen-backend rife'."
        ) from e
    cls = getattr(nvvfx, "VideoFrameGeneration", None)
    if cls is None:
        raise RuntimeError(
            "RTX Video Frame Generation is not available in the installed nvidia-vfx "
            f"{getattr(nvvfx, '__version__', '?')} (it ships with nvidia-vfx >= "
            f"{MIN_NVVFX_VERSION}). Upgrade nvidia-vfx or use '--frame-gen-backend rife'."
        )
    return cls


def _resolve_mode(effect_cls, name: str):
    mode_enum = effect_cls.Mode
    table = {
        "low": mode_enum.LOW,
        "medium": mode_enum.MEDIUM,
        "high": mode_enum.HIGH,
    }
    try:
        return table[str(name).lower()]
    except KeyError:
        raise ValueError(f"Unsupported RTX frame-gen mode: {name!r} (choices: {', '.join(MODE_CHOICES)})") from None


def _to_sdk_input(frame: torch.Tensor, device: torch.device) -> torch.Tensor:
    """CHW uint8 -> the SDK's RGB8 encoding: contiguous (3, H, W) float32 in [0, 1].

    ``copy=True`` guarantees a fresh buffer so the in-place divide never touches
    the caller's frame (the writer still holds it as the next pair's "previous").
    """
    return frame.to(device=device, dtype=torch.float32, copy=True).div_(255.0).contiguous()


def _to_chw_uint8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_(0.0, 1.0).mul_(255.0).round_().to(dtype=torch.uint8)


class RtxFrameGenerator(FrameGenerator):
    """NVIDIA RTX Video Frame Generation backend (nvvfx ``VideoFrameGeneration``).

    Mirrors the nvvfx ``Effect`` usage of ``RtxSuperresSecondaryRestorer``: CUDA
    tensors cross the boundary via DLPack and every call runs on the caller's
    current torch stream. Unlike RIFE there are no weights to supply and no
    padding requirement; the SDK accepts arbitrary frame sizes. The effect is
    bound to one input size, so it is loaded lazily from the first frame pair
    and reloaded (about 15 ms) whenever the size changes, which lets a folder
    batch share one generator across videos of different resolutions.

    ``model_path`` and ``fp16`` are accepted for signature parity with the RIFE
    backend and ignored: the SDK owns its model and precision.
    """

    name = "rtx-frame-gen"

    def __init__(
        self,
        *,
        device: torch.device,
        mode: str = DEFAULT_MODE,
        shot_change_detection: bool = True,
        model_path: Optional[str] = None,
        fp16: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self._check_gpu()
        effect_cls = _import_effect()
        sdk_mode = _resolve_mode(effect_cls, mode)
        self._mode_name = str(mode).lower()

        gpu = self.device.index or 0
        self._fg = effect_cls(
            mode=sdk_mode,
            automatic_shot_change_detection_enabled=bool(shot_change_detection),
            device=gpu,
        )
        self._size: tuple[int, int] | None = None  # (H, W) the effect is loaded for
        logger.info(
            "RtxFrameGenerator initialized (mode=%s, shot_change_detection=%s)",
            self._mode_name, bool(shot_change_detection),
        )

    def _check_gpu(self) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("RTX Video Frame Generation requires a CUDA device")
        major, minor = torch.cuda.get_device_capability(self.device)
        if (major, minor) < MIN_COMPUTE_CAPABILITY:
            raise RuntimeError(
                "RTX Video Frame Generation requires an NVIDIA Ada (RTX 40 series) or "
                f"newer GPU (compute capability {MIN_COMPUTE_CAPABILITY[0]}.{MIN_COMPUTE_CAPABILITY[1]}+); "
                f"this GPU is {major}.{minor}. Use '--frame-gen-backend rife' instead."
            )

    def _ensure_loaded(self, height: int, width: int) -> None:
        if self._size == (height, width):
            return
        self._fg.input_width = int(width)
        self._fg.input_height = int(height)
        self._fg.load()
        self._size = (height, width)
        logger.info("RtxFrameGenerator: effect loaded for %dx%d", width, height)

    @torch.inference_mode()
    def interpolate(self, frame_a: torch.Tensor, frame_b: torch.Tensor, positions: list[float]) -> list[torch.Tensor]:
        if not positions:
            return []
        if frame_a.ndim != 3 or frame_a.shape[0] != 3 or frame_a.shape != frame_b.shape:
            raise ValueError(
                f"expected two CHW RGB frames of equal shape, got {tuple(frame_a.shape)} and {tuple(frame_b.shape)}"
            )
        _, h, w = frame_a.shape
        self._ensure_loaded(int(h), int(w))

        # The blend-encode thread drives this on its current stream; the SDK's
        # RGB8 conversion kernels run on the stream it is handed, so passing
        # the same stream keeps input conversion -> SDK -> output clone ordered.
        stream_ptr = torch.cuda.current_stream(self.device).cuda_stream
        prev = _to_sdk_input(frame_a, self.device)
        curr = _to_sdk_input(frame_b, self.device)

        out: list[torch.Tensor] = []
        for p in positions:
            result = self._fg.run_at_timestep(prev, curr, float(p), stream_ptr=stream_ptr)
            # The SDK returns a view of its single output buffer, which the next
            # run overwrites, so copy it out before continuing.
            mid = torch.from_dlpack(result.image).clone()
            out.append(_to_chw_uint8(mid))
        return out

    def close(self) -> None:
        fg = getattr(self, "_fg", None)
        if fg is not None:
            fg.close()
            self._fg = None
        self._size = None

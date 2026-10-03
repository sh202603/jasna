from __future__ import annotations

import ctypes
import logging
import os
import sys
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _preload_tensorrt_runtime() -> None:
    """Pin the pip ``tensorrt`` runtime before nvvfx loads its bundled copy.

    nvidia-vfx <= 0.1.0.1 bundled an older TensorRT (libnvinfer.so.10 == 10.9)
    and loaded it with ``RTLD_GLOBAL`` (see nvvfx/_lib_loader.py). Because both
    share the soname ``libnvinfer.so.10``, ELF symbol resolution uses whichever
    entered the global scope first. If nvvfx won, torch-tensorrt bound to 10.9
    and failed to deserialize jasna's 10.16-built engines ("Serialized Engine
    Version" mismatch). Loading tensorrt_libs' 10.16 RTLD_GLOBAL first makes its
    symbols win. nvidia-vfx 0.2.0.0 (the pinned version) no longer ships
    TensorRT at all, so this is now a harmless no-op kept for older venvs and
    for parity with upstream, which has the same function. The Windows build
    does the equivalent via DLL load ordering; this is the Linux counterpart.
    """
    if sys.platform == "win32":
        return  # Windows handles ordering via windows_dll_paths / tensorrt_libs import
    import importlib.util
    try:
        spec = importlib.util.find_spec("tensorrt_libs")
        if spec is None or not spec.submodule_search_locations:
            return
        libdir = list(spec.submodule_search_locations)[0]
        for name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10"):
            path = os.path.join(libdir, name)
            if os.path.exists(path):
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        logger.debug("Could not pre-load tensorrt_libs before nvvfx", exc_info=True)


# Must run before nvvfx is imported anywhere in this module.
_preload_tensorrt_runtime()

RTX_SUPERRES_INPUT_SIZE = 256
SCALE_CHOICES = [2, 4]
QUALITY_CHOICES = ["low", "medium", "high", "ultra"]
DENOISE_CHOICES = ["none", "low", "medium", "high", "ultra"]
DEBLUR_CHOICES = ["none", "low", "medium", "high", "ultra"]


def _resolve_quality(name: str, *, highbitrate: bool = False):
    """Map a level name to the upscale model.

    The standard models (LOW..ULTRA) are trained on compressed video and
    suppress compression artifacts while upscaling. The HIGHBITRATE_* family
    skips that suppression for clean sources; BasicVSR++ output has no
    compression noise, so ``highbitrate=True`` may preserve more detail.
    """
    from nvvfx import VideoSuperRes
    q = VideoSuperRes.QualityLevel
    standard, clean = {
        "low": (q.LOW, q.HIGHBITRATE_LOW),
        "medium": (q.MEDIUM, q.HIGHBITRATE_MEDIUM),
        "high": (q.HIGH, q.HIGHBITRATE_HIGH),
        "ultra": (q.ULTRA, q.HIGHBITRATE_ULTRA),
    }[name.lower()]
    return clean if highbitrate else standard


def _resolve_denoise(name: str):
    from nvvfx import VideoSuperRes
    return {
        "low": VideoSuperRes.QualityLevel.DENOISE_LOW,
        "medium": VideoSuperRes.QualityLevel.DENOISE_MEDIUM,
        "high": VideoSuperRes.QualityLevel.DENOISE_HIGH,
        "ultra": VideoSuperRes.QualityLevel.DENOISE_ULTRA,
    }[name.lower()]


def _resolve_deblur(name: str):
    from nvvfx import VideoSuperRes
    return {
        "low": VideoSuperRes.QualityLevel.DEBLUR_LOW,
        "medium": VideoSuperRes.QualityLevel.DEBLUR_MEDIUM,
        "high": VideoSuperRes.QualityLevel.DEBLUR_HIGH,
        "ultra": VideoSuperRes.QualityLevel.DEBLUR_ULTRA,
    }[name.lower()]


def _make_effect(VideoSuperRes, *, gpu: int, quality, strength: float, output_size: int):
    """Construct, size and load one VideoSuperRes pass.

    ``strength`` is only forwarded when it differs from the SDK default (1.0):
    the keyword exists from nvidia-vfx 0.2.0.0 on, so the default configuration
    keeps working on a venv that still has 0.1.0.1.
    """
    kwargs = {"device": gpu, "quality": quality}
    if strength != 1.0:
        kwargs["strength"] = float(strength)
    effect = VideoSuperRes(**kwargs)
    effect.output_width = output_size
    effect.output_height = output_size
    effect.load()
    return effect


class RtxSuperresSecondaryRestorer:
    name = "rtx-super-res"
    num_workers = 1
    preferred_queue_size = 2
    prefers_cpu_input = False

    def __init__(
        self,
        *,
        device: torch.device,
        scale: int = 4,
        quality: str = "high",
        denoise: Optional[str] = "medium",
        deblur: Optional[str] = None,
        strength: float = 1.0,
        highbitrate: bool = False,
        input_size: int = RTX_SUPERRES_INPUT_SIZE,
    ) -> None:
        from nvvfx import VideoSuperRes

        if input_size < 1:
            raise ValueError("input_size must be positive")
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError(f"strength must be in [0.0, 1.0], got {strength}")
        output_size = input_size * scale

        self.device = torch.device(device)
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.strength = float(strength)
        self.highbitrate = bool(highbitrate)
        gpu = self.device.index or 0
        self._stream_ptr = torch.cuda.current_stream(self.device).cuda_stream

        # The same strength applies to every pass: it is "how strong is the RTX
        # chain", not a per-pass knob.
        self._sr = _make_effect(
            VideoSuperRes, gpu=gpu,
            quality=_resolve_quality(quality, highbitrate=self.highbitrate),
            strength=self.strength, output_size=output_size,
        )

        self._denoise = None
        if denoise is not None and denoise.lower() != "none":
            self._denoise = _make_effect(
                VideoSuperRes, gpu=gpu, quality=_resolve_denoise(denoise),
                strength=self.strength, output_size=output_size,
            )

        self._deblur = None
        if deblur is not None and deblur.lower() != "none":
            self._deblur = _make_effect(
                VideoSuperRes, gpu=gpu, quality=_resolve_deblur(deblur),
                strength=self.strength, output_size=output_size,
            )

        logger.info(
            "RtxSuperresSecondaryRestorer: scale=%dx quality=%s%s denoise=%s deblur=%s strength=%.2f (%dx%d -> %dx%d)",
            scale, quality, " (highbitrate)" if self.highbitrate else "", denoise, deblur, self.strength,
            self.input_size, self.input_size, output_size, output_size,
        )

    def restore(self, frames: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        t = int(frames.shape[0])
        if t == 0:
            return []
        if frames.ndim != 4 or tuple(frames.shape[1:]) != (
            3,
            self.input_size,
            self.input_size,
        ):
            raise ValueError(
                f"expected frames shaped (T, 3, {self.input_size}, {self.input_size}), "
                f"got {tuple(frames.shape)}"
            )

        ks = max(0, int(keep_start))
        ke = min(t, int(keep_end))
        if ks >= ke:
            return []
        frames = frames[ks:ke]
        t = int(frames.shape[0])

        out: list[torch.Tensor] = []
        for i in range(t):
            frame = frames[i].to(device=self.device, dtype=torch.float32).contiguous()

            result = torch.from_dlpack(self._sr.run(frame, stream_ptr=self._stream_ptr).image).clone()
            if self._denoise is not None:
                result = torch.from_dlpack(self._denoise.run(result, stream_ptr=self._stream_ptr).image).clone()
            if self._deblur is not None:
                result = torch.from_dlpack(self._deblur.run(result, stream_ptr=self._stream_ptr).image).clone()

            out_u8 = result.clamp(0, 1).mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)
            out.append(out_u8)

        return out

    def close(self) -> None:
        if self._sr is not None:
            self._sr.close()
            self._sr = None
        if self._denoise is not None:
            self._denoise.close()
            self._denoise = None
        if self._deblur is not None:
            self._deblur.close()
            self._deblur = None

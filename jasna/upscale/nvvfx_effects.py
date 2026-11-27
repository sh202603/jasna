"""NVIDIA RTX Video Effects (nvvfx): importing it after the pip TensorRT runtime, and its video
super resolution effects."""

from __future__ import annotations

import ctypes
import functools
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

import torch

from jasna.session_config import RtxLevelName, RtxQualityName
from jasna.upscale import Size


@functools.cache
def _preload_tensorrt_runtime() -> None:
    """Pin the pip ``tensorrt`` runtime before nvvfx loads its bundled copy.

    nvvfx (NVIDIA Maxine) ships its own TensorRT and loads ``libnvinfer.so.10``
    with ``RTLD_GLOBAL``. Our BasicVSR++ engines are serialized against the pip
    ``tensorrt`` (a different version), and TRT engines are version-locked. If
    nvvfx loads first, its TensorRT symbols win global symbol resolution,
    torch_tensorrt binds to them, and BasicVSR++ engine deserialization fails
    with a serialization-version mismatch. Loading the pip libs into the global
    scope first makes torch_tensorrt bind to the matching runtime; nvvfx then
    happily reuses it.
    """
    # modi: the TensorRT-RTX flavour has no ``tensorrt_libs``, and nvidia-vfx 0.2.0.0 no longer
    # ships a TensorRT of its own, so a missing package or library is not an error here.
    import importlib.util

    spec = importlib.util.find_spec("tensorrt_libs")
    if spec is None or not spec.submodule_search_locations:
        return
    libs_dir = list(spec.submodule_search_locations)[0]
    if sys.platform == "win32":
        names, load = ("nvinfer_10.dll", "nvinfer_plugin_10.dll"), ctypes.WinDLL
    else:
        names, load = ("libnvinfer.so.10", "libnvinfer_plugin.so.10"), functools.partial(ctypes.CDLL, mode=ctypes.RTLD_GLOBAL)
    for name in names:
        path = os.path.join(libs_dir, name)
        if os.path.exists(path):
            load(path)


def import_nvvfx() -> ModuleType:
    """``nvvfx``, imported only once the pip TensorRT runtime holds the global symbol scope."""
    _preload_tensorrt_runtime()
    import nvvfx

    return nvvfx


def cleanup_level_names(denoise: RtxLevelName, deblur: RtxLevelName) -> list[str]:
    """The denoise and deblur ``VideoSuperRes`` levels to run, in that order (``"none"`` skips one)."""
    return [f"{kind}_{level}" for kind, level in (("denoise", denoise), ("deblur", deblur)) if level != "none"]


def upscale_level_name(quality: RtxQualityName, *, highbitrate: bool = False) -> str:
    """+modi: the ``VideoSuperRes`` upscale level for ``quality``. The standard models (LOW..ULTRA)
    are trained on compressed video and suppress compression artifacts while upscaling; the
    ``HIGHBITRATE_*`` family (nvidia-vfx 0.2.0.0 on) skips that for clean sources, which restored
    crops are. Denoise and deblur have no such family."""
    return f"highbitrate_{quality}" if highbitrate else quality


def video_super_res(level_name: str, output_size: Size, device: torch.device, *, strength: float = 1.0):
    """A loaded ``VideoSuperRes`` effect at ``level_name`` (``"high"``, ``"denoise_low"``, ...) with
    a ``(width, height)`` output.

    +modi: ``strength`` is forwarded only when it differs from the SDK default (1.0): the keyword
    exists from nvidia-vfx 0.2.0.0 on, so the default keeps working on a venv with 0.1.0.1."""
    VideoSuperRes = import_nvvfx().VideoSuperRes
    kwargs = {"device": device.index or 0, "quality": getattr(VideoSuperRes.QualityLevel, level_name.upper())}
    if strength != 1.0:
        kwargs["strength"] = float(strength)
    effect = VideoSuperRes(**kwargs)
    effect.output_width, effect.output_height = output_size
    effect.load()
    return effect


@contextmanager
def on_default_stream(device: torch.device) -> Iterator[torch.cuda.Stream]:
    """Runs the block on the default stream, ordered after the caller's stream and before its later
    work: nvvfx effects fail on any stream but the default one they load with."""
    current = torch.cuda.current_stream(device)
    default = torch.cuda.default_stream(device)
    default.wait_stream(current)
    with torch.cuda.stream(default):
        yield default
    current.wait_stream(default)


class SuperResChain:
    """Super resolution to ``output_size``, then denoise and deblur at that size.

    +modi: ``strength`` (0.0 to 1.0, every pass: it is "how strong is the RTX chain", not a
    per-pass knob) and ``highbitrate`` (the clean-source upscale models) are the secondary
    restorer's ``--rtx-strength`` / ``--rtx-highbitrate``. Their defaults are the SDK's, which is
    what whole-frame ``--upscale`` runs with."""

    def __init__(
        self,
        device: torch.device,
        *,
        output_size: Size,
        quality: RtxQualityName,
        denoise: RtxLevelName,
        deblur: RtxLevelName,
        strength: float = 1.0,
        highbitrate: bool = False,
    ) -> None:
        self.output_size = output_size
        levels = (upscale_level_name(quality, highbitrate=highbitrate), *cleanup_level_names(denoise, deblur))
        self._effects = [video_super_res(level, output_size, device, strength=strength) for level in levels]

    def __call__(self, image: torch.Tensor, stream_ptr: int) -> torch.Tensor:
        """Float ``[3, h, w]`` in ``[0, 1]`` to the uint8 ``[3, H, W]`` output, run on ``stream_ptr``."""
        for effect in self._effects:
            image = torch.from_dlpack(effect.run(image, stream_ptr=stream_ptr).image).clone()
        return image.clamp(0, 1).mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)

    def close(self) -> None:
        for effect in self._effects:
            effect.close()
        self._effects.clear()

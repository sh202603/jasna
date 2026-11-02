"""RTX Super Res for LTX windows whose crop is larger than the model canvas.

Compose enlarges such a window back to its crop. With super resolution the window is
first upscaled 2x, so compose shrinks it to the crop, or enlarges it by the remainder
past 2x.

+modi: with a deblur pass, every window is upscaled (the deblur is what sharpens, and it
sharpens windows compose shrinks too); without one only the enlarged windows are, as
upstream does (a 2x that compose shrinks back changes nothing then).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from jasna.ltx.camera import CropFrame
from jasna.ltx.plan import Window
from jasna.session_config import RtxLevelName, RtxQualityName

SUPER_RES_SCALE = 2


def enlarged_windows(windows: Sequence[Window]) -> set[int]:
    """Indices of the windows that compose enlarges on some frame."""
    return {window.index for window in windows if max(crop.side for crop in window.crops) > window.canvas}


def upscaled_crop(crop: CropFrame) -> CropFrame:
    """``crop`` as it lies on the canvas after a 2x upscale."""
    height, width = crop.resize_hw
    return CropFrame(
        crop_box=crop.crop_box,
        resize_hw=(height * SUPER_RES_SCALE, width * SUPER_RES_SCALE),
        pad=tuple(side * SUPER_RES_SCALE for side in crop.pad),
    )


class LtxSuperRes:
    """One RTX Super Res 2x upscaler per canvas size, made on first use or by ``prepare``."""

    def __init__(
        self,
        *,
        device: torch.device,
        quality: RtxQualityName,
        denoise: RtxLevelName,
        deblur: RtxLevelName,
        strength: float = 1.0,
        highbitrate: bool = False,
    ) -> None:
        self._device = device
        self._quality = quality
        self._denoise = None if denoise == "none" else denoise
        self._deblur = None if deblur == "none" else deblur
        self._strength = float(strength)
        self._highbitrate = bool(highbitrate)
        self._upscalers = {}

    @property
    def sharpens(self) -> bool:
        """Whether the deblur pass runs (+modi: then every window is upscaled)."""
        return self._deblur is not None

    def windows_to_upscale(self, windows: Sequence[Window]) -> set[int]:
        """Indices of the windows compose upscales: all of them with a deblur pass, else the
        enlarged ones."""
        return {window.index for window in windows} if self.sharpens else enlarged_windows(windows)

    def _upscaler(self, size: int):
        from jasna.restorer.rtx_superres_secondary_restorer import RtxSuperresSecondaryRestorer

        upscaler = self._upscalers.get(size)
        if upscaler is None:
            upscaler = RtxSuperresSecondaryRestorer(
                device=self._device,
                scale=SUPER_RES_SCALE,
                quality=self._quality,
                denoise=self._denoise,
                deblur=self._deblur,
                strength=self._strength,
                highbitrate=self._highbitrate,
                input_size=size,
            )
            self._upscalers[size] = upscaler
        return upscaler

    def prepare(self, sizes: Sequence[int]) -> None:
        """Load the upscaler of every canvas size up front, on the default stream (+modi):
        an LTX decode's budget is the VRAM free at that decode, so an upscaler loading on
        first use would tile the first window differently from the rest."""
        with torch.cuda.stream(torch.cuda.default_stream(self._device)):
            for size in sizes:
                self._upscaler(int(size))

    def upscale(self, canvas: torch.Tensor) -> torch.Tensor:
        """uint8 ``[3, S, S]`` on the GPU to uint8 ``[3, 2S, 2S]``, ordered after the caller's stream."""
        current = torch.cuda.current_stream(self._device)
        # nvvfx effects fail on any stream but the default one they load with.
        default = torch.cuda.default_stream(self._device)
        default.wait_stream(current)
        with torch.cuda.stream(default):
            upscaled = self._upscaler(int(canvas.shape[-1])).restore(canvas[None].float().div_(255.0), keep_start=0, keep_end=1)[0]
        current.wait_stream(default)
        upscaled.record_stream(current)
        return upscaled

    def release(self) -> None:
        """Close every upscaler; the next ``prepare`` or ``upscale`` makes them again."""
        for upscaler in self._upscalers.values():
            upscaler.close()
        self._upscalers.clear()

    def close(self) -> None:
        self.release()

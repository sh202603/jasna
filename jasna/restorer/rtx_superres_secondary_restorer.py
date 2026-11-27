from __future__ import annotations

import logging

import torch

from jasna.session_config import RtxLevelName, RtxQualityName
from jasna.upscale.nvvfx_effects import SuperResChain

logger = logging.getLogger(__name__)

RTX_SUPERRES_INPUT_SIZE = 256


class RtxSuperresSecondaryRestorer:
    """+modi: ``input_size`` is the side of the square crops the frame pipeline hands over (256);
    ``None`` sizes the passes from the frames instead, one chain per input side met, so a caller
    whose crops come at several sides (the LTX canvases of ``--ltx-crop-vsr``, 512 and 768 px)
    never reloads an effect. ``strength`` and ``highbitrate`` are ``--rtx-strength`` and
    ``--rtx-highbitrate`` (see ``SuperResChain``)."""

    name = "rtx-super-res"
    num_workers = 1
    prefers_cpu_input = False

    def __init__(
        self,
        *,
        device: torch.device,
        scale: int,
        quality: RtxQualityName,
        denoise: RtxLevelName,
        deblur: RtxLevelName,
        strength: float = 1.0,
        highbitrate: bool = False,
        input_size: int | None = RTX_SUPERRES_INPUT_SIZE,
    ) -> None:
        if input_size is not None and input_size < 1:
            raise ValueError("input_size must be positive")
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError(f"strength must be in [0.0, 1.0], got {strength}")
        self.device = torch.device(device)
        self.scale = int(scale)
        self.input_size = None if input_size is None else int(input_size)
        self.output_size = None if self.input_size is None else self.input_size * self.scale
        self.strength = float(strength)
        self.highbitrate = bool(highbitrate)
        self._quality = quality
        self._denoise = denoise
        self._deblur = deblur
        self._stream_ptr = torch.cuda.current_stream(self.device).cuda_stream
        self._chains: dict[int, SuperResChain] = {}
        if self.input_size is not None:
            self._chain_for(self.input_size)
        logger.info(
            "RtxSuperresSecondaryRestorer: scale=%dx quality=%s%s denoise=%s deblur=%s strength=%.2f (%s)",
            self.scale, quality, " (highbitrate)" if self.highbitrate else "", denoise, deblur, self.strength,
            f"{self.input_size}x{self.input_size} -> {self.output_size}x{self.output_size}"
            if self.input_size is not None else "sized from the frames",
        )

    def _chain_for(self, size: int) -> SuperResChain:
        chain = self._chains.get(size)
        if chain is None:
            output = size * self.scale
            chain = self._chains[size] = SuperResChain(
                self.device,
                output_size=(output, output),
                quality=self._quality,
                denoise=self._denoise,
                deblur=self._deblur,
                strength=self.strength,
                highbitrate=self.highbitrate,
            )
            if self.input_size is None:
                logger.info("RtxSuperresSecondaryRestorer: passes loaded for %dx%d -> %dx%d", size, size, output, output)
        return chain

    def restore(self, frames: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        t = int(frames.shape[0])
        if t == 0:
            return []
        if frames.ndim != 4 or frames.shape[1] != 3 or frames.shape[2] != frames.shape[3]:
            raise ValueError(f"expected frames shaped (T, 3, S, S), got {tuple(frames.shape)}")
        size = int(frames.shape[-1])
        if self.input_size is not None and size != self.input_size:
            raise ValueError(
                f"expected frames shaped (T, 3, {self.input_size}, {self.input_size}), "
                f"got {tuple(frames.shape)}"
            )

        ks = max(0, int(keep_start))
        ke = min(t, int(keep_end))
        if ks >= ke:
            return []
        chain = self._chain_for(size)
        return [
            chain(frame.to(device=self.device, dtype=torch.float32).contiguous(), self._stream_ptr)
            for frame in frames[ks:ke]
        ]

    def close(self) -> None:
        for chain in self._chains.values():
            chain.close()
        self._chains.clear()

from __future__ import annotations

import logging
from typing import Optional

import torch

from jasna.framegen.frame_generator import FrameGenerator

logger = logging.getLogger(__name__)


def _find_frame_gen_effect():
    """Return the nvvfx frame-generation Effect class, or None if not shipped.

    NVIDIA's "RTX Video Frame Generation" was announced as a Python wheel +
    ComfyUI node alongside RTX Spark, but is not yet part of the released
    nvidia-vfx package (1.2.0 exposes only ``VideoSuperRes``). When NVIDIA ships
    the effect it will appear as another ``nvvfx.effects.Effect`` subclass; we
    probe the likely public names so this backend activates automatically
    without further code changes here.
    """
    try:
        import nvvfx
    except Exception:
        return None
    for attr in ("VideoFrameGeneration", "FrameGeneration", "FrameInterpolation",
                 "VideoFrameInterpolation", "FrameRateUpConversion"):
        cls = getattr(nvvfx, attr, None)
        if cls is not None:
            return cls
    return None


class RtxFrameGenerator(FrameGenerator):
    """NVIDIA RTX Video Frame Generation backend (pending SDK release).

    This adapter mirrors the nvvfx ``Effect`` usage pattern of
    ``RtxSuperresSecondaryRestorer`` (DLPack CUDA arrays, ``run(..., stream_ptr=...)``).
    The actual inference call is left as a TODO because the installed nvidia-vfx
    does not yet expose a frame-generation effect; constructing this backend
    today raises a clear error rather than failing deep in the pipeline.
    """

    name = "rtx-frame-gen"

    def __init__(self, *, device: torch.device, model_path: Optional[str] = None, fp16: bool = True) -> None:
        effect_cls = _find_frame_gen_effect()
        if effect_cls is None:
            raise RuntimeError(
                "RTX Video Frame Generation is not available in the installed nvidia-vfx "
                "package (only VideoSuperRes is exposed). This NVIDIA feature ships with a "
                "future nvidia-vfx release (announced alongside RTX Spark). Use "
                "'--frame-gen-backend rife' for now."
            )

        self.device = torch.device(device)
        gpu = self.device.index or 0
        self._stream_ptr = torch.cuda.current_stream(self.device).cuda_stream
        # When the effect ships, instantiate + load() here, mirroring
        # RtxSuperresSecondaryRestorer.__init__.
        self._fg = effect_cls(device=gpu)
        self._fg.load()
        logger.info("RtxFrameGenerator initialized via %s", effect_cls.__name__)

    def interpolate(self, frame_a: torch.Tensor, frame_b: torch.Tensor, positions: list[float]) -> list[torch.Tensor]:
        # TODO(rtx-vfg): once the nvvfx frame-gen effect is released, convert
        # frame_a/frame_b to the effect's expected CUDA/DLPack input and call
        # self._fg.run(frame_a, frame_b, timestep=p, stream_ptr=self._stream_ptr)
        # for each p in positions, returning CHW RGB uint8 tensors.
        raise NotImplementedError(
            "RTX Video Frame Generation inference is not implemented yet (SDK effect unavailable)."
        )

    def close(self) -> None:
        fg = getattr(self, "_fg", None)
        if fg is not None:
            fg.close()
            self._fg = None

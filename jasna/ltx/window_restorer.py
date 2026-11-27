"""+modi: a temporal x2 restorer for the windows of an LTX job's compose pass
(``--secondary-restoration swiftvr-inline`` or ``swiftvr-distill`` with an LTX model).

It takes the place of ``LtxSuperRes`` (``super_res.py``) in the compose pass. RTX Super Res
works on one frame, so it upscales a canvas when a frame composites it; these models read
the frames around the one they draw, so a window is restored whole, once, when its first
frame is composited. The restorer is the one ``--ltx-crop-vsr`` runs (``refine.py``): a
window's canvases in, the same frames at twice the side out.

The restorer loads when a compose pass starts and closes when it ends: the denoiser is gone
by then (``restore._restore_spans``), and the next segment's denoise gets the VRAM back.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
import torch

from jasna.ltx.super_res import SUPER_RES_SCALE
from jasna.ltx.windows import Window

logger = logging.getLogger(__name__)


def kept_frames(weights: np.ndarray) -> range:
    """The frames of a window that reach the output: from its first to its last frame with a
    compose weight above 0 (``crossfade_weights``). A later window's unfused lead frames have
    none, and the decoder garbles their colour (``windows.UNFUSED_LEAD_FRAMES``), so they must
    not serve as the restorer's temporal context either."""
    live = np.flatnonzero(np.asarray(weights) > 0.0)
    if live.size == 0:
        return range(0)
    return range(int(live[0]), int(live[-1]) + 1)


class LtxWindowRestorer:
    """Whole windows through a x2 secondary restorer (``restore(frames, keep_start, keep_end)``),
    built by ``make_restorer`` when a compose pass prepares and closed when it releases."""

    whole_windows = True
    sharpens = True

    def __init__(self, name: str, make_restorer: Callable[[], object]) -> None:
        self.name = name
        self._make_restorer = make_restorer
        self._restorer = None

    @property
    def profile_tag(self) -> str:
        """Keys the speed profile: the compose rate holds this restorer's time."""
        return f"+{self.name}"

    def windows_to_upscale(self, windows: Sequence[Window]) -> set[int]:
        return {window.index for window in windows}

    def prepare(self, sizes: Sequence[int]) -> None:
        """Load the restorer before the first window decodes: a decode's budget is the VRAM
        free at that decode."""
        if self._restorer is None:
            self._restorer = self._make_restorer()
            logger.info("LtxWindowRestorer: %s loaded for the compose pass (canvases %s)", self.name, list(sizes))

    def upscale_window(self, video: torch.Tensor, kept: range) -> torch.Tensor:
        """A decoded window, uint8 ``[F, 3, S, S]``, as host uint8 ``[F, 3, 2S, 2S]``: the frames
        of ``kept`` restored together, the others (never composited) doubled pixel by pixel."""
        self.prepare(sorted({int(video.shape[-1])}))
        out = video.to("cpu").repeat_interleave(SUPER_RES_SCALE, dim=-2).repeat_interleave(SUPER_RES_SCALE, dim=-1)
        if len(kept) == 0:
            return out
        frames = video[kept.start : kept.stop].to("cpu").float().div_(255.0)
        restored = self._restorer.restore(frames, keep_start=0, keep_end=len(kept))
        if len(restored) != len(kept):
            raise RuntimeError(f"{self.name} returned {len(restored)} frames for a {len(kept)}-frame window")
        result = torch.stack([frame.to("cpu") for frame in restored])
        if result.shape[1:] != out.shape[1:] or result.dtype != torch.uint8:
            raise RuntimeError(
                f"{self.name} returned {tuple(result.shape[1:])} {result.dtype} frames for a "
                f"{int(video.shape[-1])} px canvas; expected uint8 {tuple(out.shape[1:])}"
            )
        out[kept.start : kept.stop] = result
        return out

    def release(self) -> None:
        """Close the restorer; the next ``prepare`` makes it again."""
        restorer, self._restorer = self._restorer, None
        close = getattr(restorer, "close", None)
        if close is not None:
            close()

    def close(self) -> None:
        self.release()

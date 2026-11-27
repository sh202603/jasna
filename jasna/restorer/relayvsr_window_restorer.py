"""+modi: RelayVSR as the x2 restorer of LTX windows (``--secondary-restoration relayvsr`` with
an LTX model, ``jasna/ltx/window_restorer.py``).

One window is one video to the resident worker of ``jasna-postprocess --upscale relayvsr``
(``jasna.upscale.relayvsr.RelayvsrWorker``): the window's canvases go in one at a time, the
model answers with the frames it has finished, and the end of the window brings the rest. The
Wan K/V cache therefore starts empty at every window; the frames two windows share are
restored once in each and blended by the compose crossfade.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

RELAYVSR_WINDOW_SCALE = 2


class RelayvsrWindowRestorer:
    """``restore(frames, keep_start, keep_end)`` of the secondary restorers, on a RelayVSR worker
    of its own: ``close`` ends the worker."""

    name = "relayvsr"
    num_workers = 1
    prefers_cpu_input = True

    def __init__(
        self,
        *,
        rv_python: Path,
        weights_dir: Path,
        device: str,
        window_size: int = 16,
        wan_weights: str = "fp8",
        kv_cache: str = "auto",
        kv_cache_dtype: str = "native",
        verbose: bool = False,
    ) -> None:
        from jasna.upscale.relayvsr import RelayvsrWorker

        self._worker = RelayvsrWorker(
            rv_python=rv_python, weights_dir=weights_dir, device=device, scale=RELAYVSR_WINDOW_SCALE,
            window_size=window_size, wan_weights=wan_weights, kv_cache=kv_cache, kv_cache_dtype=kv_cache_dtype,
            verbose=verbose,
        )

    def restore(self, frames: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        """Float RGB ``[T, 3, H, W]`` in 0..1 to host uint8 ``[3, 2H, 2W]`` frames ``keep_start``
        to ``keep_end``. Every frame is restored: the ones outside the kept range are context."""
        count = int(frames.shape[0])
        first, last = max(0, int(keep_start)), min(count, int(keep_end))
        if count == 0 or first >= last:
            return []
        hwc = (
            frames.detach().to("cpu", torch.float32, copy=True).clamp_(0.0, 1.0).mul_(255.0).round_()
            .to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
        )
        height, width = int(hwc.shape[1]), int(hwc.shape[2])
        restored: list[np.ndarray] = []
        try:
            self._worker.begin(width, height, RELAYVSR_WINDOW_SCALE)
            for frame in hwc:
                # The worker reuses the array it yields, so each frame is copied out.
                restored.extend(out.copy() for out in self._worker.push(frame))
            restored.extend(out.copy() for out in self._worker.end())
        except BaseException:
            self._worker.abandon()
            raise
        if len(restored) != count:
            raise RuntimeError(f"[relayvsr] frame-count mismatch: sent {count}, got {len(restored)}")
        return [torch.from_numpy(np.ascontiguousarray(out.transpose(2, 0, 1))) for out in restored[first:last]]

    def close(self) -> None:
        self._worker.close()

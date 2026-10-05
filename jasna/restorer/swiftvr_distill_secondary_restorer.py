from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.nn import functional as F

from jasna.restorer.swiftvr_distill_model import SWIFTVR_DISTILL_BASE_MODE, load_swiftvr_distill_model

logger = logging.getLogger(__name__)

# Larger batches are slower for this model (RTX 5060 Ti, FP32: 238 crop-fps at
# 4, 188 at 16) and only add VRAM.
SWIFTVR_DISTILL_BATCH_SIZE = 4

# Temporal stabilization of the added detail, after mioh's stabilizeSwiftVRFrame:
# a neighbour stops contributing where its input differs from the centre
# frame's by more than a few levels, and counts less than the centre anyway.
SWIFTVR_DISTILL_STABILIZE_MAX_RADIUS = 8
SWIFTVR_DISTILL_STABILIZE_SIGMA = 8.0 / 255.0
SWIFTVR_DISTILL_STABILIZE_NEIGHBOUR_WEIGHT = 0.8
_STABILIZE_CHUNK = 16

SWIFTVR_DISTILL_MAX_STRENGTH = 2.0


def _upscale_base(base: torch.Tensor, scale: int) -> torch.Tensor:
    """The model's own base: what its output is when it adds nothing."""
    return F.interpolate(base, scale_factor=scale, mode=SWIFTVR_DISTILL_BASE_MODE, align_corners=False)


def clamped_window_indices(start: int, end: int, total: int, radius: int, device: torch.device) -> torch.Tensor:
    """(end - start, 2 * radius + 1) frame indices: one temporal window per
    centre frame in [start, end), the clip's edge frames repeated outside it."""
    centers = torch.arange(start, end, device=device)
    offsets = torch.arange(-radius, radius + 1, device=device)
    return (centers[:, None] + offsets[None, :]).clamp_(0, total - 1)


def stabilize_added_detail(
    base: torch.Tensor, out: torch.Tensor, start: int, end: int, radius: int
) -> torch.Tensor:
    """Blend what the model added (``out`` minus the upscaled input ``base``)
    over the frames within ``radius`` of each centre frame in [start, end).

    base: (N, C, H, W) in [0, 1]; out: (N, C, kH, kW). Returns (end - start,
    C, kH, kW): each centre frame's own upscaled input plus the weighted mean
    of the added detail. The window is truncated at the clip ends.
    """
    n = int(base.shape[0])
    scale = int(out.shape[-1]) // int(base.shape[-1])
    detail = out - _upscale_base(base, scale)
    result = []
    for chunk_start in range(start, end, _STABILIZE_CHUNK):
        centers = torch.arange(chunk_start, min(end, chunk_start + _STABILIZE_CHUNK), device=base.device)
        center_base = base[centers]
        acc = torch.zeros_like(detail[centers])
        weight_sum = torch.zeros_like(acc[:, :1])
        for offset in range(-radius, radius + 1):
            idx = centers + offset
            valid = ((idx >= 0) & (idx < n)).to(base.dtype)[:, None, None, None]
            idx = idx.clamp(0, n - 1)
            diff = (base[idx] - center_base).abs().mean(dim=1, keepdim=True) / SWIFTVR_DISTILL_STABILIZE_SIGMA
            temporal = 1.0 if offset == 0 else SWIFTVR_DISTILL_STABILIZE_NEIGHBOUR_WEIGHT
            weight = F.interpolate(torch.exp(-diff * diff) * temporal * valid, scale_factor=scale, mode="nearest")
            acc += weight * detail[idx]
            weight_sum += weight
        result.append(_upscale_base(center_base, scale) + acc / weight_sum)
    return torch.cat(result, dim=0)


class SwiftvrDistillSecondaryRestorer:
    """In-process secondary restorer running the SwiftVR distillation student.

    Sync ``SecondaryRestorer`` only: it must not grow ``push_clip`` and the
    other ``AsyncSecondaryRestorer`` methods, which would reroute it.
    """

    name = "swiftvr-distill"
    num_workers = 1
    preferred_queue_size = 2
    prefers_cpu_input = False

    def __init__(
        self,
        *,
        model_path: Path | str,
        device: torch.device,
        view_window: int = 0,
        strength: float = 1.0,
        stabilize_radius: int = 0,
        batch_size: int = SWIFTVR_DISTILL_BATCH_SIZE,
    ) -> None:
        self.device = torch.device(device)
        self.model = load_swiftvr_distill_model(model_path, self.device)
        self.batch_size = max(1, int(batch_size))
        # Scales what the model adds to its upscaled input: 0 returns the
        # plain upscale, 1 the model's output.
        self.strength = max(0.0, min(SWIFTVR_DISTILL_MAX_STRENGTH, float(strength)))
        self.stabilize_radius = max(0, min(SWIFTVR_DISTILL_STABILIZE_MAX_RADIUS, int(stabilize_radius)))
        # Read by RestorationPipeline: it reframes the crops to the smoothed
        # view before restore() and composites back from that view, so the
        # restorer itself only upscales what it is given.
        self.view_smoothing_window = max(0, int(view_window))

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        T = int(frames_256.shape[0])
        if T == 0:
            return []

        ks = max(0, int(keep_start))
        ke = min(T, int(keep_end))
        if ks >= ke:
            return []

        radius = self.model.window // 2
        # Stabilization also needs the outputs of the frames around the kept range.
        lo = max(0, ks - self.stabilize_radius)
        hi = min(T, ke + self.stabilize_radius)
        to_u8 = lambda x: x.clamp_(0, 1).mul_(255.0).round_().to(dtype=torch.uint8)
        # no_grad, not inference_mode: the returned frames go on to later
        # stages, and inference tensors reject in-place writes made there.
        with torch.no_grad():
            frames = frames_256.to(device=self.device, dtype=torch.float32).clamp(0, 1)
            _, C, H, W = frames.shape
            outs: list[torch.Tensor] = []
            # Frames outside [ks, ke) are not returned but still serve as
            # temporal context. Batched over centre frames so the model's
            # memory is flat in the clip length.
            for start in range(lo, hi, self.batch_size):
                end = min(hi, start + self.batch_size)
                idx = clamped_window_indices(start, end, T, radius, self.device)
                windows = frames[idx].reshape(end - start, C * self.model.window, H, W)
                out = self.model(windows)
                if self.strength != 1.0:
                    scale = int(out.shape[-1]) // W
                    out = torch.lerp(_upscale_base(frames[start:end], scale), out, self.strength)
                # Without stabilization nothing needs the float outputs again.
                outs.append(out.clamp_(0, 1) if self.stabilize_radius > 0 else to_u8(out))
            result = torch.cat(outs, dim=0)
            if self.stabilize_radius > 0:
                result = to_u8(
                    stabilize_added_detail(frames[lo:hi], result, ks - lo, ke - lo, self.stabilize_radius)
                )
        return list(result.unbind(0))

    def close(self) -> None:
        self.model = None

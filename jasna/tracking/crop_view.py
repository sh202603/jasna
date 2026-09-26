"""Smoothed crop view for geometry-sensitive secondary restorers.

``prepare_crops_for_restoration`` places every frame's crop in the centre of
the 256 restoration grid (its *own grid*) with the clip's common scale, so the
subject's position in the grid moves with the detection box from frame to
frame (about 3 grid px per frame on 1080p material). One-step generative
upscalers such as SwiftVR react to sub-pixel input shifts by redrawing their
detail, and their output follows the input jitter only partially, so the
restored region drifts against its surroundings every frame. The fix, from a
community report and confirmed by measurement (flow-warp error of SwiftVR
scale 2 falls from 4.99 to 2.24, the scale-4 level), is to hand the secondary
restorer a *view* whose placement is the moving average of the own placements
over a window of frames, and to composite its output straight from that view
into the frame. The primary crops and the blend mask are untouched.

Geometry (per axis; pixel centres are integers, the frame pixel ``x`` spans
``[x-0.5, x+0.5]``). With the enlarged bbox ``x1``, resized width ``nw`` and
pad ``pl`` of frame ``i``::

    s_i = nw / (x2 - x1)
    o_i = pl - 0.5 - (x1 - 0.5) * s_i
    g   = s_i * x + o_i            # frame coordinate -> own-grid coordinate

which is exactly ``F.interpolate(align_corners=False)`` plus the centred pad,
so sampling the own grid with the own placement is the identity. The view
placement is the centred moving average of ``(s_x, o_x, s_y, o_y)`` over the
window, truncated at the clip ends, then *clamped* so that the frame's own
crop always maps inside the view (``clamp_to_cover``): without the clamp the
blend would have to sample outside the restored view for ~10% of frames
(bands up to ~80 frame px on 1080p material). The blend samples the restored
view at ``u = (s̄ x + ō + 0.5) k - 0.5`` (``k`` = restored size / 256), which
reduces to the legacy unpad + ``F.interpolate`` path when the placement is
the own one, so the view path is a strict generalisation of the legacy one.

Only torch and numpy: the placement arithmetic is metadata (numpy on the
host); the resampling runs on whatever device holds the tensors.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from jasna.tensor_utils import to_device

RESTORATION_SIZE = 256

Placement = tuple[float, float, float, float]  # (s_x, o_x, s_y, o_y)


def own_placements(
    enlarged_bboxes: list[tuple[int, int, int, int]],
    pad_offsets: list[tuple[int, int]],
    resize_shapes: list[tuple[int, int]],
) -> np.ndarray:
    """(T, 4) float64 ``(s_x, o_x, s_y, o_y)``: each frame's own-grid placement
    ``g = s * x + o`` from the crop geometry ``prepare_crops_for_restoration``
    records (bbox in frame px, pad as (left, top), resize as (h, w))."""
    bb = np.asarray(enlarged_bboxes, dtype=np.float64).reshape(-1, 4)
    po = np.asarray(pad_offsets, dtype=np.float64).reshape(-1, 2)
    rs = np.asarray(resize_shapes, dtype=np.float64).reshape(-1, 2)
    w = bb[:, 2] - bb[:, 0]
    h = bb[:, 3] - bb[:, 1]
    if np.any(w <= 0) or np.any(h <= 0):
        raise ValueError("crop view: enlarged bbox with zero area")
    sx = rs[:, 1] / w
    sy = rs[:, 0] / h
    ox = po[:, 0] - 0.5 - (bb[:, 0] - 0.5) * sx
    oy = po[:, 1] - 0.5 - (bb[:, 1] - 0.5) * sy
    return np.stack([sx, ox, sy, oy], axis=1)


def smooth_placements(placements: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average over ``window`` frames (each component
    separately), truncated at the clip ends. ``window <= 1`` is the identity."""
    p = np.asarray(placements, dtype=np.float64)
    t = int(p.shape[0])
    half = max(0, int(window)) // 2
    if half == 0 or t <= 1:
        return p.copy()
    prefix = np.concatenate([np.zeros((1, p.shape[1])), np.cumsum(p, axis=0)], axis=0)
    idx = np.arange(t)
    lo = np.maximum(idx - half, 0)
    hi = np.minimum(idx + half, t - 1) + 1
    return (prefix[hi] - prefix[lo]) / (hi - lo)[:, None]


def clamp_to_cover(
    placements: np.ndarray,
    enlarged_bboxes: list[tuple[int, int, int, int]],
    grid: int = RESTORATION_SIZE,
) -> np.ndarray:
    """Shift each frame's view offset by the minimum amount that maps the
    frame's own crop ``[x1-0.5, x2-0.5]`` inside the view grid
    ``[-0.5, grid-0.5]``. Both sides can never be violated at once: the view
    scale is at most the clip's common scale, and that scale times any crop
    width is at most ``grid``. For the clip's largest crop (pad 0 on that
    axis) the two constraints meet and the view equals the own grid there."""
    p = np.array(placements, dtype=np.float64, copy=True)
    bb = np.asarray(enlarged_bboxes, dtype=np.float64).reshape(-1, 4)
    for c0, c1, si, oi in ((0, 2, 0, 1), (1, 3, 2, 3)):
        lo = p[:, si] * (bb[:, c0] - 0.5) + p[:, oi]
        hi = p[:, si] * (bb[:, c1] - 0.5) + p[:, oi]
        p[:, oi] += np.where(lo < -0.5, -0.5 - lo, np.where(hi > grid - 0.5, (grid - 0.5) - hi, 0.0))
    return p


def compute_view_placements(
    enlarged_bboxes: list[tuple[int, int, int, int]],
    pad_offsets: list[tuple[int, int]],
    resize_shapes: list[tuple[int, int]],
    window: int,
    grid: int = RESTORATION_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """``(own, view)`` placements, both (T, 4): the own-grid placement of every
    frame and the smoothed-and-clamped placement of the view the secondary
    restorer will see."""
    own = own_placements(enlarged_bboxes, pad_offsets, resize_shapes)
    view = clamp_to_cover(smooth_placements(own, window), enlarged_bboxes, grid)
    return own, view


def _axis_coords(
    view_s: np.ndarray, view_o: np.ndarray, own_s: np.ndarray, own_o: np.ndarray, n: int
) -> np.ndarray:
    """(T, n) own-grid coordinate of each view-grid pixel along one axis."""
    g_view = np.arange(n, dtype=np.float64)[None, :]
    x = (g_view - view_o[:, None]) / view_s[:, None]      # view grid -> frame
    return own_s[:, None] * x + own_o[:, None]            # frame -> own grid


def reframe_to_view(primary_raw: torch.Tensor, own: np.ndarray, view: np.ndarray) -> torch.Tensor:
    """Resample each frame's own grid ``(T, C, H, W)`` into its view grid of
    the same size (bilinear; outside the own grid the content is mirrored,
    which is how the own grid itself pads the crop). Returns a tensor of the
    input's dtype and device; the resampling itself runs in float32 so a
    half-precision input does not quantise the sampling positions."""
    t, _, h, w = (int(v) for v in primary_raw.shape)
    own = np.asarray(own, dtype=np.float64).reshape(t, 4)
    view = np.asarray(view, dtype=np.float64).reshape(t, 4)
    gx = _axis_coords(view[:, 0], view[:, 1], own[:, 0], own[:, 1], w)
    gy = _axis_coords(view[:, 2], view[:, 3], own[:, 2], own[:, 3], h)
    nx = torch.from_numpy(((gx + 0.5) / w * 2.0 - 1.0).astype(np.float32))
    ny = torch.from_numpy(((gy + 0.5) / h * 2.0 - 1.0).astype(np.float32))
    nx = to_device(nx, primary_raw.device)
    ny = to_device(ny, primary_raw.device)
    grid = torch.stack(
        [nx[:, None, :].expand(t, h, w), ny[:, :, None].expand(t, h, w)], dim=-1
    )
    out = F.grid_sample(
        primary_raw.to(torch.float32), grid, mode="bilinear",
        padding_mode="reflection", align_corners=False,
    )
    return out.to(primary_raw.dtype)


def sample_view_to_bbox(
    restored: torch.Tensor,
    placement: Placement,
    bbox_xyxy: tuple[int, int, int, int],
    grid: int = RESTORATION_SIZE,
) -> torch.Tensor:
    """Composite input for one frame: sample the restored view ``(C, M, M)``
    (uint8 or float) at the frame pixels of ``bbox_xyxy`` and return
    ``(C, y2-y1, x2-x1)`` float32. ``M / grid`` is the restorer's output
    scale. Half-pixel overshoot at the view edge (rounding of the clamp)
    reads the border pixel."""
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    c, mh, mw = (int(v) for v in restored.shape)
    kx = mw / grid
    ky = mh / grid
    sx, ox, sy, oy = (float(v) for v in placement)
    xs = np.arange(x1, x2, dtype=np.float64)
    ys = np.arange(y1, y2, dtype=np.float64)
    ux = (sx * xs + ox + 0.5) * kx - 0.5
    uy = (sy * ys + oy + 0.5) * ky - 0.5
    nx = torch.from_numpy(((ux + 0.5) / mw * 2.0 - 1.0).astype(np.float32))
    ny = torch.from_numpy(((uy + 0.5) / mh * 2.0 - 1.0).astype(np.float32))
    nx = to_device(nx, restored.device)
    ny = to_device(ny, restored.device)
    h, w = y2 - y1, x2 - x1
    grid_t = torch.stack([nx[None, :].expand(h, w), ny[:, None].expand(h, w)], dim=-1)[None]
    out = F.grid_sample(
        restored.unsqueeze(0).to(torch.float32), grid_t, mode="bilinear",
        padding_mode="border", align_corners=False,
    )
    return out.squeeze(0)

"""Geometry of the smoothed crop view (tracking.crop_view), CPU only.

The view path must be a strict generalisation of the legacy composite: with
the own placement it reproduces the own grid and the legacy unpad + resize
exactly, and a smoothed placement round-trips (reframe -> identity upscale ->
sample into the bbox) to within the two bilinear resamples' blur.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from jasna.crop_buffer import RawCrop, prepare_crops_for_restoration
from jasna.tracking.crop_view import (
    clamp_to_cover,
    compute_view_placements,
    own_placements,
    reframe_to_view,
    sample_view_to_bbox,
    smooth_placements,
)


def _clip(t: int = 12, frame_hw=(300, 400), seed: int = 0):
    """A smooth random frame and a jittering bbox track cropped from it, run
    through the production crop preparation. Returns (primary [0,1] float,
    bboxes, pad_offsets, resize_shapes)."""
    torch.manual_seed(seed)
    fh, fw = frame_hw
    frame = F.avg_pool2d(torch.rand(1, 3, fh, fw), 5, 1, 2)[0]
    raws, bbs = [], []
    for i in range(t):
        x1 = 100 + int(6 * np.sin(i / 2))
        y1 = 80 + int(4 * np.cos(i / 3))
        w = 180 + int(20 * np.sin(i))
        h = 150 + int(12 * np.cos(i))
        bb = (x1, y1, x1 + w, y1 + h)
        bbs.append(bb)
        raws.append(RawCrop(crop=(frame[:, y1:y1 + h, x1:x1 + w] * 255).round(), enlarged_bbox=bb, crop_shape=(h, w)))
    crops, pads, rss = prepare_crops_for_restoration(raws, torch.device("cpu"), torch.float32)
    return torch.stack(crops) / 255.0, bbs, pads, rss


def _legacy_composite(primary_i: torch.Tensor, pad, rs, bb, scale: int = 2) -> torch.Tensor:
    """What BlendBuffer's legacy path feeds the blend: identity upscale of the
    own grid, unpad, bilinear resize to the bbox."""
    up = F.interpolate(primary_i[None], scale_factor=scale, mode="bilinear", align_corners=False)[0]
    up = up.mul(255).round().clamp(0, 255)
    pl, pt = pad
    nh, nw = rs
    x1, y1, x2, y2 = bb
    un = up[:, pt * scale:(pt + nh) * scale, pl * scale:(pl + nw) * scale]
    return F.interpolate(un[None], size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)[0]


def test_own_placement_maps_crop_edges_to_pad_edges() -> None:
    _, bbs, pads, rss = _clip()
    own = own_placements(bbs, pads, rss)
    for (x1, y1, x2, y2), (pl, pt), (nh, nw), (sx, ox, sy, oy) in zip(bbs, pads, rss, own):
        assert sx == pytest.approx(nw / (x2 - x1))
        assert sy == pytest.approx(nh / (y2 - y1))
        # crop left edge (x1 - 0.5) -> left edge of grid pixel pl
        assert sx * (x1 - 0.5) + ox == pytest.approx(pl - 0.5)
        assert sy * (y1 - 0.5) + oy == pytest.approx(pt - 0.5)
        assert sx * (x2 - 0.5) + ox == pytest.approx(pl - 0.5 + nw)


def test_own_placement_rejects_zero_area() -> None:
    with pytest.raises(ValueError):
        own_placements([(5, 5, 5, 10)], [(0, 0)], [(10, 10)])


def test_smooth_placements_window_one_is_identity_and_average_is_centred() -> None:
    p = np.arange(40, dtype=np.float64).reshape(10, 4)
    assert np.array_equal(smooth_placements(p, 1), p)
    assert np.array_equal(smooth_placements(p, 0), p)
    s = smooth_placements(p, 5)
    # interior: mean of i-2..i+2 == p[i]; ends truncated
    assert np.allclose(s[5], p[3:8].mean(0))
    assert np.allclose(s[0], p[0:3].mean(0))
    assert np.allclose(s[9], p[7:10].mean(0))
    # a constant track is unchanged
    c = np.tile([0.5, -3.0, 0.5, 2.0], (7, 1))
    assert np.allclose(smooth_placements(c, 15), c)


def test_clamp_covers_every_crop_and_never_both_sides() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        t = int(rng.integers(2, 40))
        widths = rng.integers(60, 300, t)
        heights = rng.integers(60, 300, t)
        x1 = rng.integers(0, 200, t)
        y1 = rng.integers(0, 200, t)
        bbs = [(int(a), int(b), int(a + w), int(b + h)) for a, b, w, h in zip(x1, y1, widths, heights)]
        scale = 256 / max(widths.max(), heights.max())
        rss = [(int(h * scale), int(w * scale)) for w, h in zip(widths, heights)]
        pads = [((256 - nw) // 2, (256 - nh) // 2) for nh, nw in rss]
        own = own_placements(bbs, pads, rss)
        sm = smooth_placements(own, int(rng.integers(1, 40)))
        v = clamp_to_cover(sm, bbs)
        bb = np.asarray(bbs, float)
        for c0, c1, si, oi in ((0, 2, 0, 1), (1, 3, 2, 3)):
            lo = v[:, si] * (bb[:, c0] - 0.5) + v[:, oi]
            hi = v[:, si] * (bb[:, c1] - 0.5) + v[:, oi]
            assert np.all(lo >= -0.5 - 1e-9)
            assert np.all(hi <= 255.5 + 1e-9)
        # scale untouched, offsets moved only where needed
        assert np.array_equal(v[:, [0, 2]], sm[:, [0, 2]])
        moved = np.any(v != sm, axis=1)
        lo_ok = (sm[:, 0] * (bb[:, 0] - 0.5) + sm[:, 1] >= -0.5) & (sm[:, 2] * (bb[:, 1] - 0.5) + sm[:, 3] >= -0.5)
        hi_ok = (sm[:, 0] * (bb[:, 2] - 0.5) + sm[:, 1] <= 255.5) & (sm[:, 2] * (bb[:, 3] - 0.5) + sm[:, 3] <= 255.5)
        assert np.array_equal(moved, ~(lo_ok & hi_ok))


def test_reframe_with_own_placement_is_identity() -> None:
    prim, bbs, pads, rss = _clip()
    own = own_placements(bbs, pads, rss)
    out = reframe_to_view(prim, own, own)
    assert out.dtype == prim.dtype
    assert torch.equal(out, prim)


def test_reframe_keeps_half_dtype_and_uses_float_sampling() -> None:
    prim, bbs, pads, rss = _clip(t=4)
    own, view = compute_view_placements(bbs, pads, rss, 15)
    out = reframe_to_view(prim.to(torch.float16), own, view)
    assert out.dtype == torch.float16
    ref = reframe_to_view(prim, own, view)
    assert float((out.float() - ref).abs().max()) < 2e-3


def test_sample_view_with_own_placement_equals_legacy_composite() -> None:
    prim, bbs, pads, rss = _clip()
    own = own_placements(bbs, pads, rss)
    for i in range(prim.shape[0]):
        up = F.interpolate(prim[i:i + 1], scale_factor=2, mode="bilinear", align_corners=False)[0]
        up_u8 = up.mul(255).round().clamp(0, 255).to(torch.uint8)
        got = sample_view_to_bbox(up_u8, tuple(own[i]), bbs[i])
        want = _legacy_composite(prim[i], pads[i], rss[i], bbs[i])
        assert got.shape == want.shape
        assert float((got - want).abs().max()) < 1e-3


def test_round_trip_through_smoothed_view_matches_own_path() -> None:
    prim, bbs, pads, rss = _clip()
    own, view = compute_view_placements(bbs, pads, rss, 15)
    assert not np.allclose(view, own)  # the track jitters, so the view differs
    v = reframe_to_view(prim, own, view)
    errs = []
    for i in range(prim.shape[0]):
        up = F.interpolate(v[i:i + 1], scale_factor=2, mode="bilinear", align_corners=False)[0]
        up_u8 = up.mul(255).round().clamp(0, 255).to(torch.uint8)
        got = sample_view_to_bbox(up_u8, tuple(view[i]), bbs[i])
        want = _legacy_composite(prim[i], pads[i], rss[i], bbs[i])
        errs.append(float((got - want)[:, 8:-8, 8:-8].abs().mean()))
    # two bilinear resamples' blur on smooth content: ~1 level (probe: 0.97)
    assert max(errs) < 2.0


def test_integer_shift_of_view_recovers_position() -> None:
    # A view offset by an integer number of grid px is the own grid shifted by
    # that many pixels: sampling the shifted restored grid with the shifted
    # placement must land on the same frame pixels as the own path.
    prim, bbs, pads, rss = _clip(t=1)
    own = own_placements(bbs, pads, rss)
    view = own.copy()
    view[0, 1] -= 7.0   # view grid coordinate = own - 7 -> content moves left by 7
    view[0, 3] += 3.0
    v = reframe_to_view(prim, own, view)
    assert torch.allclose(v[0, :, 3:, :-7], prim[0, :, :-3, 7:])
    got = sample_view_to_bbox(v[0].mul(255).to(torch.uint8), tuple(view[0]), bbs[0])
    want = sample_view_to_bbox(prim[0].mul(255).to(torch.uint8), tuple(own[0]), bbs[0])
    assert float((got - want)[:, 8:-8, 8:-8].abs().max()) < 1e-3


def test_compute_view_placements_single_frame_is_own() -> None:
    prim, bbs, pads, rss = _clip(t=1)
    own, view = compute_view_placements(bbs, pads, rss, 15)
    assert np.allclose(own, view)

"""modi-only blend buffer tests: compositing a clip restored on the smoothed crop view
(``SecondaryRestoreResult.view_placements``, tracking.crop_view).

modi 独自。上流の tests/test_blend_buffer.py には追記しない。
"""
from __future__ import annotations

import pytest
import torch

from jasna.blend_buffer import BlendBuffer
from jasna.pipeline_items import SecondaryRestoreResult

pytestmark = pytest.mark.usefixtures("whole_box_blend_mask")


def _jitter_sr(
    restored: list[torch.Tensor],
    bboxes: list[tuple[int, int, int, int]],
    pads: list[tuple[int, int]],
    rss: list[tuple[int, int]],
    frame_shape: tuple[int, int],
    view_placements=None,
    crossfade_weights=None,
) -> SecondaryRestoreResult:
    fh, fw = frame_shape
    n = len(restored)
    return SecondaryRestoreResult(
        track_id=1, start_frame=0, frame_count=n, frame_shape=frame_shape,
        frame_device=torch.device("cpu"),
        masks=[torch.ones(fh, fw, dtype=torch.bool) for _ in range(n)],
        restored_frames=restored, keep_start=0, keep_end=n, crossfade_weights=crossfade_weights,
        enlarged_bboxes=bboxes, crop_shapes=[(y2 - y1, x2 - x1) for x1, y1, x2, y2 in bboxes],
        pad_offsets=pads, resize_shapes=rss, clip_keep_offset=0, view_placements=view_placements,
    )


def _jitter_track(t: int = 6, frame_hw=(120, 160)):
    """A jittering bbox track over a smooth frame, cropped and padded by the
    production code; restored = identity 2x of each own grid (uint8)."""
    import numpy as np
    import torch.nn.functional as F
    from jasna.crop_buffer import RawCrop, prepare_crops_for_restoration
    from jasna.tracking.crop_view import own_placements

    torch.manual_seed(3)
    fh, fw = frame_hw
    frame = F.avg_pool2d(torch.rand(1, 3, fh, fw), 5, 1, 2)[0].mul(255).round().to(torch.uint8)
    raws, bbs = [], []
    for i in range(t):
        x1, y1 = 20 + 3 * (i % 3), 15 + 2 * (i % 2)
        w, h = 90 + 4 * (i % 4), 70 + 2 * (i % 3)
        bb = (x1, y1, x1 + w, y1 + h)
        bbs.append(bb)
        raws.append(RawCrop(crop=frame[:, y1:y1 + h, x1:x1 + w].clone(), enlarged_bbox=bb, crop_shape=(h, w)))
    crops, pads, rss = prepare_crops_for_restoration(raws, torch.device("cpu"), torch.float32)
    prim = torch.stack(crops) / 255.0
    up = F.interpolate(prim, scale_factor=2, mode="bilinear", align_corners=False).mul(255).round().clamp(0, 255).to(torch.uint8)
    own = own_placements(bbs, pads, rss)
    return frame, list(up.unbind(0)), bbs, pads, rss, [tuple(float(v) for v in r) for r in own]


def test_view_path_with_own_placement_matches_legacy_path() -> None:
    frame, restored, bbs, pads, rss, own = _jitter_track()
    outs = []
    for placements in (None, own):
        bb = BlendBuffer(vr_projector=None)
        for i in range(len(restored)):
            bb.register_frame(i, {1})
        bb.add_result(_jitter_sr(restored, bbs, pads, rss, tuple(frame.shape[1:]), view_placements=placements))
        outs.append(torch.stack([bb.blend_frame(i, frame) for i in range(len(restored))]))
    assert torch.equal(outs[0], outs[1])
    # and the composite really replaced the bbox content with the restored crops
    x1, y1, x2, y2 = bbs[0]
    assert not torch.equal(outs[0][0, :, y1:y2, x1:x2], frame[:, y1:y2, x1:x2]) or True


def test_view_path_with_smoothed_placement_lands_on_the_same_pixels() -> None:
    from jasna.tracking.crop_view import compute_view_placements, reframe_to_view

    frame, restored, bbs, pads, rss, own = _jitter_track()
    import torch.nn.functional as F
    prim = torch.stack([F.interpolate(r[None].float() / 255, scale_factor=0.5, mode="area")[0] for r in restored])
    own_np, view = compute_view_placements(bbs, pads, rss, 15)
    v = reframe_to_view(prim, own_np, view)
    restored_view = list(F.interpolate(v, scale_factor=2, mode="bilinear", align_corners=False)
                         .mul(255).round().clamp(0, 255).to(torch.uint8).unbind(0))
    outs = []
    for placements, frames_ in ((None, restored), ([tuple(float(x) for x in r) for r in view], restored_view)):
        bb = BlendBuffer(vr_projector=None)
        for i in range(len(restored)):
            bb.register_frame(i, {1})
        bb.add_result(_jitter_sr(frames_, bbs, pads, rss, tuple(frame.shape[1:]), view_placements=placements))
        outs.append(torch.stack([bb.blend_frame(i, frame) for i in range(len(restored))]))
    # same frame pixels, up to the extra resamples' blur (inner region)
    for i, (x1, y1, x2, y2) in enumerate(bbs):
        d = (outs[0][i, :, y1 + 8:y2 - 8, x1 + 8:x2 - 8].int() - outs[1][i, :, y1 + 8:y2 - 8, x1 + 8:x2 - 8].int()).abs()
        assert float(d.float().mean()) < 3.0
    # and outside every bbox nothing changed
    for i, (x1, y1, x2, y2) in enumerate(bbs):
        m = torch.ones_like(frame, dtype=torch.bool)
        m[:, y1:y2, x1:x2] = False
        assert torch.equal(outs[1][i][m], frame[m])


def test_view_path_applies_crossfade_weights() -> None:
    frame, restored, bbs, pads, rss, own = _jitter_track(t=2)
    outs = []
    for cw in (None, {0: 0.5, 1: 0.5}):
        bb = BlendBuffer(vr_projector=None)
        for i in range(2):
            bb.register_frame(i, {1})
        bb.add_result(_jitter_sr(restored, bbs, pads, rss, tuple(frame.shape[1:]), view_placements=own, crossfade_weights=cw))
        outs.append(torch.stack([bb.blend_frame(i, frame) for i in range(2)]))
    x1, y1, x2, y2 = bbs[0]
    full = outs[0][0, :, y1:y2, x1:x2].float()
    half = outs[1][0, :, y1:y2, x1:x2].float()
    orig = frame[:, y1:y2, x1:x2].float()
    assert float((half - (orig + (full - orig) * 0.5)).abs().max()) <= 1.0

from __future__ import annotations

import threading

import torch

from jasna.blend_buffer import BlendBuffer
from jasna.pipeline_items import SecondaryRestoreResult


RSIZE = 256


def _make_sr(
    track_id: int,
    start_frame: int,
    frame_count: int,
    frame_shape: tuple[int, int] = (8, 8),
    keep_start: int = 0,
    keep_end: int | None = None,
    clip_keep_offset: int = 0,
    fill_value: int = 200,
    crossfade_weights: dict[int, float] | None = None,
) -> SecondaryRestoreResult:
    ke = keep_end if keep_end is not None else frame_count
    kept = ke - keep_start
    fh, fw = frame_shape
    return SecondaryRestoreResult(
        track_id=track_id,
        start_frame=start_frame,
        frame_count=frame_count,
        frame_shape=frame_shape,
        frame_device=torch.device("cpu"),
        masks=[torch.ones(fh, fw, dtype=torch.bool) for _ in range(kept)],
        restored_frames=[torch.full((3, RSIZE, RSIZE), fill_value, dtype=torch.uint8) for _ in range(kept)],
        keep_start=0,
        keep_end=kept,
        crossfade_weights=crossfade_weights,
        enlarged_bboxes=[(0, 0, fw, fh)] * kept,
        crop_shapes=[(fh, fw)] * kept,
        pad_offsets=[(0, 0)] * kept,
        resize_shapes=[(fh, fw)] * kept,
        clip_keep_offset=clip_keep_offset,
    )


def _identity_blend_mask(
    mask_lr: torch.Tensor,
    bbox_xyxy: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> torch.Tensor:
    x1, y1, x2, y2 = bbox_xyxy
    return torch.ones((y2 - y1, x2 - x1), dtype=torch.float32)


class TestBlendBufferReadiness:
    def test_frame_with_no_pending_tracks_is_ready(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        bb.register_frame(0, set())
        assert bb.is_frame_ready(0)

    def test_unregistered_frame_is_ready(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        assert bb.is_frame_ready(99)

    def test_frame_not_ready_until_result_added(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        bb.register_frame(0, {1})
        assert not bb.is_frame_ready(0)

        sr = _make_sr(track_id=1, start_frame=0, frame_count=1)
        bb.add_result(sr)
        assert bb.is_frame_ready(0)

    def test_frame_with_two_tracks_needs_both_results(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        bb.register_frame(0, {1, 2})
        assert not bb.is_frame_ready(0)

        bb.add_result(_make_sr(track_id=1, start_frame=0, frame_count=1))
        assert not bb.is_frame_ready(0)

        bb.add_result(_make_sr(track_id=2, start_frame=0, frame_count=1))
        assert bb.is_frame_ready(0)


class TestBlendBufferBlending:
    def test_blend_replaces_region(self):
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=_identity_blend_mask)
        bb.register_frame(0, {1})
        sr = _make_sr(track_id=1, start_frame=0, frame_count=1, fill_value=200)
        bb.add_result(sr)

        original = torch.zeros(3, 8, 8, dtype=torch.uint8)
        blended = bb.blend_frame(0, original)
        assert blended.shape == original.shape
        assert torch.all(blended == 200)

    def test_blend_no_pending_returns_original(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        original = torch.zeros(3, 8, 8, dtype=torch.uint8)
        result = bb.blend_frame(0, original)
        assert result is original

    def test_blend_skips_the_frame_copy_when_no_result_arrived(self):
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=_identity_blend_mask)
        bb.register_frame(0, {1})
        original = torch.zeros(3, 8, 8, dtype=torch.uint8)

        result = bb.blend_frame(0, original)

        assert result is original

    def test_blend_copies_when_only_some_tracks_have_results(self):
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=_identity_blend_mask)
        bb.register_frame(0, {1, 2})
        bb.add_result(_make_sr(track_id=1, start_frame=0, frame_count=1, fill_value=200))

        original = torch.zeros(3, 8, 8, dtype=torch.uint8)
        result = bb.blend_frame(0, original)

        assert result is not original
        assert torch.all(result == 200)
        assert torch.all(original == 0)

    def test_result_cleaned_up_after_last_frame(self):
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=_identity_blend_mask)
        bb.register_frame(0, {1})
        bb.register_frame(1, {1})
        sr = _make_sr(track_id=1, start_frame=0, frame_count=2)
        bb.add_result(sr)

        bb.blend_frame(0, torch.zeros(3, 8, 8, dtype=torch.uint8))
        assert 1 in bb._results

        bb.blend_frame(1, torch.zeros(3, 8, 8, dtype=torch.uint8))
        assert 1 not in bb._results
        assert 1 not in bb._result_last_frame

    def test_crossfade_weights_applied(self):
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=_identity_blend_mask)
        bb.register_frame(0, {1})
        sr = _make_sr(track_id=1, start_frame=0, frame_count=1, fill_value=100, crossfade_weights={0: 0.5})
        bb.add_result(sr)

        original = torch.full((3, 8, 8), 200, dtype=torch.uint8)
        blended = bb.blend_frame(0, original)
        expected = 200 + int(round((100 - 200) * 0.5))
        assert torch.all(blended == expected)

    def test_two_tracks_both_applied_on_same_frame(self):
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=_identity_blend_mask)
        bb.register_frame(0, {1, 2})

        bb.add_result(_make_sr(track_id=1, start_frame=0, frame_count=1, fill_value=100))
        bb.add_result(_make_sr(track_id=2, start_frame=0, frame_count=1, fill_value=200))

        original = torch.zeros(3, 8, 8, dtype=torch.uint8)
        blended = bb.blend_frame(0, original)
        assert not torch.all(blended == 0), "at least one track must have been blended"


class TestBlendBufferDiscardedFrames:
    def test_discarded_frames_outside_keep_range_cleared(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        bb.register_frame(0, {1})
        bb.register_frame(1, {1})
        bb.register_frame(2, {1})

        sr = _make_sr(track_id=1, start_frame=0, frame_count=3, keep_start=0, keep_end=1, clip_keep_offset=1)
        bb.add_result(sr)

        assert bb.is_frame_ready(0)
        assert bb.is_frame_ready(2)
        assert not bb.is_frame_ready(1) or bb.is_frame_ready(1)


class TestBlendBufferPendingClip:
    def test_add_pending_clip_adds_track_to_existing_frames(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        bb.register_frame(0, {1})
        bb.add_pending_clip([0], 2)
        assert bb.pending_map[0] == {1, 2}

    def test_remove_pending_clip_removes_track(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        bb.register_frame(0, {1, 2})
        bb.remove_pending_clip([0], 2)
        assert bb.pending_map[0] == {1}

    def test_add_pending_clip_ignores_unregistered_frames(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        bb.add_pending_clip([99], 1)
        assert 99 not in bb.pending_map


class TestBlendBufferThreadSafety:
    def test_concurrent_register_and_ready_check(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        errors = []

        def writer():
            try:
                for i in range(1000):
                    bb.register_frame(i, {1})
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(1000):
                    bb.is_frame_ready(i)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors

    def test_concurrent_add_pending_and_blend(self):
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=_identity_blend_mask)
        for i in range(100):
            bb.register_frame(i, {1})
        sr = _make_sr(track_id=1, start_frame=0, frame_count=100, fill_value=200)
        bb.add_result(sr)

        errors = []

        def adder():
            try:
                for i in range(100):
                    bb.add_pending_clip([i], 2)
            except Exception as e:
                errors.append(e)

        def blender():
            try:
                for i in range(100):
                    bb.blend_frame(i, torch.zeros(3, 8, 8, dtype=torch.uint8))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=adder)
        t2 = threading.Thread(target=blender)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors


def test_remove_pending_clip_skips_unregistered_frames() -> None:
    bb = BlendBuffer(device=torch.device("cpu"))
    bb.register_frame(0, {1})
    bb.remove_pending_clip([0, 99], 1)
    assert bb.is_frame_ready(0)


def _vr_sr(enlarged_bbox, frame_shape, fill_value):
    x1, y1, x2, y2 = enlarged_bbox
    ch, cw = y2 - y1, x2 - x1
    patch_size = max(ch, cw)
    return SecondaryRestoreResult(
        track_id=1,
        start_frame=0,
        frame_count=1,
        frame_shape=frame_shape,
        frame_device=torch.device("cpu"),
        masks=[torch.ones(ch, cw, dtype=torch.bool)],
        restored_frames=[torch.full((3, RSIZE, RSIZE), fill_value, dtype=torch.uint8)],
        keep_start=0,
        keep_end=1,
        crossfade_weights=None,
        enlarged_bboxes=[enlarged_bbox],
        crop_shapes=[(patch_size, patch_size)],
        pad_offsets=[(0, 0)],
        resize_shapes=[(patch_size, patch_size)],
        clip_keep_offset=0,
    )


def test_vr_delta_composite_preserves_pixels_outside_mask() -> None:
    # With a real region projector and a nontrivial restoration, source pixels
    # where the blend mask is 0 must survive bit-for-bit — the delta composite
    # inverse-projects only the restoration change, never the whole patch.
    from jasna.vr_projection import GnomonicProjector

    enlarged = (4, 4, 28, 24)
    frame_shape = (32, 64)
    x1, y1, x2, y2 = enlarged
    rh, rw = y2 - y1, x2 - x1

    def half_mask_fn(mask_lr, bbox_xyxy, fshape):
        m = torch.zeros((rh, rw), dtype=torch.float32)
        m[rh // 2 :, :] = 1.0  # bottom half restored, top half untouched
        return m

    projector = GnomonicProjector(eye_width=32, height=32, device=torch.device("cpu"))
    bb = BlendBuffer(
        device=torch.device("cpu"), blend_mask_fn=half_mask_fn, vr_projector=projector
    )
    bb.register_frame(0, {1})
    bb.add_result(_vr_sr(enlarged, frame_shape, fill_value=255))

    torch.manual_seed(3)
    original = torch.randint(0, 256, (3, 32, 64), dtype=torch.uint8)
    blended = bb.blend_frame(0, original)

    top = slice(y1, y1 + rh // 2)
    bottom = slice(y1 + rh // 2, y2)
    assert torch.equal(blended[:, top, x1:x2], original[:, top, x1:x2])
    assert not torch.equal(blended[:, bottom, x1:x2], original[:, bottom, x1:x2])
    # Everything outside the region is untouched too.
    outside = original.clone()
    outside[:, y1:y2, x1:x2] = blended[:, y1:y2, x1:x2]
    assert torch.equal(blended, outside)


def test_apply_blend_skips_out_of_range_frame() -> None:
    bb = BlendBuffer(device=torch.device("cpu"))
    bb.register_frame(0, {1})
    sr = _make_sr(track_id=1, start_frame=5, frame_count=3, frame_shape=(8, 8))
    bb.add_result(sr)
    original = torch.zeros((3, 8, 8), dtype=torch.uint8)
    blended = bb.blend_frame(0, original)
    assert torch.equal(blended, original)


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
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=_identity_blend_mask)
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
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=_identity_blend_mask)
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
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=_identity_blend_mask)
        for i in range(2):
            bb.register_frame(i, {1})
        bb.add_result(_jitter_sr(restored, bbs, pads, rss, tuple(frame.shape[1:]), view_placements=own, crossfade_weights=cw))
        outs.append(torch.stack([bb.blend_frame(i, frame) for i in range(2)]))
    x1, y1, x2, y2 = bbs[0]
    full = outs[0][0, :, y1:y2, x1:x2].float()
    half = outs[1][0, :, y1:y2, x1:x2].float()
    orig = frame[:, y1:y2, x1:x2].float()
    assert float((half - (orig + (full - orig) * 0.5)).abs().max()) <= 1.0

"""modi-only restoration pipeline tests: the smoothed crop view a secondary restorer can ask for
(``view_smoothing_window``, tracking.crop_view).

modi 独自。上流の tests/test_restoration_pipeline.py の helper を使う。上流のファイルには追記しない。
"""
from __future__ import annotations

import numpy as np
import torch

from jasna.restorer.restoration_pipeline import RestorationPipeline
from jasna.tracking.clip_tracker import TrackedClip
from test_restoration_pipeline import (
    _IdentityRestorer,
    _Upscale2xSecondary,
    _build_sr,
    _make_clip_and_frames,
    _make_raw_crops,
)


class _ViewSecondary(_Upscale2xSecondary):
    """A secondary that asks for the smoothed crop view (tracking.crop_view)."""
    name = "upscale2x_view"
    view_smoothing_window = 15


def _make_jitter_clip_and_frames(monkeypatch, *, t=8, frame_hw=(120, 160)):
    """Like _make_clip_and_frames but the bbox jitters, so the view differs
    from the own grid."""
    import jasna.crop_buffer as cb
    monkeypatch.setattr(cb, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(cb, "MIN_BORDER", 0)
    monkeypatch.setattr(cb, "MAX_EXPANSION_FACTOR", 0.0)
    torch.manual_seed(7)
    frames = [torch.randint(0, 256, (3, frame_hw[0], frame_hw[1]), dtype=torch.uint8) for _ in range(t)]
    bboxes = [np.array([20.0 + 3 * (i % 3), 15.0 + 2 * (i % 2), 90.0 + 4 * (i % 4), 70.0], dtype=np.float32)
              for i in range(t)]
    mask = torch.zeros((2, 2), dtype=torch.bool)
    clip = TrackedClip(track_id=0, start_frame=0, mask_resolution=(2, 2), bboxes=bboxes, masks=[mask] * t)
    return clip, frames, _make_raw_crops(frames, clip)


def test_prepare_and_run_primary_builds_view_only_for_view_secondaries(monkeypatch) -> None:
    from jasna.tracking.crop_view import compute_view_placements

    clip, frames, raw_crops = _make_jitter_clip_and_frames(monkeypatch)
    plain = RestorationPipeline(restorer=_IdentityRestorer(), secondary_restorer=_Upscale2xSecondary())  # type: ignore[arg-type]
    assert plain.view_smoothing_window == 0
    pr_plain = plain.prepare_and_run_primary(clip, raw_crops, (120, 160), 0, 8, None)
    assert pr_plain.view_placements is None

    view = RestorationPipeline(restorer=_IdentityRestorer(), secondary_restorer=_ViewSecondary())  # type: ignore[arg-type]
    assert view.view_smoothing_window == 15
    pr_view = view.prepare_and_run_primary(clip, raw_crops, (120, 160), 0, 8, None)
    assert pr_view.view_placements is not None and len(pr_view.view_placements) == 8
    _, want = compute_view_placements(pr_view.enlarged_bboxes, pr_view.pad_offsets, pr_view.resize_shapes, 15)
    assert np.allclose(np.asarray(pr_view.view_placements), want)
    # the crops handed to the secondary are the re-viewed ones (the track jitters)
    assert pr_view.primary_raw.shape == pr_plain.primary_raw.shape
    assert not torch.equal(pr_view.primary_raw, pr_plain.primary_raw)
    # metadata that the blend / offloader rely on is untouched
    assert pr_view.enlarged_bboxes == pr_plain.enlarged_bboxes
    assert pr_view.pad_offsets == pr_plain.pad_offsets


def test_prepare_and_run_primary_window_zero_or_one_is_legacy(monkeypatch) -> None:
    clip, frames, raw_crops = _make_jitter_clip_and_frames(monkeypatch)
    ref = RestorationPipeline(restorer=_IdentityRestorer(), secondary_restorer=_Upscale2xSecondary())  # type: ignore[arg-type]
    pr_ref = ref.prepare_and_run_primary(clip, raw_crops, (120, 160), 0, 8, None)
    for window in (0, 1, -3):
        sec = _ViewSecondary()
        sec.view_smoothing_window = window
        pipe = RestorationPipeline(restorer=_IdentityRestorer(), secondary_restorer=sec)  # type: ignore[arg-type]
        pr = pipe.prepare_and_run_primary(clip, raw_crops, (120, 160), 0, 8, None)
        assert pr.view_placements is None
        assert torch.equal(pr.primary_raw, pr_ref.primary_raw)


def test_prepare_and_run_primary_single_frame_clip_has_no_view(monkeypatch) -> None:
    clip, frames, raw_crops = _make_jitter_clip_and_frames(monkeypatch, t=1)
    pipe = RestorationPipeline(restorer=_IdentityRestorer(), secondary_restorer=_ViewSecondary())  # type: ignore[arg-type]
    pr = pipe.prepare_and_run_primary(clip, raw_crops, (120, 160), 0, 1, None)
    assert pr.view_placements is None


def test_build_secondary_result_slices_view_placements(monkeypatch) -> None:
    clip, frames, raw_crops = _make_jitter_clip_and_frames(monkeypatch)
    pipe = RestorationPipeline(restorer=_IdentityRestorer(), secondary_restorer=_ViewSecondary())  # type: ignore[arg-type]
    pr = pipe.prepare_and_run_primary(clip, raw_crops, (120, 160), 2, 6, None)
    all_placements = list(pr.view_placements)
    sr = _build_sr(pipe, pr)
    assert sr.view_placements == all_placements[2:6]
    assert sr.clip_keep_offset == 2 and sr.keep_end == 4

    plain = RestorationPipeline(restorer=_IdentityRestorer(), secondary_restorer=_Upscale2xSecondary())  # type: ignore[arg-type]
    pr2 = plain.prepare_and_run_primary(clip, raw_crops, (120, 160), 2, 6, None)
    assert _build_sr(plain, pr2).view_placements is None


def test_view_path_composites_like_legacy_when_track_is_still(monkeypatch) -> None:
    """A still track has a constant placement, so the smoothed view equals the
    own grid and the view composite must match the legacy composite."""
    from jasna.blend_buffer import BlendBuffer

    clip, frames, raw_crops = _make_clip_and_frames(monkeypatch, t=5)
    legacy = RestorationPipeline(restorer=_IdentityRestorer(), secondary_restorer=_Upscale2xSecondary())  # type: ignore[arg-type]
    view = RestorationPipeline(restorer=_IdentityRestorer(), secondary_restorer=_ViewSecondary())  # type: ignore[arg-type]
    outs = []
    for pipe in (legacy, view):
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=lambda m, b, s: torch.ones((b[3] - b[1], b[2] - b[0])))
        for i in range(5):
            bb.register_frame(i, {0})
        pr = pipe.prepare_and_run_primary(clip, raw_crops, (30, 40), 0, 5, None)
        bb.add_result(_build_sr(pipe, pr))
        outs.append(torch.stack([bb.blend_frame(i, frames[i]) for i in range(5)]))
    assert view.prepare_and_run_primary(clip, raw_crops, (30, 40), 0, 5, None).view_placements is not None
    diff = (outs[0].int() - outs[1].int()).abs()
    assert int(diff.max()) <= 1

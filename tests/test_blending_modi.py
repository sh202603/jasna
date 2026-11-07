"""modi-only blending tests: the free-field falloff of the bbox blend mask near the bbox edge.

modi 独自。上流の tests/test_blending.py の ``_reference_fullres_blend_mask`` を使う。上流のファイルには追記しない。
"""
from __future__ import annotations

import torch

from jasna.tracking.blending import create_bbox_blend_mask
from test_blending import _reference_fullres_blend_mask


def test_bbox_blend_mask_near_edge_mask_is_free_field() -> None:
    """A mask within kernel reach of the bbox edge must get the free-field
    falloff there, not the reflect-pad pinning of a bare-slice blur (which
    printed the bbox rectangle into the composite under sharp restorers)."""
    mask = torch.zeros((64, 64), dtype=torch.bool)
    mask[20:40, 20:40] = True
    # dilation=falloff=3px at this size; mask edge (40) sits 4px inside the
    # bbox edge (44), inside the 6px reach, so a bare-slice blur would differ.
    bbox = (16, 16, 44, 44)
    frame_shape = (64, 64)

    ref = _reference_fullres_blend_mask(mask, bbox, frame_shape)
    out = create_bbox_blend_mask(mask, bbox, frame_shape)
    assert torch.allclose(out, ref, atol=1e-5)
    # The falloff must actually be under way on the bbox edge row.
    assert float(out[-1].max()) < 0.999

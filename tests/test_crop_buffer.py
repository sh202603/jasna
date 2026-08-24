from __future__ import annotations

import pytest
import torch

from jasna.crop_buffer import RESTORATION_SIZE, CropBuffer, RawCrop, prepare_crops_for_restoration


def _make_raw_crop(value: int = 0) -> RawCrop:
    return RawCrop(
        crop=torch.full((3, 4, 4), value, dtype=torch.uint8),
        enlarged_bbox=(0, 0, 4, 4),
        crop_shape=(4, 4),
    )


class TestCropBufferSplitOverlap:
    def test_split_creates_child_with_overlap_crops(self):
        parent = CropBuffer(track_id=0, start_frame=0)
        for i in range(5):
            parent.add(_make_raw_crop(value=i))

        child = parent.split_overlap(overlap_len=2, new_track_id=1, new_start_frame=3)

        assert child.track_id == 1
        assert child.start_frame == 3
        assert child.frame_count == 2
        assert child.crops[0].crop.unique().item() == 3
        assert child.crops[1].crop.unique().item() == 4

    def test_split_does_not_modify_parent(self):
        parent = CropBuffer(track_id=0, start_frame=0)
        for i in range(5):
            parent.add(_make_raw_crop(value=i))

        child = parent.split_overlap(overlap_len=2, new_track_id=1, new_start_frame=3)

        assert parent.frame_count == 5
        assert child.frame_count == 2

    def test_split_child_is_shallow_copy(self):
        parent = CropBuffer(track_id=0, start_frame=0)
        for i in range(3):
            parent.add(_make_raw_crop(value=i))

        child = parent.split_overlap(overlap_len=2, new_track_id=1, new_start_frame=1)

        assert child.crops[0] is parent.crops[1]
        assert child.crops[1] is parent.crops[2]

    def test_mutating_child_does_not_affect_parent(self):
        parent = CropBuffer(track_id=0, start_frame=0)
        for i in range(3):
            parent.add(_make_raw_crop(value=i))

        child = parent.split_overlap(overlap_len=2, new_track_id=1, new_start_frame=1)
        child.add(_make_raw_crop(value=99))

        assert parent.frame_count == 3
        assert child.frame_count == 3

    def test_split_full_overlap(self):
        parent = CropBuffer(track_id=0, start_frame=0)
        for i in range(3):
            parent.add(_make_raw_crop(value=i))

        child = parent.split_overlap(overlap_len=3, new_track_id=1, new_start_frame=0)
        assert child.frame_count == 3


class TestPrepareCropsDtype:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_output_dtype_follows_requested_precision(self, dtype: torch.dtype) -> None:
        if dtype == torch.float16 and not torch.cuda.is_available():
            pytest.skip("fp16 interpolate requires CUDA")
        device = torch.device("cuda" if dtype == torch.float16 else "cpu")

        crop = torch.randint(0, 256, (3, 40, 50), dtype=torch.uint8)
        raw = RawCrop(crop=crop, enlarged_bbox=(0, 0, 50, 40), crop_shape=(40, 50))

        crops, _, _ = prepare_crops_for_restoration([raw], device, dtype)

        assert len(crops) == 1
        out = crops[0]
        assert out.dtype == dtype
        assert out.shape == (3, RESTORATION_SIZE, RESTORATION_SIZE)
        assert out.device.type == device.type
        assert float(out.max()) <= 255.0


class TestExpandBboxBlendSafeBorder:
    def test_floor_widens_default_border(self):
        from jasna.crop_buffer import expand_bbox

        # 300px box at 1080p: default border = max(20, 6% of 300) = 20, and the
        # 340px crop needs no aspect expansion toward 256.
        x1, y1, x2, y2 = expand_bbox(500, 400, 800, 700, 1080, 1920)
        assert (500 - x1, 400 - y1, x2 - 800, y2 - 700) == (20, 20, 20, 20)

        x1, y1, x2, y2 = expand_bbox(500, 400, 800, 700, 1080, 1920, blend_safe_border_px=62)
        assert (500 - x1, 400 - y1, x2 - 800, y2 - 700) == (62, 62, 62, 62)

    def test_floor_below_default_is_inert(self):
        from jasna.crop_buffer import expand_bbox

        base = expand_bbox(500, 400, 800, 700, 1080, 1920)
        floored = expand_bbox(500, 400, 800, 700, 1080, 1920, blend_safe_border_px=10)
        assert base == floored

    def test_floor_still_clamps_to_frame(self):
        from jasna.crop_buffer import expand_bbox

        x1, y1, x2, y2 = expand_bbox(10, 10, 310, 310, 1080, 1920, blend_safe_border_px=62)
        assert x1 >= 0 and y1 >= 0 and x2 <= 1920 and y2 <= 1080

    def test_compute_enlarged_bbox_threads_floor(self):
        import numpy as np

        from jasna.crop_buffer import compute_enlarged_bbox

        bbox = np.array([500, 400, 800, 700])
        base = compute_enlarged_bbox(bbox, 1080, 1920)
        wide = compute_enlarged_bbox(bbox, 1080, 1920, blend_safe_border_px=62)
        assert base == (480, 380, 820, 720)
        assert wide == (438, 338, 862, 762)

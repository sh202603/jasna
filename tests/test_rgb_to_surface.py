"""Unit tests for the RGB→NVENC-surface conversion (NV12 / P010, BT.601/709/2020)."""
from __future__ import annotations

import pytest
import torch

from jasna.media.rgb_to_p010 import chw_rgb_to_surface, chw_rgb_to_p010_bt709_limited


def _as_uint16(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret a packed-P010 int16 tensor as the unsigned 16-bit values NVENC reads."""
    return t.to(torch.int32) & 0xFFFF


class TestShapeAndDtype:
    @pytest.mark.parametrize("cs", ["bt601", "bt709", "bt2020"])
    def test_nv12_8bit(self, cs):
        img = torch.randint(0, 256, (3, 64, 48), dtype=torch.uint8)
        s = chw_rgb_to_surface(img, cs, 8)
        assert s.shape == (96, 48)  # Y(64) + interleaved UV(32)
        assert s.dtype == torch.uint8
        assert s.is_contiguous()

    @pytest.mark.parametrize("cs", ["bt601", "bt709", "bt2020"])
    def test_p010_10bit(self, cs):
        img = torch.randint(0, 256, (3, 64, 48), dtype=torch.uint8)
        s = chw_rgb_to_surface(img, cs, 10)
        assert s.shape == (96, 48)
        assert s.dtype == torch.int16
        assert s.is_contiguous()


class TestLimitedRange:
    @pytest.mark.parametrize("cs", ["bt601", "bt709", "bt2020"])
    def test_nv12_luma_range(self, cs):
        img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        Y = chw_rgb_to_surface(img, cs, 8)[:32]
        assert Y.min() >= 16 and Y.max() <= 235

    @pytest.mark.parametrize("cs", ["bt601", "bt709", "bt2020"])
    def test_p010_luma_range(self, cs):
        img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        Y = _as_uint16(chw_rgb_to_surface(img, cs, 10)[:32]) >> 6
        assert Y.min() >= 64 and Y.max() <= 940


class TestKnownValues:
    def test_white_black_8bit(self):
        white = torch.full((3, 8, 8), 255, dtype=torch.uint8)
        black = torch.zeros((3, 8, 8), dtype=torch.uint8)
        assert int(chw_rgb_to_surface(white, "bt709", 8)[:8].float().mean()) == 235
        assert int(chw_rgb_to_surface(black, "bt709", 8)[:8].float().mean()) == 16

    def test_gray_chroma_is_neutral_8bit(self):
        # An achromatic (R=G=B) frame must have neutral chroma (128) in NV12.
        gray = torch.full((3, 8, 8), 120, dtype=torch.uint8)
        uv = chw_rgb_to_surface(gray, "bt709", 8)[8:]
        assert uv.float().mean().round().item() == 128


class TestErrors:
    def test_bad_colorspace(self):
        img = torch.zeros((3, 8, 8), dtype=torch.uint8)
        with pytest.raises(ValueError, match="colorspace"):
            chw_rgb_to_surface(img, "bt2100", 8)

    def test_bad_bit_depth(self):
        img = torch.zeros((3, 8, 8), dtype=torch.uint8)
        with pytest.raises(ValueError, match="bit depth"):
            chw_rgb_to_surface(img, "bt709", 12)


class TestBackwardCompat:
    def test_wrapper_matches_bt709_p010(self):
        img = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
        assert torch.equal(chw_rgb_to_p010_bt709_limited(img), chw_rgb_to_surface(img, "bt709", 10))

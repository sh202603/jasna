"""The whole-frame SwiftVR worker's tile layout and blend weights (CPU; no SwiftVR needed: the
worker's top level is stdlib-only)."""
from __future__ import annotations

import pytest
import torch

from jasna.restorer import swiftvr_frame_worker as worker


def _shared(positions: list[int], length: int) -> list[int]:
    return [positions[index] + length - positions[index + 1] for index in range(len(positions) - 1)]


def test_1080p_at_2x_is_eight_equal_tiles():
    tile_width, tile_height, xs, ys = worker.plan_tiles(1920, 1080, 2)

    assert (len(xs), len(ys)) == (4, 2)
    # Multiples of 16: the 2x tile is a multiple of 32, so SwiftVR pads nothing.
    assert (tile_width, tile_height) == (528, 576)
    assert tile_width * tile_height * 4 <= 1.25e6
    assert xs[0] == 0 and xs[-1] + tile_width == 1920
    assert ys[0] == 0 and ys[-1] + tile_height == 1080
    assert min(_shared(xs, tile_width) + _shared(ys, tile_height)) >= 48


@pytest.mark.parametrize(
    "width, height, scale, megapixels",
    [(1920, 1080, 2, 1.25), (1920, 1080, 2, 0.6), (1920, 1080, 4, 1.25), (854, 480, 4, 1.25),
     (1280, 720, 2, 1.25), (3840, 2160, 2, 1.64), (1080, 1920, 2, 1.25)],
)
def test_layout_covers_the_frame_within_the_tile_limit(width, height, scale, megapixels):
    tile_width, tile_height, xs, ys = worker.plan_tiles(width, height, scale, megapixels=megapixels)

    assert tile_width * tile_height * scale * scale <= megapixels * 1e6
    for positions, length, size in ((xs, tile_width, width), (ys, tile_height, height)):
        assert positions[0] == 0 and positions[-1] + length == size
        assert positions == sorted(positions)
        if len(positions) > 1:
            assert length % (32 // scale) == 0
            assert min(_shared(positions, length)) >= 96 // scale
            # Only neighbours overlap, so a pixel belongs to at most two tiles per axis.
            assert all(positions[index] + length <= positions[index + 2] for index in range(len(positions) - 2))


def test_a_frame_that_fits_is_one_tile():
    assert worker.plan_tiles(640, 360, 2) == (640, 360, [0], [0])


def test_larger_tiles_mean_fewer_tiles():
    small = worker.plan_tiles(1920, 1080, 2, megapixels=1.05)
    large = worker.plan_tiles(1920, 1080, 2, megapixels=2.4)

    assert len(large[2]) * len(large[3]) < len(small[2]) * len(small[3])


def test_impossible_tile_limit_is_refused():
    with pytest.raises(ValueError, match="no tile layout"):
        worker.plan_tiles(1920, 1080, 4, megapixels=0.01)


def test_blend_of_constant_tiles_is_constant_and_weights_cover_the_frame():
    layout = worker._Layout(torch, 320, 200, 2, megapixels=0.12, overlap=32, device="cpu")
    assert len(layout.tiles) > 1
    assert float(layout.total.min()) > 0.0

    shape = (layout.tile_height * 2, layout.tile_width * 2, 3)
    frame = layout.blend(torch, [torch.full(shape, 77, dtype=torch.uint8) for _ in layout.tiles])

    assert frame.shape == (400, 640, 3) and frame.dtype == torch.uint8
    assert int(frame.min()) == 77 and int(frame.max()) == 77


def test_blend_ramps_between_neighbouring_tiles():
    layout = worker._Layout(torch, 320, 100, 2, megapixels=0.09, overlap=32, device="cpu")
    assert len(layout.ys) == 1 and len(layout.xs) == 2

    shape = (layout.tile_height * 2, layout.tile_width * 2, 3)
    frame = layout.blend(torch, [torch.full(shape, value, dtype=torch.uint8) for value in (0, 200)])
    row = frame[0, :, 0].tolist()

    start = layout.xs[1] * 2
    end = (layout.xs[0] + layout.tile_width) * 2
    assert set(row[:start]) == {0} and set(row[end:]) == {200}
    shared = row[start:end]
    assert shared == sorted(shared) and 0 < shared[0] < shared[-1] < 200


def test_write_all_finishes_partial_writes():
    class Dribble:
        def __init__(self):
            self.data = b""

        def write(self, view):
            self.data += bytes(view[:3])
            return min(3, len(view))

    stream = Dribble()
    worker._write_all(stream, b"0123456789")
    assert stream.data == b"0123456789"

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer, SecondaryRestorer
from jasna.restorer.swiftvr_distill_model import (
    SWIFTVR_DISTILL_KNOWN_VERSIONS,
    TinyROIEnhancer,
    load_swiftvr_distill_model,
    resolve_swiftvr_distill_model_path,
)
from jasna.restorer.swiftvr_distill_secondary_restorer import (
    SWIFTVR_DISTILL_STABILIZE_NEIGHBOUR_WEIGHT,
    SwiftvrDistillSecondaryRestorer,
    clamped_window_indices,
    stabilize_added_detail,
)

_ARCH = {"channels": 4, "blocks": 2, "window": 5, "scale": 2}
_CPU = torch.device("cpu")


def _write_checkpoint(path: Path, **overrides) -> Path:
    torch.manual_seed(0)
    architecture = dict(_ARCH)
    checkpoint = {
        "version": SWIFTVR_DISTILL_KNOWN_VERSIONS[0],
        "architecture": architecture,
        "model": TinyROIEnhancer(**architecture).state_dict(),
        "step": 1,
    }
    checkpoint.update(overrides)
    torch.save(checkpoint, path)
    return path


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    return _write_checkpoint(tmp_path / "distill.pt")


@pytest.fixture
def restorer(checkpoint_path: Path) -> SwiftvrDistillSecondaryRestorer:
    return SwiftvrDistillSecondaryRestorer(model_path=checkpoint_path, device=_CPU)


class _RecordingModel:
    """Stands in for the model: records its inputs, returns the centre frame 2x."""

    window = 5
    scale = 2

    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def __call__(self, windows: torch.Tensor) -> torch.Tensor:
        self.inputs.append(windows.clone())
        return windows[:, 6:9].repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)


class TestTinyROIEnhancer:
    def test_output_shape(self):
        model = TinyROIEnhancer(**_ARCH).eval()
        with torch.no_grad():
            out = model(torch.rand(3, 15, 32, 48))
        assert out.shape == (3, 3, 64, 96)

    def test_state_dict_layout_matches_the_checkpoint_format(self):
        keys = set(TinyROIEnhancer(channels=24, blocks=12, window=5, scale=2).state_dict())
        assert {"stem.weight", "stem.bias", "head.weight", "head.bias"} <= keys
        assert {"body.0.body.0.weight", "body.11.body.2.bias"} <= keys
        assert len(keys) == 52

    def test_zero_weights_reduce_to_the_bilinear_base(self):
        model = TinyROIEnhancer(**_ARCH).eval()
        for p in model.parameters():
            torch.nn.init.zeros_(p)
        windows = torch.rand(1, 15, 16, 16)
        with torch.no_grad():
            out = model(windows)
        expected = torch.nn.functional.interpolate(
            windows[:, 6:9], scale_factor=2, mode="bilinear", align_corners=False
        )
        assert torch.equal(out, expected)


class TestLoadSwiftvrDistillModel:
    def test_loads_a_valid_checkpoint(self, checkpoint_path: Path):
        model = load_swiftvr_distill_model(checkpoint_path, _CPU)
        assert model.window == 5
        assert model.scale == 2
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())

    def test_unknown_version_warns_but_loads(self, tmp_path: Path, caplog):
        path = _write_checkpoint(tmp_path / "v.pt", version="roi-distill-other")
        with caplog.at_level("WARNING"):
            load_swiftvr_distill_model(path, _CPU)
        assert "roi-distill-other" in caplog.text

    def test_rejects_a_non_dict(self, tmp_path: Path):
        path = tmp_path / "list.pt"
        torch.save([1, 2, 3], path)
        with pytest.raises(ValueError, match="not a dict"):
            load_swiftvr_distill_model(path, _CPU)

    def test_rejects_a_missing_architecture(self, tmp_path: Path):
        path = _write_checkpoint(tmp_path / "a.pt", architecture=None)
        with pytest.raises(ValueError, match="no 'architecture'"):
            load_swiftvr_distill_model(path, _CPU)

    @pytest.mark.parametrize("key,value", [("channels", 0), ("blocks", "12"), ("scale", 8), ("window", True)])
    def test_rejects_an_unsupported_architecture_value(self, tmp_path: Path, key, value):
        path = _write_checkpoint(tmp_path / "a.pt", architecture={**_ARCH, key: value})
        with pytest.raises(ValueError, match=f"architecture.{key}"):
            load_swiftvr_distill_model(path, _CPU)

    def test_rejects_an_even_window(self, tmp_path: Path):
        path = _write_checkpoint(tmp_path / "a.pt", architecture={**_ARCH, "window": 4})
        with pytest.raises(ValueError, match="even architecture.window"):
            load_swiftvr_distill_model(path, _CPU)

    def test_rejects_a_missing_state_dict(self, tmp_path: Path):
        path = _write_checkpoint(tmp_path / "m.pt", model=None)
        with pytest.raises(ValueError, match="no 'model'"):
            load_swiftvr_distill_model(path, _CPU)

    def test_rejects_a_state_dict_of_another_architecture(self, tmp_path: Path):
        other = TinyROIEnhancer(channels=8, blocks=2, window=5, scale=2).state_dict()
        path = _write_checkpoint(tmp_path / "m.pt", model=other)
        with pytest.raises(ValueError, match="does not match its architecture"):
            load_swiftvr_distill_model(path, _CPU)

    def test_rejects_non_finite_weights(self, tmp_path: Path):
        state_dict = TinyROIEnhancer(**_ARCH).state_dict()
        state_dict["head.bias"][0] = float("nan")
        path = _write_checkpoint(tmp_path / "m.pt", model=state_dict)
        with pytest.raises(ValueError, match="non-finite"):
            load_swiftvr_distill_model(path, _CPU)


class TestResolveModelPath:
    def test_empty_is_rejected(self):
        with pytest.raises(ValueError, match="--swiftvr-distill-model is required"):
            resolve_swiftvr_distill_model_path("  ")

    def test_missing_file_is_rejected(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="--swiftvr-distill-model not found"):
            resolve_swiftvr_distill_model_path(str(tmp_path / "nope.pt"))

    def test_existing_file_is_returned(self, checkpoint_path: Path):
        assert resolve_swiftvr_distill_model_path(str(checkpoint_path)) == checkpoint_path


class TestClampedWindowIndices:
    def test_edges_repeat_the_clip_ends(self):
        idx = clamped_window_indices(0, 5, 5, 2, _CPU)
        assert idx[0].tolist() == [0, 0, 0, 1, 2]
        assert idx[2].tolist() == [0, 1, 2, 3, 4]
        assert idx[4].tolist() == [2, 3, 4, 4, 4]

    def test_single_frame_clip(self):
        assert clamped_window_indices(0, 1, 1, 2, _CPU).tolist() == [[0, 0, 0, 0, 0]]

    def test_two_frame_clip(self):
        assert clamped_window_indices(0, 2, 2, 2, _CPU).tolist() == [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]

    def test_partial_range(self):
        assert clamped_window_indices(3, 5, 8, 2, _CPU).tolist() == [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]


class TestStabilizeAddedDetail:
    @staticmethod
    def _upscale(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def test_identical_frames_are_unchanged(self):
        base = torch.rand(1, 3, 8, 8).expand(5, 3, 8, 8).contiguous()
        out = (self._upscale(base) + 0.1 * torch.rand(1, 3, 16, 16)).contiguous()
        got = stabilize_added_detail(base, out, 0, 5, 2)
        assert torch.allclose(got, out, atol=1e-6)

    def test_neighbours_with_the_same_input_are_averaged(self):
        base = torch.full((3, 3, 8, 8), 0.5)
        detail = torch.tensor([0.0, 0.3, 0.6]).view(3, 1, 1, 1).expand(3, 3, 16, 16)
        got = stabilize_added_detail(base, self._upscale(base) + detail, 1, 2, 1)
        w = SWIFTVR_DISTILL_STABILIZE_NEIGHBOUR_WEIGHT
        expected = 0.5 + (w * 0.0 + 0.3 + w * 0.6) / (1 + 2 * w)
        assert got.shape == (1, 3, 16, 16)
        assert torch.allclose(got, torch.full_like(got, expected), atol=1e-6)

    def test_a_neighbour_with_a_different_input_is_left_out(self):
        base = torch.stack([torch.full((3, 8, 8), 0.2), torch.full((3, 8, 8), 0.8)])
        detail = torch.tensor([0.1, -0.1]).view(2, 1, 1, 1).expand(2, 3, 16, 16)
        out = self._upscale(base) + detail
        got = stabilize_added_detail(base, out, 0, 2, 1)
        assert torch.allclose(got, out, atol=1e-5)

    def test_the_window_is_truncated_at_the_clip_ends(self):
        base = torch.full((2, 3, 8, 8), 0.5)
        detail = torch.tensor([0.0, 0.2]).view(2, 1, 1, 1).expand(2, 3, 16, 16)
        got = stabilize_added_detail(base, self._upscale(base) + detail, 0, 1, 4)
        w = SWIFTVR_DISTILL_STABILIZE_NEIGHBOUR_WEIGHT
        assert torch.allclose(got, torch.full_like(got, 0.5 + w * 0.2 / (1 + w)), atol=1e-6)


class TestSwiftvrDistillSecondaryRestorer:
    def test_protocol_attrs(self, restorer: SwiftvrDistillSecondaryRestorer):
        assert restorer.name == "swiftvr-distill"
        assert restorer.num_workers == 1
        assert restorer.prefers_cpu_input is False
        assert restorer.view_smoothing_window == 0

    def test_is_a_sync_restorer_only(self, restorer: SwiftvrDistillSecondaryRestorer):
        assert isinstance(restorer, SecondaryRestorer)
        assert not isinstance(restorer, AsyncSecondaryRestorer)
        assert not hasattr(restorer, "push_clip")

    def test_view_window_is_exposed(self, checkpoint_path: Path):
        r = SwiftvrDistillSecondaryRestorer(model_path=checkpoint_path, device=_CPU, view_window=15)
        assert r.view_smoothing_window == 15

    @pytest.mark.parametrize("T", [1, 2, 5, 37])
    def test_restore_returns_one_uint8_frame_per_input(self, restorer: SwiftvrDistillSecondaryRestorer, T: int):
        frames = torch.rand(T, 3, 32, 32)
        result = restorer.restore(frames, keep_start=0, keep_end=T)
        assert len(result) == T
        for frame in result:
            assert frame.shape == (3, 64, 64)
            assert frame.dtype == torch.uint8

    def test_restore_empty_clip(self, restorer: SwiftvrDistillSecondaryRestorer):
        assert restorer.restore(torch.rand(0, 3, 32, 32), keep_start=0, keep_end=0) == []

    def test_restore_empty_keep(self, restorer: SwiftvrDistillSecondaryRestorer):
        frames = torch.rand(4, 3, 32, 32)
        assert restorer.restore(frames, keep_start=2, keep_end=2) == []
        assert restorer.restore(frames, keep_start=5, keep_end=6) == []

    def test_keep_range_is_clamped_to_the_clip(self, restorer: SwiftvrDistillSecondaryRestorer):
        frames = torch.rand(4, 3, 32, 32)
        assert len(restorer.restore(frames, keep_start=-2, keep_end=9)) == 4

    def test_partial_keep_matches_the_full_clip(self, restorer: SwiftvrDistillSecondaryRestorer):
        frames = torch.rand(8, 3, 32, 32)
        full = restorer.restore(frames, keep_start=0, keep_end=8)
        part = restorer.restore(frames, keep_start=3, keep_end=6)
        assert len(part) == 3
        for got, want in zip(part, full[3:6]):
            assert torch.equal(got, want)

    def test_batch_size_does_not_change_the_output(self, checkpoint_path: Path):
        frames = torch.rand(7, 3, 32, 32)
        one = SwiftvrDistillSecondaryRestorer(model_path=checkpoint_path, device=_CPU, batch_size=1)
        many = SwiftvrDistillSecondaryRestorer(model_path=checkpoint_path, device=_CPU, batch_size=3)
        a = torch.stack(one.restore(frames, keep_start=0, keep_end=7)).to(torch.int16)
        b = torch.stack(many.restore(frames, keep_start=0, keep_end=7)).to(torch.int16)
        # Rounding may flip one level where the float results differ in the last bit.
        assert int((a - b).abs().max()) <= 1

    def test_windows_are_stacked_frame_by_frame(self, restorer: SwiftvrDistillSecondaryRestorer):
        model = _RecordingModel()
        restorer.model = model
        frames = torch.stack([torch.full((3, 8, 8), v) for v in (0.2, 0.4, 0.8)])
        frames[:, 1] += 0.05  # tell the channels apart
        result = restorer.restore(frames, keep_start=0, keep_end=3)

        (windows,) = model.inputs
        assert windows.shape == (3, 15, 8, 8)
        # Frame 0's window is [0, 0, 0, 1, 2]: RGB of each frame in turn.
        assert torch.equal(windows[0, 0:3], frames[0])
        assert torch.equal(windows[0, 6:9], frames[0])
        assert torch.equal(windows[0, 9:12], frames[1])
        assert torch.equal(windows[0, 12:15], frames[2])
        # The output is the centre frame, so frame i comes back as frame i.
        assert [int(f[0, 0, 0]) for f in result] == [51, 102, 204]

    def test_input_is_clamped_and_cast(self, restorer: SwiftvrDistillSecondaryRestorer):
        model = _RecordingModel()
        restorer.model = model
        frames = torch.tensor([-0.5, 1.5], dtype=torch.float16).view(2, 1, 1, 1).expand(2, 3, 8, 8)
        result = restorer.restore(frames, keep_start=0, keep_end=2)

        (windows,) = model.inputs
        assert windows.dtype == torch.float32
        assert float(windows.min()) == 0.0
        assert float(windows.max()) == 1.0
        assert int(result[0].max()) == 0
        assert int(result[1].min()) == 255

    def test_returned_frames_accept_in_place_writes(self, restorer: SwiftvrDistillSecondaryRestorer):
        result = restorer.restore(torch.rand(1, 3, 32, 32), keep_start=0, keep_end=1)
        result[0].fill_(0)

    def test_strength_is_clamped(self, checkpoint_path: Path):
        make = lambda s: SwiftvrDistillSecondaryRestorer(model_path=checkpoint_path, device=_CPU, strength=s)
        assert make(-1.0).strength == 0.0
        assert make(0.75).strength == 0.75
        assert make(9.0).strength == 2.0

    def test_strength_zero_is_the_plain_upscale(self, checkpoint_path: Path):
        frames = torch.rand(5, 3, 32, 32)
        r = SwiftvrDistillSecondaryRestorer(model_path=checkpoint_path, device=_CPU, strength=0.0)
        got = torch.stack(r.restore(frames, keep_start=0, keep_end=5))
        expected = torch.nn.functional.interpolate(frames, scale_factor=2, mode="bilinear", align_corners=False)
        assert torch.equal(got, expected.mul(255.0).round().to(torch.uint8))

    def test_strength_scales_the_added_detail(self, checkpoint_path: Path):
        frames = torch.rand(5, 3, 32, 32)
        make = lambda s: SwiftvrDistillSecondaryRestorer(model_path=checkpoint_path, device=_CPU, strength=s)
        run = lambda s: torch.stack(make(s).restore(frames, keep_start=1, keep_end=4)).float()
        plain, full, half = run(0.0), run(1.0), run(0.5)
        assert float((full - plain).abs().mean()) > 1.0  # the random weights do add something
        # Linear wherever the full output was not clipped; rounding to uint8
        # three times leaves at most a level and a half.
        unclipped = (full > 0) & (full < 255)
        assert float((half - (plain + full) / 2)[unclipped].abs().max()) <= 1.5

    def test_stabilize_radius_is_clamped(self, checkpoint_path: Path):
        make = lambda r: SwiftvrDistillSecondaryRestorer(model_path=checkpoint_path, device=_CPU, stabilize_radius=r)
        assert make(-3).stabilize_radius == 0
        assert make(2).stabilize_radius == 2
        assert make(99).stabilize_radius == 8

    def test_stabilized_static_clip_matches_the_plain_output(self, checkpoint_path: Path):
        frames = torch.rand(1, 3, 32, 32).expand(6, 3, 32, 32).contiguous()
        plain = SwiftvrDistillSecondaryRestorer(model_path=checkpoint_path, device=_CPU)
        stable = SwiftvrDistillSecondaryRestorer(model_path=checkpoint_path, device=_CPU, stabilize_radius=2)
        a = torch.stack(plain.restore(frames, keep_start=0, keep_end=6)).to(torch.int16)
        b = torch.stack(stable.restore(frames, keep_start=0, keep_end=6)).to(torch.int16)
        assert int((a - b).abs().max()) <= 1

    @pytest.mark.parametrize("T", [1, 2, 9])
    def test_stabilized_restore_shapes(self, checkpoint_path: Path, T: int):
        r = SwiftvrDistillSecondaryRestorer(model_path=checkpoint_path, device=_CPU, stabilize_radius=3)
        result = r.restore(torch.rand(T, 3, 32, 32), keep_start=0, keep_end=T)
        assert len(result) == T
        assert all(f.shape == (3, 64, 64) and f.dtype == torch.uint8 for f in result)

    def test_stabilized_partial_keep_matches_the_full_clip(self, checkpoint_path: Path):
        r = SwiftvrDistillSecondaryRestorer(model_path=checkpoint_path, device=_CPU, stabilize_radius=2)
        still = torch.rand(1, 3, 32, 32)
        frames = (still + 0.01 * torch.rand(10, 3, 32, 32)).clamp(0, 1)  # close enough to blend
        full = r.restore(frames, keep_start=0, keep_end=10)
        part = r.restore(frames, keep_start=4, keep_end=7)
        assert len(part) == 3
        for got, want in zip(part, full[4:7]):
            assert int((got.to(torch.int16) - want.to(torch.int16)).abs().max()) <= 1

    def test_stabilization_calms_a_noisy_still(self, checkpoint_path: Path):
        still = torch.rand(1, 3, 32, 32)
        frames = (still + 0.01 * torch.rand(6, 3, 32, 32)).clamp(0, 1)
        plain = SwiftvrDistillSecondaryRestorer(model_path=checkpoint_path, device=_CPU)
        stable = SwiftvrDistillSecondaryRestorer(model_path=checkpoint_path, device=_CPU, stabilize_radius=2)
        a = torch.stack(plain.restore(frames, keep_start=0, keep_end=6)).float()
        b = torch.stack(stable.restore(frames, keep_start=0, keep_end=6)).float()
        change = lambda x: float((x[1:] - x[:-1]).abs().mean())
        assert change(b) < change(a)

    def test_close_releases_the_model(self, restorer: SwiftvrDistillSecondaryRestorer):
        restorer.close()
        assert restorer.model is None


_REAL_MODEL = os.environ.get("JASNA_SWIFTVR_DISTILL_MODEL", "")


@pytest.mark.skipif(not _REAL_MODEL or not Path(_REAL_MODEL).is_file(),
                    reason="JASNA_SWIFTVR_DISTILL_MODEL does not point to the real checkpoint")
def test_real_weights_leave_flat_colours_flat():
    """Guards the forward-pass constants (the published definition). On a flat colour the
    real weights must add neither a PixelShuffle grid nor a colour shift; a
    wrong residual scale, activation or stem ReLU fails one of the two."""
    model = load_swiftvr_distill_model(_REAL_MODEL, _CPU)
    colours = [(g, g, g) for g in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)] + [
        (0.8, 0.5, 0.4), (0.6, 0.35, 0.3), (0.3, 0.2, 0.2), (0.9, 0.7, 0.6), (0.2, 0.4, 0.6),
    ]
    windows = torch.stack([
        torch.tensor(c, dtype=torch.float32).repeat(5)[:, None, None].expand(15, 64, 64) for c in colours
    ])
    with torch.no_grad():
        out = model(windows)
    flat = torch.tensor(colours, dtype=torch.float32)[:, :, None, None]
    residual = (out - flat)[:, :, 60:68, 60:68]  # centre, away from the borders
    phases = torch.stack([residual[:, :, i::2, j::2].mean((2, 3)) for i in (0, 1) for j in (0, 1)], dim=-1)
    grid = float((phases.max(-1).values - phases.min(-1).values).mean()) * 255
    shift = float(phases.mean(-1).abs().mean()) * 255
    assert grid < 0.12, f"PixelShuffle grid on flat colours: {grid:.3f} levels"
    assert shift < 2.3, f"colour shift on flat colours: {shift:.2f} levels"

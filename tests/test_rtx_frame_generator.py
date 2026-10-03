"""Tests for the RTX Video Frame Generation backend (nvvfx ``VideoFrameGeneration``).

The unit tests run GPU-free against a fake ``nvvfx`` module installed in
``sys.modules`` for the duration of each test: the backend imports nvvfx lazily
(inside ``__init__``), so the fake is picked up without touching a real SDK. The
last test exercises the real effect and is gated on CUDA + nvvfx >= 0.2.0.0 +
an Ada-or-newer GPU.
"""
from __future__ import annotations

import sys
from enum import IntEnum
from importlib.util import find_spec
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from jasna.framegen.rtx_frame_generator import (
    MIN_COMPUTE_CAPABILITY,
    RtxFrameGenerator,
    _to_chw_uint8,
    _to_sdk_input,
)


class _Mode(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class _FakeVideoFrameGeneration:
    """Mimics the nvvfx 0.2.0.0 effect: dims must be set before load(), run()
    needs load(), and the output buffer is shared between runs (so the backend
    must clone it). Interpolation is a linear blend so results are checkable."""

    Mode = _Mode
    instances: list["_FakeVideoFrameGeneration"] = []

    def __init__(self, *, mode=_Mode.MEDIUM, automatic_shot_change_detection_enabled=True, device=0):
        self.mode = mode
        self.auto_shot_change = automatic_shot_change_detection_enabled
        self.device = device
        self.input_width = None
        self.input_height = None
        self.load_calls = 0
        self.run_calls: list[tuple[float, int]] = []
        self.closed = False
        self._loaded_for = None
        self._out = None
        _FakeVideoFrameGeneration.instances.append(self)

    @property
    def is_loaded(self):
        return self._loaded_for is not None

    def load(self):
        if self.input_width is None or self.input_height is None:
            raise ValueError("input_width and input_height must be set before load()")
        self.load_calls += 1
        self._loaded_for = (self.input_height, self.input_width)
        self._out = None

    def run_at_timestep(self, prev, cur, timestep, *, shot_change=False, non_blocking=False, stream_ptr=0):
        assert self.is_loaded, "run before load()"
        assert prev.dtype == torch.float32 and cur.dtype == torch.float32
        assert prev.is_contiguous() and cur.is_contiguous()
        assert tuple(prev.shape[1:]) == self._loaded_for, "frame size != loaded size"
        assert 0.0 < timestep < 1.0
        self.run_calls.append((float(timestep), int(stream_ptr)))
        blend = prev * (1.0 - timestep) + cur * timestep
        if self._out is None:
            self._out = blend
        else:
            self._out.copy_(blend)  # same buffer every run, like the SDK
        return SimpleNamespace(image=self._out)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_nvvfx(monkeypatch):
    _FakeVideoFrameGeneration.instances.clear()
    mod = ModuleType("nvvfx")
    mod.__version__ = "0.2.0.0"
    mod.VideoFrameGeneration = _FakeVideoFrameGeneration
    monkeypatch.setitem(sys.modules, "nvvfx", mod)
    return mod


@pytest.fixture
def ada_gpu():
    """GPU-free stand-in for the CUDA bits the backend touches at runtime.

    The GPU gate is skipped (it has its own tests below), the "current stream"
    is a stub whose pointer we can assert on, and ``torch.from_dlpack`` becomes
    the identity because the fake effect already hands back torch tensors (the
    real call would otherwise ask torch.cuda for the current stream).
    """
    stream = SimpleNamespace(cuda_stream=4242)
    with (
        patch.object(RtxFrameGenerator, "_check_gpu", lambda self: None),
        patch("torch.cuda.current_stream", return_value=stream),
        patch("torch.from_dlpack", side_effect=lambda x: x),
    ):
        yield stream


@pytest.fixture
def ada_capability():
    """Report an Ada GPU for the compute-capability gate (no CUDA needed)."""
    with patch("torch.cuda.get_device_capability", return_value=(8, 9)):
        yield


def _frames(h=24, w=40, device="cpu"):
    g = torch.Generator().manual_seed(0)
    a = torch.randint(0, 256, (3, h, w), dtype=torch.uint8, generator=g).to(device)
    b = torch.randint(0, 256, (3, h, w), dtype=torch.uint8, generator=g).to(device)
    return a, b


def _make(**kw):
    # With the GPU gate stubbed out a CPU device keeps every tensor op on the
    # host, so the fake-SDK tests run on a box without CUDA.
    return RtxFrameGenerator(device=torch.device("cpu"), **kw)


def _make_cuda(**kw):
    return RtxFrameGenerator(device=torch.device("cuda:0"), **kw)


class TestHelpers:
    def test_to_sdk_input_is_float01_contiguous_and_does_not_alias(self):
        a, _ = _frames()
        x = _to_sdk_input(a.permute(0, 2, 1).contiguous().permute(0, 2, 1), torch.device("cpu"))
        assert x.dtype == torch.float32 and x.is_contiguous()
        assert float(x.max()) <= 1.0 and float(x.min()) >= 0.0
        f = torch.full((3, 2, 2), 0.5)
        y = _to_sdk_input(f, torch.device("cpu"))
        assert y.data_ptr() != f.data_ptr()
        assert float(f[0, 0, 0]) == 0.5  # caller's buffer untouched

    def test_to_chw_uint8_rounds_and_clamps(self):
        x = torch.tensor([[[-0.1, 0.0, 0.5, 1.0, 1.7]]])
        assert _to_chw_uint8(x.clone()).tolist() == [[[0, 0, 128, 255, 255]]]


class TestInit:
    def test_requires_nvvfx_with_frame_generation(self, monkeypatch, ada_capability):
        old = ModuleType("nvvfx")
        old.__version__ = "0.1.0.1"
        monkeypatch.setitem(sys.modules, "nvvfx", old)
        with pytest.raises(RuntimeError, match="0.2.0.0"):
            _make_cuda()

    def test_requires_nvvfx_installed(self, monkeypatch, ada_capability):
        monkeypatch.setitem(sys.modules, "nvvfx", None)  # makes `import nvvfx` fail
        with pytest.raises(RuntimeError, match="nvidia-vfx"):
            _make_cuda()

    def test_rejects_non_cuda_device(self, fake_nvvfx):
        with pytest.raises(RuntimeError, match="CUDA"):
            RtxFrameGenerator(device=torch.device("cpu"))
        assert _FakeVideoFrameGeneration.instances == []

    def test_rejects_pre_ada_gpu_before_touching_sdk(self, fake_nvvfx):
        with patch("torch.cuda.get_device_capability", return_value=(8, 6)):
            with pytest.raises(RuntimeError, match="RTX 40"):
                _make_cuda()
        assert _FakeVideoFrameGeneration.instances == []

    @pytest.mark.parametrize("cap", [(8, 9), (9, 0), (10, 0), (12, 0)])
    def test_accepts_ada_and_newer(self, fake_nvvfx, cap):
        assert cap >= MIN_COMPUTE_CAPABILITY
        with patch("torch.cuda.get_device_capability", return_value=cap):
            gen = _make_cuda()
        assert gen.name == "rtx-frame-gen"
        assert _FakeVideoFrameGeneration.instances[-1].device == 0

    def test_mode_and_flags_forwarded(self, fake_nvvfx, ada_gpu):
        gen = _make(mode="HIGH", shot_change_detection=False)
        fg = _FakeVideoFrameGeneration.instances[-1]
        assert fg.mode == _Mode.HIGH
        assert fg.auto_shot_change is False
        assert fg.load_calls == 0  # lazy: no size known yet
        gen.close()

    def test_invalid_mode(self, fake_nvvfx, ada_gpu):
        with pytest.raises(ValueError, match="mode"):
            _make(mode="ultra")

    def test_ignores_rife_only_kwargs(self, fake_nvvfx, ada_gpu):
        gen = _make(model_path="/nope/rife.pth", fp16=False)
        gen.close()


class TestInterpolate:
    def test_outputs_match_contract_and_are_independent_copies(self, fake_nvvfx, ada_gpu):
        gen = _make()
        a, b = _frames()
        out = gen.interpolate(a, b, [0.25, 0.5, 0.75])
        fg = _FakeVideoFrameGeneration.instances[-1]

        assert len(out) == 3
        for frame in out:
            assert frame.shape == a.shape and frame.dtype == torch.uint8
        # Each result must survive the next run overwriting the SDK buffer.
        ptrs = {f.data_ptr() for f in out}
        assert len(ptrs) == 3
        # The fake blends in [0, 1] and the backend rescales, so values that
        # land on .5 may round either way: allow one code of slack.
        expected_mid = (a.float() * 0.5 + b.float() * 0.5).round()
        assert int((out[1].float() - expected_mid).abs().max()) <= 1
        assert [t for t, _ in fg.run_calls] == [0.25, 0.5, 0.75]
        assert all(sp == ada_gpu.cuda_stream for _, sp in fg.run_calls)

    def test_lazy_load_once_per_size_and_reload_on_change(self, fake_nvvfx, ada_gpu):
        gen = _make()
        fg = _FakeVideoFrameGeneration.instances[-1]
        a, b = _frames(24, 40)
        gen.interpolate(a, b, [0.5])
        gen.interpolate(a, b, [0.5])
        assert fg.load_calls == 1
        assert (fg.input_height, fg.input_width) == (24, 40)

        c, d = _frames(16, 32)
        gen.interpolate(c, d, [0.5])
        assert fg.load_calls == 2
        assert (fg.input_height, fg.input_width) == (16, 32)

        gen.interpolate(a, b, [0.5])  # back to the first size -> reload again
        assert fg.load_calls == 3

    def test_empty_positions(self, fake_nvvfx, ada_gpu):
        gen = _make()
        a, b = _frames()
        assert gen.interpolate(a, b, []) == []
        assert _FakeVideoFrameGeneration.instances[-1].load_calls == 0

    def test_shape_mismatch_rejected(self, fake_nvvfx, ada_gpu):
        gen = _make()
        a, _ = _frames(24, 40)
        b, _ = _frames(24, 48)
        with pytest.raises(ValueError, match="equal shape"):
            gen.interpolate(a, b, [0.5])

    def test_close_releases_effect_and_is_idempotent(self, fake_nvvfx, ada_gpu):
        gen = _make()
        fg = _FakeVideoFrameGeneration.instances[-1]
        gen.close()
        gen.close()
        assert fg.closed
        assert gen._fg is None


def test_build_frame_generator_routes_rtx_mode(fake_nvvfx, ada_gpu):
    from jasna.framegen import build_frame_generator

    gen = build_frame_generator("rtx", device=torch.device("cpu"), model_path=None, fp16=True, rtx_mode="low")
    assert isinstance(gen, RtxFrameGenerator)
    assert _FakeVideoFrameGeneration.instances[-1].mode == _Mode.LOW
    gen.close()


# --------------------------------------------------------------------------
# Real SDK (skipped without CUDA / nvvfx 0.2+ / Ada-or-newer GPU)
# --------------------------------------------------------------------------

def _real_sdk_available() -> bool:
    if not torch.cuda.is_available() or find_spec("nvvfx") is None:
        return False
    if torch.cuda.get_device_capability(0) < MIN_COMPUTE_CAPABILITY:
        return False
    import nvvfx  # noqa: WPS433

    return hasattr(nvvfx, "VideoFrameGeneration")


def _psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    import math

    mse = float(((x.float() - y.float()) ** 2).mean())
    return 99.0 if mse == 0 else 10 * math.log10(255.0**2 / mse)


@pytest.mark.skipif(not _real_sdk_available(), reason="needs CUDA, nvidia-vfx >= 0.2.0.0 and an Ada+ GPU")
def test_real_effect_interpolates_and_handles_resize():
    device = torch.device("cuda:0")
    gen = RtxFrameGenerator(device=device, mode="medium")
    try:
        with torch.inference_mode():
            # A smooth synthetic frame at a size that is not a multiple of 8
            # (the SDK needs no padding, unlike RIFE).
            h, w = 270, 482
            yy, xx = torch.meshgrid(
                torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device), indexing="ij"
            )
            blob = torch.sin(xx * 12) * torch.cos(yy * 9) * 0.5 + 0.5
            a = torch.stack([blob, blob.roll(20, 1), blob.roll(40, 0)]).mul(255).round().to(torch.uint8)

            # Identical frames: the interpolation must reproduce the frame.
            out = gen.interpolate(a, a, [0.25, 0.5, 0.75])
            assert len(out) == 3
            for f in out:
                assert f.shape == a.shape and f.dtype == torch.uint8 and f.device == a.device
                assert int((f.int() - a.int()).abs().max()) <= 3

            # Horizontal motion: the midpoint must track the half-shifted truth
            # better than a plain cross-fade would (i.e. it really interpolates).
            b = a.roll(6, 2)
            truth = a.roll(3, 2)
            mid = gen.interpolate(a, b, [0.5])[0]
            blend = ((a.float() + b.float()) / 2).round().to(torch.uint8)
            assert _psnr(mid, truth) > _psnr(blend, truth) + 1.0

            # A size change mid-stream reloads the effect instead of failing.
            b0 = torch.zeros((3, 180, 320), dtype=torch.uint8, device=device)
            b1 = torch.full((3, 180, 320), 200, dtype=torch.uint8, device=device)
            mid = gen.interpolate(b0, b1, [0.5])[0]
            assert mid.shape == b0.shape
            assert 60 <= float(mid.float().mean()) <= 140
    finally:
        gen.close()

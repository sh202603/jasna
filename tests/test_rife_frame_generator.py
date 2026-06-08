"""Tests for the RIFE frame-generation backend.

The neural model itself needs weights, so end-to-end interpolation is exercised
with an injected dummy model (linear blend). Padding helpers are tested directly.
"""
from __future__ import annotations

import torch

from jasna.framegen.rife_frame_generator import (
    RifeFrameGenerator,
    _pad_to_multiple,
    _unpad,
)

requires_cuda = __import__("pytest").mark.skipif(
    not torch.cuda.is_available(), reason="RIFE backend requires CUDA"
)


class _BlendModel(torch.nn.Module):
    """Stand-in RIFE model: linear blend by the (broadcast) timestep channel."""

    def forward(self, img0, img1, timestep):
        return img0 * (1 - timestep) + img1 * timestep


def test_pad_to_multiple_pads_and_unpads():
    x = torch.zeros(1, 3, 100, 130)
    padded, h, w = _pad_to_multiple(x, multiple=64)
    assert padded.shape[2] % 64 == 0 and padded.shape[3] % 64 == 0
    assert padded.shape[2] == 128 and padded.shape[3] == 192
    assert (h, w) == (100, 130)
    assert _unpad(padded, h, w).shape == (1, 3, 100, 130)


def test_pad_to_multiple_noop_when_aligned():
    x = torch.zeros(1, 3, 128, 64)
    padded, h, w = _pad_to_multiple(x, multiple=64)
    assert padded is x
    assert (h, w) == (128, 64)


@requires_cuda
def test_interpolate_shapes_and_dtype():
    device = torch.device("cuda:0")
    gen = RifeFrameGenerator(device=device, fp16=False, model=_BlendModel())
    a = torch.randint(0, 256, (3, 100, 130), dtype=torch.uint8, device=device)
    b = torch.randint(0, 256, (3, 100, 130), dtype=torch.uint8, device=device)

    out = gen.interpolate(a, b, [0.25, 0.5, 0.75])
    assert len(out) == 3
    for frame in out:
        assert frame.shape == (3, 100, 130)
        assert frame.dtype == torch.uint8
        assert frame.device.type == "cuda"


@requires_cuda
def test_interpolate_empty_positions():
    device = torch.device("cuda:0")
    gen = RifeFrameGenerator(device=device, fp16=False, model=_BlendModel())
    a = torch.zeros(3, 64, 64, dtype=torch.uint8, device=device)
    assert gen.interpolate(a, a, []) == []


@requires_cuda
def test_loads_torchscript_checkpoint(tmp_path):
    """The backend must load a TorchScript checkpoint (the recommended format)
    via its jit.load path and run it through the (img0,img1,timestep) convention."""
    device = torch.device("cuda:0")
    example = (
        torch.rand(1, 3, 64, 64, device=device),
        torch.rand(1, 3, 64, 64, device=device),
        torch.full((1, 1, 64, 64), 0.5, device=device),
    )
    traced = torch.jit.trace(_BlendModel().to(device).eval(), example)
    ckpt = tmp_path / "rife_ts.pth"
    torch.jit.save(traced, str(ckpt))

    gen = RifeFrameGenerator(device=device, model_path=str(ckpt))
    a = torch.zeros(3, 64, 64, dtype=torch.uint8, device=device)
    b = torch.full((3, 64, 64), 200, dtype=torch.uint8, device=device)
    (mid,) = gen.interpolate(a, b, [0.5])
    assert mid.shape == (3, 64, 64) and mid.dtype == torch.uint8
    assert torch.allclose(mid.float().mean(), torch.tensor(100.0), atol=1.0)


@requires_cuda
def test_fp16_is_default_and_interpolates():
    device = torch.device("cuda:0")
    gen = RifeFrameGenerator(device=device, model=_BlendModel())
    assert gen._dtype == torch.float16
    a = torch.zeros(3, 64, 64, dtype=torch.uint8, device=device)
    b = torch.full((3, 64, 64), 200, dtype=torch.uint8, device=device)
    (mid,) = gen.interpolate(a, b, [0.5])
    assert mid.dtype == torch.uint8
    assert torch.allclose(mid.float().mean(), torch.tensor(100.0), atol=1.0)


class _Fp32OnlyModel(torch.nn.Module):
    """Mimics a TorchScript checkpoint whose warp bakes in a float32 grid:
    half inputs fail the way grid_sample's dtype check does."""

    def forward(self, img0, img1, timestep):
        if img0.dtype is not torch.float32:
            raise RuntimeError("grid_sample(): expected grid and input tensors to have same dtype")
        return img0 * (1 - timestep) + img1 * timestep


@requires_cuda
def test_fp16_probe_falls_back_to_fp32():
    device = torch.device("cuda:0")
    gen = RifeFrameGenerator(device=device, fp16=True, model=_Fp32OnlyModel())
    assert gen._dtype == torch.float32
    a = torch.zeros(3, 64, 64, dtype=torch.uint8, device=device)
    (out,) = gen.interpolate(a, a, [0.5])
    assert out.shape == (3, 64, 64) and out.dtype == torch.uint8


def test_vendored_ifnet_default_scale_list_matches_upstream_v46():
    """The 4.6 weights are trained against the [8, 4, 2, 1] coarse-to-fine
    pyramid; a different default silently degrades the state_dict path."""
    import inspect

    from jasna.models.rife.ifnet import IFNet

    default = inspect.signature(IFNet.forward).parameters["scale_list"].default
    assert default == (8, 4, 2, 1)


@requires_cuda
def test_interpolate_midpoint_blend_is_average():
    device = torch.device("cuda:0")
    gen = RifeFrameGenerator(device=device, fp16=False, model=_BlendModel())
    a = torch.zeros(3, 64, 64, dtype=torch.uint8, device=device)
    b = torch.full((3, 64, 64), 200, dtype=torch.uint8, device=device)
    (mid,) = gen.interpolate(a, b, [0.5])
    # 0.5 blend of 0 and 200 → 100.
    assert torch.allclose(mid.float().mean(), torch.tensor(100.0), atol=1.0)

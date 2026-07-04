from __future__ import annotations

import os
from unittest.mock import MagicMock

# Must be set before jasna.restorer.fp8_upsample is first imported anywhere in
# the test session so the glue ops stay eager (no triton/inductor compiles).
os.environ.setdefault("JASNA_FP8_RECON_NOCOMPILE", "1")

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from jasna.restorer.basicvsrpp_sub_engines import (
    DIRECTIONS,
    BasicVSRPlusPlusNetSplit,
    _try_create_fp8_upsample,
    create_split_forward,
)
from jasna.restorer.fp8_upsample import _ps_int8_impl, _round_up_bucket

_HAS_FP8_GPU = torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 9)
requires_fp8_gpu = pytest.mark.skipif(not _HAS_FP8_GPU, reason="needs CUDA GPU with sm89+")


class _FakeFP8Upsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1


# ---------------------------------------------------------------------------
# Gate / fallback (CPU-only)
# ---------------------------------------------------------------------------

def test_gate_returns_none_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JASNA_FP8_RECON", raising=False)
    assert _try_create_fp8_upsample(MagicMock(), torch.device("cuda"), fp16=True) is None


def test_gate_returns_none_with_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JASNA_FP8_RECON", "1")
    assert _try_create_fp8_upsample(MagicMock(), torch.device("cuda"), fp16=False) is None


def test_gate_returns_none_with_cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JASNA_FP8_RECON", "1")
    assert _try_create_fp8_upsample(MagicMock(), torch.device("cpu"), fp16=True) is None


def test_gate_falls_back_on_constructor_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("JASNA_FP8_RECON", "1")
    import jasna.restorer.fp8_upsample as fp8_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("no fp8 kernels")

    monkeypatch.setattr(fp8_mod, "CudnnFP8Upsample", _boom)
    with caplog.at_level("WARNING"):
        result = _try_create_fp8_upsample(MagicMock(), torch.device("cuda"), fp16=True)
    assert result is None
    assert any("falling back to TRT upsample engine" in r.message for r in caplog.records)


def test_create_split_forward_uses_fp8_and_skips_trt_upsample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jasna.restorer.basicvsrpp_sub_engines as se

    # The default-on CUDA graphs warmup would call the fake loop bodies.
    monkeypatch.setenv("JASNA_TRT_CUDAGRAPHS", "0")
    sentinel = _FakeFP8Upsample()
    monkeypatch.setattr(se, "_try_create_fp8_upsample", lambda *a, **k: sentinel)
    recorded: dict[str, object] = {}

    def _fake_load(model_weights_path, device, fp16, load_upsample=True):
        recorded["load_upsample"] = load_upsample
        return {d: nn.Module() for d in DIRECTIONS}, nn.Module(), None

    monkeypatch.setattr(se, "load_sub_engines", _fake_load)
    split = se.create_split_forward(MagicMock(), "model.pth", torch.device("cuda"), fp16=True)
    assert split is not None
    assert split._upsample_engine is sentinel
    assert recorded["load_upsample"] is False


def test_create_split_forward_releases_fp8_when_engines_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jasna.restorer.basicvsrpp_sub_engines as se

    sentinel = _FakeFP8Upsample()
    monkeypatch.setattr(se, "_try_create_fp8_upsample", lambda *a, **k: sentinel)
    monkeypatch.setattr(se, "load_sub_engines", lambda *a, **k: None)
    result = se.create_split_forward(MagicMock(), "model.pth", torch.device("cuda"), fp16=True)
    assert result is None
    assert sentinel.release_calls == 1


def test_close_dispatches_release() -> None:
    fp8 = _FakeFP8Upsample()
    split = BasicVSRPlusPlusNetSplit(
        MagicMock(),
        {d: nn.Module() for d in DIRECTIONS},
        nn.Module(),
        fp8,
    )
    split.close()
    assert fp8.release_calls == 1


# ---------------------------------------------------------------------------
# Glue-op math / helpers (CPU-only)
# ---------------------------------------------------------------------------

def test_ps_int8_matches_pixel_shuffle() -> None:
    torch.manual_seed(0)
    x = torch.randint(-128, 127, (3, 16, 8, 8), dtype=torch.int8)
    x = x.contiguous(memory_format=torch.channels_last)
    out = _ps_int8_impl(x, 2)
    ref = F.pixel_shuffle(x.float(), 2)
    assert torch.equal(out.float(), ref)
    assert out.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.parametrize(
    ("t", "expected"),
    [(1, 10), (9, 10), (10, 10), (11, 20), (60, 60), (91, 100)],
)
def test_bucket_rounding(t: int, expected: int) -> None:
    assert _round_up_bucket(t) == expected


def test_compile_or_eager_falls_back_permanently(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    import jasna.restorer.fp8_upsample as fp8_mod

    compiled_calls = {"n": 0}

    def _broken_compiled(*args):
        compiled_calls["n"] += 1
        raise RuntimeError("inductor broken")

    monkeypatch.setattr(fp8_mod.torch, "compile", lambda fn, **k: _broken_compiled)
    wrapped = fp8_mod._compile_or_eager(lambda x: x + 1)
    with caplog.at_level("WARNING"):
        assert wrapped(1) == 2  # first call: compiled raises -> eager result
    assert wrapped(2) == 3      # stays eager, compiled not retried
    assert compiled_calls["n"] == 1
    assert any("falling back to eager" in r.message for r in caplog.records)


def test_cli_flag_parsed() -> None:
    from jasna.main import build_parser

    parser = build_parser()
    assert parser.parse_args(["--fp8-recon"]).fp8_recon is True
    assert parser.parse_args([]).fp8_recon is False


# ---------------------------------------------------------------------------
# GPU-gated (CUDA sm89+ and nvidia-cudnn-frontend required)
# ---------------------------------------------------------------------------

def _make_net():
    from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet

    torch.manual_seed(42)
    net = BasicVSRPlusPlusNet(mid_channels=64, num_blocks=2, spynet_pretrained=None)
    return net.eval()


@requires_fp8_gpu
def test_fp8_forward_parity_small(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cudnn")
    from jasna.restorer.basicvsrpp_sub_engines import _UpsampleWrapper
    from jasna.restorer.fp8_upsample import CudnnFP8Upsample

    monkeypatch.setenv("JASNA_FP8_RECON_NOWARM", "1")
    device = torch.device("cuda")
    net = _make_net().half().to(device)
    fp8 = CudnnFP8Upsample(net, device, max_clip_size=20)
    ref_mod = _UpsampleWrapper(
        net.reconstruction, net.upsample1, net.upsample2, net.conv_hr, net.conv_last,
    ).float().to(device).eval()

    torch.manual_seed(7)
    x = (torch.randn(12, 320, 64, 64, device=device) * 0.5).half()
    with torch.inference_mode():
        out = fp8(x).float().clone()
        ref = ref_mod(x.float())
    snr_db = 10 * torch.log10(ref.pow(2).mean() / (out - ref).pow(2).mean()).item()
    assert out.shape == ref.shape
    # Loose smoke gate on random weights; the strict gate (>=60dB PSNR on real
    # weights/activations) lives in the fp8 A/B benchmark.
    assert snr_db >= 15.0, f"FP8 vs FP32 SNR too low: {snr_db:.1f}dB"
    fp8.release()


@requires_fp8_gpu
def test_fp8_bit_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cudnn")
    from jasna.restorer.fp8_upsample import CudnnFP8Upsample

    monkeypatch.setenv("JASNA_FP8_RECON_NOWARM", "1")
    device = torch.device("cuda")
    net = _make_net().half().to(device)
    fp8 = CudnnFP8Upsample(net, device, max_clip_size=20)
    torch.manual_seed(7)
    x = (torch.randn(15, 320, 64, 64, device=device) * 0.5).half()  # pad path (15 -> bucket 20)
    with torch.inference_mode():
        first = fp8(x).clone()
        second = fp8(x.clone())
        assert torch.equal(first, second)
    fp8.release()


@requires_fp8_gpu
def test_fp8_release_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cudnn")
    from jasna.restorer.fp8_upsample import CudnnFP8Upsample

    monkeypatch.setenv("JASNA_FP8_RECON_NOWARM", "1")
    device = torch.device("cuda")
    net = _make_net().half().to(device)
    fp8 = CudnnFP8Upsample(net, device, max_clip_size=10)
    fp8.release()
    fp8.release()
    second = CudnnFP8Upsample(net, device, max_clip_size=10)
    with torch.inference_mode():
        out = second(torch.zeros(10, 320, 64, 64, dtype=torch.float16, device=device))
    assert out.shape == (10, 3, 256, 256)
    second.release()

"""modi-only RTX Super Res tests: ``--rtx-strength`` / ``--rtx-highbitrate`` and the sized-from-frames
restorer of ``--ltx-crop-vsr rtx-super-res``.

modi 独自。上流の tests/test_rtx_superres_restorer.py には追記しない。その偽 nvvfx は strength と
HIGHBITRATE_* を知らないので、ここでは自前の偽 ``VideoSuperRes`` を各テストの間だけ差し込む。
"""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch


class _Quality:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"
    HIGHBITRATE_LOW = "highbitrate_low"
    HIGHBITRATE_MEDIUM = "highbitrate_medium"
    HIGHBITRATE_HIGH = "highbitrate_high"
    HIGHBITRATE_ULTRA = "highbitrate_ultra"
    DENOISE_LOW = "denoise_low"
    DENOISE_MEDIUM = "denoise_medium"
    DENOISE_HIGH = "denoise_high"
    DENOISE_ULTRA = "denoise_ultra"
    DEBLUR_LOW = "deblur_low"
    DEBLUR_MEDIUM = "deblur_medium"
    DEBLUR_HIGH = "deblur_high"
    DEBLUR_ULTRA = "deblur_ultra"


class _MockVideoSuperRes:
    QualityLevel = _Quality

    def __init__(self, *, device, quality, strength=1.0):
        self.device_id = device
        self.quality = quality
        self.strength = strength
        self.output_width = 0
        self.output_height = 0

    def load(self):
        pass

    def run(self, frame, stream_ptr=None):
        result = MagicMock()
        result.image = torch.rand_like(frame)
        return result

    def close(self):
        pass


sys.modules.setdefault("nvvfx", ModuleType("nvvfx"))

from jasna.restorer.rtx_superres_secondary_restorer import RtxSuperresSecondaryRestorer, _resolve_quality  # noqa: E402


@pytest.fixture(autouse=True)
def _modi_fake_nvvfx(monkeypatch):
    """The fake SDK of these tests, whatever nvvfx (real or upstream's fake) is installed."""
    monkeypatch.setattr(sys.modules["nvvfx"], "VideoSuperRes", _MockVideoSuperRes, raising=False)


def _make_restorer(*, scale=4, quality="high", denoise="medium", deblur=None, **kwargs):
    with patch("torch.cuda.current_stream") as mock_stream:
        mock_stream.return_value.cuda_stream = 12345
        return RtxSuperresSecondaryRestorer(
            device=torch.device("cuda:0"), scale=scale, quality=quality, denoise=denoise, deblur=deblur, **kwargs,
        )


class TestStrengthAndHighBitrate:
    def test_resolve_quality_highbitrate_family(self):
        q = _Quality
        assert _resolve_quality("low", highbitrate=True) == q.HIGHBITRATE_LOW
        assert _resolve_quality("medium", highbitrate=True) == q.HIGHBITRATE_MEDIUM
        assert _resolve_quality("HIGH", highbitrate=True) == q.HIGHBITRATE_HIGH
        assert _resolve_quality("ultra", highbitrate=True) == q.HIGHBITRATE_ULTRA
        # The flag only switches the upscale family; off keeps the standard models.
        assert _resolve_quality("high", highbitrate=False) == q.HIGH

    def test_defaults_do_not_pass_strength_to_sdk(self):
        # strength=1.0 is the SDK default and the keyword only exists from
        # nvidia-vfx 0.2.0.0 on, so the default config must not forward it.
        from jasna.restorer import rtx_superres_secondary_restorer as mod

        with patch.object(mod, "_make_effect", wraps=mod._make_effect) as spy, \
                patch.object(sys.modules["nvvfx"], "VideoSuperRes") as cls:
            cls.QualityLevel = _Quality
            _make_restorer()
            for call in cls.call_args_list:
                assert "strength" not in call.kwargs
            assert spy.call_count == 2  # upscale + denoise

    def test_strength_applies_to_every_pass(self):
        restorer = _make_restorer(denoise="medium", deblur="low", strength=0.4)
        assert restorer.strength == 0.4
        assert restorer._sr.strength == 0.4
        assert restorer._denoise.strength == 0.4
        assert restorer._deblur.strength == 0.4

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_strength_out_of_range_rejected(self, bad):
        with pytest.raises(ValueError, match="strength"):
            _make_restorer(strength=bad)

    def test_highbitrate_switches_upscale_model_only(self):
        q = _Quality
        restorer = _make_restorer(quality="high", denoise="medium", deblur="low", highbitrate=True)
        assert restorer.highbitrate is True
        assert restorer._sr.quality == q.HIGHBITRATE_HIGH
        assert restorer._denoise.quality == q.DENOISE_MEDIUM
        assert restorer._deblur.quality == q.DEBLUR_LOW


class TestSizedFromFrames:
    """``input_size=None`` (the ``--ltx-crop-vsr`` restorer): the passes are loaded for
    an input side the first time frames of that side arrive, one chain per side, so the
    512 and 768 px canvases of one video never reload an effect."""

    @staticmethod
    def _fake_effect(VideoSuperRes, *, gpu, quality, strength, output_size):
        effect = MagicMock(output_width=output_size, output_height=output_size, quality=quality)
        effect.run.side_effect = lambda frame, stream_ptr=None: MagicMock(
            image=torch.rand((3, output_size, output_size), dtype=torch.float32)
        )
        return effect

    def _restorer(self, monkeypatch, **kwargs):
        from jasna.restorer import rtx_superres_secondary_restorer as mod

        monkeypatch.setattr(mod, "_make_effect", self._fake_effect)
        with patch("torch.cuda.current_stream") as mock_stream:
            mock_stream.return_value.cuda_stream = 12345
            return RtxSuperresSecondaryRestorer(device=torch.device("cuda:0"), scale=2, input_size=None, **kwargs)

    def test_loads_nothing_until_the_first_frames(self, monkeypatch):
        restorer = self._restorer(monkeypatch)
        assert restorer.input_size is None and restorer.output_size is None
        assert restorer._chains == {} and restorer._sr is None

    def test_one_chain_per_input_side_at_twice_the_side(self, monkeypatch):
        restorer = self._restorer(monkeypatch, denoise="medium", deblur=None)
        out = restorer.restore(torch.rand((2, 3, 8, 8)), keep_start=0, keep_end=2)
        restorer.restore(torch.rand((1, 3, 12, 12)), keep_start=0, keep_end=1)
        restorer.restore(torch.rand((1, 3, 8, 8)), keep_start=0, keep_end=1)
        assert sorted(restorer._chains) == [8, 12]
        assert [tuple(frame.shape) for frame in out] == [(3, 16, 16), (3, 16, 16)] and out[0].dtype == torch.uint8
        chain = restorer._chains[12]
        assert (chain.sr.output_width, chain.sr.output_height) == (24, 24)
        assert chain.denoise.output_width == 24 and chain.deblur is None
        assert restorer._chains[8].sr.run.call_count == 3

    def test_rejects_frames_that_are_not_square(self, monkeypatch):
        restorer = self._restorer(monkeypatch)
        with pytest.raises(ValueError, match="expected frames shaped"):
            restorer.restore(torch.rand((1, 3, 8, 12)), keep_start=0, keep_end=1)

    def test_close_closes_every_chain(self, monkeypatch):
        restorer = self._restorer(monkeypatch, denoise="low", deblur="low")
        restorer.restore(torch.rand((1, 3, 8, 8)), keep_start=0, keep_end=1)
        restorer.restore(torch.rand((1, 3, 12, 12)), keep_start=0, keep_end=1)
        effects = [e for chain in restorer._chains.values() for e in (chain.sr, chain.denoise, chain.deblur)]
        assert len(effects) == 6
        restorer.close()
        assert restorer._chains == {} and all(effect.close.call_count == 1 for effect in effects)
        restorer.close()

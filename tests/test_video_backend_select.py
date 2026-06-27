"""Pure unit tests for the video backend selection predicates.

No GPU, no torchcodec, no native libs required: ``torchcodec_available`` is
monkeypatched and the predicates are exercised directly. This locks the
torchcodec eligibility matrix (the narrow HEVC/8-bit/BT.709/default-settings
case) and the native-default behavior.
"""
from __future__ import annotations

from fractions import Fraction

import pytest
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media import Colorspace, VideoMetadata
from jasna.media import backend as backend_mod
from jasna.media.backend import (
    VideoBackend,
    select_decoder_backend,
    select_encoder_backend,
    torchcodec_encoder_eligibility,
)


def _meta(**overrides) -> VideoMetadata:
    defaults = dict(
        video_file="fake.mkv",
        num_frames=100,
        video_fps=24.0,
        average_fps=24.0,
        video_fps_exact=Fraction(24, 1),
        codec_name="hevc",
        duration=100.0 / 24.0,
        video_width=1920,
        video_height=1080,
        time_base=Fraction(1, 24),
        start_pts=0,
        color_space=AvColorspace.ITU709,
        color_range=AvColorRange.MPEG,
        is_10bit=False,
        yuv_colorspace=Colorspace.BT709,
    )
    defaults.update(overrides)
    return VideoMetadata(**defaults)


@pytest.fixture
def tc_available(monkeypatch):
    monkeypatch.setattr(backend_mod, "torchcodec_available", lambda: True)


@pytest.fixture
def tc_unavailable(monkeypatch):
    monkeypatch.setattr(backend_mod, "torchcodec_available", lambda: False)


# --- decoder selection ------------------------------------------------------


class TestDecoderSelection:
    def test_native_always_native(self, tc_available):
        assert select_decoder_backend(VideoBackend.NATIVE, metadata=_meta()) == "native"

    def test_auto_uses_torchcodec_when_available(self, tc_available):
        assert select_decoder_backend(VideoBackend.AUTO, metadata=_meta()) == "torchcodec"

    def test_auto_falls_back_when_unavailable(self, tc_unavailable):
        assert select_decoder_backend(VideoBackend.AUTO, metadata=_meta()) == "native"

    def test_torchcodec_forced_ok_when_available(self, tc_available):
        assert select_decoder_backend(VideoBackend.TORCHCODEC, metadata=_meta()) == "torchcodec"

    def test_torchcodec_forced_raises_when_unavailable(self, tc_unavailable):
        with pytest.raises(ValueError, match="not installed"):
            select_decoder_backend(VideoBackend.TORCHCODEC, metadata=_meta())

    def test_string_backend_accepted(self, tc_available):
        assert select_decoder_backend("auto", metadata=_meta()) == "torchcodec"


# --- encoder eligibility matrix ---------------------------------------------


class TestEncoderEligibility:
    def test_baseline_hevc_8bit_bt709_eligible(self):
        ok, _ = torchcodec_encoder_eligibility(
            codec="hevc", bit_depth=None, metadata=_meta(is_10bit=False),
            encoder_settings={}, output_fps_multiplier=1, stream_mode=False,
        )
        assert ok is True

    def test_av1_8bit_eligible(self):
        ok, _ = torchcodec_encoder_eligibility(
            codec="av1", bit_depth=None, metadata=_meta(is_10bit=False),
            encoder_settings={}, output_fps_multiplier=1, stream_mode=False,
        )
        assert ok is True

    def test_av1_10bit_ineligible(self):
        ok, reason = torchcodec_encoder_eligibility(
            codec="av1", bit_depth=None, metadata=_meta(is_10bit=True),
            encoder_settings={}, output_fps_multiplier=1, stream_mode=False,
        )
        assert ok is False and "10-bit" in reason

    def test_other_codec_ineligible(self):
        ok, reason = torchcodec_encoder_eligibility(
            codec="vp9", bit_depth=None, metadata=_meta(is_10bit=False),
            encoder_settings={}, output_fps_multiplier=1, stream_mode=False,
        )
        assert ok is False and "HEVC or AV1" in reason

    def test_10bit_source_ineligible(self):
        ok, reason = torchcodec_encoder_eligibility(
            codec="hevc", bit_depth=None, metadata=_meta(is_10bit=True),
            encoder_settings={}, output_fps_multiplier=1, stream_mode=False,
        )
        assert ok is False and "10-bit" in reason

    def test_bit_depth_override_10_ineligible(self):
        ok, _ = torchcodec_encoder_eligibility(
            codec="hevc", bit_depth=10, metadata=_meta(is_10bit=False),
            encoder_settings={}, output_fps_multiplier=1, stream_mode=False,
        )
        assert ok is False

    def test_bit_depth_override_8_on_10bit_source_eligible(self):
        ok, _ = torchcodec_encoder_eligibility(
            codec="hevc", bit_depth=8, metadata=_meta(is_10bit=True),
            encoder_settings={}, output_fps_multiplier=1, stream_mode=False,
        )
        assert ok is True

    def test_mappable_encoder_settings_eligible(self):
        # cq/lookahead/aq map to torchcodec nvenc options.
        ok, _ = torchcodec_encoder_eligibility(
            codec="hevc", bit_depth=None, metadata=_meta(is_10bit=False),
            encoder_settings={"cq": 22, "lookahead": 16, "aq": 4},
            output_fps_multiplier=1, stream_mode=False,
        )
        assert ok is True

    def test_unmappable_encoder_settings_ineligible(self):
        # rc/tflevel have no torchcodec nvenc mapping -> native.
        ok, reason = torchcodec_encoder_eligibility(
            codec="hevc", bit_depth=None, metadata=_meta(is_10bit=False),
            encoder_settings={"rc": "cbr"}, output_fps_multiplier=1, stream_mode=False,
        )
        assert ok is False and "encoder-settings" in reason

    def test_bt601_eligible(self):
        # Colorspace is handled by the shared remux, so BT.601 is not a gate.
        ok, _ = torchcodec_encoder_eligibility(
            codec="hevc", bit_depth=None,
            metadata=_meta(is_10bit=False, yuv_colorspace=Colorspace.BT601),
            encoder_settings={}, output_fps_multiplier=1, stream_mode=False,
        )
        assert ok is True

    def test_bt2020_eligible(self):
        ok, _ = torchcodec_encoder_eligibility(
            codec="hevc", bit_depth=None,
            metadata=_meta(is_10bit=False, yuv_colorspace=Colorspace.BT2020),
            encoder_settings={}, output_fps_multiplier=1, stream_mode=False,
        )
        assert ok is True

    def test_full_range_ineligible(self):
        ok, _ = torchcodec_encoder_eligibility(
            codec="hevc", bit_depth=None,
            metadata=_meta(is_10bit=False, color_range=AvColorRange.JPEG),
            encoder_settings={}, output_fps_multiplier=1, stream_mode=False,
        )
        assert ok is False

    def test_frame_gen_ineligible(self):
        ok, reason = torchcodec_encoder_eligibility(
            codec="hevc", bit_depth=None, metadata=_meta(is_10bit=False),
            encoder_settings={}, output_fps_multiplier=2, stream_mode=False,
        )
        assert ok is False and "frame generation" in reason

    def test_streaming_ineligible(self):
        ok, reason = torchcodec_encoder_eligibility(
            codec="hevc", bit_depth=None, metadata=_meta(is_10bit=False),
            encoder_settings={}, output_fps_multiplier=1, stream_mode=True,
        )
        assert ok is False and "streaming" in reason


# --- encoder selection (predicate + backend mode) ---------------------------


def _eligible_kwargs():
    return dict(
        codec="hevc", bit_depth=None, metadata=_meta(is_10bit=False),
        encoder_settings={}, output_fps_multiplier=1, stream_mode=False,
    )


class TestEncoderSelection:
    def test_native_always_native(self, tc_available):
        assert select_encoder_backend(VideoBackend.NATIVE, **_eligible_kwargs()) == "native"

    def test_auto_eligible_uses_torchcodec(self, tc_available):
        assert select_encoder_backend(VideoBackend.AUTO, **_eligible_kwargs()) == "torchcodec"

    def test_auto_ineligible_falls_back(self, tc_available):
        kw = _eligible_kwargs()
        kw["bit_depth"] = 10  # 10-bit is ineligible → AUTO falls back to native
        assert select_encoder_backend(VideoBackend.AUTO, **kw) == "native"

    def test_auto_unavailable_falls_back(self, tc_unavailable):
        assert select_encoder_backend(VideoBackend.AUTO, **_eligible_kwargs()) == "native"

    def test_torchcodec_eligible_ok(self, tc_available):
        assert select_encoder_backend(VideoBackend.TORCHCODEC, **_eligible_kwargs()) == "torchcodec"

    def test_torchcodec_ineligible_raises(self, tc_available):
        kw = _eligible_kwargs()
        kw["bit_depth"] = 10
        with pytest.raises(ValueError, match="cannot satisfy"):
            select_encoder_backend(VideoBackend.TORCHCODEC, **kw)

    def test_torchcodec_unavailable_raises(self, tc_unavailable):
        with pytest.raises(ValueError, match="not installed"):
            select_encoder_backend(VideoBackend.TORCHCODEC, **_eligible_kwargs())

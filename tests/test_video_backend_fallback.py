"""Tests for the AUTO fallback mechanism and factory dispatch (no GPU).

Exercises ``_FallbackVideoReader`` directly with fake factories, and checks that
``make_video_reader`` / ``make_video_encoder`` route to the right backend without
constructing GPU resources. Reader construction is lightweight (``__init__`` only
stores attributes); the torchcodec path stays lazy until ``__enter__``.
"""
from __future__ import annotations

from fractions import Fraction

import pytest
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media import VideoMetadata
from jasna.media import backend as backend_mod
from jasna.media.backend import (
    VideoBackend,
    _FallbackVideoReader,
    make_video_encoder,
    make_video_reader,
)


class _FakeReader:
    def __init__(self, name, enter_raises=False):
        self.name = name
        self.enter_raises = enter_raises
        self.entered = False
        self.exited = False

    def __enter__(self):
        if self.enter_raises:
            raise RuntimeError(f"{self.name} __enter__ boom")
        self.entered = True
        return self

    def __exit__(self, *a):
        self.exited = True
        return False


class TestFallbackReaderAdapter:
    def test_uses_torchcodec_when_it_succeeds(self):
        tc = _FakeReader("tc")
        native = _FakeReader("native")
        adapter = _FallbackVideoReader(lambda: tc, lambda: native)
        with adapter as r:
            assert r is tc and tc.entered
        assert tc.exited and not native.entered

    def test_falls_back_to_native_when_torchcodec_enter_raises(self):
        tc = _FakeReader("tc", enter_raises=True)
        native = _FakeReader("native")
        adapter = _FallbackVideoReader(lambda: tc, lambda: native)
        with adapter as r:
            assert r is native and native.entered
        assert native.exited

    def test_falls_back_when_torchcodec_factory_raises(self):
        def _tc():
            raise ImportError("no torchcodec")

        native = _FakeReader("native")
        adapter = _FallbackVideoReader(_tc, lambda: native)
        with adapter as r:
            assert r is native


class TestMakeVideoReaderDispatch:
    def test_native_returns_reader_directly(self, monkeypatch):
        from jasna.media.video_decoder import NvidiaVideoReader

        monkeypatch.setattr(backend_mod, "torchcodec_available", lambda: True)
        r = make_video_reader(file="x.mp4", batch_size=4, device=object(), metadata=object(),
                              backend=VideoBackend.NATIVE)
        assert isinstance(r, NvidiaVideoReader)

    def test_auto_available_wraps_in_fallback(self, monkeypatch):
        monkeypatch.setattr(backend_mod, "torchcodec_available", lambda: True)
        r = make_video_reader(file="x.mp4", batch_size=4, device=object(), metadata=object(),
                              backend=VideoBackend.AUTO)
        assert isinstance(r, _FallbackVideoReader)

    def test_auto_unavailable_returns_native(self, monkeypatch):
        from jasna.media.video_decoder import NvidiaVideoReader

        monkeypatch.setattr(backend_mod, "torchcodec_available", lambda: False)
        r = make_video_reader(file="x.mp4", batch_size=4, device=object(), metadata=object(),
                              backend=VideoBackend.AUTO)
        assert isinstance(r, NvidiaVideoReader)

    def test_torchcodec_unavailable_raises(self, monkeypatch):
        monkeypatch.setattr(backend_mod, "torchcodec_available", lambda: False)
        with pytest.raises(ValueError, match="not installed"):
            make_video_reader(file="x.mp4", batch_size=4, device=object(), metadata=object(),
                              backend=VideoBackend.TORCHCODEC)


def _meta(**ov) -> VideoMetadata:
    d = dict(
        video_file="f.mkv", num_frames=10, video_fps=24.0, average_fps=24.0,
        video_fps_exact=Fraction(24, 1), codec_name="hevc", duration=1.0,
        video_width=320, video_height=240, time_base=Fraction(1, 24), start_pts=0,
        color_space=AvColorspace.ITU709, color_range=AvColorRange.MPEG,
        is_10bit=False,
    )
    d.update(ov)
    return VideoMetadata(**d)


class TestMakeVideoEncoderDispatch:
    def test_torchcodec_forced_ineligible_raises_before_construction(self, monkeypatch):
        monkeypatch.setattr(backend_mod, "torchcodec_available", lambda: True)
        with pytest.raises(ValueError, match="cannot satisfy"):
            make_video_encoder(
                file="o.mkv", device=object(), metadata=_meta(is_10bit=True), codec="hevc",
                encoder_settings={}, backend=VideoBackend.TORCHCODEC,
            )

    def test_torchcodec_unavailable_raises(self, monkeypatch):
        monkeypatch.setattr(backend_mod, "torchcodec_available", lambda: False)
        with pytest.raises(ValueError, match="not installed"):
            make_video_encoder(
                file="o.mkv", device=object(), metadata=_meta(), codec="hevc",
                encoder_settings={}, backend=VideoBackend.TORCHCODEC,
            )

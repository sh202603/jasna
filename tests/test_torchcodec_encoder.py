"""GPU + unit tests for the torchcodec encoder.

The round-trip test (GPU/ffmpeg-guarded) encodes synthetic frames through
``TorchcodecVideoEncoder`` and checks the muxed output with ffprobe: codec,
8-bit pixel format, frame count, source color tags, and carried audio. The
rejection tests are pure (no GPU): ineligible requests must raise
``TorchcodecEncodeUnsupported`` from the constructor.
"""
from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchcodec")

from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange  # noqa: E402

from jasna.media import Colorspace, VideoMetadata  # noqa: E402


def _meta(video_file="f.mkv", **ov) -> VideoMetadata:
    d = dict(
        video_file=video_file, num_frames=24, video_fps=24.0, average_fps=24.0,
        video_fps_exact=Fraction(24, 1), codec_name="hevc", duration=1.0,
        video_width=320, video_height=240, time_base=Fraction(1, 24), start_pts=0,
        color_space=AvColorspace.ITU709, color_range=AvColorRange.MPEG,
        is_10bit=False, yuv_colorspace=Colorspace.BT709,
    )
    d.update(ov)
    return VideoMetadata(**d)


class TestRejection:
    def _enc(self, **ov):
        from jasna.media.torchcodec_encoder import TorchcodecVideoEncoder

        kw = dict(file="o.mkv", device=torch.device("cuda:0"), metadata=_meta(),
                  codec="hevc", encoder_settings={})
        meta_ov = {k: ov.pop(k) for k in ("is_10bit", "yuv_colorspace", "color_range") if k in ov}
        if meta_ov:
            kw["metadata"] = _meta(**meta_ov)
        kw.update(ov)
        return TorchcodecVideoEncoder(**kw)

    def test_10bit_rejected(self):
        from jasna.media.torchcodec_encoder import TorchcodecEncodeUnsupported
        with pytest.raises(TorchcodecEncodeUnsupported, match="10-bit"):
            self._enc(bit_depth=10)

    def test_unmappable_settings_rejected(self):
        from jasna.media.torchcodec_encoder import TorchcodecEncodeUnsupported
        with pytest.raises(TorchcodecEncodeUnsupported):
            self._enc(encoder_settings={"rc": "cbr"})  # rc has no torchcodec mapping

    def test_mappable_settings_build_params(self):
        # Mappable settings are accepted and overlaid onto the default tuning.
        enc = self._enc(encoder_settings={"cq": 30, "lookahead": 8, "aq": 5, "preset": "P6"})
        assert enc._preset == "p6"
        assert enc._extra_options["cq"] == "30"
        assert enc._extra_options["rc-lookahead"] == "8"
        assert enc._extra_options["spatial-aq"] == "1" and enc._extra_options["aq-strength"] == "5"

    def test_streaming_rejected(self):
        from jasna.media.torchcodec_encoder import TorchcodecEncodeUnsupported
        with pytest.raises(TorchcodecEncodeUnsupported):
            self._enc(stream_mode=True)

    def test_framegen_rejected(self):
        from jasna.media.torchcodec_encoder import TorchcodecEncodeUnsupported
        with pytest.raises(TorchcodecEncodeUnsupported):
            self._enc(output_fps_multiplier=2)

    def test_other_codec_rejected(self):
        from jasna.media.torchcodec_encoder import TorchcodecEncodeUnsupported
        with pytest.raises(TorchcodecEncodeUnsupported):
            self._enc(codec="vp9")


# --- GPU round-trip ---------------------------------------------------------

_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or shutil.which("ffmpeg") is None,
    reason="CUDA GPU + ffmpeg required",
)


def _ffprobe(path, stream, entries):
    # key=value output: ffprobe emits fields in its own fixed order, so parse by name.
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream, "-count_frames",
         "-show_entries", f"stream={entries}", "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True,
    )
    d = {}
    for line in out.stdout.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    return d


@_gpu
@pytest.mark.parametrize("codec,expect_codec", [("hevc", "hevc"), ("av1", "av1")])
def test_encode_roundtrip(tmp_path, codec, expect_codec):
    from jasna.media import get_video_meta_data
    from jasna.media.torchcodec_encoder import TorchcodecVideoEncoder

    device = torch.device("cuda:0")
    src = tmp_path / "src.mkv"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=duration=1:size=320x240:rate=24",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(src)],
        check=True,
    )
    meta = get_video_meta_data(str(src))
    out = tmp_path / f"out_{codec}.mkv"

    enc = TorchcodecVideoEncoder(
        str(out), device=device, metadata=meta, codec=codec, encoder_settings={},
        working_directory=tmp_path,
    )
    n_frames = int(meta.num_frames)
    with enc:
        for _ in range(n_frames):
            enc.encode((torch.rand(3, 240, 320, device=device) * 255).to(torch.uint8), pts=0)

    assert out.exists() and out.stat().st_size > 0
    v = _ffprobe(out, "v:0", "codec_name,pix_fmt,nb_read_frames,color_space")
    assert v["codec_name"] == expect_codec, v
    assert v["pix_fmt"] == "yuv420p", v
    assert int(v["nb_read_frames"]) == n_frames, v
    assert v["color_space"] == "bt709", v
    assert _ffprobe(out, "a:0", "codec_name")["codec_name"] == "aac"  # audio carried by remux

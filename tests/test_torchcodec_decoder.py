"""GPU parity tests for the torchcodec decoder (opt-in, availability-guarded).

Skipped unless a CUDA GPU, torchcodec, and ffmpeg are all present. Generates a
short clip with ffmpeg, decodes it through ``TorchcodecVideoReader``, and checks
the ``NvidiaVideoReader`` contract plus parity facts established in pre-flight:

- output is ``uint8`` NCHW on CUDA;
- PTS are strictly increasing integer ticks and match the native reader exactly
  for the overlapping prefix;
- torchcodec returns the ground-truth frame count (matches the number of frames
  ffmpeg encoded), and pixels are close to native but not bit-identical.

Note: the native vali reader over-decodes by 2 frames at EOF on these clips
(an NVDEC flush quirk), so we assert torchcodec == ground truth, not == native.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchcodec")

if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)
if shutil.which("ffmpeg") is None:
    pytest.skip("ffmpeg required to synthesize a clip", allow_module_level=True)


N_FRAMES = 48
FPS = 24
W, H = 320, 240


@pytest.fixture(scope="module")
def _cuda_ctx():
    torch.cuda.init()
    torch.cuda.set_device(0)
    _ = torch.zeros(1, device="cuda")  # context for vali's cuStreamCreate
    return torch.device("cuda:0")


def _make_clip(path, pix_fmt="yuv420p", vcodec="libx264"):
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc2=duration={N_FRAMES / FPS}:size={W}x{H}:rate={FPS}",
         "-pix_fmt", pix_fmt, "-c:v", vcodec, str(path)],
        check=True,
    )
    return str(path)


def _drain(reader):
    total, pts_all, first = 0, [], None
    with reader as r:
        for data, pts in r.frames():
            assert data.dtype == torch.uint8 and data.ndim == 4 and data.shape[1] == 3
            assert data.device.type == "cuda"
            # torchcodec returns a non-contiguous NCHW view of its internal HWC buffer;
            # the reader must make it contiguous, or raw-memory consumers (the detection
            # TensorRT engine) read garbage and silently detect nothing.
            assert data.is_contiguous(), "reader must yield contiguous NCHW tensors"
            if first is None:
                first = data[0].float().clone()
            total += data.shape[0]
            pts_all.extend(pts)
    return total, pts_all, first


def test_torchcodec_decoder_contract_and_parity(tmp_path, _cuda_ctx):
    from jasna.media import get_video_meta_data
    from jasna.media.torchcodec_decoder import TorchcodecVideoReader

    device = _cuda_ctx
    clip = _make_clip(tmp_path / "clip.mp4")
    meta = get_video_meta_data(clip)

    total, pts, first = _drain(
        TorchcodecVideoReader(clip, batch_size=5, device=device, metadata=meta)
    )

    assert total == N_FRAMES, f"torchcodec frame count {total} != ground truth {N_FRAMES}"
    assert all(pts[i] < pts[i + 1] for i in range(len(pts) - 1)), "PTS not strictly increasing"
    assert pts[0] == 0


def test_torchcodec_matches_native_pts_and_pixels(tmp_path, _cuda_ctx):
    from jasna.media import get_video_meta_data
    from jasna.media.torchcodec_decoder import TorchcodecVideoReader
    from jasna.media.video_decoder import NvidiaVideoReader

    device = _cuda_ctx
    clip = _make_clip(tmp_path / "clip.mp4")
    meta = get_video_meta_data(clip)

    n_total, n_pts, n_first = _drain(NvidiaVideoReader(clip, batch_size=5, device=device, metadata=meta))
    t_total, t_pts, t_first = _drain(TorchcodecVideoReader(clip, batch_size=5, device=device, metadata=meta))

    # PTS match exactly on the overlapping prefix.
    k = min(len(n_pts), len(t_pts))
    assert n_pts[:k] == t_pts[:k], "PTS diverge from native"

    # torchcodec is frame-exact to ffmpeg; native over-decodes here, so only
    # require torchcodec >= ground truth and not wildly off from native.
    assert t_total == N_FRAMES
    assert abs(n_total - t_total) <= 2

    # Close but not bit-identical (different color conversion).
    mad = (n_first - t_first).abs().mean().item()
    assert mad < 20.0, f"first-frame mean-abs-diff too large: {mad:.2f}"

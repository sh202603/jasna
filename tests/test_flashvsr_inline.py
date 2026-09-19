"""Tests for the inline FlashVSR secondary restorer (no GPU, no FlashVSR checkout).

The real worker is replaced by a tiny stub that speaks the same length-prefixed
protocol (uint8 **BGR** on the wire, like the lada-ex worker the real one is
kept identical with) and echoes a per-frame solid color, so we can exercise the
parent's wire handling (handshake, send, receive, frame-count contract, [ks:ke]
slicing, the RGB<->BGR flips) on a CPU box.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

import jasna.restorer.flashvsr_inline_secondary_restorer as fi
from jasna.restorer.flashvsr_inline_secondary_restorer import (
    FlashvsrInlineSecondaryRestorer,
    _check_patched_repo,
)
# The worker module top-level is stdlib-only (numpy/torch/run are imported inside
# main()), so these helpers import cleanly on a CPU box without a FlashVSR checkout.
from jasna.restorer.flashvsr_inline_worker import (
    _adain,
    _color_fix_frames,
    _feather_mask_numpy,
    _stitch_tiles,
    _strip_coords,
    _wavelet_reconstruct,
)

# A stub "worker": ready handshake, then echo each input frame as a solid
# (256*scale)^2 frame of that frame's per-channel mean (preserves the wire
# channel order, so a round trip through the parent's two flips must come back
# RGB). It can be told to lie about the frame count or emit an error, to test
# those paths, and has two modes that pin the wire color order itself:
#   assert_red_wire: every input frame must arrive as wire (0,0,255) = BGR red,
#                    else it reports an error (catches a missing outbound flip);
#   wire_marker:     returns wire (0,0,255) frames, which the parent must hand
#                    back as RGB red (catches a missing inbound flip).
_STUB_WORKER = textwrap.dedent(
    """
    import os, sys, json
    import numpy as np

    MODE = os.environ.get("STUB_MODE", "ok")

    def rh(f):
        line = bytearray()
        while True:
            b = f.read(1)
            if not b:
                return None
            if b == b"\\n":
                break
            line += b
        return json.loads(line.decode())

    def rx(f, n):
        buf = bytearray()
        while len(buf) < n:
            c = f.read(n - len(buf))
            if not c:
                raise EOFError()
            buf += c
        return bytes(buf)

    out = sys.stdout.buffer
    stdin = sys.stdin.buffer
    if os.environ.get("STUB_STDERR"):
        sys.stderr.write("[W716 07:57:26.7 CUDACachingAllocator.cpp:508] expandable_segments: memory mapping failed with OOM on device 0\\n")
        sys.stderr.write("STUB_MARKER real diagnostic line\\n")
        sys.stderr.flush()
    out.write((json.dumps({"status": "ready"}) + "\\n").encode()); out.flush()
    O = 256 * int(sys.argv[sys.argv.index("--scale") + 1]) if "--scale" in sys.argv else 1024
    while True:
        h = rh(stdin)
        if h is None:
            break
        n, ih, iw = h["n"], h["h"], h["w"]
        crops = np.frombuffer(rx(stdin, n * ih * iw * 3), np.uint8).reshape(n, ih, iw, 3)
        if MODE == "error":
            out.write((json.dumps({"seq": h.get("seq", 0), "error": "boom"}) + "\\n").encode()); out.flush()
            continue
        if MODE == "assert_red_wire":
            ok = all(int(crops[i, :, :, 2].min()) == 255 and int(crops[i, :, :, 0].max()) == 0
                     and int(crops[i, :, :, 1].max()) == 0 for i in range(n))
            if not ok:
                out.write((json.dumps({"seq": h.get("seq", 0), "error": "wire is not BGR"}) + "\\n").encode()); out.flush()
                continue
        rn = n - 1 if MODE == "shortcount" else n
        arr = np.empty((rn, O, O, 3), np.uint8)
        for i in range(rn):
            for c in range(3):
                arr[i, :, :, c] = 255 if (MODE == "wire_marker" and c == 2) else (
                    0 if MODE == "wire_marker" else int(round(crops[i, :, :, c].mean())))
        out.write((json.dumps({"seq": h.get("seq", 0), "n": rn, "h": O, "w": O}) + "\\n").encode())
        out.write(arr.tobytes()); out.flush()
    """
)


@pytest.fixture
def stub_env(tmp_path, monkeypatch):
    """A fake patched repo + a stub worker script, wired into the restorer."""
    repo = tmp_path / "repo"
    (repo / "models" / "FlashVSR-v1.1").mkdir(parents=True)
    tinylong = repo / "src" / "pipelines"
    tinylong.mkdir(parents=True)
    (tinylong / "flashvsr_tiny_long.py").write_text("# FIX(jasna) applied\n")
    worker = tmp_path / "stub_worker.py"
    worker.write_text(_STUB_WORKER)
    monkeypatch.setattr(fi, "_resolve_worker_script", lambda: worker)
    return {"repo": repo, "worker": worker, "model_dir": repo / "models" / "FlashVSR-v1.1"}


def _make_restorer(stub_env):
    # sys.executable (test venv) runs the stub worker. The stub reads its mode
    # from the STUB_MODE env var (inherited by the child); tests set it via
    # monkeypatch.setenv before calling this.
    return FlashvsrInlineSecondaryRestorer(
        repo=stub_env["repo"],
        model_dir=stub_env["model_dir"],
        fv_python=Path(sys.executable),
        startup_timeout_s=30.0,
    )


class TestPatchedRepoCheck:
    def test_accepts_patched(self, tmp_path):
        p = tmp_path / "src" / "pipelines"
        p.mkdir(parents=True)
        (p / "flashvsr_tiny_long.py").write_text("x = 1  # FIX(jasna)\n")
        _check_patched_repo(tmp_path)  # no raise

    def test_accepts_lada_ex_marker(self, tmp_path):
        # The identical fix ships in lada-ex under its own marker; a checkout
        # patched from there must not be rejected.
        p = tmp_path / "src" / "pipelines"
        p.mkdir(parents=True)
        (p / "flashvsr_tiny_long.py").write_text("x = 1  # FIX(lada-ex)\n")
        _check_patched_repo(tmp_path)  # no raise

    def test_rejects_unpatched(self, tmp_path):
        p = tmp_path / "src" / "pipelines"
        p.mkdir(parents=True)
        (p / "flashvsr_tiny_long.py").write_text("x = 1\n")
        with pytest.raises(RuntimeError, match="multi-chunk fix"):
            _check_patched_repo(tmp_path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _check_patched_repo(tmp_path)


class TestRestoreWireRoundTrip:
    def test_class_attrs_route_sync(self):
        # must not accidentally satisfy AsyncSecondaryRestorer
        from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer
        assert FlashvsrInlineSecondaryRestorer.name == "flashvsr-inline"
        assert FlashvsrInlineSecondaryRestorer.prefers_cpu_input is True
        for m in ("push_clip", "pop_completed", "has_pending", "flush_pending", "flush_all", "_to_tensors"):
            assert not hasattr(FlashvsrInlineSecondaryRestorer, m)

    def test_roundtrip_frame_count_and_rgb(self, stub_env):
        r = _make_restorer(stub_env)
        try:
            # 4 distinct solid-color frames: red, green, blue, white
            colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)]
            frames = torch.zeros(4, 3, 256, 256)
            for i, (rr, gg, bb) in enumerate(colors):
                frames[i, 0] = rr
                frames[i, 1] = gg
                frames[i, 2] = bb
            out = r.restore(frames, keep_start=0, keep_end=4)
            assert len(out) == 4
            for t in out:
                assert t.shape == (3, 1024, 1024)
                assert t.dtype == torch.uint8
            # RGB channel order preserved: frame 0 red -> ch0 high, ch1/2 low
            assert out[0][0].float().mean() > 200 and out[0][1].float().mean() < 40
            assert out[1][1].float().mean() > 200 and out[1][0].float().mean() < 40
            assert out[2][2].float().mean() > 200 and out[2][0].float().mean() < 40
        finally:
            r.close()

    def test_outbound_wire_is_bgr(self, stub_env, monkeypatch):
        # RGB red in -> the stub must see wire (0,0,255), i.e. the parent flips
        # RGB->BGR before sending (the stub errors otherwise).
        monkeypatch.setenv("STUB_MODE", "assert_red_wire")
        r = _make_restorer(stub_env)
        try:
            frames = torch.zeros(3, 3, 256, 256)
            frames[:, 0] = 1.0
            out = r.restore(frames, keep_start=0, keep_end=3)
            assert len(out) == 3
        finally:
            r.close()

    def test_inbound_wire_is_bgr(self, stub_env, monkeypatch):
        # The stub returns wire (0,0,255) = BGR red -> the parent must hand back
        # RGB red (channel 0 high, channel 2 zero).
        monkeypatch.setenv("STUB_MODE", "wire_marker")
        r = _make_restorer(stub_env)
        try:
            out = r.restore(torch.zeros(2, 3, 256, 256), keep_start=0, keep_end=2)
            for t in out:
                assert int(t[0].min()) == 255 and int(t[1].max()) == 0 and int(t[2].max()) == 0
        finally:
            r.close()

    def test_keep_window_slice(self, stub_env):
        r = _make_restorer(stub_env)
        try:
            frames = torch.zeros(6, 3, 256, 256)
            for i in range(6):
                frames[i, 0] = i / 10.0  # distinct red ramp
            out = r.restore(frames, keep_start=2, keep_end=5)
            assert len(out) == 3  # keep_end - keep_start
        finally:
            r.close()

    def test_empty_window(self, stub_env):
        r = _make_restorer(stub_env)
        try:
            frames = torch.zeros(4, 3, 256, 256)
            assert r.restore(frames, keep_start=2, keep_end=2) == []
            assert r.restore(torch.zeros(0, 3, 256, 256), keep_start=0, keep_end=0) == []
        finally:
            r.close()


class TestErrorPaths:
    def test_frame_count_mismatch_raises(self, stub_env, monkeypatch):
        monkeypatch.setenv("STUB_MODE", "shortcount")
        r = _make_restorer(stub_env)
        try:
            frames = torch.zeros(4, 3, 256, 256)
            with pytest.raises(RuntimeError, match="frame-count mismatch"):
                r.restore(frames, keep_start=0, keep_end=4)
        finally:
            r.close()

    def test_worker_error_raises(self, stub_env, monkeypatch):
        monkeypatch.setenv("STUB_MODE", "error")
        r = _make_restorer(stub_env)
        try:
            frames = torch.zeros(4, 3, 256, 256)
            with pytest.raises(RuntimeError, match="worker error"):
                r.restore(frames, keep_start=0, keep_end=4)
        finally:
            r.close()


class TestStripGeometry:
    """The 256px crop splits into N uniform, fully-covering full-width strips."""

    # Strip height snaps to a (128 // scale)-multiple so the upscaled strip is a
    # 128-multiple: 32 at 4x (unchanged geometry), 64 at 2x.
    @pytest.mark.parametrize("scale,n_tiles,strip_h,overlap_px", [
        (4, 2, 160, 64), (4, 3, 128, 64), (4, 4, 96, 43),
        (2, 2, 192, 128), (2, 3, 128, 64), (2, 4, 128, 85),
    ])
    def test_uniform_full_width_and_covering(self, scale, n_tiles, strip_h, overlap_px):
        coords, overlap = _strip_coords(256, 256, n_tiles, scale)
        assert len(coords) == n_tiles
        assert overlap == overlap_px
        rows = set()
        for (x1, y1, x2, y2) in coords:
            assert (x1, x2) == (0, 256)          # full width
            assert y2 - y1 == strip_h            # uniform height
            assert (strip_h * scale) % 128 == 0  # upscaled strip is a DiT 128-multiple
            rows.update(range(y1, y2))
        assert rows == set(range(256))           # full vertical coverage

    @pytest.mark.parametrize("scale", [2, 4])
    def test_single_tile_is_full_frame(self, scale):
        coords, overlap = _strip_coords(256, 256, 1, scale)
        assert coords == [(0, 0, 256, 256)] and overlap == 0


class TestStitchTiles:
    """The CPU feather-composite that replaces run.py's mp4 stitch_video_tiles."""

    @pytest.mark.parametrize("n_tiles", [2, 3, 4])
    def test_uniform_field_reconstructs(self, n_tiles):
        # A uniform value V across all strips must reconstruct to V everywhere
        # except the outermost 1px (feather mask ramps to exactly 0 there — an
        # inherited FlashVSR trait; weight-sum normalisation recovers the interior).
        scale, m, V = 1, 2, 100
        coords, overlap = _strip_coords(256, 256, n_tiles, 4)
        tiles = [
            np.full((m, y2 - y1, x2 - x1, 3), V, dtype=np.uint8)
            for (x1, y1, x2, y2) in coords
        ]
        out = _stitch_tiles(tiles, coords, scale, overlap, 256, 256)
        assert out.shape == (m, 256, 256, 3) and out.dtype == np.float32
        assert np.allclose(out[:, 1:-1, 1:-1, :], V, atol=1e-3)
        # true canvas border: weight 0 -> normalised to 0
        assert np.all(out[:, 0, :, :] == 0) and np.all(out[:, -1, :, :] == 0)
        assert np.all(out[:, :, 0, :] == 0) and np.all(out[:, :, -1, :] == 0)

    def test_two_strip_crossfade(self):
        # Two vertical strips: single-coverage rows keep the exact strip value;
        # the overlap band is a monotone feather crossfade between them.
        scale, overlap, m = 1, 4, 1
        A, B = 100.0, 200.0
        coords = [(0, 0, 24, 16), (0, 8, 24, 24)]  # full width, 8px overlap at y in [8,16)
        tiles = [
            np.full((m, 16, 24, 3), A, dtype=np.uint8),
            np.full((m, 16, 24, 3), B, dtype=np.uint8),
        ]
        out = _stitch_tiles(tiles, coords, scale, overlap, 24, 24)
        col = out[0, :, 12, 0]  # a mid-width column, red channel
        # top A-only interior (past the 1px border, before the overlap) == A
        assert np.allclose(col[1:8], A)
        # bottom B-only interior (after the overlap, before the 1px border) == B
        assert np.allclose(col[16:23], B)
        # overlap band strictly between A and B and non-decreasing
        band = col[8:16]
        assert np.all(band >= A - 1e-3) and np.all(band <= B + 1e-3)
        assert np.all(np.diff(band) >= -1e-3)
        assert band[0] < band[-1]

    def test_feather_mask_edges_zero_center_one(self):
        mask = _feather_mask_numpy((32, 32), 8)
        assert mask.shape == (32, 32, 1)
        assert mask[0, 0, 0] == 0.0 and mask[-1, -1, 0] == 0.0
        assert mask[16, 16, 0] == pytest.approx(1.0)


class TestTileCountPlumbing:
    """--tiles reaches the worker spawn cmd; the wire contract is unchanged."""

    def test_tiles_forwarded_to_cmd(self, stub_env, monkeypatch):
        seen = {}
        real_popen = subprocess.Popen

        def spy(cmd, *a, **k):
            seen["cmd"] = list(cmd)
            return real_popen(cmd, *a, **k)

        monkeypatch.setattr(subprocess, "Popen", spy)
        r = FlashvsrInlineSecondaryRestorer(
            repo=stub_env["repo"],
            model_dir=stub_env["model_dir"],
            fv_python=Path(sys.executable),
            tiles=2,
            startup_timeout_s=30.0,
        )
        try:
            cmd = seen["cmd"]
            assert "--tiles" in cmd
            assert cmd[cmd.index("--tiles") + 1] == "2"
            # the stub worker ignores the extra flag, so the round-trip still holds
            frames = torch.zeros(4, 3, 256, 256)
            out = r.restore(frames, keep_start=0, keep_end=4)
            assert len(out) == 4
        finally:
            r.close()

    def test_default_tiles_one_in_cmd(self, stub_env, monkeypatch):
        seen = {}
        real_popen = subprocess.Popen

        def spy(cmd, *a, **k):
            seen["cmd"] = list(cmd)
            return real_popen(cmd, *a, **k)

        monkeypatch.setattr(subprocess, "Popen", spy)
        r = _make_restorer(stub_env)
        try:
            cmd = seen["cmd"]
            assert cmd[cmd.index("--tiles") + 1] == "1"
        finally:
            r.close()


def _spy_popen(monkeypatch):
    seen = {}
    real_popen = subprocess.Popen

    def spy(cmd, *a, **k):
        seen["cmd"] = list(cmd)
        return real_popen(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "Popen", spy)
    return seen


class TestScaleAndColorFixPlumbing:
    """--scale and the JASNA_FLASHVSR_COLOR_FIX override reach the worker cmd."""

    def test_default_scale_4_and_no_color_fix_flag(self, stub_env, monkeypatch):
        monkeypatch.delenv("JASNA_FLASHVSR_COLOR_FIX", raising=False)
        seen = _spy_popen(monkeypatch)
        r = _make_restorer(stub_env)
        try:
            cmd = seen["cmd"]
            assert cmd[cmd.index("--scale") + 1] == "4"
            # always-on color correction is the worker's default; the parent
            # passes the method only when the verification override is set
            assert "--color-fix-method" not in cmd
        finally:
            r.close()

    def test_scale_2_forwarded_and_512px_output_accepted(self, stub_env, monkeypatch):
        seen = _spy_popen(monkeypatch)
        r = FlashvsrInlineSecondaryRestorer(
            repo=stub_env["repo"], model_dir=stub_env["model_dir"],
            fv_python=Path(sys.executable), scale=2, startup_timeout_s=30.0,
        )
        try:
            cmd = seen["cmd"]
            assert cmd[cmd.index("--scale") + 1] == "2"
            out = r.restore(torch.zeros(3, 3, 256, 256), keep_start=0, keep_end=3)
            assert len(out) == 3 and all(t.shape == (3, 512, 512) for t in out)
        finally:
            r.close()

    def test_invalid_scale_rejected(self, stub_env):
        with pytest.raises(ValueError, match="scale must be 2 or 4"):
            FlashvsrInlineSecondaryRestorer(
                repo=stub_env["repo"], model_dir=stub_env["model_dir"],
                fv_python=Path(sys.executable), scale=3,
            )

    @pytest.mark.parametrize("method", ["none", "adain", "wavelet"])
    def test_color_fix_override_forwarded(self, stub_env, monkeypatch, method):
        monkeypatch.setenv("JASNA_FLASHVSR_COLOR_FIX", method)
        seen = _spy_popen(monkeypatch)
        r = _make_restorer(stub_env)
        try:
            cmd = seen["cmd"]
            assert cmd[cmd.index("--color-fix-method") + 1] == method
        finally:
            r.close()

    def test_invalid_color_fix_override_rejected(self, stub_env, monkeypatch):
        monkeypatch.setenv("JASNA_FLASHVSR_COLOR_FIX", "magic")
        with pytest.raises(ValueError, match="JASNA_FLASHVSR_COLOR_FIX"):
            _make_restorer(stub_env)


class TestColorFixPrimitives:
    """The worker's color correction (ports of FlashVSR_plus AdaIN / wavelet),
    exercised on CPU tensors."""

    def test_adain_transfers_style_stats(self):
        g = torch.Generator().manual_seed(0)
        content = torch.rand(1, 3, 32, 32, generator=g) * 0.5 + 0.1
        style = torch.rand(1, 3, 32, 32, generator=g) * 0.2 + 0.6
        out = _adain(content, style)
        for c in range(3):
            assert out[0, c].mean().item() == pytest.approx(style[0, c].mean().item(), abs=1e-3)
            assert out[0, c].std(unbiased=False).item() == pytest.approx(
                style[0, c].std(unbiased=False).item(), abs=1e-3)

    def test_wavelet_identity_when_content_equals_style(self):
        g = torch.Generator().manual_seed(1)
        x = torch.rand(1, 3, 64, 64, generator=g)
        assert torch.allclose(_wavelet_reconstruct(x, x), x, atol=1e-5)

    def test_wavelet_constant_content_takes_style_low_frequencies(self):
        # A constant content has no high frequencies, so the result is the
        # style's low-frequency band; for a constant style that is the style.
        content = torch.full((1, 3, 64, 64), 0.7)
        style = torch.full((1, 3, 64, 64), 0.2)
        assert torch.allclose(_wavelet_reconstruct(content, style), style, atol=1e-5)

    @pytest.mark.parametrize("method", ["wavelet", "adain"])
    def test_color_fix_frames_matches_constant_input(self, method):
        # (n, oh, ow, 3) float32 0..255 output vs (n, h, w, 3) [0,1] input crops:
        # a constant 180 output against a constant 0.25 input must come back as
        # 0.25*255 = 63.75 everywhere (bicubic upscale of a constant is exact).
        out = np.full((2, 32, 32, 3), 180.0, dtype=np.float32)
        lq = torch.full((2, 8, 8, 3), 0.25)
        fixed = _color_fix_frames(out, lq, method, "cpu")
        assert fixed.shape == (2, 32, 32, 3) and fixed.dtype == np.float32
        assert np.allclose(fixed, 63.75, atol=0.05)


class TestStderrSuppression:
    """--log-level error mutes the benign expandable_segments OOM warning; other
    stderr (and info/warning level) passes through."""

    def test_regex_matches_benign_only(self):
        benign = (b"[W716 07:57:26.7 CUDACachingAllocator.cpp:508] "
                  b"expandable_segments: memory mapping failed with OOM on device 0")
        assert fi._STDERR_SUPPRESS_RE.search(benign)
        assert not fi._STDERR_SUPPRESS_RE.search(b"Traceback (most recent call last):")

    def test_suppressed_at_error_level(self, stub_env, monkeypatch, capfd):
        monkeypatch.setenv("STUB_STDERR", "1")
        r = FlashvsrInlineSecondaryRestorer(
            repo=stub_env["repo"],
            model_dir=stub_env["model_dir"],
            fv_python=Path(sys.executable),
            log_level="error",
            startup_timeout_s=30.0,
        )
        r.close()  # joins the stderr pump thread, flushing forwarded lines
        err = capfd.readouterr().err
        assert "expandable_segments: memory mapping failed with OOM" not in err
        assert "STUB_MARKER" in err  # genuine stderr still forwarded

    def test_shown_at_info_level(self, stub_env, monkeypatch, capfd):
        monkeypatch.setenv("STUB_STDERR", "1")
        r = FlashvsrInlineSecondaryRestorer(
            repo=stub_env["repo"],
            model_dir=stub_env["model_dir"],
            fv_python=Path(sys.executable),
            log_level="info",
            startup_timeout_s=30.0,
        )
        r.close()
        err = capfd.readouterr().err
        assert "expandable_segments: memory mapping failed with OOM" in err


class TestUnpatchedRepoRejectedAtConstruction:
    def test_construction_fails_on_unpatched(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        (repo / "models" / "FlashVSR-v1.1").mkdir(parents=True)
        p = repo / "src" / "pipelines"
        p.mkdir(parents=True)
        (p / "flashvsr_tiny_long.py").write_text("no marker here\n")
        with pytest.raises(RuntimeError, match="multi-chunk fix"):
            FlashvsrInlineSecondaryRestorer(
                repo=repo, model_dir=repo / "models" / "FlashVSR-v1.1",
                fv_python=Path(sys.executable),
            )

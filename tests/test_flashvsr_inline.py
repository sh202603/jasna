"""Tests for the inline FlashVSR secondary restorer (no GPU, no FlashVSR checkout).

The real worker is replaced by a tiny stub that speaks the same length-prefixed
protocol and echoes a per-frame solid color, so we can exercise the parent's
wire handling (handshake, send, receive, frame-count contract, [ks:ke] slicing,
RGB channel order) on a CPU box.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
import torch

import jasna.restorer.flashvsr_inline_secondary_restorer as fi
from jasna.restorer.flashvsr_inline_secondary_restorer import (
    FlashvsrInlineSecondaryRestorer,
    _check_patched_repo,
)

# A stub "worker": ready handshake, then echo each input frame as a solid
# 1024x1024 frame of that frame's per-channel mean (preserves RGB order). It can
# be told to lie about the frame count or emit an error, to test those paths.
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
    out.write((json.dumps({"status": "ready"}) + "\\n").encode()); out.flush()
    O = 1024
    while True:
        h = rh(stdin)
        if h is None:
            break
        n, ih, iw = h["n"], h["h"], h["w"]
        crops = np.frombuffer(rx(stdin, n * ih * iw * 3), np.uint8).reshape(n, ih, iw, 3)
        if MODE == "error":
            out.write((json.dumps({"seq": h.get("seq", 0), "error": "boom"}) + "\\n").encode()); out.flush()
            continue
        rn = n - 1 if MODE == "shortcount" else n
        arr = np.empty((rn, O, O, 3), np.uint8)
        for i in range(rn):
            for c in range(3):
                arr[i, :, :, c] = int(round(crops[i, :, :, c].mean()))
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

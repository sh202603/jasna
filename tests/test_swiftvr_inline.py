"""Tests for the inline SwiftVR secondary restorer (no GPU, no SwiftVR checkout).

The real worker is replaced by a tiny stub that speaks the same length-prefixed
protocol (uint8 **BGR** on the wire, like the FlashVSR/SeedVR2 workers) and
echoes a per-frame solid color, so we can exercise the parent's wire handling
(handshake, send, receive, frame-count contract, [ks:ke] slicing, the
RGB<->BGR flips, the accel flags) on a CPU box. The worker module's CPU-safe
helpers (accel decision reporting, the GPU color fix on a CPU device) are
tested directly.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

import jasna.restorer.swiftvr_inline_secondary_restorer as si
from jasna.restorer.swiftvr_common import (
    add_swiftvr_arguments,
    check_restore_clip_api,
    default_swiftvr_model_dir,
    default_swiftvr_python,
    resolve_swiftvr_paths,
)
from jasna.restorer.swiftvr_inline_secondary_restorer import SwiftvrInlineSecondaryRestorer
# The worker module top-level is stdlib-only (numpy/torch/swiftvr are imported
# inside main()), so these helpers import cleanly without a SwiftVR checkout.
from jasna.restorer.swiftvr_inline_worker import _color_fix_frames_gpu, _color_fix_primitives

# A stub "worker": ready handshake, then echo each input frame as a solid
# (256*scale)^2 frame of that frame's per-channel mean (preserves the wire
# channel order, so a round trip through the parent's two flips must come back
# RGB). Modes: error / shortcount / assert_red_wire / wire_marker (see the
# FlashVSR stub). STUB_ACCEL_REPORT controls the handshake's accel fields.
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
    if os.environ.get("STUB_ARGV_FILE"):
        open(os.environ["STUB_ARGV_FILE"], "w").write(json.dumps(sys.argv[1:]))
    if MODE == "die_at_start":
        sys.exit(3)
    ready = {"status": "ready"}
    report = os.environ.get("STUB_ACCEL_REPORT", "full")
    if report == "full":
        ready["accel"] = ["fp8_dit", "torch_compile"]
        ready["accel_log"] = ["enabled fp8_dit, torch_compile."]
    elif report == "nofp8":
        ready["accel"] = ["torch_compile"]
        ready["accel_log"] = ["fp8_dit disabled: needs compute capability 8.9+", "enabled torch_compile."]
    elif report == "off":
        ready["accel"] = []
        ready["accel_log"] = []
    out.write((json.dumps(ready) + "\\n").encode()); out.flush()
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
        resp = {"seq": h.get("seq", 0), "n": rn, "h": O, "w": O}
        out.write((json.dumps(resp) + "\\n").encode())
        out.write(arr.tobytes()); out.flush()
    """
)


def _make_repo(root: Path, *, with_api: bool = True) -> Path:
    repo = root / "repo"
    (repo / "swiftvr").mkdir(parents=True)
    (repo / "swiftvr" / "pipeline.py").write_text(
        "class SwiftVRPipeline:\n    def restore_clip(self, frames_uint8, *, upscale=4):\n        pass\n"
        if with_api else
        "class SwiftVRPipeline:\n    def restore_video(self, a, b):\n        pass\n"
    )
    ck = repo / "checkpoints"
    (ck / "transformer").mkdir(parents=True)
    for name in ("reae.safetensors", "prompt_embedding.safetensors"):
        (ck / name).write_bytes(b"x")
    (ck / "transformer" / "config.json").write_text("{}")
    return repo


@pytest.fixture
def stub_env(tmp_path, monkeypatch):
    """A fake checkout (with restore_clip) + a stub worker script, wired into the restorer."""
    repo = _make_repo(tmp_path)
    worker = tmp_path / "stub_worker.py"
    worker.write_text(_STUB_WORKER)
    monkeypatch.setattr(si, "_resolve_worker_script", lambda: worker)
    return {"repo": repo, "worker": worker, "model_dir": repo / "checkpoints"}


def _make_restorer(stub_env, **kw):
    # sys.executable (test venv) runs the stub worker. The stub reads its mode
    # from the STUB_MODE env var (inherited by the child); tests set it via
    # monkeypatch.setenv before calling this.
    kw.setdefault("startup_timeout_s", 30.0)
    return SwiftvrInlineSecondaryRestorer(
        repo=stub_env["repo"],
        model_dir=stub_env["model_dir"],
        sv_python=Path(sys.executable),
        **kw,
    )


def _spawned_argv(stub_env, monkeypatch, tmp_path, **kw) -> list[str]:
    argv_file = tmp_path / "argv.json"
    monkeypatch.setenv("STUB_ARGV_FILE", str(argv_file))
    r = _make_restorer(stub_env, **kw)
    r.close()
    import json
    return json.loads(argv_file.read_text())


class TestCommon:
    def test_defaults(self, tmp_path):
        repo = tmp_path / "r"
        assert default_swiftvr_model_dir(repo) == repo / "checkpoints"
        py = default_swiftvr_python(repo)
        assert py.parts[-3] == ".venv" and py.name.startswith("python")

    def test_api_check_accepts_fork(self, tmp_path):
        check_restore_clip_api(_make_repo(tmp_path))

    def test_api_check_rejects_upstream(self, tmp_path):
        with pytest.raises(RuntimeError, match="restore_clip"):
            check_restore_clip_api(_make_repo(tmp_path, with_api=False))

    def test_api_check_missing_pipeline(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            check_restore_clip_api(tmp_path)

    def test_resolve_paths_defaults(self, tmp_path):
        repo = _make_repo(tmp_path)
        py = default_swiftvr_python(repo)
        py.parent.mkdir(parents=True)
        py.write_text("")
        r, p, m = resolve_swiftvr_paths(str(repo), "", "")
        assert (r, p, m) == (repo, py, repo / "checkpoints")

    def test_resolve_paths_errors(self, tmp_path):
        repo = _make_repo(tmp_path)
        with pytest.raises(ValueError, match="--swiftvr-repo is required"):
            resolve_swiftvr_paths("", "", "")
        with pytest.raises(FileNotFoundError, match="--swiftvr-repo not found"):
            resolve_swiftvr_paths(str(tmp_path / "nope"), "", "")
        with pytest.raises(FileNotFoundError, match="SwiftVR Python not found"):
            resolve_swiftvr_paths(str(repo), "", "")  # no venv in the fake checkout
        (repo / "checkpoints" / "reae.safetensors").unlink()
        with pytest.raises(FileNotFoundError, match="missing reae.safetensors"):
            resolve_swiftvr_paths(str(repo), sys.executable, "")

    def test_argument_defaults(self):
        import argparse
        p = argparse.ArgumentParser()
        add_swiftvr_arguments(p.add_argument_group("SwiftVR"))
        a = p.parse_args([])
        assert (a.swiftvr_repo, a.swiftvr_python, a.swiftvr_model_dir) == ("", "", "")
        assert a.swiftvr_scale == 4 and a.swiftvr_accel is True
        a = p.parse_args(["--no-swiftvr-accel", "--swiftvr-scale", "2"])
        assert a.swiftvr_accel is False and a.swiftvr_scale == 2
        with pytest.raises(SystemExit):
            p.parse_args(["--swiftvr-scale", "3"])


class TestWire:
    def test_class_attrs_route_sync(self):
        # must not accidentally satisfy AsyncSecondaryRestorer
        assert SwiftvrInlineSecondaryRestorer.name == "swiftvr-inline"
        assert SwiftvrInlineSecondaryRestorer.prefers_cpu_input is True
        for m in ("push_clip", "pop_completed", "has_pending", "flush_pending", "flush_all", "_to_tensors"):
            assert not hasattr(SwiftvrInlineSecondaryRestorer, m)

    def test_roundtrip_frame_count_and_rgb(self, stub_env):
        r = _make_restorer(stub_env)
        try:
            colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)]
            frames = torch.zeros(4, 3, 256, 256)
            for i, (rr, gg, bb) in enumerate(colors):
                frames[i, 0], frames[i, 1], frames[i, 2] = rr, gg, bb
            out = r.restore(frames, keep_start=0, keep_end=4)
            assert len(out) == 4
            for t in out:
                assert t.shape == (3, 1024, 1024) and t.dtype == torch.uint8
            assert out[0][0].float().mean() > 200 and out[0][1].float().mean() < 40
            assert out[1][1].float().mean() > 200 and out[1][0].float().mean() < 40
            assert out[2][2].float().mean() > 200 and out[2][0].float().mean() < 40
        finally:
            r.close()

    def test_outbound_wire_is_bgr(self, stub_env, monkeypatch):
        monkeypatch.setenv("STUB_MODE", "assert_red_wire")
        r = _make_restorer(stub_env)
        try:
            frames = torch.zeros(3, 3, 256, 256)
            frames[:, 0] = 1.0
            assert len(r.restore(frames, keep_start=0, keep_end=3)) == 3
        finally:
            r.close()

    def test_inbound_wire_is_bgr(self, stub_env, monkeypatch):
        monkeypatch.setenv("STUB_MODE", "wire_marker")
        r = _make_restorer(stub_env)
        try:
            for t in r.restore(torch.zeros(2, 3, 256, 256), keep_start=0, keep_end=2):
                assert int(t[0].min()) == 255 and int(t[1].max()) == 0 and int(t[2].max()) == 0
        finally:
            r.close()

    def test_keep_window_slice_and_empty(self, stub_env):
        r = _make_restorer(stub_env)
        try:
            frames = torch.zeros(6, 3, 256, 256)
            assert len(r.restore(frames, keep_start=2, keep_end=5)) == 3
            assert r.restore(frames, keep_start=2, keep_end=2) == []
            assert r.restore(torch.zeros(0, 3, 256, 256), keep_start=0, keep_end=0) == []
        finally:
            r.close()

    def test_scale_2_gives_512px(self, stub_env):
        r = _make_restorer(stub_env, scale=2)
        try:
            out = r.restore(torch.zeros(2, 3, 256, 256), keep_start=0, keep_end=2)
            assert out[0].shape == (3, 512, 512)
        finally:
            r.close()

    def test_cpu_float_input_unchanged(self, stub_env):
        r = _make_restorer(stub_env)
        try:
            frames = torch.full((2, 3, 256, 256), 0.5)
            r.restore(frames, keep_start=0, keep_end=2)
            assert torch.equal(frames, torch.full((2, 3, 256, 256), 0.5))
        finally:
            r.close()


class TestErrorPaths:
    def test_frame_count_mismatch_raises(self, stub_env, monkeypatch):
        monkeypatch.setenv("STUB_MODE", "shortcount")
        r = _make_restorer(stub_env)
        try:
            with pytest.raises(RuntimeError, match="frame-count mismatch"):
                r.restore(torch.zeros(4, 3, 256, 256), keep_start=0, keep_end=4)
        finally:
            r.close()

    def test_worker_error_raises(self, stub_env, monkeypatch):
        monkeypatch.setenv("STUB_MODE", "error")
        r = _make_restorer(stub_env)
        try:
            with pytest.raises(RuntimeError, match="worker error"):
                r.restore(torch.zeros(4, 3, 256, 256), keep_start=0, keep_end=4)
        finally:
            r.close()

    def test_worker_dies_at_start_hints_accel(self, stub_env, monkeypatch):
        monkeypatch.setenv("STUB_MODE", "die_at_start")
        with pytest.raises(RuntimeError, match="failed to start.*--no-swiftvr-accel"):
            _make_restorer(stub_env)
        with pytest.raises(RuntimeError) as ei:
            _make_restorer(stub_env, accel=False)
        assert "--no-swiftvr-accel" not in str(ei.value)

    def test_invalid_scale_rejected(self, stub_env):
        with pytest.raises(ValueError, match="scale must be 2 or 4"):
            _make_restorer(stub_env, scale=3)

    def test_construction_fails_without_restore_clip(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path, with_api=False)
        with pytest.raises(RuntimeError, match="restore_clip"):
            SwiftvrInlineSecondaryRestorer(
                repo=repo, model_dir=repo / "checkpoints", sv_python=Path(sys.executable),
            )


class TestWorkerCommandLine:
    def test_accel_passes_both_parts(self, stub_env, monkeypatch, tmp_path):
        argv = _spawned_argv(stub_env, monkeypatch, tmp_path)
        assert "--fp8-dit" in argv and "--torch-compile" in argv
        assert argv[argv.index("--scale") + 1] == "4"
        assert "--color-fix-method" not in argv  # worker default (wavelet)

    def test_no_accel_passes_neither(self, stub_env, monkeypatch, tmp_path):
        argv = _spawned_argv(stub_env, monkeypatch, tmp_path, accel=False)
        assert "--fp8-dit" not in argv and "--torch-compile" not in argv

    @pytest.mark.parametrize("method", ["adain", "wavelet", "none"])
    def test_color_fix_override_forwarded(self, stub_env, monkeypatch, tmp_path, method):
        monkeypatch.setenv("JASNA_SWIFTVR_COLOR_FIX", method)
        argv = _spawned_argv(stub_env, monkeypatch, tmp_path)
        assert argv[argv.index("--color-fix-method") + 1] == method

    def test_invalid_color_fix_override_rejected(self, stub_env, monkeypatch):
        monkeypatch.setenv("JASNA_SWIFTVR_COLOR_FIX", "lab")
        with pytest.raises(ValueError, match="JASNA_SWIFTVR_COLOR_FIX"):
            _make_restorer(stub_env)

    def test_env_isolation(self, stub_env, monkeypatch):
        monkeypatch.setenv("PYTHONPATH", "/should/not/leak")
        r = _make_restorer(stub_env)
        try:
            assert "PYTHONPATH" not in r._env
            if sys.platform != "win32":
                assert r._env.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"
        finally:
            r.close()


class TestHandshakeReport:
    def test_full_accel_logged(self, stub_env, caplog):
        with caplog.at_level(logging.INFO):
            r = _make_restorer(stub_env)
            r.close()
        assert "acceleration: fp8_dit, torch_compile" in caplog.text
        assert "runs its DiT in bf16" not in caplog.text

    def test_missing_fp8_warns(self, stub_env, monkeypatch, caplog):
        monkeypatch.setenv("STUB_ACCEL_REPORT", "nofp8")
        with caplog.at_level(logging.INFO):
            r = _make_restorer(stub_env)
            r.close()
        assert "fp8_dit disabled" in caplog.text
        assert "runs its DiT in bf16" in caplog.text
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    def test_no_accel_requested_no_bf16_warning(self, stub_env, monkeypatch, caplog):
        monkeypatch.setenv("STUB_ACCEL_REPORT", "off")
        with caplog.at_level(logging.INFO):
            r = _make_restorer(stub_env, accel=False)
            r.close()
        assert "acceleration: off" in caplog.text
        assert "runs its DiT in bf16" not in caplog.text


class TestStderrSuppression:
    def test_suppressed_at_error_level(self, stub_env, monkeypatch, capfd):
        monkeypatch.setenv("STUB_STDERR", "1")
        r = _make_restorer(stub_env, log_level="error")
        r.close()  # joins the stderr pump thread, flushing forwarded lines
        err = capfd.readouterr().err
        assert "expandable_segments: memory mapping failed with OOM" not in err
        assert "STUB_MARKER" in err

    def test_shown_at_info_level(self, stub_env, monkeypatch, capfd):
        monkeypatch.setenv("STUB_STDERR", "1")
        r = _make_restorer(stub_env, log_level="info")
        r.close()
        assert "expandable_segments: memory mapping failed with OOM" in capfd.readouterr().err


class TestColorFix:
    """The worker's device-resident color fix must match the FlashVSR worker's
    host-round-trip ``_color_fix_frames`` bit for bit on the same inputs."""

    @pytest.mark.parametrize("method", ["wavelet", "adain"])
    def test_matches_flashvsr_reference(self, method):
        from jasna.restorer.flashvsr_inline_worker import _color_fix_frames

        g = torch.Generator().manual_seed(0)
        out_u8 = torch.randint(0, 256, (3, 64, 64, 3), generator=g, dtype=torch.uint8)
        lq_u8 = torch.randint(0, 256, (3, 16, 16, 3), generator=g, dtype=torch.uint8)
        ours = _color_fix_frames_gpu(out_u8, lq_u8, method)
        ref = _color_fix_frames(out_u8.numpy().astype(np.float32), lq_u8.float().div(255.0), method, "cpu")
        ref_u8 = np.clip(ref, 0, 255).round().astype(np.uint8)
        assert ours.dtype == torch.uint8 and tuple(ours.shape) == (3, 64, 64, 3)
        assert np.array_equal(ours.numpy(), ref_u8)

    def test_primitives_come_from_flashvsr_worker(self):
        import jasna.restorer.flashvsr_inline_worker as fw
        prims = _color_fix_primitives()
        assert prims.__file__ == fw.__file__

    def test_constant_input_takes_reference_tone(self):
        out = torch.full((2, 32, 32, 3), 180, dtype=torch.uint8)
        lq = torch.full((2, 8, 8, 3), 64, dtype=torch.uint8)
        fixed = _color_fix_frames_gpu(out, lq, "wavelet")
        assert int(fixed.min()) >= 63 and int(fixed.max()) <= 65


class TestSessionFactoryBranch:
    def test_builds_restorer_from_config(self, tmp_path, monkeypatch):
        from jasna.session_config import SessionConfig
        import jasna.session_factory as sf

        repo = _make_repo(tmp_path)
        py = default_swiftvr_python(repo)
        py.parent.mkdir(parents=True)
        py.write_text("")
        captured = {}

        class _Fake:
            def __init__(self, **kw):
                captured.update(kw)

        monkeypatch.setattr(
            "jasna.restorer.swiftvr_inline_secondary_restorer.SwiftvrInlineSecondaryRestorer", _Fake
        )
        cfg = SessionConfig(
            device="cuda:0", fp16=True, batch_size=4, detection_model_name="rfdetr-v6",
            detection_model_path=Path("det.onnx"), detection_score_threshold=0.25,
            max_detection_gap=2, min_detection_duration=2, scene_detection=True,
            restoration_model_path=Path("restore.pth"), compile_basicvsrpp=True,
            max_clip_size=90, temporal_overlap=8, enable_crossfade=True,
            denoise_strength="none", denoise_step="after_primary",
            secondary_restoration="swiftvr-inline", tvai_ffmpeg_path="ffmpeg", tvai_model="iris-2",
            tvai_scale=4, tvai_args="", tvai_workers=2, rtx_scale=4, rtx_quality="high",
            rtx_denoise="medium", rtx_deblur="none", vr_mode="auto", codec="hevc",
            encoder_settings={}, lut_path=None, retarget_high_fps=False, disable_progress=False,
            working_dir=None, swiftvr_repo=str(repo), swiftvr_scale=2, swiftvr_accel=False,
            swiftvr_log_level="info",
        )
        assert isinstance(sf._build_secondary_restorer(cfg, "cuda:0"), _Fake)
        assert captured["repo"] == repo and captured["sv_python"] == py
        assert captured["model_dir"] == repo / "checkpoints"
        assert captured["scale"] == 2 and captured["accel"] is False
        assert captured["log_level"] == "info" and captured["device"] == "cuda:0"


class TestWorkerScript:
    def test_runs_help_without_swiftvr(self):
        # The worker's top level must not import torch/swiftvr (frozen builds
        # hand it to an external interpreter; tests import its helpers).
        script = si._resolve_worker_script()
        res = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        assert res.returncode == 0 and "--fp8-dit" in res.stdout and "--torch-compile" in res.stdout

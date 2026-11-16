"""jasna.upscale.swiftvr: the factor choice and the worker wire, against a stand-in worker
script that speaks the protocol of ``swiftvr_frame_worker.py`` (CPU, no SwiftVR needed)."""
from __future__ import annotations

import sys
import textwrap

import pytest
import torch

import jasna.upscale.swiftvr as swiftvr

_FAKE_WORKER = textwrap.dedent(
    """
    import json, sys
    import numpy as np

    args = sys.argv[1:]
    out, stdin = sys.stdout.buffer, sys.stdin.buffer
    if "--fail-start" in args:
        out.write(b'{"status":"failed"}\\n'); out.flush(); sys.exit(1)
    ready = {"status": "ready", "accel": ["fp8_dit"] if "--fp8-dit" in args else [], "accel_log": [], "args": args}
    out.write((json.dumps(ready) + "\\n").encode()); out.flush()
    while True:
        line = stdin.readline()
        if not line:
            break
        header = json.loads(line)
        n, h, w, scale = header["n"], header["h"], header["w"], header["scale"]
        frames = np.frombuffer(stdin.read(n * h * w * 3), np.uint8).reshape(n, h, w, 3)
        for index in range(n):
            if int(frames[index].max()) == 255:
                out.write((json.dumps({"seq": header["seq"], "error": "boom"}) + "\\n").encode()); out.flush()
                break
            frame = np.ascontiguousarray(frames[index].repeat(scale, axis=0).repeat(scale, axis=1))
            resp = {"seq": header["seq"], "i": index, "h": h * scale, "w": w * scale}
            out.write((json.dumps(resp) + "\\n").encode()); out.write(frame.tobytes()); out.flush()
    """
)


@pytest.fixture
def worker_script(tmp_path, monkeypatch):
    script = tmp_path / "fake_frame_worker.py"
    script.write_text(_FAKE_WORKER, encoding="utf-8")
    monkeypatch.setattr(swiftvr, "_resolve_worker_script", lambda: script)
    monkeypatch.setattr("jasna.restorer.swiftvr_common.check_restore_clip_api", lambda repo: None)
    return script


def _worker(tmp_path, **kwargs) -> swiftvr.SwiftvrWorker:
    options = dict(
        repo=tmp_path, model_dir=tmp_path, sv_python=sys.executable, device="cpu",
        warmup_size=(8, 6), warmup_scale=2, startup_timeout_s=60,
    )
    options.update(kwargs)
    return swiftvr.SwiftvrWorker(**options)


@pytest.mark.parametrize(
    "source, target, scale",
    [(1080, 2160, 2), (1080, 1440, 2), (720, 1440, 2), (720, 1800, 2), (720, 2160, 4), (480, 2160, 4), (480, 1080, 2)],
)
def test_scale_follows_the_target_height(source, target, scale):
    assert swiftvr.swiftvr_scale(source, target) == scale


def test_window_round_trip_and_reuse_across_windows(tmp_path, worker_script):
    worker = _worker(tmp_path)
    try:
        upscaler = swiftvr.SwiftvrClipUpscaler(
            worker, torch.device("cpu"), source_size=(8, 6), output_size=(16, 12), strength=1.0
        )
        for _ in range(2):
            frames = torch.randint(0, 255, (5, 3, 6, 8), dtype=torch.uint8)
            restored = list(upscaler.upscale(frames))

            assert len(restored) == 5
            expected = frames.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
            for got, want in zip(restored, expected):
                assert got.dtype == torch.uint8 and torch.equal(got, want)
                assert got.is_contiguous()  # the encoder reads rows
        upscaler.close()  # borrowed: the worker keeps running
        assert worker._proc.poll() is None
    finally:
        worker.close()
    assert worker._proc.poll() is not None


def test_output_is_resized_to_the_exact_size_and_mixed_by_strength(tmp_path, worker_script):
    worker = _worker(tmp_path)
    try:
        upscaler = swiftvr.SwiftvrClipUpscaler(
            worker, torch.device("cpu"), source_size=(8, 6), output_size=(12, 9), strength=0.5
        )
        frames = torch.full((2, 3, 6, 8), 120, dtype=torch.uint8)
        restored = list(upscaler.upscale(frames))
    finally:
        worker.close()

    assert all(frame.shape == (3, 9, 12) for frame in restored)
    assert all(int(frame.min()) == 120 and int(frame.max()) == 120 for frame in restored)


def test_command_carries_the_options(tmp_path, worker_script, monkeypatch):
    seen = {}
    real_popen = swiftvr.subprocess.Popen

    def record(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs["env"]
        return real_popen(cmd, **kwargs)

    monkeypatch.setattr(swiftvr.subprocess, "Popen", record)
    monkeypatch.setenv("PYTHONPATH", "should-not-leak")
    _worker(tmp_path, warmup_size=(1920, 1080), tile_megapixels=1.64).close()
    cmd = seen["cmd"]

    assert cmd[cmd.index("--warmup-width") + 1] == "1920" and cmd[cmd.index("--warmup-height") + 1] == "1080"
    assert cmd[cmd.index("--warmup-scale") + 1] == "2"
    assert cmd[cmd.index("--tile-megapixels") + 1] == "1.64"
    assert "--fp8-dit" in cmd and "--torch-compile" in cmd
    assert "PYTHONPATH" not in seen["env"]

    _worker(tmp_path, accel=False).close()
    assert "--fp8-dit" not in seen["cmd"] and "--torch-compile" not in seen["cmd"]


def test_color_fix_override_is_validated(tmp_path, worker_script, monkeypatch):
    monkeypatch.setenv("JASNA_SWIFTVR_COLOR_FIX", "bogus")
    with pytest.raises(ValueError, match="JASNA_SWIFTVR_COLOR_FIX"):
        _worker(tmp_path)


def test_worker_error_fails_the_window_and_keeps_the_worker(tmp_path, worker_script):
    worker = _worker(tmp_path)
    try:
        bad = torch.zeros((3, 3, 6, 8), dtype=torch.uint8)
        bad[1] = 255
        with pytest.raises(RuntimeError, match="boom"):
            list(worker.restore(bad.permute(0, 2, 3, 1).contiguous().numpy(), 2))
        good = torch.zeros((1, 6, 8, 3), dtype=torch.uint8).numpy()
        assert len(list(worker.restore(good, 2))) == 1
    finally:
        worker.close()


def test_abandoned_window_breaks_the_worker(tmp_path, worker_script):
    worker = _worker(tmp_path)
    frames = torch.zeros((4, 6, 8, 3), dtype=torch.uint8).numpy()
    window = worker.restore(frames, 2)
    next(window)
    window.close()

    with pytest.raises(RuntimeError, match="not running"):
        list(worker.restore(frames, 2))
    worker.close()
    assert worker._proc.poll() is not None


def test_failed_start_names_the_accel_switch(tmp_path, worker_script, monkeypatch):
    real_popen = swiftvr.subprocess.Popen
    monkeypatch.setattr(swiftvr.subprocess, "Popen", lambda cmd, **kwargs: real_popen([*cmd, "--fail-start"], **kwargs))

    with pytest.raises(RuntimeError, match="--no-swiftvr-accel"):
        _worker(tmp_path)


def test_missing_worker_script_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(swiftvr, "_resolve_worker_script", lambda: tmp_path / "absent.py")
    monkeypatch.setattr("jasna.restorer.swiftvr_common.check_restore_clip_api", lambda repo: None)

    with pytest.raises(FileNotFoundError):
        _worker(tmp_path)

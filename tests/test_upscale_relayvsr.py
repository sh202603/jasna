"""jasna.upscale.relayvsr: the worker wire and one video's upscaler, against a stand-in worker
script that speaks the protocol of ``relayvsr_frame_worker.py`` (CPU, no RelayVSR needed)."""
from __future__ import annotations

import sys
import textwrap

import pytest
import torch

import jasna.upscale.relayvsr as relayvsr
from jasna.restorer import relayvsr_common

# Answers a frame three frames late, the way RelayVSR waits for its next keyframe.
_FAKE_WORKER = textwrap.dedent(
    """
    import json, sys
    import numpy as np

    args = sys.argv[1:]
    out, stdin = sys.stdout.buffer, sys.stdin.buffer
    LAG = 3

    def send(message):
        out.write((json.dumps(message) + "\\n").encode()); out.flush()

    def scale_of(wan):
        return 2 if wan.endswith("wan_x2.safetensors") else 4

    if "--fail-start" in args:
        send({"status": "failed"}); sys.exit(1)
    weights = "bf16" if "--no-fp8" in args else args[args.index("--wan-weights") + 1]
    send({"status": "ready", "scale": scale_of(args[args.index("--wan-checkpoint") + 1]), "wan_weights": weights})
    while True:
        line = stdin.readline()
        if not line:
            break
        header = json.loads(line)
        h, w, scale = header["h"], header["w"], scale_of(header["wan"])
        send({"status": "begin", "scale": scale, "wan_weights": weights})
        held, count = [], 0

        def emit():
            global count
            frame = held.pop(0)
            frame = np.ascontiguousarray(frame.repeat(scale, axis=0).repeat(scale, axis=1))
            out.write((json.dumps({"i": count, "h": h * scale, "w": w * scale}) + "\\n").encode())
            out.write(frame.tobytes()); out.flush()
            count += 1

        while True:
            op = json.loads(stdin.readline())["op"]
            if op == "end":
                while held:
                    emit()
                send({"done": True, "frames": count, "kv_cache": "gpu"})
                break
            frame = np.frombuffer(stdin.read(h * w * 3), np.uint8).reshape(h, w, 3)
            if int(frame.max()) == 255:
                send({"error": "boom"})
                break
            held.append(frame)
            if len(held) > LAG:
                emit()
            send({"more": True})
    """
)


@pytest.fixture
def weights_dir(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in ("wan.safetensors", "flash.safetensors", "wan_x2.safetensors", "flash_x2.safetensors"):
        (weights / name).touch()
    return weights


@pytest.fixture
def worker_script(tmp_path, monkeypatch):
    script = tmp_path / "fake_frame_worker.py"
    script.write_text(_FAKE_WORKER, encoding="utf-8")
    monkeypatch.setattr(relayvsr, "_resolve_worker_script", lambda: script)
    return script


def _worker(weights_dir, **kwargs) -> relayvsr.RelayvsrWorker:
    options = dict(rv_python=sys.executable, weights_dir=weights_dir, device="cpu", scale=2, startup_timeout_s=60)
    options.update(kwargs)
    return relayvsr.RelayvsrWorker(**options)


def _upscaler(worker, source=(8, 6), output=(16, 12), strength=1.0) -> relayvsr.RelayvsrStreamUpscaler:
    return relayvsr.RelayvsrStreamUpscaler(
        worker, torch.device("cpu"), source_size=source, output_size=output, strength=strength
    )


def _run(upscaler, frames):
    """``(frames per push, all output frames)``."""
    counts, restored = [], []
    for frame in frames:
        emitted = list(upscaler.push(frame))
        counts.append(len(emitted))
        restored += emitted
    restored += list(upscaler.finish())
    return counts, restored


def test_late_frames_round_trip_and_the_worker_serves_the_next_video(weights_dir, worker_script):
    worker = _worker(weights_dir)
    try:
        for scale, output in ((2, (16, 12)), (4, (32, 24)), (2, (16, 12))):
            upscaler = _upscaler(worker, output=output)
            frames = torch.randint(0, 255, (7, 3, 6, 8), dtype=torch.uint8)
            counts, restored = _run(upscaler, frames)
            upscaler.close()  # borrowed: the worker keeps running

            assert counts == [0, 0, 0, 1, 1, 1, 1] and len(restored) == 7
            expected = frames.repeat_interleave(scale, dim=2).repeat_interleave(scale, dim=3)
            for got, want in zip(restored, expected):
                assert got.dtype == torch.uint8 and torch.equal(got, want)
                assert got.is_contiguous()  # the encoder reads rows
            assert worker._proc.poll() is None
    finally:
        worker.close()
    assert worker._proc.poll() is not None


def test_output_is_resized_to_the_exact_size_and_mixed_by_strength(weights_dir, worker_script):
    worker = _worker(weights_dir)
    try:
        frames = torch.stack([torch.full((3, 6, 8), value, dtype=torch.uint8) for value in (40, 120, 200, 80, 160)])
        _, restored = _run(_upscaler(worker, output=(12, 9), strength=0.5), frames)
    finally:
        worker.close()

    # Each output is mixed with its own source frame, although it comes out three frames late.
    assert [tuple(frame.shape) for frame in restored] == [(3, 9, 12)] * 5
    assert [(int(frame.min()), int(frame.max())) for frame in restored] == [(v, v) for v in (40, 120, 200, 80, 160)]


def test_video_without_frames_never_reaches_the_worker(weights_dir, worker_script):
    worker = _worker(weights_dir)
    try:
        upscaler = _upscaler(worker)
        assert list(upscaler.finish()) == []
        upscaler.close()
        assert _run(_upscaler(worker), torch.zeros((1, 3, 6, 8), dtype=torch.uint8))[0] == [0]
    finally:
        worker.close()


def test_command_carries_the_options(weights_dir, worker_script, monkeypatch):
    seen = {}
    real_popen = relayvsr.subprocess.Popen

    def record(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs["env"]
        return real_popen(cmd, **kwargs)

    monkeypatch.setattr(relayvsr.subprocess, "Popen", record)
    monkeypatch.setenv("PYTHONPATH", "should-not-leak")
    _worker(weights_dir, scale=4, window_size=32, wan_weights="bf16", kv_cache="cpu", kv_cache_dtype="int8").close()
    cmd = seen["cmd"]

    assert cmd[cmd.index("--wan-checkpoint") + 1] == str(weights_dir / "wan.safetensors")
    assert cmd[cmd.index("--flash-checkpoint") + 1] == str(weights_dir / "flash.safetensors")
    assert cmd[cmd.index("--window-size") + 1] == "32"
    assert cmd[cmd.index("--wan-weights") + 1] == "bf16"
    assert cmd[cmd.index("--kv-cache") + 1] == "cpu" and cmd[cmd.index("--kv-cache-dtype") + 1] == "int8"
    assert "PYTHONPATH" not in seen["env"]

    _worker(weights_dir).close()
    assert seen["cmd"][seen["cmd"].index("--wan-checkpoint") + 1] == str(weights_dir / "wan_x2.safetensors")
    assert seen["cmd"][seen["cmd"].index("--wan-weights") + 1] == "fp8"


def test_fp8_fallback_is_reported(weights_dir, worker_script, monkeypatch, caplog):
    real_popen = relayvsr.subprocess.Popen
    monkeypatch.setattr(relayvsr.subprocess, "Popen", lambda cmd, **kwargs: real_popen([*cmd, "--no-fp8"], **kwargs))

    with caplog.at_level("WARNING", logger=relayvsr.__name__):
        _worker(weights_dir).close()
    assert "run as bf16, not fp8" in caplog.text


def test_worker_error_ends_the_video_and_keeps_the_worker(weights_dir, worker_script):
    worker = _worker(weights_dir)
    try:
        upscaler = _upscaler(worker)
        assert list(upscaler.push(torch.zeros((3, 6, 8), dtype=torch.uint8))) == []
        with pytest.raises(RuntimeError, match="boom"):
            list(upscaler.push(torch.full((3, 6, 8), 255, dtype=torch.uint8)))
        upscaler.close()

        counts, restored = _run(_upscaler(worker), torch.zeros((2, 3, 6, 8), dtype=torch.uint8))
        assert counts == [0, 0] and len(restored) == 2
    finally:
        worker.close()


def test_abandoned_video_breaks_the_worker(weights_dir, worker_script):
    worker = _worker(weights_dir)
    upscaler = _upscaler(worker)
    list(upscaler.push(torch.zeros((3, 6, 8), dtype=torch.uint8)))
    upscaler.close()

    with pytest.raises(RuntimeError, match="not running"):
        list(_upscaler(worker).push(torch.zeros((3, 6, 8), dtype=torch.uint8)))
    worker.close()
    assert worker._proc.poll() is not None


def test_missing_pair_is_reported_with_its_source(weights_dir, worker_script):
    (weights_dir / "flash.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="kopper/RelayVSR"):
        _worker(weights_dir, scale=4)

    worker = _worker(weights_dir)
    try:
        with pytest.raises(FileNotFoundError, match="4x checkpoint not found.*flash.safetensors"):
            list(_upscaler(worker, output=(32, 24)).push(torch.zeros((3, 6, 8), dtype=torch.uint8)))
    finally:
        worker.close()


def test_failed_start_is_reported(weights_dir, worker_script, monkeypatch):
    real_popen = relayvsr.subprocess.Popen
    monkeypatch.setattr(relayvsr.subprocess, "Popen", lambda cmd, **kwargs: real_popen([*cmd, "--fail-start"], **kwargs))

    with pytest.raises(RuntimeError, match="failed to start"):
        _worker(weights_dir)


def test_missing_worker_script_is_reported(weights_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(relayvsr, "_resolve_worker_script", lambda: tmp_path / "absent.py")
    with pytest.raises(FileNotFoundError):
        _worker(weights_dir)


# --------------------------------------------------------------------------- #
# relayvsr_common
# --------------------------------------------------------------------------- #

def _checkout(tmp_path, *, session="def build_stream(): ...\n"):
    repo = tmp_path / "RelayVSR"
    (repo / "refvsr_inference" / "runtime").mkdir(parents=True)
    (repo / "refvsr_inference" / "runtime" / "session.py").write_text(session, encoding="utf-8")
    python = relayvsr_common.default_relayvsr_python(repo)
    python.parent.mkdir(parents=True)
    python.touch()
    (repo / "weights").mkdir()
    return repo


def test_paths_default_to_the_checkout(tmp_path):
    repo = _checkout(tmp_path)
    assert relayvsr_common.resolve_relayvsr_paths(str(repo), "", "") == (
        repo, relayvsr_common.default_relayvsr_python(repo), repo / "weights"
    )


def test_paths_are_checked_one_by_one(tmp_path):
    with pytest.raises(ValueError, match="--relayvsr-repo is required"):
        relayvsr_common.resolve_relayvsr_paths("", "", "")
    with pytest.raises(FileNotFoundError, match="--relayvsr-repo not found"):
        relayvsr_common.resolve_relayvsr_paths(str(tmp_path / "absent"), "", "")

    repo = _checkout(tmp_path)
    with pytest.raises(FileNotFoundError, match="RelayVSR Python not found"):
        relayvsr_common.resolve_relayvsr_paths(str(repo), str(tmp_path / "no-python"), "")
    with pytest.raises(FileNotFoundError, match="--relayvsr-weights-dir not found"):
        relayvsr_common.resolve_relayvsr_paths(str(repo), "", str(tmp_path / "no-weights"))


def test_checkout_without_build_stream_names_the_fork(tmp_path):
    repo = _checkout(tmp_path, session="# upstream\n")
    with pytest.raises(RuntimeError, match="sh202603/RelayVSR"):
        relayvsr_common.resolve_relayvsr_paths(str(repo), "", "")

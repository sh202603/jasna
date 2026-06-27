"""Throughput regression checks for the video backends (opt-in, GPU-only).

Run explicitly with ``-m perf``; skipped by default and whenever its
prerequisites are missing (no CUDA GPU, no torchcodec, or no sample clip).

Two judgments (see ``docs/TORCHCODEC_BACKEND_*.md``):

  (1) torchcodec must not be slower than native by more than ``THRESHOLD``.
  (2) the most important one — the native path must not regress vs a baseline
      captured BEFORE the backend layer was introduced. Set the baseline FPS via
      ``JASNA_NATIVE_DECODE_FPS_BASELINE`` (measured once on the same clip + GPU);
      when unset, judgment (2) is skipped with a note.

Timing is wall-clock (``time.perf_counter``) with ``torch.cuda.synchronize()``
bracketing the measured region so async GPU work is included.

Sample clip: ``JASNA_PERF_SAMPLE`` (path to a short video). Without it the test
skips — there is no bundled fixture because decode throughput is hardware- and
content-dependent.
"""
from __future__ import annotations

import os
import statistics
import time

import pytest

pytestmark = pytest.mark.perf

THRESHOLD = 0.8          # torchcodec_fps >= native_fps * THRESHOLD  (judgment 1)
BASELINE_MARGIN = 0.05   # native_fps >= baseline * (1 - margin)     (judgment 2)
WARMUP_BATCHES = 2
REPEATS = 3


def _require(cond, reason):
    if not cond:
        pytest.skip(reason)


def _sample_clip() -> str:
    path = os.environ.get("JASNA_PERF_SAMPLE", "")
    _require(path and os.path.exists(path), "set JASNA_PERF_SAMPLE to a short video clip")
    return path


def _decode_fps(make_reader, *, clip, batch_size, device, metadata) -> float:
    """Median frames/sec over REPEATS full decode passes (wall-clock)."""
    import torch

    samples = []
    for rep in range(REPEATS):
        with make_reader() as reader:
            it = reader.frames()
            # warmup: a few batches not counted
            for _ in range(WARMUP_BATCHES):
                next(it, None)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            frames = 0
            for _batch, pts in it:
                frames += len(pts)
            torch.cuda.synchronize(device)
            dt = time.perf_counter() - t0
        if frames and dt > 0:
            samples.append(frames / dt)
    _require(samples, "no frames decoded from sample clip")
    return statistics.median(samples)


def test_decode_throughput_no_regression():
    import torch

    _require(torch.cuda.is_available(), "CUDA GPU required")
    from jasna.media import backend as backend_mod
    _require(backend_mod.torchcodec_available(), "torchcodec not installed")

    from jasna.media import get_video_meta_data
    from jasna.media.backend import VideoBackend, make_video_reader

    clip = _sample_clip()
    device = torch.device("cuda:0")
    # Establish the primary CUDA context before the native reader's cuStreamCreate
    # (vali needs a current context; pytest doesn't init it the way a script does).
    torch.cuda.set_device(device)
    _ = torch.zeros(1, device=device)
    batch_size = int(os.environ.get("JASNA_PERF_BATCH", "8"))
    metadata = get_video_meta_data(clip)

    def _reader(backend):
        return lambda: make_video_reader(
            file=clip, batch_size=batch_size, device=device, metadata=metadata, backend=backend,
        )

    native_fps = _decode_fps(_reader(VideoBackend.NATIVE), clip=clip, batch_size=batch_size, device=device, metadata=metadata)
    tc_fps = _decode_fps(_reader(VideoBackend.TORCHCODEC), clip=clip, batch_size=batch_size, device=device, metadata=metadata)

    print(f"\n[perf] decode FPS: native={native_fps:.1f} torchcodec={tc_fps:.1f} "
          f"ratio={tc_fps / native_fps:.2f}")

    # Judgment (1): torchcodec vs native.
    assert tc_fps >= native_fps * THRESHOLD, (
        f"torchcodec decode {tc_fps:.1f} fps regressed vs native {native_fps:.1f} fps "
        f"(threshold {THRESHOLD})"
    )

    # Judgment (2): native vs pre-change baseline (the important one).
    baseline_env = os.environ.get("JASNA_NATIVE_DECODE_FPS_BASELINE", "")
    if baseline_env:
        baseline = float(baseline_env)
        print(f"[perf] native baseline={baseline:.1f} (min allowed={baseline * (1 - BASELINE_MARGIN):.1f})")
        assert native_fps >= baseline * (1 - BASELINE_MARGIN), (
            f"native decode {native_fps:.1f} fps regressed vs baseline {baseline:.1f} fps "
            f"(margin {BASELINE_MARGIN}) — the backend layer must add zero hot-path overhead"
        )
    else:
        print("[perf] JASNA_NATIVE_DECODE_FPS_BASELINE unset — judgment (2) skipped; "
              "capture it once on this clip+GPU before the backend layer to guard the native path")

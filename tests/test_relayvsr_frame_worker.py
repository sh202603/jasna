"""relayvsr_frame_worker.py itself, run as the subprocess it is: jasna's ``RelayvsrWorker`` talks
to the real script, which imports a stand-in ``refvsr_inference`` whose stream cuts its windows
the way RelayVSR's ``frames()`` does (CPU, no RelayVSR needed)."""
from __future__ import annotations

import os
import sys
import textwrap

import pytest
import torch

import jasna.upscale.relayvsr as relayvsr
from jasna.restorer import relayvsr_frame_worker

_FAKE_STREAMING = textwrap.dedent(
    """
    from dataclasses import dataclass
    import torch

    @dataclass(frozen=True)
    class InputFrame:
        index: int
        rgb: torch.Tensor

    @dataclass(frozen=True)
    class OutputFrame:
        index: int
        rgb: torch.Tensor
    """
)

_FAKE_SESSION = textwrap.dedent(
    """
    import os, sys
    from itertools import islice
    import torch
    from .streaming import OutputFrame

    class Stream:
        def __init__(self, scale, window_size, wan_weights, kv_cache):
            self.scale, self.window_size, self.wan_weights = scale, window_size, wan_weights
            self.resolved_wan_kv_cache = None
            self._kv_cache = kv_cache

        def reset(self):
            print("fake-reset", file=sys.stderr, flush=True)

        def frames(self, source):
            iterator = iter(source)
            first = next(iterator, None)
            if first is None:
                raise ValueError("Cannot infer an empty stream.")
            self.resolved_wan_kv_cache = "cpu" if self._kv_cache == "auto" else self._kv_cache
            shape = tuple(first.rgb.shape)
            start, first_window = first, True
            while True:
                window = [start, *islice(iterator, self.window_size - 1)]
                if len(window) == 1 and not first_window:
                    return
                for offset, packet in enumerate(window):
                    if packet.index != start.index + offset or tuple(packet.rgb.shape) != shape:
                        raise ValueError("Streaming frames must be consecutive with constant dimensions.")
                    if first_window or offset:
                        if int(packet.rgb.max()) == 255:
                            raise RuntimeError("boom")
                        rgb = packet.rgb.repeat_interleave(self.scale, 1).repeat_interleave(self.scale, 2)
                        yield OutputFrame(packet.index, rgb[None].float().div(127.5).sub(1.0))
                first_window = False
                start = window[-1]
                if len(window) < self.window_size:
                    return

    def build_stream(wan, flash, *, device, dtype, window_size, seed, kv_cache, kv_cache_dtype, wan_weights):
        assert (dtype, seed) == ("bf16", 42)
        print("this print must not reach the wire")
        scale = 2 if os.path.basename(wan) == "wan_x2.safetensors" else 4
        assert os.path.basename(flash) == ("flash_x2.safetensors" if scale == 2 else "flash.safetensors")
        return Stream(scale, window_size, wan_weights, kv_cache)

    def to_uint8(rgb):
        return rgb[0].add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()
    """
)


@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    root = tmp_path_factory.mktemp("relayvsr")
    package = root / "site" / "refvsr_inference" / "runtime"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    (package / "streaming.py").write_text(_FAKE_STREAMING, encoding="utf-8")
    (package / "session.py").write_text(_FAKE_SESSION, encoding="utf-8")
    weights = root / "weights"
    weights.mkdir()
    for name in ("wan.safetensors", "flash.safetensors", "wan_x2.safetensors", "flash_x2.safetensors"):
        (weights / name).touch()

    env = dict(os.environ, PYTHONPATH=str(root / "site"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("jasna.restorer.worker_env.external_worker_env", lambda: dict(env))
        worker = relayvsr.RelayvsrWorker(
            rv_python=sys.executable, weights_dir=weights, device="cpu", scale=2, window_size=16,
            wan_weights="bf16", startup_timeout_s=120,
        )
    yield worker
    worker.close()
    assert worker._proc.poll() == 0  # stdin closed: a clean exit


def _video(worker, frames, scale=2):
    """``(frames answered per pushed frame, frames answered at the end, all output frames)``."""
    height, width = frames.shape[2:]
    upscaler = relayvsr.RelayvsrStreamUpscaler(
        worker, torch.device("cpu"), source_size=(width, height),
        output_size=(width * scale, height * scale), strength=1.0,
    )
    counts, restored = [], []
    for frame in frames:
        emitted = list(upscaler.push(frame))
        counts.append(len(emitted))
        restored += emitted
    tail = list(upscaler.finish())
    upscaler.close()
    return counts, len(tail), restored + tail


def _frames(count, height=6, width=8):
    return torch.randint(0, 255, (count, 3, height, width), dtype=torch.uint8)


@pytest.mark.parametrize(
    "count, answered, tail",
    [
        (1, {}, 1),
        (15, {}, 15),
        (16, {15: 16}, 0),
        (17, {15: 16}, 1),
        (31, {15: 16, 30: 15}, 0),
        (40, {15: 16, 30: 15}, 9),
    ],
)
def test_every_frame_comes_back_in_order_at_relayvsrs_pace(worker, count, answered, tail):
    frames = _frames(count)
    counts, at_end, restored = _video(worker, frames)

    assert counts == [answered.get(index, 0) for index in range(count)] and at_end == tail
    assert len(restored) == count
    expected = frames.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    assert all(torch.equal(got, want) for got, want in zip(restored, expected))


def test_factor_and_frame_size_change_between_videos(worker):
    frames = _frames(18, height=10, width=14)
    _, _, restored = _video(worker, frames, scale=4)
    assert len(restored) == 18 and tuple(restored[0].shape) == (3, 40, 56)
    assert torch.equal(restored[-1], frames[-1].repeat_interleave(4, dim=1).repeat_interleave(4, dim=2))

    _, _, restored = _video(worker, _frames(3), scale=2)
    assert len(restored) == 3 and tuple(restored[0].shape) == (3, 12, 16)


def test_failure_inside_a_video_leaves_the_worker_for_the_next(worker):
    frames = _frames(20)
    frames[17] = 255  # fails in the second window, after 16 frames went out
    with pytest.raises(RuntimeError, match="worker error: RuntimeError: boom"):
        _video(worker, frames)

    counts, at_end, restored = _video(worker, _frames(5))
    assert (sum(counts), at_end, len(restored)) == (0, 5, 5)


def test_defaults_are_relayvsrs():
    assert relayvsr_frame_worker.DEFAULT_WINDOW_SIZE == 16
    assert relayvsr_frame_worker._SEED == 42

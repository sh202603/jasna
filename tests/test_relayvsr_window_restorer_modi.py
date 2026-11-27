"""``--secondary-restoration relayvsr`` (``jasna/restorer/relayvsr_window_restorer.py``): RelayVSR as
the x2 restorer of LTX windows. One window is one video to the worker of ``jasna-postprocess
--upscale relayvsr``; the CLI and the session accept it with an LTX model without auto split only.

modi 独自。factories.py の ``cli_args`` / ``cli_environment`` / ``session_config`` に依存する。
上流のファイルには追記しない。
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from factories import cli_args, cli_environment, session_config
from jasna.restorer.relayvsr_window_restorer import RelayvsrWindowRestorer

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


class _FakeWorker:
    """Answers late like RelayVSR (nothing until ``delay`` frames are in) and reuses the array it
    yields, like the real worker. A restored frame is its input doubled, plus 1."""

    instances: list["_FakeWorker"] = []

    def __init__(self, *, delay: int = 3, drop_last: bool = False, **options):
        self.options, self.delay, self.drop_last = options, delay, drop_last
        self.begun: list[tuple[int, int, int]] = []
        self.pending: list[np.ndarray] = []
        self.abandoned = self.closed = 0
        self._out = None
        _FakeWorker.instances.append(self)

    def begin(self, width, height, scale):
        self.begun.append((width, height, scale))
        self._out = np.empty((height * scale, width * scale, 3), np.uint8)

    def _emit(self, frames):
        for frame in frames:
            self._out[...] = frame.repeat(2, axis=0).repeat(2, axis=1) + 1
            yield self._out

    def push(self, frame):
        assert frame.dtype == np.uint8 and frame.shape[-1] == 3
        self.pending.append(frame.copy())
        if len(self.pending) >= self.delay:
            ready, self.pending = self.pending, []
            yield from self._emit(ready)

    def end(self):
        ready, self.pending = self.pending, []
        yield from self._emit(ready[:-1] if self.drop_last else ready)

    def abandon(self):
        self.abandoned += 1

    def close(self):
        self.closed += 1


@pytest.fixture
def worker_cls():
    _FakeWorker.instances.clear()
    with patch("jasna.upscale.relayvsr.RelayvsrWorker", _FakeWorker):
        yield _FakeWorker


def _restorer(**options):
    return RelayvsrWindowRestorer(rv_python=Path("py"), weights_dir=Path("w"), device="cuda:0", **options)


def _clip(count: int, height: int = 4, width: int = 6) -> torch.Tensor:
    values = torch.arange(count, dtype=torch.float32).mul(10).div(255.0)
    return values.reshape(count, 1, 1, 1).expand(count, 3, height, width).contiguous()


def test_a_window_is_one_video_of_the_worker(worker_cls):
    restorer = _restorer(window_size=32, wan_weights="bf16", kv_cache="cpu", kv_cache_dtype="int8")
    restored = restorer.restore(_clip(8), keep_start=0, keep_end=8)
    worker = worker_cls.instances[0]
    assert worker.options == dict(
        rv_python=Path("py"), weights_dir=Path("w"), device="cuda:0", scale=2, window_size=32, wan_weights="bf16",
        kv_cache="cpu", kv_cache_dtype="int8", verbose=False,
    )
    assert worker.begun == [(6, 4, 2)]  # width, height, the 2x pair
    assert len(restored) == 8 and all(frame.shape == (3, 8, 12) and frame.dtype == torch.uint8 for frame in restored)
    # Each frame is its own copy of the array the worker reuses, in order.
    assert [int(frame[0, 0, 0]) for frame in restored] == [10 * index + 1 for index in range(8)]
    restorer.restore(_clip(5), keep_start=0, keep_end=5)
    assert worker.begun == [(6, 4, 2)] * 2 and len(worker_cls.instances) == 1  # the worker stays for the next window


def test_only_the_kept_frames_come_back_but_every_frame_is_restored(worker_cls):
    restored = _restorer().restore(_clip(8), keep_start=2, keep_end=6)
    assert [int(frame[0, 0, 0]) for frame in restored] == [21, 31, 41, 51]
    assert _restorer().restore(_clip(0), keep_start=0, keep_end=0) == []
    assert _restorer().restore(_clip(4), keep_start=3, keep_end=3) == []


def test_a_missing_frame_is_an_error_and_a_failed_window_abandons_the_worker(worker_cls):
    with patch("jasna.upscale.relayvsr.RelayvsrWorker", lambda **options: _FakeWorker(drop_last=True, **options)):
        with pytest.raises(RuntimeError, match="sent 5, got 4"):
            _restorer().restore(_clip(5), keep_start=0, keep_end=5)

    restorer = _restorer()
    worker = worker_cls.instances[-1]
    with patch.object(worker, "push", side_effect=RuntimeError("worker error: out of memory")):
        with pytest.raises(RuntimeError, match="out of memory"):
            restorer.restore(_clip(3), keep_start=0, keep_end=3)
    assert worker.abandoned == 1
    restorer.close()
    assert worker.closed == 1


def _run_main(tmp_path, extra_args, *, model=("--restoration-model-name", "ltx-8step", "--ltx-trial")):
    argv = cli_args(tmp_path, *model, *extra_args)
    with (
        cli_environment() as pipeline_cls,
        patch("jasna.accelerator.is_nvidia_device", return_value=True),
        patch("jasna.ltx.transformer.require_kernels"),
        patch("jasna.session_factory._relayvsr_paths", return_value=(tmp_path,) * 3),
        patch.object(sys, "argv", ["jasna", *argv]),
    ):
        from jasna.main import main

        main()
    return pipeline_cls


class TestCli:
    def test_relayvsr_reaches_the_pipeline_with_its_options(self, tmp_path):
        config = _run_main(
            tmp_path,
            ["--secondary-restoration", "relayvsr", "--relayvsr-repo", "repo", "--relayvsr-window-size", "32",
             "--relayvsr-wan-weights", "bf16", "--relayvsr-kv-cache", "cpu", "--relayvsr-kv-cache-dtype", "int8"],
        ).call_args.kwargs["config"]
        assert config.restoration_model_name == "ltx" and config.secondary_restoration == "relayvsr"
        assert (config.relayvsr_repo, config.relayvsr_window_size, config.relayvsr_wan_weights) == ("repo", 32, "bf16")
        assert (config.relayvsr_kv_cache, config.relayvsr_kv_cache_dtype) == ("cpu", "int8")

    def test_relayvsr_needs_an_ltx_model(self, tmp_path):
        with pytest.raises(ValueError, match="works on LTX windows only"):
            _run_main(tmp_path, ["--secondary-restoration", "relayvsr"], model=())

    def test_relayvsr_is_refused_with_auto_split(self, tmp_path):
        with (
            patch("jasna.session_factory.ltx_auto_split", return_value=None),
            pytest.raises(ValueError, match="works on LTX windows only"),
        ):
            _run_main(tmp_path, ["--secondary-restoration", "relayvsr", "--ltx-auto-split", "12"])

    def test_the_window_size_must_hold_a_keyframe_interval(self, tmp_path):
        with pytest.raises(ValueError, match="--relayvsr-window-size must be at least 2"):
            _run_main(tmp_path, ["--secondary-restoration", "relayvsr", "--relayvsr-window-size", "1"])


class TestSession:
    def test_the_restorer_gets_the_resolved_paths_and_every_option(self, tmp_path):
        from jasna import session_factory

        config = session_config(
            restoration_model_name="ltx", secondary_restoration="relayvsr", relayvsr_repo="repo",
            relayvsr_window_size=32, relayvsr_wan_weights="bf16", relayvsr_kv_cache="cpu",
            relayvsr_kv_cache_dtype="int8",
        )
        paths = (tmp_path / "repo", tmp_path / "py", tmp_path / "weights")
        with (
            patch.object(session_factory, "_relayvsr_paths", return_value=paths),
            patch("jasna.restorer.relayvsr_window_restorer.RelayvsrWindowRestorer") as restorer_cls,
        ):
            restorer = session_factory._build_secondary_restorer(
                session_factory.ltx_window_restorer_config(config), torch.device("cuda:0"), size_from_frames=True
            )
        assert restorer is restorer_cls.return_value
        assert restorer_cls.call_args.kwargs == dict(
            rv_python=paths[1], weights_dir=paths[2], device="cuda:0", window_size=32, wan_weights="bf16",
            kv_cache="cpu", kv_cache_dtype="int8",
        )
        assert session_factory.restores_ltx_windows(config)

    def test_the_paths_are_checked_with_the_2x_pair_before_anything_loads(self, tmp_path):
        from jasna import session_factory

        config = session_config(restoration_model_name="ltx", secondary_restoration="relayvsr")
        with pytest.raises(ValueError, match="--relayvsr-repo is required for --secondary-restoration relayvsr"):
            session_factory._validate_window_restorer(config)

        repo = tmp_path / "RelayVSR"
        (repo / "refvsr_inference" / "runtime").mkdir(parents=True)
        (repo / "refvsr_inference" / "runtime" / "session.py").write_text("def build_stream(): ...\n")
        (repo / "weights").mkdir()
        python = repo / "python"
        python.touch()
        config = session_config(
            restoration_model_name="ltx", secondary_restoration="relayvsr", relayvsr_repo=str(repo),
            relayvsr_python=str(python),
        )
        with pytest.raises(FileNotFoundError, match="RelayVSR 2x checkpoint not found"):
            session_factory._validate_window_restorer(config)
        for name in ("wan_x2.safetensors", "flash_x2.safetensors"):
            (repo / "weights" / name).touch()
        session_factory._validate_window_restorer(config)

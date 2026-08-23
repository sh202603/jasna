from __future__ import annotations

import queue
import threading
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from av.video.reformatter import ColorRange as AvColorRange
from av.video.reformatter import Colorspace as AvColorspace
from PIL import Image

from jasna.gui import ab_compare_worker
from jasna.gui.ab_compare_worker import (
    ABClipResult,
    ABCompareWorker,
    ABCompileFinished,
    ABFailed,
    ABFrameResult,
    ABSideConfig,
    ABStatus,
    _Cancel,
    _CompileRequest,
    _Request,
    _Stop,
)
from jasna.gui.models import AppSettings
from jasna.media import VideoMetadata


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        video_file="video.mp4",
        num_frames=300,
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        codec_name="h264",
        duration=10.0,
        video_width=1920,
        video_height=1080,
        time_base=Fraction(1, 90000),
        start_pts=0,
        color_space=AvColorspace.ITU709,
        color_range=AvColorRange.MPEG,
        is_10bit=False,
    )


def _side(checkpoint: str = "model.pth") -> ABSideConfig:
    return ABSideConfig(
        detection_model="rfdetr-v6",
        detection_score_threshold=0.35,
        restoration_model_path=Path(checkpoint),
    )


def _request(worker: ABCompareWorker, *, view: str = "restored") -> _Request:
    return _Request(
        center_seconds=2.0,
        view=view,
        side_a=_side(),
        side_b=_side(),
        base_settings=AppSettings(),
        projection="auto",
        generation=worker._generation + 1,
    )


def _drain(worker: ABCompareWorker) -> list:
    events = []
    try:
        while True:
            events.append(worker.events.get_nowait())
    except queue.Empty:
        pass
    return events


def test_request_coalesces_and_increments_generation() -> None:
    worker = ABCompareWorker("unused.mp4", _metadata())

    first = worker.request(1.0, "restored", _side(), _side(), AppSettings(), "auto")
    second = worker.request(2.0, "detection", _side(), _side(), AppSettings(), "auto")

    command = worker._commands.get_nowait()
    assert command.center_seconds == 2.0
    assert command.view == "detection"
    assert command.generation == second == first + 1
    with pytest.raises(queue.Empty):
        worker._commands.get_nowait()
    worker.close()


def test_request_rejects_unknown_view() -> None:
    worker = ABCompareWorker("unused.mp4", _metadata())

    with pytest.raises(ValueError):
        worker.request(1.0, "bogus", _side(), _side(), AppSettings(), "auto")
    worker.close()


def test_process_runs_legs_strictly_sequentially_a_then_b() -> None:
    worker = ABCompareWorker("unused.mp4", _metadata())
    order = []

    def run_side(side, config, request):
        order.append((side, "start"))
        order.append((side, "end"))

    worker._run_side = run_side

    worker._process(_request(worker))

    assert order == [("a", "start"), ("a", "end"), ("b", "start"), ("b", "end")]


def test_new_request_between_legs_aborts_leg_b() -> None:
    worker = ABCompareWorker("unused.mp4", _metadata())
    calls = []

    def run_side(side, config, request):
        calls.append(side)
        if side == "a":
            worker.request(5.0, "restored", _side(), _side(), AppSettings(), "auto")

    worker._run_side = run_side

    worker._process(_request(worker))

    assert calls == ["a"]


def test_close_between_legs_aborts_leg_b() -> None:
    worker = ABCompareWorker("unused.mp4", _metadata())
    calls = []

    def run_side(side, config, request):
        calls.append(side)
        worker._closed.set()

    worker._run_side = run_side

    worker._process(_request(worker))

    assert calls == ["a"]


def test_checkpoint_without_ema_is_refused_and_other_side_still_runs(monkeypatch) -> None:
    import jasna.restorer.checkpoint_info as checkpoint_info

    monkeypatch.setattr(checkpoint_info, "checkpoint_has_ema_weights", lambda _path: False)
    worker = ABCompareWorker("unused.mp4", _metadata())
    worker._ensure_detector = MagicMock()

    worker._process(_request(worker, view="restored"))

    events = _drain(worker)
    failed = [e for e in events if isinstance(e, ABFailed)]
    assert [e.side for e in failed] == ["a", "b"]
    assert all(e.code == "checkpoint_missing_ema" for e in failed)
    assert not worker._ensure_detector.called


def test_detection_view_skips_ema_check(monkeypatch) -> None:
    import jasna.restorer.checkpoint_info as checkpoint_info

    def _boom(_path):
        raise AssertionError("EMA check must not run for the detection view")

    monkeypatch.setattr(checkpoint_info, "checkpoint_has_ema_weights", _boom)
    worker = ABCompareWorker("unused.mp4", _metadata())
    worker._ensure_detector = MagicMock()
    worker._run_detection_view = MagicMock()

    worker._process(_request(worker, view="detection"))

    assert worker._run_detection_view.call_count == 2
    assert [c.args[0] for c in worker._run_detection_view.call_args_list] == ["a", "b"]


def test_cuda_oom_releases_both_slots_and_retries_once() -> None:
    worker = ABCompareWorker("unused.mp4", _metadata())
    worker._slots = {"a": MagicMock(), "b": MagicMock()}
    calls = []
    outcomes = [torch.cuda.OutOfMemoryError("out of memory"), None, None]

    def run_side(side, config, request):
        calls.append(side)
        outcome = outcomes.pop(0)
        if outcome is not None:
            raise outcome

    worker._run_side = run_side

    worker._process(_request(worker))

    assert calls == ["a", "a", "b"]
    assert worker._slots["a"].release.called
    assert worker._slots["b"].release.called
    assert not [e for e in _drain(worker) if isinstance(e, ABFailed)]


def test_failed_leg_emits_error_and_continues_with_other_side() -> None:
    worker = ABCompareWorker("unused.mp4", _metadata())
    calls = []

    def run_side(side, config, request):
        calls.append(side)
        if side == "a":
            raise ValueError("boom")

    worker._run_side = run_side

    worker._process(_request(worker))

    assert calls == ["a", "b"]
    failed = [e for e in _drain(worker) if isinstance(e, ABFailed)]
    assert len(failed) == 1
    assert failed[0].side == "a"
    assert failed[0].code == "error"
    assert "boom" in failed[0].message


def test_restoration_view_maps_pass_result_to_frame_event(monkeypatch) -> None:
    worker = ABCompareWorker("unused.mp4", _metadata())
    image = Image.new("RGB", (4, 4))
    monkeypatch.setattr(
        ab_compare_worker,
        "run_restoration_pass",
        lambda **kwargs: SimpleNamespace(
            superseded=False, clip_frames=None, frame_image=image, frame_seconds=1.5
        ),
    )
    request = _request(worker, view="restored")

    worker._run_restoration_view("a", MagicMock(), request, AppSettings())

    events = _drain(worker)
    assert isinstance(events[0], ABStatus)
    frame = next(e for e in events if isinstance(e, ABFrameResult))
    assert frame.side == "a"
    assert frame.seconds == 1.5
    assert frame.image is image
    assert frame.generation == request.generation


def test_restoration_view_superseded_pass_emits_no_result(monkeypatch) -> None:
    worker = ABCompareWorker("unused.mp4", _metadata())
    monkeypatch.setattr(
        ab_compare_worker,
        "run_restoration_pass",
        lambda **kwargs: SimpleNamespace(
            superseded=True, clip_frames=None, frame_image=None, frame_seconds=None
        ),
    )

    worker._run_restoration_view("a", MagicMock(), _request(worker), AppSettings())

    events = _drain(worker)
    assert all(isinstance(e, ABStatus) for e in events)


def test_clip_view_emits_clip_result_even_when_empty(monkeypatch) -> None:
    worker = ABCompareWorker("unused.mp4", _metadata())
    monkeypatch.setattr(
        ab_compare_worker,
        "run_restoration_pass",
        lambda **kwargs: SimpleNamespace(
            superseded=False, clip_frames=None, frame_image=None, frame_seconds=None
        ),
    )

    worker._run_restoration_view("b", MagicMock(), _request(worker, view="clip"), AppSettings())

    clip = next(e for e in _drain(worker) if isinstance(e, ABClipResult))
    assert clip.side == "b"
    assert clip.frames == ()


def test_restoration_pass_receives_side_settings_and_clip_playback(monkeypatch) -> None:
    worker = ABCompareWorker("unused.mp4", _metadata())
    captured = {}

    def fake_pass(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            superseded=False, clip_frames=(), frame_image=None, frame_seconds=None
        )

    monkeypatch.setattr(ab_compare_worker, "run_restoration_pass", fake_pass)
    slot = MagicMock()
    settings = AppSettings(detection_model="rfdetr-v5", detection_score_threshold=0.25)

    worker._run_restoration_view("a", slot, _request(worker, view="clip"), settings)

    assert captured["playback"] is True
    assert captured["settings"] is settings
    assert captured["session"] is slot.session
    assert captured["detection_model"] is slot.detector
    assert captured["max_size"] == ab_compare_worker.AB_PREVIEW_MAX_SIZE


def test_close_cancels_active_pass_and_queues_stop() -> None:
    worker = ABCompareWorker("unused.mp4", _metadata())
    active = threading.Event()
    worker._active_cancel = active

    worker.close()

    assert active.is_set()
    assert isinstance(worker._commands.get_nowait(), _Stop)


def test_cancel_stops_pass_without_closing_worker() -> None:
    worker = ABCompareWorker("unused.mp4", _metadata())
    active = threading.Event()
    worker._active_cancel = active

    worker.cancel()

    assert active.is_set()
    assert not worker._closed.is_set()
    assert isinstance(worker._commands.get_nowait(), _Cancel)
    worker.close()


def test_request_compile_coalesces_and_increments_generation() -> None:
    worker = ABCompareWorker("unused.mp4", _metadata())

    first = worker.request(1.0, "restored", _side(), _side(), AppSettings(), "auto")
    second = worker.request_compile("b", Path("model.pth"), AppSettings())

    command = worker._commands.get_nowait()
    assert isinstance(command, _CompileRequest)
    assert command.side == "b"
    assert command.checkpoint == Path("model.pth")
    assert command.generation == second == first + 1
    with pytest.raises(queue.Empty):
        worker._commands.get_nowait()
    worker.close()


def test_compile_streams_status_and_reports_success(monkeypatch) -> None:
    import jasna.engine_compiler as engine_compiler
    import jasna.engine_paths as engine_paths

    captured = {}

    def fake_ensure(request, log_callback=None):
        captured["request"] = request
        log_callback("Compiling TensorRT engines...")

    monkeypatch.setattr(engine_compiler, "ensure_engines_compiled", fake_ensure)
    monkeypatch.setattr(
        engine_paths, "all_basicvsrpp_sub_engines_exist", lambda _path, _fp16: True
    )
    worker = ABCompareWorker("unused.mp4", _metadata())

    worker._compile(
        _CompileRequest(
            side="a",
            checkpoint=Path("model.pth"),
            base_settings=AppSettings(),
            generation=3,
        )
    )

    events = _drain(worker)
    assert isinstance(events[0], ABStatus) and events[0].message == "compiling"
    assert any(
        isinstance(e, ABStatus) and "Compiling" in e.message for e in events
    )
    finished = next(e for e in events if isinstance(e, ABCompileFinished))
    assert finished.side == "a"
    assert finished.ok is True
    assert finished.generation == 3
    assert captured["request"].basicvsrpp is True
    assert captured["request"].basicvsrpp_model_path == "model.pth"
    assert captured["request"].fp16 is True


def test_compile_failure_reports_not_ok(monkeypatch) -> None:
    import jasna.engine_compiler as engine_compiler

    def fake_ensure(request, log_callback=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(engine_compiler, "ensure_engines_compiled", fake_ensure)
    worker = ABCompareWorker("unused.mp4", _metadata())

    worker._compile(
        _CompileRequest(
            side="b",
            checkpoint=Path("model.pth"),
            base_settings=AppSettings(),
            generation=4,
        )
    )

    finished = next(e for e in _drain(worker) if isinstance(e, ABCompileFinished))
    assert finished.ok is False
    assert "boom" in finished.message


def test_worker_loop_dispatches_compile_requests() -> None:
    worker = ABCompareWorker("unused.mp4", _metadata())
    done = threading.Event()
    worker._compile = lambda request: done.set()
    worker.start()

    worker.request_compile("a", Path("model.pth"), AppSettings())

    assert done.wait(timeout=5)
    worker.close()
    worker.join(timeout=5)


def test_worker_reports_completed_teardown() -> None:
    stopped = threading.Event()
    worker = ABCompareWorker("unused.mp4", _metadata(), on_stopped=stopped.set)
    worker.start()

    worker.close()
    worker.join(timeout=5)

    assert stopped.is_set()


def _seedvr2_side() -> ABSideConfig:
    return ABSideConfig(
        detection_model="rfdetr-v6",
        detection_score_threshold=0.35,
        restoration_model_path=Path("lada_seedvr2_lora_v2.pt"),
        restoration_model_name="seedvr2",
        seedvr2_repo="/opt/seedvr2_videoupscaler",
    )


def test_seedvr2_side_skips_ema_check(monkeypatch) -> None:
    import jasna.restorer.checkpoint_info as checkpoint_info

    def _boom(_path):
        raise AssertionError("EMA check must not run for a seedvr2 side")

    monkeypatch.setattr(checkpoint_info, "checkpoint_has_ema_weights", _boom)
    worker = ABCompareWorker("unused.mp4", _metadata())
    worker._ensure_detector = MagicMock()
    worker._ensure_session = MagicMock()
    worker._run_restoration_view = MagicMock()

    request = _Request(
        center_seconds=2.0,
        view="restored",
        side_a=_seedvr2_side(),
        side_b=_seedvr2_side(),
        base_settings=AppSettings(),
        projection="auto",
        generation=worker._generation + 1,
    )
    worker._process(request)

    assert worker._run_restoration_view.call_count == 2


def test_seedvr2_session_key_and_build_kwargs(monkeypatch) -> None:
    built = []

    def fake_build(settings, *, disable_basicvsrpp_tensorrt, log, restoration_model_path,
                   restoration_model_name="basicvsrpp", seedvr2_repo=""):
        built.append((restoration_model_name, seedvr2_repo, str(restoration_model_path)))
        session = MagicMock()
        session.restoration_pipeline.restorer.tensorrt_active = False
        return session

    monkeypatch.setattr(ab_compare_worker, "build_video_session", fake_build)
    monkeypatch.setattr(ab_compare_worker, "release_session_memory", lambda _device: None)
    monkeypatch.setattr(
        "jasna.engine_paths.all_basicvsrpp_sub_engines_exist", lambda *_a: False
    )
    worker = ABCompareWorker("unused.mp4", _metadata())
    slot = ab_compare_worker._SessionSlot()

    worker._ensure_session("a", slot, AppSettings(), _seedvr2_side(), 1)
    assert built == [("seedvr2", "/opt/seedvr2_videoupscaler", "lada_seedvr2_lora_v2.pt")]
    seedvr2_key = slot.session_key
    assert "seedvr2" in seedvr2_key

    # Switching the same slot to a BasicVSR++ checkpoint must change the key
    # and rebuild the session.
    worker._ensure_session("a", slot, AppSettings(), _side("lada_seedvr2_lora_v2.pt"), 1)
    assert slot.session_key != seedvr2_key
    assert built[-1][0] == "basicvsrpp"
    worker.close()

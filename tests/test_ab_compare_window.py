from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from jasna.gui.ab_compare import ABCompareWindow, _PaneState
from jasna.gui.ab_compare_worker import (
    ABClipResult,
    ABCompileFinished,
    ABDetectionResult,
    ABFailed,
    ABFrameResult,
    ABSessionInfo,
    ABSideConfig,
    ABStatus,
)
from jasna.gui.locales import t
from jasna.gui.restoration_preview import RestoredClipFrame


def _window(*, view: str = "restored", checkpoint: Path | None = Path("model.pth")):
    window = object.__new__(ABCompareWindow)
    window._closed = threading.Event()
    window._view = view
    window._generation = 5
    window._pending_sides = {"a", "b"}
    window._panes = {
        "a": _PaneState("rfdetr-v6", 0.35, checkpoint),
        "b": _PaneState("rfdetr-v6", 0.35, checkpoint),
    }
    window._pane_widgets = {"a": MagicMock(), "b": MagicMock()}
    window._status_label = MagicMock()
    window._refresh_pane = MagicMock()
    window._refresh_badge = MagicMock()
    window._refresh_pane_images = MagicMock()
    window._stop_clip_playback = MagicMock()
    window._left_eye = False
    window._clip_position = None
    window._fps = 30.0
    window._duration = 60.0
    window._center_seconds = 5.0
    window._initial_detection_model = "rfdetr-v6"
    window._initial_threshold = 0.35
    window._projection = "auto"
    window._worker = MagicMock()
    window._get_settings = MagicMock()
    window._time_label = MagicMock()
    window._compiling_side = None
    window._set_controls_enabled = MagicMock()
    return window


def test_events_from_stale_generation_are_ignored() -> None:
    window = _window()
    image = Image.new("RGB", (4, 4))

    window._handle_worker_event(ABFrameResult("a", 1.0, image, generation=4))

    assert window._panes["a"].frame_image is None
    assert not window._refresh_pane.called


def test_frame_result_routes_to_its_side_only() -> None:
    window = _window()
    image = Image.new("RGB", (4, 4))

    window._handle_worker_event(ABFrameResult("a", 1.0, image, generation=5))

    assert window._panes["a"].frame_image is image
    assert window._panes["a"].frame_seconds == 1.0
    assert window._panes["b"].frame_image is None
    window._refresh_pane.assert_called_once_with("a")
    assert window._pending_sides == {"b"}


def test_all_sides_finished_clears_status_line() -> None:
    window = _window()
    image = Image.new("RGB", (4, 4))

    window._handle_worker_event(ABFrameResult("a", 1.0, image, generation=5))
    window._handle_worker_event(ABFrameResult("b", 1.0, image, generation=5))

    assert window._pending_sides == set()
    window._status_label.configure.assert_called_with(text="")


def test_status_event_updates_pane_and_status_line() -> None:
    window = _window()

    window._handle_worker_event(ABStatus("b", "loading_models", generation=5))

    assert window._panes["b"].status == t("segments_restore_loading_models")
    text = window._status_label.configure.call_args.kwargs["text"]
    assert text.startswith("B:")


def test_session_info_confirms_badge() -> None:
    window = _window()

    window._handle_worker_event(
        ABSessionInfo("b", generation=5, tensorrt=False, checkpoint_name="x.pth")
    )

    assert window._panes["b"].tensorrt is False
    window._refresh_badge.assert_called_once_with("b")


def test_failed_ema_checkpoint_shows_refusal_message() -> None:
    window = _window()

    window._handle_worker_event(
        ABFailed("a", "checkpoint_missing_ema", "old.pth", generation=5)
    )

    assert window._panes["a"].error == t("ab_checkpoint_missing_ema", name="old.pth")
    assert window._panes["a"].status is None


def test_failed_generic_error_uses_restore_failed_message() -> None:
    window = _window()

    window._handle_worker_event(ABFailed("b", "error", "boom", generation=5))

    assert window._panes["b"].error == t("segments_restore_failed", message="boom")


def test_badge_maps_confirmed_and_provisional_states(monkeypatch) -> None:
    # Pin the standard flavor: in a TensorRT-RTX venv the badge reads TRT-RTX.
    from jasna import engine_paths

    monkeypatch.setattr(engine_paths, "_trt_flavor_cache", "standard")
    window = _window()
    window._refresh_badge = ABCompareWindow._refresh_badge.__get__(window)
    pane = window._panes["a"]
    widgets = window._pane_widgets["a"]

    pane.tensorrt = None
    pane.provisional_tensorrt = True
    window._refresh_badge("a")
    assert widgets.badge.configure.call_args.kwargs["text"] == t("ab_badge_trt")

    pane.tensorrt = False
    window._refresh_badge("a")
    assert widgets.badge.configure.call_args.kwargs["text"] == t("ab_badge_pytorch")
    widgets.badge_tooltip.set_text.assert_called_with(t("ab_badge_pytorch_hint"))

    pane.tensorrt = None
    pane.provisional_tensorrt = None
    window._refresh_badge("a")
    assert widgets.badge.configure.call_args.kwargs["text"] == ""


def test_model_change_marks_stale_and_applies_recommended_threshold() -> None:
    window = _window()
    window._mark_stale = MagicMock()

    window._on_detection_model_selected("a", "rfdetr-v6-large")

    assert window._panes["a"].detection_model == "rfdetr-v6-large"
    assert window._panes["a"].threshold == pytest.approx(0.40)
    window._mark_stale.assert_called_once_with("a")

    window._mark_stale.reset_mock()
    window._on_detection_model_selected("a", "rfdetr-v6")
    assert window._panes["a"].threshold == pytest.approx(0.35)
    window._mark_stale.assert_called_once_with("a")


def test_checkpoint_change_resets_confirmed_badge_and_marks_stale() -> None:
    window = _window()
    window._checkpoints = {"other.pth": Path("other.pth")}
    window._provisional_tensorrt = MagicMock(return_value=False)
    window._mark_stale = MagicMock()
    window._panes["b"].tensorrt = True

    window._on_checkpoint_selected("b", "other.pth")

    assert window._panes["b"].checkpoint == Path("other.pth")
    assert window._panes["b"].tensorrt is None
    assert window._panes["b"].provisional_tensorrt is False
    window._mark_stale.assert_called_once_with("b")
    window._refresh_badge.assert_called_once_with("b")


def test_set_center_marks_both_sides_stale_and_stops_playback() -> None:
    window = _window()
    window._mark_stale = MagicMock()

    window._set_center(6.0)

    assert window._center_seconds == 6.0
    window._stop_clip_playback.assert_called_once_with()
    assert [c.args[0] for c in window._mark_stale.call_args_list] == ["a", "b"]

    window._mark_stale.reset_mock()
    window._set_center(6.0)
    assert not window._mark_stale.called


def test_pane_source_clip_picks_frame_nearest_to_playback_position() -> None:
    window = _window(view="clip")
    first = Image.new("RGB", (4, 4))
    second = Image.new("RGB", (4, 4))
    window._panes["a"].clip_frames = (
        RestoredClipFrame(0.0, first),
        RestoredClipFrame(0.5, second),
    )

    assert window._pane_source("a") is first
    window._clip_position = 0.4
    assert window._pane_source("a") is second


def test_pane_source_detection_overlays_mask() -> None:
    window = _window(view="detection")
    original = Image.new("RGB", (160, 90), "black")
    mask = torch.ones((90, 160), dtype=torch.uint8)
    window._panes["a"].detection = (0.0, original, mask, 0.9)

    source = window._pane_source("a")

    assert source is not original
    assert source.getpixel((80, 45))[0] > 0
    assert window._pane_source("b") is None


def test_clip_clock_falls_back_to_side_b() -> None:
    window = _window(view="clip")
    frames = (RestoredClipFrame(0.0, Image.new("RGB", (4, 4))),)
    window._panes["b"].clip_frames = frames

    assert window._clock_frames() is frames


def test_run_without_checkpoint_is_blocked_for_restoration_views() -> None:
    window = _window(checkpoint=None)

    window._run_clicked()

    assert not window._worker.request.called
    assert window._status_label.configure.call_args.kwargs["text"] == t("ab_no_checkpoints")


def test_run_requests_current_selection_and_clears_stale() -> None:
    window = _window()
    window._panes["a"].stale = True
    window._panes["b"].detection_model = "rfdetr-v5"
    window._panes["b"].threshold = 0.25
    window._panes["b"].checkpoint = Path("b.pth")
    settings = object()
    window._get_settings.return_value = settings
    window._worker.request.return_value = 6

    window._run_clicked()

    assert window._generation == 6
    args = window._worker.request.call_args.args
    assert args[0] == window._center_seconds
    assert args[1] == "restored"
    assert args[2] == ABSideConfig("rfdetr-v6", 0.35, Path("model.pth"))
    assert args[3] == ABSideConfig("rfdetr-v5", 0.25, Path("b.pth"))
    assert args[4] is settings
    assert window._pending_sides == {"a", "b"}
    assert not window._panes["a"].stale
    assert window._panes["a"].status == t("segments_restore_restoring")
    window._refresh_pane_images.assert_called_once_with()


def test_threshold_change_marks_stale_and_updates_label() -> None:
    window = _window()
    window._mark_stale = MagicMock()

    window._on_side_threshold("a", 0.6)

    assert window._panes["a"].threshold == pytest.approx(0.6)
    label_text = window._pane_widgets["a"].threshold_label.configure.call_args.kwargs["text"]
    assert label_text == "0.60"
    window._mark_stale.assert_called_once_with("a")

    window._mark_stale.reset_mock()
    window._on_side_threshold("a", 0.6)
    assert not window._mark_stale.called


def test_threshold_is_clamped_to_scan_floor() -> None:
    from jasna.gui.mosaic_scan import SCAN_SCORE_FLOOR

    window = _window()
    window._mark_stale = MagicMock()

    window._on_side_threshold("b", 0.0)

    assert window._panes["b"].threshold == pytest.approx(SCAN_SCORE_FLOOR)


def test_model_change_syncs_threshold_slider() -> None:
    window = _window()
    window._mark_stale = MagicMock()

    window._on_detection_model_selected("a", "rfdetr-v6-large")

    widgets = window._pane_widgets["a"]
    widgets.threshold_slider.set.assert_called_once_with(pytest.approx(0.40))
    assert widgets.threshold_label.configure.call_args.kwargs["text"] == "0.40"


def test_compile_clicked_locks_controls_and_requests_compilation() -> None:
    window = _window()
    window._worker.request_compile.return_value = 9

    window._compile_clicked("a")

    assert window._compiling_side == "a"
    assert window._generation == 9
    args = window._worker.request_compile.call_args.args
    assert args[0] == "a"
    assert args[1] == Path("model.pth")
    window._set_controls_enabled.assert_called_once_with(False)
    assert window._panes["a"].status == t("ab_compiling")

    window._worker.request_compile.reset_mock()
    window._compile_clicked("b")
    assert not window._worker.request_compile.called


def test_compile_clicked_ignores_missing_checkpoint() -> None:
    window = _window(checkpoint=None)

    window._compile_clicked("a")

    assert window._compiling_side is None
    assert not window._worker.request_compile.called


def test_compile_finished_ok_unlocks_and_refreshes_both_sides() -> None:
    window = _window()
    window._compiling_side = "a"
    window._mark_stale = MagicMock()
    window._provisional_tensorrt = MagicMock(return_value=True)
    for pane in window._panes.values():
        pane.tensorrt = False
        pane.provisional_tensorrt = False

    window._handle_worker_event(
        ABCompileFinished("a", ok=True, message="", generation=5)
    )

    assert window._compiling_side is None
    window._set_controls_enabled.assert_called_once_with(True)
    assert window._status_label.configure.call_args.kwargs["text"] == t("ab_compile_done")
    for side, pane in window._panes.items():
        assert pane.tensorrt is None
        assert pane.provisional_tensorrt is True
    assert {c.args[0] for c in window._mark_stale.call_args_list} == {"a", "b"}


def test_compile_finished_only_touches_sides_with_same_checkpoint() -> None:
    window = _window()
    window._compiling_side = "a"
    window._mark_stale = MagicMock()
    window._provisional_tensorrt = MagicMock(return_value=True)
    window._panes["b"].checkpoint = Path("other.pth")
    window._panes["b"].tensorrt = True

    window._handle_worker_event(
        ABCompileFinished("a", ok=True, message="", generation=5)
    )

    assert window._panes["a"].tensorrt is None
    assert window._panes["b"].tensorrt is True
    assert [c.args[0] for c in window._mark_stale.call_args_list] == ["a"]


def test_compile_finished_failure_shows_message_without_stale() -> None:
    window = _window()
    window._compiling_side = "a"
    window._mark_stale = MagicMock()
    window._provisional_tensorrt = MagicMock(return_value=False)

    window._handle_worker_event(
        ABCompileFinished("a", ok=False, message="boom", generation=5)
    )

    assert window._compiling_side is None
    assert window._status_label.configure.call_args.kwargs["text"] == t(
        "ab_compile_failed", message="boom"
    )
    assert not window._mark_stale.called


def test_run_is_blocked_while_compiling() -> None:
    window = _window()
    window._compiling_side = "a"

    window._run_clicked()

    assert not window._worker.request.called


def test_view_change_stops_playback_and_rerenders() -> None:
    window = _window()
    window._view_by_label = {t("ab_view_clip"): "clip"}
    window._update_play_button_visibility = MagicMock()
    window._clip_position = 1.0

    window._on_view_selected(t("ab_view_clip"))

    assert window._view == "clip"
    assert window._clip_position is None
    window._stop_clip_playback.assert_called_once_with()
    window._update_play_button_visibility.assert_called_once_with()
    window._refresh_pane_images.assert_called_once_with()

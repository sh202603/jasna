"""modi-only segment editor tests: the A/B model comparison button and window.

modi 独自。上流の tests/test_segment_editor.py の ``build_editor`` fixture（と conftest の ``tk_root``）に
依存する。上流のファイルには追記しない。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from jasna.gui.locales import t
from jasna.gui.segment_editor import SegmentEditor
from jasna.gui.segment_editor_state import SegmentEditorState
from jasna.segments import SegmentRestoration
from test_segment_editor import build_editor  # noqa: F401  (fixture)

_STANDARD = SegmentRestoration("basicvsrpp", None)


def _bare_editor() -> SegmentEditor:
    editor = object.__new__(SegmentEditor)
    editor._state = SegmentEditorState(duration=10, fps=30, segments=(), default_restoration=_STANDARD)
    editor._ab_window = None
    return editor


def test_ab_compare_button_builds_and_locks_during_scan(build_editor) -> None:  # noqa: F811
    editor = build_editor()
    assert editor._ab_btn.cget("text") == t("segments_ab_compare")
    assert editor._ab_btn in editor._lockable_widgets()
    assert editor._ab_window is None

    editor._on_scan_lock_changed(True)
    assert editor._ab_btn.cget("state") == "disabled"
    editor._on_scan_lock_changed(False)
    assert editor._ab_btn.cget("state") == "normal"


def test_open_ab_compare_is_blocked_while_gpu_is_busy() -> None:
    editor = _bare_editor()
    editor._scan_panel = SimpleNamespace(active=False)
    editor._is_gpu_busy = MagicMock(return_value=True)
    editor._deactivate_restoration_preview = MagicMock()

    editor._open_ab_compare()

    assert editor._ab_window is None
    editor._deactivate_restoration_preview.assert_not_called()


def test_open_ab_compare_stops_restore_preview_and_hands_over_context(monkeypatch) -> None:
    import jasna.gui.ab_compare as ab_compare

    opened = MagicMock()
    monkeypatch.setattr(ab_compare, "ABCompareWindow", opened)
    editor = _bare_editor()
    editor._scan_panel = SimpleNamespace(active=False, detection_model="rfdetr-v6", threshold=0.4)
    editor._is_gpu_busy = MagicMock(return_value=False)
    editor._restore_active = True
    editor._deactivate_restoration_preview = MagicMock()
    editor._restoration_worker = MagicMock()
    restoration_worker = editor._restoration_worker
    editor._set_playing = MagicMock()
    editor._job = MagicMock(path=Path("video.mp4"))
    editor._metadata = MagicMock()
    editor._current_video_settings = MagicMock()
    editor._current = 3.5
    editor._vr_projection = "auto"
    editor._set_preview_gpu_busy = MagicMock()

    editor._open_ab_compare()

    editor._deactivate_restoration_preview.assert_called_once_with()
    restoration_worker.close.assert_called_once_with()
    assert editor._restoration_worker is None
    kwargs = opened.call_args.kwargs
    assert kwargs["center_seconds"] == 3.5
    assert kwargs["initial_detection_model"] == "rfdetr-v6"
    assert kwargs["initial_threshold"] == 0.4
    assert kwargs["projection"] == "auto"
    assert editor._ab_window is opened.return_value


def test_finish_close_closes_open_ab_window(build_editor) -> None:  # noqa: F811
    editor = build_editor()
    ab_window = MagicMock()
    editor._ab_window = ab_window

    editor._finish_close()

    ab_window.close.assert_called_once_with()
    assert editor._ab_window is None

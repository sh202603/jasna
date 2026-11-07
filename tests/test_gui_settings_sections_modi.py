"""modi-only settings panel tests: the modi widgets (RTX strength / high bitrate, FP8 recon,
frame generation, video backend) are on the real panel and round-trip their values.

modi 独自。上流の tests/test_gui_settings_sections.py の ``panel_factory`` fixture と conftest の
``tk_root`` / ``nvidia_vendor`` fixture に依存する。上流のファイルには追記しない。
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from jasna.gui.models import AppSettings
from test_gui_settings_sections import _apply, panel_factory  # noqa: F401  (fixture)

pytestmark = pytest.mark.usefixtures("nvidia_vendor")

_MODI_FIELDS = (
    "rtx_strength",
    "rtx_highbitrate",
    "fp8_recon",
    "frame_generation_backend",
    "frame_generation_model_path",
    "frame_generation_rtx_mode",
    "video_backend",
)


def test_modi_widgets_are_collected(panel_factory) -> None:  # noqa: F811
    panel = panel_factory()

    collected = {key for section in panel._sections for key in section.collect()}

    assert set(_MODI_FIELDS) <= collected


def test_modi_settings_round_trip_through_panel(panel_factory) -> None:  # noqa: F811
    panel = panel_factory()
    settings = replace(
        AppSettings(),
        rtx_strength=0.5,
        rtx_highbitrate=True,
        fp8_recon=True,
        frame_generation="4x",
        frame_generation_backend="rife",
        frame_generation_model_path="/opt/rife/rife.pth",
        frame_generation_rtx_mode="high",
        video_backend="auto",
    )

    _apply(panel, settings)

    got = panel.get_settings()
    assert {name: getattr(got, name) for name in _MODI_FIELDS} == {
        name: getattr(settings, name) for name in _MODI_FIELDS
    }


def test_frame_generation_4x_round_trips(panel_factory) -> None:  # noqa: F811
    panel = panel_factory()
    _apply(panel, replace(AppSettings(), frame_generation="4x"))
    assert panel.get_settings().frame_generation == "4x"

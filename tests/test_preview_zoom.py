from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import customtkinter as ctk
import numpy as np
import pytest
from PIL import Image

from jasna.gui.preview_zoom import (
    ZOOM_MAX,
    ZOOM_MIN,
    ZOOM_STEP,
    ZoomPanController,
    apply_mask_overlay,
    clamp_center,
    crop_normalized,
    fit_to_label,
    image_geometry,
)


def test_clamp_center_limits_to_visible_half() -> None:
    assert clamp_center((0.0, 0.0), 2.0) == (0.25, 0.25)
    assert clamp_center((1.0, 1.0), 2.0) == (0.75, 0.75)
    assert clamp_center((0.5, 0.5), 4.0) == (0.5, 0.5)


def test_crop_normalized_at_default_zoom_returns_source() -> None:
    source = Image.new("RGB", (400, 200))

    assert crop_normalized(source, 1.0, (0.3, 0.7)) is source


def test_crop_normalized_crops_and_clamps_to_source() -> None:
    source = Image.new("RGB", (400, 200), "black")
    source.putpixel((0, 0), (255, 0, 0))

    cropped = crop_normalized(source, 2.0, (0.0, 0.0))

    assert cropped.size == (200, 100)
    assert cropped.getpixel((0, 0)) == (255, 0, 0)


def test_image_geometry_letterboxes_with_frame_padding() -> None:
    left, top, width, height = image_geometry(500, 300, (400, 200))

    assert width == pytest.approx(484)
    assert height == pytest.approx(242)
    assert left == pytest.approx((500 - 484) / 2)
    assert top == pytest.approx((300 - 242) / 2)


@pytest.mark.parametrize("scaling", [1.0, 1.5])
def test_fit_to_label_compensates_widget_scaling(scaling: float) -> None:
    label = MagicMock()
    label.winfo_width.return_value = 916
    label.winfo_height.return_value = 556
    source = Image.new("RGB", (1920, 1080))

    result = fit_to_label(label, source, scaling)

    rendered = (
        round(result._size[0] * scaling),
        round(result._size[1] * scaling),
    )
    assert rendered[0] <= 900 and rendered[1] <= 540
    assert result._light_image.size == (900, 506)


def test_apply_mask_overlay_returns_source_when_mask_is_empty() -> None:
    image = Image.new("RGB", (160, 90), "black")
    mask = np.zeros((90, 160), dtype=np.uint8)

    assert apply_mask_overlay(image, mask) is image


def test_apply_mask_overlay_blends_red_over_detections() -> None:
    image = Image.new("RGB", (160, 90), "black")
    mask = np.ones((90, 160), dtype=np.uint8)

    overlaid = apply_mask_overlay(image, mask)

    assert overlaid is not image
    assert overlaid.getpixel((80, 45))[0] > 0


def test_apply_mask_overlay_crops_left_eye_before_resizing() -> None:
    image = Image.new("RGB", (80, 90), "black")
    mask = np.zeros((90, 160), dtype=np.uint8)
    mask[:, :80] = 1

    overlaid = apply_mask_overlay(image, mask, left_eye=True)

    assert overlaid.getpixel((0, 45))[0] > 0
    assert overlaid.getpixel((79, 45))[0] > 0


def test_controller_set_zoom_clamps_and_notifies() -> None:
    changes = []
    controller = ZoomPanController(on_change=lambda: changes.append(1))

    controller.set_zoom(1.5)

    assert controller.zoom == 1.5
    assert changes == [1]

    controller.set_zoom(ZOOM_MAX + 5)
    assert controller.zoom == ZOOM_MAX

    changes.clear()
    controller.set_zoom(ZOOM_MAX + 10)
    assert changes == []


def test_controller_crop_resets_center_at_default_zoom() -> None:
    controller = ZoomPanController(on_change=lambda: None)
    controller.center = (0.9, 0.9)
    source = Image.new("RGB", (400, 200))

    assert controller.crop(source) is source
    assert controller.center == (0.5, 0.5)


def test_controller_crop_clamps_center_when_zoomed() -> None:
    controller = ZoomPanController(on_change=lambda: None)
    controller.zoom = 2.0
    controller.center = (0.0, 0.0)
    source = Image.new("RGB", (400, 200))

    cropped = controller.crop(source)

    assert cropped.size == (200, 100)
    assert controller.center == pytest.approx((0.25, 0.25))


def test_controller_pan_drag_moves_shared_center() -> None:
    changes = []
    controller = ZoomPanController(on_change=lambda: changes.append(1))
    controller.zoom = 2.0
    controller.center = (0.5, 0.5)
    controller._pan_anchor = (100, 100)
    label = MagicMock()
    label.winfo_width.return_value = 500
    label.winfo_height.return_value = 300
    source = Image.new("RGB", (400, 200))
    controller._sources[label] = lambda: source

    result = controller._pan_drag(SimpleNamespace(x=150, y=100), label)

    assert result == "break"
    assert controller.center[0] < 0.5
    assert controller.center[1] == pytest.approx(0.5)
    assert changes == [1]


def test_controller_reset_restores_defaults_and_notifies() -> None:
    changes = []
    controller = ZoomPanController(on_change=lambda: changes.append(1))
    controller.zoom = 3.0
    controller.center = (0.7, 0.6)

    assert controller.reset(SimpleNamespace()) == "break"
    assert controller.zoom == ZOOM_MIN
    assert controller.center == (0.5, 0.5)
    assert changes == [1]

    changes.clear()
    assert controller.reset() is None
    assert changes == []


def test_controller_wheel_zooms_by_step() -> None:
    controller = ZoomPanController(on_change=lambda: None)
    label = MagicMock()
    label.winfo_width.return_value = 500
    label.winfo_height.return_value = 300
    controller._sources[label] = lambda: None

    assert controller._on_wheel(SimpleNamespace(num=0, delta=120, x=10, y=10), label) == "break"
    assert controller.zoom == pytest.approx(ZOOM_MIN + ZOOM_STEP)

    assert controller._on_wheel(SimpleNamespace(num=5, delta=0, x=10, y=10), label) == "break"
    assert controller.zoom == pytest.approx(ZOOM_MIN)


def test_controller_attach_binds_gestures() -> None:
    controller = ZoomPanController(on_change=lambda: None)
    label = MagicMock()

    controller.attach(label, lambda: None)

    bound = {call.args[0] for call in label.bind.call_args_list}
    assert {
        "<MouseWheel>",
        "<Button-4>",
        "<Button-5>",
        "<ButtonPress-1>",
        "<B1-Motion>",
        "<ButtonRelease-1>",
        "<Double-Button-1>",
    } <= bound

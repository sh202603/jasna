"""Shared zoom/pan math for preview panes, in normalized source coordinates.

The formulas mirror the segment editor's preview (crop by zoom around a
normalized center, letterbox-fit into the label, HiDPI-compensated CTkImage
sizing per issue #229). ``ZoomPanController`` holds one zoom/pan state and
drives any number of labels in lockstep, which is what keeps the A/B compare
panes synchronized.
"""

from __future__ import annotations

from collections.abc import Callable

import customtkinter as ctk
from PIL import Image

ZOOM_MIN = 1.0
ZOOM_MAX = 8.0
ZOOM_STEP = 0.25

MASK_OVERLAY_COLOR = "#ef4444"
MASK_OVERLAY_ALPHA = 130


def clamp_center(center: tuple[float, float], zoom: float) -> tuple[float, float]:
    half_visible = 0.5 / zoom
    return (
        min(1.0 - half_visible, max(half_visible, float(center[0]))),
        min(1.0 - half_visible, max(half_visible, float(center[1]))),
    )


def crop_normalized(
    source: Image.Image,
    zoom: float,
    center: tuple[float, float],
) -> Image.Image:
    zoom = max(ZOOM_MIN, float(zoom))
    if zoom <= ZOOM_MIN:
        return source
    center = clamp_center(center, zoom)
    crop_width = max(1, min(source.width, round(source.width / zoom)))
    crop_height = max(1, min(source.height, round(source.height / zoom)))
    left = round(center[0] * source.width - crop_width / 2)
    top = round(center[1] * source.height - crop_height / 2)
    left = min(source.width - crop_width, max(0, left))
    top = min(source.height - crop_height, max(0, top))
    return source.crop((left, top, left + crop_width, top + crop_height))


def image_geometry(
    widget_width: int,
    widget_height: int,
    source_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Placement of a letterboxed source inside a label: ``(left, top,
    display_width, display_height)`` with the 16px frame padding."""
    widget_width = max(2, int(widget_width))
    widget_height = max(2, int(widget_height))
    available_width = max(2, widget_width - 16)
    available_height = max(2, widget_height - 16)
    source_width, source_height = source_size
    scale = min(
        available_width / source_width,
        available_height / source_height,
    )
    display_width = max(1.0, source_width * scale)
    display_height = max(1.0, source_height * scale)
    return (
        (widget_width - display_width) / 2,
        (widget_height - display_height) / 2,
        display_width,
        display_height,
    )


def fit_to_label(
    label: ctk.CTkLabel,
    source: Image.Image,
    widget_scaling: float,
) -> ctk.CTkImage:
    # winfo_* measures physical pixels while CTkImage's size is multiplied by
    # the widget scaling factor at render time; divide it back out so HiDPI
    # displays do not overflow and clip the preview (issue #229).
    width = max(2, label.winfo_width() - 16)
    height = max(2, label.winfo_height() - 16)
    source_width, source_height = source.size
    scale = min(width / source_width, height / source_height)
    pixel_size = (
        max(2, round(source_width * scale)),
        max(2, round(source_height * scale)),
    )
    image = source.resize(pixel_size, Image.Resampling.LANCZOS)
    return ctk.CTkImage(
        image,
        size=(
            max(1, round(pixel_size[0] / widget_scaling)),
            max(1, round(pixel_size[1] / widget_scaling)),
        ),
    )


def apply_mask_overlay(
    image: Image.Image,
    mask_np,
    *,
    left_eye: bool = False,
) -> Image.Image:
    """Blend a low-res detection mask over ``image`` in red, matching the
    segment editor's scan overlay."""
    if left_eye:
        mask_np = mask_np[:, : mask_np.shape[1] // 2]
    if not mask_np.any():
        return image
    alpha = Image.fromarray(
        (mask_np * MASK_OVERLAY_ALPHA).astype("uint8"), "L"
    ).resize(image.size, Image.Resampling.NEAREST)
    overlay = Image.new("RGB", image.size, MASK_OVERLAY_COLOR)
    composed = image.copy()
    composed.paste(overlay, (0, 0), alpha)
    return composed


class ZoomPanController:
    """One zoom/pan state shared by every attached preview label.

    Wheel zoom (anchored under the cursor), left-drag pan, and double-click
    reset are bound per label; any change fires ``on_change`` so the owner
    re-renders all panes from the same state.
    """

    def __init__(self, *, on_change: Callable[[], None]) -> None:
        self._on_change = on_change
        self.zoom = ZOOM_MIN
        self.center = (0.5, 0.5)
        self._pan_anchor: tuple[int, int] | None = None
        self._sources: dict[object, Callable[[], Image.Image | None]] = {}

    @property
    def is_default(self) -> bool:
        return self.zoom <= ZOOM_MIN and self.center == (0.5, 0.5)

    def attach(
        self,
        label,
        get_source: Callable[[], Image.Image | None],
    ) -> None:
        self._sources[label] = get_source
        label.bind("<MouseWheel>", lambda event, l=label: self._on_wheel(event, l))
        label.bind("<Button-4>", lambda event, l=label: self._on_wheel(event, l))
        label.bind("<Button-5>", lambda event, l=label: self._on_wheel(event, l))
        label.bind("<ButtonPress-1>", self._pan_start)
        label.bind("<B1-Motion>", lambda event, l=label: self._pan_drag(event, l))
        label.bind("<ButtonRelease-1>", self._pan_end)
        label.bind("<Double-Button-1>", self.reset)

    def crop(self, source: Image.Image) -> Image.Image:
        if self.zoom <= ZOOM_MIN:
            self.center = (0.5, 0.5)
            return source
        self.center = clamp_center(self.center, self.zoom)
        return crop_normalized(source, self.zoom, self.center)

    def set_zoom(
        self,
        zoom: float,
        *,
        anchor: tuple[float, float] | None = None,
        label=None,
    ) -> None:
        old_zoom = float(self.zoom)
        new_zoom = min(ZOOM_MAX, max(ZOOM_MIN, float(zoom)))
        if new_zoom == old_zoom:
            return
        center = clamp_center(self.center, old_zoom)
        source = None
        if anchor is not None and label is not None:
            get_source = self._sources.get(label)
            source = get_source() if get_source is not None else None
        if anchor is not None and source is not None:
            left, top, width, height = image_geometry(
                label.winfo_width(), label.winfo_height(), source.size
            )
            fx = min(1.0, max(0.0, (anchor[0] - left) / width))
            fy = min(1.0, max(0.0, (anchor[1] - top) / height))
            source_x = center[0] + (fx - 0.5) / old_zoom
            source_y = center[1] + (fy - 0.5) / old_zoom
            center = (
                source_x - (fx - 0.5) / new_zoom,
                source_y - (fy - 0.5) / new_zoom,
            )
        self.zoom = new_zoom
        self.center = clamp_center(center, new_zoom)
        self._on_change()

    def adjust(self, amount: float) -> None:
        self.set_zoom(self.zoom + float(amount))

    def reset(self, event=None):
        changed = not self.is_default or self._pan_anchor is not None
        self.zoom = ZOOM_MIN
        self.center = (0.5, 0.5)
        self._pan_anchor = None
        if changed:
            self._on_change()
        return "break" if event is not None else None

    def _on_wheel(self, event, label):
        button = int(getattr(event, "num", 0))
        delta = int(getattr(event, "delta", 0))
        direction = (
            1 if button == 4 or delta > 0 else -1 if button == 5 or delta < 0 else 0
        )
        if not direction:
            return None
        self.set_zoom(
            self.zoom + direction * ZOOM_STEP,
            anchor=(float(event.x), float(event.y)),
            label=label,
        )
        return "break"

    def _pan_start(self, event):
        if self.zoom <= ZOOM_MIN:
            self._pan_anchor = None
            return None
        self._pan_anchor = (int(event.x), int(event.y))
        return "break"

    def _pan_drag(self, event, label):
        anchor = self._pan_anchor
        get_source = self._sources.get(label)
        source = get_source() if get_source is not None else None
        if anchor is None or source is None or self.zoom <= ZOOM_MIN:
            return None
        _, _, display_width, display_height = image_geometry(
            label.winfo_width(), label.winfo_height(), source.size
        )
        dx = int(event.x) - anchor[0]
        dy = int(event.y) - anchor[1]
        self.center = clamp_center(
            (
                self.center[0] - dx / display_width / self.zoom,
                self.center[1] - dy / display_height / self.zoom,
            ),
            self.zoom,
        )
        self._pan_anchor = (int(event.x), int(event.y))
        self._on_change()
        return "break"

    def _pan_end(self, _event=None):
        was_panning = self._pan_anchor is not None
        self._pan_anchor = None
        return "break" if was_panning else None

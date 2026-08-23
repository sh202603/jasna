"""A/B compare window for the segment editor.

Two panes render the same frame (or clip window) processed with two
independently selected detection-model / restoration-checkpoint combinations.
Runs are explicit (the Run button); selection or time changes only mark the
panes stale. The single ``ABCompareWorker`` thread executes leg A then leg B
sequentially (a CUDA-graphs constraint, see the worker module docstring).
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path

import customtkinter as ctk
from PIL import Image

from jasna.gui import scaling
from jasna.gui.ab_compare_worker import (
    ABClipResult,
    ABCompareWorker,
    ABCompileFinished,
    ABDetectionResult,
    ABFailed,
    ABFrameResult,
    ABSessionInfo,
    ABSideConfig,
    ABStatus,
)
from jasna.gui.mosaic_scan import SCAN_SCORE_FLOOR
from jasna.gui.components import Tooltip
from jasna.gui.locales import t
from jasna.gui.models import AppSettings
from jasna.gui.preview_zoom import ZoomPanController, apply_mask_overlay, fit_to_label
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.media import VideoMetadata
from jasna.segments import format_timestamp


class _PaneState:
    """Model state of one compare pane; widget-free so window logic stays
    testable without a Tk display."""

    def __init__(
        self,
        detection_model: str,
        threshold: float,
        checkpoint: Path | None,
    ) -> None:
        self.detection_model = str(detection_model)
        self.threshold = float(threshold)
        self.checkpoint = checkpoint
        # "basicvsrpp" (checkpoint = BasicVSR++ .pth) or "seedvr2"
        # (checkpoint = the LoRA file; restored by the external worker).
        self.restoration_model_name = "basicvsrpp"
        self.stale = False
        self.status: str | None = None
        self.error: str | None = None
        self.tensorrt: bool | None = None
        self.provisional_tensorrt: bool | None = None
        self.frame_image: Image.Image | None = None
        self.frame_seconds: float | None = None
        self.detection: tuple | None = None  # (seconds, original, mask, score)
        self.clip_frames: tuple = ()


class _ComparePane(ctk.CTkFrame):
    def __init__(
        self,
        master,
        *,
        side_label: str,
        detection_models: list[str],
        checkpoint_names: list[str],
        initial_model: str,
        initial_checkpoint_name: str | None,
        initial_threshold: float,
        on_model_selected: Callable[[str], None],
        on_checkpoint_selected: Callable[[str], None],
        on_threshold_changed: Callable[[float], None],
        on_compile: Callable[[], None],
    ) -> None:
        super().__init__(
            master,
            fg_color=Colors.BG_CARD,
            corner_radius=Sizing.BORDER_RADIUS,
        )
        self.image_ref = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
        ctk.CTkLabel(
            header,
            text=side_label,
            font=(Fonts.FAMILY, Fonts.SIZE_HEADING, "bold"),
            text_color=Colors.TEXT_PRIMARY,
        ).pack(side="left")
        self.badge = ctk.CTkLabel(
            header,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY, "bold"),
            corner_radius=4,
            padx=6,
        )
        self.badge.pack(side="left", padx=(8, 0))
        self.badge_tooltip = Tooltip(self.badge, "")
        self.compile_btn = ctk.CTkButton(
            header,
            text=t("ab_compile_engines"),
            height=22,
            width=130,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=on_compile,
        )
        # Packed by the window only while the checkpoint lacks TRT engines.
        Tooltip(self.compile_btn, t("ab_compile_engines_hint"))
        self.stale_label = ctk.CTkLabel(
            header,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PAUSED,
        )
        self.stale_label.pack(side="right")

        self.image = ctk.CTkLabel(
            self,
            text="",
            fg_color=Colors.BG_PANEL,
            text_color=Colors.STATUS_PENDING,
            corner_radius=Sizing.BORDER_RADIUS,
        )
        self.image.grid(row=1, column=0, sticky="nsew", padx=8, pady=2)

        self.info = ctk.CTkLabel(
            self,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
            anchor="w",
        )
        self.info.grid(row=2, column=0, sticky="ew", padx=8)

        selectors = ctk.CTkFrame(self, fg_color="transparent")
        selectors.grid(row=3, column=0, sticky="ew", padx=8, pady=(2, 8))
        selectors.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            selectors,
            text=t("ab_detection_model"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
        ).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=1)
        self.model_menu = ctk.CTkOptionMenu(
            selectors,
            values=detection_models,
            height=26,
            command=on_model_selected,
        )
        self.model_menu.set(initial_model)
        self.model_menu.grid(row=0, column=1, sticky="ew", pady=1)
        ctk.CTkLabel(
            selectors,
            text=t("segments_scan_threshold"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
        ).grid(row=1, column=0, sticky="w", padx=(0, 6), pady=1)
        threshold_control = ctk.CTkFrame(selectors, fg_color="transparent")
        threshold_control.grid(row=1, column=1, sticky="ew", pady=1)
        self.threshold_slider = ctk.CTkSlider(
            threshold_control,
            from_=SCAN_SCORE_FLOOR,
            to=1.0,
            command=on_threshold_changed,
        )
        self.threshold_slider.set(initial_threshold)
        self.threshold_slider.pack(side="left", fill="x", expand=True)
        self.threshold_label = ctk.CTkLabel(
            threshold_control,
            text=f"{initial_threshold:.2f}",
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
            width=42,
        )
        self.threshold_label.pack(side="left", padx=(6, 0))
        ctk.CTkLabel(
            selectors,
            text=t("ab_restoration_model"),
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            text_color=Colors.STATUS_PENDING,
        ).grid(row=2, column=0, sticky="w", padx=(0, 6), pady=1)
        self.ckpt_menu = ctk.CTkOptionMenu(
            selectors,
            values=checkpoint_names or [t("ab_no_checkpoints")],
            height=26,
            command=on_checkpoint_selected,
        )
        if initial_checkpoint_name is not None:
            self.ckpt_menu.set(initial_checkpoint_name)
        else:
            self.ckpt_menu.set(t("ab_no_checkpoints"))
            self.ckpt_menu.configure(state="disabled")
        self.ckpt_menu.grid(row=2, column=1, sticky="ew", pady=1)


class ABCompareWindow(ctk.CTkToplevel):
    """Modal A/B comparison window opened from the segment editor."""

    # Class-level defaults so partially-constructed windows (tests build the
    # window without running __init__) behave as "seedvr2 unavailable".
    _seedvr2_name: str | None = None
    _seedvr2_repo: str = ""

    def __init__(
        self,
        master,
        *,
        path: str | Path,
        metadata: VideoMetadata,
        get_settings: Callable[[], AppSettings],
        center_seconds: float,
        projection: str,
        initial_detection_model: str,
        initial_threshold: float,
        is_gpu_busy: Callable[[], bool],
        set_preview_gpu_busy: Callable[[bool], None],
        on_closed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self._path = Path(path)
        self._metadata = metadata
        self._get_settings = get_settings
        self._projection = projection
        self._initial_detection_model = str(initial_detection_model)
        self._initial_threshold = float(initial_threshold)
        self._is_gpu_busy = is_gpu_busy
        self._set_preview_gpu_busy = set_preview_gpu_busy
        self._on_closed = on_closed

        self._closed = threading.Event()
        self._view = "restored"
        self._generation = 0
        self._pending_sides: set[str] = set()
        self._compiling_side: str | None = None
        self._playing = False
        self._clip_after: str | None = None
        self._clip_index = 0
        self._clip_position: float | None = None
        self._poll_after: str | None = None
        self._resize_after: str | None = None
        self._fps = max(1.0, float(metadata.video_fps))
        self._duration = float(metadata.duration)
        self._center_seconds = min(self._duration, max(0.0, float(center_seconds)))

        from jasna.vr180 import resolve_vr_mode

        self._left_eye = resolve_vr_mode(
            get_settings().vr_mode,
            metadata,
            self._path,
            projection=projection,
        ).is_sbs

        from jasna.restorer.checkpoint_info import discover_restoration_checkpoints, discover_seedvr2

        checkpoints = discover_restoration_checkpoints()
        self._checkpoints: dict[str, Path] = {p.name: p for p in checkpoints}
        # The seedvr2 primary restorer joins the dropdown as a pseudo entry
        # (value = its LoRA path) when the external checkout and LoRA exist.
        # Checkpoint keys carry a .pth suffix, so the bare name cannot collide.
        self._seedvr2_name: str | None = None
        self._seedvr2_repo: str = ""
        seedvr2 = discover_seedvr2()
        if seedvr2 is not None:
            seedvr2_repo, seedvr2_lora = seedvr2
            self._seedvr2_name = "seedvr2"
            self._seedvr2_repo = str(seedvr2_repo)
            self._checkpoints[self._seedvr2_name] = seedvr2_lora
        initial_checkpoint = self._default_checkpoint(checkpoints)

        from jasna.mosaic.detection_registry import detection_model_choices

        models = detection_model_choices()
        if self._initial_detection_model not in models:
            models.insert(0, self._initial_detection_model)
        self._detection_models = models

        self._panes = {
            "a": _PaneState(
                self._initial_detection_model, self._initial_threshold, initial_checkpoint
            ),
            "b": _PaneState(
                self._initial_detection_model, self._initial_threshold, initial_checkpoint
            ),
        }
        for pane in self._panes.values():
            pane.provisional_tensorrt = self._provisional_tensorrt(
                pane.checkpoint, pane.restoration_model_name
            )

        self._zoom = ZoomPanController(on_change=self._on_zoom_changed)

        self.title(t("ab_title"))
        self.configure(fg_color=Colors.BG_MAIN)
        self.transient(master.winfo_toplevel())
        self.protocol("WM_DELETE_WINDOW", self.close)
        screen_w, screen_h = scaling.to_logical(self, *scaling.screen_rect(self)[2:])
        width = min(screen_w - 80, max(1100, screen_w - 300))
        height = min(screen_h - 120, max(680, screen_h - 240))
        scaling.place_centered_on_screen(self, *scaling.to_physical(self, width, height))

        self._build_ui(initial_checkpoint)
        for side in ("a", "b"):
            self._refresh_badge(side)
        self.bind("<Escape>", lambda _event: self.close())
        self.update_idletasks()
        self.wait_visibility()
        self.grab_set()

        self._set_preview_gpu_busy(True)
        try:
            self._worker = ABCompareWorker(
                self._path,
                metadata,
                on_stopped=lambda: self._set_preview_gpu_busy(False),
            )
            self._worker.start()
        except Exception:
            self._set_preview_gpu_busy(False)
            raise
        self._poll_after = self.after(25, self._poll_worker)

    @staticmethod
    def _default_checkpoint(checkpoints: list[Path]) -> Path | None:
        from jasna.engine_paths import default_restoration_model_path

        default = default_restoration_model_path()
        if default in checkpoints:
            return default
        return checkpoints[0] if checkpoints else None

    def _build_ui(self, initial_checkpoint: Path | None) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(12, 4))
        ctk.CTkLabel(
            header,
            text=self._path.name,
            font=(Fonts.FAMILY, Fonts.SIZE_LARGE, "bold"),
            text_color=Colors.TEXT_PRIMARY,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        transport = ctk.CTkFrame(header, fg_color="transparent")
        transport.pack(side="right")
        for text, command, tip in (
            ("−1s", lambda: self._step_seconds(-1.0), "ab_step_second_back"),
            ("|◀", lambda: self._step_frames(-1), "segments_previous_frame"),
            ("▶|", lambda: self._step_frames(1), "segments_next_frame"),
            ("+1s", lambda: self._step_seconds(1.0), "ab_step_second_forward"),
        ):
            button = ctk.CTkButton(
                transport,
                text=text,
                width=44,
                height=26,
                fg_color=Colors.BG_PANEL,
                hover_color=Colors.BORDER_LIGHT,
                command=command,
            )
            button.pack(side="left", padx=2)
            Tooltip(button, t(tip))
        self._time_label = ctk.CTkLabel(
            transport,
            text=self._time_text(),
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._time_label.pack(side="left", padx=(10, 0))

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.pack(fill="x", padx=16, pady=(0, 4))
        self._view_by_label = {
            t("ab_view_restored"): "restored",
            t("ab_view_detection"): "detection",
            t("ab_view_clip"): "clip",
        }
        self._view_selector = ctk.CTkSegmentedButton(
            controls,
            values=list(self._view_by_label),
            command=self._on_view_selected,
        )
        self._view_selector.set(t("ab_view_restored"))
        self._view_selector.pack(side="left")
        self._run_btn = ctk.CTkButton(
            controls,
            text=t("ab_run"),
            width=110,
            command=self._run_clicked,
        )
        self._run_btn.pack(side="right")
        self._status_label = ctk.CTkLabel(
            controls,
            text="",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            text_color=Colors.STATUS_PENDING,
            anchor="w",
        )
        self._status_label.pack(side="left", fill="x", expand=True, padx=(12, 12))

        panes = ctk.CTkFrame(self, fg_color="transparent")
        panes.pack(fill="both", expand=True, padx=16, pady=4)
        panes.grid_columnconfigure(0, weight=1, uniform="ab")
        panes.grid_columnconfigure(1, weight=1, uniform="ab")
        panes.grid_rowconfigure(0, weight=1)
        checkpoint_names = list(self._checkpoints)
        initial_name = initial_checkpoint.name if initial_checkpoint is not None else None
        self._pane_widgets: dict[str, _ComparePane] = {}
        for column, side in enumerate(("a", "b")):
            pane = _ComparePane(
                panes,
                side_label=side.upper(),
                detection_models=list(self._detection_models),
                checkpoint_names=checkpoint_names,
                initial_model=self._initial_detection_model,
                initial_checkpoint_name=initial_name,
                initial_threshold=self._panes[side].threshold,
                on_model_selected=lambda model, s=side: self._on_detection_model_selected(s, model),
                on_checkpoint_selected=lambda name, s=side: self._on_checkpoint_selected(s, name),
                on_threshold_changed=lambda value, s=side: self._on_side_threshold(s, value),
                on_compile=lambda s=side: self._compile_clicked(s),
            )
            pane.grid(row=0, column=column, sticky="nsew", padx=(0, 6) if column == 0 else (6, 0))
            pane.image.bind("<Configure>", self._pane_resized)
            self._zoom.attach(pane.image, lambda s=side: self._pane_source(s))
            self._pane_widgets[side] = pane

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=16, pady=(4, 12))
        self._play_btn = ctk.CTkButton(
            footer,
            text="▶",
            width=48,
            command=self._toggle_clip_play,
        )
        # Packed only in the clip view (see _update_play_button_visibility).
        zoom_controls = ctk.CTkFrame(footer, fg_color="transparent")
        zoom_controls.pack(side="left", expand=True)
        from jasna.gui.preview_zoom import ZOOM_STEP

        self._zoom_out_btn = ctk.CTkButton(
            zoom_controls,
            text="−",
            width=30,
            height=26,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=lambda: self._zoom.adjust(-ZOOM_STEP),
        )
        self._zoom_out_btn.pack(side="left")
        Tooltip(self._zoom_out_btn, t("segments_preview_zoom_out"))
        self._zoom_label = ctk.CTkLabel(
            zoom_controls,
            text="100%",
            width=48,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_TINY),
            text_color=Colors.TEXT_PRIMARY,
        )
        self._zoom_label.pack(side="left", padx=3)
        self._zoom_in_btn = ctk.CTkButton(
            zoom_controls,
            text="+",
            width=30,
            height=26,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=lambda: self._zoom.adjust(ZOOM_STEP),
        )
        self._zoom_in_btn.pack(side="left")
        Tooltip(self._zoom_in_btn, t("segments_preview_zoom_in"))
        self._reset_view_btn = ctk.CTkButton(
            zoom_controls,
            text=t("segments_preview_reset_view"),
            width=84,
            height=26,
            fg_color=Colors.BG_PANEL,
            hover_color=Colors.BORDER_LIGHT,
            command=self._zoom.reset,
        )
        self._reset_view_btn.pack(side="left", padx=(6, 0))
        self._close_btn = ctk.CTkButton(
            footer,
            text=t("ab_close"),
            fg_color=Colors.BG_CARD,
            hover_color=Colors.BORDER_LIGHT,
            command=self.close,
        )
        self._close_btn.pack(side="right")

    # ------------------------------------------------------------------ run

    def _view_needs_restoration(self) -> bool:
        return self._view in ("restored", "clip")

    def _run_clicked(self) -> None:
        if self._closed.is_set() or self._compiling_side is not None:
            return
        if self._view_needs_restoration() and any(
            pane.checkpoint is None for pane in self._panes.values()
        ):
            self._status_label.configure(
                text=t("ab_no_checkpoints"), text_color=Colors.STATUS_ERROR
            )
            return
        if (
            self._view_needs_restoration()
            and self._left_eye
            and any(
                pane.restoration_model_name == "seedvr2"
                for pane in self._panes.values()
            )
        ):
            # jasna's VR crops are projection-conditioned — out of distribution
            # for the LoRA and unverified (same rejection as the CLI).
            self._status_label.configure(
                text=t("ab_seedvr2_vr_unsupported"), text_color=Colors.STATUS_ERROR
            )
            return
        self._stop_clip_playback()
        self._clip_position = None
        side_a, side_b = (
            ABSideConfig(
                detection_model=pane.detection_model,
                detection_score_threshold=pane.threshold,
                restoration_model_path=pane.checkpoint,
                restoration_model_name=pane.restoration_model_name,
                seedvr2_repo=self._seedvr2_repo,
            )
            for pane in (self._panes["a"], self._panes["b"])
        )
        self._generation = self._worker.request(
            self._center_seconds,
            self._view,
            side_a,
            side_b,
            self._get_settings(),
            self._projection,
        )
        self._pending_sides = {"a", "b"}
        for pane in self._panes.values():
            pane.stale = False
            pane.error = None
            pane.status = t("segments_restore_restoring")
        self._status_label.configure(
            text=t("segments_restore_restoring"), text_color=Colors.STATUS_PENDING
        )
        self._refresh_pane_images()

    # ------------------------------------------------------------- worker I/O

    def _poll_worker(self) -> None:
        if self._closed.is_set():
            return
        try:
            while True:
                self._handle_worker_event(self._worker.events.get_nowait())
        except queue.Empty:
            pass
        self._poll_after = self.after(25, self._poll_worker)

    def _handle_worker_event(self, event) -> None:
        if event.generation != self._generation:
            return
        pane = self._panes.get(event.side)
        if pane is None:
            return
        if isinstance(event, ABStatus):
            message = self._status_text(event.message)
            pane.status = message
            self._status_label.configure(
                text=f"{event.side.upper()}: {message}",
                text_color=Colors.STATUS_PENDING,
            )
            self._refresh_pane(event.side)
        elif isinstance(event, ABSessionInfo):
            # For seedvr2, no-TRT is the normal state, not a warning badge.
            pane.tensorrt = (
                None if pane.restoration_model_name == "seedvr2" else bool(event.tensorrt)
            )
            self._refresh_badge(event.side)
        elif isinstance(event, ABFrameResult):
            pane.status = None
            pane.error = None
            pane.frame_image = event.image
            pane.frame_seconds = event.seconds
            self._finish_side(event.side)
        elif isinstance(event, ABDetectionResult):
            pane.status = None
            pane.error = None
            pane.detection = (event.seconds, event.original, event.mask, event.score)
            self._finish_side(event.side)
        elif isinstance(event, ABClipResult):
            pane.status = None
            pane.error = None
            pane.clip_frames = event.frames
            self._finish_side(event.side)
        elif isinstance(event, ABFailed):
            pane.status = None
            if event.code == "checkpoint_missing_ema":
                pane.error = t("ab_checkpoint_missing_ema", name=event.message)
            else:
                pane.error = t("segments_restore_failed", message=event.message)
            self._finish_side(event.side)
        elif isinstance(event, ABCompileFinished):
            self._finish_compile(event)

    def _finish_side(self, side: str) -> None:
        self._pending_sides.discard(side)
        if not self._pending_sides:
            self._status_label.configure(text="")
        self._refresh_pane(side)

    @staticmethod
    def _status_text(message: str) -> str:
        if message == "loading_models":
            return t("segments_restore_loading_models")
        if message == "restoring":
            return t("segments_restore_restoring")
        if message == "compiling":
            return t("ab_compiling")
        return message

    # ----------------------------------------------------------- compilation

    def _compile_available(self) -> bool:
        # The compiler only builds BasicVSR++ sub-engines for fp16; with TRT or
        # fp16 globally off a compile could never change the execution path.
        settings = self._get_settings()
        return bool(settings.compile_basicvsrpp) and bool(settings.fp16_mode)

    def _compile_clicked(self, side: str) -> None:
        if self._closed.is_set() or self._compiling_side is not None:
            return
        pane = self._panes[side]
        if pane.checkpoint is None or pane.restoration_model_name == "seedvr2":
            return
        self._stop_clip_playback()
        self._compiling_side = side
        self._pending_sides = set()
        self._generation = self._worker.request_compile(
            side, pane.checkpoint, self._get_settings()
        )
        # A superseded in-flight run never delivers results; drop its
        # "Restoring…" placeholders so only the compiling side shows activity.
        for other in self._panes.values():
            other.status = None
        pane.status = t("ab_compiling")
        self._status_label.configure(
            text=t("ab_compiling"), text_color=Colors.STATUS_PENDING
        )
        self._set_controls_enabled(False)
        self._refresh_pane_images()

    def _finish_compile(self, event: ABCompileFinished) -> None:
        self._compiling_side = None
        self._set_controls_enabled(True)
        compiled_pane = self._panes[event.side]
        compiled_pane.status = None
        if event.ok:
            self._status_label.configure(
                text=t("ab_compile_done"), text_color=Colors.STATUS_COMPLETED
            )
        else:
            self._status_label.configure(
                text=t("ab_compile_failed", message=event.message),
                text_color=Colors.STATUS_ERROR,
            )
        # Engines are per checkpoint, so the other side benefits too when it
        # points at the same file; results computed on PyTorch are now stale.
        for side, pane in self._panes.items():
            if pane.checkpoint != compiled_pane.checkpoint:
                continue
            pane.tensorrt = None
            pane.provisional_tensorrt = self._provisional_tensorrt(
                pane.checkpoint, pane.restoration_model_name
            )
            if event.ok:
                self._mark_stale(side)
            self._refresh_badge(side)
            self._refresh_pane(side)

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._run_btn.configure(state=state)
        self._view_selector.configure(state=state)
        for widgets in self._pane_widgets.values():
            widgets.model_menu.configure(state=state)
            widgets.ckpt_menu.configure(
                state=state if self._checkpoints else "disabled"
            )
            widgets.threshold_slider.configure(state=state)
            widgets.compile_btn.configure(state=state)

    # ------------------------------------------------------------- selection

    def _on_detection_model_selected(self, side: str, model: str) -> None:
        pane = self._panes[side]
        if model == pane.detection_model:
            return
        pane.detection_model = str(model)
        self._set_side_threshold(side, self._threshold_for_model(model))
        self._mark_stale(side)

    def _threshold_for_model(self, model: str) -> float:
        if model == self._initial_detection_model:
            return self._initial_threshold
        from jasna.mosaic.detection_registry import recommended_score_threshold

        try:
            return float(recommended_score_threshold(model))
        except ValueError:
            return self._initial_threshold

    def _on_side_threshold(self, side: str, value: float) -> None:
        pane = self._panes[side]
        value = min(1.0, max(SCAN_SCORE_FLOOR, float(value)))
        if value == pane.threshold:
            return
        pane.threshold = value
        self._pane_widgets[side].threshold_label.configure(text=f"{value:.2f}")
        self._mark_stale(side)

    def _set_side_threshold(self, side: str, value: float) -> None:
        pane = self._panes[side]
        pane.threshold = min(1.0, max(SCAN_SCORE_FLOOR, float(value)))
        widgets = self._pane_widgets[side]
        widgets.threshold_slider.set(pane.threshold)
        widgets.threshold_label.configure(text=f"{pane.threshold:.2f}")

    def _on_checkpoint_selected(self, side: str, name: str) -> None:
        pane = self._panes[side]
        checkpoint = self._checkpoints.get(name)
        model_name = "seedvr2" if name == self._seedvr2_name else "basicvsrpp"
        if checkpoint == pane.checkpoint and model_name == pane.restoration_model_name:
            return
        pane.checkpoint = checkpoint
        pane.restoration_model_name = model_name
        pane.tensorrt = None
        pane.provisional_tensorrt = self._provisional_tensorrt(checkpoint, model_name)
        self._mark_stale(side)
        self._refresh_badge(side)

    def _provisional_tensorrt(
        self, checkpoint: Path | None, model_name: str = "basicvsrpp"
    ) -> bool | None:
        """Cheap file-system-only prediction of the execution backend, shown
        until the run's ABSessionInfo confirms it."""
        if checkpoint is None or model_name == "seedvr2":
            return None
        settings = self._get_settings()
        if not bool(settings.compile_basicvsrpp):
            return False
        from jasna.engine_paths import all_basicvsrpp_sub_engines_exist

        return all_basicvsrpp_sub_engines_exist(str(checkpoint), bool(settings.fp16_mode))

    def _mark_stale(self, side: str) -> None:
        self._panes[side].stale = True
        self._refresh_stale(side)

    # ------------------------------------------------------------------ time

    def _step_frames(self, frames: int) -> None:
        self._set_center(self._center_seconds + int(frames) / self._fps)

    def _step_seconds(self, seconds: float) -> None:
        self._set_center(self._center_seconds + float(seconds))

    def _set_center(self, seconds: float) -> None:
        seconds = min(self._duration, max(0.0, float(seconds)))
        if seconds == self._center_seconds:
            return
        self._center_seconds = seconds
        self._time_label.configure(text=self._time_text())
        self._stop_clip_playback()
        for side in ("a", "b"):
            self._mark_stale(side)

    def _time_text(self) -> str:
        return f"{format_timestamp(self._center_seconds)} / {format_timestamp(self._duration)}"

    # ------------------------------------------------------------------ view

    def _on_view_selected(self, label: str) -> None:
        view = self._view_by_label.get(label, "restored")
        if view == self._view:
            return
        self._view = view
        self._stop_clip_playback()
        self._clip_position = None
        self._update_play_button_visibility()
        self._refresh_pane_images()

    def _update_play_button_visibility(self) -> None:
        if self._view == "clip":
            if not self._play_btn.winfo_manager():
                self._play_btn.pack(side="left", padx=(0, 12))
        elif self._play_btn.winfo_manager():
            self._play_btn.pack_forget()

    # ------------------------------------------------------------------ clip

    def _clock_frames(self) -> tuple:
        frames = self._panes["a"].clip_frames
        return frames if frames else self._panes["b"].clip_frames

    def _toggle_clip_play(self) -> None:
        if self._playing:
            self._stop_clip_playback()
            return
        if len(self._clock_frames()) < 2:
            return
        self._playing = True
        self._clip_index = 0
        self._play_btn.configure(text="⏸")
        self._advance_clip()

    def _advance_clip(self) -> None:
        self._clip_after = None
        if not self._playing or self._closed.is_set():
            return
        frames = self._clock_frames()
        if self._clip_index >= len(frames):
            self._stop_clip_playback()
            return
        self._clip_position = frames[self._clip_index].seconds
        self._clip_index += 1
        self._refresh_pane_images()
        delay = max(10, round(1000 / min(60.0, self._fps)))
        self._clip_after = self.after(delay, self._advance_clip)

    def _stop_clip_playback(self) -> None:
        self._playing = False
        if self._clip_after is not None:
            try:
                self.after_cancel(self._clip_after)
            except tk.TclError:
                pass
            self._clip_after = None
        if hasattr(self, "_play_btn"):
            try:
                self._play_btn.configure(text="▶")
            except tk.TclError:
                pass

    # ------------------------------------------------------------- rendering

    def _pane_source(self, side: str) -> Image.Image | None:
        pane = self._panes[side]
        if self._view == "restored":
            return pane.frame_image
        if self._view == "detection":
            if pane.detection is None:
                return None
            _, original, mask, _ = pane.detection
            return apply_mask_overlay(original, mask.numpy(), left_eye=self._left_eye)
        if not pane.clip_frames:
            return None
        if self._clip_position is None:
            return pane.clip_frames[0].image
        return min(
            pane.clip_frames,
            key=lambda frame: abs(frame.seconds - self._clip_position),
        ).image

    def _refresh_pane_images(self) -> None:
        for side in ("a", "b"):
            self._refresh_pane(side)

    def _refresh_pane(self, side: str) -> None:
        if self._closed.is_set():
            return
        pane = self._panes[side]
        widgets = self._pane_widgets[side]

        info = ""
        if self._view == "detection" and pane.detection is not None and not pane.error:
            score = pane.detection[3]
            info = (
                t("ab_score", score=f"{score:.2f}")
                if score > 0
                else t("ab_no_detection")
            )
        widgets.info.configure(text=info)

        source = None if pane.error else self._pane_source(side)
        if source is None:
            text = pane.error or pane.status or ""
            color = Colors.STATUS_ERROR if pane.error else Colors.STATUS_PENDING
            # CTkLabel.configure(image=None) updates its Python-side image
            # reference but does not clear the underlying tkinter.Label image;
            # clear that first so releasing our CTkImage cannot leave Tk
            # pointing at a deleted ``pyimage``.
            widgets.image._label.configure(image="")
            widgets.image.configure(image=None, text=text, text_color=color)
            widgets.image_ref = None
        else:
            widgets.image_ref = fit_to_label(
                widgets.image,
                self._zoom.crop(source),
                scaling.widget_scaling(widgets.image),
            )
            widgets.image.configure(image=widgets.image_ref, text="")
        self._refresh_stale(side)

    def _refresh_stale(self, side: str) -> None:
        widgets = self._pane_widgets[side]
        pane = self._panes[side]
        widgets.stale_label.configure(text=t("ab_stale_hint") if pane.stale else "")

    def _refresh_badge(self, side: str) -> None:
        pane = self._panes[side]
        widgets = self._pane_widgets[side]
        tensorrt = pane.tensorrt if pane.tensorrt is not None else pane.provisional_tensorrt
        if tensorrt is None:
            widgets.badge.configure(text="", fg_color="transparent")
            widgets.badge_tooltip.set_text("")
        elif tensorrt:
            from jasna.engine_paths import trt_flavor

            badge_key = "ab_badge_trt_rtx" if trt_flavor() == "rtx" else "ab_badge_trt"
            widgets.badge.configure(
                text=t(badge_key),
                fg_color=Colors.BG_PANEL,
                text_color=Colors.STATUS_COMPLETED,
            )
            widgets.badge_tooltip.set_text("")
        else:
            widgets.badge.configure(
                text=t("ab_badge_pytorch"),
                fg_color=Colors.BG_PANEL,
                text_color=Colors.STATUS_PAUSED,
            )
            widgets.badge_tooltip.set_text(t("ab_badge_pytorch_hint"))
        show_compile = (
            tensorrt is False
            and pane.checkpoint is not None
            and self._compile_available()
        )
        if show_compile:
            if not widgets.compile_btn.winfo_manager():
                widgets.compile_btn.pack(side="left", padx=(6, 0))
        elif widgets.compile_btn.winfo_manager():
            widgets.compile_btn.pack_forget()

    def _on_zoom_changed(self) -> None:
        zoom = self._zoom.zoom
        self._zoom_label.configure(text=f"{round(zoom * 100)}%")
        from jasna.gui.preview_zoom import ZOOM_MAX, ZOOM_MIN

        self._zoom_out_btn.configure(
            state="disabled" if zoom <= ZOOM_MIN else "normal"
        )
        self._zoom_in_btn.configure(
            state="disabled" if zoom >= ZOOM_MAX else "normal"
        )
        for side in ("a", "b"):
            self._pane_widgets[side].image.configure(
                cursor="fleur" if zoom > ZOOM_MIN else ""
            )
        self._refresh_pane_images()

    def _pane_resized(self, _event=None) -> None:
        if self._resize_after is not None:
            try:
                self.after_cancel(self._resize_after)
            except tk.TclError:
                pass
        self._resize_after = self.after(60, self._refresh_pane_images)

    # ----------------------------------------------------------------- close

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._stop_clip_playback()
        if self._poll_after is not None:
            try:
                self.after_cancel(self._poll_after)
            except tk.TclError:
                pass
            self._poll_after = None
        if getattr(self, "_worker", None) is not None:
            self._worker.close()
        try:
            self.grab_release()
        except tk.TclError:
            pass
        if self._on_closed is not None:
            self._on_closed()
        self.destroy()

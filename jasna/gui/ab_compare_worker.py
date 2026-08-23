"""Background worker for the segment editor's A/B compare window.

Runs two configurations (detection model x restoration checkpoint) over the
same frame or window and emits per-side results. Both legs run **sequentially
on this one thread**: torch_tensorrt's CUDA-graphs mode is process-global and
capture is poisoned by CUDA calls from other threads (see the warmup docstring
in ``jasna.restorer.basicvsrpp_sub_engines``), so the A and B sessions must
never execute concurrently.

Requests are coalesced (queue of one) with generation tagging, like
``RestorationPreviewWorker``. A request that arrives while leg A is running
cancels the in-flight pass; one that arrives between legs skips leg B.
"""

from __future__ import annotations

import queue
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image

from jasna.gui.models import AppSettings
from jasna.gui.restoration_preview import (
    RestoredClipFrame,
    _frame_image,
    run_restoration_pass,
)
from jasna.gui.video_session import (
    build_video_session,
    release_session_memory,
    video_session_key,
)
from jasna.media import VideoMetadata

AB_PREVIEW_MAX_SIZE = (960, 540)

AB_VIEWS = ("restored", "detection", "clip")


@dataclass(frozen=True)
class ABSideConfig:
    detection_model: str
    detection_score_threshold: float
    # For "seedvr2", restoration_model_path carries the LoRA checkpoint and
    # seedvr2_repo the external checkout (same convention as the CLI).
    restoration_model_path: Path
    restoration_model_name: str = "basicvsrpp"
    seedvr2_repo: str = ""


@dataclass(frozen=True)
class ABStatus:
    side: str
    message: str
    generation: int


@dataclass(frozen=True)
class ABSessionInfo:
    side: str
    generation: int
    tensorrt: bool
    checkpoint_name: str


@dataclass(frozen=True)
class ABFrameResult:
    side: str
    seconds: float
    image: Image.Image
    generation: int


@dataclass(frozen=True)
class ABDetectionResult:
    side: str
    seconds: float
    original: Image.Image
    mask: object
    score: float
    generation: int


@dataclass(frozen=True)
class ABClipResult:
    side: str
    frames: tuple[RestoredClipFrame, ...]
    generation: int


@dataclass(frozen=True)
class ABFailed:
    side: str
    code: str  # "checkpoint_missing_ema" | "error"
    message: str
    generation: int


@dataclass(frozen=True)
class ABCompileFinished:
    side: str
    ok: bool
    message: str
    generation: int


ABEvent = (
    ABStatus
    | ABSessionInfo
    | ABFrameResult
    | ABDetectionResult
    | ABClipResult
    | ABFailed
    | ABCompileFinished
)


@dataclass(frozen=True)
class _Request:
    center_seconds: float
    view: str
    side_a: ABSideConfig
    side_b: ABSideConfig
    base_settings: AppSettings
    projection: str
    generation: int


@dataclass(frozen=True)
class _CompileRequest:
    side: str
    checkpoint: Path
    base_settings: AppSettings
    generation: int


@dataclass(frozen=True)
class _Stop:
    pass


@dataclass(frozen=True)
class _Cancel:
    pass


class _SessionSlot:
    """Lazy per-side cache of the detector and the restoration session.

    Keyed independently so changing one side's checkpoint never tears down the
    other side. The session key extends ``video_session_key`` with the
    checkpoint path and the engines-disabled flag, which the shared key does
    not carry.
    """

    def __init__(self) -> None:
        self.detector = None
        self.detector_key: tuple | None = None
        self.session = None
        self.session_key: tuple | None = None
        self.tensorrt = False

    def release_session(self) -> None:
        if self.session is None:
            return
        device = self.session.device
        self.session.close()
        self.session = None
        self.session_key = None
        self.tensorrt = False
        release_session_memory(device)

    def release(self) -> None:
        if self.detector is not None and hasattr(self.detector, "close"):
            self.detector.close()
        self.detector = None
        self.detector_key = None
        self.release_session()


class ABCompareWorker:
    """Single background thread serving one A/B request at a time."""

    def __init__(
        self,
        path: str | Path,
        metadata: VideoMetadata,
        *,
        on_stopped: Callable[[], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self.metadata = metadata
        self._on_stopped = on_stopped
        self.events: queue.Queue[ABEvent] = queue.Queue()
        self._commands: queue.Queue[_Request | _Cancel | _Stop] = queue.Queue(maxsize=1)
        self._closed = threading.Event()
        self._generation = 0
        self._active_cancel: threading.Event | None = None
        self._cancel_lock = threading.Lock()
        self._slots = {"a": _SessionSlot(), "b": _SessionSlot()}
        self._thread = threading.Thread(
            target=self._run,
            name=f"ab-compare-{self.path.name}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def request(
        self,
        center_seconds: float,
        view: str,
        side_a: ABSideConfig,
        side_b: ABSideConfig,
        base_settings: AppSettings,
        projection: str,
    ) -> int:
        if view not in AB_VIEWS:
            raise ValueError(f"Unknown A/B view '{view}'. Valid views: {', '.join(AB_VIEWS)}")
        self._generation += 1
        self._cancel_active_pass()
        self._replace_command(
            _Request(
                center_seconds=max(0.0, float(center_seconds)),
                view=view,
                side_a=side_a,
                side_b=side_b,
                base_settings=base_settings,
                projection=projection,
                generation=self._generation,
            )
        )
        return self._generation

    def request_compile(
        self,
        side: str,
        checkpoint: Path,
        base_settings: AppSettings,
    ) -> int:
        """Queue a BasicVSR++ sub-engine compilation for one side's checkpoint.

        The compile subprocess (15-60 min) cannot be interrupted once started;
        the window locks its controls until ABCompileFinished arrives.
        """
        self._generation += 1
        self._cancel_active_pass()
        self._replace_command(
            _CompileRequest(
                side=side,
                checkpoint=Path(checkpoint),
                base_settings=base_settings,
                generation=self._generation,
            )
        )
        return self._generation

    def cancel(self) -> None:
        if self._closed.is_set():
            return
        self._generation += 1
        self._cancel_active_pass()
        self._replace_command(_Cancel())

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._replace_command(_Stop(), allow_closed=True)
        self._cancel_active_pass()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def _cancel_active_pass(self) -> None:
        with self._cancel_lock:
            if self._active_cancel is not None:
                self._active_cancel.set()

    def _replace_command(
        self,
        command: _Request | _Cancel | _Stop,
        *,
        allow_closed: bool = False,
    ) -> None:
        if self._closed.is_set() and not allow_closed:
            return
        try:
            while True:
                self._commands.get_nowait()
        except queue.Empty:
            pass
        try:
            self._commands.put_nowait(command)
        except queue.Full:
            pass

    def _superseded(self) -> bool:
        return not self._commands.empty() or self._closed.is_set()

    def _run(self) -> None:
        try:
            while not self._closed.is_set():
                try:
                    command = self._commands.get(timeout=0.1)
                except queue.Empty:
                    continue
                if isinstance(command, _Stop):
                    break
                if isinstance(command, _Cancel):
                    continue
                if isinstance(command, _CompileRequest):
                    self._compile(command)
                    continue
                self._process(command)
        finally:
            for slot in self._slots.values():
                slot.release()
            if "torch" in sys.modules:
                import torch

                if torch.cuda.is_available():
                    release_session_memory(torch.device("cuda:0"))
            if self._on_stopped is not None:
                self._on_stopped()

    def _process(self, request: _Request) -> None:
        for side in ("a", "b"):
            if self._superseded():
                return
            config = request.side_a if side == "a" else request.side_b
            try:
                self._run_side(side, config, request)
            except Exception as exc:
                if self._closed.is_set():
                    return
                if self._is_cuda_oom(exc):
                    # Both sessions resident may not fit (8K VR): free both
                    # slots and retry this leg once with a cold rebuild of only
                    # what it needs.
                    for slot in self._slots.values():
                        slot.release()
                    try:
                        self._run_side(side, config, request)
                        continue
                    except Exception as retry_exc:
                        if self._closed.is_set():
                            return
                        exc = retry_exc
                self.events.put(ABFailed(side, "error", str(exc), request.generation))

    def _compile(self, request: _CompileRequest) -> None:
        side = request.side
        generation = request.generation
        settings = request.base_settings
        self.events.put(ABStatus(side, "compiling", generation))
        try:
            from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled

            ensure_engines_compiled(
                EngineCompilationRequest(
                    device="cuda:0",
                    fp16=bool(settings.fp16_mode),
                    basicvsrpp=True,
                    basicvsrpp_model_path=str(request.checkpoint),
                ),
                log_callback=lambda message: self.events.put(
                    ABStatus(side, message, generation)
                ),
            )
        except Exception as exc:
            if not self._closed.is_set():
                self.events.put(
                    ABCompileFinished(side, ok=False, message=str(exc), generation=generation)
                )
            return
        from jasna.engine_paths import all_basicvsrpp_sub_engines_exist

        # The cached session key carries the engines-disabled flag, so the next
        # run of a side using this checkpoint rebuilds on the TRT path without
        # any explicit slot invalidation here.
        ok = all_basicvsrpp_sub_engines_exist(
            str(request.checkpoint), bool(settings.fp16_mode)
        )
        if not self._closed.is_set():
            self.events.put(
                ABCompileFinished(side, ok=ok, message="", generation=generation)
            )

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        if "torch" not in sys.modules:
            return False
        import torch

        return isinstance(exc, torch.cuda.OutOfMemoryError)

    def _run_side(self, side: str, config: ABSideConfig, request: _Request) -> None:
        generation = request.generation
        needs_restoration = request.view in ("restored", "clip")
        settings = replace(
            request.base_settings,
            detection_model=str(config.detection_model),
            detection_score_threshold=float(config.detection_score_threshold),
        )
        checkpoint = Path(config.restoration_model_path)
        is_seedvr2 = config.restoration_model_name == "seedvr2"

        if needs_restoration and not is_seedvr2:
            from jasna.restorer.checkpoint_info import checkpoint_has_ema_weights

            # load_model is hardwired to is_use_ema=True; a checkpoint without
            # generator_ema.* keys would run with random weights, so refuse it.
            # (The seedvr2 LoRA is not a BasicVSR++ checkpoint; skipped there.)
            if not checkpoint_has_ema_weights(checkpoint):
                self.events.put(
                    ABFailed(side, "checkpoint_missing_ema", checkpoint.name, generation)
                )
                return

        slot = self._slots[side]
        self._ensure_detector(side, slot, settings, generation)
        if self._superseded():
            return
        if needs_restoration:
            self._ensure_session(side, slot, settings, config, generation)
            if self._superseded():
                return
            self.events.put(
                ABSessionInfo(
                    side,
                    generation,
                    tensorrt=slot.tensorrt,
                    checkpoint_name=checkpoint.name,
                )
            )
            self._run_restoration_view(side, slot, request, settings)
        else:
            self._run_detection_view(side, slot, request, settings)

    @staticmethod
    def _detector_key(settings: AppSettings) -> tuple:
        return (
            str(settings.detection_model),
            float(settings.detection_score_threshold),
            int(settings.batch_size),
            bool(settings.fp16_mode),
        )

    def _ensure_detector(
        self,
        side: str,
        slot: _SessionSlot,
        settings: AppSettings,
        generation: int,
    ) -> None:
        key = self._detector_key(settings)
        if slot.detector is not None and slot.detector_key == key:
            return
        if slot.detector is not None and hasattr(slot.detector, "close"):
            slot.detector.close()
        slot.detector = None
        slot.detector_key = None
        self.events.put(ABStatus(side, "loading_models", generation))

        from jasna._suppress_noise import install as _install_noise_filters

        _install_noise_filters()
        import torch

        from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
        from jasna.mosaic.detection_registry import (
            build_detection_model,
            coerce_detection_model_name,
            require_detection_model_weights,
        )

        device = torch.device("cuda:0")
        det_name = coerce_detection_model_name(str(settings.detection_model))
        detection_model_path = require_detection_model_weights(det_name)
        ensure_engines_compiled(
            EngineCompilationRequest(
                device=str(device),
                fp16=bool(settings.fp16_mode),
                detection=True,
                detection_model_name=det_name,
                detection_model_path=str(detection_model_path),
                detection_batch_size=int(settings.batch_size),
            ),
            log_callback=lambda message: self.events.put(
                ABStatus(side, message, generation)
            ),
        )
        slot.detector = build_detection_model(
            det_name,
            detection_model_path,
            batch_size=int(settings.batch_size),
            device=device,
            score_threshold=float(settings.detection_score_threshold),
            fp16=bool(settings.fp16_mode),
        )
        slot.detector_key = key

    def _ensure_session(
        self,
        side: str,
        slot: _SessionSlot,
        settings: AppSettings,
        config: ABSideConfig,
        generation: int,
    ) -> None:
        checkpoint = Path(config.restoration_model_path)
        is_seedvr2 = config.restoration_model_name == "seedvr2"

        if is_seedvr2:
            # No BasicVSR++ engines are involved; ABSessionInfo(tensorrt=False)
            # is the normal state, not a warning.
            engines_disabled = False
        else:
            from jasna.engine_paths import all_basicvsrpp_sub_engines_exist

            # A checkpoint without current-generation sub-engines runs on PyTorch
            # instead of triggering a 15-60 min compile subprocess; the resulting
            # ABSessionInfo(tensorrt=False) drives the warning badge.
            engines_disabled = not all_basicvsrpp_sub_engines_exist(
                str(checkpoint), bool(settings.fp16_mode)
            )
        key = video_session_key(settings) + (
            config.restoration_model_name,
            str(checkpoint),
            engines_disabled,
            str(config.seedvr2_repo),
        )
        if slot.session is not None and slot.session_key == key:
            return
        slot.release_session()
        self.events.put(ABStatus(side, "loading_models", generation))
        slot.session = build_video_session(
            settings,
            disable_basicvsrpp_tensorrt=engines_disabled,
            log=lambda message: self.events.put(ABStatus(side, message, generation)),
            restoration_model_path=checkpoint,
            restoration_model_name=config.restoration_model_name,
            seedvr2_repo=str(config.seedvr2_repo),
        )
        slot.session_key = key
        slot.tensorrt = bool(
            getattr(slot.session.restoration_pipeline.restorer, "tensorrt_active", False)
        )

    def _run_restoration_view(
        self,
        side: str,
        slot: _SessionSlot,
        request: _Request,
        settings: AppSettings,
    ) -> None:
        generation = request.generation
        self.events.put(ABStatus(side, "restoring", generation))
        cancel_event = threading.Event()
        with self._cancel_lock:
            self._active_cancel = cancel_event
        try:
            result = run_restoration_pass(
                path=self.path,
                metadata=self.metadata,
                settings=settings,
                projection=request.projection,
                center_seconds=request.center_seconds,
                playback=(request.view == "clip"),
                session=slot.session,
                detection_model=slot.detector,
                max_size=AB_PREVIEW_MAX_SIZE,
                cancel_event=cancel_event,
                should_abort=self._superseded,
            )
        finally:
            with self._cancel_lock:
                self._active_cancel = None
        if result.superseded:
            return
        if request.view == "clip":
            self.events.put(ABClipResult(side, result.clip_frames or (), generation))
        elif result.frame_image is not None:
            self.events.put(
                ABFrameResult(side, result.frame_seconds, result.frame_image, generation)
            )
        else:
            self.events.put(
                ABFailed(side, "error", "No frame could be restored", generation)
            )

    def _run_detection_view(
        self,
        side: str,
        slot: _SessionSlot,
        request: _Request,
        settings: AppSettings,
    ) -> None:
        generation = request.generation
        self.events.put(ABStatus(side, "restoring", generation))

        import torch

        from jasna.gui.mosaic_scan import SCAN_MASK_HW
        from jasna.media.video_decoder import NvidiaVideoReader
        from jasna.vr180 import SbsDetectionAdapter, resolve_vr_mode

        metadata = self.metadata
        device = torch.device("cuda:0")
        batch_size = int(settings.batch_size)
        vr_resolution = resolve_vr_mode(
            settings.vr_mode,
            metadata,
            self.path,
            projection=request.projection,
        )
        detector = (
            SbsDetectionAdapter(slot.detector)
            if vr_resolution.is_sbs
            else slot.detector
        )
        reader = NvidiaVideoReader(str(self.path), batch_size, device, metadata)
        with reader:
            batch_and_pts = next(
                reader.frames(seek_ts=max(0.0, request.center_seconds)), None
            )
            if batch_and_pts is None:
                raise RuntimeError("Could not decode the requested preview frame")
            batch, pts_list = batch_and_pts
            if batch.shape[0] < batch_size:
                pad = batch[-1:].expand(batch_size - batch.shape[0], -1, -1, -1)
                batch = torch.cat((batch, pad))
            scores, masks = detector.scan_scores_masks(batch, mask_hw=SCAN_MASK_HW)
            seconds = max(
                0.0,
                (pts_list[0] - reader.start_pts) * float(metadata.time_base),
            )
            original = _frame_image(
                batch[0],
                AB_PREVIEW_MAX_SIZE,
                left_eye_only=vr_resolution.is_sbs,
            )
            mask = masks[0].to(torch.uint8).cpu()
            score = float(scores[0].cpu())
        if self._superseded():
            return
        self.events.put(
            ABDetectionResult(side, seconds, original, mask, score, generation)
        )

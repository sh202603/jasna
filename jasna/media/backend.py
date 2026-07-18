"""Video decode/encode backend selection.

This is the single place that knows about both the native backend (the
PyAV/NVDEC ``NvidiaVideoReader`` and PyAV/NVENC ``NvidiaVideoEncoder``) and the
optional ``torchcodec`` backend. Callers construct readers/encoders through the
factories here instead of instantiating the concrete classes directly, so the
backend can be chosen per request with automatic fallback to native.

Design constraints (see ``docs/TORCHCODEC_BACKEND_*.md``):

- The native path must stay byte-/perf-identical to before. For
  ``VideoBackend.NATIVE`` (and any request torchcodec cannot satisfy) the
  factories return the existing ``NvidiaVideoReader`` / ``NvidiaVideoEncoder``
  instance *directly* — no wrapper, no per-frame indirection.
- Selection runs once at construction; the decode/encode hot path is untouched.
- No torch / torchcodec import at module load — every heavy import is deferred
  into the factory bodies so the selection predicates stay cheap and
  unit-testable without a GPU installed.

torchcodec 0.15.0 capability limits that drive the predicates (unchanged since
0.14.0; re-verified on 0.15.0):

- Decode: covers essentially every input (jasna only needs 8-bit RGB for its
  models), so the decoder predicate is availability plus ``frame_stride == 1``
  (the fps-retarget stride is only implemented by the native reader).
- Encode: GPU NVENC is hardcoded to 8-bit ``nv12``, only ``hevc_nvenc`` /
  ``av1_nvenc`` are usable, no FFmpeg AVOptions passthrough beyond the mapped
  keys, and no color-metadata control (tags are applied by remux). The native
  encoder always outputs 10-bit for HEVC/AV1, which torchcodec cannot match, so
  torchcodec encode is never auto-selected: it runs only when forced via
  ``--encode-backend torchcodec``, and only for 8-bit sources.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from jasna.media import VideoMetadata

log = logging.getLogger(__name__)


class VideoBackend(str, Enum):
    """Which decode/encode backend a request should use.

    ``AUTO`` prefers torchcodec where it is available/eligible and falls back to
    native otherwise. ``NATIVE`` always uses the PyAV-based reader/encoder (the
    default, i.e. current behavior). ``TORCHCODEC`` forces torchcodec and errors
    loudly when it cannot satisfy the request.
    """

    AUTO = "auto"
    NATIVE = "native"
    TORCHCODEC = "torchcodec"


def coerce_backend(backend: "VideoBackend | str | None") -> VideoBackend:
    if backend is None:
        return VideoBackend.NATIVE
    if isinstance(backend, VideoBackend):
        return backend
    return VideoBackend(str(backend).lower())


@functools.lru_cache(maxsize=1)
def torchcodec_available() -> bool:
    """Whether the ``torchcodec`` package is importable.

    Uses ``find_spec`` only (cheap, no import, never raises). A package that is
    present but fails to *import* (e.g. the known Windows/cu130 DLL-load issue)
    is still reported available here; the real import/NVDEC failure is caught at
    construction time and, under ``AUTO``, falls back to native.
    """

    try:
        return importlib.util.find_spec("torchcodec") is not None
    except Exception:
        return False


# --- selection predicates ---------------------------------------------------


def select_decoder_backend(
    backend: "VideoBackend | str | None", *, metadata: "VideoMetadata | None" = None
) -> str:
    """Return ``"native"`` or ``"torchcodec"`` for the decoder.

    torchcodec decode covers all inputs jasna cares about, so the only gate is
    availability. ``metadata`` is accepted for symmetry / future codec gating.
    """

    b = coerce_backend(backend)
    if b is VideoBackend.NATIVE:
        return "native"
    if b is VideoBackend.TORCHCODEC:
        if not torchcodec_available():
            raise ValueError(
                "--video-backend torchcodec requested but the torchcodec package is not installed"
            )
        return "torchcodec"
    # AUTO
    return "torchcodec" if torchcodec_available() else "native"


# jasna encoder-setting key -> ffmpeg nvenc `extra_options` key. Validated against
# torchcodec 0.15.0 (which rejects rc/tune/b_ref_mode/multipass/tier). "preset" goes
# to the dedicated add_video param and "aq" expands to spatial-aq + aq-strength, so
# both are mappable but handled specially in the encoder (not via this dict).
NVENC_EXTRA_KEYMAP = {
    "cq": "cq",
    "qmin": "qmin",
    "qmax": "qmax",
    "nonrefp": "nonref_p",
    "gop": "g",
    "maxbitrate": "maxrate",
    "vbvbufsize": "bufsize",
    "temporalaq": "temporal-aq",
    "lookahead": "rc-lookahead",
}
NVENC_MAPPABLE_KEYS = set(NVENC_EXTRA_KEYMAP) | {"preset", "aq"}


def encoder_settings_mappable(settings: "dict[str, object]") -> bool:
    """Whether every key in ``settings`` maps to a torchcodec nvenc option."""
    return all(k in NVENC_MAPPABLE_KEYS for k in settings)


def torchcodec_encoder_eligibility(
    *,
    codec: str,
    metadata: "VideoMetadata",
    encoder_settings: "dict[str, object]",
    output_fps=None,
    mux_audio: bool = True,
    smart_fragment: bool = False,
) -> "tuple[bool, str]":
    """Pure predicate: can the torchcodec encoder satisfy this request?

    Returns ``(eligible, reason)``; ``reason`` explains the first failing gate
    (and is surfaced in the error when ``--encode-backend torchcodec`` is forced).
    """

    from av.video.reformatter import ColorRange as AvColorRange

    if smart_fragment or not mux_audio:
        return False, "--segments smart-render fragments require the native encoder"
    if str(codec).lower() not in ("hevc", "av1"):
        return False, f"codec {codec!r} unsupported (torchcodec GPU encode is HEVC or AV1)"
    if metadata.is_10bit:
        return False, (
            "10-bit sources require native (torchcodec GPU encode is 8-bit nv12 only; "
            "note the native encoder always outputs 10-bit HEVC/AV1)"
        )
    if not encoder_settings_mappable(encoder_settings):
        unmapped = sorted(k for k in encoder_settings if k not in NVENC_MAPPABLE_KEYS)
        return False, (
            f"--encoder-settings {unmapped} cannot be mapped to torchcodec nvenc options "
            "(rc/tuning_info/vbvinit/initqp/lookahead_level/tflevel); use native"
        )
    if metadata.color_range == AvColorRange.JPEG:
        return False, "full/JPEG range is not supported"
    if output_fps is not None and output_fps != metadata.video_fps_exact:
        return False, (
            "frame generation / fps retargeting require the native encoder "
            "(torchcodec encodes at the source frame rate only)"
        )
    # Colorspace (BT.601/709/2020) is NOT gated: color tags + the HEVC VUI rewrite
    # are applied by the remux step, identically to the modi-native behavior, so
    # torchcodec + remux preserves the source colorspace.
    return True, "ok"


def select_encoder_backend(
    backend: "VideoBackend | str | None",
    *,
    codec: str,
    metadata: "VideoMetadata",
    encoder_settings: "dict[str, object]",
    output_fps=None,
    mux_audio: bool = True,
    smart_fragment: bool = False,
    lut_path: "str | None" = None,  # accepted for parity; not an eligibility gate
) -> str:
    """Return ``"native"`` or ``"torchcodec"`` for the encoder.

    Since upstream v0.8.0 the native encoder always outputs 10-bit for HEVC/AV1,
    which torchcodec's 8-bit nv12 NVENC cannot match, so AUTO never picks
    torchcodec for encode (output parity with native would silently break).
    torchcodec encode runs only when forced.
    """

    b = coerce_backend(backend)
    if b is not VideoBackend.TORCHCODEC:
        return "native"

    if not torchcodec_available():
        raise ValueError(
            "--encode-backend torchcodec requested but the torchcodec package is not installed"
        )
    eligible, reason = torchcodec_encoder_eligibility(
        codec=codec,
        metadata=metadata,
        encoder_settings=encoder_settings,
        output_fps=output_fps,
        mux_audio=mux_audio,
        smart_fragment=smart_fragment,
    )
    if not eligible:
        raise ValueError(f"torchcodec encoder cannot satisfy this request: {reason}")
    return "torchcodec"


# --- factories --------------------------------------------------------------


class _FallbackVideoReader:
    """Context-manager adapter: try torchcodec, fall back to native on failure.

    Used only for AUTO. The fallback must happen at ``__enter__`` time because
    torchcodec's NVDEC init / CPU-fallback detection occurs when the decoder is
    opened and the first frame is probed, not at construction. ``frames`` and
    ``__exit__`` delegate to whichever reader was actually entered, so the
    per-frame hot path adds nothing.
    """

    def __init__(self, tc_factory, native_factory):
        self._tc_factory = tc_factory
        self._native_factory = native_factory
        self._active = None

    def __enter__(self):
        try:
            reader = self._tc_factory()
            entered = reader.__enter__()
            self._active = reader
            return entered
        except Exception:
            log.warning("torchcodec decoder failed; falling back to native", exc_info=True)
            reader = self._native_factory()
            entered = reader.__enter__()
            self._active = reader
            return entered

    def __exit__(self, exc_type, exc_value, traceback):
        if self._active is not None:
            return self._active.__exit__(exc_type, exc_value, traceback)
        return False


def make_video_reader(
    *,
    file: str,
    batch_size: int,
    device: "torch.device",
    metadata: "VideoMetadata",
    frame_stride: int = 1,
    backend: "VideoBackend | str | None" = VideoBackend.NATIVE,
):
    """Construct a video reader for ``backend`` with native fallback under AUTO.

    Returns an object with the ``NvidiaVideoReader`` contract: a context manager
    whose ``frames(seek_ts=None)`` yields ``(uint8[B,3,H,W] CUDA, list[int] PTS)``.
    ``NATIVE`` returns the existing reader directly (no wrapper).
    """

    b = coerce_backend(backend)

    def _native():
        from jasna.media.video_decoder import NvidiaVideoReader

        # Only pass frame_stride when active so reader fakes/mocks written
        # against the plain contract keep working.
        kwargs = {"frame_stride": frame_stride} if int(frame_stride) != 1 else {}
        return NvidiaVideoReader(
            file, batch_size=batch_size, device=device, metadata=metadata, **kwargs
        )

    if int(frame_stride) != 1:
        # The fps-retarget stride is only implemented by the native reader.
        if b is VideoBackend.TORCHCODEC:
            raise ValueError(
                "--decode-backend torchcodec does not support --retarget-high-fps "
                "(frame stride); use the native decode backend"
            )
        return _native()

    if select_decoder_backend(b, metadata=metadata) != "torchcodec":
        return _native()

    def _torchcodec():
        from jasna.media.torchcodec_decoder import TorchcodecVideoReader

        return TorchcodecVideoReader(file, batch_size=batch_size, device=device, metadata=metadata)

    if b is VideoBackend.TORCHCODEC:
        # Forced: any failure (import / NVDEC / CPU fallback) propagates.
        return _torchcodec()
    # AUTO: wrap so an __enter__-time failure falls back to native.
    return _FallbackVideoReader(_torchcodec, _native)


def make_video_encoder(
    *,
    file: str,
    device: "torch.device",
    metadata: "VideoMetadata",
    codec: str,
    encoder_settings: "dict[str, object]",
    lut_path=None,
    output_fps=None,
    mux_audio: bool = True,
    pts_origin: int = 0,
    match_input_bit_depth: bool = False,
    smart_fragment: bool = False,
    backend: "VideoBackend | str | None" = VideoBackend.NATIVE,
):
    """Construct a video encoder for ``backend``.

    Returns an object with the ``NvidiaVideoEncoder`` contract (context manager +
    ``encode(frame, pts)``). Parameters mirror ``NvidiaVideoEncoder``; torchcodec
    is used only when forced and eligible (see ``select_encoder_backend``).
    """

    b = coerce_backend(backend)
    selected = select_encoder_backend(
        b,
        codec=codec,
        metadata=metadata,
        encoder_settings=encoder_settings,
        output_fps=output_fps,
        mux_audio=mux_audio,
        smart_fragment=smart_fragment,
        lut_path=lut_path,
    )
    if selected == "torchcodec":
        from jasna.media.torchcodec_encoder import TorchcodecVideoEncoder

        return TorchcodecVideoEncoder(
            file,
            device=device,
            metadata=metadata,
            codec=codec,
            encoder_settings=encoder_settings,
            lut_path=lut_path,
            output_fps=output_fps,
            mux_audio=mux_audio,
        )

    from jasna.media.video_encoder import NvidiaVideoEncoder

    return NvidiaVideoEncoder(
        file,
        device=device,
        metadata=metadata,
        codec=codec,
        encoder_settings=encoder_settings,
        lut_path=lut_path,
        output_fps=output_fps,
        mux_audio=mux_audio,
        pts_origin=pts_origin,
        match_input_bit_depth=match_input_bit_depth,
        smart_fragment=smart_fragment,
    )

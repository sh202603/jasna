"""FlashVSR offline secondary restoration (3-phase, VRAM-disjoint subprocesses).

FlashVSR (one-step streaming diffusion 4x VSR) peaks at 12-16 GB VRAM by itself,
so it cannot co-reside with jasna's ~9 GB primary pipeline on a 16 GB card. This
module runs it *offline* in three separate processes whose peak VRAM never
overlaps in time:

    Phase 1 (jasna env, ~9 GB) -- decode + detect + primary restoration, then
        serialize each clip's ``PrimaryRestoreResult`` (256px crops + masks +
        geometry) to a persistent *bundle* on disk. blend/encode is throwaway.
    Phase 2 (FlashVSR env, 12-16 GB) -- upscale every clip's 256px crops to
        1024px with FlashVSR and write them back into the bundle.
    Phase 3 (jasna env, light) -- re-decode the source, re-assemble
        ``SecondaryRestoreResult`` from the bundle + FlashVSR crops, and
        blend + encode the final output.

Each phase is a fresh subprocess; process exit guarantees the VRAM release that
an in-process ``close()`` cannot (same rationale as the ``--compile-engines``
subprocess boundary). The bundle is persistent, so a failed run can be resumed
from the phase that failed.

The orchestrator ``run_flashvsr_offline`` is invoked from ``jasna.main`` when
``--secondary-restoration flashvsr`` is selected; it spawns Phase 1 and Phase 3
as ``jasna --flashvsr-phase {dump,reblend}`` subprocesses (dispatched in
``jasna.__main__`` before the multiprocessing PID guard, mirroring
``--compile-engines``) and Phase 2 as the standalone
``flashvsr_phase2_driver.py`` run under the FlashVSR virtualenv's Python.

Design notes: ``FLASHVSR_OFFLINE_DESIGN_ja.md``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from jasna._frozen import is_frozen
from jasna.restorer import bundled_script_path

if TYPE_CHECKING:  # pragma: no cover - typing only
    import argparse

    import torch

    from jasna.pipeline_items import PrimaryRestoreResult

logger = logging.getLogger(__name__)

BUNDLE_VERSION = 1

# FlashVSR tiny mode holds every latent frame in VRAM; the Phase-0 pilot measured
# ~13.5 GB @ 21 frames and near-OOM @ 65 frames on a 16 GB RTX 5080. Capping the
# primary clip length keeps every clip within tiny's budget so Phase 2 can return
# *lossless* tensors (tiny mode) rather than falling back to a lossy mp4 round-trip
# (tiny-long). This is the flashvsr-specific default for Phase 1's --max-clip-size.
DEFAULT_MAX_CLIP_FRAMES = 32


# ---------------------------------------------------------------------------
# Bundle format
#
# <bundle-dir>/
#   manifest.json                     written by Phase 1 after primary completes
#   clip_<track>_<start>.npz          written by Phase 1 (256px primary crops)
#   clip_<track>_<start>_fvsr.npz     written by Phase 2 (1024px FlashVSR crops)
#
# The clip npz holds `primary_u8` (T,3,256,256 uint8 RGB CHW), `masks_packed`
# (np.packbits of a (T,Hm,Wm) bool array), `mask_shape` (T,Hm,Wm), and `geom`
# (a JSON string with all per-frame geometry needed to re-blend). All fields are
# plain numpy/JSON so the Phase 2 driver can read/write the bundle without any
# jasna import (it runs under a different virtualenv).
# ---------------------------------------------------------------------------


def clip_key(track_id: int, start_frame: int) -> str:
    return f"clip_{int(track_id)}_{int(start_frame)}"


def clip_npz_path(bundle_dir: Path, key: str) -> Path:
    return Path(bundle_dir) / f"{key}.npz"


def fvsr_npz_path(bundle_dir: Path, key: str) -> Path:
    return Path(bundle_dir) / f"{key}_fvsr.npz"


def manifest_path(bundle_dir: Path) -> Path:
    return Path(bundle_dir) / "manifest.json"


def _geom_from_primary(pr: "PrimaryRestoreResult") -> dict[str, Any]:
    """Serialize the geometry (everything re-blend needs except pixels/masks)."""
    return {
        "track_id": int(pr.track_id),
        "start_frame": int(pr.start_frame),
        "frame_count": int(pr.frame_count),
        "frame_shape": [int(pr.frame_shape[0]), int(pr.frame_shape[1])],
        "keep_start": int(pr.keep_start),
        "keep_end": int(pr.keep_end),
        "enlarged_bboxes": [[int(v) for v in bb] for bb in pr.enlarged_bboxes],
        "crop_shapes": [[int(v) for v in cs] for cs in pr.crop_shapes],
        "pad_offsets": [[int(v) for v in po] for po in pr.pad_offsets],
        "resize_shapes": [[int(v) for v in rs] for rs in pr.resize_shapes],
        # JSON object keys must be strings; restored to int on read.
        "crossfade_weights": (
            {str(k): float(v) for k, v in pr.crossfade_weights.items()}
            if pr.crossfade_weights is not None
            else None
        ),
    }


def write_primary_clip(bundle_dir: Path, pr: "PrimaryRestoreResult") -> str:
    """Serialize one ``PrimaryRestoreResult`` (all T frames) into the bundle.

    Returns the clip key. Called from the Phase 1 dump hook on the primary
    thread. Dumps every frame (not keep-windowed) so Phase 2's FlashVSR keeps the
    clip's temporal context; the keep-window slice happens in Phase 3.
    """
    import torch  # local: keep module import light for the orchestrator

    key = clip_key(pr.track_id, pr.start_frame)

    primary = pr.primary_raw.detach()
    primary_u8 = (
        primary.float().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    )  # (T,3,256,256) RGB CHW

    # masks: list of (Hm,Wm) bool GPU tensors, one per frame. Stack -> packbits.
    if pr.masks:
        mask_stack = torch.stack([m.to("cpu", torch.bool) for m in pr.masks], dim=0)
        masks_np = mask_stack.numpy()  # (T,Hm,Wm) bool
    else:
        masks_np = np.zeros((0, 0, 0), dtype=bool)
    mask_shape = np.asarray(masks_np.shape, dtype=np.int64)
    masks_packed = np.packbits(masks_np) if masks_np.size else np.zeros((0,), dtype=np.uint8)

    geom = _geom_from_primary(pr)

    # Compressed: the 256px dump is the bulk of the persistent bundle and Phase 1
    # is not perf-critical (the 1024px Phase 2 output stays uncompressed for the
    # GPU loop's sake). ~1-2 GiB/video at 256px.
    np.savez_compressed(
        clip_npz_path(bundle_dir, key),
        primary_u8=primary_u8,
        masks_packed=masks_packed,
        mask_shape=mask_shape,
        geom=np.asarray(json.dumps(geom)),
    )
    return key


def read_clip_geom(bundle_dir: Path, key: str) -> dict[str, Any]:
    with np.load(clip_npz_path(bundle_dir, key), allow_pickle=False) as data:
        geom = json.loads(str(data["geom"]))
    # Restore int keys on crossfade_weights.
    cw = geom.get("crossfade_weights")
    if cw is not None:
        geom["crossfade_weights"] = {int(k): float(v) for k, v in cw.items()}
    return geom


def read_clip_masks(bundle_dir: Path, key: str) -> np.ndarray:
    """Return the (T,Hm,Wm) bool mask array for a clip."""
    with np.load(clip_npz_path(bundle_dir, key), allow_pickle=False) as data:
        mask_shape = tuple(int(v) for v in data["mask_shape"])
        packed = data["masks_packed"]
    if not mask_shape or int(np.prod(mask_shape)) == 0:
        return np.zeros(mask_shape if mask_shape else (0, 0, 0), dtype=bool)
    total = int(np.prod(mask_shape))
    return np.unpackbits(packed, count=total).astype(bool).reshape(mask_shape)


# ---------------------------------------------------------------------------
# Phase 1: dump
# ---------------------------------------------------------------------------

# Accumulates one manifest entry per clip during the hooked primary run.
_DUMPED_CLIPS: list[dict[str, Any]] = []


def _install_dump_hook(bundle_dir: Path) -> None:
    """Monkeypatch ``prepare_and_run_primary`` to serialize each clip.

    Mirrors the Phase-0 pilot dumper (``~/flashvsr-pilot/dump_primary_crops.py``)
    but writes the full bundle instead of PNGs. The hook returns the result
    unchanged so the (throwaway) rest of the pipeline runs normally.
    """
    import jasna.restorer.restoration_pipeline as rp

    Path(bundle_dir).mkdir(parents=True, exist_ok=True)
    _DUMPED_CLIPS.clear()
    orig = rp.RestorationPipeline.prepare_and_run_primary

    def patched(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        pr = orig(self, *args, **kwargs)
        try:
            write_primary_clip(bundle_dir, pr)
            _DUMPED_CLIPS.append(
                {
                    "key": clip_key(pr.track_id, pr.start_frame),
                    "track_id": int(pr.track_id),
                    "start_frame": int(pr.start_frame),
                    "frame_count": int(pr.frame_count),
                    "keep_start": int(pr.keep_start),
                    "keep_end": int(pr.keep_end),
                }
            )
        except Exception:
            # A dump failure must abort the run (the bundle would be incomplete);
            # re-raise so the Phase 1 subprocess exits non-zero.
            logger.exception("[flashvsr] failed to dump clip %s", clip_key(pr.track_id, pr.start_frame))
            raise
        return pr

    rp.RestorationPipeline.prepare_and_run_primary = patched
    logger.info("[flashvsr] Phase 1 dump hook installed -> %s", bundle_dir)


def _write_manifest(bundle_dir: Path, input_path: str) -> None:
    from jasna.media import get_video_meta_data

    meta = get_video_meta_data(str(input_path))
    manifest = {
        "version": BUNDLE_VERSION,
        "input": str(input_path),
        "fps": float(meta.video_fps),
        "frame_count": int(meta.num_frames),
        "total_frames": int(meta.num_frames),
        "clips": list(_DUMPED_CLIPS),
    }
    manifest_path(bundle_dir).write_text(json.dumps(manifest, indent=2))
    logger.info("[flashvsr] wrote manifest: %d clips -> %s", len(_DUMPED_CLIPS), manifest_path(bundle_dir))


def read_manifest(bundle_dir: Path) -> dict[str, Any]:
    return json.loads(manifest_path(bundle_dir).read_text())


def _run_phase_dump(cfg: dict[str, Any]) -> None:
    """Phase 1 entry (in the ``--flashvsr-phase dump`` subprocess).

    Installs the dump hook, then hands the (rewritten) argv to jasna's normal CLI
    so decode/detect/primary run through the fully-tested pipeline. The secondary
    restorer is forced to ``none`` and the output to a throwaway temp file.
    """
    bundle_dir = Path(cfg["bundle_dir"])
    _install_dump_hook(bundle_dir)

    sys.argv = ["jasna", *cfg["argv"]]
    from jasna.main import main as jasna_main

    jasna_main()
    _write_manifest(bundle_dir, cfg["input"])


# ---------------------------------------------------------------------------
# Phase 3: reblend
# ---------------------------------------------------------------------------


def _assemble_secondary_result(
    geom: dict[str, Any],
    restored_kept: list["torch.Tensor"],
    masks_kept: list["torch.Tensor"],
    device: "torch.device",
) -> "SecondaryRestoreResult":
    """Rebuild a ``SecondaryRestoreResult`` from bundle geometry + FlashVSR crops.

    Mirrors ``RestorationPipeline.build_secondary_result``: the geometry is sliced
    to the keep window ``[ks:ke]`` and the restored frames (already sliced to the
    same window by the caller) become ``restored_frames`` with
    ``clip_keep_offset=ks``. ``scale_offsets`` derives pad/resize from the 1024px
    frames at blend time, so the 4x crops need no geometry rewrite.
    """
    from jasna.pipeline_items import SecondaryRestoreResult

    frame_count = int(geom["frame_count"])
    ks = max(0, int(geom["keep_start"]))
    ke = min(frame_count, int(geom["keep_end"]))
    kept_count = ke - ks

    return SecondaryRestoreResult(
        track_id=int(geom["track_id"]),
        start_frame=int(geom["start_frame"]),
        frame_count=frame_count,
        frame_shape=(int(geom["frame_shape"][0]), int(geom["frame_shape"][1])),
        frame_device=device,
        masks=masks_kept,
        restored_frames=restored_kept,
        keep_start=0,
        keep_end=kept_count,
        crossfade_weights=geom.get("crossfade_weights"),
        enlarged_bboxes=[tuple(bb) for bb in geom["enlarged_bboxes"][ks:ke]],
        crop_shapes=[tuple(cs) for cs in geom["crop_shapes"][ks:ke]],
        pad_offsets=[tuple(po) for po in geom["pad_offsets"][ks:ke]],
        resize_shapes=[tuple(rs) for rs in geom["resize_shapes"][ks:ke]],
        clip_keep_offset=ks,
    )


def _kept_range(entry: dict[str, Any]) -> tuple[int, int]:
    """The [first, last+1) source-frame range a clip contributes to, from manifest
    fields alone (no pixels loaded). Matches ``_apply_blend``'s touched frames."""
    frame_count = int(entry["frame_count"])
    ks = max(0, int(entry["keep_start"]))
    ke = min(frame_count, int(entry["keep_end"]))
    start = int(entry["start_frame"]) + ks
    return start, start + (ke - ks)


def _plan_reblend(
    bundle_dir: Path,
) -> tuple[dict[int, set[int]], dict[int, list[str]]]:
    """Plan the re-blend from the manifest only (no pixel load).

    Returns ``(frame_to_tracks, loads_at)``: the frame->track_ids pending map, and
    ``loads_at[first_frame] = [clip keys]`` telling Phase 3 which clip crops to
    load the moment decode reaches that clip's first frame. Clips whose FlashVSR
    output is missing (or with an empty keep window) are skipped with a warning.
    """
    manifest = read_manifest(bundle_dir)
    frame_to_tracks: dict[int, set[int]] = {}
    loads_at: dict[int, list[str]] = {}

    for entry in manifest["clips"]:
        key = entry["key"]
        if not fvsr_npz_path(bundle_dir, key).exists():
            logger.warning("[flashvsr] missing FlashVSR output for %s; skipping clip", key)
            continue
        first, end = _kept_range(entry)
        if end <= first:
            continue
        for f in range(first, end):
            frame_to_tracks.setdefault(f, set()).add(int(entry["track_id"]))
        loads_at.setdefault(first, []).append(key)

    return frame_to_tracks, loads_at


def _load_clip_sr(
    bundle_dir: Path, key: str, device: "torch.device"
) -> "SecondaryRestoreResult | None":
    """Load one clip's ``SecondaryRestoreResult`` (crops + masks + geometry).

    Pixels stay on CPU; the blend moves each frame to the device on demand, so
    only the clips overlapping the current decode position are resident (Phase 3
    peak host-RAM is bounded by concurrent clips, not the whole video).
    """
    import torch

    geom = read_clip_geom(bundle_dir, key)
    frame_count = int(geom["frame_count"])
    ks = max(0, int(geom["keep_start"]))
    ke = min(frame_count, int(geom["keep_end"]))
    if ke <= ks:
        return None

    with np.load(fvsr_npz_path(bundle_dir, key), allow_pickle=False) as data:
        restored_u8 = data["restored_u8"]  # (T,3,1024,1024) uint8 RGB CHW
    # The blend indexes restored_frames[local_i] and masks[local_i] by the same
    # index, so a Phase-2 frame-count mismatch would silently corrupt (or crash)
    # the blend. Realign to exactly frame_count (Phase 2 already aligns to T; this
    # is defensive).
    n = int(restored_u8.shape[0])
    if n == 0:
        logger.warning("[flashvsr] %s: FlashVSR returned 0 frames; skipping clip", key)
        return None
    if n != frame_count:
        logger.warning(
            "[flashvsr] %s: FlashVSR returned %d frames, expected %d; realigning",
            key, n, frame_count,
        )
        if n < frame_count:
            pad = np.repeat(restored_u8[-1:], frame_count - n, axis=0)
            restored_u8 = np.concatenate([restored_u8, pad], axis=0)
        else:
            restored_u8 = restored_u8[:frame_count]
    restored_kept = list(torch.from_numpy(restored_u8[ks:ke]).unbind(0))

    masks_np = read_clip_masks(bundle_dir, key)  # (T,Hm,Wm) bool
    masks_kept = [torch.from_numpy(masks_np[i].copy()) for i in range(ks, ke)]

    return _assemble_secondary_result(geom, restored_kept, masks_kept, device)


def _run_phase_reblend(cfg: dict[str, Any]) -> None:
    """Phase 3 entry (in the ``--flashvsr-phase reblend`` subprocess).

    Re-decodes the source in decode order, blends the FlashVSR 1024px crops back
    in via ``BlendBuffer``, and encodes the final output.
    """
    import torch

    from jasna.blend_buffer import BlendBuffer
    from jasna.media import get_video_meta_data
    from jasna.media.backend import VideoBackend, make_video_encoder, make_video_reader

    bundle_dir = Path(cfg["bundle_dir"])
    device = torch.device(str(cfg["device"]))
    torch.cuda.set_device(device)

    input_path = str(cfg["input"])
    output_path = str(cfg["output"])
    metadata = get_video_meta_data(input_path)

    # Plan from the manifest (cheap), then stream each clip's crops in as decode
    # reaches it so host-RAM stays bounded to concurrent clips.
    frame_to_tracks, loads_at = _plan_reblend(bundle_dir)
    blend_buffer = BlendBuffer(device=device)
    for f, tracks in frame_to_tracks.items():
        blend_buffer.register_frame(f, tracks)

    decode_backend = VideoBackend(str(cfg.get("decode_backend", "native")))
    encode_backend = VideoBackend(str(cfg.get("encode_backend", "native")))

    encoder = make_video_encoder(
        file=output_path,
        device=device,
        metadata=metadata,
        codec=str(cfg["codec"]),
        encoder_settings=dict(cfg.get("encoder_settings", {})),
        lut_path=cfg.get("lut_path"),
        backend=encode_backend,
    )

    frames_encoded = 0
    with torch.inference_mode():
        with make_video_reader(
            file=input_path,
            batch_size=int(cfg.get("batch_size", 4)),
            device=device,
            metadata=metadata,
            backend=decode_backend,
        ) as reader, encoder as enc:
            frame_idx = 0
            for batch, pts_list in reader.frames():
                for i in range(len(pts_list)):
                    for key in loads_at.get(frame_idx, ()):  # load clips due at this frame
                        sr = _load_clip_sr(bundle_dir, key, device)
                        if sr is not None:
                            blend_buffer.add_result(sr)
                    original = batch[i]
                    blended = blend_buffer.blend_frame(frame_idx, original)
                    enc.encode(blended, int(pts_list[i]))
                    frame_idx += 1
                    frames_encoded += 1

    logger.info("[flashvsr] Phase 3 reblend complete: %d frames -> %s", frames_encoded, output_path)


# ---------------------------------------------------------------------------
# Subprocess dispatch (called from jasna.__main__ before the PID guard)
# ---------------------------------------------------------------------------


def run_phase_subprocess(phase: str, json_path: str) -> None:
    """Entry point for ``jasna --flashvsr-phase {dump,reblend} <json>``."""
    # Claim main-process status so any child guards in the reused pipeline behave.
    os.environ["JASNA_MAIN_PID"] = str(os.getpid())

    cfg = json.loads(Path(json_path).read_text())
    if phase == "dump":
        _run_phase_dump(cfg)
    elif phase == "reblend":
        _run_phase_reblend(cfg)
    else:
        raise ValueError(f"unknown --flashvsr-phase: {phase!r}")


# ---------------------------------------------------------------------------
# Orchestrator (Phase 1 -> 2 -> 3), invoked from jasna.main
# ---------------------------------------------------------------------------


def _rewrite_argv_for_dump(argv: list[str], temp_output: str) -> list[str]:
    """Rewrite the user's CLI argv for the Phase 1 primary-only run.

    Swaps ``--secondary-restoration`` to ``none`` (so Phase 1 does not recurse
    into flashvsr), redirects ``--output`` to a throwaway temp, and disables
    ``--frame-gen`` (frame generation belongs to the final Phase 3 encode). Every
    other flag the user passed is preserved verbatim.
    """
    out: list[str] = []
    i = 0
    n = len(argv)
    override_next: str | None = None
    while i < n:
        a = argv[i]
        if override_next is not None:
            # skip the original value; the replacement was already appended
            override_next = None
            i += 1
            continue
        if a == "--secondary-restoration":
            out += ["--secondary-restoration", "none"]
            override_next = "value"
            i += 1
            continue
        if a.startswith("--secondary-restoration="):
            out += ["--secondary-restoration=none"]
            i += 1
            continue
        if a == "--output":
            out += ["--output", temp_output]
            override_next = "value"
            i += 1
            continue
        if a.startswith("--output="):
            out += [f"--output={temp_output}"]
            i += 1
            continue
        if a == "--frame-gen":
            out += ["--frame-gen", "none"]
            override_next = "value"
            i += 1
            continue
        if a.startswith("--frame-gen="):
            out += ["--frame-gen=none"]
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _cap_max_clip_size(argv: list[str], cap: int) -> list[str]:
    """Ensure ``--max-clip-size`` in *argv* is at most *cap* (keeps clips within
    FlashVSR tiny-mode VRAM). Inserts the flag if absent."""
    out: list[str] = []
    i = 0
    seen = False
    while i < len(argv):
        a = argv[i]
        if a == "--max-clip-size" and i + 1 < len(argv):
            seen = True
            try:
                val = min(int(argv[i + 1]), cap)
            except ValueError:
                val = cap
            out += ["--max-clip-size", str(val)]
            i += 2
            continue
        if a.startswith("--max-clip-size="):
            seen = True
            try:
                val = min(int(a.split("=", 1)[1]), cap)
            except ValueError:
                val = cap
            out += [f"--max-clip-size={val}"]
            i += 1
            continue
        out.append(a)
        i += 1
    if not seen:
        out += ["--max-clip-size", str(cap)]
    return out


def default_flashvsr_python(repo: Path) -> Path:
    """Platform default for --flashvsr-python inside the FlashVSR checkout's uv venv."""
    if os.name == "nt":
        return repo / ".venv" / "Scripts" / "python.exe"
    return repo / ".venv" / "bin" / "python"


def _validate_flashvsr_args(args: "argparse.Namespace") -> tuple[Path, Path, Path, Path]:
    """Validate the flashvsr-specific inputs; return resolved paths."""
    from jasna.media.image_io import is_image_path

    if bool(getattr(args, "stream", False)):
        raise ValueError("--secondary-restoration flashvsr is file-output only (not compatible with --stream)")
    if args.input is None or args.output is None:
        raise ValueError("--secondary-restoration flashvsr requires --input and --output")
    if str(getattr(args, "frame_gen", "none")).lower() != "none":
        raise ValueError(
            "--secondary-restoration flashvsr does not support --frame-gen yet "
            "(run frame generation as a separate pass)"
        )
    if bool(getattr(args, "retarget_high_fps", False)):
        # Phase 1 would decode with the fps-retarget frame stride while Phase 3
        # re-reads every source frame, so the bundle's start_frame indices would
        # no longer match the reblend's frame counter.
        raise ValueError(
            "--secondary-restoration flashvsr does not support --retarget-high-fps"
        )
    if str(getattr(args, "segments", "") or "").strip():
        raise ValueError(
            "--secondary-restoration flashvsr does not support --segments smart rendering"
        )

    if not str(args.flashvsr_repo).strip():
        raise ValueError("--flashvsr-repo is required for --secondary-restoration flashvsr")
    repo = Path(str(args.flashvsr_repo)).expanduser()
    if not repo.is_dir():
        raise FileNotFoundError(f"--flashvsr-repo not found: {repo}")

    python_arg = str(args.flashvsr_python).strip()
    fv_python = Path(python_arg).expanduser() if python_arg else default_flashvsr_python(repo)
    if not fv_python.exists():
        raise FileNotFoundError(
            f"FlashVSR Python not found: {fv_python}. Pass --flashvsr-python. "
            "It must be a uv-managed standalone Python venv (system Python lacks the "
            "dev headers Triton JIT needs)."
        )

    model_arg = str(args.flashvsr_model_dir).strip()
    model_dir = Path(model_arg).expanduser() if model_arg else repo / "models" / "FlashVSR-v1.1"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"--flashvsr-model-dir not found: {model_dir}")

    input_path = Path(str(args.input)).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    if input_path.is_dir():
        raise ValueError("--secondary-restoration flashvsr does not support folder input")
    if is_image_path(input_path):
        raise ValueError("--secondary-restoration flashvsr is video-only (image input not supported)")

    # The Phase 3 reblend pastes crops onto plain source frames with no VR
    # projector, so any VR processing in Phase 1 would blend into the wrong
    # geometry. "auto" is only rejected when it actually detects VR content.
    vr_mode = str(getattr(args, "vr_mode", "off") or "off").strip().lower()
    if vr_mode in ("sbs", "sbs-fisheye"):
        raise ValueError(
            f"--secondary-restoration flashvsr does not support VR processing (--vr-mode {vr_mode})"
        )
    if vr_mode == "auto":
        from jasna.media import get_video_meta_data
        from jasna.vr180 import resolve_vr_mode

        resolution = resolve_vr_mode("auto", get_video_meta_data(str(input_path)), input_path)
        if resolution.resolved != "off":
            raise ValueError(
                "--secondary-restoration flashvsr does not support VR processing, but "
                f"--vr-mode auto detected VR content ({resolution.resolved}: {resolution.reason}). "
                "Pass --vr-mode off to force flat processing."
            )

    return repo, fv_python, model_dir, input_path


# ---------------------------------------------------------------------------
# Disk-space safeguards
#
# The bundle is dominated by Phase 2's uncompressed 1024px crops (~3 MiB/frame).
# The default bundle lands under the system temp dir, which on Linux is often
# tmpfs (RAM-backed) and small — a large bundle there fills /tmp / exhausts RAM.
# ---------------------------------------------------------------------------

_GIB = 1024 ** 3
_FVSR_BYTES_PER_FRAME = 3 * 1024 * 1024  # 1024x1024x3 uint8 (uncompressed)
# ~1024px fvsr (~3 MiB/frame x ~1.5 overlap) + tiny 256px dump, per mosaic frame.
_BYTES_PER_MOSAIC_FRAME_UPPER = int(4.6 * 1024 * 1024)


def _fstype(path: Path) -> str | None:
    """Best-effort Linux filesystem type for the mount containing *path*."""
    try:
        resolved = str(Path(path).resolve())
        best_mnt, best_type = "", None
        with open("/proc/mounts", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt, fstype = parts[1], parts[2]
                if resolved == mnt or resolved.startswith(mnt.rstrip("/") + "/"):
                    if len(mnt) >= len(best_mnt):
                        best_mnt, best_type = mnt, fstype
        return best_type
    except OSError:
        return None


def _preflight_bundle_disk(bundle_dir: Path, input_path: Path) -> None:
    """Warn (before Phase 1) if the bundle lands on a RAM-backed / tight filesystem.

    Mosaic coverage is unknown here, so this only reports free space + a
    whole-video-mosaic upper bound and flags tmpfs (the /tmp default footgun). The
    precise, fatal check runs after Phase 1 in ``_gate_phase2_disk``.
    """
    free = shutil.disk_usage(str(bundle_dir)).free
    fstype = _fstype(bundle_dir)
    try:
        from jasna.media import get_video_meta_data
        total_frames = int(get_video_meta_data(str(input_path)).num_frames)
    except Exception:
        total_frames = 0
    upper = total_frames * _BYTES_PER_MOSAIC_FRAME_UPPER

    logger.info(
        "[flashvsr] bundle fs=%s free=%.1f GiB; worst-case (all-mosaic) estimate=%.1f GiB",
        fstype or "?", free / _GIB, upper / _GIB,
    )
    if fstype == "tmpfs":
        print(
            f"WARNING: FlashVSR bundle dir {bundle_dir} is on tmpfs (RAM-backed, "
            f"{free / _GIB:.0f} GiB free). A large bundle will exhaust RAM. Pass "
            f"--flashvsr-bundle-dir <path on a real disk> for anything beyond a short clip."
        )
    if total_frames and upper > free:
        print(
            f"NOTE: if this video is mosaiced throughout, the bundle may reach ~{upper / _GIB:.0f} GiB "
            f"but only {free / _GIB:.0f} GiB is free at {bundle_dir}. Actual size scales with mosaic "
            f"coverage; the run stops before Phase 2 if space is short."
        )


def _gate_phase2_disk(bundle_dir: Path) -> None:
    """After Phase 1, before the expensive Phase 2: refuse to start if the exact
    remaining 1024px output won't fit. The bundle is kept, so the user can free
    space / move it to a bigger disk and resume."""
    manifest = read_manifest(bundle_dir)
    needed = sum(
        int(e["frame_count"]) * _FVSR_BYTES_PER_FRAME
        for e in manifest["clips"]
        if not fvsr_npz_path(bundle_dir, e["key"]).exists()
    )
    free = shutil.disk_usage(str(bundle_dir)).free
    logger.info(
        "[flashvsr] Phase 2 needs ~%.1f GiB of 1024px output; %.1f GiB free at %s",
        needed / _GIB, free / _GIB, bundle_dir,
    )
    if needed * 1.03 > free:  # 3% headroom for npz/filesystem overhead
        raise RuntimeError(
            f"Not enough disk for FlashVSR Phase 2: need ~{needed / _GIB:.0f} GiB of 1024px output "
            f"but only {free / _GIB:.0f} GiB is free at {bundle_dir}. Re-run with "
            f"--flashvsr-bundle-dir <path on a bigger disk> (the current bundle is kept, so "
            f"completed clips are reused on resume)."
        )


def run_flashvsr_offline(args: "argparse.Namespace") -> None:
    """Orchestrate the 3-phase offline FlashVSR run (called from ``jasna.main``).

    Spawns Phase 1 (dump) and Phase 3 (reblend) as ``jasna --flashvsr-phase``
    subprocesses and Phase 2 as the standalone driver under the FlashVSR Python.
    Each phase runs to completion before the next starts, so their peak VRAM is
    never live at the same time.
    """
    repo, fv_python, model_dir, input_path = _validate_flashvsr_args(args)
    output_path = Path(str(args.output)).expanduser()

    keep_bundle = bool(getattr(args, "flashvsr_keep_bundle", False))
    bundle_arg = str(getattr(args, "flashvsr_bundle_dir", "") or "").strip()
    if bundle_arg:
        bundle_dir = Path(bundle_arg).expanduser()
        bundle_dir.mkdir(parents=True, exist_ok=True)
        cleanup_bundle = False
    else:
        bundle_dir = Path(tempfile.mkdtemp(prefix="jasna_flashvsr_"))
        cleanup_bundle = not keep_bundle

    max_clip_frames = int(getattr(args, "flashvsr_max_clip_frames", DEFAULT_MAX_CLIP_FRAMES))
    logger.info("[flashvsr] bundle dir: %s (keep=%s)", bundle_dir, keep_bundle or not cleanup_bundle)

    success = False
    try:
        _preflight_bundle_disk(bundle_dir, input_path)
        _phase1_dump(args, bundle_dir, input_path, max_clip_frames)
        _gate_phase2_disk(bundle_dir)  # exact 1024px estimate now that clips are known
        _phase2_upscale(args, bundle_dir, repo, fv_python, model_dir)
        _phase3_reblend(args, bundle_dir, input_path, output_path)
        success = True
        print(f"FlashVSR offline restoration complete -> {output_path}")
    finally:
        # Only discard the bundle on success. On failure keep it (completed phases
        # are resumable) and tell the user how to resume from where it broke.
        if cleanup_bundle and success:
            shutil.rmtree(bundle_dir, ignore_errors=True)
        elif not success:
            print(
                f"FlashVSR run failed. Bundle kept for resume at: {bundle_dir}\n"
                f"  Re-run the same command with: --flashvsr-bundle-dir {bundle_dir}"
            )


def _jasna_phase_cmd(phase: str, cfg: dict[str, Any], bundle_dir: Path) -> list[str]:
    json_path = bundle_dir / f"phase_{phase}.json"
    json_path.write_text(json.dumps(cfg))
    # The frozen binary takes --flashvsr-phase directly (jasna/__main__.py handles
    # it before dispatch); "-m jasna" only exists for a real interpreter. Mirrors
    # the --compile-engines relaunch in engine_compiler.
    if is_frozen():
        return [sys.executable, "--flashvsr-phase", phase, str(json_path)]
    return [sys.executable, "-m", "jasna", "--flashvsr-phase", phase, str(json_path)]


def _run_checked(cmd: list[str], phase_name: str, env: dict[str, str] | None = None) -> None:
    logger.info("[flashvsr] %s: %s", phase_name, " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"FlashVSR {phase_name} failed (exit code {result.returncode})")


def _phase1_dump(args: "argparse.Namespace", bundle_dir: Path, input_path: Path, max_clip_frames: int) -> None:
    temp_output = str(bundle_dir / "phase1_throwaway.mkv")
    argv = _rewrite_argv_for_dump(sys.argv[1:], temp_output)
    argv = _cap_max_clip_size(argv, max_clip_frames)
    cfg = {"bundle_dir": str(bundle_dir), "input": str(input_path), "argv": argv}
    _run_checked(_jasna_phase_cmd("dump", cfg, bundle_dir), "Phase 1 (primary + dump)")
    # The throwaway encode output is not needed downstream.
    try:
        Path(temp_output).unlink(missing_ok=True)
    except OSError:
        pass


def _phase2_upscale(
    args: "argparse.Namespace",
    bundle_dir: Path,
    repo: Path,
    fv_python: Path,
    model_dir: Path,
) -> None:
    driver = bundled_script_path("flashvsr_phase2_driver.py")
    cmd = [
        str(fv_python),
        str(driver),
        "--bundle-dir", str(bundle_dir),
        "--repo", str(repo),
        "--model-dir", str(model_dir),
        "--version", str(getattr(args, "flashvsr_version", "11")),
        "--dtype", str(getattr(args, "flashvsr_dtype", "bf16")),
        "--device", str(args.device),
        "--scale", "4",
    ]
    if bool(getattr(args, "flashvsr_unload_dit", True)):
        cmd.append("--unload-dit")
    if bool(getattr(args, "flashvsr_tiled_vae", True)):
        cmd.append("--tiled-vae")
    # The FlashVSR venv must import its own repo, not inherit jasna's PYTHONPATH.
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if os.name == "nt":
        # FlashVSR prints a block-character banner at pipeline init; when stdout is
        # a pipe (GUI runs, output redirection) Windows defaults the child's text
        # layer to cp932 and the print raises UnicodeEncodeError before inference.
        env["PYTHONUTF8"] = "1"
    _run_checked(cmd, "Phase 2 (FlashVSR 4x)", env=env)


def _phase3_reblend(
    args: "argparse.Namespace",
    bundle_dir: Path,
    input_path: Path,
    output_path: Path,
) -> None:
    from jasna.media import parse_encoder_settings, validate_encoder_settings

    video_backend = str(getattr(args, "video_backend", "native")).lower()

    def _resolve(override: str) -> str:
        override = str(override).lower()
        return video_backend if override == "inherit" else override

    lut_arg = str(getattr(args, "lut", "") or "").strip()
    cfg = {
        "bundle_dir": str(bundle_dir),
        "input": str(input_path),
        "output": str(output_path),
        "device": str(args.device),
        "codec": str(args.codec).lower(),
        "encoder_settings": validate_encoder_settings(
            parse_encoder_settings(str(getattr(args, "encoder_settings", "") or "")),
            codec=str(args.codec).lower(),
        ),
        "lut_path": lut_arg or None,
        "batch_size": int(args.batch_size),
        "decode_backend": _resolve(str(getattr(args, "decode_backend", "inherit"))),
        "encode_backend": _resolve(str(getattr(args, "encode_backend", "inherit"))),
    }
    _run_checked(_jasna_phase_cmd("reblend", cfg, bundle_dir), "Phase 3 (reblend + encode)")


# ---------------------------------------------------------------------------
# CLI argument registration (called from jasna.main.build_parser)
# ---------------------------------------------------------------------------


def add_flashvsr_arguments(group: "argparse._ArgumentGroup") -> None:
    """Register the ``--flashvsr-*`` flags on the 2nd-restoration arg group."""
    import argparse

    group.add_argument(
        "--flashvsr-repo",
        type=str,
        default="",
        help="Path to the FlashVSR_plus checkout (required for --secondary-restoration flashvsr).",
    )
    group.add_argument(
        "--flashvsr-python",
        type=str,
        default="",
        help="Python for the FlashVSR env (default: <repo>/.venv/bin/python; on Windows "
             "<repo>/.venv/Scripts/python.exe). MUST be a uv-managed standalone Python venv; "
             "system Python lacks the dev headers Triton JIT needs.",
    )
    group.add_argument(
        "--flashvsr-model-dir",
        type=str,
        default="",
        help="FlashVSR weights dir (default: <repo>/models/FlashVSR-v1.1).",
    )
    group.add_argument(
        "--flashvsr-version",
        type=str,
        default="11",
        choices=["10", "11"],
        help="FlashVSR model version (default: %(default)s).",
    )
    group.add_argument(
        "--flashvsr-dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16"],
        help="FlashVSR compute dtype (default: %(default)s).",
    )
    group.add_argument(
        "--flashvsr-max-clip-frames",
        type=int,
        default=DEFAULT_MAX_CLIP_FRAMES,
        help="Cap Phase 1 --max-clip-size so each clip fits FlashVSR tiny-mode VRAM "
             "(default: %(default)s). Larger values risk OOM in Phase 2.",
    )
    group.add_argument(
        "--flashvsr-unload-dit",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Offload the FlashVSR DiT before VAE decode to save VRAM (default: %(default)s).",
    )
    group.add_argument(
        "--flashvsr-tiled-vae",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Tile the FlashVSR VAE decode to save VRAM (default: %(default)s).",
    )
    group.add_argument(
        "--flashvsr-tiles",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="flashvsr-inline only: number of full-width horizontal DiT strips per "
             "clip (tiled-dit) to cut FlashVSR peak VRAM on 16GB GPUs. 1 disables "
             "(default). Use the largest that fits VRAM: 2 (~1.25x slower) first, then "
             "3 or 4 (~1.5x) if it still OOMs. The offline path ignores this.",
    )
    group.add_argument(
        "--flashvsr-bundle-dir",
        type=str,
        default="",
        help="Persist the intermediate bundle here (default: a temp dir removed on completion). "
             "A persisted bundle lets a failed run resume from the phase that failed.",
    )
    group.add_argument(
        "--flashvsr-keep-bundle",
        action="store_true",
        help="Keep the bundle dir after completion (implied by --flashvsr-bundle-dir).",
    )

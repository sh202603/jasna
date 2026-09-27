"""FlashVSR offline secondary restoration (3-phase, VRAM-disjoint subprocesses).

FlashVSR (one-step streaming diffusion 4x VSR) peaks at 12-16 GB VRAM by itself,
so it cannot co-reside with jasna's ~9 GB primary pipeline on a 16 GB card. This
module runs it *offline* in three separate processes whose peak VRAM never
overlaps in time:

    Phase 1 (jasna env, ~9 GB) -- decode + detect + primary restoration, then
        serialize each clip's ``PrimaryRestoreResult`` (256px crops + masks +
        geometry) to a persistent *bundle* on disk. blend/encode is throwaway.
    Phase 2 (FlashVSR env, 12-16 GB) -- upscale every clip's 256px crops to
        256*scale px (``--flashvsr-scale``, 4 = 1024px) with FlashVSR, color-
        correct them, and write them back into the bundle.
    Phase 3 (jasna env, light) -- re-decode the source, re-assemble
        ``SecondaryRestoreResult`` from the bundle + FlashVSR crops, and
        blend + encode the final output.

Each phase is a fresh subprocess; process exit guarantees the VRAM release that
an in-process ``close()`` cannot (same rationale as the ``--compile-engines``
subprocess boundary). The bundle is persistent, so a failed run can be resumed
from the phase that failed.

The orchestrator ``run_flashvsr_offline`` is invoked from ``jasna.main`` when
``--secondary-restoration flashvsr`` (or ``swiftvr``) is selected; it spawns
Phase 1 and Phase 3 as ``jasna --flashvsr-phase {dump,reblend}`` subprocesses
(dispatched in ``jasna.__main__`` before the multiprocessing PID guard,
mirroring ``--compile-engines``) and Phase 2 as a standalone driver run under
the model's own virtualenv Python (``flashvsr_phase2_driver.py`` or
``swiftvr_phase2_driver.py``).

The SwiftVR engine reuses everything here except what ``OfflineEngine``
parametrises: the path resolution and flag names, the scale, Phase 1's clip
cap (FlashVSR only) and crop view window (SwiftVR only), and the Phase 2
command. The ``--flashvsr-phase`` hook name, the ``_fvsr.npz`` suffix and the
``[flashvsr]`` log prefix are internal names shared by both engines.

Design notes: ``FLASHVSR_OFFLINE_DESIGN_ja.md``, ``SWIFTVR_OFFLINE_DESIGN_ja.md``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from jasna._frozen import is_frozen
from jasna.restorer import bundled_script_path

if TYPE_CHECKING:  # pragma: no cover - typing only
    import argparse

    import torch

    from jasna.pipeline_items import PrimaryRestoreResult

logger = logging.getLogger(__name__)

# Bundle format version written to manifest.json. Version 2 adds the optional
# per-frame ``view_placements`` to the clip geometry (the smoothed crop view a
# geometry-sensitive engine such as SwiftVR sees, tracking.crop_view); Phase 3
# reads version 1 bundles too (no placements: the legacy blend path) and
# refuses newer versions, whose geometry it could not blend correctly.
BUNDLE_VERSION = 2

# Offline (Phase 2) only: tiny mode holds every latent frame, so Phase 1's clip
# length is capped. 90 (= the primary's default) measured flat vs 32 on a 16 GB
# card at both scales; longer is unmeasured. Inline (tiny-long, flat VRAM) does
# not cap.
DEFAULT_MAX_CLIP_FRAMES = 90


# ---------------------------------------------------------------------------
# Bundle format
#
# <bundle-dir>/
#   manifest.json                     written by Phase 1 after primary completes
#   clip_<track>_<start>.npz          written by Phase 1 (256px primary crops)
#   clip_<track>_<start>_fvsr.npz     written by Phase 2 ((256*scale)px FlashVSR crops)
#
# The clip npz holds `primary_u8` (T,3,256,256 uint8 RGB CHW), `masks_packed`
# (np.packbits of a (T,Hm,Wm) bool array), `mask_shape` (T,Hm,Wm), and `geom`
# (a JSON string with all per-frame geometry needed to re-blend; since version
# 2 also `view_placements`, T x 4 floats or null when the crops are in each
# frame's own grid). All fields are plain numpy/JSON so the Phase 2 driver can
# read/write the bundle without any jasna import (it runs under a different
# virtualenv).
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
        # Version 2: the placement of the smoothed crop view the engine sees
        # (None = own grid, blended through the legacy unpad + resize path).
        "view_placements": (
            [[float(v) for v in row] for row in pr.view_placements]
            if pr.view_placements is not None
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
    # is not perf-critical (the upscaled Phase 2 output stays uncompressed for the
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


def _install_dump_hook(bundle_dir: Path, view_window: int = 0) -> None:
    """Monkeypatch ``prepare_and_run_primary`` to serialize each clip.

    Mirrors the Phase-0 pilot dumper (``~/flashvsr-pilot/dump_primary_crops.py``)
    but writes the full bundle instead of PNGs. The hook returns the result
    unchanged so the (throwaway) rest of the pipeline runs normally.

    ``view_window`` > 1 makes Phase 1 hand the engine the smoothed crop view an
    inline run would (SwiftVR's ``--swiftvr-view-window``). The pipeline reads
    the window from its secondary restorer, which Phase 1 runs without, so the
    property is overridden to return the configured window; the same code as
    inline then builds the view and its placements, which are dumped with the
    geometry. The override is confined to this subprocess, like the hook.
    """
    import jasna.restorer.restoration_pipeline as rp

    Path(bundle_dir).mkdir(parents=True, exist_ok=True)
    _DUMPED_CLIPS.clear()
    if int(view_window) > 1:
        window = int(view_window)
        rp.RestorationPipeline.view_smoothing_window = property(lambda self: window)
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
    manifest = json.loads(manifest_path(bundle_dir).read_text())
    version = int(manifest.get("version", 1))
    if version > BUNDLE_VERSION:
        # A newer jasna may carry geometry this one cannot blend (version 1
        # would have composited version 2's view grids as own grids, silently).
        raise RuntimeError(
            f"bundle at {bundle_dir} is version {version}, newer than this jasna's "
            f"{BUNDLE_VERSION}; re-run with the jasna that wrote it or start a new bundle"
        )
    return manifest


def _run_phase_dump(cfg: dict[str, Any]) -> None:
    """Phase 1 entry (in the ``--flashvsr-phase dump`` subprocess).

    Installs the dump hook, then hands the (rewritten) argv to jasna's normal CLI
    so decode/detect/primary run through the fully-tested pipeline. The secondary
    restorer is forced to ``none`` and the output to a throwaway temp file.
    """
    bundle_dir = Path(cfg["bundle_dir"])
    _install_dump_hook(bundle_dir, int(cfg.get("view_window", 0)))

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
    ``clip_keep_offset=ks``. ``scale_offsets`` derives pad/resize from the restored
    frames' actual size at blend time, so the upscaled crops (1024px at
    ``--flashvsr-scale 4``, 512px at 2) need no geometry rewrite. A version 2
    bundle's ``view_placements`` (SwiftVR's smoothed crop view) are sliced the
    same way; the blend takes its view path only when they are set.
    """
    from jasna.pipeline_items import SecondaryRestoreResult

    frame_count = int(geom["frame_count"])
    ks = max(0, int(geom["keep_start"]))
    ke = min(frame_count, int(geom["keep_end"]))
    kept_count = ke - ks
    placements = geom.get("view_placements")
    view_placements = (
        [tuple(float(v) for v in row) for row in placements[ks:ke]]
        if placements is not None
        else None
    )

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
        view_placements=view_placements,
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
        restored_u8 = data["restored_u8"]  # (T,3,S,S) uint8 RGB CHW, S = 256*scale
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

    Re-decodes the source in decode order, blends the FlashVSR upscaled crops back
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


FLASHVSR_FORK_URL = "https://github.com/sh202603/FlashVSR_plus"


def apply_flashvsr_accel_env(env: dict, accel: bool, repo: Path) -> None:
    """Set the FlashVSR_plus fork's acceleration switch in a worker/driver env.

    The fork reads ``FLASHVSR_ACCEL=1`` (its ``--accel``) inside
    ``run.init_pipeline``, which both of our FlashVSR processes call, and falls
    back to the standard path per part on GPUs/checkouts that can't run it.
    ``--no-flashvsr-accel`` removes the variable so a value left in the shell
    can't override the flag; the per-part variables (``FLASHVSR_FP8_DIT=0``,
    ...) are left alone as verification overrides. An upstream checkout has no
    acceleration and would silently ignore the variable, so warn instead.
    """
    if not accel:
        env.pop("FLASHVSR_ACCEL", None)
        return
    env["FLASHVSR_ACCEL"] = "1"
    if not (repo / "vsrlib" / "accel.py").is_file():
        logger.warning(
            "[flashvsr] --flashvsr-accel needs the FlashVSR_plus fork (%s); the checkout at "
            "%s has no acceleration, so FlashVSR runs at standard speed.",
            FLASHVSR_FORK_URL, repo,
        )


def _validate_offline_flags(args: "argparse.Namespace", mode: str) -> None:
    """The engine-independent constraints of the offline 3-phase path (no I/O).

    ``mode`` is the ``--secondary-restoration`` value, for the messages.
    """
    flag = f"--secondary-restoration {mode}"
    if bool(getattr(args, "stream", False)):
        raise ValueError(f"{flag} is file-output only (not compatible with --stream)")
    if args.input is None or args.output is None:
        raise ValueError(f"{flag} requires --input and --output")
    if str(getattr(args, "frame_gen", "none")).lower() != "none":
        raise ValueError(
            f"{flag} does not support --frame-gen yet (run frame generation as a separate pass)"
        )
    if bool(getattr(args, "retarget_high_fps", False)):
        # Phase 1 would decode with the fps-retarget frame stride while Phase 3
        # re-reads every source frame, so the bundle's start_frame indices would
        # no longer match the reblend's frame counter.
        raise ValueError(f"{flag} does not support --retarget-high-fps")
    if str(getattr(args, "segments", "") or "").strip():
        raise ValueError(f"{flag} does not support --segments smart rendering")


def _validate_offline_input(args: "argparse.Namespace", mode: str) -> Path:
    """The engine-independent input constraints; returns the input path."""
    from jasna.media.image_io import is_image_path

    flag = f"--secondary-restoration {mode}"
    input_path = Path(str(args.input)).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    if input_path.is_dir():
        raise ValueError(f"{flag} does not support folder input")
    if is_image_path(input_path):
        raise ValueError(f"{flag} is video-only (image input not supported)")

    # The Phase 3 reblend pastes crops onto plain source frames with no VR
    # projector, so any VR processing in Phase 1 would blend into the wrong
    # geometry. "auto" is only rejected when it actually detects VR content.
    vr_mode = str(getattr(args, "vr_mode", "off") or "off").strip().lower()
    if vr_mode in ("sbs", "sbs-fisheye"):
        raise ValueError(f"{flag} does not support VR processing (--vr-mode {vr_mode})")
    if vr_mode == "auto":
        from jasna.media import get_video_meta_data
        from jasna.vr180 import resolve_vr_mode

        resolution = resolve_vr_mode("auto", get_video_meta_data(str(input_path)), input_path)
        if resolution.resolved != "off":
            raise ValueError(
                f"{flag} does not support VR processing, but "
                f"--vr-mode auto detected VR content ({resolution.resolved}: {resolution.reason}). "
                "Pass --vr-mode off to force flat processing."
            )
    return input_path


def _resolve_flashvsr_paths(args: "argparse.Namespace") -> tuple[Path, Path, Path]:
    """Validate the ``--flashvsr-*`` paths; return ``(repo, python, model_dir)``."""
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
            "Its base Python must ship the dev headers the Triton JIT needs (a "
            "uv-managed standalone Python, or a system Python with its -dev package)."
        )

    model_arg = str(args.flashvsr_model_dir).strip()
    model_dir = Path(model_arg).expanduser() if model_arg else repo / "models" / "FlashVSR-v1.1"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"--flashvsr-model-dir not found: {model_dir}")
    return repo, fv_python, model_dir


def _validate_flashvsr_args(args: "argparse.Namespace") -> tuple[Path, Path, Path, Path]:
    """Validate the flashvsr-specific inputs; return resolved paths."""
    _validate_offline_flags(args, "flashvsr")
    repo, fv_python, model_dir = _resolve_flashvsr_paths(args)
    input_path = _validate_offline_input(args, "flashvsr")
    return repo, fv_python, model_dir, input_path


# ---------------------------------------------------------------------------
# Disk-space safeguards
#
# The bundle is dominated by Phase 2's uncompressed upscaled crops (~3 MiB/frame
# at --flashvsr-scale 4, a quarter of that at 2).
# The default bundle lands under the system temp dir, which on Linux is often
# tmpfs (RAM-backed) and small — a large bundle there fills /tmp / exhausts RAM.
# ---------------------------------------------------------------------------

_GIB = 1024 ** 3
_MIB = 1024 ** 2


def _fvsr_bytes_per_frame(scale: int) -> int:
    """One uncompressed (256*scale)^2 x3 uint8 Phase 2 frame: 3 MiB at 4x, 0.75 MiB at 2x."""
    return 3 * (256 * int(scale)) ** 2


def _bytes_per_mosaic_frame_upper(scale: int) -> int:
    """Per mosaic frame upper bound: fvsr output x ~1.5 clip overlap + the tiny
    256px dump (~0.1 MiB). 4.6 MiB at 4x."""
    return int(1.5 * _fvsr_bytes_per_frame(scale) + 0.1 * _MIB)


# The scale-4 values (kept as module constants: referenced by tests).
_FVSR_BYTES_PER_FRAME = _fvsr_bytes_per_frame(4)
_BYTES_PER_MOSAIC_FRAME_UPPER = _bytes_per_mosaic_frame_upper(4)


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


def _preflight_bundle_disk(
    bundle_dir: Path,
    input_path: Path,
    scale: int = 4,
    *,
    display: str = "FlashVSR",
    bundle_dir_flag: str = "--flashvsr-bundle-dir",
) -> None:
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
    upper = total_frames * _bytes_per_mosaic_frame_upper(scale)

    logger.info(
        "[flashvsr] bundle fs=%s free=%.1f GiB; worst-case (all-mosaic) estimate=%.1f GiB",
        fstype or "?", free / _GIB, upper / _GIB,
    )
    if fstype == "tmpfs":
        print(
            f"WARNING: {display} bundle dir {bundle_dir} is on tmpfs (RAM-backed, "
            f"{free / _GIB:.0f} GiB free). A large bundle will exhaust RAM. Pass "
            f"{bundle_dir_flag} <path on a real disk> for anything beyond a short clip."
        )
    if total_frames and upper > free:
        print(
            f"NOTE: if this video is mosaiced throughout, the bundle may reach ~{upper / _GIB:.0f} GiB "
            f"but only {free / _GIB:.0f} GiB is free at {bundle_dir}. Actual size scales with mosaic "
            f"coverage; the run stops before Phase 2 if space is short."
        )


def _gate_phase2_disk(
    bundle_dir: Path,
    scale: int = 4,
    *,
    display: str = "FlashVSR",
    bundle_dir_flag: str = "--flashvsr-bundle-dir",
) -> None:
    """After Phase 1, before the expensive Phase 2: refuse to start if the exact
    remaining (256*scale)px output won't fit. The bundle is kept, so the user
    can free space / move it to a bigger disk and resume."""
    manifest = read_manifest(bundle_dir)
    needed = sum(
        int(e["frame_count"]) * _fvsr_bytes_per_frame(scale)
        for e in manifest["clips"]
        if not fvsr_npz_path(bundle_dir, e["key"]).exists()
    )
    free = shutil.disk_usage(str(bundle_dir)).free
    logger.info(
        "[flashvsr] Phase 2 needs ~%.1f GiB of %dpx output; %.1f GiB free at %s",
        needed / _GIB, 256 * int(scale), free / _GIB, bundle_dir,
    )
    if needed * 1.03 > free:  # 3% headroom for npz/filesystem overhead
        raise RuntimeError(
            f"Not enough disk for {display} Phase 2: need ~{needed / _GIB:.0f} GiB of "
            f"{256 * int(scale)}px output "
            f"but only {free / _GIB:.0f} GiB is free at {bundle_dir}. Re-run with "
            f"{bundle_dir_flag} <path on a bigger disk> (the current bundle is kept, so "
            f"completed clips are reused on resume)."
        )


@dataclass(frozen=True)
class OfflineEngine:
    """What differs between the offline engines (FlashVSR, SwiftVR).

    Everything else in the 3-phase path is shared: Phase 1's argv rewrite and
    dump hook, the bundle format, Phase 3, and the disk checks.
    """

    name: str                    # the --secondary-restoration value
    display: str                 # user-facing name in messages
    bundle_dir_flag: str         # --<engine>-bundle-dir, quoted in resume hints
    bundle_dir: str              # that flag's value ("" = a temp dir)
    keep_bundle: bool
    scale: int                   # Phase 2 writes (256*scale)px crops
    max_clip_frames: int | None  # cap on Phase 1's --max-clip-size (None = none)
    view_window: int             # Phase 1's crop view window (0 = own grid)
    input_path: Path
    run_phase2: Callable[[Path], None]  # bundle_dir -> runs the Phase 2 driver


def _flashvsr_engine(args: "argparse.Namespace") -> OfflineEngine:
    _validate_offline_flags(args, "flashvsr")
    repo, fv_python, model_dir = _resolve_flashvsr_paths(args)
    input_path = _validate_offline_input(args, "flashvsr")
    return OfflineEngine(
        name="flashvsr",
        display="FlashVSR",
        bundle_dir_flag="--flashvsr-bundle-dir",
        bundle_dir=str(getattr(args, "flashvsr_bundle_dir", "") or ""),
        keep_bundle=bool(getattr(args, "flashvsr_keep_bundle", False)),
        scale=int(getattr(args, "flashvsr_scale", 4)),
        # tiny mode holds every latent frame, so the clip length is capped.
        max_clip_frames=int(getattr(args, "flashvsr_max_clip_frames", DEFAULT_MAX_CLIP_FRAMES)),
        view_window=0,
        input_path=input_path,
        run_phase2=lambda bundle_dir: _phase2_upscale(args, bundle_dir, repo, fv_python, model_dir),
    )


def _swiftvr_engine(args: "argparse.Namespace") -> OfflineEngine:
    from jasna.restorer.swiftvr_common import resolve_swiftvr_paths, swiftvr_phase2_command

    _validate_offline_flags(args, "swiftvr")
    repo, sv_python, model_dir = resolve_swiftvr_paths(
        str(getattr(args, "swiftvr_repo", "") or ""),
        str(getattr(args, "swiftvr_python", "") or ""),
        str(getattr(args, "swiftvr_model_dir", "") or ""),
        mode="swiftvr",
    )
    input_path = _validate_offline_input(args, "swiftvr")
    scale = int(getattr(args, "swiftvr_scale", 4))

    def run_phase2(bundle_dir: Path) -> None:
        cmd, env = swiftvr_phase2_command(args, bundle_dir, repo, sv_python, model_dir)
        _run_checked(cmd, f"Phase 2 (SwiftVR {scale}x)", env=env, display="SwiftVR")

    return OfflineEngine(
        name="swiftvr",
        display="SwiftVR",
        bundle_dir_flag="--swiftvr-bundle-dir",
        bundle_dir=str(getattr(args, "swiftvr_bundle_dir", "") or ""),
        keep_bundle=bool(getattr(args, "swiftvr_keep_bundle", False)),
        scale=scale,
        # SwiftVR restores in fixed causal chunks: VRAM is flat in the clip length.
        max_clip_frames=None,
        # The same smoothed crop view the inline restorer asks for.
        view_window=int(getattr(args, "swiftvr_view_window", 0)),
        input_path=input_path,
        run_phase2=run_phase2,
    )


_ENGINES: dict[str, Callable[["argparse.Namespace"], OfflineEngine]] = {
    "flashvsr": _flashvsr_engine,
    "swiftvr": _swiftvr_engine,
}


def run_flashvsr_offline(args: "argparse.Namespace", engine: str = "flashvsr") -> None:
    """Orchestrate the 3-phase offline run (called from ``jasna.main``).

    Spawns Phase 1 (dump) and Phase 3 (reblend) as ``jasna --flashvsr-phase``
    subprocesses and Phase 2 as the engine's standalone driver under its own
    Python. Each phase runs to completion before the next starts, so their peak
    VRAM is never live at the same time. ``engine`` selects FlashVSR or SwiftVR.
    """
    if engine not in _ENGINES:
        raise ValueError(f"unknown offline engine: {engine!r}")
    eng = _ENGINES[engine](args)
    output_path = Path(str(args.output)).expanduser()

    if eng.bundle_dir.strip():
        bundle_dir = Path(eng.bundle_dir).expanduser()
        bundle_dir.mkdir(parents=True, exist_ok=True)
        cleanup_bundle = False
    else:
        bundle_dir = Path(tempfile.mkdtemp(prefix=f"jasna_{eng.name}_"))
        cleanup_bundle = not eng.keep_bundle

    logger.info("[flashvsr] %s bundle dir: %s (keep=%s)", eng.display, bundle_dir,
                eng.keep_bundle or not cleanup_bundle)

    success = False
    try:
        _preflight_bundle_disk(bundle_dir, eng.input_path, eng.scale,
                               display=eng.display, bundle_dir_flag=eng.bundle_dir_flag)
        _phase1_dump(bundle_dir, eng.input_path, max_clip_frames=eng.max_clip_frames,
                     view_window=eng.view_window, display=eng.display)
        # exact output estimate now that clips are known
        _gate_phase2_disk(bundle_dir, eng.scale, display=eng.display, bundle_dir_flag=eng.bundle_dir_flag)
        eng.run_phase2(bundle_dir)
        _phase3_reblend(args, bundle_dir, eng.input_path, output_path, display=eng.display)
        success = True
        print(f"{eng.display} offline restoration complete -> {output_path}")
    finally:
        # Only discard the bundle on success. On failure keep it (completed phases
        # are resumable) and tell the user how to resume from where it broke.
        if cleanup_bundle and success:
            shutil.rmtree(bundle_dir, ignore_errors=True)
        elif not success:
            print(
                f"{eng.display} run failed. Bundle kept for resume at: {bundle_dir}\n"
                f"  Re-run the same command with: {eng.bundle_dir_flag} {bundle_dir}"
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


def _run_checked(
    cmd: list[str], phase_name: str, env: dict[str, str] | None = None, *, display: str = "FlashVSR"
) -> None:
    logger.info("[flashvsr] %s: %s", phase_name, " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{display} {phase_name} failed (exit code {result.returncode})")


def _phase1_dump(
    bundle_dir: Path,
    input_path: Path,
    *,
    max_clip_frames: int | None,
    view_window: int = 0,
    display: str = "FlashVSR",
) -> None:
    temp_output = str(bundle_dir / "phase1_throwaway.mkv")
    argv = _rewrite_argv_for_dump(sys.argv[1:], temp_output)
    if max_clip_frames is not None:
        argv = _cap_max_clip_size(argv, max_clip_frames)
    cfg = {
        "bundle_dir": str(bundle_dir),
        "input": str(input_path),
        "argv": argv,
        "view_window": int(view_window),
    }
    _run_checked(_jasna_phase_cmd("dump", cfg, bundle_dir), "Phase 1 (primary + dump)", display=display)
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
        "--scale", str(int(getattr(args, "flashvsr_scale", 4))),
    ]
    if bool(getattr(args, "flashvsr_unload_dit", True)):
        cmd.append("--unload-dit")
    if bool(getattr(args, "flashvsr_tiled_vae", True)):
        cmd.append("--tiled-vae")
    # Color correction is always on (driver default: wavelet).
    # JASNA_FLASHVSR_COLOR_FIX is a verification-only override (adain|wavelet|
    # none) for A/B runs — deliberately an env var, not a CLI flag; same
    # contract as the inline restorer.
    color_fix = os.environ.get("JASNA_FLASHVSR_COLOR_FIX")
    if color_fix:
        if color_fix not in ("adain", "wavelet", "none"):
            raise ValueError(
                "[flashvsr] JASNA_FLASHVSR_COLOR_FIX must be adain|wavelet|none, "
                f"got {color_fix!r}"
            )
        cmd += ["--color-fix-method", color_fix]
    # The FlashVSR venv must import its own repo, not inherit jasna's PYTHONPATH.
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if os.name == "nt":
        # FlashVSR prints a block-character banner at pipeline init; when stdout is
        # a pipe (GUI runs, output redirection) Windows defaults the child's text
        # layer to cp932 and the print raises UnicodeEncodeError before inference.
        env["PYTHONUTF8"] = "1"
    apply_flashvsr_accel_env(env, bool(getattr(args, "flashvsr_accel", False)), repo)
    _run_checked(cmd, f"Phase 2 (FlashVSR {int(getattr(args, 'flashvsr_scale', 4))}x)", env=env)


def _phase3_reblend(
    args: "argparse.Namespace",
    bundle_dir: Path,
    input_path: Path,
    output_path: Path,
    *,
    display: str = "FlashVSR",
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
    _run_checked(_jasna_phase_cmd("reblend", cfg, bundle_dir), "Phase 3 (reblend + encode)", display=display)


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
        help="Path to the FlashVSR_plus checkout (required for --secondary-restoration flashvsr "
             f"and flashvsr-inline). Use the fork {FLASHVSR_FORK_URL}: it includes the tiny-long "
             "fix flashvsr-inline needs and the --flashvsr-accel speed-up.",
    )
    group.add_argument(
        "--flashvsr-python",
        type=str,
        default="",
        help="Python for the FlashVSR env (default: <repo>/.venv/bin/python; on Windows "
             "<repo>/.venv/Scripts/python.exe). Its base Python MUST ship the dev headers "
             "the Triton JIT needs: a uv-managed standalone Python, or a system Python "
             "with its -dev package.",
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
             "(default: %(default)s, measured flat up to there on 16 GB). Values above "
             "the default are unmeasured and may OOM in Phase 2.",
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
        "--flashvsr-scale",
        type=int,
        default=4,
        choices=[2, 4],
        help="Processing scale for both FlashVSR modes (default: %(default)s). 4 is "
             "model-native (256px crops processed at 1024px); 2 processes at 512px, "
             "roughly 5x faster with a few GB lower peak VRAM (the model is 4x-trained, "
             "so 2 is an opt-in). The output video resolution is unchanged either way: "
             "the blend shrink-composites the crops back onto the frame.",
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
        "--flashvsr-accel",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Both FlashVSR modes: speed FlashVSR up ~1.4x with FP8 convolutions/linears "
             "and fused DiT kernels, and lower its peak VRAM ~1.4 GiB (default: %(default)s). "
             "Needs the FlashVSR_plus fork, an RTX 40 series or newer GPU, "
             "--flashvsr-version 11 and --flashvsr-dtype bf16; anything else falls back "
             "to the standard path automatically. The output differs slightly from a "
             "run without it.",
    )
    group.add_argument(
        "--flashvsr-lora",
        type=str,
        default="",
        help="flashvsr-inline only: a Lada LoRA for the FlashVSR DiT, as a path or a bare file "
             "name looked up in the model_weights directory (e.g. "
             "lada_flashvsr_secondary_lora_v1.pt from huggingface.co/sh202603/lada-seedvr2-lora). "
             "Applied as bf16 adapters when the worker starts (a few percent slower, +32 MB "
             "VRAM); combines with --flashvsr-accel. The published LoRA tones down FlashVSR's "
             "mid-frequency over-sharpening and slightly reduces flicker at the cost of some "
             "fine grain; validated with --flashvsr-scale 2. Default: none (base FlashVSR).",
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

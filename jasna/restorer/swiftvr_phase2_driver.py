#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 sh202603
# SPDX-License-Identifier: AGPL-3.0
"""SwiftVR offline Phase 2 driver — runs under the SwiftVR virtualenv.

Reads a jasna offline *bundle* (written by Phase 1 of ``--secondary-restoration
swiftvr``), restores every clip's 256px primary crops to 256*scale px with
``SwiftVRPipeline.restore_clip()`` (``--scale`` 4 = the model-native 1024px,
2 = 512px), color-corrects them against the bicubic-upscaled input, and writes
the results back into the bundle for Phase 3 to re-blend.

This script is intentionally free of any ``jasna`` (or ``lada``) import: it is
executed by the SwiftVR env's Python (``--swiftvr-python``), not jasna's. It
uses numpy/torch plus the ``swiftvr`` package, and shares its model handling
with the inline worker (``swiftvr_inline_worker.py``, loaded by path from the
same directory; its top level is stdlib-only): the acceleration decision
(``_decide_accel``), the model load and warmup, the checked ``restore_clip()``
call and the GPU color correction. An offline clip therefore goes through the
same functions as an inline one; only the transport differs (bundle files
instead of the BGR wire, so there is no channel flip here: the bundle is RGB).

Invocation (by ``jasna.restorer.swiftvr_common.swiftvr_phase2_command``):

    <swiftvr-python> swiftvr_phase2_driver.py \\
        --bundle-dir <dir> --repo <SwiftVR> --model-dir <checkpoints> \\
        --device cuda:0 --scale 4 [--fp8-dit --torch-compile] \\
        [--color-fix-method wavelet]

Bundle contract (numpy/JSON, see ``flashvsr_offline.py``):
  in : manifest.json, clip_<track>_<start>.npz  (``primary_u8`` = (T,3,256,256) uint8 RGB CHW)
  out: clip_<track>_<start>_fvsr.npz            (``restored_u8`` = (T,3,S,S) uint8 RGB CHW, S = 256*scale)

Unlike the inline worker, bf16 is a regular path here: the SwiftVR phase has
the GPU to itself, so the ~12 GiB DiT fits a 16 GB card (this is the mode for
GPUs without FP8). Whatever ``_decide_accel`` switches off is only reported.

Idempotent: clips whose ``*_fvsr.npz`` already exists are skipped (stage
resume). A clip that runs out of VRAM is retried once (``_restore_checked``);
a second failure stops the driver with a non-zero exit, the bundle stays, and
the run resumes with ``--swiftvr-bundle-dir``. SwiftVR's FP8 has no fallback
path, so there is no mid-run demotion to retry after (unlike FlashVSR's).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _load_sibling_module(name: str):
    """Load ``<this dir>/<name>.py`` by path (no sys.path edit, no package).

    The inline worker lives next to this script, also in the frozen build where
    both are copied as real files to <dist>/jasna/restorer/. Its top level is
    stdlib-only, so loading it under the SwiftVR venv is safe.
    """
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_worker = _load_sibling_module("swiftvr_inline_worker")
DEFAULT_CLIP_LEN = _worker.DEFAULT_CLIP_LEN


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SwiftVR offline Phase 2 (256px crops -> 256*scale px)")
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--repo", required=True, help="SwiftVR checkout (fork sh202603/SwiftVR, branch modi)")
    ap.add_argument("--model-dir", required=True, help="SwiftVR checkpoint directory")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scale", type=int, default=4, choices=[2, 4],
                    help="Processing upscale factor: crops are processed at 256*scale px "
                         "(4 = model-native 1024px, 2 = 512px).")
    ap.add_argument("--clip-len", type=int, default=DEFAULT_CLIP_LEN,
                    help="SwiftVR MIDDLE chunk length in frames (multiple of 4).")
    ap.add_argument("--fp8-dit", action="store_true",
                    help="Run the DiT's linear layers in FP8 (sm89+ GPU; switched off with a "
                         "notice where unsupported, bf16 then).")
    ap.add_argument("--torch-compile", action="store_true",
                    help="torch.compile the DiT blocks (needs a working Triton; switched off "
                         "with a notice otherwise).")
    ap.add_argument(
        "--color-fix-method",
        default="wavelet",
        choices=["adain", "wavelet", "none"],
        help="Color correction of the output crops against the bicubic-upscaled input "
             "(always on in production; 'none' exists only for A/B baselines and is "
             "reachable via the JASNA_SWIFTVR_COLOR_FIX env override, not the jasna CLI).",
    )
    ap.add_argument("--verbose", action="store_true",
                    help="keep the model libraries' warnings (diffusers) on stderr")
    return ap.parse_args()


def _restore_bundle_clip(pipe, torch, primary_u8: np.ndarray, device, scale: int, clip_len: int,
                         color_fix_method: str) -> np.ndarray:
    """One clip: (T,3,256,256) uint8 RGB CHW -> (T,3,S,S) uint8 RGB CHW, S = 256*scale.

    The same two calls as the inline worker makes per clip (the checked
    ``restore_clip()`` with one out-of-VRAM retry, then the GPU color fix
    against the input crops); only the CHW <-> HWC layout change is ours.
    """
    lq = torch.from_numpy(np.ascontiguousarray(primary_u8.transpose(0, 2, 3, 1))).to(device)  # (T,256,256,3)
    out = _worker._restore_checked(pipe, torch, lq, scale, clip_len)  # (T,S,S,3) uint8 on device
    if color_fix_method != "none":
        out = _worker._color_fix_frames_gpu(out, lq, color_fix_method)
    restored = out.permute(0, 3, 1, 2).contiguous().cpu().numpy()
    del lq, out
    return restored


def main() -> None:
    args = _parse_args()
    if args.clip_len % 4 != 0 or args.clip_len < 4:
        raise SystemExit(f"--clip-len must be a positive multiple of 4, got {args.clip_len}")
    bundle_dir = Path(args.bundle_dir)
    if not Path(args.model_dir).is_dir():
        raise FileNotFoundError(f"--model-dir not found: {args.model_dir}")
    os.environ.setdefault("TQDM_DISABLE", "1")

    # The venv normally has the checkout installed (uv sync, editable); the
    # sys.path entry is the fallback for a venv built elsewhere.
    sys.path.insert(0, args.repo)
    import torch

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)

    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    clips = manifest["clips"]
    scale = int(args.scale)
    clip_len = int(args.clip_len)
    print(f"[swiftvr-phase2] {len(clips)} clips to restore (scale={scale}, clip_len={clip_len}, "
          f"color_fix={args.color_fix_method})", flush=True)

    fp8, comp, accel_parts, accel_log = _worker._decide_accel(args.fp8_dit, args.torch_compile, device)
    for line in accel_log:
        print(f"[swiftvr-phase2] accel: {line}", flush=True)

    t0 = time.perf_counter()
    pipe = _worker._load_pipeline(args.model_dir, device, fp8=fp8, torch_compile=comp,
                                  quiet=not args.verbose)
    t_load = time.perf_counter() - t0
    t1 = time.perf_counter()
    _worker._warmup(pipe, torch, device, scale, clip_len, args.color_fix_method)
    print(f"[swiftvr-phase2] ready: model load {t_load:.1f}s, warmup {time.perf_counter() - t1:.1f}s "
          f"(scale {scale}, clip_len {clip_len}, accel: {', '.join(accel_parts) or 'off'})", flush=True)

    done = 0
    skipped = 0
    for idx, entry in enumerate(clips):
        key = entry["key"]
        out_path = bundle_dir / f"{key}_fvsr.npz"
        if out_path.exists():
            skipped += 1
            continue
        with np.load(bundle_dir / f"{key}.npz", allow_pickle=False) as data:
            primary_u8 = data["primary_u8"]  # (T,3,256,256) uint8 RGB CHW

        t2 = time.perf_counter()
        restored = _restore_bundle_clip(pipe, torch, primary_u8, device, scale, clip_len,
                                        args.color_fix_method)
        # atomic-ish write: temp then rename. The temp name MUST end in .npz or
        # np.savez appends .npz to it (writing a different file than os.replace
        # then looks for).
        tmp = out_path.with_name(out_path.stem + ".partial.npz")
        np.savez(tmp, restored_u8=restored)
        os.replace(tmp, out_path)
        done += 1
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[swiftvr-phase2] [{idx + 1}/{len(clips)}] {key}: {primary_u8.shape[0]} frames -> "
              f"{restored.shape} in {time.perf_counter() - t2:.1f}s", flush=True)

    print(f"[swiftvr-phase2] done: {done} restored, {skipped} already present", flush=True)


if __name__ == "__main__":
    main()

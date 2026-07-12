#!/usr/bin/env python3
"""FlashVSR offline Phase 2 driver — runs under the FlashVSR virtualenv.

Reads a jasna flashvsr *bundle* (produced by Phase 1), upscales every clip's
256px primary crops to 1024px with FlashVSR (tiny mode, 4x), and writes the
results back into the bundle for Phase 3 to re-blend.

This script is intentionally free of any ``jasna`` import: it is executed by the
FlashVSR env's uv-managed standalone Python (``--flashvsr-python``), not jasna's
interpreter. It only uses numpy/torch/PIL plus the FlashVSR repo (added to
``sys.path`` at runtime).

Invocation (by the orchestrator in ``jasna.restorer.flashvsr_offline``):

    <flashvsr-python> flashvsr_phase2_driver.py \
        --bundle-dir <dir> --repo <FlashVSR_plus> --model-dir <weights> \
        --version 11 --dtype bf16 --device cuda:0 --scale 4 \
        --unload-dit --tiled-vae

Bundle contract (numpy/JSON, see ``flashvsr_offline.py``):
  in : manifest.json, clip_<track>_<start>.npz  (field ``primary_u8`` = (T,3,256,256) uint8 RGB CHW)
  out: clip_<track>_<start>_fvsr.npz            (field ``restored_u8`` = (T,3,1024,1024) uint8 RGB CHW)

Idempotent: clips whose ``*_fvsr.npz`` already exists are skipped (stage resume).

Only *tiny* mode is used, so Phase 2 returns lossless tensors (tiny-long would
write a lossy mp4). jasna caps Phase 1's clip length (``--flashvsr-max-clip-frames``,
default 32) so every clip fits tiny-mode VRAM. FlashVSR's 8n+5 / +4-warmup frame
padding is applied to the input and stripped from the output so exactly T frames
are written (the blend contract silently corrupts on a frame-count mismatch).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FlashVSR offline Phase 2 (256px -> 1024px, 4x)")
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--repo", required=True, help="FlashVSR_plus checkout")
    ap.add_argument("--model-dir", required=True, help="FlashVSR weights dir (validated only)")
    ap.add_argument("--version", default="11", choices=["10", "11"])
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attention", default="sage", choices=["sage", "block"])
    ap.add_argument("--unload-dit", action="store_true")
    ap.add_argument("--tiled-vae", action="store_true")
    ap.add_argument("--color-fix", action="store_true")
    return ap.parse_args()


def _import_flashvsr(repo: str, version: str, attention: str):
    """Import the FlashVSR ``run`` module + attention global.

    ``run.py`` calls ``argparse.parse_args()` at module top level, so a valid
    dummy ``sys.argv`` is installed before import (its ``main()`` is guarded by
    ``if __name__ == '__main__'`` and does not execute on import).
    """
    sys.path.insert(0, repo)
    saved_argv = sys.argv
    sys.argv = ["run.py", "-i", "__flashvsr_dummy__", "-v", version, "__flashvsr_dummy_out__"]
    try:
        import run  # type: ignore
        from src.models import wan_video_dit  # type: ignore
    finally:
        sys.argv = saved_argv
    # sage (default) uses the Triton Sparse_SageAttention kernel; block = dense.
    wan_video_dit.USE_BLOCK_ATTN = attention != "sage"
    return run


def _upscale_clip(run, pipe, torch, primary_u8: np.ndarray, device, dtype, scale: int, *,
                  seed: int, tiled_vae: bool, unload_dit: bool, color_fix: bool) -> np.ndarray:
    """Upscale one clip's (T,3,256,256) uint8 crops to (T,3,1024,1024) uint8.

    Mirrors run.py's ``main`` preprocessing: build (T,H,W,C) [0,1], replicate the
    last frame up to the nearest 8n+5 (min 21), then ``prepare_input_tensor``
    (adds the +4 warmup, normalizes to [-1,1], upscales bicubic 4x). The DiT runs
    one DMD step; ``tensor2video`` denorms back to [0,1] and the output is trimmed
    to exactly T real frames.
    """
    t_real = int(primary_u8.shape[0])
    # (T,3,256,256) uint8 CHW  ->  (T,256,256,3) float[0,1] RGB
    frames = torch.from_numpy(primary_u8).permute(0, 2, 3, 1).to(dtype).div(255.0)

    add = run.next_8n5(t_real) - t_real
    if add > 0:
        pad = frames[-1:].repeat(add, 1, 1, 1)
        frames = torch.cat([frames, pad], dim=0)

    LQ, th, tw, F = run.prepare_input_tensor(frames, device, scale=scale, dtype=dtype)
    LQ = LQ.to(device)

    # `unload_dit` offloads the DiT to CPU during each clip's VAE decode. When the
    # pipe is reused across many clips (unlike run.py, which loads once per clip),
    # the DiT must be back on the device before the next clip's inference.
    if unload_dit and hasattr(pipe, "load_models_to_device"):
        pipe.load_models_to_device(["dit", "vae"])

    video = pipe(
        prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed,
        tiled=tiled_vae, LQ_video=LQ, num_frames=F, height=th, width=tw,
        is_full_block=False, if_buffer=True,
        topk_ratio=2 * 768 * 1280 / (th * tw), kv_ratio=3, local_range=11,
        color_fix=color_fix, unload_dit=unload_dit, fps=30, output_path=os.devnull,
        tiled_dit=True,
    )
    out = run.tensor2video(video).to("cpu")[:t_real]  # (T,1024,1024,3) [0,1]
    restored = (out.float() * 255.0).round().clamp(0, 255).to(torch.uint8)
    restored = restored.permute(0, 3, 1, 2).contiguous().numpy()  # (T,3,1024,1024) CHW
    return restored


def main() -> None:
    args = _parse_args()
    bundle_dir = Path(args.bundle_dir)
    if not Path(args.model_dir).is_dir():
        raise FileNotFoundError(f"--model-dir not found: {args.model_dir}")

    import torch

    run = _import_flashvsr(args.repo, args.version, args.attention)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)

    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    clips = manifest["clips"]
    print(f"[flashvsr-phase2] {len(clips)} clips to upscale (version={args.version}, mode=tiny)", flush=True)

    pipe = run.init_pipeline(args.version, "tiny", device, dtype)

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

        restored = _upscale_clip(
            run, pipe, torch, primary_u8, device, dtype, args.scale,
            seed=args.seed, tiled_vae=args.tiled_vae, unload_dit=args.unload_dit,
            color_fix=args.color_fix,
        )
        # atomic-ish write: temp then rename. The temp name MUST end in .npz or
        # np.savez appends .npz to it (writing a different file than os.replace
        # then looks for).
        tmp = out_path.with_name(out_path.stem + ".partial.npz")
        np.savez(tmp, restored_u8=restored)
        os.replace(tmp, out_path)
        done += 1
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[flashvsr-phase2] [{idx + 1}/{len(clips)}] {key}: {primary_u8.shape[0]} frames -> {restored.shape}", flush=True)

    print(f"[flashvsr-phase2] done: {done} upscaled, {skipped} already present", flush=True)


if __name__ == "__main__":
    main()

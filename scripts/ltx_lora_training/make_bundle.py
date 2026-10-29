"""Assemble a jasna LTX model folder from a converted transformer.

The folder gets the transformer under the chosen slot's file name (``--model distilled`` = the
8-step slot, ``distilled4`` = 4-step, ``undistilled`` = 15-step with STG), the public LTX-2.5
DiffVAE as ``vae.safetensors`` (checked against jasna's recorded tensor names, shapes and
config) and an empty ``vae-decoder.safetensors`` (no tuned-decoder overlay, so the stock 2.5
diffusion decoder runs). The transformer and VAE are hard-linked when possible and symlinked
otherwise, so the folder costs no extra disk.

Run with jasna's venv. Use the folder with ``--restoration-model-name ltx-8step
--restoration-model-path <folder>`` (``ltx-4step`` or ``ltx-undistilled`` for the other slots),
or copy its files into ``model_weights/ltx-restore/`` where the dev install finds them by name.

  .venv/bin/python scripts/ltx_lora_training/make_bundle.py --transformer my-model-int8.safetensors \
      --vae ltx-2.5-video-vae-bf16.safetensors --model-dir model_weights/my-ltx-model
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from jasna.ltx import trial_shapes as S
from jasna.ltx.model_files import bundle_names


def link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        dst.symlink_to(src.resolve())


def check_vae(path: Path) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        names = list(f.keys())
        specs = {n: (f.get_slice(n).get_dtype(), tuple(f.get_slice(n).get_shape())) for n in names}
    config = json.loads(meta["config"])
    expected = {n: (dt, tuple(sh)) for n, (dt, sh) in S.VAE.items()}
    missing = sorted(set(expected) - set(specs))
    extra = sorted(set(specs) - set(expected))
    mismatch = [n for n in expected if n in specs and specs[n] != expected[n]]
    print(f"VAE: {len(names)} tensors; missing {len(missing)} {missing[:3]}, extra {len(extra)} {extra[:3]}, dtype/shape mismatch {len(mismatch)} {mismatch[:3]}")
    print(f"VAE config == trial_shapes.VAE_CONFIG: {config == S.VAE_CONFIG}")
    if config != S.VAE_CONFIG:
        a, b = config["vae"], S.VAE_CONFIG["vae"]
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                print(f"  config differs at {k}: {str(a.get(k))[:120]} | {str(b.get(k))[:120]}")
    if missing or mismatch:
        raise SystemExit("VAE file does not match jasna's expected tensors")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer", type=Path, required=True, help="converted transformer (convert_ltx_to_jasna.py output)")
    ap.add_argument("--vae", type=Path, required=True, help="public LTX-2.5 DiffVAE, ltx-2.5-video-vae-bf16.safetensors")
    ap.add_argument("--model-dir", type=Path, required=True, help="model folder to create")
    ap.add_argument("--model", choices=("distilled", "distilled4", "distilled2", "undistilled"), default="distilled",
                    help="slot whose file name the transformer gets (distilled = ltx-8step, default)")
    args = ap.parse_args()
    check_vae(args.vae)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    transformer_name, vae_name, decoder_name = bundle_names(args.model, fast=False)
    link(args.transformer, args.model_dir / transformer_name)
    link(args.vae, args.model_dir / vae_name)
    save_file({}, str(args.model_dir / decoder_name))
    for p in sorted(args.model_dir.iterdir()):
        print(f"  {p.name}: {p.stat().st_size / 1e9:.3f} GB")


if __name__ == "__main__":
    main()

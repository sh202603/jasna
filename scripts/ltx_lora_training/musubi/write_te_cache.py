"""Write Musubi Tuner LTX-2 text-encoder cache files from a precomputed prompt_context.

Musubi normally runs Gemma 3 12B over every caption (``ltx2_cache_text_encoder_outputs.py``); on a
16 GB card that does not fit even in 8-bit, and jasna has no text encoder at all. The caption is
fixed anyway, so this writes the tensor jasna sees at inference (encode_prompt.py's output) as the
text cache of every latent in the given cache folders: ``<stem>_ltx2_te.safetensors`` holding
``video_prompt_embeds_bfloat16`` [1024, 4096] (valid tokens first, registers after, all-valid
mask), the same tensor again as ``text_bfloat16``, and int64 ones as ``prompt_attention_mask`` and
``text_mask``. One file is written per folder, the rest are hard links (copies if linking fails).
Training and inference then condition on the same constant.

Run after cache_latents.sh, on the target cache folders (not the ``_ref`` reference folders), with
any Python that has torch and safetensors (jasna's venv works):

  python scripts/ltx_lora_training/musubi/write_te_cache.py --prompt-context prompt_context.safetensors \\
      /path/to/cache/train /path/to/cache/val
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LATENT_RE = re.compile(r"^(?P<stem>.+)_\d+-\d+_\d+x\d+_ltx2\.safetensors$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt-context", type=Path, required=True, help="encode_prompt.py output")
    ap.add_argument("--caption", default=None, help="caption for the cache metadata (default: the prompt stored in --prompt-context)")
    ap.add_argument("cache_dirs", type=Path, nargs="+", help="Musubi cache folders holding *_ltx2.safetensors latents")
    args = ap.parse_args()
    with safe_open(str(args.prompt_context), framework="pt", device="cpu") as f:
        emb = f.get_tensor("prompt_context")[0].to(torch.bfloat16).contiguous()  # [1024, 4096]
        caption = args.caption if args.caption is not None else (f.metadata() or {}).get("prompt", "")
    ones = torch.ones(emb.shape[0], dtype=torch.int64)
    tensors = {"video_prompt_embeds_bfloat16": emb, "prompt_attention_mask": ones, "text_bfloat16": emb.clone(), "text_mask": ones.clone()}
    meta = {"architecture": "ltx2_v1", "caption1": caption, "format_version": "1.0.1", "source": args.prompt_context.name}
    for folder in args.cache_dirs:
        latents = sorted(p for p in folder.iterdir() if LATENT_RE.match(p.name))
        if not latents:
            raise SystemExit(f"{folder}: no *_ltx2.safetensors latents (run cache_latents.sh first)")
        first: Path | None = None
        for p in latents:
            out = folder / f"{LATENT_RE.match(p.name)['stem']}_ltx2_te.safetensors"
            if out.exists():
                out.unlink()
            if first is None:
                save_file(tensors, str(out), metadata=meta)
                first = out
            else:
                try:
                    os.link(first, out)
                except OSError:
                    shutil.copyfile(first, out)
        print(f"{folder}: {len(latents)} latents, {len(list(folder.glob('*_ltx2_te.safetensors')))} text caches, first {first.name}")


if __name__ == "__main__":
    main()

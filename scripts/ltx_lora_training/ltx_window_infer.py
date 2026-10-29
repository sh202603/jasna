"""Window-level LTX inference in the training geometry, for checking a LoRA without the full pipeline.

The whole 512x512x121 input clip (a dataset ``lq`` pair, mosaic + degradation) is the reference window:
no detection, no camera. Encode with jasna's VAE encoder, denoise one window with the model folder's
transformer (``denoise_chain``, the same call the compose pass makes), decode, write ``<stem>.mkv``
(libx264 crf 10). About 7 s per window on an RTX 5080 once the model is loaded. Compare the output
with the pair's ``gt`` clip (PSNR inside the mask, sharpness) to rank checkpoints; the full pipeline is
the wrong tool for that, because its detector splits a synthetic mosaic into several tracks and the
windows no longer match the training geometry.

Run with jasna's venv from the jasna checkout:

  .venv/bin/python scripts/ltx_lora_training/ltx_window_infer.py --model-dir <model folder> \
      --out-dir <dir> --inputs a.mp4 b.mp4 ...
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import av
import numpy as np
import torch

from jasna.ltx.camera import PLAN_CANVAS, WINDOW_FRAMES
from jasna.ltx.inference import ModelRuntime
from jasna.ltx.model_files import LtxModelFiles
from jasna.ltx.restore import _decode_window, _vae_pixels, free_vram_budget
from jasna.session_config import LTX_DEFAULT_SEED


def read_bgr(path: Path) -> np.ndarray:
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="bgr24") for f in c.decode(video=0)])


def write_bgr(path: Path, frames: np.ndarray, fps: int = 24) -> None:
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps, options={"crf": "10", "preset": "medium"})
        s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        for f in frames:
            for pk in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="bgr24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True, help="model folder (make_bundle.py output)")
    ap.add_argument("--model", choices=("distilled", "distilled4", "distilled2", "undistilled"), default="distilled",
                    help="slot to load (distilled = the 8-step file name, default)")
    ap.add_argument("--inputs", type=Path, nargs="+", required=True, help="512x512 clips of at least 121 frames (dataset lq pairs)")
    ap.add_argument("--out-dir", type=Path, required=True, help="folder for <stem>.mkv outputs")
    ap.add_argument("--seed", type=int, default=LTX_DEFAULT_SEED, help="window seed (jasna's default)")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    files = LtxModelFiles.from_dir(a.model_dir, a.model, fast=False)
    t0 = time.time()
    clips = []
    for p in a.inputs:
        frames = read_bgr(p)[:WINDOW_FRAMES]
        assert frames.shape[0] == WINDOW_FRAMES and frames.shape[1:3] == (PLAN_CANVAS, PLAN_CANVAS), (p, frames.shape)
        clips.append(torch.from_numpy(frames[..., ::-1].copy()).permute(3, 0, 1, 2).contiguous())  # [3, F, H, W] RGB uint8
    print(f"read {len(clips)} clips in {time.time() - t0:.1f} s", flush=True)
    with ModelRuntime(files, device, threading.Event()) as runtime:
        encoder = runtime.load_encoder()
        refs = []
        with torch.inference_mode():
            for pixels in clips:
                refs.append(encoder(_vae_pixels(pixels, device)).to("cpu", torch.bfloat16).contiguous())
        runtime.unload_encoder()
        print(f"encoded: latent {tuple(refs[0].shape)} {refs[0].dtype}, {time.time() - t0:.1f} s", flush=True)
        transformer = runtime.load_transformer()
        finals = []
        for ref in refs:
            t1 = time.time()
            finals.append(transformer.denoise_chain([ref], [a.seed], 88, lambda n: None)[0])
            print(f"  denoised one window in {time.time() - t1:.1f} s", flush=True)
        runtime.unload_transformer()
        decoder = runtime.load_decoder()
        for p, latent in zip(a.inputs, finals):
            video = _decode_window(decoder, latent, a.seed, PLAN_CANVAS, free_vram_budget(device), device)  # [F, 3, S, S]
            out = video.permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]
            write_bgr(a.out_dir / (p.stem + ".mkv"), out)
            print(f"  wrote {p.stem}.mkv {tuple(video.shape)}", flush=True)
    print(f"done in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()

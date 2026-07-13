#!/usr/bin/env python3
"""FlashVSR inline worker — runs under the FlashVSR virtualenv.

Loads the FlashVSR **tiny-long** pipeline once (weights resident) and, for each
clip received over a length-prefixed stdin/stdout protocol, upscales the 256px
primary crops to 1024px (4x) and streams them back. tiny-long is O(1) in VRAM
(flat ~11.85GB regardless of clip length) which is what lets FlashVSR co-reside
with jasna's primary pipeline inside a 16GB GPU (see FLASHVSR_INLINE_FEASIBILITY
§12); the O(T) *tiny* path OOMs when co-resident.

This script is intentionally free of any ``jasna`` import: it is executed by the
FlashVSR env's uv-managed standalone Python (``--flashvsr-python``), not jasna's
interpreter. It only uses numpy/torch plus the FlashVSR repo (added to
``sys.path`` at runtime).

Requires a FlashVSR_plus checkout with the tiny-long multi-chunk fix
(``tinylong_multichunk_fix.patch``); the parent restorer verifies this before
spawning us.

Wire protocol (parent = jasna venv, child = this):
  parent -> child : header ``{"seq","n","h","w"}\n`` (UTF-8) then n*h*w*3 raw
                    uint8 RGB bytes  (the 256px primary crops, HWC)
  child  -> parent: header ``{"seq","n","h","w"}\n`` then n*h*w*3 raw uint8 RGB
                    bytes  (the 1024px restored crops, HWC), exactly n frames
  child  -> parent (once, at startup): ``{"status":"ready"}\n``
  child  -> parent (on per-clip failure): ``{"seq","error":"..."}\n`` then stays
                    alive for the next clip.

fd handling (SEEDVR2 design §4.1): the *real* stdout fd is dup'd to a private
protocol fd, then fd 1 is repointed to /dev/null (or stderr if --verbose) so the
FlashVSR banner / prints never corrupt the wire. ``TQDM_DISABLE=1`` is set before
tqdm is imported.
"""
from __future__ import annotations

import argparse
import json
import os
import struct  # noqa: F401  (kept for parity; framing uses newline+raw)
import sys
import traceback


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FlashVSR inline worker (tiny-long, 256->1024 4x)")
    ap.add_argument("--repo", required=True, help="FlashVSR_plus checkout")
    ap.add_argument("--model-dir", required=True, help="FlashVSR weights dir (validated only)")
    ap.add_argument("--version", default="11", choices=["10", "11"])
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--attention", default="sage", choices=["sage", "block"])
    ap.add_argument("--verbose", action="store_true", help="send worker stdout to stderr, not /dev/null")
    return ap.parse_args()


def _install_protocol_fd(verbose: bool):
    """Reserve the real stdout as the protocol channel and mute fd 1.

    Returns a binary file object for writing protocol messages. Reads use the
    real stdin (fd 0). After this, any library print()/banner goes to /dev/null
    (or stderr when --verbose) instead of the wire.
    """
    proto_fd = os.dup(1)
    proto = os.fdopen(proto_fd, "wb", buffering=0)
    sink = sys.stderr.fileno() if verbose else os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink, 1)
    return proto


def _read_exact(stream, n: int) -> bytes:
    """Read exactly n bytes or raise EOFError (child dies -> parent sees pipe close)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("stdin closed while reading payload")
        buf.extend(chunk)
    return bytes(buf)


def _read_header(stream) -> dict | None:
    """Read one newline-terminated JSON header, or None on EOF (parent gone)."""
    line = bytearray()
    while True:
        b = stream.read(1)
        if not b:
            return None
        if b == b"\n":
            break
        line.extend(b)
    return json.loads(line.decode("utf-8"))


class _CaptureWriter:
    """Stand-in for imageio's writer: capture append_data frames losslessly."""

    def __init__(self, sink: list):
        self._sink = sink

    def append_data(self, frame):
        # frame is already HWC uint8 RGB (tensor_to_imageio_frame output).
        self._sink.append(frame)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main() -> None:
    args = _parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"--model-dir not found: {args.model_dir}")

    # Mute the FlashVSR banner/tqdm before importing it; reserve the protocol fd.
    os.environ.setdefault("TQDM_DISABLE", "1")
    proto = _install_protocol_fd(args.verbose)
    stdin = sys.stdin.buffer

    # Import the FlashVSR repo (adds it to sys.path, installs a dummy argv so
    # run.py's module-level parse_args doesn't fire).
    sys.path.insert(0, args.repo)
    os.chdir(args.repo)
    _saved_argv = sys.argv
    sys.argv = ["run.py", "-i", "__inline_dummy__", "-v", args.version, "__inline_dummy_out__"]
    try:
        import numpy as np
        import torch
        import imageio
        import run  # type: ignore
        from src.models import wan_video_dit  # type: ignore
    finally:
        sys.argv = _saved_argv

    wan_video_dit.USE_BLOCK_ATTN = args.attention != "sage"

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)

    # Lossless tensor capture: replace imageio.get_writer with a capturer. The
    # tiny-long pipe writes tensor_to_imageio_frame(...) (HWC uint8 RGB) per
    # frame, so capturing append_data yields the exact output frames without an
    # mp4 round-trip.
    captured: list = []

    def _fake_get_writer(*_a, **_k):
        captured.clear()
        return _CaptureWriter(captured)

    imageio.get_writer = _fake_get_writer

    pipe = run.init_pipeline(args.version, "tiny-long", device, dtype)

    # Handshake: tell the parent we're ready to accept clips.
    proto.write((json.dumps({"status": "ready"}) + "\n").encode("utf-8"))

    scale = int(args.scale)
    devnull_out = os.path.join("/tmp", "flashvsr_inline_devnull.mp4")

    while True:
        header = _read_header(stdin)
        if header is None:
            break  # parent closed stdin -> shut down
        seq = int(header.get("seq", -1))
        n = int(header["n"])
        h = int(header["h"])
        w = int(header["w"])
        payload = _read_exact(stdin, n * h * w * 3)

        try:
            crops = np.frombuffer(payload, dtype=np.uint8).reshape(n, h, w, 3)
            frames = torch.from_numpy(crops.copy()).to(dtype).div_(255.0)  # (n,h,w,3) [0,1]

            # Pad up to next_8n5 (min 21) by replicating the last frame, exactly
            # like run.py main() (lines 513-515): tiny-long's chunk loop needs a
            # seed-sized clip, so a small clip (e.g. a 1-frame mosaic track) would
            # otherwise yield F=1 -> 0 chunks -> 0 frames. Output is trimmed to n.
            add = run.next_8n5(n) - n
            if add > 0:
                frames = torch.cat([frames, frames[-1:].repeat(add, 1, 1, 1)], dim=0)

            th, tw, F = run.get_input_params(frames, scale=scale)
            LQ = run.input_tensor_generator(frames, device, scale=scale, dtype=dtype)

            captured.clear()
            pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
                tiled=False, LQ_video=LQ, num_frames=F, height=th, width=tw,
                is_full_block=False, if_buffer=True,
                topk_ratio=2 * 768 * 1280 / (th * tw), kv_ratio=3, local_range=11,
                color_fix=False, unload_dit=False, fps=30, output_path=devnull_out,
                tiled_dit=True,
            )
            # captured: list of (1024,1024,3) uint8 HWC RGB, length >= n (see
            # FLASHVSR_INLINE_FEASIBILITY: output >= input for tiny-long). Align
            # to exactly n frames (trim; defensively pad by repeating the last).
            if len(captured) == 0:
                raise RuntimeError(f"tiny-long produced 0 frames for {n}-frame clip")
            out = captured[:n]
            while len(out) < n:
                out.append(out[-1])
            out_arr = np.ascontiguousarray(np.stack(out, axis=0))  # (n,1024,1024,3) uint8
            oh, ow = int(out_arr.shape[1]), int(out_arr.shape[2])

            resp = json.dumps({"seq": seq, "n": n, "h": oh, "w": ow}) + "\n"
            proto.write(resp.encode("utf-8"))
            proto.write(out_arr.tobytes())

            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()  # co-residence discipline: return reserved each clip
        except Exception as e:  # keep the worker alive; report per-clip failure
            traceback.print_exc()
            err = json.dumps({"seq": seq, "error": f"{type(e).__name__}: {e}"}) + "\n"
            proto.write(err.encode("utf-8"))


if __name__ == "__main__":
    main()

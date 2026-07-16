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
import tempfile
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
    ap.add_argument(
        "--tiles",
        type=int,
        default=1,
        help="Number of full-width horizontal DiT strips per clip for tiled-dit "
             "(1 = off, max 4). More strips cut DiT peak VRAM for 16GB GPUs at a "
             "latency cost (~1.25x at 2, ~1.5x at 3-4). Strip height is picked as a "
             "32-multiple so the 4x-upscaled strip stays a 128-multiple.",
    )
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


def _feather_mask_numpy(size, overlap):
    """Per-tile feather mask (HWC, 1 channel), identical to run.py's
    ``create_feather_mask_numpy`` (run.py:313-324) but inlined so this module has
    zero FlashVSR-checkout dependency and stays unit-testable under jasna pytest.

    Ramps each of the four edges over ``overlap`` px via ``linspace(0,1)`` and
    multiplies (so corners decay in both axes). The outermost px weight is exactly
    0 — an inherited FlashVSR trait; weight-sum normalisation in ``_stitch_tiles``
    keeps it out of interior seams and only darkens the true crop border by 1px.
    """
    import numpy as np

    H, W = size
    mask = np.ones((H, W, 1), dtype=np.float32)
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    mask[:, :overlap, :] *= ramp[np.newaxis, :, np.newaxis]
    mask[:, -overlap:, :] *= np.flip(ramp)[np.newaxis, :, np.newaxis]
    mask[:overlap, :, :] *= ramp[:, np.newaxis, np.newaxis]
    mask[-overlap:, :, :] *= np.flip(ramp)[:, np.newaxis, np.newaxis]
    return mask


def _strip_coords(H, W, n_tiles):
    """Coordinates for ``n_tiles`` full-width horizontal strips covering ``[0,H]``.

    Splitting only the height (width stays full) is far cheaper than a square grid
    for a square 256px crop: it reaches the same DiT peak-VRAM cut (the attn mask
    scales with strip-area squared) with far less overlap-redundant compute. Strip
    height is a multiple of 32 so the 4x-upscaled strip is a multiple of 128 (the
    DiT window requirement) with no center-crop loss; a ~32px margin guarantees a
    feather overlap between neighbours. Strips are evenly spaced, the last snapped
    to the bottom. Returns ``(coords, overlap)`` where ``overlap`` is the vertical
    overlap (LQ px) fed to the feather composite. ``n_tiles <= 1`` -> one full tile.
    """
    if n_tiles <= 1:
        return [(0, 0, W, H)], 0
    base = -(-H // n_tiles) + 32               # ceil(H/n_tiles) + feather margin
    sh = min(H, ((base + 31) // 32) * 32)      # round up to a 32-multiple
    stride = (H - sh) / (n_tiles - 1)
    coords = []
    for i in range(n_tiles):
        y1 = int(round(i * stride))
        y2 = y1 + sh
        if y2 > H:
            y1, y2 = H - sh, H
        coords.append((0, y1, W, y2))
    overlap = max(1, sh - int(round(stride)))
    return coords, overlap


def _stitch_tiles(tile_frames_list, tile_coords, scale, overlap, out_h, out_w):
    """CPU weighted-feather composite of per-tile restored frames.

    Mirror of run.py's non-tiny-long tiled composite (run.py:572-597), but sourced
    from captured HWC uint8 frames instead of ``tensor2video(output_tile_gpu)``.
    Because the inline worker replaces ``imageio.get_writer`` with a tensor
    capturer, run.py's mp4-round-trip ``stitch_video_tiles`` cannot be reused.

    Tiles are the full-width horizontal strips from ``_strip_coords``; all share the
    same height, so every ``tf`` has the same frame count ``M`` and spatial size.
    The feather mask ramps all four edges, but the full-width left/right ramps
    cancel under weight normalisation (every strip covers the full width), so only
    the vertical seams crossfade; the outermost 1px stays darkened (inherited from
    run.py, blended away downstream). The weight mask is frame- and channel-
    invariant, so ``wsum`` is a single ``(out_h, out_w, 1)`` plane (saves ~415 MB
    vs a per-frame weight canvas). Returns float32 ``(M, out_h, out_w, 3)`` 0..255.
    """
    import numpy as np

    M = tile_frames_list[0].shape[0]
    canvas = np.zeros((M, out_h, out_w, 3), dtype=np.float32)
    wsum = np.zeros((out_h, out_w, 1), dtype=np.float32)
    ramp_px = overlap * scale  # feather over the scaled overlap (run.py:577)
    for (x1, y1, x2, y2), tf in zip(tile_coords, tile_frames_list):
        tf = tf.astype(np.float32)  # (M, tsH, tsW, 3), 0..255
        tsH, tsW = tf.shape[1], tf.shape[2]
        mask = _feather_mask_numpy((tsH, tsW), ramp_px)  # (tsH, tsW, 1)
        oy1, ox1 = y1 * scale, x1 * scale
        oy2, ox2 = oy1 + tsH, ox1 + tsW
        canvas[:, oy1:oy2, ox1:ox2, :] += tf * mask[np.newaxis]
        wsum[oy1:oy2, ox1:ox2, :] += mask
    wsum[wsum == 0] = 1.0
    return canvas / wsum[np.newaxis]


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
    # Never written (imageio.get_writer is patched above); must merely be a
    # valid-looking path on every platform.
    devnull_out = os.path.join(tempfile.gettempdir(), "flashvsr_inline_devnull.mp4")

    # tiled-dit knob: number of full-width horizontal strips (1 = off, max 4).
    n_tiles = max(1, min(4, int(args.tiles)))

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

            H_lq, W_lq = int(frames.shape[1]), int(frames.shape[2])
            if n_tiles >= 2:
                # tiled-dit: split the (padded) clip into full-width horizontal
                # strips, restore each independently, then feather-blend into the
                # full 1024^2 canvas. Mirrors run.py:522-597's tiled-dit tiny-long
                # branch, but strips (not a square grid) and a CPU-side composite
                # (run.py's mp4-based stitch_video_tiles is unusable here since we
                # capture tensors, not mp4). Per-strip pipe() calls are safe:
                # tiny-long self-resets its causal cache (LQ_proj_in/TCDecoder) at
                # the top of every __call__.
                tile_coords, feather_overlap = _strip_coords(H_lq, W_lq, n_tiles)
                tile_frames_list = []
                for (x1, y1, x2, y2) in tile_coords:
                    input_tile = frames[:, y1:y2, x1:x2, :]
                    tth, ttw, F = run.get_input_params(input_tile, scale=scale)
                    LQ_tile = run.input_tensor_generator(input_tile, device, scale=scale, dtype=dtype)
                    # _fake_get_writer clears `captured` at each pipe() start, so
                    # after this call `captured` holds only this strip's frames.
                    pipe(
                        prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
                        tiled=False, LQ_video=LQ_tile, num_frames=F, height=tth, width=ttw,
                        is_full_block=False, if_buffer=True,
                        topk_ratio=2 * 768 * 1280 / (tth * ttw), kv_ratio=3, local_range=11,
                        color_fix=False, unload_dit=False, fps=30, output_path=devnull_out,
                        tiled_dit=True,
                    )
                    if len(captured) == 0:
                        raise RuntimeError(f"tiled strip produced 0 frames for {n}-frame clip")
                    # Snapshot now: the next strip's pipe() will clear `captured`.
                    tile_frames_list.append(np.ascontiguousarray(np.stack(captured, axis=0)))
                    del LQ_tile
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()  # release each strip's peak before the next
                stitched = _stitch_tiles(
                    tile_frames_list, tile_coords, scale, feather_overlap, H_lq * scale, W_lq * scale
                )
                out_arr = np.ascontiguousarray(
                    np.clip(stitched[:n], 0, 255).round().astype(np.uint8)
                )
                if out_arr.shape[0] < n:  # defensive pad, mirrors the single-shot path
                    pad = np.repeat(out_arr[-1:], n - out_arr.shape[0], axis=0)
                    out_arr = np.ascontiguousarray(np.concatenate([out_arr, pad], axis=0))
            else:
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

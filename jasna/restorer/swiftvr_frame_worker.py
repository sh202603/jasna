# SPDX-FileCopyrightText: 2026 sh202603
# SPDX-License-Identifier: AGPL-3.0
"""
Persistent SwiftVR worker for whole frames: runs inside the SwiftVR venv (NOT
the jasna venv), spawned by ``jasna.upscale.swiftvr.SwiftvrWorker`` for
``jasna-postprocess --upscale swiftvr``.

``swiftvr_inline_worker.py`` restores 256px mosaic crops. This worker restores
whole frames: a window of frames arrives, is cut into tiles that all have the
same size, every tile runs through ``SwiftVRPipeline.restore_clip()`` over the
whole window, and the tiles are blended back into frames with feather weights.
A whole 4K frame does not fit in consumer VRAM (SwiftVR's memory follows the
output pixels of one call), and ``restore_clip`` keeps no state between calls,
so the unit of work is one window of one tile.

The tiles are not the fork's (``swiftvr.tiling``): those are squares shifted
inwards at the frame edge, which for 1080p means 15 tiles and 1.9x the frame's
pixels. Here the frame is divided evenly into rectangles (8 tiles, 1.12x).

The blended frame gets the same color correction as the crops of the inline
worker, against the bicubic-upscaled input frame. It runs on the whole frame,
not per tile: the wavelet filter reaches 16 px and would see replicated pixels
at a tile edge where the frame has real ones.

The model handling (``_decide_accel``, ``_load_pipeline``, ``_restore_checked``,
``_color_fix_frames_gpu``) and the protocol helpers are the inline worker's,
loaded by path like the offline Phase 2 driver does. The top level is
stdlib-only, so jasna's tests can import ``plan_tiles``.

Wire protocol (parent = jasna venv, child = this); colors are RGB:
  child  -> parent (once): ``{"status":"ready","accel":[...],"accel_log":[...]}\\n``
  parent -> child : ``{"seq","n","h","w","scale"}\\n`` then n*h*w*3 raw uint8
                    bytes (the window's frames, HWC)
  child  -> parent: per frame ``{"seq","i","h","w"}\\n`` then h*w*3 raw uint8
                    bytes (the frame at ``scale`` times the input size), for
                    i = 0 .. n-1 in order
  child  -> parent (on failure, instead of the next frame):
                    ``{"seq","error":"..."}\\n``; the worker stays alive.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

# Megapixels one tile may have after upscaling; decides the VRAM of a call. The fork measured
# 7.5 GiB for 1024x1024 and 9.0 GiB for 1280x1280 with the FP8 DiT.
DEFAULT_TILE_MEGAPIXELS = 1.25
# Output pixels neighbouring tiles share (the fork's default: 24 input px at 4x).
DEFAULT_TILE_OVERLAP = 96
_MAX_TILES_PER_AXIS = 32


def _load_sibling_module(name: str):
    """Load ``<this dir>/<name>.py`` by path (see ``swiftvr_phase2_driver``)."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_worker = _load_sibling_module("swiftvr_inline_worker")
DEFAULT_CLIP_LEN = _worker.DEFAULT_CLIP_LEN


def _tile_length(size: int, count: int, overlap: int, align: int) -> int | None:
    """The tile length that covers ``size`` with ``count`` tiles sharing ``overlap`` px, rounded
    up to ``align`` so the upscaled tile needs no padding; ``None`` when ``count`` tiles are
    more than the axis can take."""
    if count == 1:
        return size
    length = -(-(size + (count - 1) * overlap) // count)
    length = -(-length // align) * align
    if length >= size or length <= 2 * overlap:
        return None
    return length


def _positions(size: int, length: int, count: int) -> list[int]:
    if count == 1:
        return [0]
    return [round(index * (size - length) / (count - 1)) for index in range(count)]


def plan_tiles(
    width: int,
    height: int,
    scale: int,
    *,
    megapixels: float = DEFAULT_TILE_MEGAPIXELS,
    overlap: int = DEFAULT_TILE_OVERLAP,
) -> tuple[int, int, list[int], list[int]]:
    """``(tile_width, tile_height, xs, ys)`` in input pixels: the fewest equal tiles that cover a
    ``width`` x ``height`` frame, each at most ``megapixels`` after upscaling by ``scale`` and
    sharing at least ``overlap`` output pixels with its neighbours. Among the layouts with the
    fewest tiles, the one whose tiles have the longest short side wins (SwiftVR's shifted
    windows need about 1024 output px to shift at all)."""
    align = 32 // math.gcd(32, scale)
    shared = -(-overlap // scale)
    limit = megapixels * 1_000_000 / (scale * scale)
    best = None
    for columns in range(1, _MAX_TILES_PER_AXIS + 1):
        tile_width = _tile_length(width, columns, shared, align)
        if tile_width is None:
            break
        for rows in range(1, _MAX_TILES_PER_AXIS + 1):
            tile_height = _tile_length(height, rows, shared, align)
            if tile_height is None:
                break
            if tile_width * tile_height > limit:
                continue
            key = (columns * rows, -min(tile_width, tile_height))
            if best is None or key < best[0]:
                best = (key, tile_width, tile_height, columns, rows)
            break  # more rows only add tiles for this column count
    if best is None:
        raise ValueError(
            f"no tile layout fits {megapixels} megapixels per tile for a {width}x{height} frame at {scale}x "
            f"with {overlap} px overlap; raise the tile size"
        )
    _, tile_width, tile_height, columns, rows = best
    return tile_width, tile_height, _positions(width, tile_width, columns), _positions(height, tile_height, rows)


def _axis_weights(torch, positions: list[int], length: int, scale: int, device) -> list:
    """Per tile along one axis, its weight at each output pixel: a linear ramp over the pixels
    it shares with a neighbour, 1 elsewhere. No ramp touches 0, so the sum is positive."""
    weights = []
    for index, start in enumerate(positions):
        weight = torch.ones(length * scale, dtype=torch.float32, device=device)
        if index > 0:
            shared = (positions[index - 1] + length - start) * scale
            weight[:shared] *= torch.arange(1, shared + 1, dtype=torch.float32, device=device) / (shared + 1)
        if index + 1 < len(positions):
            shared = (start + length - positions[index + 1]) * scale
            weight[-shared:] *= torch.arange(shared, 0, -1, dtype=torch.float32, device=device) / (shared + 1)
        weights.append(weight)
    return weights


class _Layout:
    """The tiles of one frame size and scale, and their blend weights on the device."""

    def __init__(self, torch, width: int, height: int, scale: int, megapixels: float, overlap: int, device) -> None:
        self.scale = scale
        self.tile_width, self.tile_height, self.xs, self.ys = plan_tiles(
            width, height, scale, megapixels=megapixels, overlap=overlap
        )
        x_weights = _axis_weights(torch, self.xs, self.tile_width, scale, device)
        y_weights = _axis_weights(torch, self.ys, self.tile_height, scale, device)
        self.tiles = []
        self.total = torch.zeros((height * scale, width * scale, 1), dtype=torch.float32, device=device)
        for y, y_weight in zip(self.ys, y_weights):
            for x, x_weight in zip(self.xs, x_weights):
                weight = (y_weight[:, None] * x_weight[None, :])[..., None]
                self.tiles.append((x, y, weight))
                self._region(self.total, x, y).add_(weight)

    def _region(self, canvas, x: int, y: int):
        s = self.scale
        return canvas[y * s:(y + self.tile_height) * s, x * s:(x + self.tile_width) * s]

    def blend(self, torch, tile_frames: list):
        """One output frame, uint8 ``[H, W, 3]``, from each tile's uint8 frame on the device."""
        canvas = torch.zeros((*self.total.shape[:2], 3), dtype=torch.float32, device=self.total.device)
        for (x, y, weight), frame in zip(self.tiles, tile_frames):
            self._region(canvas, x, y).add_(frame.to(torch.float32) * weight)
        return canvas.div_(self.total).round_().clamp_(0.0, 255.0).to(torch.uint8)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SwiftVR whole-frame worker (tiled windows)")
    ap.add_argument("--repo", required=True, help="SwiftVR checkout (fork sh202603/SwiftVR, branch modi)")
    ap.add_argument("--model-dir", required=True, help="SwiftVR checkpoint directory")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--clip-len", type=int, default=DEFAULT_CLIP_LEN,
                    help="SwiftVR MIDDLE chunk length in frames (multiple of 4).")
    ap.add_argument("--fp8-dit", action="store_true")
    ap.add_argument("--torch-compile", action="store_true")
    ap.add_argument("--color-fix-method", default="wavelet", choices=["adain", "wavelet", "none"])
    ap.add_argument("--tile-megapixels", type=float, default=DEFAULT_TILE_MEGAPIXELS,
                    help="Largest upscaled tile in megapixels (decides the VRAM of a call).")
    ap.add_argument("--tile-overlap", type=int, default=DEFAULT_TILE_OVERLAP,
                    help="Output pixels neighbouring tiles share.")
    ap.add_argument("--warmup-width", type=int, required=True, help="Frame width to warm up for")
    ap.add_argument("--warmup-height", type=int, required=True, help="Frame height to warm up for")
    ap.add_argument("--warmup-scale", type=int, required=True, choices=[2, 4])
    ap.add_argument("--verbose", action="store_true", help="send worker stdout to stderr, not /dev/null")
    return ap.parse_args()


def _read_payload(stream, size: int) -> bytearray:
    """Exactly ``size`` bytes into a writable buffer (numpy and torch share it without a copy)."""
    buffer = bytearray(size)
    view = memoryview(buffer)
    while view:
        count = stream.readinto(view)
        if not count:
            raise EOFError("stdin closed while reading payload")
        view = view[count:]
    return buffer


def _write_all(stream, data) -> None:
    """The protocol stream is unbuffered, so one write may take only part of a frame."""
    view = memoryview(data).cast("B")
    while view:
        view = view[stream.write(view):]


def _restore_window(pipe, torch, frames, layout: _Layout, clip_len: int, device) -> list:
    """Every tile of the window through SwiftVR; uint8 ``[n, h, w, 3]`` on the CPU per tile. The
    results wait in host memory: the tiles of a window come to about the window's upscaled
    frames, too much to keep next to the DiT."""
    restored = []
    for x, y, _ in layout.tiles:
        tile = frames[:, y:y + layout.tile_height, x:x + layout.tile_width].contiguous().to(device)
        restored.append(_worker._restore_checked(pipe, torch, tile, layout.scale, clip_len).cpu())
        del tile
    if str(device).startswith("cuda"):
        # The blend and the color correction work on whole frames: give them the DiT's cache.
        torch.cuda.empty_cache()
    return restored


def main() -> None:
    args = _parse_args()
    if args.clip_len % 4 != 0 or args.clip_len < 4:
        raise SystemExit(f"--clip-len must be a positive multiple of 4, got {args.clip_len}")

    os.environ.setdefault("TQDM_DISABLE", "1")
    proto = _worker._install_protocol_fd(args.verbose)
    stdin = sys.stdin.buffer

    sys.path.insert(0, args.repo)
    import numpy as np
    import torch

    device = args.device
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)

    fp8, comp, accel_parts, accel_log = _worker._decide_accel(args.fp8_dit, args.torch_compile, device)
    for line in accel_log:
        print(f"[swiftvr-frame-worker] accel: {line}", file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    pipe = _worker._load_pipeline(args.model_dir, device, fp8=fp8, torch_compile=comp, quiet=not args.verbose)
    t_load = time.perf_counter() - t0

    clip_len = int(args.clip_len)
    color_fix_method = args.color_fix_method
    layouts: dict[tuple[int, int, int], _Layout] = {}

    def layout_for(width: int, height: int, scale: int) -> _Layout:
        key = (width, height, scale)
        if key not in layouts:
            layouts.clear()  # one frame size at a time: the weights of a 4K layout are ~100 MB
            layouts[key] = _Layout(torch, width, height, scale, args.tile_megapixels, args.tile_overlap, device)
            found = layouts[key]
            print(
                f"[swiftvr-frame-worker] {width}x{height} at {scale}x: {len(found.xs)}x{len(found.ys)} tiles of "
                f"{found.tile_width}x{found.tile_height} px",
                file=sys.stderr, flush=True,
            )
        return layouts[key]

    # Warm up at the tile shape of the first video: 2*clip_len+5 frames are a FIRST, a MIDDLE
    # and a LAST chunk, which under torch.compile builds every DiT graph of that shape. Random
    # pixels, not zeros: FP8's activation scale is an abs-max.
    t1 = time.perf_counter()
    warm_layout = layout_for(args.warmup_width, args.warmup_height, args.warmup_scale)
    warm = torch.randint(
        0, 256, (2 * clip_len + 5, warm_layout.tile_height, warm_layout.tile_width, 3),
        dtype=torch.uint8, device=device,
    )
    warm_out = _worker._restore_checked(pipe, torch, warm, warm_layout.scale, clip_len)
    del warm, warm_out
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    print(
        f"[swiftvr-frame-worker] ready: model load {t_load:.1f}s, warmup {time.perf_counter() - t1:.1f}s "
        f"(clip_len {clip_len}, accel: {', '.join(accel_parts) or 'off'})",
        file=sys.stderr, flush=True,
    )

    ready = {"status": "ready", "accel": accel_parts, "accel_log": accel_log}
    proto.write((json.dumps(ready) + "\n").encode("utf-8"))

    while True:
        header = _worker._read_header(stdin)
        if header is None:
            break  # parent closed stdin -> shut down
        seq = int(header.get("seq", -1))
        n, h, w, scale = int(header["n"]), int(header["h"]), int(header["w"]), int(header["scale"])
        payload = _read_payload(stdin, n * h * w * 3)

        try:
            frames = torch.from_numpy(np.frombuffer(payload, dtype=np.uint8).reshape(n, h, w, 3))
            layout = layout_for(w, h, scale)
            t_window = time.perf_counter()
            restored = _restore_window(pipe, torch, frames, layout, clip_len, device)
            t_restored = time.perf_counter()
            for index in range(n):
                frame = layout.blend(torch, [tile[index].to(device) for tile in restored])
                if color_fix_method != "none":
                    source = frames[index:index + 1].to(device)
                    frame = _worker._color_fix_frames_gpu(frame[None], source, color_fix_method)[0]
                data = frame.contiguous().cpu().numpy()
                resp = {"seq": seq, "i": index, "h": int(data.shape[0]), "w": int(data.shape[1])}
                _write_all(proto, (json.dumps(resp) + "\n").encode("utf-8"))
                _write_all(proto, data.data)
            del frames, restored
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            if args.verbose:
                print(
                    f"[swiftvr-frame-worker] window {seq}: {n} frames, restore {t_restored - t_window:.1f}s, "
                    f"blend and send {time.perf_counter() - t_restored:.1f}s",
                    file=sys.stderr, flush=True,
                )
        except Exception as e:  # keep the worker alive; the parent fails the job
            traceback.print_exc()
            err = json.dumps({"seq": seq, "error": f"{type(e).__name__}: {e}"}) + "\n"
            proto.write(err.encode("utf-8"))


if __name__ == "__main__":
    main()

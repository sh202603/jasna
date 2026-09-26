# SPDX-FileCopyrightText: 2026 sh202603
# SPDX-License-Identifier: AGPL-3.0
"""
Persistent SwiftVR inference worker — runs inside the SwiftVR venv (NOT the
jasna venv), spawned by ``SwiftvrInlineSecondaryRestorer``.

Loads ``SwiftVRPipeline`` once (weights resident) and, for each clip received
over stdin/stdout, upscales the 256px primary crops to 256*scale px
(``--scale``: 4 = the model-native 1024px, 2 = 512px) with
``SwiftVRPipeline.restore_clip()`` and streams them back. SwiftVR restores in
fixed causal chunks, so its VRAM does not grow with the clip length, and with
FP8 it co-resides with jasna's primary pipeline on a 16 GB GPU.

Output crops get an always-on color correction against the bicubic-upscaled
input (``--color-fix-method``), the same wavelet/AdaIN math as the FlashVSR
worker: the primitives are loaded by path from the sibling
``flashvsr_inline_worker.py`` (whose top level is stdlib-only) and applied on
the GPU here, since SwiftVR's output already lives there.

This script is intentionally free of any ``jasna`` (or ``lada``) import: it is
executed by the SwiftVR env's Python (``--swiftvr-python``) and only uses
numpy/torch plus the ``swiftvr`` package (the checkout is added to ``sys.path``
as a fallback to its editable install). The wire color order is lada-native
BGR, like the FlashVSR and SeedVR2 workers, so the same file can serve lada-ex;
the parent adapter flips RGB<->BGR around the wire.

Acceleration (``--fp8-dit``, ``--torch-compile``; both passed by the parent for
``--swiftvr-accel``) is decided BEFORE the ~20 GB model load: FP8 needs an
sm89+ GPU, and both parts need a Triton that can build and run kernels (on
Linux that means the base Python's dev headers). Whatever fails is switched
off with a reason in the handshake; there is no runtime demotion, since
SwiftVR's FP8 replaces the DiT linears in place and has no fallback path.

Wire protocol (parent = jasna venv, child = this):
  parent -> child : header ``{"seq","n","h","w"}\\n`` (UTF-8) then n*h*w*3 raw
                    uint8 BGR bytes  (the 256px primary crops, HWC)
  child  -> parent: header ``{"seq","n","h","w"}\\n`` then n*h*w*3 raw uint8
                    BGR bytes  (the 256*scale px restored crops, HWC), exactly
                    n frames.
  child  -> parent (once, at startup):
                    ``{"status":"ready","accel":[...],"accel_log":[...]}\\n``
                    (the active acceleration parts among ``fp8_dit`` /
                    ``torch_compile``, and the decision log lines)
  child  -> parent (on per-clip failure): ``{"seq","error":"..."}\\n`` then
                    stays alive for the next clip.

fd handling: the *real* stdout fd is dup'd to a private protocol fd, then fd 1
is repointed to /dev/null (or stderr if --verbose) so SwiftVR's progress
prints never corrupt the wire.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

# Frames per MIDDLE chunk of SwiftVR's fixed causal protocol (its default).
DEFAULT_CLIP_LEN = 24


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SwiftVR worker (256px crops -> 256*scale px)")
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
                         "warning where unsupported).")
    ap.add_argument("--torch-compile", action="store_true",
                    help="torch.compile the DiT blocks (needs a working Triton; switched off "
                         "with a warning otherwise).")
    ap.add_argument(
        "--color-fix-method",
        default="wavelet",
        choices=["adain", "wavelet", "none"],
        help="Color correction of the output crops against the bicubic-upscaled input "
             "(always on in production; 'none' exists only for A/B baselines and is "
             "reachable via the JASNA_SWIFTVR_COLOR_FIX env override, not the CLI).",
    )
    ap.add_argument("--verbose", action="store_true", help="send worker stdout to stderr, not /dev/null")
    return ap.parse_args()


def _install_protocol_fd(verbose: bool):
    """Reserve the real stdout as the protocol channel and mute fd 1.

    Returns a binary file object for writing protocol messages. Reads use the
    real stdin (fd 0). After this, any library print()/progress line goes to
    /dev/null (or stderr when --verbose) instead of the wire.
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


# ---------------------------------------------------------------------------- #
# Color correction (always on in production)
#
# Same method and reference as the FlashVSR worker: the output crop's high
# frequencies on the low frequencies of the bicubic-upscaled input crop
# (wavelet), or a mean/std transfer (adain). The primitives are shared by
# loading the sibling worker by path (no copy to drift, and the FlashVSR worker
# stays verbatim-identical with lada-ex's). Unlike its ``_color_fix_frames``,
# which ferries host float32 frames to the device one by one, this runs on
# device-resident uint8 tensors, since ``restore_clip`` returns them there.
# ---------------------------------------------------------------------------- #

_PRIMITIVES = None


def _color_fix_primitives():
    """``flashvsr_inline_worker`` loaded by path (cached); stdlib-only top level."""
    global _PRIMITIVES
    if _PRIMITIVES is None:
        import importlib.util

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flashvsr_inline_worker.py")
        spec = importlib.util.spec_from_file_location("_swiftvr_color_fix_primitives", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PRIMITIVES = module
    return _PRIMITIVES


def _color_fix_frames_gpu(out_u8, lq_u8, method: str):
    """Correct ``out_u8`` (n, oh, ow, 3) uint8 RGB against ``lq_u8`` (n, h, w, 3)
    uint8 RGB, both on the same device; returns (n, oh, ow, 3) uint8 there.

    Numerically the same as ``flashvsr_inline_worker._color_fix_frames`` on the
    same float32 inputs (content = out/255, style = bicubic(lq/255) to the
    output size, per frame, then clamp, *255, round-half-even): only the
    host/device traffic differs.
    """
    import torch
    import torch.nn.functional as F

    prims = _color_fix_primitives()
    n, oh, ow = int(out_u8.shape[0]), int(out_u8.shape[1]), int(out_u8.shape[2])
    fixed_all = torch.empty_like(out_u8)
    for i in range(n):
        content = out_u8[i].permute(2, 0, 1).unsqueeze(0).to(torch.float32).div_(255.0)
        style = lq_u8[i].permute(2, 0, 1).unsqueeze(0).to(torch.float32).div_(255.0)
        style = F.interpolate(style, size=(oh, ow), mode="bicubic", align_corners=False)
        if method == "adain":
            fixed = prims._adain(content, style)
        else:
            fixed = prims._wavelet_reconstruct(content, style)
        fixed_all[i] = (
            fixed.clamp_(0.0, 1.0).mul_(255.0).round_().squeeze(0).permute(1, 2, 0).to(torch.uint8)
        )
    return fixed_all


# ---------------------------------------------------------------------------- #
# Acceleration decision (before the model load)
# ---------------------------------------------------------------------------- #

def _triton_works(device) -> tuple[bool, str]:
    """Build and run a trivial Triton kernel once.

    Catches, in a second and before the 20 GB checkpoint is read, the failure
    mode both accelerated parts share: Triton's launcher build needs the base
    Python's dev headers on Linux (``fatal error: Python.h``).
    """
    try:
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def _probe(x_ptr, n, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            mask = offs < n
            tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1.0, mask=mask)

        x = torch.zeros(16, device=device, dtype=torch.float32)
        _probe[(1,)](x, 16, BLOCK=16)
        torch.cuda.synchronize(device)
        if not bool((x == 1.0).all()):
            return False, "probe kernel produced wrong values"
        return True, ""
    except Exception as e:  # ImportError, CalledProcessError (launcher build), RuntimeError
        lines = [ln for ln in str(e).strip().splitlines() if ln.strip()]
        return False, f"{type(e).__name__}: {lines[-1] if lines else ''}"


def _decide_accel(want_fp8: bool, want_compile: bool, device) -> tuple[bool, bool, list, list]:
    """Return ``(fp8, compile, active_parts, log_lines)`` for the handshake."""
    parts: list[str] = []
    log: list[str] = []
    fp8, comp = bool(want_fp8), bool(want_compile)
    if not (fp8 or comp):
        return False, False, parts, log

    import torch
    from swiftvr.models.fp8 import fp8_supported

    if fp8 and not fp8_supported(device):
        cap = torch.cuda.get_device_capability(torch.device(device))
        log.append(
            f"fp8_dit disabled: needs compute capability 8.9+ (RTX 40 series or newer), this GPU "
            f"is {cap[0]}.{cap[1]}; the DiT runs in bf16 (~12 GiB, which does not co-reside with "
            "the primary pipeline on a 16 GB card)."
        )
        fp8 = False
    if fp8 or comp:
        ok, why = _triton_works(device)
        if not ok:
            wanted = " and ".join(p for p, on in (("fp8_dit", fp8), ("torch_compile", comp)) if on)
            log.append(
                f"{wanted} disabled: Triton cannot build or run kernels here ({why}); on Linux "
                "install the dev headers (python3.X-dev) of the venv's base Python."
            )
            fp8 = comp = False
    if fp8:
        parts.append("fp8_dit")
    if comp:
        parts.append("torch_compile")
    if parts:
        log.append("enabled " + ", ".join(parts) + ".")
    return fp8, comp, parts, log


def _restore_checked(pipe, torch, lq, scale: int, clip_len: int):
    """``restore_clip`` with one retry on an out-of-VRAM error and the
    frame-count contract checked (a mismatch would silently corrupt the blend
    alignment in the parent)."""
    oom = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    n = int(lq.shape[0])
    for attempt in (1, 2):
        try:
            out = pipe.restore_clip(lq, upscale=scale, clip_len=clip_len)
        except oom:
            if attempt == 2:
                raise
            print("[swiftvr-worker] out of VRAM mid-clip; retrying once after releasing the cache",
                  file=sys.stderr, flush=True)
            torch.cuda.empty_cache()
            continue
        if int(out.shape[0]) != n:
            raise RuntimeError(f"SwiftVR returned {int(out.shape[0])} frames for a {n}-frame clip")
        return out
    raise AssertionError("unreachable")


def main() -> None:
    args = _parse_args()
    if args.clip_len % 4 != 0 or args.clip_len < 4:
        raise SystemExit(f"--clip-len must be a positive multiple of 4, got {args.clip_len}")

    # Mute progress output before importing anything noisy; reserve the protocol fd.
    os.environ.setdefault("TQDM_DISABLE", "1")
    proto = _install_protocol_fd(args.verbose)
    stdin = sys.stdin.buffer

    # The venv normally has the checkout installed (uv sync, editable); the
    # sys.path entry is the fallback for a venv built elsewhere.
    sys.path.insert(0, args.repo)
    import numpy as np
    import torch
    from swiftvr import SwiftVRPipeline

    if not args.verbose:
        # The fork's to() casts the whole DiT to bf16 on purpose; diffusers'
        # "modules that should be kept in float32" warning about it is noise.
        try:
            from diffusers.utils import logging as diffusers_logging
            diffusers_logging.set_verbosity_error()
        except Exception:
            pass

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)

    fp8, comp, accel_parts, accel_log = _decide_accel(args.fp8_dit, args.torch_compile, device)
    for line in accel_log:
        print(f"[swiftvr-worker] accel: {line}", file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    pipe = SwiftVRPipeline.from_pretrained(args.model_dir).to(
        device, dtype="bfloat16", attention_backend="auto", torch_compile=comp,
        cudnn_benchmark=False, fp8_dit=fp8)
    t_load = time.perf_counter() - t0

    scale = int(args.scale)
    clip_len = int(args.clip_len)
    color_fix_method = args.color_fix_method

    # Warm up at the shape every clip has (256px crops): 2*clip_len+5 frames
    # split into FIRST (clip_len+4), MIDDLE (clip_len) and LAST (1), which
    # under torch.compile builds all four DiT graphs (clip_len//4+1 and
    # clip_len//4 latents, with and without the window shift). Random pixels,
    # not zeros: FP8's activation scale is an abs-max.
    t1 = time.perf_counter()
    warm_n = 2 * clip_len + 5
    warm = torch.randint(0, 256, (warm_n, 256, 256, 3), dtype=torch.uint8, device=device)
    warm_out = pipe.restore_clip(warm, upscale=scale, clip_len=clip_len)
    if tuple(warm_out.shape) != (warm_n, 256 * scale, 256 * scale, 3):
        raise RuntimeError(f"warmup produced {tuple(warm_out.shape)}, expected {(warm_n, 256 * scale, 256 * scale, 3)}")
    if color_fix_method != "none":
        _color_fix_frames_gpu(warm_out, warm, color_fix_method)
    del warm, warm_out
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    print(f"[swiftvr-worker] ready: model load {t_load:.1f}s, warmup {time.perf_counter() - t1:.1f}s "
          f"(scale {scale}, clip_len {clip_len}, accel: {', '.join(accel_parts) or 'off'})",
          file=sys.stderr, flush=True)

    # Handshake: tell the parent we're ready to accept clips.
    ready = {"status": "ready", "accel": accel_parts, "accel_log": accel_log}
    proto.write((json.dumps(ready) + "\n").encode("utf-8"))

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
            # Wire is BGR (lada-native); SwiftVR wants RGB.
            lq = torch.from_numpy(np.ascontiguousarray(crops[..., ::-1])).to(device)  # (n,h,w,3) uint8

            out = _restore_checked(pipe, torch, lq, scale, clip_len)  # (n, h*scale, w*scale, 3) uint8
            if color_fix_method != "none":
                out = _color_fix_frames_gpu(out, lq, color_fix_method)
            # RGB (SwiftVR output) -> BGR (lada wire), then to the host.
            out_arr = out.flip(-1).contiguous().cpu().numpy()
            oh, ow = int(out_arr.shape[1]), int(out_arr.shape[2])

            resp = {"seq": seq, "n": n, "h": oh, "w": ow}
            proto.write((json.dumps(resp) + "\n").encode("utf-8"))
            proto.write(out_arr.tobytes())

            del lq, out, out_arr
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()  # co-residence discipline: return reserved each clip
        except Exception as e:  # keep the worker alive; report per-clip failure
            traceback.print_exc()
            err = json.dumps({"seq": seq, "error": f"{type(e).__name__}: {e}"}) + "\n"
            proto.write(err.encode("utf-8"))


if __name__ == "__main__":
    main()

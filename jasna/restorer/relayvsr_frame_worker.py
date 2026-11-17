# SPDX-FileCopyrightText: 2026 sh202603
# SPDX-License-Identifier: AGPL-3.0
"""
Persistent RelayVSR worker for whole frames: runs inside the RelayVSR venv (NOT
the jasna venv), spawned by ``jasna.upscale.relayvsr.RelayvsrWorker`` for
``jasna-postprocess --upscale relayvsr``.

RelayVSR keeps its state (the Wan K/V cache) over the whole video and emits a
frame only once the keyframe after it exists, so the answer to one input frame
is zero or more output frames. The video's frames are handed to
``stream.frames()`` as a generator that reads stdin: the model pulls the next
frame when it needs one, and the worker announces that pull with ``more``. The
parent sends one frame and reads until ``more``, so the two never write at the
same time and neither has to know how RelayVSR cuts its windows.

One stream is loaded at a time. A video that needs the other factor names the
other pair of checkpoints in its ``begin``, and the stream is rebuilt: two
pairs would hold the Wan weights twice.

The top level is stdlib-only, so jasna's tests can import this file.

Wire protocol (parent = jasna venv, child = this); frames are RGB uint8, HWC:
  child  -> parent (once): ``{"status":"ready","scale","wan_weights"}\\n``
  parent -> child : ``{"op":"begin","h","w","wan","flash"}\\n``
  child  -> parent: ``{"status":"begin","scale","wan_weights"}\\n``
  parent -> child : ``{"op":"frame"}\\n`` then h*w*3 bytes
  child  -> parent: zero or more [``{"i","h","w"}\\n`` then h*w*3 bytes], then
                    ``{"more":true}\\n``
  parent -> child : ``{"op":"end"}\\n``
  child  -> parent: the remaining frames, then
                    ``{"done":true,"frames":n,"kv_cache":"gpu"|"cpu"}\\n``
  child  -> parent (on failure, instead of the next answer):
                    ``{"error":"..."}\\n``; the video is over and the worker
                    waits for the next ``begin``.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback

DEFAULT_WINDOW_SIZE = 16
# The same input gives the same output (RelayVSR's own default).
_SEED = 42


class _ProtocolError(Exception):
    """The parent sent something the protocol does not allow here; stdin cannot be trusted
    after it, so the worker exits."""


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RelayVSR whole-frame worker (one stream per video)")
    ap.add_argument("--wan-checkpoint", required=True, help="Wan checkpoint of the first video's factor")
    ap.add_argument("--flash-checkpoint", required=True, help="Flash checkpoint of the same factor")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="Keyframe interval + 1")
    ap.add_argument("--wan-weights", default="bf16", choices=["bf16", "fp8"])
    ap.add_argument("--kv-cache", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--kv-cache-dtype", default="native", choices=["native", "int8"])
    ap.add_argument("--verbose", action="store_true", help="send worker stdout to stderr, not /dev/null")
    return ap.parse_args()


def _install_protocol_fd(verbose: bool):
    """Reserve the real stdout as the protocol channel and mute fd 1 (stderr when verbose), so
    a library's print() cannot land on the wire."""
    proto = os.fdopen(os.dup(1), "wb", buffering=0)
    sink = sys.stderr.fileno() if verbose else os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink, 1)
    return proto


def _read_header(stream) -> dict | None:
    """One newline-terminated JSON header, or ``None`` on EOF (the parent is gone)."""
    line = stream.readline()
    return json.loads(line.decode("utf-8")) if line.endswith(b"\n") else None


def _read_payload(stream, size: int) -> bytearray:
    """Exactly ``size`` bytes into a writable buffer of its own: the model keeps a window of
    frames, and torch shares the buffer without a copy."""
    buffer = bytearray(size)
    view = memoryview(buffer)
    while view:
        count = stream.readinto(view)
        if not count:
            raise EOFError("stdin closed while reading a frame")
        view = view[count:]
    return buffer


def _write_all(stream, data) -> None:
    """The protocol stream is unbuffered, so one write may take only part of a frame."""
    view = memoryview(data).cast("B")
    while view:
        view = view[stream.write(view):]


def _send(proto, message: dict) -> None:
    _write_all(proto, (json.dumps(message) + "\n").encode("utf-8"))


def _log(text: str) -> None:
    print(f"[relayvsr-frame-worker] {text}", file=sys.stderr, flush=True)


def _source(stdin, proto, torch, input_frame, height: int, width: int):
    """The video's frames as ``stream.frames()`` pulls them. ``more`` goes out when the model
    asks for the frame after one it was given: everything it could emit so far is on the wire."""
    index = 0
    while True:
        header = _read_header(stdin)
        if header is None:
            raise EOFError("stdin closed inside a video")
        op = header.get("op")
        if op == "end":
            return
        if op != "frame":
            raise _ProtocolError(f"expected a frame or the end of the video, got {header}")
        data = _read_payload(stdin, height * width * 3)
        rgb = torch.frombuffer(data, dtype=torch.uint8).view(height, width, 3).permute(2, 0, 1)
        yield input_frame(index, rgb)
        index += 1
        _send(proto, {"more": True})


def _run_video(stream, stdin, proto, torch, input_frame, to_uint8, height: int, width: int) -> dict:
    count = 0
    for out in stream.frames(_source(stdin, proto, torch, input_frame, height, width)):
        frame = to_uint8(out.rgb).cpu().numpy()
        _send(proto, {"i": int(out.index), "h": int(frame.shape[0]), "w": int(frame.shape[1])})
        _write_all(proto, frame.data)
        count += 1
    return {"done": True, "frames": count, "kv_cache": stream.resolved_wan_kv_cache}


def main() -> None:
    args = _parse_args()
    if args.window_size < 2:
        raise SystemExit(f"--window-size must be at least 2, got {args.window_size}")

    os.environ.setdefault("TQDM_DISABLE", "1")
    proto = _install_protocol_fd(args.verbose)
    stdin = sys.stdin.buffer

    import torch
    from refvsr_inference.runtime.session import build_stream, to_uint8
    from refvsr_inference.runtime.streaming import InputFrame

    cuda = str(args.device).startswith("cuda")

    def load(wan: str, flash: str):
        started = time.perf_counter()
        stream = build_stream(
            wan, flash,
            device=args.device, dtype="bf16", window_size=args.window_size, seed=_SEED,
            kv_cache=args.kv_cache, kv_cache_dtype=args.kv_cache_dtype, wan_weights=args.wan_weights,
        )
        _log(
            f"loaded the {stream.scale}x pair in {time.perf_counter() - started:.1f}s "
            f"(Wan weights {stream.wan_weights}, window size {args.window_size})"
        )
        return stream

    def release() -> None:
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()

    loaded = (os.path.abspath(args.wan_checkpoint), os.path.abspath(args.flash_checkpoint))
    stream = load(*loaded)
    _send(proto, {"status": "ready", "scale": int(stream.scale), "wan_weights": stream.wan_weights})

    while True:
        header = _read_header(stdin)
        if header is None:
            break  # parent closed stdin -> shut down
        try:
            if header.get("op") != "begin":
                raise _ProtocolError(f"expected the beginning of a video, got {header}")
            height, width = int(header["h"]), int(header["w"])
            wanted = (os.path.abspath(header["wan"]), os.path.abspath(header["flash"]))
            if stream is None or wanted != loaded:
                stream = None
                release()  # the old Wan weights leave before the new ones arrive
                stream = load(*wanted)
                loaded = wanted
            _send(proto, {"status": "begin", "scale": int(stream.scale), "wan_weights": stream.wan_weights})
            started = time.perf_counter()
            done = _run_video(stream, stdin, proto, torch, InputFrame, to_uint8, height, width)
            if args.verbose:
                elapsed = time.perf_counter() - started
                _log(
                    f"{done['frames']} frames of {width}x{height} at {stream.scale}x in {elapsed:.1f}s "
                    f"({done['frames'] / max(elapsed, 1e-9):.2f} fps, K/V cache on the {done['kv_cache']})"
                )
            # The K/V cache of the finished video must not sit next to the next video's.
            stream.reset()
            release()
            _send(proto, done)
        except (EOFError, _ProtocolError):
            traceback.print_exc()
            break
        except Exception as e:  # the video failed; the worker stays for the next one
            traceback.print_exc()
            error = f"{type(e).__name__}: {e}"
        else:
            continue
        if stream is not None:
            stream.reset()
        release()
        _send(proto, {"error": error})


if __name__ == "__main__":
    main()

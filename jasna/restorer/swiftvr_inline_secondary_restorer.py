"""Inline SwiftVR secondary restorer (synchronous ``SecondaryRestorer``).

Spawns a resident SwiftVR-venv worker (``swiftvr_inline_worker.py``) that
upscales each clip's 256px primary crops to 256*scale px (``scale`` 4 = the
model-native 1024px, 2 = 512px) with ``SwiftVRPipeline.restore_clip()`` and
streams them back over a length-prefixed stdin/stdout protocol. Runs inside
jasna's normal streaming pipeline — no bundle, no intermediate files — because
SwiftVR restores in fixed causal chunks (VRAM independent of the clip length)
and, with FP8 (``accel``, the default), co-resides with the primary pipeline
inside 16 GB.

The worker speaks the same wire as the FlashVSR and SeedVR2 workers (lada-native
**BGR**) and imports neither jasna nor lada, so it can serve lada-ex unchanged;
this adapter flips RGB<->BGR around the wire. The worker also applies an
always-on color correction of the output crops against the bicubic-upscaled
input (wavelet by default; ``JASNA_SWIFTVR_COLOR_FIX`` is a verification-only
override, see ``__init__``).

Synchronous by choice, like the FlashVSR inline restorer: SwiftVR is the rate
limiter, so an async/out-of-order restorer buys nothing, and
``AsyncSecondaryRestorer`` is a structural Protocol (defining ``push_clip``
etc. would silently route this to the async loop). Only the sync
``SecondaryRestorer`` methods are implemented.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import torch

from jasna.restorer import bundled_script_path
from jasna.restorer.swiftvr_common import check_restore_clip_api

logger = logging.getLogger(__name__)

# Benign co-residence noise emitted by the worker's torch when expandable_segments
# cannot memory-map under VRAM pressure (harmless, not a crash). Dropped from the
# worker's stderr only at --log-level error; genuine stderr (tracebacks) is
# forwarded unchanged. Matched on bytes to avoid a decode step.
_STDERR_SUPPRESS_RE = re.compile(rb"expandable_segments: memory mapping failed with OOM")

# The 20 GB checkpoint read plus the warmup (and, with accel, the torch.compile
# of the four DiT graphs) take longer than the FlashVSR worker's startup.
DEFAULT_STARTUP_TIMEOUT_S = 600.0


def _resolve_worker_script() -> Path:
    """Path to the worker script (module-level for test monkeypatching)."""
    return bundled_script_path("swiftvr_inline_worker.py")


class SwiftvrInlineSecondaryRestorer:
    name = "swiftvr-inline"
    num_workers = 1
    prefers_cpu_input = True  # crops handed to restore() are on CPU (host RAM)

    def __init__(
        self,
        *,
        repo: Path,
        model_dir: Path,
        sv_python: Path,
        device: str = "cuda:0",
        scale: int = 4,
        accel: bool = True,
        log_level: str = "error",
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
        verbose: bool = False,
    ) -> None:
        if int(scale) not in (2, 4):
            raise ValueError(f"[swiftvr-inline] scale must be 2 or 4, got {scale}")
        check_restore_clip_api(Path(repo))
        worker = _resolve_worker_script()
        if not worker.is_file():
            raise FileNotFoundError(f"inline worker script not found: {worker}")

        cmd = [
            str(sv_python), str(worker),
            "--repo", str(repo),
            "--model-dir", str(model_dir),
            "--device", str(device),
            "--scale", str(int(scale)),
        ]
        if accel:
            # One switch on the jasna side (like --flashvsr-accel); the worker
            # takes the two parts separately and drops whichever its GPU or
            # Triton cannot run, reporting that in the handshake.
            cmd += ["--fp8-dit", "--torch-compile"]
        # Color correction is always on (worker default: wavelet).
        # JASNA_SWIFTVR_COLOR_FIX is a verification-only override
        # (adain|wavelet|none) for A/B runs — deliberately an env var, not a
        # CLI flag (same contract as the FlashVSR restorer).
        color_fix = os.environ.get("JASNA_SWIFTVR_COLOR_FIX")
        if color_fix:
            if color_fix not in ("adain", "wavelet", "none"):
                raise ValueError(
                    "[swiftvr-inline] JASNA_SWIFTVR_COLOR_FIX must be adain|wavelet|none, "
                    f"got {color_fix!r}"
                )
            cmd += ["--color-fix-method", color_fix]
        if verbose:
            cmd.append("--verbose")

        # The SwiftVR venv must import its own package, not inherit jasna's
        # PYTHONPATH.
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        if os.name == "nt":
            # expandable_segments is not supported on Windows; force UTF-8 so a
            # non-ASCII log line cannot raise on the cp932 text layer our
            # stderr pipe gets by default.
            env["PYTHONUTF8"] = "1"
        else:
            # expandable_segments keeps the worker's reserved VRAM tight next
            # to the primary pipeline (same discipline as the FlashVSR worker).
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        self._accel_requested = bool(accel)
        self._lock = threading.Lock()
        self._closed = False
        # At --log-level error, mute the worker's benign expandable_segments OOM
        # warnings by piping stderr through a line filter (still forwarding real
        # stderr like tracebacks). At info/warning/debug, inherit stderr unchanged.
        self._quiet = str(log_level).lower() == "error"
        self._stderr_thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._cmd = cmd
        self._env = env
        self._startup_timeout_s = startup_timeout_s
        self._spawn()

    # -- lifecycle -----------------------------------------------------------

    def _spawn(self) -> None:
        """Start the worker and block until its ready handshake."""
        logger.info("[swiftvr-inline] spawning worker: %s", " ".join(self._cmd))
        self._proc = subprocess.Popen(
            self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if self._quiet else None, env=self._env,
        )
        if self._quiet:
            self._stderr_thread = threading.Thread(
                target=self._pump_stderr, name="swiftvr-inline-stderr", daemon=True
            )
            self._stderr_thread.start()
        self._await_ready(self._startup_timeout_s)

    def _pump_stderr(self) -> None:
        """Forward the worker's stderr, dropping only the benign expandable_segments
        OOM warnings. Runs on a daemon thread when --log-level error; it ends on
        stderr EOF (worker exit)."""
        stream = self._proc.stderr
        if stream is None:
            return
        try:
            for raw in iter(stream.readline, b""):
                if _STDERR_SUPPRESS_RE.search(raw):
                    continue
                self._emit_stderr(raw)
        except (ValueError, OSError):
            pass  # stream closed during shutdown

    @staticmethod
    def _emit_stderr(raw: bytes) -> None:
        try:
            buf = getattr(sys.stderr, "buffer", None)
            if buf is not None:
                buf.write(raw)
                buf.flush()
            else:  # pytest / captured stderr has no .buffer
                sys.stderr.write(raw.decode("utf-8", "replace"))
                sys.stderr.flush()
        except Exception:
            pass

    def _await_ready(self, timeout_s: float) -> None:
        """Block until the worker's ``{"status":"ready"}`` handshake (or fail)."""
        result: dict = {}

        def _reader():
            result["header"] = self._read_header()

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            self._kill()
            raise RuntimeError(
                f"[swiftvr-inline] worker did not become ready within {timeout_s:.0f}s "
                "(model load stalled?)"
            )
        header = result.get("header")
        if not header or header.get("status") != "ready":
            self._kill()
            hint = " (if the failure is in FP8 or torch.compile, retry with --no-swiftvr-accel)" \
                if self._accel_requested else ""
            raise RuntimeError(f"[swiftvr-inline] worker failed to start: {header}{hint}")
        # The worker's stdout is muted, so it relays its acceleration decisions
        # and the active parts in the handshake.
        for line in header.get("accel_log") or ():
            level = logging.INFO if line.startswith("enabled") else logging.WARNING
            logger.log(level, "[swiftvr-inline] accel: %s", line)
        accel = list(header.get("accel") or [])
        logger.info("[swiftvr-inline] worker ready (acceleration: %s)",
                    ", ".join(accel) if accel else "off")
        if self._accel_requested and "fp8_dit" not in accel:
            logger.warning(
                "[swiftvr-inline] SwiftVR runs its DiT in bf16 (~12 GiB peak): on a 16 GB GPU "
                "this will not fit next to the primary pipeline and the run may fail with an "
                "out-of-VRAM error mid-clip."
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._kill()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2)

    def _kill(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass

    # -- wire helpers --------------------------------------------------------

    def _read_header(self) -> dict | None:
        stream = self._proc.stdout
        line = bytearray()
        while True:
            b = stream.read(1)
            if not b:
                return None
            if b == b"\n":
                break
            line.extend(b)
        return json.loads(line.decode("utf-8"))

    def _read_exact(self, n: int) -> bytes:
        stream = self._proc.stdout
        buf = bytearray()
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk:
                raise RuntimeError("[swiftvr-inline] worker closed the pipe mid-payload")
            buf.extend(chunk)
        return bytes(buf)

    # -- SecondaryRestorer ---------------------------------------------------

    def restore(
        self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int
    ) -> list[torch.Tensor]:
        t = int(frames_256.shape[0])
        ks = max(0, int(keep_start))
        ke = min(t, int(keep_end))
        if t == 0 or ks >= ke:
            return []

        # (T,C,256,256) float [0,1] RGB -> (T,256,256,3) uint8 BGR HWC,
        # C-contiguous (the wire is lada-native BGR). copy=True: a CPU float32
        # input would otherwise be returned as-is by .to() and the in-place ops
        # below would rescale the caller's tensor.
        hwc = (
            frames_256.detach().to("cpu", torch.float32, copy=True).clamp_(0.0, 1.0)
            .mul_(255.0).round_().to(torch.uint8)
            .permute(0, 2, 3, 1).contiguous().numpy()
        )
        hwc = np.ascontiguousarray(hwc[..., ::-1])
        h, w = int(hwc.shape[1]), int(hwc.shape[2])

        with self._lock:
            if self._closed or self._proc.poll() is not None:
                raise RuntimeError("[swiftvr-inline] worker process is not running")
            header = json.dumps({"seq": 0, "n": t, "h": h, "w": w}) + "\n"
            self._proc.stdin.write(header.encode("utf-8"))
            self._proc.stdin.write(hwc.tobytes())
            self._proc.stdin.flush()

            resp = self._read_header()
            if resp is None:
                raise RuntimeError("[swiftvr-inline] worker died (no response)")
            if "error" in resp:
                raise RuntimeError(f"[swiftvr-inline] worker error: {resp['error']}")
            rn, rh, rw = int(resp["n"]), int(resp["h"]), int(resp["w"])
            data = self._read_exact(rn * rh * rw * 3)

        # Frame-count contract: a mismatch silently corrupts the blend alignment.
        if rn != t:
            raise RuntimeError(
                f"[swiftvr-inline] frame-count mismatch: sent {t}, got {rn}"
            )
        out = np.frombuffer(data, dtype=np.uint8).reshape(rn, rh, rw, 3)  # BGR
        # keep window, BGR -> RGB, -> CHW uint8 CPU tensors (blend moves each
        # to device).
        kept = np.ascontiguousarray(out[ks:ke, ..., ::-1].transpose(0, 3, 1, 2))
        return list(torch.from_numpy(kept).unbind(0))

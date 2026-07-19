"""Inline FlashVSR secondary restorer (synchronous ``SecondaryRestorer``).

Spawns a resident FlashVSR-venv worker (``flashvsr_inline_worker.py``) that
upscales each clip's 256px primary crops to 1024px (4x) with the FlashVSR
**tiny-long** pipeline, and streams them back over a length-prefixed
stdin/stdout protocol (RGB payload). Runs inside jasna's normal streaming
pipeline — no bundle, no 1024px intermediate files — because tiny-long is O(1)
in VRAM and co-resides with the primary pipeline inside 16GB (see
FLASHVSR_INLINE_FEASIBILITY §12).

This is the inline counterpart of the offline 3-phase path
(``flashvsr_offline.py``), which stays as the fallback for 12GB-class GPUs and
un-patched FlashVSR checkouts.

Synchronous by choice: FlashVSR (~15 crop-fps) is the rate limiter, so an
async/out-of-order restorer buys nothing, and ``AsyncSecondaryRestorer`` is a
structural Protocol (defining ``push_clip`` etc. would silently route this to
the async loop). We implement only the sync ``SecondaryRestorer`` methods.
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
from typing import TYPE_CHECKING

import numpy as np
import torch

from jasna.restorer import bundled_script_path

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Benign co-residence noise emitted by the worker's torch when expandable_segments
# cannot memory-map under VRAM pressure (see FLASHVSR docs: harmless, not a crash).
# Dropped from the worker's stderr only at --log-level error; genuine stderr
# (tracebacks) is forwarded unchanged. Matched on bytes to avoid a decode step.
_STDERR_SUPPRESS_RE = re.compile(rb"expandable_segments: memory mapping failed with OOM")

# Marker left by tinylong_multichunk_fix.patch. The inline worker needs the
# tiny-long multi-chunk fix; without it tiny-long crashes on the 2nd chunk.
_FIX_MARKER = "FIX(jasna)"
_TINYLONG_REL = Path("src") / "pipelines" / "flashvsr_tiny_long.py"


def _resolve_worker_script() -> Path:
    """Path to the worker script (module-level for test monkeypatching)."""
    return bundled_script_path("flashvsr_inline_worker.py")


def _check_patched_repo(repo: Path) -> None:
    """Fail fast if the FlashVSR checkout lacks the tiny-long multi-chunk fix."""
    tinylong = repo / _TINYLONG_REL
    if not tinylong.is_file():
        raise FileNotFoundError(f"FlashVSR tiny-long pipeline not found: {tinylong}")
    if _FIX_MARKER not in tinylong.read_text(encoding="utf-8", errors="ignore"):
        raise RuntimeError(
            f"FlashVSR checkout at {repo} is missing the tiny-long multi-chunk fix.\n"
            "  --secondary-restoration flashvsr-inline uses tiny-long, which crashes on the\n"
            "  second chunk without the fix. Apply tinylong_multichunk_fix.patch to the\n"
            "  checkout, or use --secondary-restoration flashvsr (offline, tiny-mode)."
        )


class FlashvsrInlineSecondaryRestorer:
    name = "flashvsr-inline"
    num_workers = 1
    prefers_cpu_input = True  # crops handed to restore() are on CPU (host RAM)

    def __init__(
        self,
        *,
        repo: Path,
        model_dir: Path,
        fv_python: Path,
        version: str = "11",
        dtype: str = "bf16",
        device: str = "cuda:0",
        scale: int = 4,
        tiles: int = 1,
        log_level: str = "error",
        startup_timeout_s: float = 300.0,
        verbose: bool = False,
    ) -> None:
        _check_patched_repo(Path(repo))
        worker = _resolve_worker_script()
        if not worker.is_file():
            raise FileNotFoundError(f"inline worker script not found: {worker}")

        cmd = [
            str(fv_python), str(worker),
            "--repo", str(repo),
            "--model-dir", str(model_dir),
            "--version", str(version),
            "--dtype", str(dtype),
            "--device", str(device),
            "--scale", str(int(scale)),
            "--tiles", str(int(tiles)),
        ]
        if verbose:
            cmd.append("--verbose")

        # The FlashVSR venv must import its own repo, not inherit jasna's
        # PYTHONPATH.
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        if os.name == "nt":
            # expandable_segments is not supported on Windows (torch warns and
            # falls back to the default caching allocator), so the worker's
            # reserved VRAM runs ~1-2GB above the Linux flat budget from
            # fragmentation, plus ~1GB held by the WDDM desktop. Measured on a
            # 16GB RTX 5080: 256px clips peak ~13GB reserved, so co-residence
            # with the primary does not fit in 16GB. Warn (larger GPUs still
            # work) and force UTF-8: tiny-long's block-character init banner
            # raises UnicodeEncodeError on the cp932 text layer our stdout pipe
            # gets by default.
            env["PYTHONUTF8"] = "1"
            logger.warning(
                "[flashvsr-inline] Windows: expandable_segments is unavailable, so the "
                "FlashVSR worker peaks ~13GB reserved VRAM in addition to the primary "
                "pipeline; 16GB GPUs will likely OOM. Prefer --secondary-restoration "
                "flashvsr (offline 3-phase) on Windows."
            )
        else:
            # expandable_segments keeps the worker's reserved VRAM tight (the
            # co-residence discipline, §12).
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        self._lock = threading.Lock()
        self._closed = False
        # At --log-level error, mute the worker's benign expandable_segments OOM
        # warnings by piping stderr through a line filter (still forwarding real
        # stderr like tracebacks). At info/warning/debug, inherit stderr unchanged
        # so those warnings show.
        self._quiet = str(log_level).lower() == "error"
        self._stderr_thread: threading.Thread | None = None
        logger.info("[flashvsr-inline] spawning worker: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if self._quiet else None, env=env,
        )
        if self._quiet:
            self._stderr_thread = threading.Thread(
                target=self._pump_stderr, name="flashvsr-inline-stderr", daemon=True
            )
            self._stderr_thread.start()
        self._await_ready(startup_timeout_s)

    # -- lifecycle -----------------------------------------------------------

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
                f"[flashvsr-inline] worker did not become ready within {timeout_s:.0f}s "
                "(model load stalled?)"
            )
        header = result.get("header")
        if not header or header.get("status") != "ready":
            self._kill()
            raise RuntimeError(f"[flashvsr-inline] worker failed to start: {header}")
        logger.info("[flashvsr-inline] worker ready")

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
                raise RuntimeError("[flashvsr-inline] worker closed the pipe mid-payload")
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

        # (T,C,256,256) float [0,1] -> (T,256,256,3) uint8 RGB HWC, C-contiguous.
        hwc = (
            frames_256.detach().to("cpu", torch.float32).clamp_(0.0, 1.0)
            .mul_(255.0).round_().to(torch.uint8)
            .permute(0, 2, 3, 1).contiguous().numpy()
        )
        h, w = int(hwc.shape[1]), int(hwc.shape[2])

        with self._lock:
            if self._closed or self._proc.poll() is not None:
                raise RuntimeError("[flashvsr-inline] worker process is not running")
            header = json.dumps({"seq": 0, "n": t, "h": h, "w": w}) + "\n"
            self._proc.stdin.write(header.encode("utf-8"))
            self._proc.stdin.write(hwc.tobytes())
            self._proc.stdin.flush()

            resp = self._read_header()
            if resp is None:
                raise RuntimeError("[flashvsr-inline] worker died (no response)")
            if "error" in resp:
                raise RuntimeError(f"[flashvsr-inline] worker error: {resp['error']}")
            rn, rh, rw = int(resp["n"]), int(resp["h"]), int(resp["w"])
            data = self._read_exact(rn * rh * rw * 3)

        # Frame-count contract: a mismatch silently corrupts the blend alignment.
        if rn != t:
            raise RuntimeError(
                f"[flashvsr-inline] frame-count mismatch: sent {t}, got {rn}"
            )
        out = np.frombuffer(data, dtype=np.uint8).reshape(rn, rh, rw, 3)
        # keep window -> CHW uint8 CPU tensors (blend moves each to device).
        kept = np.ascontiguousarray(out[ks:ke].transpose(0, 3, 1, 2))
        return list(torch.from_numpy(kept).unbind(0))

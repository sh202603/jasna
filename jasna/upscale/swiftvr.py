"""+modi: whole-frame upscaling with SwiftVR (``jasna-postprocess --upscale swiftvr``).

SwiftVR is the one-step diffusion video restorer the secondary restoration runs on 256px mosaic
crops. Here it restores whole frames: ``SwiftvrWorker`` keeps a worker of the SwiftVR venv
resident (``jasna/restorer/swiftvr_frame_worker.py``), which cuts each window of frames into
tiles, and ``SwiftvrClipUpscaler`` is the ``ClipUpscaler`` of one video on top of it.

SwiftVR upscales 2x or 4x only. The factor follows the target height the way the network
upscalers choose their passes, and ``finish_frame`` resizes the result to the exact output size."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from jasna.upscale import Size, finish_frame

logger = logging.getLogger(__name__)

# The plain resize at the end may enlarge this much; more needs the 4x factor (as in network.py).
_RESIZE_FINISH_MAX = 1.25
# The checkpoint read (about 20 GB), the warmup at the tile size and torch.compile.
DEFAULT_STARTUP_TIMEOUT_S = 900.0


def swiftvr_scale(source_height: int, output_height: int) -> int:
    """SwiftVR's factor for this upscale: 2 when the final resize then enlarges by at most
    ``_RESIZE_FINISH_MAX``, else 4."""
    return 2 if output_height <= source_height * 2 * _RESIZE_FINISH_MAX else 4


def _resolve_worker_script() -> Path:
    """Path to the worker script (module-level for test monkeypatching)."""
    from jasna.restorer import bundled_script_path

    return bundled_script_path("swiftvr_frame_worker.py")


class SwiftvrWorker:
    """The resident whole-frame worker. A folder batch shares one across its videos: the frame
    size and the factor travel with every window."""

    def __init__(
        self,
        *,
        repo: Path,
        model_dir: Path,
        sv_python: Path,
        device: str,
        warmup_size: Size,
        warmup_scale: int,
        accel: bool = True,
        tile_megapixels: float | None = None,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
        verbose: bool = False,
    ) -> None:
        from jasna.restorer.swiftvr_common import check_restore_clip_api
        from jasna.restorer.worker_env import external_worker_env

        check_restore_clip_api(Path(repo))
        worker = _resolve_worker_script()
        if not worker.is_file():
            raise FileNotFoundError(f"SwiftVR frame worker script not found: {worker}")
        cmd = [
            str(sv_python), str(worker),
            "--repo", str(repo),
            "--model-dir", str(model_dir),
            "--device", str(device),
            "--warmup-width", str(int(warmup_size[0])),
            "--warmup-height", str(int(warmup_size[1])),
            "--warmup-scale", str(int(warmup_scale)),
        ]
        if tile_megapixels is not None:
            cmd += ["--tile-megapixels", str(float(tile_megapixels))]
        if accel:
            cmd += ["--fp8-dit", "--torch-compile"]
        # Verification-only override, the same contract as the secondary restorers.
        color_fix = os.environ.get("JASNA_SWIFTVR_COLOR_FIX")
        if color_fix:
            if color_fix not in ("adain", "wavelet", "none"):
                raise ValueError(f"[swiftvr] JASNA_SWIFTVR_COLOR_FIX must be adain|wavelet|none, got {color_fix!r}")
            cmd += ["--color-fix-method", color_fix]
        if verbose:
            cmd.append("--verbose")

        env = external_worker_env()
        if os.name == "nt":
            env["PYTHONUTF8"] = "1"
        else:
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        self._accel_requested = bool(accel)
        self._seq = 0
        self._broken = False
        self._closed = False
        logger.info("[swiftvr] spawning worker: %s", " ".join(cmd))
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        self._await_ready(startup_timeout_s)

    def _await_ready(self, timeout_s: float) -> None:
        result: dict = {}

        def _reader() -> None:
            result["header"] = self._read_header()

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        thread.join(timeout_s)
        if thread.is_alive():
            self._kill()
            raise RuntimeError(f"[swiftvr] worker did not become ready within {timeout_s:.0f}s")
        header = result.get("header")
        if not header or header.get("status") != "ready":
            self._kill()
            hint = " (if the failure is in FP8 or torch.compile, retry with --no-swiftvr-accel)" \
                if self._accel_requested else ""
            raise RuntimeError(f"[swiftvr] worker failed to start: {header}{hint}")
        for line in header.get("accel_log") or ():
            logger.log(logging.INFO if line.startswith("enabled") else logging.WARNING, "[swiftvr] accel: %s", line)
        accel = list(header.get("accel") or [])
        logger.info("[swiftvr] worker ready (acceleration: %s)", ", ".join(accel) if accel else "off")

    def _read_header(self) -> dict | None:
        line = self._proc.stdout.readline()
        return json.loads(line.decode("utf-8")) if line.endswith(b"\n") else None

    def _read_into(self, target: np.ndarray) -> None:
        view = memoryview(target).cast("B")
        while view:
            count = self._proc.stdout.readinto(view)
            if not count:
                raise RuntimeError("[swiftvr] worker closed the pipe mid-frame")
            view = view[count:]

    def restore(self, frames: np.ndarray, scale: int) -> Iterator[np.ndarray]:
        """uint8 RGB ``[n, h, w, 3]`` frames to the ``n`` frames at ``scale`` times the size, one
        at a time. Each yielded array is reused for the next frame."""
        if self._closed or self._broken or self._proc.poll() is not None:
            raise RuntimeError("[swiftvr] worker process is not running")
        n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        self._seq += 1
        # A window left half read leaves frames in the pipe; nothing can be sent after that.
        self._broken = True
        header = json.dumps({"seq": self._seq, "n": n, "h": h, "w": w, "scale": int(scale)}) + "\n"
        self._proc.stdin.write(header.encode("utf-8"))
        self._proc.stdin.write(np.ascontiguousarray(frames).data)
        self._proc.stdin.flush()
        frame = np.empty((h * scale, w * scale, 3), np.uint8)
        for index in range(n):
            resp = self._read_header()
            if resp is None:
                raise RuntimeError("[swiftvr] worker died (no response)")
            if "error" in resp:
                self._broken = False  # the worker answered the window with the error alone
                raise RuntimeError(f"[swiftvr] worker error: {resp['error']}")
            if (resp["seq"], resp["i"], resp["h"], resp["w"]) != (self._seq, index, *frame.shape[:2]):
                raise RuntimeError(f"[swiftvr] unexpected response {resp} for frame {index} of window {self._seq}")
            self._read_into(frame)
            yield frame
        self._broken = False

    def _kill(self) -> None:
        try:
            self._proc.kill()
            self._proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._broken:
            self._kill()
            return
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._kill()


class SwiftvrClipUpscaler:
    """One video's ``ClipUpscaler`` on a borrowed ``SwiftvrWorker``: ``close`` leaves the worker
    running for the next video."""

    def __init__(
        self,
        worker: SwiftvrWorker,
        device: torch.device,
        *,
        source_size: Size,
        output_size: Size,
        strength: float,
    ) -> None:
        self._worker = worker
        self._device = device
        self._output_size = output_size
        self._strength = strength
        self._scale = swiftvr_scale(source_size[1], output_size[1])

    @torch.no_grad()
    def upscale(self, frames: torch.Tensor) -> Iterator[torch.Tensor]:
        hwc = frames.permute(0, 2, 3, 1).contiguous().numpy()
        for index, restored in enumerate(self._worker.restore(hwc, self._scale)):
            # Contiguous CHW: finish_frame returns the layout it is given and the encoder needs rows.
            image = torch.from_numpy(restored).to(self._device).permute(2, 0, 1).contiguous()[None].float().div_(255.0)
            source = frames[index].to(self._device) if self._strength < 1.0 else frames[index]
            yield finish_frame(image, source, self._output_size, self._strength)
        if self._device.type == "cuda":
            # The worker needs the VRAM for its next window more than this process needs a cache.
            torch.cuda.empty_cache()

    def close(self) -> None:
        pass

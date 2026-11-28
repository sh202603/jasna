"""+modi: whole-frame upscaling with RelayVSR (``jasna-postprocess --upscale relayvsr``).

RelayVSR is a streaming video super-resolution model: a large Wan model restores a keyframe
every few frames and a small decoder fills in the frames between two keyframes. It runs in its
own venv (it pins another torch than jasna), so ``RelayvsrWorker`` keeps a worker of that venv
resident (``jasna/restorer/relayvsr_frame_worker.py``) and ``RelayvsrStreamUpscaler`` is the
``StreamUpscaler`` of one video on top of it.

RelayVSR upscales 2x or 4x, each with its own pair of checkpoints. The factor follows the target
height by SwiftVR's rule, and ``finish_frame`` resizes the result to the exact output size."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from jasna.upscale import Size, finish_frame
from jasna.upscale.swiftvr import swiftvr_scale as relayvsr_scale

logger = logging.getLogger(__name__)

# Two checkpoints of about 10 GB, the FP8 conversion and the Triton kernels.
DEFAULT_STARTUP_TIMEOUT_S = 900.0
# The tallest output the K/V cache of a 16 GB setup is known for (about 12 GB in bf16).
HEAVY_MODEL_HEIGHT = 2160


def _resolve_worker_script() -> Path:
    """Path to the worker script (module-level for test monkeypatching)."""
    from jasna.restorer import bundled_script_path

    return bundled_script_path("relayvsr_frame_worker.py")


class RelayvsrWorker:
    """The resident worker. A folder batch shares one across its videos: the frame size and the
    pair of checkpoints travel with every video, and the worker reloads when the pair changes."""

    def __init__(
        self,
        *,
        rv_python: Path,
        weights_dir: Path,
        device: str,
        scale: int,
        window_size: int = 16,
        wan_weights: str = "fp8",
        kv_cache: str = "auto",
        kv_cache_dtype: str = "native",
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
        verbose: bool = False,
    ) -> None:
        from jasna.restorer.relayvsr_common import relayvsr_checkpoints
        from jasna.restorer.worker_env import external_worker_env

        worker = _resolve_worker_script()
        if not worker.is_file():
            raise FileNotFoundError(f"RelayVSR frame worker script not found: {worker}")
        wan, flash = relayvsr_checkpoints(weights_dir, scale)
        cmd = [
            str(rv_python), str(worker),
            "--wan-checkpoint", str(wan),
            "--flash-checkpoint", str(flash),
            "--device", str(device),
            "--window-size", str(int(window_size)),
            "--wan-weights", str(wan_weights),
            "--kv-cache", str(kv_cache),
            "--kv-cache-dtype", str(kv_cache_dtype),
        ]
        if verbose:
            cmd.append("--verbose")

        env = external_worker_env()
        if os.name == "nt":
            env["PYTHONUTF8"] = "1"

        self._weights_dir = Path(weights_dir)
        self._wan_weights_requested = str(wan_weights)
        self._held: tuple | None = None
        self._broken = False
        self._closed = False
        self._streaming = False
        self._sent = self._received = 0
        self._frame = np.empty((0, 0, 3), np.uint8)
        logger.info("[relayvsr] spawning worker: %s", " ".join(cmd))
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
            raise RuntimeError(f"[relayvsr] worker did not become ready within {timeout_s:.0f}s")
        header = result.get("header")
        if not header or header.get("status") != "ready":
            self._kill()
            raise RuntimeError(f"[relayvsr] worker failed to start: {header} (its traceback is on stderr)")
        self._report_weights(header)

    def _report_weights(self, header: dict) -> None:
        used = header.get("wan_weights")
        held = (header.get("scale"), used)
        if held == self._held:
            return  # the same pair as for the video before
        self._held = held
        if used != self._wan_weights_requested:
            logger.warning(
                "[relayvsr] Wan weights run as %s, not %s (FP8 needs an RTX 40 series or newer GPU)",
                used, self._wan_weights_requested,
            )
        logger.info("[relayvsr] worker holds the %sx pair (Wan weights %s)", header.get("scale"), used)

    def _read_header(self) -> dict | None:
        line = self._proc.stdout.readline()
        return json.loads(line.decode("utf-8")) if line.endswith(b"\n") else None

    def _read_into(self, target: np.ndarray) -> None:
        view = memoryview(target).cast("B")
        while view:
            count = self._proc.stdout.readinto(view)
            if not count:
                raise RuntimeError("[relayvsr] worker closed the pipe mid-frame")
            view = view[count:]

    def _check_running(self) -> None:
        if self._closed or self._broken or self._proc.poll() is not None:
            raise RuntimeError("[relayvsr] worker process is not running")

    def _send(self, message: dict, payload: np.ndarray | None = None) -> None:
        self._check_running()
        try:
            self._proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
            if payload is not None:
                self._proc.stdin.write(np.ascontiguousarray(payload).data)
            self._proc.stdin.flush()
        except OSError as e:
            self._broken = True
            raise RuntimeError(f"[relayvsr] worker died ({e})") from e

    def _answer(self) -> dict:
        """The next header, with the worker's failures raised. An ``error`` ends the video and
        leaves the worker waiting for the next one."""
        resp = self._read_header()
        if resp is None:
            raise RuntimeError("[relayvsr] worker died (no response; its traceback is on stderr)")
        if "error" in resp:
            self._broken = False
            self._streaming = False
            raise RuntimeError(f"[relayvsr] worker error: {resp['error']}")
        return resp

    def begin(self, width: int, height: int, scale: int) -> None:
        """Start a video of ``width`` x ``height`` frames at ``scale``. The worker loads the
        pair of that factor first when it holds the other one."""
        from jasna.restorer.relayvsr_common import relayvsr_checkpoints

        self._check_running()
        if self._streaming:
            raise RuntimeError("[relayvsr] the previous video has not ended")
        wan, flash = relayvsr_checkpoints(self._weights_dir, scale)
        self._send({"op": "begin", "h": int(height), "w": int(width), "wan": str(wan), "flash": str(flash)})
        # An answer left unread leaves the worker out of step; nothing can be sent after that.
        self._broken = True
        resp = self._answer()
        if resp.get("status") != "begin" or int(resp.get("scale", 0)) != int(scale):
            raise RuntimeError(f"[relayvsr] unexpected answer {resp} to the beginning of a {scale}x video")
        self._broken = False
        self._report_weights(resp)
        self._frame = np.empty((height * scale, width * scale, 3), np.uint8)
        self._sent = self._received = 0
        self._streaming = True

    def push(self, frame: np.ndarray) -> Iterator[np.ndarray]:
        """One uint8 RGB ``[h, w, 3]`` frame in; the output frames the model emits for it, one
        at a time. Each yielded array is reused for the next frame."""
        if not self._streaming:
            raise RuntimeError("[relayvsr] no video has begun")
        self._send({"op": "frame"}, frame)
        self._sent += 1
        yield from self._frames("more")

    def end(self) -> Iterator[np.ndarray]:
        """The video has ended: its remaining output frames."""
        if not self._streaming:
            raise RuntimeError("[relayvsr] no video has begun")
        self._send({"op": "end"})
        yield from self._frames("done")

    def _frames(self, last: str) -> Iterator[np.ndarray]:
        self._broken = True  # until the answer is read to its end
        while True:
            resp = self._answer()
            if resp.get(last):
                break
            if (resp.get("i"), resp.get("h"), resp.get("w")) != (self._received, *self._frame.shape[:2]):
                raise RuntimeError(f"[relayvsr] unexpected response {resp} for frame {self._received}")
            self._read_into(self._frame)
            self._received += 1
            yield self._frame
        if last == "done":
            if not resp.get("frames") == self._received == self._sent:
                raise RuntimeError(
                    f"[relayvsr] sent {self._sent} frames, received {self._received}, the worker counted "
                    f"{resp.get('frames')}"
                )
            self._streaming = False
            logger.info("[relayvsr] %d frames done (K/V cache on the %s)", self._received, resp.get("kv_cache"))
        self._broken = False

    def abandon(self) -> None:
        """Give up the video in progress. The worker is waiting for its next frame and holds
        frames of it, so it cannot take another video: ``close`` kills it."""
        if self._streaming:
            self._broken = True

    @property
    def alive(self) -> bool:
        """Whether the worker can begin another video (an abandoned one cannot be resumed)."""
        return not (self._closed or self._broken or self._streaming or self._proc.poll() is not None)

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
        if self._broken or self._streaming:
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


class RelayvsrStreamUpscaler:
    """One video's ``StreamUpscaler`` on a borrowed ``RelayvsrWorker``: ``close`` leaves the
    worker running for the next video."""

    def __init__(
        self,
        worker: RelayvsrWorker,
        device: torch.device,
        *,
        source_size: Size,
        output_size: Size,
        strength: float,
    ) -> None:
        self._worker = worker
        self._device = device
        self._source_size = source_size
        self._output_size = output_size
        self._strength = strength
        self._scale = relayvsr_scale(source_size[1], output_size[1])
        # Below full strength the frames the model has not answered yet wait here for their mix.
        self._sources: deque[torch.Tensor] = deque()
        self._begun = False
        self._ended = False

    @torch.no_grad()
    def push(self, frame: torch.Tensor) -> Iterator[torch.Tensor]:
        if not self._begun:
            self._worker.begin(*self._source_size, self._scale)
            self._begun = True
        if self._strength < 1.0:
            self._sources.append(frame.cpu())
        hwc = frame.permute(1, 2, 0).contiguous().cpu().numpy()
        yield from self._finished(self._worker.push(hwc))

    @torch.no_grad()
    def finish(self) -> Iterator[torch.Tensor]:
        if self._begun and not self._ended:
            yield from self._finished(self._worker.end())
            self._ended = True

    def _finished(self, restored_frames: Iterator[np.ndarray]) -> Iterator[torch.Tensor]:
        emitted = False
        for restored in restored_frames:
            # Contiguous CHW: finish_frame returns the layout it is given and the encoder needs rows.
            image = torch.from_numpy(restored).to(self._device).permute(2, 0, 1).contiguous()[None].float().div_(255.0)
            source = self._sources.popleft().to(self._device) if self._strength < 1.0 else None
            yield finish_frame(image, source, self._output_size, self._strength)
            emitted = True
        if emitted and self._device.type == "cuda":
            # The worker needs the VRAM for its next window more than this process needs a cache.
            torch.cuda.empty_cache()

    def close(self) -> None:
        self._sources.clear()
        if self._begun and not self._ended:
            self._worker.abandon()

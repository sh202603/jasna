"""SeedVR2 (1-step SR diffusion) + LoRA as the primary mosaic restorer.

Implements the primary-restorer contract consumed by
``RestorationPipeline.prepare_and_run_primary`` (the same duck-typed surface
as ``BasicvsrppMosaicRestorer``): ``raw_process(list[(C,256,256) float RGB
0..255]) -> (T,C,256,256) float [0,1]`` on ``self.device``, plus ``device``,
``input_dtype``, ``dtype``, ``tensorrt_active`` and ``close()``. It also
declares ``pad_mode = "zero"``: the LoRA was trained and validated on
zero-padded crops, so ``prepare_crops_for_restoration`` must not reflect-pad
for this restorer.

Process model: inference runs in a persistent child process from the SeedVR2
venv (``seedvr2_lora_worker.py`` — a verbatim copy of the lada-ex worker; keep
it byte-identical so upstream breakage is fixed once and diff-copied). One
clip in -> one clip out over stdin/stdout as a JSON header line plus a raw
uint8 BGR HWC buffer. The child loads the base model once, injects the LoRA at
startup, and stays resident across input files (``RestorationSession.close()``
owns the final ``close()``); it is respawned if found dead.

Error policy — harder than a secondary restorer's: a failed clip here means
the mosaic stays in the output, so there is no pass-through fallback. A worker
error or death triggers one respawn+retry of the same clip; a second failure
propagates and stops the job via the pipeline's error_holder path.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from pathlib import Path

import torch

from jasna.os_utils import subprocess_no_window_kwargs

logger = logging.getLogger(__name__)

# Marker file used to verify a seedvr2_videoupscaler checkout before spawning.
_REPO_MARKER = Path("src") / "core" / "generation_utils.py"


def default_seedvr2_python(repo: Path) -> Path:
    if os.name == "nt":
        return repo / ".venv" / "Scripts" / "python.exe"
    return repo / ".venv" / "bin" / "python"


class Seedvr2ContractError(RuntimeError):
    """Output/input shape mismatch: a contract violation, not a transient
    worker failure — never retried (a silent mismatch corrupts blend alignment
    downstream)."""


class Seedvr2LoraRestorer:
    name = "seedvr2-lora"
    # The LoRA was trained on zero-padded 256px crops; reflect padding would
    # leak mirrored content into the window statistics and the color-fix
    # reference. Read by RestorationPipeline._prepare_from_raw_crops.
    pad_mode = "zero"
    # The diffusion VAE/DiT bleeds the zero padding's black 2-3px into the
    # valid region, which composites as a dark line along the bbox edge (the
    # blend mask reaches past the crop border margin). The pipeline reverts
    # this band to the input (see _revert_valid_edges); BasicVSR++ needs 0.
    edge_revert_px = 4
    tensorrt_active = False

    def __init__(
        self,
        *,
        repo_path: str,
        lora_path: str,
        device: "torch.device | str" = "cuda:0",
        python_path: str | None = None,
        model_dir: str | None = None,
        dit_model: str = "seedvr2_ema_3b_fp16.safetensors",
        lora_rank: int = 16,
        window: int = 33,
        overlap: int = 9,
        color_fix: str = "lab",
        startup_timeout_s: float = 600.0,
    ) -> None:
        if window % 4 != 1 or window < 1:
            raise ValueError(f"[seedvr2] --seedvr2-window must be 4n+1, got {window}")
        if not 0 <= overlap < window:
            raise ValueError(f"[seedvr2] --seedvr2-overlap must be in [0, window), got {overlap}")
        if color_fix not in ("none", "lab", "wavelet"):
            raise ValueError(f"[seedvr2] --seedvr2-color-fix must be none|lab|wavelet, got {color_fix}")

        repo = Path(repo_path).expanduser()
        if not (repo / _REPO_MARKER).is_file():
            raise RuntimeError(
                f"[seedvr2] not a seedvr2_videoupscaler checkout (missing {_REPO_MARKER}): {repo}"
            )

        # Absolutize: the worker os.chdir()s into the SeedVR2 repo before
        # loading, so a CWD-relative path (the model_weights default) breaks.
        lora = Path(lora_path).expanduser().resolve()
        if not lora.is_file():
            raise RuntimeError(f"[seedvr2] LoRA checkpoint not found: {lora}")

        python = Path(python_path).expanduser() if python_path else default_seedvr2_python(repo)
        if not python.exists():
            raise RuntimeError(f"[seedvr2] worker python not found: {python} (pass --seedvr2-python)")

        if model_dir:
            resolved_model_dir = Path(model_dir).expanduser()
            if not resolved_model_dir.is_dir():
                raise RuntimeError(f"[seedvr2] --seedvr2-model-dir not found: {resolved_model_dir}")
        else:
            resolved_model_dir = repo / "models" / "SEEDVR2"
            if not resolved_model_dir.is_dir():
                logger.warning(
                    "[seedvr2] base weights not found at %s — SeedVR2 will download them on first load.",
                    resolved_model_dir,
                )

        self.device = torch.device(device)
        self.input_dtype = torch.float16
        self.dtype = torch.float16
        self._lock = threading.Lock()
        self._closed = False
        self._next_seq = 0

        from jasna.restorer import bundled_script_path

        worker = bundled_script_path("seedvr2_lora_worker.py")
        cmd = [str(python), "-u", str(worker),
               "--repo", str(repo),
               "--model-dir", str(resolved_model_dir),
               "--dit", dit_model,
               "--lora", str(lora),
               "--rank", str(int(lora_rank)),
               "--alpha", str(int(lora_rank)),
               "--window", str(int(window)),
               "--overlap", str(int(overlap)),
               "--color-fix", color_fix,
               "--device", "cuda:0"]
        if logger.isEnabledFor(logging.DEBUG):
            cmd.append("--verbose")

        # The SeedVR2 venv must import its own repo, not inherit jasna's PYTHONPATH.
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        device_index = self.device.index if self.device.index is not None else 0
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", str(device_index))
        # Same allocator as the lada-ex training/eval harness (bit-parity
        # precondition). Deliberately NOT expandable_segments.
        env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
        if os.name == "nt":
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"

        self._cmd = cmd
        self._env = env
        self._startup_timeout_s = startup_timeout_s

        self._spawn()
        logger.info(
            "[seedvr2] worker ready (lora=%s, window=%d, overlap=%d, color_fix=%s)",
            lora.name, window, overlap, color_fix,
        )

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def _spawn(self) -> None:
        """(Re)start the worker and block until its ready handshake."""
        logger.info("[seedvr2] starting worker: %s", " ".join(self._cmd))
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=self._env,
            **subprocess_no_window_kwargs(),
        )
        self._await_ready(self._startup_timeout_s)
        self._closed = False

    def _await_ready(self, timeout_s: float) -> None:
        result: dict = {}

        def _reader():
            result["header"] = self._read_header()

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            self._kill()
            raise RuntimeError(
                f"[seedvr2] worker did not become ready within {timeout_s:.0f}s "
                "(model load or weight download stalled?)"
            )
        header = result.get("header")
        if not header or header.get("status") != "ready":
            self._kill()
            raise RuntimeError(f"[seedvr2] worker failed to start: {header}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                proc.wait(timeout=15)
        except Exception:
            self._kill()
        logger.info("[seedvr2] worker stopped")

    def _kill(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass

    # ------------------------------------------------------------------ #
    # wire protocol helpers
    # ------------------------------------------------------------------ #

    def _read_header(self) -> dict | None:
        line = self._proc.stdout.readline()
        if not line:
            return None
        return json.loads(line)

    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError("[seedvr2] worker closed stdout mid-payload")
            buf.extend(chunk)
        return bytes(buf)

    # ------------------------------------------------------------------ #
    # primary-restorer contract
    # ------------------------------------------------------------------ #

    def raw_process(self, video: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            video: list of (C, 256, 256) float tensors in RGB format, [0, 255]
        Returns:
            (T, C, 256, 256) float16 tensor in [0, 1] on ``self.device``
        """
        import numpy as np

        with torch.inference_mode():
            stacked = torch.stack(video).to(self.device)
            # Quantize on-device, flip RGB->BGR (the wire is lada-native BGR,
            # keeping the worker byte-identical with lada-ex), then download.
            u8 = stacked.round().clamp(0, 255).to(torch.uint8)
            bgr_hwc = u8.permute(0, 2, 3, 1).flip(-1).contiguous()
            payload = np.ascontiguousarray(bgr_hwc.cpu().numpy())

        out = self._restore_clip(payload)  # (T, H, W, 3) uint8 BGR

        with torch.inference_mode():
            rgb = torch.from_numpy(np.ascontiguousarray(out[..., ::-1]))
            result = rgb.to(self.device).permute(0, 3, 1, 2).to(self.dtype).div_(255.0)
        return result

    def _restore_clip(self, payload) -> "object":
        try:
            return self._push_clip(payload)
        except Seedvr2ContractError:
            raise
        except (RuntimeError, OSError, BrokenPipeError) as e:
            # One respawn+retry; a second failure propagates and stops the job.
            # No pass-through fallback: a skipped clip = mosaic in the output.
            logger.warning("[seedvr2] clip failed (%s), respawning worker and retrying once", e)
            self._kill()
            self._spawn()
            return self._push_clip(payload)

    def _push_clip(self, payload):
        """payload: (n, h, w, 3) uint8 BGR numpy array; returns same-shape array."""
        import numpy as np

        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            n, h, w = payload.shape[0], payload.shape[1], payload.shape[2]

            if self._closed or self._proc.poll() is not None:
                logger.warning("[seedvr2] worker not running (closed or died), respawning")
                self._spawn()
            header = json.dumps({"seq": seq, "n": n, "h": h, "w": w}) + "\n"
            self._proc.stdin.write(header.encode())
            self._proc.stdin.write(payload.tobytes())
            self._proc.stdin.flush()

            resp = self._read_header()
            if resp is None or "error" in resp:
                raise RuntimeError(f"[seedvr2] worker error: {resp}")
            rn, rh, rw = resp["n"], resp["h"], resp["w"]
            data = self._read_exact(rn * rh * rw * 3)
            if (rn, rh, rw) != (n, h, w):
                raise Seedvr2ContractError(
                    f"[seedvr2] shape mismatch: sent {n}x{h}x{w}, got {rn}x{rh}x{rw}"
                )
            return np.frombuffer(data, dtype=np.uint8).reshape(rn, rh, rw, 3)

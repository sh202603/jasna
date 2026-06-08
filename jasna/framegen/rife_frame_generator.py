from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from jasna.framegen.frame_generator import FrameGenerator

logger = logging.getLogger(__name__)

# Spatial dims must be divisible by this for RIFE's multi-scale flow (two
# stride-2 convs + 4x downscale in the coarsest block).
_PAD_MULTIPLE = 64


def _pad_to_multiple(x: torch.Tensor, multiple: int = _PAD_MULTIPLE) -> tuple[torch.Tensor, int, int]:
    """Pad an NCHW tensor's H/W up to a multiple. Returns (padded, orig_h, orig_w)."""
    _, _, h, w = x.shape
    ph = ((h + multiple - 1) // multiple) * multiple
    pw = ((w + multiple - 1) // multiple) * multiple
    if ph == h and pw == w:
        return x, h, w
    return F.pad(x, (0, pw - w, 0, ph - h), mode="replicate"), h, w


def _unpad(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return x[..., :h, :w]


def _chw_uint8_to_nchw_float(frame: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return frame.to(device=device, dtype=dtype).div(255.0).unsqueeze(0)


def _nchw_float_to_chw_uint8(x: torch.Tensor) -> torch.Tensor:
    return x[0].clamp_(0.0, 1.0).mul_(255.0).round_().clamp_(0, 255).to(dtype=torch.uint8)


class RifeFrameGenerator(FrameGenerator):
    """Neural frame interpolation via RIFE, running on CUDA.

    Model loading supports two checkpoint formats (auto-detected):

    1. **TorchScript** (recommended, guaranteed): a ``torch.jit`` module whose
       forward signature is ``model(img0, img1, timestep) -> img`` with NCHW
       float [0,1] tensors. Bundles architecture + weights, so it always runs.
    2. **state_dict** (``.pth``/``.pkl``): loaded into the vendored RIFE 4.6
       ``IFNet`` (``jasna.models.rife``). Requires the checkpoint to match that
       architecture; otherwise prefer TorchScript.

    The backend owns all GPU pre/post-processing (uint8<->float, padding to a
    multiple of 64, per-position timestep) so any RIFE-compatible model drops in.
    """

    name = "rife"

    def __init__(
        self,
        *,
        device: torch.device,
        model_path: Optional[str] = None,
        fp16: bool = True,
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        self.device = torch.device(device)
        self._dtype = torch.float16 if fp16 else torch.float32
        self._stream = torch.cuda.current_stream(self.device)

        if model is not None:
            self._model = model.to(self.device).eval()
        else:
            self._model = self._load_model(model_path).to(self.device).eval()

        if self._dtype is torch.float16:
            # The vendored IFNet is dtype-agnostic (warp builds its grid in the
            # flow's dtype), but external TorchScript checkpoints may bake a
            # float32 sampling grid into their warp, which grid_sample rejects
            # against half inputs. Probe once and fall back to fp32 if so.
            try:
                self._model = self._model.half()
                self._probe()
            except Exception as e:
                logger.warning(
                    "RIFE fp16 inference failed (%s); falling back to fp32. "
                    "TorchScript checkpoints with a baked-in float32 warp grid "
                    "only support fp32.", e,
                )
                self._dtype = torch.float32

        if self._dtype is torch.float32:
            self._model = self._model.float()
        logger.info(
            "RifeFrameGenerator initialized (%s)",
            "fp16" if self._dtype is torch.float16 else "fp32",
        )

    @torch.inference_mode()
    def _probe(self) -> None:
        """Run one minimal-size inference to validate the current dtype."""
        z = torch.zeros((1, 3, _PAD_MULTIPLE, _PAD_MULTIPLE), device=self.device, dtype=self._dtype)
        t = torch.full((1, 1, _PAD_MULTIPLE, _PAD_MULTIPLE), 0.5, device=self.device, dtype=self._dtype)
        self._model(z, z, t)

    def _load_model(self, model_path: Optional[str]) -> torch.nn.Module:
        if model_path:
            path = Path(model_path)
        else:
            from jasna.model_weights_resolver import resolve_model_weights_file
            path = resolve_model_weights_file("rife.pth")
        if not path.exists():
            raise FileNotFoundError(
                f"RIFE weights not found: {path}. Provide a RIFE checkpoint via "
                "--frame-gen-model-path (TorchScript recommended; see RifeFrameGenerator docs) "
                "or place 'rife.pth' in model_weights/."
            )

        # Try TorchScript first (self-contained, guaranteed architecture).
        try:
            return torch.jit.load(str(path), map_location="cpu")
        except Exception:
            pass

        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(obj, torch.nn.Module):
            return obj
        state = obj
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Unrecognized RIFE checkpoint format: {path}")
        # Strip common prefixes (DataParallel 'module.', RIFE 'flownet.').
        cleaned = {}
        for k, v in state.items():
            nk = k
            for prefix in ("module.", "flownet."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v
        from jasna.models.rife import IFNet
        net = IFNet()
        missing, unexpected = net.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            logger.warning(
                "RIFE state_dict loaded non-strictly (missing=%d, unexpected=%d). "
                "If interpolation looks wrong, use a TorchScript checkpoint instead.",
                len(missing), len(unexpected),
            )
        return net

    @torch.inference_mode()
    def interpolate(self, frame_a: torch.Tensor, frame_b: torch.Tensor, positions: list[float]) -> list[torch.Tensor]:
        if not positions:
            return []
        img0 = _chw_uint8_to_nchw_float(frame_a, self.device, self._dtype)
        img1 = _chw_uint8_to_nchw_float(frame_b, self.device, self._dtype)
        img0, h, w = _pad_to_multiple(img0)
        img1, _, _ = _pad_to_multiple(img1)

        out: list[torch.Tensor] = []
        for p in positions:
            t = torch.full((1, 1, img0.shape[2], img0.shape[3]), float(p), device=self.device, dtype=self._dtype)
            mid = self._model(img0, img1, t)
            mid = _unpad(mid.float(), h, w)
            out.append(_nchw_float_to_chw_uint8(mid))
        return out

    def close(self) -> None:
        self._model = None

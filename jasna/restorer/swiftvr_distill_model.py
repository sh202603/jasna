"""TinyROIEnhancer: the lightweight 2x student distilled from SwiftVR outputs.

The checkpoint (``roi-distill-pilot-v1``) ships without its model definition.
The layer layout below follows the state dict. Three points the state dict
cannot tell are estimates, recovered by probing the weights with flat-colour
inputs (the only combination that leaves no PixelShuffle grid and no colour
shift): the ReLU right after the stem, the residual scale, and the bilinear
base the head's output is added to. Replace them once the author's definition
is available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

SWIFTVR_DISTILL_KNOWN_VERSIONS = ("roi-distill-pilot-v1",)

# Estimates, see the module docstring.
SWIFTVR_DISTILL_RESIDUAL_SCALE = 0.1
SWIFTVR_DISTILL_STEM_RELU = True
SWIFTVR_DISTILL_BASE_MODE = "bilinear"

_ARCHITECTURE_LIMITS = {
    "channels": (1, 512),
    "blocks": (1, 64),
    "window": (1, 15),
    "scale": (1, 4),
}


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + SWIFTVR_DISTILL_RESIDUAL_SCALE * self.body(x)


class TinyROIEnhancer(nn.Module):
    def __init__(self, *, channels: int = 24, blocks: int = 12, window: int = 5, scale: int = 2) -> None:
        super().__init__()
        self.window = int(window)
        self.scale = int(scale)
        self.stem = nn.Conv2d(3 * self.window, channels, 3, padding=1)
        self.body = nn.Sequential(*[_ResidualBlock(channels) for _ in range(blocks)])
        self.head = nn.Conv2d(channels, 3 * self.scale * self.scale, 3, padding=1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """
        Args:
            windows: (B, 3 * window, H, W) float RGB in [0, 1]; the frames
                t - window // 2 .. t + window // 2 stacked frame by frame
        Returns:
            (B, 3, H * scale, W * scale) float, the upscaled centre frame (not clamped)
        """
        center = 3 * (self.window // 2)
        features = self.stem(windows)
        if SWIFTVR_DISTILL_STEM_RELU:
            features = F.relu(features)
        features = self.body(features)
        base = F.interpolate(
            windows[:, center : center + 3],
            scale_factor=self.scale,
            mode=SWIFTVR_DISTILL_BASE_MODE,
            align_corners=False,
        )
        return base + F.pixel_shuffle(self.head(features), self.scale)


def resolve_swiftvr_distill_model_path(model_arg: str) -> Path:
    if not str(model_arg).strip():
        raise ValueError("--swiftvr-distill-model is required for --secondary-restoration swiftvr-distill")
    model_path = Path(str(model_arg).strip()).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"--swiftvr-distill-model not found: {model_path}")
    return model_path


def _read_architecture(checkpoint: dict, model_path: Path) -> dict[str, int]:
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError(f"SwiftVR distill checkpoint has no 'architecture' dict: {model_path}")
    result: dict[str, int] = {}
    for key, (low, high) in _ARCHITECTURE_LIMITS.items():
        value = architecture.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(
                f"SwiftVR distill checkpoint has an unsupported architecture.{key}={value!r} "
                f"(expected an int in [{low}, {high}]): {model_path}"
            )
        result[key] = value
    if result["window"] % 2 == 0:
        raise ValueError(
            f"SwiftVR distill checkpoint has an even architecture.window={result['window']} "
            f"(the window must have a centre frame): {model_path}"
        )
    return result


def load_swiftvr_distill_model(model_path: Path | str, device: torch.device) -> TinyROIEnhancer:
    model_path = Path(model_path)
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"SwiftVR distill checkpoint is not a dict: {model_path}")

    version = checkpoint.get("version")
    if version not in SWIFTVR_DISTILL_KNOWN_VERSIONS:
        # The forward is an estimate made for the known versions; another
        # version with the same layers may still need a different one.
        logger.warning(
            "SwiftVR distill checkpoint version %r is not one of %s; the forward pass may not match it",
            version, list(SWIFTVR_DISTILL_KNOWN_VERSIONS),
        )

    architecture = _read_architecture(checkpoint, model_path)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError(f"SwiftVR distill checkpoint has no 'model' state dict: {model_path}")
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"SwiftVR distill checkpoint has a non-finite or non-tensor entry '{key}': {model_path}")

    model = TinyROIEnhancer(**architecture)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise ValueError(f"SwiftVR distill checkpoint does not match its architecture {architecture}: {model_path}: {e}") from e

    model.requires_grad_(False)
    model.eval().to(device=device, dtype=torch.float32)
    logger.info(
        "SwiftVR distill model loaded: %s (version=%s, %s, step=%s)",
        model_path, version, architecture, checkpoint.get("step"),
    )
    return model

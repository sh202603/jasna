# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import pickle
import re
import warnings

import torch

from jasna.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGanNet

warnings.filterwarnings(
    "ignore",
    message="The given buffer is not writable, and PyTorch does not support non-writable tensors",
    module="torch",
)

logger = logging.getLogger(__name__)

_INFERENCE_GENERATOR_PREFIXES = ("generator_ema.", "generator.")


def inference_generator_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    """The generator weights a restoration checkpoint is run with: its EMA generator's, or the
    trained generator's when it has no EMA copy. Accepts a nested ``state_dict`` and
    ``module.``-prefixed (DataParallel) keys."""
    state = checkpoint.get("state_dict", checkpoint)
    state = {re.sub(r"^module\.", "", key): value for key, value in state.items()}
    for prefix in _INFERENCE_GENERATOR_PREFIXES:
        weights = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
        if weights:
            return weights
    raise ValueError("The checkpoint holds no BasicVSR++ generator weights")


def _load_checkpoint(checkpoint_path: str) -> dict:
    """+modi: a checkpoint written by the Lada / mmengine trainer (the original Lada releases, a
    fine-tune of your own) carries the trainer's log buffers beside the weights, which the
    weights-only loader refuses. Such a file is loaded the way mmengine did; upstream's
    republished weights hold tensors only and never take this path."""
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        logger.warning(
            "%s holds more than tensors (a checkpoint saved by the trainer); loading it without the "
            "weights-only check. Only do this with checkpoints you trust.",
            checkpoint_path,
        )
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def load_model(checkpoint_path: str, device: torch.device, fp16: bool) -> BasicVSRPlusPlusGanNet:
    """The BasicVSR++ restoration network of a Lada GAN checkpoint, in eval mode on ``device``."""
    generator = BasicVSRPlusPlusGanNet(mid_channels=64, num_blocks=15, spynet_pretrained=None)
    checkpoint = _load_checkpoint(checkpoint_path)
    generator.load_state_dict(inference_generator_state(checkpoint))
    generator = generator.to(device).eval()
    return generator.half() if fp16 else generator

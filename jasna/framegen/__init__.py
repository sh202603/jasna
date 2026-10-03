"""Frame generation (frame-rate up-conversion) for the output stage.

Frame generation inserts AI-interpolated frames between the pipeline's final
blended frames, doubling (2x) or quadrupling (4x) the output frame rate. Unlike
the secondary restorers (which process 256x256 mosaic crops and never change the
frame count), frame generation operates on full-resolution output frames and
*increases* the frame count + presentation timestamps (PTS).

It is wired in as a thin decorator (``FrameGenWriter``) around the pipeline's
``FrameWriter``, so the rest of the pipeline and the encoder are untouched.

Backends implement the ``FrameGenerator`` protocol and are pluggable by name:
- ``rife``: neural interpolation (RIFE) in PyTorch; needs a RIFE checkpoint
  (``model_weights/rife.pth``), runs on every supported GPU.
- ``rtx``: NVIDIA RTX Video Frame Generation via ``nvidia-vfx`` (>= 0.2.0.0);
  no weights to supply, roughly 10x faster than RIFE, Ada (RTX 40) and newer only.
"""

from jasna.framegen.frame_generator import FrameGenerator
from jasna.framegen.frame_gen_writer import FrameGenWriter

__all__ = [
    "FrameGenerator",
    "FrameGenWriter",
    "build_frame_generator",
    "BACKEND_CHOICES",
    "MULTIPLIER_CHOICES",
    "RTX_MODE_CHOICES",
]

MULTIPLIER_CHOICES = {"none": 1, "2x": 2, "4x": 4}
BACKEND_CHOICES = ["rife", "rtx"]
# Kept as a literal (not imported from rtx_frame_generator) so that argparse
# construction never pulls the backend module in.
RTX_MODE_CHOICES = ["low", "medium", "high"]


def build_frame_generator(
    backend: str,
    *,
    device,
    model_path=None,
    fp16: bool = True,
    rtx_mode: str = "medium",
) -> FrameGenerator:
    """Construct a frame-generation backend by name.

    Heavy backend imports stay local so that ``--frame-gen none`` and error
    paths never import torch model code or the nvvfx SDK. ``model_path`` and
    ``fp16`` only matter to ``rife``; ``rtx_mode`` only to ``rtx``.
    """
    backend = str(backend).lower()
    if backend == "rife":
        from jasna.framegen.rife_frame_generator import RifeFrameGenerator
        return RifeFrameGenerator(device=device, model_path=model_path, fp16=fp16)
    if backend == "rtx":
        from jasna.framegen.rtx_frame_generator import RtxFrameGenerator
        return RtxFrameGenerator(device=device, mode=rtx_mode)
    raise ValueError(f"Unsupported frame-gen backend: {backend} (supported: {', '.join(BACKEND_CHOICES)})")

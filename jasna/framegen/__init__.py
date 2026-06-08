"""Frame generation (frame-rate up-conversion) for the output stage.

Frame generation inserts AI-interpolated frames between the pipeline's final
blended frames, doubling (2x) or quadrupling (4x) the output frame rate. Unlike
the secondary restorers (which process 256x256 mosaic crops and never change the
frame count), frame generation operates on full-resolution output frames and
*increases* the frame count + presentation timestamps (PTS).

It is wired in as a thin decorator (``FrameGenWriter``) around the pipeline's
``FrameWriter``, so the rest of the pipeline and the encoder are untouched.

Backends implement the ``FrameGenerator`` protocol and are pluggable by name:
- ``rife``: BasicVSR-style neural interpolation (RIFE), runs today on CUDA.
- ``rtx``: NVIDIA RTX Video Frame Generation — a stub that activates once the
  ``nvidia-vfx`` package ships the frame-generation effect (not yet released).
"""

from jasna.framegen.frame_generator import FrameGenerator
from jasna.framegen.frame_gen_writer import FrameGenWriter

__all__ = ["FrameGenerator", "FrameGenWriter", "build_frame_generator", "MULTIPLIER_CHOICES"]

MULTIPLIER_CHOICES = {"none": 1, "2x": 2, "4x": 4}


def build_frame_generator(backend: str, *, device, **kwargs) -> FrameGenerator:
    """Construct a frame-generation backend by name.

    Heavy backend imports stay local so that ``--frame-gen none`` and error
    paths never import torch model code or the nvvfx SDK.
    """
    backend = str(backend).lower()
    if backend == "rife":
        from jasna.framegen.rife_frame_generator import RifeFrameGenerator
        return RifeFrameGenerator(device=device, **kwargs)
    if backend == "rtx":
        from jasna.framegen.rtx_frame_generator import RtxFrameGenerator
        return RtxFrameGenerator(device=device, **kwargs)
    raise ValueError(f"Unsupported frame-gen backend: {backend} (supported: rife, rtx)")

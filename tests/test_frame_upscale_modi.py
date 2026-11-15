"""+modi: upstream's whole-video upscaling (``--upscale``, v0.12.3) next to modi's own paths."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from factories import make_pipeline, video_metadata
from jasna import update_check


class _Writer:
    def write(self, frame, pts):
        pass

    def after_write(self, frames_written, stop_event):
        pass

    def finish(self):
        pass

    def close(self):
        pass


@pytest.mark.usefixtures("nvidia_vendor")
@pytest.mark.parametrize("backend", ["rtx", "rife"])
def test_frame_generation_after_upscaling_is_built_at_the_upscaled_size(backend):
    pipeline = make_pipeline(
        device=torch.device("cuda", 0), upscale_model="span", upscale_height=1080,
        frame_generation="2x", frame_generation_backend=backend,
    )
    metadata = video_metadata(video_width=960, video_height=540)

    with patch("jasna.pipeline.build_frame_upscaler"), \
            patch("jasna.frame_generation.build_generator") as build_generator:
        pipeline._frame_writer(_Writer(), metadata)

    assert build_generator.call_args.args == (backend,)
    assert (build_generator.call_args.kwargs["width"], build_generator.call_args.kwargs["height"]) == (1920, 1080)


@pytest.mark.usefixtures("nvidia_vendor")
def test_frame_generation_without_upscaling_keeps_the_source_size():
    pipeline = make_pipeline(device=torch.device("cuda", 0), frame_generation="4x", frame_generation_backend="rife")
    metadata = video_metadata(video_width=960, video_height=540)

    with patch("jasna.frame_generation.build_generator") as build_generator:
        pipeline._frame_writer(_Writer(), metadata)

    assert (build_generator.call_args.kwargs["width"], build_generator.call_args.kwargs["height"]) == (960, 540)


@pytest.mark.parametrize(
    ("version", "current", "newer"),
    [("0.12.4", "0.12.3+modi", True), ("0.12.3", "0.12.3+modi", False), ("0.12.2", "0.12.3+modi", False)],
)
def test_the_update_check_compares_the_release_part_of_the_fork_version(version, current, newer):
    assert update_check.is_newer(version, current) is newer

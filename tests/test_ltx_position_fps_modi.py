"""+modi: the frame rate of the temporal positions is read from the model file (``position_fps``)."""

from __future__ import annotations

import torch

from jasna.ltx import sampler
from jasna.ltx import transformer as T


def test_position_fps_is_upstreams_without_the_metadata_key() -> None:
    assert sampler.position_fps({"sigmas": "[1.0, 0.0]"}) == T.POSITION_FPS == 24.0


def test_position_fps_comes_from_the_model_file() -> None:
    assert sampler.position_fps({"position_fps": "30.0"}) == 30.0


def test_latent_positions_default_to_upstreams_rate() -> None:
    assert torch.equal(T.latent_positions(4, 2, 3), T.latent_positions(4, 2, 3, T.POSITION_FPS))


def test_latent_time_positions_scale_inversely_with_the_rate() -> None:
    at_24 = T.latent_positions(4, 2, 3, 24.0)
    at_30 = T.latent_positions(4, 2, 3, 30.0)

    assert torch.allclose(at_24[0] * 24.0, at_30[0] * 30.0)
    assert at_30[0].max() < at_24[0].max()
    assert torch.equal(at_24[1:], at_30[1:])


def test_rope_follows_the_rate() -> None:
    cpu = torch.device("cpu")
    default = T.build_rope(4, 2, 2, dim=4096, heads=T.HEADS, device=cpu)
    same = T.build_rope(4, 2, 2, dim=4096, heads=T.HEADS, device=cpu, fps=T.POSITION_FPS)
    other = T.build_rope(4, 2, 2, dim=4096, heads=T.HEADS, device=cpu, fps=30.0)

    assert torch.equal(default.cos, same.cos) and torch.equal(default.sin, same.sin)
    assert not torch.equal(default.cos, other.cos)

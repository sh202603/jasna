"""+modi: LTX-2.5's conv VAE (``CausalVideoAutoencoder``) loads beside the diffusion VAE."""

from __future__ import annotations

import json

import pytest
import torch

from jasna.models.ltx_vae import loader
from jasna.models.ltx_vae.conv_video_decoder import ConvVideoDecoder

CONV_VAE_CONFIG = {
    "_class_name": "CausalVideoAutoencoder",
    "dims": 3,
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 8,
    "encoder_blocks": [["res_x", {"num_layers": 1}], ["compress_all_res", {"multiplier": 2}]],
    "decoder_blocks": [
        ["res_x", {"num_layers": 1}],
        ["compress_space", {"multiplier": 2}],
        ["compress_time", {"multiplier": 2}],
        ["compress_all", {"multiplier": 1}],
    ],
    "norm_layer": "pixel_norm",
    "patch_size": 4,
    "causal_decoder": False,
    "timestep_conditioning": False,
    "decoder_base_channels": 8,
    "spatial_padding_mode": "zeros",
}


class _Reader:
    def __init__(self, vae_config: dict) -> None:
        self._metadata = {"config": json.dumps({"vae": vae_config})}

    def metadata(self) -> dict[str, str]:
        return self._metadata


def test_read_vae_config_accepts_the_conv_vae() -> None:
    assert loader.read_vae_config(_Reader(CONV_VAE_CONFIG))["_class_name"] == "CausalVideoAutoencoder"


def test_read_vae_config_still_rejects_other_classes() -> None:
    with pytest.raises(ValueError, match="not supported"):
        loader.read_vae_config(_Reader({"_class_name": "AutoencoderKL"}))


def test_encoder_kwargs_of_the_conv_vae_come_from_its_flat_config() -> None:
    assert loader.encoder_kwargs(CONV_VAE_CONFIG) == {
        "in_channels": 3,
        "out_channels": 8,
        "encoder_blocks": CONV_VAE_CONFIG["encoder_blocks"],
        "patch_size": 4,
    }


def test_conv_decoder_kwargs() -> None:
    assert loader.conv_decoder_kwargs(CONV_VAE_CONFIG) == {
        "in_channels": 8,
        "out_channels": 3,
        "decoder_blocks": CONV_VAE_CONFIG["decoder_blocks"],
        "patch_size": 4,
        "base_channels": 8,
    }


@pytest.mark.parametrize("key, value", [("causal_decoder", True), ("timestep_conditioning", True), ("norm_layer", "group_norm")])
def test_conv_decoder_kwargs_reject_variants_that_are_not_ported(key, value) -> None:
    with pytest.raises(ValueError):
        loader.conv_decoder_kwargs({**CONV_VAE_CONFIG, key: value})


@pytest.mark.parametrize(
    "blocks",
    [
        [["res_x", {"num_layers": 1, "inject_noise": True}]],
        [["compress_all", {"multiplier": 1, "residual": True}]],
        [["attn_res_x", {"num_layers": 1}]],
    ],
)
def test_conv_decoder_rejects_blocks_that_are_not_ported(blocks) -> None:
    with pytest.raises(ValueError):
        ConvVideoDecoder(8, 3, blocks, patch_size=4, base_channels=8)


def test_conv_decoder_output_shape_and_scale_factors() -> None:
    torch.manual_seed(0)
    decoder = ConvVideoDecoder(**loader.conv_decoder_kwargs(CONV_VAE_CONFIG)).eval()
    scale = decoder.video_downscale_factors
    assert (scale.time, scale.height, scale.width) == (4, 16, 16)

    with torch.no_grad():
        video = decoder(torch.randn(1, 8, 3, 2, 2))

    assert video.shape == (1, 3, scale.time * (3 - 1) + 1, 2 * scale.height, 2 * scale.width)
    assert torch.isfinite(video).all()


def test_decode_latent_runs_the_conv_decoder_whole() -> None:
    torch.manual_seed(0)
    decoder = ConvVideoDecoder(**loader.conv_decoder_kwargs(CONV_VAE_CONFIG)).eval()
    latent = torch.randn(1, 8, 2, 2, 2)

    with torch.no_grad():
        video = loader.decode_latent(decoder, latent, 0, torch.Generator())
        expected = decoder(latent)

    assert video.dtype == latent.dtype
    assert torch.equal(video, expected)

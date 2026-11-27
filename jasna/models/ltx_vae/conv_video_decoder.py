"""Convolutional video VAE decoder of the LTX-2.5 ``CausalVideoAutoencoder`` checkpoint (modi).

Ported from ``ltx_core/model/video_vae/conv_video_decoder.py`` (with ``convolution.py``,
``resnet.py`` and ``sampling.py``), commit f0f2f8607189c78a72420b2b63cb1975b5eca179, restricted
to what the published conv VAE uses: pixel norm, zero spatial padding, no timestep conditioning,
no noise injection, non-causal temporal padding. The whole window is decoded in one pass.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from jasna.models.ltx_vae.ops import PerChannelStatistics, PixelNorm, unpatchify
from jasna.models.ltx_vae.tiling import SpatioTemporalScaleFactors

_UPSAMPLE_STRIDES = {
    "compress_space": (1, 2, 2),
    "compress_time": (2, 1, 1),
    "compress_all": (2, 2, 2),
}


class SymmetricConv3d(nn.Module):
    """3x3x3 conv; time is padded by repeating the first and the last frame."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=(0, 1, 1), padding_mode="zeros")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat((x[:, :, :1], x, x[:, :, -1:]), dim=2))


class ResnetBlock3D(nn.Module):
    """PixelNorm -> SiLU -> conv, twice, plus identity skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = PixelNorm()
        self.non_linearity = nn.SiLU()
        self.conv1 = SymmetricConv3d(channels, channels)
        self.norm2 = PixelNorm()
        self.conv2 = SymmetricConv3d(channels, channels)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(self.non_linearity(self.norm1(input_tensor)))
        hidden_states = self.conv2(self.non_linearity(self.norm2(hidden_states)))
        return input_tensor + hidden_states


class UNetMidBlock3D(nn.Module):
    def __init__(self, channels: int, num_layers: int) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList([ResnetBlock3D(channels) for _ in range(num_layers)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states)
        return hidden_states


def depth_to_space(x: torch.Tensor, stride: tuple[int, int, int]) -> torch.Tensor:
    """``b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)``."""
    b, c, d, h, w = x.shape
    p1, p2, p3 = stride
    x = x.reshape(b, c // (p1 * p2 * p3), p1, p2, p3, d, h, w).permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(b, c // (p1 * p2 * p3), d * p1, h * p2, w * p3)


class DepthToSpaceUpsample(nn.Module):
    """Conv + depth-to-space; a temporal upsample drops its first frame."""

    def __init__(self, in_channels: int, stride: tuple[int, int, int], reduction: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = SymmetricConv3d(in_channels, math.prod(stride) * in_channels // reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = depth_to_space(self.conv(x), self.stride)
        return x[:, :, 1:] if self.stride[0] == 2 else x


class ConvVideoDecoder(nn.Module):
    """Decodes normalized ``(B, 128, F', H', W')`` latents into ``(B, 3, 8(F'-1)+1, 32H', 32W')``
    pixels in ~[-1, 1]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        decoder_blocks: list[tuple[str, dict[str, Any]]],
        patch_size: int,
        base_channels: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.video_downscale_factors = SpatioTemporalScaleFactors.from_blocks(decoder_blocks, patch_size)
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        channels = base_channels
        for name, config in decoder_blocks:
            if name in _UPSAMPLE_STRIDES:
                channels *= config.get("multiplier", 1)
        self.conv_in = SymmetricConv3d(in_channels, channels)

        self.up_blocks = nn.ModuleList([])
        for name, config in reversed(decoder_blocks):
            if name == "res_x":
                if config.get("inject_noise", False):
                    raise ValueError("conv VAE decoders with noise injection are not supported")
                self.up_blocks.append(UNetMidBlock3D(channels, config["num_layers"]))
            elif name in _UPSAMPLE_STRIDES:
                if config.get("residual", False):
                    raise ValueError("conv VAE decoders with residual upsampling are not supported")
                reduction = config.get("multiplier", 1)
                self.up_blocks.append(DepthToSpaceUpsample(channels, _UPSAMPLE_STRIDES[name], reduction))
                channels //= reduction
            else:
                raise ValueError(f"unsupported decoder block: {name}")

        self.conv_norm_out = PixelNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = SymmetricConv3d(channels, out_channels * patch_size**2)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.per_channel_statistics.un_normalize(sample)
        sample = self.conv_in(sample)
        for up_block in self.up_blocks:
            sample = up_block(sample)
        sample = self.conv_out(self.conv_act(self.conv_norm_out(sample)))
        return unpatchify(sample, patch_size_hw=self.patch_size)

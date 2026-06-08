# SPDX-FileCopyrightText: hzwer / Practical-RIFE contributors
# SPDX-License-Identifier: MIT AND AGPL-3.0
# Code vendored from: https://github.com/hzwer/Practical-RIFE
"""RIFE 4.6 "HDv3" IFNet, transcribed for inference.

This targets the upstream Practical-RIFE 4.6 ``flownet.pkl`` weights. The block
channel widths must match the checkpoint exactly for ``load_state_dict`` to
succeed; if you hit key/shape mismatches, use a TorchScript checkpoint instead
(``RifeFrameGenerator`` loads either format).

License note: RIFE training weights from Practical-RIFE carry non-commercial
terms — verify the license of any weights you ship.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from jasna.models.rife.warp import warp


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes),
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes: int, c: int = 64):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock0 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock1 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock2 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock3 = nn.Sequential(conv(c, c), conv(c, c))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 4, 4, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 1, 4, 2, 1),
        )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * (1.0 / scale)
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat
        tmp = self.conv1(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp * scale
        mask = self.conv2(feat)
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False)
        return flow, mask


class IFNet(nn.Module):
    # RIFE 4.6 HDv3 channel widths. block0 takes (img0,img1,timestep)=7 ch;
    # later blocks take (warped0,warped1,timestep,mask)=8 ch + flow(4) inside.
    def __init__(self, channels=(192, 128, 96, 64)):
        super().__init__()
        self.block0 = IFBlock(7, c=channels[0])
        self.block1 = IFBlock(8 + 4, c=channels[1])
        self.block2 = IFBlock(8 + 4, c=channels[2])
        self.block3 = IFBlock(8 + 4, c=channels[3])

    # scale_list matches upstream RIFE 4.6's default [8, 4, 2, 1]: the weights
    # were trained with this coarse-to-fine pyramid, so other values degrade flow.
    def forward(self, img0, img1, timestep=0.5, scale_list=(8, 4, 2, 1)):
        if not torch.is_tensor(timestep):
            timestep = (img0[:, :1].clone() * 0 + 1) * timestep
        blocks = [self.block0, self.block1, self.block2, self.block3]
        flow = None
        mask = None
        warped_img0 = img0
        warped_img1 = img1
        for i in range(4):
            if flow is None:
                flow, mask = blocks[i](torch.cat((img0, img1, timestep), 1), None, scale=scale_list[i])
            else:
                fd, md = blocks[i](
                    torch.cat((warped_img0, warped_img1, timestep, mask), 1), flow, scale=scale_list[i]
                )
                flow = flow + fd
                mask = mask + md
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return warped_img0 * mask + warped_img1 * (1 - mask)

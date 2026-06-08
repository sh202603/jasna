# SPDX-FileCopyrightText: hzwer / Practical-RIFE contributors
# SPDX-License-Identifier: MIT AND AGPL-3.0
# Code vendored from: https://github.com/hzwer/Practical-RIFE
"""Vendored RIFE (Real-time Intermediate Flow Estimation) model code.

Used by ``jasna.framegen.rife_frame_generator`` for neural frame interpolation.
``IFNet`` is a faithful transcription of the RIFE 4.6 "HDv3" architecture and is
intended to load the upstream ``flownet.pkl`` weights. When in doubt about
weight compatibility, prefer a TorchScript checkpoint (see
``RifeFrameGenerator`` docs), which bundles architecture + weights and is the
guaranteed-correct path.
"""

from jasna.models.rife.ifnet import IFNet

__all__ = ["IFNet"]

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(
    in_channels: int,
    out_channels: int,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = None,
) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )

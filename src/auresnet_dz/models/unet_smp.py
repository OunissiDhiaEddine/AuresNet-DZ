from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(
    in_channels: int,
    out_channels: int,
    encoder_name: str = "resnet50",
    encoder_weights: str | None = None,
    architecture: str = "unet",
) -> nn.Module:
    if architecture.lower() == "unetplusplus":
        model_cls = smp.UnetPlusPlus
    else:
        model_cls = smp.Unet

    return model_cls(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )

from __future__ import annotations

from typing import Sequence

import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class GfsToEra5LightningModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        target_variable_names: Sequence[str] | None = None,
        target_mean: torch.Tensor | None = None,
        target_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.target_variable_names = list(target_variable_names or [])
        self._target_mean = target_mean.detach().clone().float() if target_mean is not None else None
        self._target_std = target_std.detach().clone().float() if target_std is not None else None
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _denormalize_targets(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._target_mean is None or self._target_std is None:
            return tensor

        view_shape = [1] * tensor.ndim
        view_shape[1] = -1
        mean = self._target_mean.to(device=tensor.device, dtype=tensor.dtype).view(*view_shape)
        std = self._target_std.to(device=tensor.device, dtype=tensor.dtype).view(*view_shape)
        return tensor * std + mean

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"), on_epoch=True, on_step=False)

        y_hat_real = self._denormalize_targets(y_hat)
        y_real = self._denormalize_targets(y)
        per_variable_mae = (y_hat_real - y_real).abs().mean(dim=(0, 2, 3))
        aggregate_mae = per_variable_mae.mean()

        self.log(f"{stage}_mae", aggregate_mae, prog_bar=True, on_epoch=True, on_step=False)

        if self.target_variable_names and len(self.target_variable_names) == int(per_variable_mae.numel()):
            metric_values = {
                f"{stage}_mae/{variable_name}": per_variable_mae[index]
                for index, variable_name in enumerate(self.target_variable_names)
            }
            self.log_dict(metric_values, on_epoch=True, on_step=False, prog_bar=False)

        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "train")
        self.log("train_loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

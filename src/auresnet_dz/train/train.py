from __future__ import annotations

from dataclasses import dataclass

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from auresnet_dz.data.datamodule import DataConfig, WrfEra5DataModule
from auresnet_dz.models.unet_smp import build_unet
from auresnet_dz.train.lightning_module import WrfToEra5LightningModule
from auresnet_dz.utils import seed_everything


@dataclass
class ModelConfig:
    in_channels: int
    out_channels: int
    encoder_name: str
    encoder_weights: str | None = None


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))

    data_cfg = DataConfig(**cfg.data)
    datamodule = WrfEra5DataModule(cfg=data_cfg)

    model_cfg = ModelConfig(**cfg.model)
    net = build_unet(
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        encoder_name=model_cfg.encoder_name,
        encoder_weights=model_cfg.encoder_weights,
    )

    lightning_module = WrfToEra5LightningModule(
        model=net,
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )

    ckpt = ModelCheckpoint(
        dirpath=cfg.train.checkpoint_dir,
        monitor="val_mae",
        mode="min",
        save_top_k=2,
        save_last=True,
    )
    logger = TensorBoardLogger("logs", name=cfg.experiment_name)

    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        callbacks=[ckpt],
        logger=logger,
        log_every_n_steps=int(cfg.train.log_every_n_steps),
    )

    trainer.fit(lightning_module, datamodule=datamodule)
    trainer.test(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()

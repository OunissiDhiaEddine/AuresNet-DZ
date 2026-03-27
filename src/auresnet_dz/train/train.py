from __future__ import annotations

from dataclasses import dataclass

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
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

    torch.set_float32_matmul_precision(str(cfg.train.matmul_precision))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(cfg.train.cudnn_benchmark)
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.train.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.train.allow_tf32)

    data_cfg = DataConfig(**cfg.data)
    datamodule = WrfEra5DataModule(cfg=data_cfg)

    model_cfg = ModelConfig(**cfg.model)
    net = build_unet(
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        encoder_name=model_cfg.encoder_name,
        encoder_weights=model_cfg.encoder_weights,
    )
    if bool(cfg.train.torch_compile):
        net = torch.compile(net, mode=str(cfg.train.torch_compile_mode))

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
    early_stopping = EarlyStopping(
        monitor="val_mae",
        mode="min",
        patience=int(cfg.train.early_stopping_patience),
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger("logs", name=cfg.experiment_name)

    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        num_nodes=int(cfg.train.num_nodes),
        strategy=cfg.train.strategy,
        precision=cfg.train.precision,
        accumulate_grad_batches=int(cfg.train.accumulate_grad_batches),
        gradient_clip_val=float(cfg.train.gradient_clip_val),
        deterministic=bool(cfg.train.deterministic),
        callbacks=[ckpt, early_stopping, lr_monitor],
        logger=logger,
        log_every_n_steps=int(cfg.train.log_every_n_steps),
    )

    trainer.fit(lightning_module, datamodule=datamodule)
    trainer.test(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()

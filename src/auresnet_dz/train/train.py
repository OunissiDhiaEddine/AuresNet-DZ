from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from auresnet_dz.data.datamodule import CclmEra5DataModule, DataConfig
from auresnet_dz.models.unet_smp import build_unet
from auresnet_dz.train.lightning_module import CclmToEra5LightningModule
from auresnet_dz.utils import seed_everything

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    in_channels: int
    out_channels: int
    encoder_name: str
    encoder_weights: str | None = None
    architecture: str | None = None


def verify_training_data(cfg: DictConfig) -> None:
    """Verify CCLM and ERA5 data before training.

    Args:
        cfg: Hydra configuration

    Raises:
        RuntimeError: If data verification fails
    """
    from auresnet_dz.data.verification import verify_cclm_era5_pair

    logger.info("=" * 70)
    logger.info("VERIFYING TRAINING DATA")
    logger.info("=" * 70)

    if not bool(cfg.data.get("verify_data", False)):
        logger.info("Data verification disabled in config")
        return

    import glob

    cclm_files = glob.glob(cfg.data.raw_cclm_glob)
    era5_files = glob.glob(cfg.data.raw_era5_glob)

    if not cclm_files:
        raise FileNotFoundError(f"No CCLM files found for glob: {cfg.data.raw_cclm_glob}")
    if not era5_files:
        raise FileNotFoundError(f"No ERA5 files found for glob: {cfg.data.raw_era5_glob}")

    cclm_file = cclm_files[0]
    era5_file = era5_files[0]

    logger.info(f"Verifying CCLM: {cclm_file}")
    logger.info(f"Verifying ERA5: {era5_file}")

    required_vars = list(set(cfg.data.input_variables + cfg.data.target_variables))
    report = verify_cclm_era5_pair(cclm_file, era5_file, required_variables=required_vars)

    if not report.is_valid:
        logger.error("\n" + "=" * 70)
        logger.error("DATA VERIFICATION FAILED!")
        logger.error("=" * 70)
        for error in report.errors:
            logger.error(f"  ✗ {error}")
        raise RuntimeError(f"Data verification failed: {'; '.join(report.errors)}")

    logger.info("\n✓ Data verification passed successfully!")
    logger.info("=" * 70 + "\n")


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))

    # Verify data before proceeding with training
    verify_training_data(cfg)

    require_gpu = bool(cfg.train.require_gpu)
    accelerator = str(cfg.train.accelerator).lower()
    if require_gpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 0 or torch.version.cuda is None:
            raise RuntimeError(
                "GPU-only mode is enabled (train.require_gpu=true), but CUDA is not available in the current "
                "Python environment. Install a CUDA-enabled PyTorch build and run on the RTX GPU."
            )
        if accelerator == "cpu":
            raise ValueError(
                "GPU-only mode is enabled (train.require_gpu=true), but train.accelerator is set to 'cpu'. "
                "Use train.accelerator=gpu."
            )

    torch.set_float32_matmul_precision(str(cfg.train.matmul_precision))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(cfg.train.cudnn_benchmark)
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.train.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.train.allow_tf32)

    data_cfg_dict = cast(dict[str, Any], OmegaConf.to_container(cfg.data, resolve=True))
    data_cfg = DataConfig(**data_cfg_dict)
    datamodule = CclmEra5DataModule(cfg=data_cfg)

    expected_in_channels = len(data_cfg.input_variables)
    expected_out_channels = len(data_cfg.target_variables)
    cfg.model.in_channels = expected_in_channels
    cfg.model.out_channels = expected_out_channels

    model_cfg = ModelConfig(**cfg.model)
    net = build_unet(
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        encoder_name=model_cfg.encoder_name,
        encoder_weights=model_cfg.encoder_weights,
    )
    if bool(cfg.train.torch_compile):
        net = torch.compile(net, mode=str(cfg.train.torch_compile_mode))
    net = cast(torch.nn.Module, net)

    lightning_module = CclmToEra5LightningModule(
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

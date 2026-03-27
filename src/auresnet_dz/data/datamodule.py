from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from auresnet_dz.data.io import open_mfdataset
from auresnet_dz.data.pairs import align_time_and_space


@dataclass
class DataConfig:
    raw_wrf_glob: str
    raw_era5_glob: str
    input_variables: list[str]
    target_variables: list[str]
    time_dim: str = "time"
    batch_size: int = 4
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


class WrfEra5Dataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class WrfEra5DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        wrf = open_mfdataset(self.cfg.raw_wrf_glob)
        era5 = open_mfdataset(self.cfg.raw_era5_glob)
        wrf, era5 = align_time_and_space(wrf, era5, time_dim=self.cfg.time_dim)

        x = wrf[self.cfg.input_variables].to_array("channel").transpose(self.cfg.time_dim, "channel", ...).values
        y = era5[self.cfg.target_variables].to_array("channel").transpose(self.cfg.time_dim, "channel", ...).values

        full_ds = WrfEra5Dataset(x, y)
        n = len(full_ds)
        n_train = int(n * self.cfg.train_split)
        n_val = int(n * self.cfg.val_split)
        n_test = n - n_train - n_val
        self.train_ds, self.val_ds, self.test_ds = random_split(full_ds, [n_train, n_val, n_test])

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            pin_memory=True,
        )

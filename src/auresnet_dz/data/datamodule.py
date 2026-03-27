from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset, Subset

from auresnet_dz.data.io import open_mfdataset
from auresnet_dz.data.pairs import align_time_and_space


Engine = Literal["netcdf4", "h5netcdf"]


@dataclass
class DataConfig:
    raw_wrf_glob: str
    raw_era5_glob: str
    input_variables: list[str]
    target_variables: list[str]
    time_dim: str = "time"
    lat_dim: str = "lat"
    lon_dim: str = "lon"
    engine: Engine = "h5netcdf"
    chunks_time: int = 24
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    split_seed: int = 42


class LazyWrfEra5Dataset(Dataset):
    """Loads one timestamp sample at a time to avoid full in-memory materialization."""

    def __init__(
        self,
        wrf: xr.Dataset,
        era5: xr.Dataset,
        input_variables: list[str],
        target_variables: list[str],
        time_dim: str,
    ) -> None:
        self.wrf = wrf
        self.era5 = era5
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.time_dim = time_dim
        self.n_samples = int(wrf.sizes[time_dim])

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_da = self.wrf[self.input_variables].to_array("channel").isel({self.time_dim: idx})
        y_da = self.era5[self.target_variables].to_array("channel").isel({self.time_dim: idx})

        x_np = x_da.load().transpose("channel", ...).values
        y_np = y_da.load().transpose("channel", ...).values

        x = torch.tensor(x_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)
        return x, y


class WrfEra5DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        chunks = {self.cfg.time_dim: self.cfg.chunks_time}
        wrf = open_mfdataset(self.cfg.raw_wrf_glob, engine=self.cfg.engine, chunks=chunks)
        era5 = open_mfdataset(self.cfg.raw_era5_glob, engine=self.cfg.engine, chunks=chunks)
        wrf, era5 = align_time_and_space(wrf, era5, time_dim=self.cfg.time_dim)

        full_ds = LazyWrfEra5Dataset(
            wrf=wrf,
            era5=era5,
            input_variables=self.cfg.input_variables,
            target_variables=self.cfg.target_variables,
            time_dim=self.cfg.time_dim,
        )
        n = len(full_ds)
        if n < 3:
            raise ValueError(f"Need at least 3 aligned timesteps, found {n}.")

        n_train = int(n * self.cfg.train_split)
        n_val = int(n * self.cfg.val_split)
        n_test = n - n_train - n_val

        if min(n_train, n_val, n_test) <= 0:
            raise ValueError(
                "Invalid split sizes. Adjust train_split/val_split/test_split so all splits are non-zero."
            )

        g = torch.Generator().manual_seed(self.cfg.split_seed)
        indices = torch.randperm(n, generator=g).tolist()

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        self.train_ds = Subset(full_ds, train_idx)
        self.val_ds = Subset(full_ds, val_idx)
        self.test_ds = Subset(full_ds, test_idx)

    def _loader_kwargs(self) -> dict:
        kwargs = {
            "batch_size": self.cfg.batch_size,
            "num_workers": self.cfg.num_workers,
            "pin_memory": self.cfg.pin_memory,
        }
        if self.cfg.num_workers > 0:
            kwargs["persistent_workers"] = self.cfg.persistent_workers
            kwargs["prefetch_factor"] = self.cfg.prefetch_factor
        return kwargs

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            shuffle=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None
        return DataLoader(
            self.test_ds,
            shuffle=False,
            **self._loader_kwargs(),
        )

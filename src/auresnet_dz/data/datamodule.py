from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset, Subset

from auresnet_dz.data.io import open_mfdataset
from auresnet_dz.data.pairs import align_time_and_space
from auresnet_dz.data.verification import verify_gfs_era5_pair

logger = logging.getLogger(__name__)


Engine = Literal["netcdf4", "h5netcdf"]


@dataclass
class DataConfig:
    raw_gfs_glob: str
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
    verify_data: bool = False
    """If True, verify data compatibility before training."""


@dataclass
class ChannelNormalizationStats:
    variable_names: list[str]
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_dataset(
        cls,
        ds: xr.Dataset,
        variable_names: list[str],
        time_dim: str,
        time_indices: list[int] | None = None,
    ) -> "ChannelNormalizationStats":
        if time_indices is not None:
            ds = ds.isel({time_dim: time_indices})

        means: list[float] = []
        stds: list[float] = []
        for variable_name in variable_names:
            values = np.asarray(ds[variable_name].astype("float64").values)
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values))
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0
            means.append(mean)
            stds.append(std)

        return cls(
            variable_names=list(variable_names),
            mean=torch.tensor(means, dtype=torch.float32),
            std=torch.tensor(stds, dtype=torch.float32),
        )

    def _channel_view(self, tensor: torch.Tensor, channel_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        shape = [1] * tensor.ndim
        shape[channel_dim] = -1
        mean = self.mean.to(device=tensor.device, dtype=tensor.dtype).view(*shape)
        std = self.std.to(device=tensor.device, dtype=tensor.dtype).view(*shape)
        return mean, std

    def normalize(self, tensor: torch.Tensor, channel_dim: int) -> torch.Tensor:
        mean, std = self._channel_view(tensor, channel_dim)
        return (tensor - mean) / std

    def denormalize(self, tensor: torch.Tensor, channel_dim: int) -> torch.Tensor:
        mean, std = self._channel_view(tensor, channel_dim)
        return tensor * std + mean


class LazyGfsEra5Dataset(Dataset):
    """Loads one timestamp sample at a time to avoid full in-memory materialization."""

    def __init__(
        self,
        gfs: xr.Dataset,
        era5: xr.Dataset,
        input_variables: list[str],
        target_variables: list[str],
        time_dim: str,
        input_normalization: ChannelNormalizationStats | None = None,
        target_normalization: ChannelNormalizationStats | None = None,
    ) -> None:
        self.gfs = gfs
        self.era5 = era5
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.time_dim = time_dim
        self.input_normalization = input_normalization
        self.target_normalization = target_normalization
        self.n_samples = int(gfs.sizes[time_dim])

    def set_normalization(
        self,
        input_normalization: ChannelNormalizationStats | None,
        target_normalization: ChannelNormalizationStats | None,
    ) -> None:
        self.input_normalization = input_normalization
        self.target_normalization = target_normalization

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_da = self.gfs[self.input_variables].to_array("channel").isel({self.time_dim: idx})
        y_da = self.era5[self.target_variables].to_array("channel").isel({self.time_dim: idx})

        x_np = x_da.load().transpose("channel", ...).values
        y_np = y_da.load().transpose("channel", ...).values

        x = torch.tensor(x_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)

        if self.input_normalization is not None:
            x = self.input_normalization.normalize(x, channel_dim=0)
        if self.target_normalization is not None:
            y = self.target_normalization.normalize(y, channel_dim=0)

        return x, y


class GfsEra5DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None
        self.input_normalization: ChannelNormalizationStats | None = None
        self.target_normalization: ChannelNormalizationStats | None = None
        self._is_setup = False

    def setup(self, stage: str | None = None) -> None:
        if self._is_setup:
            return

        if self.cfg.verify_data:
            logger.info("Running data verification...")
            self._verify_raw_data()

        chunks = {self.cfg.time_dim: self.cfg.chunks_time}
        gfs = open_mfdataset(self.cfg.raw_gfs_glob, engine=self.cfg.engine, chunks=chunks)
        era5 = open_mfdataset(self.cfg.raw_era5_glob, engine=self.cfg.engine, chunks=chunks)
        gfs, era5 = align_time_and_space(gfs, era5, time_dim=self.cfg.time_dim)

        full_ds = LazyGfsEra5Dataset(
            gfs=gfs,
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

        self.input_normalization = ChannelNormalizationStats.from_dataset(
            gfs,
            self.cfg.input_variables,
            time_dim=self.cfg.time_dim,
            time_indices=train_idx,
        )
        self.target_normalization = ChannelNormalizationStats.from_dataset(
            era5,
            self.cfg.target_variables,
            time_dim=self.cfg.time_dim,
            time_indices=train_idx,
        )

        full_ds.set_normalization(self.input_normalization, self.target_normalization)

        self.train_ds = Subset(full_ds, train_idx)
        self.val_ds = Subset(full_ds, val_idx)
        self.test_ds = Subset(full_ds, test_idx)
        self._is_setup = True

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

    def _verify_raw_data(self) -> None:
        """Verify raw GFS and ERA5 data files for compatibility.

        Raises:
            RuntimeError: If verification fails
        """
        import glob

        gfs_files = glob.glob(self.cfg.raw_gfs_glob, recursive=True)
        era5_files = glob.glob(self.cfg.raw_era5_glob)

        if not gfs_files:
            raise FileNotFoundError(f"No GFS files found for glob: {self.cfg.raw_gfs_glob}")
        if not era5_files:
            raise FileNotFoundError(f"No ERA5 files found for glob: {self.cfg.raw_era5_glob}")

        # Use first file from each glob pattern for verification
        gfs_file = gfs_files[0]
        era5_file = era5_files[0]

        report = verify_gfs_era5_pair(
            gfs_file,
            era5_file,
            required_variables=self.cfg.input_variables + self.cfg.target_variables,
        )

        if not report.is_valid:
            raise RuntimeError(
                f"Data verification failed with {len(report.errors)} error(s): "
                + "; ".join(report.errors)
            )

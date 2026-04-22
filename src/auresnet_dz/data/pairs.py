from __future__ import annotations

import xarray as xr


def align_time_and_space(
    gfs: xr.Dataset,
    era5: xr.Dataset,
    time_dim: str = "time",
) -> tuple[xr.Dataset, xr.Dataset]:
    gfs_aligned, era5_aligned = xr.align(gfs, era5, join="inner")
    if time_dim in gfs_aligned.dims and time_dim in era5_aligned.dims:
        common_time = gfs_aligned[time_dim].to_index().intersection(era5_aligned[time_dim].to_index())
        gfs_aligned = gfs_aligned.sel({time_dim: common_time})
        era5_aligned = era5_aligned.sel({time_dim: common_time})
    return gfs_aligned, era5_aligned

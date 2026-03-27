from __future__ import annotations

import xarray as xr


def align_time_and_space(
    wrf: xr.Dataset,
    era5: xr.Dataset,
    time_dim: str = "time",
) -> tuple[xr.Dataset, xr.Dataset]:
    wrf_aligned, era5_aligned = xr.align(wrf, era5, join="inner")
    if time_dim in wrf_aligned.dims and time_dim in era5_aligned.dims:
        common_time = wrf_aligned[time_dim].to_index().intersection(era5_aligned[time_dim].to_index())
        wrf_aligned = wrf_aligned.sel({time_dim: common_time})
        era5_aligned = era5_aligned.sel({time_dim: common_time})
    return wrf_aligned, era5_aligned

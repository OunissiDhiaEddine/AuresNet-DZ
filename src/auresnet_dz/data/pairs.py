from __future__ import annotations

import xarray as xr


def align_time_and_space(
    cclm: xr.Dataset,
    era5: xr.Dataset,
    time_dim: str = "time",
) -> tuple[xr.Dataset, xr.Dataset]:
    cclm_aligned, era5_aligned = xr.align(cclm, era5, join="inner")
    if time_dim in cclm_aligned.dims and time_dim in era5_aligned.dims:
        common_time = cclm_aligned[time_dim].to_index().intersection(era5_aligned[time_dim].to_index())
        cclm_aligned = cclm_aligned.sel({time_dim: common_time})
        era5_aligned = era5_aligned.sel({time_dim: common_time})
    return cclm_aligned, era5_aligned

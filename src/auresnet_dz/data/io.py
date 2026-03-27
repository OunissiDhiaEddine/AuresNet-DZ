from __future__ import annotations

from typing import Literal

import xarray as xr


Engine = Literal["netcdf4", "h5netcdf"]


def open_mfdataset(path_glob: str, engine: Engine = "netcdf4", chunks: dict | None = None) -> xr.Dataset:
    return xr.open_mfdataset(
        path_glob,
        combine="by_coords",
        engine=engine,
        chunks=chunks or {"time": 24},
        parallel=True,
    )

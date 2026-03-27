from __future__ import annotations

import xarray as xr
import xesmf as xe


def build_regridder(
    source: xr.Dataset,
    target: xr.Dataset,
    method: str = "bilinear",
    periodic: bool = False,
    reuse_weights: bool = False,
) -> xe.Regridder:
    return xe.Regridder(
        source,
        target,
        method=method,
        periodic=periodic,
        reuse_weights=reuse_weights,
    )


def regrid_dataset(ds: xr.Dataset, regridder: xe.Regridder) -> xr.Dataset:
    return regridder(ds)

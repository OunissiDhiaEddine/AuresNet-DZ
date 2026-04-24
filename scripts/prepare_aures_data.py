from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


GFS_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "t2m": ("t2m", "2t"),
    "u10": ("u10", "10u"),
    "v10": ("v10", "10v"),
    "sp": ("sp", "surface_pressure"),
    "tp": ("tp", "total_precipitation"),
}

ERA5_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "t2m": ("t2m", "2t"),
    "u10": ("u10", "10u"),
    "v10": ("v10", "10v"),
    "sp": ("sp", "surface_pressure"),
    "tp": ("tp", "total_precipitation"),
    "d2m": ("d2m", "2d"),
}


def _extract_time_value(ds: xr.Dataset, path: str) -> pd.Timestamp:
    if "valid_time" in ds.coords:
        return pd.Timestamp(pd.to_datetime(ds["valid_time"].values))
    if "time" in ds.coords:
        return pd.Timestamp(pd.to_datetime(ds["time"].values))

    match = re.search(r"gfs_(\d{8})_(\d{2})_f(\d{3})", Path(path).stem, flags=re.IGNORECASE)
    if match:
        init_time = pd.Timestamp(f"{match.group(1)} {match.group(2)}:00:00")
        return init_time + pd.to_timedelta(int(match.group(3)), unit="h")

    raise ValueError(f"Could not determine valid time for source file: {path}")


def _canonicalize_variables(ds: xr.Dataset, aliases: dict[str, tuple[str, ...]]) -> xr.Dataset:
    rename_map: dict[str, str] = {}
    for canonical_var, candidate_vars in aliases.items():
        if canonical_var in ds.data_vars:
            continue
        for candidate in candidate_vars:
            if candidate in ds.data_vars:
                rename_map[candidate] = canonical_var
                break

    if rename_map:
        ds = ds.rename(rename_map)

    return ds


def _normalize_gfs(ds: xr.Dataset) -> xr.Dataset:
    rename_dims: dict[str, str] = {}
    if "latitude" in ds.dims:
        rename_dims["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename_dims["longitude"] = "lon"
    if rename_dims:
        ds = ds.rename(rename_dims)

    drop_coords = [c for c in ["step", "heightAboveGround", "surface"] if c in ds.coords]
    if drop_coords:
        ds = ds.drop_vars(drop_coords)

    return ds


def _open_gfs_merged(raw_source_glob: str) -> xr.Dataset:
    source_files = sorted(glob.glob(raw_source_glob, recursive=True))
    if not source_files:
        raise FileNotFoundError(f"No source files found for glob: {raw_source_glob}")

    pieces: list[xr.Dataset] = []
    for path in source_files:
        ds = xr.open_dataset(path)
        ds = _normalize_gfs(ds)
        ds = _canonicalize_variables(ds, GFS_VAR_ALIASES)

        if "wind10" not in ds.data_vars and {"u10", "v10"}.issubset(ds.data_vars):
            ds["wind10"] = np.hypot(ds["u10"], ds["v10"])

        keep_vars = [var for var in ["t2m", "u10", "v10", "sp", "tp", "wind10"] if var in ds.data_vars]
        if len([var for var in keep_vars if var in {"t2m", "u10", "v10", "sp", "tp"}]) < 3:
            raise ValueError(f"Source file does not expose the required GFS variables: {path}")

        time_value = _extract_time_value(ds, path)
        drop_coords = [c for c in ["time", "valid_time"] if c in ds.coords and c not in ds.dims]
        if drop_coords:
            ds = ds.drop_vars(drop_coords)

        ds = ds[keep_vars].expand_dims(time=[time_value])
        if ds["lat"].ndim == 1 and ds["lat"].values[0] > ds["lat"].values[-1]:
            ds = ds.sortby("lat")
        if ds["lon"].ndim == 1 and ds["lon"].values[0] > ds["lon"].values[-1]:
            ds = ds.sortby("lon")
        pieces.append(ds)

    source = xr.concat(pieces, dim="time").sortby("time")
    time_index = source.get_index("time")
    if time_index.has_duplicates:
        source = source.isel(time=~time_index.duplicated())

    return source


def _normalize_era5(ds: xr.Dataset) -> xr.Dataset:
    rename_dims: dict[str, str] = {}
    if "valid_time" in ds.dims:
        rename_dims["valid_time"] = "time"
    if "latitude" in ds.dims:
        rename_dims["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename_dims["longitude"] = "lon"
    if rename_dims:
        ds = ds.rename(rename_dims)

    drop_coords = [c for c in ["number", "expver"] if c in ds.coords]
    if drop_coords:
        ds = ds.drop_vars(drop_coords)

    if "wind10" not in ds.data_vars and "u10" in ds.data_vars and "v10" in ds.data_vars:
        ds["wind10"] = np.hypot(ds["u10"], ds["v10"])

    return ds


def _canonicalize_era5_variables(ds: xr.Dataset) -> xr.Dataset:
    return _canonicalize_variables(ds, ERA5_VAR_ALIASES)


def _harmonize_tp_to_meters(ds: xr.Dataset, dataset_name: str) -> xr.Dataset:
    if "tp" not in ds.data_vars:
        return ds

    tp = ds["tp"]
    units = str(tp.attrs.get("units", "")).strip().lower()
    factor = 1.0
    reason = "already in meters"

    if any(token in units for token in ("mm", "millimeter", "kg m-2", "kg/m^2", "kg m^-2", "kg m**-2")):
        factor = 1.0e-3
        reason = f"units='{units}' interpreted as mm-equivalent"
    elif units in {"m", "meter", "meters", "metre", "metres"}:
        factor = 1.0
        reason = f"units='{units}' interpreted as meters"
    else:
        # Fallback heuristic for missing/ambiguous units:
        # values above 1 are very unlikely for hourly precipitation in meters.
        n_time = int(tp.sizes.get("time", 0))
        sample = tp if n_time <= 48 else tp.isel(time=slice(0, 48))
        sample_vals = np.asarray(sample.values, dtype=np.float64)
        finite = sample_vals[np.isfinite(sample_vals)]
        if finite.size > 0:
            p99 = float(np.quantile(np.abs(finite), 0.99))
            if p99 > 1.0:
                factor = 1.0e-3
                reason = f"p99={p99:.4g} suggests mm-scale values"
            else:
                reason = f"p99={p99:.4g} suggests meter-scale values"
        else:
            reason = "no finite values; left unchanged"

    if factor != 1.0:
        ds["tp"] = tp * factor
        ds["tp"].attrs.update(tp.attrs)
        ds["tp"].attrs["original_units"] = tp.attrs.get("units", "")
        ds["tp"].attrs["units"] = "m"
        print(f"[{dataset_name}] Converted tp to meters with factor={factor:g} ({reason})")
    else:
        if "units" not in ds["tp"].attrs:
            ds["tp"].attrs["units"] = "m"
        print(f"[{dataset_name}] tp kept as-is ({reason})")

    return ds


def _regrid_source_to_era5_grid(
    source: xr.Dataset,
    target_era5: xr.Dataset,
    variables: list[str],
) -> xr.Dataset:
    if "lat" not in source.coords or "lon" not in source.coords:
        raise ValueError("Source dataset must expose lat/lon coordinates.")
    if "lat" not in target_era5.dims or "lon" not in target_era5.dims:
        raise ValueError("ERA5 target grid must use lat/lon dimensions.")

    if source["lat"].ndim == 1 and source["lon"].ndim == 1:
        source_sorted = source.sortby("lat").sortby("lon")
        regridded = source_sorted[variables].interp(lat=target_era5["lat"].values, lon=target_era5["lon"].values)
        return regridded

    if "rlat" not in source.dims or "rlon" not in source.dims:
        raise ValueError("Unsupported source grid layout for interpolation.")

    src_lat = source["lat"].values
    src_lon = source["lon"].values

    tgt_lat_1d = target_era5["lat"].values
    tgt_lon_1d = target_era5["lon"].values
    tgt_lon_2d, tgt_lat_2d = np.meshgrid(tgt_lon_1d, tgt_lat_1d)

    src_points = np.column_stack([src_lat.ravel(), src_lon.ravel()])
    tgt_points = np.column_stack([tgt_lat_2d.ravel(), tgt_lon_2d.ravel()])

    tree = cKDTree(src_points)
    distances, indices = tree.query(tgt_points, k=4)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)

    eps = 1.0e-12
    weights = 1.0 / np.maximum(distances, eps)
    weights = weights / weights.sum(axis=1, keepdims=True)

    n_time = int(source.sizes["time"])
    n_lat = int(tgt_lat_1d.shape[0])
    n_lon = int(tgt_lon_1d.shape[0])

    out_vars: dict[str, xr.DataArray] = {}
    for var_name in variables:
        if var_name not in source.data_vars:
            raise ValueError(f"Source variable missing after canonicalization: {var_name}")

        arr = source[var_name].transpose("time", "rlat", "rlon").values
        flat = arr.reshape(n_time, -1)

        gathered = flat[:, indices]
        blended = (gathered * weights[None, :, :]).sum(axis=2)
        out = blended.reshape(n_time, n_lat, n_lon)

        out_vars[var_name] = xr.DataArray(
            out,
            dims=("time", "lat", "lon"),
            coords={"time": source["time"].values, "lat": tgt_lat_1d, "lon": tgt_lon_1d},
        )

    return xr.Dataset(out_vars)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GFS source + ERA5 into train-ready Aures datasets.")
    parser.add_argument("--raw-source-glob", default="data/raw/gfs/**/*.nc")
    parser.add_argument("--raw-era5-glob", default="data/raw/era5/*.nc")
    parser.add_argument("--processed-source", default="data/processed/gfs_aures_ready.nc")
    parser.add_argument("--processed-era5", default="data/processed/era5_aures_ready.nc")
    args = parser.parse_args()

    source = _open_gfs_merged(args.raw_source_glob)
    source = _harmonize_tp_to_meters(source, dataset_name="GFS")

    era5_files = sorted(glob.glob(args.raw_era5_glob, recursive=True))
    if not era5_files:
        raise FileNotFoundError(f"No ERA5 files found for glob: {args.raw_era5_glob}")
    era5 = xr.open_mfdataset(era5_files, combine="by_coords", parallel=True)
    era5 = _normalize_era5(era5)
    era5 = _canonicalize_era5_variables(era5)
    era5 = _harmonize_tp_to_meters(era5, dataset_name="ERA5")

    required_era5 = ["t2m", "u10", "v10", "sp", "tp"]
    for var in required_era5:
        if var not in era5.data_vars:
            raise ValueError(f"ERA5 variable missing: {var}")

    source_has_sp = "sp" in source.data_vars
    variables = ["t2m", "u10", "v10", "sp", "tp"]

    if source_has_sp and "sp" not in era5.data_vars:
        raise ValueError("Source includes 'sp' but ERA5 'sp' is missing. Include 'sp' in ERA5 input files.")

    era5 = era5[variables]
    source_regridded = _regrid_source_to_era5_grid(source, era5, variables=variables)

    source_time = pd.DatetimeIndex(source_regridded["time"].values)
    era_time = pd.DatetimeIndex(era5["time"].values)
    common = source_time.intersection(era_time)
    if len(common) < 3:
        raise ValueError(f"Need at least 3 common timesteps after preprocessing, found {len(common)}")

    source_out = source_regridded.sel(time=common)
    era5_out = era5.sel(time=common)

    processed_source = Path(args.processed_source)
    processed_era5 = Path(args.processed_era5)
    processed_source.parent.mkdir(parents=True, exist_ok=True)
    processed_era5.parent.mkdir(parents=True, exist_ok=True)

    source_out.to_netcdf(processed_source)
    era5_out.to_netcdf(processed_era5)

    print("Prepared datasets written:")
    print(f"- GFS: {processed_source} -> dims {dict(source_out.sizes)} vars {list(source_out.data_vars)}")
    print(f"- ERA5: {processed_era5} -> dims {dict(era5_out.sizes)} vars {list(era5_out.data_vars)}")
    print(f"- Common timesteps: {len(common)}")


if __name__ == "__main__":
    main()

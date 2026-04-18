from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


SOURCE_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    # CCLM historical commonly provides tas + sfcWind + psl, while some CORDEX sets may include components.
    "t2m": ("tas", "t2m"),
    "wind10": ("sfcWind", "wind10"),
    "sp": ("ps", "sp", "psl"),
}

ERA5_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "t2m": ("t2m", "2t"),
    "u10": ("u10", "10u"),
    "v10": ("v10", "10v"),
    "sp": ("sp", "surface_pressure"),
}


def _open_source_merged(raw_source_glob: str) -> xr.Dataset:
    source_files = sorted(glob.glob(raw_source_glob))
    if not source_files:
        raise FileNotFoundError(f"No source files found for glob: {raw_source_glob}")

    pieces: list[xr.Dataset] = []
    for canonical_var, candidate_vars in SOURCE_VAR_ALIASES.items():
        match = None
        matched_src_var = None
        for src_var in candidate_vars:
            for path in source_files:
                stem = Path(path).name
                if f"{src_var}_" in stem or stem.startswith(f"{src_var}.") or stem.startswith(f"{src_var}_"):
                    match = path
                    matched_src_var = src_var
                    break
            if match is not None:
                break
        if match is None or matched_src_var is None:
            if canonical_var in {"t2m", "wind10"}:
                raise ValueError(
                    f"Could not find source file for required variable '{canonical_var}' in {raw_source_glob}"
                )
            # Optional channels (e.g., sp) are skipped if unavailable.
            continue

        ds = xr.open_dataset(match)
        if matched_src_var not in ds.data_vars:
            raise ValueError(f"Variable '{matched_src_var}' missing in {match}")

        ds = ds[[matched_src_var]].rename({matched_src_var: canonical_var})
        drop_candidates = [v for v in ["time_bnds", "rotated_pole", "height"] if v in ds.variables]
        if drop_candidates:
            ds = ds.drop_vars(drop_candidates)
        pieces.append(ds)

    if not pieces:
        raise ValueError("No usable source variables were found.")

    source = xr.merge(pieces, compat="override")
    return source


def _normalize_era5(ds: xr.Dataset) -> xr.Dataset:
    rename_dims = {}
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
    rename_map: dict[str, str] = {}
    for canonical_var, candidate_vars in ERA5_VAR_ALIASES.items():
        if canonical_var in ds.data_vars:
            continue
        for candidate in candidate_vars:
            if candidate in ds.data_vars:
                rename_map[candidate] = canonical_var
                break

    if rename_map:
        ds = ds.rename(rename_map)

    return ds


def _regrid_cclm_to_era5_grid(
    source: xr.Dataset,
    target_era5: xr.Dataset,
    variables: list[str],
) -> xr.Dataset:
    if "lat" not in source.coords or "lon" not in source.coords:
        raise ValueError("Source dataset must have 2D lat/lon coordinates.")
    if "rlat" not in source.dims or "rlon" not in source.dims:
        raise ValueError("Source dataset must expose rlat/rlon dimensions.")
    if "lat" not in target_era5.dims or "lon" not in target_era5.dims:
        raise ValueError("ERA5 target grid must use lat/lon dimensions.")

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
            raise ValueError(f"Source variable missing after rename: {var_name}")

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
    parser = argparse.ArgumentParser(description="Prepare CCLM/CORDEX source + ERA5 into train-ready Aures datasets.")
    parser.add_argument("--raw-source-glob", default="data/raw/cclm/*.nc")
    parser.add_argument("--raw-era5-glob", default="data/raw/era5/*.nc")
    parser.add_argument("--processed-source", default="data/processed/cclm_aures_ready.nc")
    parser.add_argument("--processed-era5", default="data/processed/era5_aures_ready.nc")
    args = parser.parse_args()

    source = _open_source_merged(args.raw_source_glob)

    era5_files = sorted(glob.glob(args.raw_era5_glob))
    if not era5_files:
        raise FileNotFoundError(f"No ERA5 files found for glob: {args.raw_era5_glob}")
    era5 = xr.open_mfdataset(era5_files, combine="by_coords", parallel=True)
    era5 = _normalize_era5(era5)
    era5 = _canonicalize_era5_variables(era5)

    required_era5 = ["t2m", "wind10"]
    for var in required_era5:
        if var not in era5.data_vars:
            raise ValueError(f"ERA5 variable missing: {var}")

    source_has_sp = "sp" in source.data_vars
    variables = ["t2m", "wind10"]
    if source_has_sp and "sp" not in era5.data_vars:
        raise ValueError("Source includes 'sp' but ERA5 'sp' is missing. Include 'sp' in ERA5 input files.")

    if source_has_sp:
        variables.append("sp")

    era5 = era5[variables]
    source_regridded = _regrid_cclm_to_era5_grid(source, era5, variables=variables)

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
    print(f"- CCLM: {processed_source} -> dims {dict(source_out.sizes)} vars {list(source_out.data_vars)}")
    print(f"- ERA5: {processed_era5} -> dims {dict(era5_out.sizes)} vars {list(era5_out.data_vars)}")
    print(f"- Common timesteps: {len(common)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


WRF_VAR_MAP = {
    "tas": "t2m",
    "uas": "u10",
    "vas": "v10",
    "ps": "sp",
}


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _open_wrf_merged(raw_wrf_glob: str) -> xr.Dataset:
    wrf_files = sorted(glob.glob(raw_wrf_glob))
    if not wrf_files:
        raise FileNotFoundError(f"No WRF files found for glob: {raw_wrf_glob}")

    pieces: list[xr.Dataset] = []
    for src_var, dst_var in WRF_VAR_MAP.items():
        match = None
        for path in wrf_files:
            stem = Path(path).name
            if f"{src_var}_" in stem or stem.startswith(f"{src_var}.") or stem.startswith(f"{src_var}_"):
                match = path
                break
        if match is None:
            raise ValueError(f"Could not find WRF file for variable '{src_var}' in {raw_wrf_glob}")

        ds = xr.open_dataset(match)
        if src_var not in ds.data_vars:
            raise ValueError(f"Variable '{src_var}' missing in {match}")

        ds = ds[[src_var]].rename({src_var: dst_var})
        drop_candidates = [v for v in ["time_bnds", "rotated_pole", "height"] if v in ds.variables]
        if drop_candidates:
            ds = ds.drop_vars(drop_candidates)
        pieces.append(ds)

    wrf = xr.merge(pieces, compat="override")
    return wrf


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
    return ds


def _download_era5_sp_if_needed(
    era5: xr.Dataset,
    out_path: Path,
) -> xr.Dataset:
    if "sp" in era5.data_vars:
        return era5

    cds_url = os.getenv("CDSAPI_URL", "").strip()
    cds_key = os.getenv("CDSAPI_KEY", "").strip()
    if not cds_url or not cds_key:
        raise RuntimeError(
            "ERA5 variable 'sp' is missing and CDS credentials are not set (CDSAPI_URL/CDSAPI_KEY)."
        )

    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError("cdsapi is required to download ERA5 surface pressure.") from exc

    times = pd.DatetimeIndex(era5["time"].values)
    years = sorted({f"{t.year:04d}" for t in times})
    months = sorted({f"{t.month:02d}" for t in times})
    days = sorted({f"{t.day:02d}" for t in times})
    hours = [f"{h:02d}:00" for h in range(24)]

    lat_vals = era5["lat"].values
    lon_vals = era5["lon"].values
    north = float(np.nanmax(lat_vals))
    south = float(np.nanmin(lat_vals))
    west = float(np.nanmin(lon_vals))
    east = float(np.nanmax(lon_vals))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client(url=cds_url, key=cds_key, quiet=True)
    request = {
        "product_type": "reanalysis",
        "variable": "surface_pressure",
        "year": years,
        "month": months,
        "day": days,
        "time": hours,
        "area": [north, west, south, east],
        "format": "netcdf",
    }
    client.retrieve("reanalysis-era5-single-levels", request, str(out_path))

    sp = xr.open_dataset(out_path)
    sp = _normalize_era5(sp)
    if "sp" not in sp.data_vars:
        raise RuntimeError("Downloaded ERA5 file did not contain 'sp'.")

    sp = sp[["sp"]]
    sp = sp.interp(lat=era5["lat"], lon=era5["lon"], method="nearest")
    sp = sp.sel(time=era5["time"])

    return xr.merge([era5, sp], compat="override")


def _regrid_wrf_to_era5_grid(
    wrf: xr.Dataset,
    target_era5: xr.Dataset,
    variables: list[str],
) -> xr.Dataset:
    if "lat" not in wrf.coords or "lon" not in wrf.coords:
        raise ValueError("WRF dataset must have 2D lat/lon coordinates.")
    if "rlat" not in wrf.dims or "rlon" not in wrf.dims:
        raise ValueError("WRF dataset must expose rlat/rlon dimensions.")
    if "lat" not in target_era5.dims or "lon" not in target_era5.dims:
        raise ValueError("ERA5 target grid must use lat/lon dimensions.")

    src_lat = wrf["lat"].values
    src_lon = wrf["lon"].values

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

    n_time = int(wrf.sizes["time"])
    n_lat = int(tgt_lat_1d.shape[0])
    n_lon = int(tgt_lon_1d.shape[0])

    out_vars: dict[str, xr.DataArray] = {}
    for var_name in variables:
        if var_name not in wrf.data_vars:
            raise ValueError(f"WRF variable missing after rename: {var_name}")

        arr = wrf[var_name].transpose("time", "rlat", "rlon").values
        flat = arr.reshape(n_time, -1)

        gathered = flat[:, indices]
        blended = (gathered * weights[None, :, :]).sum(axis=2)
        out = blended.reshape(n_time, n_lat, n_lon)

        out_vars[var_name] = xr.DataArray(
            out,
            dims=("time", "lat", "lon"),
            coords={"time": wrf["time"].values, "lat": tgt_lat_1d, "lon": tgt_lon_1d},
        )

    return xr.Dataset(out_vars)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CORDEX WRF + ERA5 into train-ready Aures datasets.")
    parser.add_argument("--raw-wrf-glob", default="data/raw/wrf/*.nc")
    parser.add_argument("--raw-era5-glob", default="data/raw/era5/*.nc")
    parser.add_argument("--processed-wrf", default="data/processed/wrf_aures_ready.nc")
    parser.add_argument("--processed-era5", default="data/processed/era5_aures_ready.nc")
    parser.add_argument("--era5-sp-cache", default="data/raw/era5/era5_sp_download.nc")
    parser.add_argument("--skip-era5-sp-download", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv(repo_root / ".env")

    wrf = _open_wrf_merged(args.raw_wrf_glob)

    era5_files = sorted(glob.glob(args.raw_era5_glob))
    if not era5_files:
        raise FileNotFoundError(f"No ERA5 files found for glob: {args.raw_era5_glob}")
    era5 = xr.open_mfdataset(era5_files, combine="by_coords", parallel=True)
    era5 = _normalize_era5(era5)

    required_era5 = ["t2m", "u10", "v10"]
    for var in required_era5:
        if var not in era5.data_vars:
            raise ValueError(f"ERA5 variable missing: {var}")

    if not args.skip_era5_sp_download:
        era5 = _download_era5_sp_if_needed(era5, Path(args.era5_sp_cache))

    variables = ["t2m", "u10", "v10"]
    if "sp" in wrf.data_vars and "sp" in era5.data_vars:
        variables.append("sp")

    era5 = era5[variables]
    wrf_regridded = _regrid_wrf_to_era5_grid(wrf, era5, variables=variables)

    wrf_time = pd.DatetimeIndex(wrf_regridded["time"].values)
    era_time = pd.DatetimeIndex(era5["time"].values)
    common = wrf_time.intersection(era_time)
    if len(common) < 3:
        raise ValueError(f"Need at least 3 common timesteps after preprocessing, found {len(common)}")

    wrf_out = wrf_regridded.sel(time=common)
    era5_out = era5.sel(time=common)

    processed_wrf = Path(args.processed_wrf)
    processed_era5 = Path(args.processed_era5)
    processed_wrf.parent.mkdir(parents=True, exist_ok=True)
    processed_era5.parent.mkdir(parents=True, exist_ok=True)

    wrf_out.to_netcdf(processed_wrf)
    era5_out.to_netcdf(processed_era5)

    print("Prepared datasets written:")
    print(f"- WRF : {processed_wrf} -> dims {dict(wrf_out.sizes)} vars {list(wrf_out.data_vars)}")
    print(f"- ERA5: {processed_era5} -> dims {dict(era5_out.sizes)} vars {list(era5_out.data_vars)}")
    print(f"- Common timesteps: {len(common)}")


if __name__ == "__main__":
    main()

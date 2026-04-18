from __future__ import annotations

import argparse
import os
from pathlib import Path

import cdsapi
from tqdm import tqdm


# CCLM filename token -> ERA5 single-level variables.
# Some CCLM vars (e.g., huss) are not directly available in ERA5 single-level output,
# so we download helper fields needed for later conversion.
CCLM_TO_ERA5: dict[str, list[str]] = {
    "tas": ["2m_temperature"],
    "sfcWind": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
    "pr": ["total_precipitation"],
    "psl": ["mean_sea_level_pressure"],
    "orog": ["geopotential"],
    "huss": ["2m_dewpoint_temperature", "surface_pressure"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ERA5 data matching existing CCLM variables, period, and region."
    )
    parser.add_argument("--cclm-dir", default="data/raw/cclm", help="Directory with raw CCLM netCDF files.")
    parser.add_argument("--out-dir", default="data/raw/era5", help="Output directory for ERA5 netCDF files.")
    parser.add_argument("--env-file", default=".env", help="Path to .env file containing CDS credentials.")
    parser.add_argument("--start-year", type=int, default=1990, help="Start year (inclusive).")
    parser.add_argument("--end-year", type=int, default=2005, help="End year (inclusive).")

    parser.add_argument("--north", type=float, default=36.5, help="North bound.")
    parser.add_argument("--south", type=float, default=35.0, help="South bound.")
    parser.add_argument("--west", type=float, default=4.5, help="West bound.")
    parser.add_argument("--east", type=float, default=8.5, help="East bound.")
    parser.add_argument("--grid", type=float, default=0.22, help="Output resolution in degrees.")

    parser.add_argument(
        "--dataset",
        default="derived-era5-single-levels-daily-statistics",
        help="CDS dataset name.",
    )
    parser.add_argument(
        "--product-type",
        default="reanalysis",
        help="ERA5 product type.",
    )
    parser.add_argument(
        "--months-per-request",
        type=int,
        default=1,
        help="How many months to include in each CDS request.",
    )
    parser.add_argument(
        "--vars-per-request",
        type=int,
        default=2,
        help="How many ERA5 variables to include in each CDS request.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--dry-run", action="store_true", help="Print requests without downloading.")
    return parser.parse_args()


def _load_env(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_file.exists():
        return values

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        values[key] = value

    return values


def _make_client(env_values: dict[str, str]) -> cdsapi.Client:
    url = (
        env_values.get("CDSAPI_URL")
        or env_values.get("CDS_URL")
        or env_values.get("COPERNICUS_API_URL")
        or os.getenv("CDSAPI_URL")
        or os.getenv("CDS_URL")
        or os.getenv("COPERNICUS_API_URL")
    )
    key = (
        env_values.get("CDSAPI_KEY")
        or env_values.get("CDS_KEY")
        or env_values.get("COPERNICUS_API_KEY")
        or os.getenv("CDSAPI_KEY")
        or os.getenv("CDS_KEY")
        or os.getenv("COPERNICUS_API_KEY")
    )

    if url and key:
        return cdsapi.Client(url=url, key=key)
    return cdsapi.Client()


def _discover_cclm_vars(cclm_dir: Path) -> list[str]:
    vars_found: set[str] = set()
    for nc_file in sorted(cclm_dir.glob("*.nc")):
        token = nc_file.name.split("_", 1)[0]
        if token:
            vars_found.add(token)
    return sorted(vars_found)


def _era5_request_vars(cclm_vars: list[str]) -> tuple[list[str], list[str]]:
    era5_vars: set[str] = set()
    unsupported: list[str] = []

    for cclm_var in cclm_vars:
        mapped = CCLM_TO_ERA5.get(cclm_var)
        if mapped is None:
            unsupported.append(cclm_var)
            continue
        era5_vars.update(mapped)

    return sorted(era5_vars), sorted(unsupported)


def _request_payload(
    *,
    year: int,
    months: list[str],
    variables: list[str],
    daily_statistic: str,
    north: float,
    west: float,
    south: float,
    east: float,
    grid: float,
    product_type: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "product_type": product_type,
        "variable": variables,
        "year": str(year),
        "month": months,
        "area": [north, west, south, east],
        "grid": [grid, grid],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    # Parameters expected by the ERA5 daily-statistics dataset.
    if daily_statistic:
        payload["daily_statistic"] = daily_statistic
        payload["time_zone"] = "UTC+00:00"
        payload["frequency"] = "1_hourly"
        payload["day"] = [f"{d:02d}" for d in range(1, 32)]
    else:
        payload["day"] = [f"{d:02d}" for d in range(1, 32)]
        payload["time"] = [f"{h:02d}:00" for h in range(24)]

    return payload


def _chunked(seq: list[str], size: int) -> list[list[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _download_group(
    *,
    client: cdsapi.Client,
    dataset: str,
    years: list[int],
    variables: list[str],
    out_dir: Path,
    north: float,
    west: float,
    south: float,
    east: float,
    grid: float,
    product_type: str,
    overwrite: bool,
    dry_run: bool,
    stat_name: str,
    months_per_request: int,
    vars_per_request: int,
) -> None:
    if not variables:
        return

    print(f"\nRequest group: {stat_name}")
    print(f"Variables ({len(variables)}): {variables}")

    if months_per_request < 1:
        raise ValueError("months_per_request must be >= 1")
    if vars_per_request < 1:
        raise ValueError("vars_per_request must be >= 1")

    all_months = [f"{m:02d}" for m in range(1, 13)]
    month_chunks = _chunked(all_months, months_per_request)
    var_chunks = _chunked(variables, vars_per_request)
    total_requests = len(years) * len(month_chunks) * len(var_chunks)

    progress = tqdm(total=total_requests, desc=f"{stat_name}", unit="req", dynamic_ncols=True)
    for year in years:
        for month_chunk in month_chunks:
            for var_chunk in var_chunks:
                month_label = "-".join(month_chunk)
                var_label = "_".join(v.replace("_", "-") for v in var_chunk)
                out_file = out_dir / f"era5_reanalysis_{stat_name}_{year}_{month_label}_{var_label}.nc"
                progress.set_postfix_str(out_file.name)

                if out_file.exists() and not overwrite:
                    print(f"[skip] {out_file} already exists")
                    progress.update(1)
                    continue

                payload = _request_payload(
                    year=year,
                    months=month_chunk,
                    variables=var_chunk,
                    daily_statistic=stat_name,
                    north=north,
                    west=west,
                    south=south,
                    east=east,
                    grid=grid,
                    product_type=product_type,
                )

                print(f"[request] {dataset} year={year} month={month_label} vars={var_chunk} -> {out_file.name}")
                if not dry_run:
                    client.retrieve(dataset, payload, str(out_file))
                    print(f"[done] {out_file}")

                progress.update(1)


def main() -> None:
    args = _parse_args()

    if args.start_year > args.end_year:
        raise ValueError("start-year must be <= end-year")
    if args.south >= args.north:
        raise ValueError("south must be < north")
    if args.west >= args.east:
        raise ValueError("west must be < east")

    cclm_dir = Path(args.cclm_dir)
    out_dir = Path(args.out_dir)
    env_file = Path(args.env_file)

    if not cclm_dir.exists():
        raise FileNotFoundError(f"CCLM directory not found: {cclm_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cclm_vars = _discover_cclm_vars(cclm_dir)
    if not cclm_vars:
        raise FileNotFoundError(f"No CCLM .nc files found in: {cclm_dir}")

    era5_vars, unsupported = _era5_request_vars(cclm_vars)
    if not era5_vars:
        raise ValueError("No ERA5 variables could be mapped from discovered CCLM variables.")

    # Separate precipitation because it should be requested as daily_sum.
    precip_var = "total_precipitation"
    daily_sum_vars = [v for v in era5_vars if v == precip_var]
    daily_mean_vars = [v for v in era5_vars if v != precip_var]

    env_values = _load_env(env_file)
    client = _make_client(env_values)
    years = list(range(args.start_year, args.end_year + 1))

    print("CCLM variables discovered:", cclm_vars)
    if unsupported:
        print("CCLM variables without ERA5 mapping (skipped):", unsupported)
    print("ERA5 variables selected:", era5_vars)
    print(f"Years: {args.start_year}-{args.end_year}")
    print(
        "Area (N,W,S,E):",
        [args.north, args.west, args.south, args.east],
        "Grid:",
        [args.grid, args.grid],
    )

    _download_group(
        client=client,
        dataset=args.dataset,
        years=years,
        variables=daily_mean_vars,
        out_dir=out_dir,
        north=args.north,
        west=args.west,
        south=args.south,
        east=args.east,
        grid=args.grid,
        product_type=args.product_type,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        stat_name="daily_mean",
        months_per_request=args.months_per_request,
        vars_per_request=args.vars_per_request,
    )

    _download_group(
        client=client,
        dataset=args.dataset,
        years=years,
        variables=daily_sum_vars,
        out_dir=out_dir,
        north=args.north,
        west=args.west,
        south=args.south,
        east=args.east,
        grid=args.grid,
        product_type=args.product_type,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        stat_name="daily_sum",
        months_per_request=args.months_per_request,
        vars_per_request=args.vars_per_request,
    )

    print("\nAll requests processed.")


if __name__ == "__main__":
    main()

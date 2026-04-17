"""Verification and compatibility checking for climate datasets."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

WRF_TO_ERA5_VAR_MAP: dict[str, str] = {
    "tas": "t2m",
    "uas": "u10",
    "vas": "v10",
    "ps": "sp",
}


@dataclass
class DataQualityReport:
    """Report on data quality and compatibility checks."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    wrf_info: dict
    era5_info: dict
    alignment_info: dict


def check_dataset_integrity(ds: xr.Dataset, dataset_type: str = "dataset") -> tuple[bool, list[str]]:
    """Check basic integrity of a climate dataset.

    Args:
        ds: xarray Dataset to check
        dataset_type: 'WRF' or 'ERA5' for custom checks

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    ds_type = dataset_type.lower()

    # Check for required dimensions
    if "time" not in ds.dims:
        errors.append(f"Missing required dimension 'time' in {dataset_type}")

    if ds_type == "wrf":
        has_wrf_grid = ("rlat" in ds.dims and "rlon" in ds.dims)
        has_geo_grid = (
            ("lat" in ds.dims and "lon" in ds.dims)
            or ("latitude" in ds.dims and "longitude" in ds.dims)
            or ("lat" in ds.coords and "lon" in ds.coords)
            or ("latitude" in ds.coords and "longitude" in ds.coords)
        )
        if not (has_wrf_grid or has_geo_grid):
            errors.append(
                f"Missing spatial grid in {dataset_type}: expected rlat/rlon or lat/lon coordinates"
            )
    else:
        if "lat" not in ds.dims and "latitude" not in ds.dims:
            errors.append(f"Missing required dimension 'lat'/'latitude' in {dataset_type}")

        if "lon" not in ds.dims and "longitude" not in ds.dims:
            errors.append(f"Missing required dimension 'lon'/'longitude' in {dataset_type}")

    # Check for NaN or infinite values
    for var in ds.data_vars:
        data = ds[var].values
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()

        if nan_count > 0:
            nan_pct = 100 * nan_count / data.size
            if nan_pct > 50:
                errors.append(f"Variable '{var}' has {nan_pct:.1f}% NaN values (>50%)")
            else:
                logger.warning(f"Variable '{var}' has {nan_pct:.1f}% NaN values")

        if inf_count > 0:
            errors.append(f"Variable '{var}' contains infinite values ({inf_count} instances)")

    return len(errors) == 0, errors


def check_time_alignment(wrf_ds: xr.Dataset, era5_ds: xr.Dataset) -> tuple[bool, list[str], pd.DatetimeIndex]:
    """Check if WRF and ERA5 datasets have compatible time dimensions.

    Args:
        wrf_ds: WRF xarray Dataset
        era5_ds: ERA5 xarray Dataset

    Returns:
        Tuple of (is_compatible, error_messages, common_times)
    """
    errors = []

    wrf_time = pd.DatetimeIndex(wrf_ds["time"].values)
    era5_time = pd.DatetimeIndex(era5_ds["time"].values)

    if len(wrf_time) == 0:
        errors.append("WRF dataset has no time steps")

    if len(era5_time) == 0:
        errors.append("ERA5 dataset has no time steps")

    if errors:
        return False, errors, pd.DatetimeIndex([])

    wrf_years = set(wrf_time.year)
    era5_years = set(era5_time.year)
    common_years = wrf_years.intersection(era5_years)

    if not common_years:
        errors.append(
            f"No year overlap: WRF years {sorted(wrf_years)}, ERA5 years {sorted(era5_years)}"
        )
        return False, errors, pd.DatetimeIndex([])

    if len(common_years) < len(wrf_years) or len(common_years) < len(era5_years):
        missing_wrf = era5_years - wrf_years
        missing_era5 = wrf_years - era5_years
        if missing_wrf:
            logger.warning(f"ERA5 has years not in WRF: {sorted(missing_era5)}")
        if missing_era5:
            logger.warning(f"WRF has years not in ERA5: {sorted(missing_wrf)}")

    common_times = wrf_time.intersection(era5_time)

    if len(common_times) == 0:
        errors.append(
            f"No common timestamps found. "
            f"WRF: {wrf_time[0]} to {wrf_time[-1]}, "
            f"ERA5: {era5_time[0]} to {era5_time[-1]}"
        )
    elif len(common_times) < 3:
        errors.append(f"Too few common timestamps ({len(common_times)}), need at least 3")

    return len(errors) == 0, errors, common_times


def check_spatial_alignment(wrf_ds: xr.Dataset, era5_ds: xr.Dataset) -> tuple[bool, list[str], list[str]]:
    """Check if WRF and ERA5 have compatible spatial grids.

    Args:
        wrf_ds: WRF xarray Dataset
        era5_ds: ERA5 xarray Dataset

    Returns:
        Tuple of (is_compatible, error_messages)
    """
    errors = []
    warnings = []

    # Get spatial dimensions
    wrf_lat_dim = "lat" if "lat" in wrf_ds.dims else "rlat"
    wrf_lon_dim = "lon" if "lon" in wrf_ds.dims else "rlon"
    era5_lat_dim = "lat" if "lat" in era5_ds.dims else "latitude"
    era5_lon_dim = "lon" if "lon" in era5_ds.dims else "longitude"

    wrf_nlat = wrf_ds.sizes.get(wrf_lat_dim)
    wrf_nlon = wrf_ds.sizes.get(wrf_lon_dim)
    era5_nlat = era5_ds.sizes.get(era5_lat_dim)
    era5_nlon = era5_ds.sizes.get(era5_lon_dim)

    if wrf_nlat is None or wrf_nlon is None:
        errors.append(f"Cannot determine WRF spatial dimensions")
    if era5_nlat is None or era5_nlon is None:
        errors.append(f"Cannot determine ERA5 spatial dimensions")

    if wrf_nlat and era5_nlat:
        if abs(wrf_nlat - era5_nlat) > 2:
            warnings.append(
                f"Different latitude grid sizes: WRF={wrf_nlat}, ERA5={era5_nlat}"
            )

    if wrf_nlon and era5_nlon:
        if abs(wrf_nlon - era5_nlon) > 2:
            warnings.append(
                f"Different longitude grid sizes: WRF={wrf_nlon}, ERA5={era5_nlon}"
            )

    # Check coordinate bounds
    if "lat" in wrf_ds.coords and "lon" in wrf_ds.coords:
        wrf_lat = wrf_ds["lat"].values
        wrf_lon = wrf_ds["lon"].values
    elif "latitude" in wrf_ds.coords and "longitude" in wrf_ds.coords:
        wrf_lat = wrf_ds["latitude"].values
        wrf_lon = wrf_ds["longitude"].values
    else:
        wrf_lat = wrf_ds[wrf_lat_dim].values
        wrf_lon = wrf_ds[wrf_lon_dim].values

    era5_lat = era5_ds[era5_lat_dim].values
    era5_lon = era5_ds[era5_lon_dim].values

    wrf_lat_range = (np.nanmin(wrf_lat), np.nanmax(wrf_lat))
    wrf_lon_range = (np.nanmin(wrf_lon), np.nanmax(wrf_lon))
    era5_lat_range = (np.nanmin(era5_lat), np.nanmax(era5_lat))
    era5_lon_range = (np.nanmin(era5_lon), np.nanmax(era5_lon))

    # WRF should be within or close to ERA5 spatial domain
    if wrf_lat_range[0] < era5_lat_range[0] - 1 or wrf_lat_range[1] > era5_lat_range[1] + 1:
        warnings.append(
            f"WRF latitude range {wrf_lat_range} extends beyond ERA5 range {era5_lat_range}"
        )

    if wrf_lon_range[0] < era5_lon_range[0] - 1 or wrf_lon_range[1] > era5_lon_range[1] + 1:
        warnings.append(
            f"WRF longitude range {wrf_lon_range} extends beyond ERA5 range {era5_lon_range}"
        )

    return len(errors) == 0, errors, warnings


def check_variables(
    wrf_ds: xr.Dataset,
    era5_ds: xr.Dataset,
    required_variables: list[str] | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Check if required variables are present in both datasets.

    Args:
        wrf_ds: WRF xarray Dataset
        era5_ds: ERA5 xarray Dataset
        required_variables: Variables that must be in both datasets

    Returns:
        Tuple of (all_present, missing_vars, common_vars)
    """
    if required_variables is None:
        required_variables = ["t2m", "u10", "v10"]

    wrf_vars_raw = set(wrf_ds.data_vars)
    wrf_vars = {WRF_TO_ERA5_VAR_MAP.get(v, v) for v in wrf_vars_raw}
    era5_vars = set(era5_ds.data_vars)

    missing_wrf = [v for v in required_variables if v not in wrf_vars]
    missing_era5 = [v for v in required_variables if v not in era5_vars]

    errors = []
    if missing_wrf:
        errors.append(f"WRF missing variables: {missing_wrf}")
    if missing_era5:
        errors.append(f"ERA5 missing variables: {missing_era5}")

    common_vars = [v for v in required_variables if v in wrf_vars and v in era5_vars]

    return len(errors) == 0, errors, common_vars


def verify_wrf_era5_pair(
    wrf_path: str | Path,
    era5_path: str | Path,
    required_variables: list[str] | None = None,
) -> DataQualityReport:
    """Comprehensive verification of WRF and ERA5 dataset pair.

    Args:
        wrf_path: Path to WRF netCDF file
        era5_path: Path to ERA5 netCDF file
        required_variables: List of required variables (default: ['t2m', 'u10', 'v10'])

    Returns:
        DataQualityReport with verification results
    """
    if required_variables is None:
        required_variables = ["t2m", "u10", "v10"]

    errors = []
    warnings = []
    wrf_info = {}
    era5_info = {}
    alignment_info = {}

    # Load datasets
    try:
        wrf_ds = xr.open_dataset(wrf_path)
        wrf_info["path"] = str(wrf_path)
        wrf_info["shape"] = dict(wrf_ds.sizes)
        wrf_info["variables"] = list(wrf_ds.data_vars)
        logger.info(f"Loaded WRF dataset: {wrf_info['shape']}")
    except Exception as e:
        errors.append(f"Failed to load WRF: {e}")
        return DataQualityReport(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            wrf_info=wrf_info,
            era5_info=era5_info,
            alignment_info=alignment_info,
        )

    try:
        era5_ds = xr.open_dataset(era5_path)
        era5_info["path"] = str(era5_path)
        era5_info["shape"] = dict(era5_ds.sizes)
        era5_info["variables"] = list(era5_ds.data_vars)
        logger.info(f"Loaded ERA5 dataset: {era5_info['shape']}")
    except Exception as e:
        errors.append(f"Failed to load ERA5: {e}")
        return DataQualityReport(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            wrf_info=wrf_info,
            era5_info=era5_info,
            alignment_info=alignment_info,
        )

    # Check integrity
    wrf_ok, wrf_errs = check_dataset_integrity(wrf_ds, "WRF")
    if not wrf_ok:
        errors.extend(wrf_errs)

    era5_ok, era5_errs = check_dataset_integrity(era5_ds, "ERA5")
    if not era5_ok:
        errors.extend(era5_errs)

    # Check time alignment
    time_ok, time_errs, common_times = check_time_alignment(wrf_ds, era5_ds)
    if not time_ok:
        errors.extend(time_errs)
    else:
        alignment_info["common_times"] = len(common_times)
        alignment_info["time_range"] = (str(common_times[0]), str(common_times[-1]))
        wrf_time = pd.DatetimeIndex(wrf_ds["time"].values)
        era5_time = pd.DatetimeIndex(era5_ds["time"].values)
        alignment_info["wrf_years"] = sorted(set(wrf_time.year))
        alignment_info["era5_years"] = sorted(set(era5_time.year))
        logger.info(f"Common timestamps: {len(common_times)}")

    # Check spatial alignment
    spatial_ok, spatial_errors, spatial_warnings = check_spatial_alignment(wrf_ds, era5_ds)
    if not spatial_ok:
        errors.extend(spatial_errors)
    warnings.extend(spatial_warnings)

    # Check variables
    vars_ok, var_errs, common_vars = check_variables(wrf_ds, era5_ds, required_variables)
    if not vars_ok:
        errors.extend(var_errs)
    alignment_info["common_variables"] = common_vars

    overall_valid = len(errors) == 0

    report = DataQualityReport(
        is_valid=overall_valid,
        errors=errors,
        warnings=warnings,
        wrf_info=wrf_info,
        era5_info=era5_info,
        alignment_info=alignment_info,
    )

    # Log the report
    _log_report(report)

    return report


def _log_report(report: DataQualityReport) -> None:
    """Log verification report to logger."""
    logger.info("=" * 70)
    logger.info("DATA VERIFICATION REPORT")
    logger.info("=" * 70)

    logger.info("\nWRF Dataset:")
    logger.info(f"  Path: {report.wrf_info.get('path', 'N/A')}")
    logger.info(f"  Shape: {report.wrf_info.get('shape', 'N/A')}")
    logger.info(f"  Variables: {report.wrf_info.get('variables', [])}")

    logger.info("\nERA5 Dataset:")
    logger.info(f"  Path: {report.era5_info.get('path', 'N/A')}")
    logger.info(f"  Shape: {report.era5_info.get('shape', 'N/A')}")
    logger.info(f"  Variables: {report.era5_info.get('variables', [])}")

    logger.info("\nAlignment Info:")
    if report.alignment_info:
        for key, value in report.alignment_info.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.info("  No alignment info available")

    if report.errors:
        logger.error("\nERRORS:")
        for err in report.errors:
            logger.error(f"  ✗ {err}")

    if report.warnings:
        logger.warning("\nWARNINGS:")
        for warn in report.warnings:
            logger.warning(f"  ⚠ {warn}")

    if report.is_valid:
        logger.info("\n✓ Data verification PASSED")
    else:
        logger.error("\n✗ Data verification FAILED")

    logger.info("=" * 70)

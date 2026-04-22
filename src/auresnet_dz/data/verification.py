"""Verification and compatibility checking for climate datasets."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

GFS_TO_ERA5_VAR_MAP: dict[str, str] = {
    "t2m": "t2m",
    "u10": "wind10",
    "v10": "wind10",
    "sp": "sp",
    "wind10": "wind10",
}


@dataclass
class DataQualityReport:
    """Report on data quality and compatibility checks."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    gfs_info: dict
    era5_info: dict
    alignment_info: dict


def check_dataset_integrity(ds: xr.Dataset, dataset_type: str = "dataset") -> tuple[bool, list[str]]:
    """Check basic integrity of a climate dataset.

    Args:
        ds: xarray Dataset to check
        dataset_type: 'GFS' or 'ERA5' for custom checks

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    ds_type = dataset_type.lower()

    # Check for required dimensions
    if "time" not in ds.dims:
        errors.append(f"Missing required dimension 'time' in {dataset_type}")

    if ds_type == "gfs":
        has_gfs_grid = ("lat" in ds.dims and "lon" in ds.dims)
        has_geo_grid = (
            ("lat" in ds.dims and "lon" in ds.dims)
            or ("latitude" in ds.dims and "longitude" in ds.dims)
            or ("lat" in ds.coords and "lon" in ds.coords)
            or ("latitude" in ds.coords and "longitude" in ds.coords)
        )
        if not (has_gfs_grid or has_geo_grid):
            errors.append(
                f"Missing spatial grid in {dataset_type}: expected lat/lon coordinates"
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


def check_time_alignment(gfs_ds: xr.Dataset, era5_ds: xr.Dataset) -> tuple[bool, list[str], pd.DatetimeIndex]:
    """Check if GFS and ERA5 datasets have compatible time dimensions.

    Args:
        gfs_ds: GFS xarray Dataset
        era5_ds: ERA5 xarray Dataset

    Returns:
        Tuple of (is_compatible, error_messages, common_times)
    """
    errors = []

    gfs_time = pd.DatetimeIndex(gfs_ds["time"].values)
    era5_time = pd.DatetimeIndex(era5_ds["time"].values)

    if len(gfs_time) == 0:
        errors.append("GFS dataset has no time steps")

    if len(era5_time) == 0:
        errors.append("ERA5 dataset has no time steps")

    if errors:
        return False, errors, pd.DatetimeIndex([])

    gfs_years = set(gfs_time.year)
    era5_years = set(era5_time.year)
    common_years = gfs_years.intersection(era5_years)

    if not common_years:
        errors.append(
            f"No year overlap: GFS years {sorted(gfs_years)}, ERA5 years {sorted(era5_years)}"
        )
        return False, errors, pd.DatetimeIndex([])

    if len(common_years) < len(gfs_years) or len(common_years) < len(era5_years):
        missing_gfs = era5_years - gfs_years
        missing_era5 = gfs_years - era5_years
        if missing_gfs:
            logger.warning(f"ERA5 has years not in GFS: {sorted(missing_gfs)}")
        if missing_era5:
            logger.warning(f"GFS has years not in ERA5: {sorted(missing_era5)}")

    common_times = gfs_time.intersection(era5_time)

    if len(common_times) == 0:
        errors.append(
            f"No common timestamps found. "
            f"GFS: {gfs_time[0]} to {gfs_time[-1]}, "
            f"ERA5: {era5_time[0]} to {era5_time[-1]}"
        )
    elif len(common_times) < 3:
        errors.append(f"Too few common timestamps ({len(common_times)}), need at least 3")

    return len(errors) == 0, errors, common_times


def check_spatial_alignment(gfs_ds: xr.Dataset, era5_ds: xr.Dataset) -> tuple[bool, list[str], list[str]]:
    """Check if GFS and ERA5 have compatible spatial grids.

    Args:
        gfs_ds: GFS xarray Dataset
        era5_ds: ERA5 xarray Dataset

    Returns:
        Tuple of (is_compatible, error_messages)
    """
    errors = []
    warnings = []

    # Get spatial dimensions
    gfs_lat_dim = "lat" if "lat" in gfs_ds.dims else "rlat"
    gfs_lon_dim = "lon" if "lon" in gfs_ds.dims else "rlon"
    era5_lat_dim = "lat" if "lat" in era5_ds.dims else "latitude"
    era5_lon_dim = "lon" if "lon" in era5_ds.dims else "longitude"

    gfs_nlat = gfs_ds.sizes.get(gfs_lat_dim)
    gfs_nlon = gfs_ds.sizes.get(gfs_lon_dim)
    era5_nlat = era5_ds.sizes.get(era5_lat_dim)
    era5_nlon = era5_ds.sizes.get(era5_lon_dim)

    if gfs_nlat is None or gfs_nlon is None:
        errors.append(f"Cannot determine GFS spatial dimensions")
    if era5_nlat is None or era5_nlon is None:
        errors.append(f"Cannot determine ERA5 spatial dimensions")

    if gfs_nlat and era5_nlat:
        if abs(gfs_nlat - era5_nlat) > 2:
            warnings.append(
                f"Different latitude grid sizes: GFS={gfs_nlat}, ERA5={era5_nlat}"
            )

    if gfs_nlon and era5_nlon:
        if abs(gfs_nlon - era5_nlon) > 2:
            warnings.append(
                f"Different longitude grid sizes: GFS={gfs_nlon}, ERA5={era5_nlon}"
            )

    # Check coordinate bounds
    if "lat" in gfs_ds.coords and "lon" in gfs_ds.coords:
        gfs_lat = gfs_ds["lat"].values
        gfs_lon = gfs_ds["lon"].values
    elif "latitude" in gfs_ds.coords and "longitude" in gfs_ds.coords:
        gfs_lat = gfs_ds["latitude"].values
        gfs_lon = gfs_ds["longitude"].values
    else:
        gfs_lat = gfs_ds[gfs_lat_dim].values
        gfs_lon = gfs_ds[gfs_lon_dim].values

    era5_lat = era5_ds[era5_lat_dim].values
    era5_lon = era5_ds[era5_lon_dim].values

    gfs_lat_range = (np.nanmin(gfs_lat), np.nanmax(gfs_lat))
    gfs_lon_range = (np.nanmin(gfs_lon), np.nanmax(gfs_lon))
    era5_lat_range = (np.nanmin(era5_lat), np.nanmax(era5_lat))
    era5_lon_range = (np.nanmin(era5_lon), np.nanmax(era5_lon))

    # GFS should be within or close to ERA5 spatial domain
    if gfs_lat_range[0] < era5_lat_range[0] - 1 or gfs_lat_range[1] > era5_lat_range[1] + 1:
        warnings.append(
            f"GFS latitude range {gfs_lat_range} extends beyond ERA5 range {era5_lat_range}"
        )

    if gfs_lon_range[0] < era5_lon_range[0] - 1 or gfs_lon_range[1] > era5_lon_range[1] + 1:
        warnings.append(
            f"GFS longitude range {gfs_lon_range} extends beyond ERA5 range {era5_lon_range}"
        )

    return len(errors) == 0, errors, warnings


def check_variables(
    gfs_ds: xr.Dataset,
    era5_ds: xr.Dataset,
    required_variables: list[str] | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Check if required variables are present in both datasets.

    Args:
        gfs_ds: GFS xarray Dataset
        era5_ds: ERA5 xarray Dataset
        required_variables: Variables that must be in both datasets

    Returns:
        Tuple of (all_present, missing_vars, common_vars)
    """
    if required_variables is None:
        required_variables = ["t2m", "wind10"]

    gfs_vars_raw = set(gfs_ds.data_vars)
    gfs_vars = {GFS_TO_ERA5_VAR_MAP.get(str(v), str(v)) for v in gfs_vars_raw}
    era5_vars = set(era5_ds.data_vars)

    missing_gfs = [v for v in required_variables if v not in gfs_vars]
    missing_era5 = [v for v in required_variables if v not in era5_vars]

    errors = []
    if missing_gfs:
        errors.append(f"GFS missing variables: {missing_gfs}")
    if missing_era5:
        errors.append(f"ERA5 missing variables: {missing_era5}")

    common_vars = [v for v in required_variables if v in gfs_vars and v in era5_vars]

    return len(errors) == 0, errors, common_vars


def verify_gfs_era5_pair(
    gfs_path: str | Path,
    era5_path: str | Path,
    required_variables: list[str] | None = None,
) -> DataQualityReport:
    """Comprehensive verification of GFS and ERA5 dataset pair.

    Args:
        gfs_path: Path to GFS netCDF file
        era5_path: Path to ERA5 netCDF file
        required_variables: List of required variables (default: ['t2m', 'u10', 'v10'])

    Returns:
        DataQualityReport with verification results
    """
    if required_variables is None:
        required_variables = ["t2m", "wind10"]

    errors = []
    warnings = []
    gfs_info = {}
    era5_info = {}
    alignment_info = {}

    # Load datasets
    try:
        gfs_ds = xr.open_dataset(gfs_path)
        gfs_info["path"] = str(gfs_path)
        gfs_info["shape"] = dict(gfs_ds.sizes)
        gfs_info["variables"] = list(gfs_ds.data_vars)
        logger.info(f"Loaded GFS dataset: {gfs_info['shape']}")
    except Exception as e:
        errors.append(f"Failed to load GFS: {e}")
        return DataQualityReport(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            gfs_info=gfs_info,
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
            gfs_info=gfs_info,
            era5_info=era5_info,
            alignment_info=alignment_info,
        )

    # Check integrity
    gfs_ok, gfs_errs = check_dataset_integrity(gfs_ds, "GFS")
    if not gfs_ok:
        errors.extend(gfs_errs)

    era5_ok, era5_errs = check_dataset_integrity(era5_ds, "ERA5")
    if not era5_ok:
        errors.extend(era5_errs)

    # Check time alignment
    time_ok, time_errs, common_times = check_time_alignment(gfs_ds, era5_ds)
    if not time_ok:
        errors.extend(time_errs)
    else:
        alignment_info["common_times"] = len(common_times)
        alignment_info["time_range"] = (str(common_times[0]), str(common_times[-1]))
        gfs_time = pd.DatetimeIndex(gfs_ds["time"].values)
        era5_time = pd.DatetimeIndex(era5_ds["time"].values)
        alignment_info["gfs_years"] = sorted(set(gfs_time.year))
        alignment_info["era5_years"] = sorted(set(era5_time.year))
        logger.info(f"Common timestamps: {len(common_times)}")

    # Check spatial alignment
    spatial_ok, spatial_errors, spatial_warnings = check_spatial_alignment(gfs_ds, era5_ds)
    if not spatial_ok:
        errors.extend(spatial_errors)
    warnings.extend(spatial_warnings)

    # Check variables
    vars_ok, var_errs, common_vars = check_variables(gfs_ds, era5_ds, required_variables)
    if not vars_ok:
        errors.extend(var_errs)
    alignment_info["common_variables"] = common_vars

    overall_valid = len(errors) == 0

    report = DataQualityReport(
        is_valid=overall_valid,
        errors=errors,
        warnings=warnings,
        gfs_info=gfs_info,
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

    logger.info("\nGFS Dataset:")
    logger.info(f"  Path: {report.gfs_info.get('path', 'N/A')}")
    logger.info(f"  Shape: {report.gfs_info.get('shape', 'N/A')}")
    logger.info(f"  Variables: {report.gfs_info.get('variables', [])}")

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



# AuresNet-DZ: AI-Enhanced Weather Downscaling for Algeria

A deep learning framework to bias-correct and downscale Global Forecast System (GFS) outputs (0.25°) to high-resolution ERA5-like accuracy (0.1°) specifically for the Aures mountain range in North-East Algeria.

## Core Objective

The model learns a mapping to correct systematic biases in GFS caused by complex orography:
$$ f(\text{GFS}) \approx \text{ERA5} $$

## Project Structure

- [src/auresnet_dz/](src/auresnet_dz/): Core Python package containing models, data modules, and training logic.
- [configs/](configs/): Hydra-based configuration system for models, datasets, and training loops.
- [scripts/](scripts/): Utility scripts for data preparation and analysis.
- [analysis_results/](analysis_results/): Generated metrics, error maps, and comparison plots.
- [checkpoints/](checkpoints/): Model weight files (`.ckpt`).

## Quick Start

### 1. Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 2. Data Preparation
Align GFS and ERA5 data to the Aures grid:
```powershell
python scripts/prepare_aures_data.py
```

### 3. Execution
- **Training**:
  ```powershell
  python -m auresnet_dz.train.train train.max_epochs=100
  ```
- **Analysis & Reporting**:
  Run the inference and generate the HTML dashboard:
  ```powershell
  python scripts/generate_analysis.py --date "2023-01-18" --ckpt checkpoints/last.ckpt
  ```
- **View Dashboard**: Open [analysis_report.html](analysis_report.html) in your browser.

## Performance Analysis Suite

The project includes a comprehensive analysis suite that generates:
- **Weather App Dashboard**: Real-time comparison of GFS vs AI vs Truth in `analysis_report.html`.
- **Improvement Metrics**: Automatically calculates % error reduction (RMSE, MAE, Bias).
- **Error Maps**: Visualizes exactly where the model improves over the baseline (e.g., in high-altitude zones).

## Stack
- **Framework**: PyTorch Lightning
- **Architecture**: SMP U-Net (ResNet backbone)
- **Data**: Xarray, Dask, NetCDF4
- **Config**: Hydra

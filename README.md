# AuresNet-DZ

Bias-correction / post-processing model for CCLM/CORDEX outputs over the Aures region (North-East Algeria), using ERA5 as target truth.

Core mapping objective:

$$
f(\text{CCLM}) \approx \text{ERA5}
$$

## Stack

- Data: `xarray`, `dask`, `netCDF4`, `h5netcdf`, `xesmf`
- ML: `PyTorch`, `PyTorch Lightning`, `segmentation-models-pytorch` (U-Net)
- Config: `Hydra` + YAML

## Runtime mode

- Local console only (no cloud/Colab workflow in this repository).

## Initial project layout

- [configs](configs)
- [src/auresnet_dz](src/auresnet_dz)
- [scripts](scripts)

## Quick start

1) Create and activate a virtual environment.
2) Install package in editable mode:

```bash
pip install -e .
```

3) Launch training with default configs:

```bash
python -m auresnet_dz.train.train
```

Or with the helper script:

```bash
bash scripts/run_local_gpu.sh
```

## Local RTX training (recommended)

1) Create and activate a local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install --upgrade pip
pip install -e .
```

3) Place datasets locally:

- `data/raw/cclm/*.nc`
- `data/raw/era5/*.nc`

4) Prepare train-ready aligned files (CCLM/CORDEX variable mapping + Aures grid alignment):

```bash
python scripts/prepare_aures_data.py --skip-era5-sp-download
```

If you have valid CDS credentials in `.env` and want to include `sp` as a 4th channel, run without `--skip-era5-sp-download`.

This writes:

- `data/processed/cclm_aures_ready.nc`
- `data/processed/era5_aures_ready.nc`

5) Run a smoke test:

```bash
bash scripts/run_local_gpu.sh train.max_epochs=1 data.batch_size=1
```

6) Run full local training:

```bash
bash scripts/run_local_gpu.sh train.max_epochs=120 data.batch_size=8
```

### Performance notes (RTX)

- Default training profile is `train: local_gpu` in [configs/config.yaml](configs/config.yaml).
- It enables mixed precision and local GPU optimizations (TF32 + cuDNN benchmark).
- If your GPU memory is tight, reduce batch size:

```bash
bash scripts/run_local_gpu.sh data.batch_size=4
```

- If memory allows, increase throughput:

```bash
bash scripts/run_local_gpu.sh data.batch_size=12 data.num_workers=8
```

### Important security note

- Keep credentials in `.env` only.
- `.env` is git-ignored.
- If a key was exposed previously, rotate it.

## Data conventions (current default)

- Inputs: CCLM variables in netCDF (`tas`, `sfcWind`, optional pressure)
- Targets: ERA5 variables in netCDF
- Canonical channels: `t2m`, `wind10` (optional `sp`)
- Spatial alignment: IDW regridding on the Aures ERA5 grid
- Temporal alignment: intersection on timestamps

## Notes

- Keep raw datasets out of git. Put them under local `data/raw/`.
- If you use CDS/Copernicus credentials, store them in `.env` (never commit secrets).

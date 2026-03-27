# AuresNet-DZ

Bias-correction / post-processing model for CORDEX-WRF outputs over the Aures region (North-East Algeria), using ERA5 as target truth.

Core mapping objective:

$$
f(\text{WRF}) \approx \text{ERA5}
$$

## Stack

- Data: `xarray`, `dask`, `netCDF4`, `h5netcdf`, `xesmf`
- ML: `PyTorch`, `PyTorch Lightning`, `segmentation-models-pytorch` (U-Net)
- Config: `Hydra` + YAML

## Initial project layout

- [configs](configs)
- [src/auresnet_dz](src/auresnet_dz)
- [scripts](scripts)
- [tests](tests)

## Quick start

1) Create and activate a virtual environment.
2) Install package in editable mode:

```bash
pip install -e .[dev]
```

3) Copy env template:

```bash
cp .env.example .env
```

4) Launch training with default configs:

```bash
python -m auresnet_dz.train.train
```

## Data conventions (current default)

- Inputs: WRF variables in netCDF
- Targets: ERA5 variables in netCDF
- Spatial alignment: regridding via `xesmf`
- Temporal alignment: intersection on timestamps

## Notes

- Keep raw datasets out of git. Put them under local `data/raw/`.
- If you later need cloud storage credentials (Copernicus/CDS, object store, etc.), add them to `.env` (never commit secrets).

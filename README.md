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

## Cloud full-scale run

Recommended for xesmf stability: conda/mamba environment using [environment.yml](environment.yml).

1) Create env on cloud instance:

```bash
conda env create -f environment.yml
conda activate auresnet-dz
pip install -e .
```

2) Create local secrets file (never commit):

```bash
cp .env.example .env
```

3) Place data on attached storage:

- `data/raw/wrf/*.nc`
- `data/raw/era5/*.nc`

4) Start full-scale training:

```bash
bash scripts/run_cloud_full.sh
```

5) Optional overrides at launch time:

```bash
bash scripts/run_cloud_full.sh train.max_epochs=200 data.batch_size=12
```

Cloud profiles are in:

- [configs/data/cloud.yaml](configs/data/cloud.yaml)
- [configs/train/cloud.yaml](configs/train/cloud.yaml)

### Important security note

- Keep credentials in `.env` only.
- `.env` is git-ignored.
- If a key was exposed previously, rotate it before any upload.

## Google Colab (VS Code extension) full-scale run

Use this when your runtime is Colab and data/checkpoints are stored in Google Drive.

1) In a Colab notebook cell, mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

2) Clone repo to Colab workspace:

```bash
%cd /content
!git clone <YOUR_GITHUB_REPO_URL> AuresNet-DZ
%cd /content/AuresNet-DZ
```

3) Install core training dependencies (Colab-safe):

```bash
!pip install -U pip
!pip install -e .
```

If you need regridding with `xesmf`, install optional extra separately (may require additional system setup depending on runtime):

```bash
!pip install -e .[regrid]
```

4) Create `.env` locally in Colab runtime:

```bash
!cp .env.example .env
```

5) Ensure your data exists in Drive paths:

- `/content/drive/MyDrive/AuresNet-DZ/data/raw/wrf/*.nc`
- `/content/drive/MyDrive/AuresNet-DZ/data/raw/era5/*.nc`

6) Start full training with Colab profile:

```bash
!bash scripts/run_colab_full.sh
```

7) Optional overrides:

```bash
!bash scripts/run_colab_full.sh train.max_epochs=150 data.batch_size=6
```

Colab profiles are in:

- [configs/data/colab.yaml](configs/data/colab.yaml)
- [configs/train/colab.yaml](configs/train/colab.yaml)

## Data conventions (current default)

- Inputs: WRF variables in netCDF
- Targets: ERA5 variables in netCDF
- Spatial alignment: regridding via `xesmf`
- Temporal alignment: intersection on timestamps

## Notes

- Keep raw datasets out of git. Put them under local `data/raw/`.
- If you later need cloud storage credentials (Copernicus/CDS, object store, etc.), add them to `.env` (never commit secrets).

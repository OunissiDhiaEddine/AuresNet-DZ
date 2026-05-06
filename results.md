# AURESNET-DZ TRAINING RESULTS

*Generated: 2026-05-06 22:31:04*

## Run Information
**Run directory**: `logs\auresnet_dz_gfs_baseline\version_0`

## Hyperparameters
- **learning_rate**: 0.0002
- **weight_decay**: 1e-05

## Saved Checkpoints
- `last.ckpt` (280.0 MB)
- `epoch=112-step=1921.ckpt` (280.0 MB)
- `epoch=110-step=1887.ckpt` (280.0 MB)

## Metrics Overview
**Scalar tags found**: 25

## Train Loss (Step)
- **Initial**: 0.7561
- **Final**: 0.1832
- **Change**: -75.77%
- **Steps**: 204

## Validation MAE
- **Initial**: 348.2122
- **Best**: 52.1258 (step 1886)
- **Final**: 53.6219
- **Improvement to best**: 85.03%

## Train MAE
- **Initial**: 385.5288
- **Final**: 58.2640


## Train MAE by Variable
- **sp**: 288.9228
- **t2m**: 1.4540
- **tp**: 5.681e-05
- **u10**: 0.4033
- **v10**: 0.5398

## Test Metrics
- **test_loss**: 0.2190
- **test_mae**: 49.3775
- **test_mae/sp**: 244.3676
- **test_mae/t2m**: 1.5574
- **test_mae/tp**: 4.379e-05
- **test_mae/u10**: 0.4671
- **test_mae/v10**: 0.4956


## Validation MAE by Variable
- **sp**: 265.7809
- **t2m**: 1.4622
- **tp**: 1.292e-05
- **u10**: 0.3865
- **v10**: 0.4800


## Test MAE by Variable
- **sp**: 244.3676
- **t2m**: 1.5574
- **tp**: 4.379e-05
- **u10**: 0.4671
- **v10**: 0.4956

## Baseline Skill vs Raw GFS (Test Split)
- **sp**: model=244.3676 | raw_gfs=514.9163 | improvement=52.54%
- **t2m**: model=1.5574 | raw_gfs=2.0446 | improvement=23.83%
- **tp**: model=4.379e-05 | raw_gfs=2.818e-04 | improvement=84.46%
- **u10**: model=0.4671 | raw_gfs=0.8041 | improvement=41.90%
- **v10**: model=0.4956 | raw_gfs=0.7611 | improvement=34.89%

## Generalization Snapshot
- **Final train_mae**: 58.2640
- **Final val_mae**: 53.6219
- **Gap (val-train)**: -4.6420

## Latest Checkpoint Summary
- **Epoch**: 112
- **Global step**: 1921

## Available Scalar Tags
```
epoch, hp_metric, lr-AdamW, test_loss, test_mae, test_mae/sp, test_mae/t2m, test_mae/tp, test_mae/u10, test_mae/v10, train_loss, train_loss_step, train_mae, train_mae/sp, train_mae/t2m, train_mae/tp, train_mae/u10, train_mae/v10, val_loss, val_mae, val_mae/sp, val_mae/t2m, val_mae/tp, val_mae/u10, val_mae/v10
```
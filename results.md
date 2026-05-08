# AURESNET-DZ TRAINING RESULTS

*Generated: 2026-05-08 11:06:48*

## Run Information
**Run directory**: `logs\auresnet_dz_optimized_v4\version_0`

## Hyperparameters
- **learning_rate**: 0.0002
- **weight_decay**: 1e-05

## Saved Checkpoints
- `last.ckpt` (561.2 MB)
- `epoch=99-step=1700.ckpt` (561.2 MB)
- `epoch=97-step=1666.ckpt` (561.2 MB)

## Metrics Overview
**Scalar tags found**: 25

## Train Loss (Step)
- **Initial**: 0.6435
- **Final**: 0.0515
- **Change**: -91.99%
- **Steps**: 170

## Validation MAE
- **Initial**: 390.3716
- **Best**: 55.9449 (step 1665)
- **Final**: 56.0536
- **Improvement to best**: 85.67%

## Train MAE
- **Initial**: 402.1576
- **Final**: 47.0602


## Train MAE by Variable
- **sp**: 234.2359
- **t2m**: 0.6688
- **tp**: 2.314e-05
- **u10**: 0.1766
- **v10**: 0.2198

## Test Metrics
- **test_loss**: 0.1527
- **test_mae**: 57.5583
- **test_mae/sp**: 285.7465
- **test_mae/t2m**: 1.1973
- **test_mae/tp**: 3.420e-05
- **test_mae/u10**: 0.4362
- **test_mae/v10**: 0.4117


## Validation MAE by Variable
- **sp**: 278.2048
- **t2m**: 1.2866
- **tp**: 1.747e-05
- **u10**: 0.3635
- **v10**: 0.4130


## Test MAE by Variable
- **sp**: 285.7465
- **t2m**: 1.1973
- **tp**: 3.420e-05
- **u10**: 0.4362
- **v10**: 0.4117

## Baseline Skill vs Raw GFS (Test Split)
- **sp**: model=285.7465 | raw_gfs=514.9163 | improvement=44.51%
- **t2m**: model=1.1973 | raw_gfs=2.0446 | improvement=41.44%
- **tp**: model=3.420e-05 | raw_gfs=2.818e-04 | improvement=87.87%
- **u10**: model=0.4362 | raw_gfs=0.8041 | improvement=45.75%
- **v10**: model=0.4117 | raw_gfs=0.7611 | improvement=45.91%

## Generalization Snapshot
- **Final train_mae**: 47.0602
- **Final val_mae**: 56.0536
- **Gap (val-train)**: 8.9934

## Latest Checkpoint Summary
- **Epoch**: 99
- **Global step**: 1700

## Available Scalar Tags
```
epoch, hp_metric, lr-AdamW, test_loss, test_mae, test_mae/sp, test_mae/t2m, test_mae/tp, test_mae/u10, test_mae/v10, train_loss, train_loss_step, train_mae, train_mae/sp, train_mae/t2m, train_mae/tp, train_mae/u10, train_mae/v10, val_loss, val_mae, val_mae/sp, val_mae/t2m, val_mae/tp, val_mae/u10, val_mae/v10
```
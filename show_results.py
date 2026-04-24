#!/usr/bin/env python3
"""Display dynamic training results from the latest AuresNet-DZ run artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show dynamic training results from logs and checkpoints.")
    parser.add_argument("--logs-root", default="logs", help="Root directory containing TensorBoard logs.")
    parser.add_argument("--experiment", default="", help="Optional experiment name under logs root.")
    parser.add_argument("--checkpoints", default="checkpoints", help="Checkpoint directory.")
    return parser.parse_args()


def _find_latest_logs_dir(logs_root: Path, experiment: str) -> Path | None:
    if not logs_root.exists():
        return None

    if experiment:
        base = logs_root / experiment
        if not base.exists():
            return None
        versions = [p for p in base.glob("version_*") if p.is_dir()]
        if versions:
            return max(versions, key=lambda p: p.stat().st_mtime)
        return base

    candidates: list[Path] = []
    for exp_dir in logs_root.iterdir():
        if not exp_dir.is_dir():
            continue
        versions = [p for p in exp_dir.glob("version_*") if p.is_dir()]
        if versions:
            candidates.extend(versions)
        else:
            candidates.append(exp_dir)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_hparams(logs_dir: Path) -> dict[str, Any]:
    hparams_path = logs_dir / "hparams.yaml"
    if not hparams_path.exists():
        return {}
    with hparams_path.open("r", encoding="utf-8") as f:
        try:
            loaded = yaml.safe_load(f) or {}
        except yaml.constructor.ConstructorError:
            f.seek(0)
            loaded = yaml.unsafe_load(f) or {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def _load_scalars(logs_dir: Path) -> dict[str, list[tuple[int, float]]]:
    metrics: dict[str, list[tuple[int, float]]] = defaultdict(list)
    event_files = sorted(logs_dir.glob("events.out.tfevents.*"))

    for event_file in event_files:
        try:
            accumulator = EventAccumulator(str(event_file))
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                for event in accumulator.Scalars(tag):
                    metrics[tag].append((int(event.step), float(event.value)))
        except Exception:
            continue

    deduped: dict[str, list[tuple[int, float]]] = {}
    for tag, points in metrics.items():
        by_step: dict[int, float] = {}
        for step, value in points:
            by_step[step] = value
        deduped[tag] = sorted(by_step.items(), key=lambda x: x[0])

    return deduped


def _format_float(value: float | None, ndigits: int = 4) -> str:
    if value is None:
        return "N/A"
    abs_value = abs(value)
    if abs_value != 0.0 and (abs_value < 10 ** (-(ndigits - 1)) or abs_value >= 10**4):
        return f"{value:.3e}"
    return f"{value:.{ndigits}f}"


def _series_values(metrics: dict[str, list[tuple[int, float]]], tag: str) -> tuple[list[int], list[float]]:
    points = metrics.get(tag, [])
    if not points:
        return [], []
    steps = [s for s, _ in points]
    values = [v for _, v in points]
    return steps, values


def _print_metric_group(metrics: dict[str, list[tuple[int, float]]], prefix: str, title: str) -> None:
    tags = sorted(tag for tag in metrics if tag.startswith(prefix))
    if not tags:
        return

    print(f"\n{title}:")
    for tag in tags:
        _, values = _series_values(metrics, tag)
        if values:
            suffix = tag.split("/", 1)[1] if "/" in tag else tag
            print(f"   {suffix}: {_format_float(values[-1])}")


def _extract_checkpoint_summary(checkpoint_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    checkpoints = sorted(
        [p for p in checkpoint_dir.glob("*.ckpt") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    summary: dict[str, Any] = {}
    if not checkpoints:
        return checkpoints, summary

    try:
        import torch
    except Exception:
        return checkpoints, summary

    latest = checkpoints[0]
    try:
        ckpt = torch.load(latest, map_location="cpu")
        summary["epoch"] = ckpt.get("epoch")
        summary["global_step"] = ckpt.get("global_step")
        callback_metrics = ckpt.get("callback_metrics")
        if isinstance(callback_metrics, dict):
            for key, value in callback_metrics.items():
                if hasattr(value, "item"):
                    summary[str(key)] = float(value.item())
                elif isinstance(value, (int, float)):
                    summary[str(key)] = float(value)
    except Exception:
        pass
    return checkpoints, summary


def _safe_read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return loaded if isinstance(loaded, dict) else {}


def _compute_test_indices(n_samples: int, train_split: float, val_split: float, split_seed: int) -> list[int]:
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    g = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(n_samples, generator=g).tolist()
    return indices[n_train + n_val :]


def _baseline_mae_by_variable(
    gfs_path: Path,
    era5_path: Path,
    variables: list[str],
    test_indices: list[int],
) -> dict[str, float]:
    gfs = xr.open_dataset(gfs_path)
    era5 = xr.open_dataset(era5_path)
    baseline: dict[str, float] = {}
    for variable_name in variables:
        if variable_name not in gfs.data_vars or variable_name not in era5.data_vars:
            continue
        a = gfs[variable_name].isel(time=test_indices).values
        b = era5[variable_name].isel(time=test_indices).values
        mask = np.isfinite(a) & np.isfinite(b)
        if not np.any(mask):
            continue
        baseline[variable_name] = float(np.abs(a[mask] - b[mask]).mean())
    return baseline


def main() -> None:
    args = _parse_args()
    logs_root = Path(args.logs_root)
    checkpoint_dir = Path(args.checkpoints)

    logs_dir = _find_latest_logs_dir(logs_root, args.experiment)

    print("\n" + "=" * 80)
    print(" " * 20 + "AURESNET-DZ TRAINING RESULTS")
    print("=" * 80)

    if logs_dir is None:
        print("\nNo training logs found.")
        print(f"Checked logs root: {logs_root.resolve()}")
        print("Run training first, then rerun this script.")
        print("=" * 80 + "\n")
        return

    print(f"\nRun directory: {logs_dir}")

    hparams = _load_hparams(logs_dir)
    if hparams:
        print("\nHYPERPARAMETERS:")
        for key in [
            "learning_rate",
            "weight_decay",
            "max_epochs",
            "early_stopping_patience",
            "batch_size",
            "accumulate_grad_batches",
        ]:
            if key in hparams:
                print(f"   {key}: {hparams[key]}")

    checkpoints, ckpt_summary = _extract_checkpoint_summary(checkpoint_dir)
    print("\nSAVED CHECKPOINTS:")
    if not checkpoints:
        print("   None found.")
    else:
        for ckpt in checkpoints[:5]:
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"   {ckpt.name} ({size_mb:.1f} MB)")
        if len(checkpoints) > 5:
            print(f"   ... and {len(checkpoints) - 5} more")

    metrics = _load_scalars(logs_dir)
    if not metrics:
        print("\nNo TensorBoard scalar metrics found in this run.")
        print("=" * 80 + "\n")
        return

    print("\nMETRICS OVERVIEW:")
    print(f"   Scalar tags found: {len(metrics)}")

    train_steps, train_loss_step = _series_values(metrics, "train_loss_step")
    _, train_loss_epoch = _series_values(metrics, "train_loss_epoch")
    _, train_mae = _series_values(metrics, "train_mae")
    val_steps, val_mae = _series_values(metrics, "val_mae")

    if train_loss_step:
        initial = train_loss_step[0]
        final = train_loss_step[-1]
        change_pct = ((final - initial) / initial * 100.0) if initial != 0 else 0.0
        print("\nTRAIN LOSS (STEP):")
        print(f"   Initial: {_format_float(initial)}")
        print(f"   Final:   {_format_float(final)}")
        print(f"   Change:  {change_pct:.2f}%")
        print(f"   Steps:   {len(train_steps)}")

    if train_loss_epoch:
        print("\nTRAIN LOSS (EPOCH):")
        print(f"   Epochs logged: {len(train_loss_epoch)}")
        print(f"   Initial:       {_format_float(train_loss_epoch[0])}")
        print(f"   Final:         {_format_float(train_loss_epoch[-1])}")

    if val_mae:
        best_val = min(val_mae)
        best_idx = val_mae.index(best_val)
        best_step = val_steps[best_idx]
        print("\nVALIDATION MAE:")
        print(f"   Initial: {_format_float(val_mae[0])}")
        print(f"   Best:    {_format_float(best_val)} (step {best_step})")
        print(f"   Final:   {_format_float(val_mae[-1])}")
        if val_mae[0] != 0:
            improvement_pct = (val_mae[0] - best_val) / val_mae[0] * 100.0
            print(f"   Improvement to best: {improvement_pct:.2f}%")

    if train_mae:
        print("\nTRAIN MAE:")
        print(f"   Initial: {_format_float(train_mae[0])}")
        print(f"   Final:   {_format_float(train_mae[-1])}")

    _print_metric_group(metrics, "train_mae/", "TRAIN MAE BY VARIABLE")

    test_tags = sorted([tag for tag in metrics if tag.startswith("test_")])
    print("\nTEST METRICS:")
    if not test_tags:
        print("   No test_* metrics were found in TensorBoard logs.")
    else:
        for tag in test_tags:
            _, values = _series_values(metrics, tag)
            if values:
                print(f"   {tag}: {_format_float(values[-1])}")

    _print_metric_group(metrics, "val_mae/", "VALIDATION MAE BY VARIABLE")
    _print_metric_group(metrics, "test_mae/", "TEST MAE BY VARIABLE")

    data_cfg = _safe_read_yaml(Path("configs/data/default.yaml"))
    gfs_path = Path(str(data_cfg.get("raw_gfs_glob", "data/processed/gfs_aures_ready.nc")))
    era5_path = Path(str(data_cfg.get("raw_era5_glob", "data/processed/era5_aures_ready.nc")))
    train_split = float(data_cfg.get("train_split", 0.8))
    val_split = float(data_cfg.get("val_split", 0.1))
    split_seed = int(data_cfg.get("split_seed", 42))

    test_mae_tags = sorted(tag for tag in metrics if tag.startswith("test_mae/"))
    test_variables = [tag.split("/", 1)[1] for tag in test_mae_tags if "/" in tag]

    if gfs_path.exists() and era5_path.exists() and test_variables:
        try:
            gfs_ds = xr.open_dataset(gfs_path)
            n_samples = int(gfs_ds.sizes.get("time", 0))
            if n_samples > 0:
                test_indices = _compute_test_indices(n_samples, train_split, val_split, split_seed)
                baseline = _baseline_mae_by_variable(gfs_path, era5_path, test_variables, test_indices)
                if baseline:
                    print("\nBASELINE SKILL VS RAW GFS (TEST SPLIT):")
                    for variable_name in test_variables:
                        model_points = metrics.get(f"test_mae/{variable_name}", [])
                        if not model_points or variable_name not in baseline:
                            continue
                        model_mae = model_points[-1][1]
                        raw_mae = baseline[variable_name]
                        if raw_mae == 0:
                            skill_pct = None
                        else:
                            skill_pct = 100.0 * (raw_mae - model_mae) / raw_mae
                        print(
                            f"   {variable_name}: model={_format_float(model_mae)} | "
                            f"raw_gfs={_format_float(raw_mae)} | "
                            f"improvement={_format_float(skill_pct, ndigits=2)}%"
                        )
        except Exception:
            pass

    if train_mae and val_mae:
        gap = val_mae[-1] - train_mae[-1]
        print("\nGENERALIZATION SNAPSHOT:")
        print(f"   Final train_mae: {_format_float(train_mae[-1])}")
        print(f"   Final val_mae:   {_format_float(val_mae[-1])}")
        print(f"   Gap (val-train): {_format_float(gap)}")

    if ckpt_summary:
        print("\nLATEST CHECKPOINT SUMMARY:")
        if "epoch" in ckpt_summary:
            print(f"   Epoch:       {ckpt_summary['epoch']}")
        if "global_step" in ckpt_summary:
            print(f"   Global step: {ckpt_summary['global_step']}")
        if "val_mae" in ckpt_summary:
            print(f"   val_mae:     {_format_float(float(ckpt_summary['val_mae']))}")

    print("\nAvailable scalar tags:")
    print("   " + ", ".join(sorted(metrics.keys())))
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

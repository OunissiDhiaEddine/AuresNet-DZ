#!/usr/bin/env python3
"""Display dynamic training results from the latest AuresNet-DZ run artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

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
        loaded = yaml.safe_load(f) or {}
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
    return f"{value:.{ndigits}f}"


def _series_values(metrics: dict[str, list[tuple[int, float]]], tag: str) -> tuple[list[int], list[float]]:
    points = metrics.get(tag, [])
    if not points:
        return [], []
    steps = [s for s, _ in points]
    values = [v for _, v in points]
    return steps, values


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

    test_tags = sorted([tag for tag in metrics if tag.startswith("test_")])
    print("\nTEST METRICS:")
    if not test_tags:
        print("   No test_* metrics were found in TensorBoard logs.")
    else:
        for tag in test_tags:
            _, values = _series_values(metrics, tag)
            if values:
                print(f"   {tag}: {_format_float(values[-1])}")

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

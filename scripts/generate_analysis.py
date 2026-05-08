import os
import sys

# Add src to path if not already there
sys.path.append(os.path.join(os.getcwd(), "src"))

import torch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

from auresnet_dz.train.lightning_module import GfsToEra5LightningModule
from auresnet_dz.models.unet_smp import build_unet
from auresnet_dz.data.datamodule import ChannelNormalizationStats

def load_model(checkpoint_path, hparams_path):
    # Load hparams to get stats
    import yaml
    with open(hparams_path, 'r') as f:
        try:
            hparams = yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            f.seek(0)
            hparams = yaml.unsafe_load(f)
    
    # Reconstruct the model structure
    encoder_name = hparams.get('encoder_name', 'resnet50')
    architecture = hparams.get('architecture', 'unetplusplus')
    in_channels = 5 # t2m, u10, v10, sp, tp
    out_channels = 5
    
    base_model = build_unet(
        encoder_name=encoder_name, 
        in_channels=in_channels, 
        out_channels=out_channels,
        architecture=architecture
    )
    model = GfsToEra5LightningModule.load_from_checkpoint(
        checkpoint_path, 
        model=base_model,
        map_location=torch.device('cpu')
    )
    model.eval()

    # Extract normalization stats from hparams if they exist
    # If not in hparams as objects, we might need to recreate them from data
    input_stats = None
    if 'input_mean' in hparams and 'input_std' in hparams:
        input_stats = ChannelNormalizationStats(
            variable_names=["t2m", "u10", "v10", "sp", "tp"],
            mean=hparams['input_mean'],
            std=hparams['input_std']
        )
    elif '_target_mean' in dir(model):
        # We can try to infer input stats from target if they were identical, 
        # but better to calculate them from the train split of the data.
        pass

    return model, hparams, input_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Date to analyze (YYYY-MM-DD)", default="2024-01-14")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--hparams", type=str, default="logs/auresnet_dz_optimized_v4/version_0/hparams.yaml")
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    gfs_ds = xr.open_dataset("data/processed/gfs_aures_ready.nc")
    era5_ds = xr.open_dataset("data/processed/era5_aures_ready.nc")
    
    target_date = np.datetime64(args.date)
    try:
        gfs_step = gfs_ds.sel(time=target_date, method='nearest')
        era5_step = era5_ds.sel(time=target_date, method='nearest')
    except Exception as e:
        print(f"Error selecting date {args.date}: {e}")
        return

    # 2. Load Model
    model, hparams, _ = load_model(args.ckpt, args.hparams)

    # Use exact same normalization stats as training
    vars = ["t2m", "u10", "v10", "sp", "tp"]
    
    # Checkpoint values for TARGET (ERA5)
    target_mean = model._target_mean.clone()
    target_std = model._target_std.clone()
    
    # Calculate GFS stats from the FULL dataset (as done in datamodule.py)
    # The datamodule usually calculates input_stats from the 'gfs' dataset
    print("Calculating Input (GFS) stats from dataset to match DataModule logic...")
    input_stats = ChannelNormalizationStats.from_dataset(gfs_ds, vars, time_dim="time")

    print(f"Using Calculated Input (GFS) Stats:")
    print(f"  Means: {input_stats.mean.numpy()}")
    print(f"  Stds:  {input_stats.std.numpy()}")
    
    print(f"Using Checkpoint Target (ERA5) Stats for Denormalization:")
    print(f"  Means: {target_mean.numpy()}")
    print(f"  Stds:  {target_std.numpy()}")
    
    # 3. Prepare Input
    gfs_data = np.stack([gfs_step[v].values for v in vars])
    gfs_tensor = torch.from_numpy(gfs_data).float() # (C, H, W)
    
    # Normalize input using GFS stats
    gfs_tensor_norm = input_stats.normalize(gfs_tensor, channel_dim=0).unsqueeze(0) # (1, C, H, W)

    with torch.no_grad():
        pred_tensor_norm = model(gfs_tensor_norm)
        # Denormalize using TARGET (ERA5) stats
        pred_tensor = model._denormalize_targets(pred_tensor_norm)
        pred_data = pred_tensor.squeeze(0).numpy()

    # Debug print to check scales
    print("\nValue Range Check (Target Date):")
    for i, var in enumerate(vars):
        print(f"{var}:")
        print(f"  GFS range:   {gfs_data[i].min():.6f} to {gfs_data[i].max():.6f}")
        print(f"  ERA5 range:  {era5_step[var].values.min():.6f} to {era5_step[var].values.max():.6f}")
        print(f"  Model range: {pred_data[i].min():.6f} to {pred_data[i].max():.6f}")
        if var == 'tp':
            print(f"  Normalized Pred TP: {pred_tensor_norm[0, i].min().item():.6f} to {pred_tensor_norm[0, i].max().item():.6f}")
    for i, var in enumerate(vars):
        plt.figure(figsize=(18, 5))
        
        # Determine color scale
        vmin = min(gfs_data[i].min(), era5_step[var].values.min(), pred_data[i].min())
        vmax = max(gfs_data[i].max(), era5_step[var].values.max(), pred_data[i].max())
        
        cmap = 'RdBu_r'
        if var == 'tp':
            cmap = 'Blues'
            vmin = 0 # Precip shouldn't be negative in plot
            vmax = max(vmax, 0.0001) # Ensure scale is visible even for small amounts
        
        # GFS
        plt.subplot(1, 4, 1)
        plt.imshow(gfs_data[i], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(f"GFS Baseline ({var})")
        plt.colorbar()

        # ERA5
        plt.subplot(1, 4, 2)
        plt.imshow(era5_step[var].values, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(f"ERA5 Truth ({var})")
        plt.colorbar()

        # Model
        plt.subplot(1, 4, 3)
        plt.imshow(pred_data[i], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(f"AuresNet Predicted ({var})")
        plt.colorbar()

        # Error Comparison
        plt.subplot(1, 4, 4)
        error_gfs = np.abs(gfs_data[i] - era5_step[var].values)
        error_model = np.abs(pred_data[i] - era5_step[var].values)
        gain = error_gfs - error_model
        plt.imshow(gain, cmap='RdYlGn')
        plt.title(f"Performance Gain (AI vs GFS)\nGreen = AI improves")
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/{var}_comparison.png")
        plt.close()

    # Scatter plot for first variable (t2m)
    plt.figure(figsize=(8, 8))
    plt.scatter(era5_step["t2m"].values.flatten(), gfs_step["t2m"].values.flatten(), alpha=0.5, label="GFS")
    plt.scatter(era5_step["t2m"].values.flatten(), pred_data[0].flatten(), alpha=0.5, label="AuresNet")
    plt.plot([era5_step["t2m"].min(), era5_step["t2m"].max()], [era5_step["t2m"].min(), era5_step["t2m"].max()], 'k--')
    plt.xlabel("ERA5 (Truth)")
    plt.ylabel("Predicted")
    plt.title("Scatter Plot: Real vs Predicted (t2m)")
    plt.legend()
    plt.savefig(f"{args.output_dir}/scatter_t2m.png")
    plt.close()

    print(f"Analysis complete. Results in {args.output_dir}/")

    # Final: Save metrics to a json to be loaded by HTML (or just print for now)
    import json
    results_metrics = {}
    forecast_details = {} # Weather app summary

    for i, var in enumerate(vars):
        y_true = era5_step[var].values
        y_gfs = gfs_data[i]
        y_pred = pred_data[i]
        
        from auresnet_dz.analysis_utils import calculate_metrics
        m_gfs = calculate_metrics(y_true, y_gfs)
        m_model = calculate_metrics(y_true, y_pred)
        
        # Calculate Improvement Percentage (Error Reduction)
        improvement = ((m_gfs["MAE"] - m_model["MAE"]) / m_gfs["MAE"] * 100) if m_gfs["MAE"] > 0 else 0
        
        results_metrics[var] = {
            "gfs": m_gfs,
            "model": m_model,
            "improvement_pct": float(improvement)
        }

        # Spatial average for "Forecast" display
        forecast_details[var] = {
            "gfs": float(np.mean(y_gfs)),
            "truth": float(np.mean(y_true)),
            "model": float(np.mean(y_pred))
        }

    # Derived values (Wind Speed)
    if 'u10' in vars and 'v10' in vars:
        ws_gfs = np.sqrt(gfs_data[vars.index('u10')]**2 + gfs_data[vars.index('v10')]**2)
        ws_true = np.sqrt(era5_step['u10'].values**2 + era5_step['v10'].values**2)
        ws_pred = np.sqrt(pred_data[vars.index('u10')]**2 + pred_data[vars.index('v10')]**2)
        forecast_details["wind_speed"] = {
            "gfs": float(np.mean(ws_gfs)),
            "truth": float(np.mean(ws_true)),
            "model": float(np.mean(ws_pred))
        }
    
    # Bundle everything
    final_output = {
        "metrics": results_metrics,
        "forecast": forecast_details,
        "date": args.date
    }
    
    # Use absolute path and explicit close for reliability
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    print(f"Saving metrics to {metrics_path}...")
    f = open(metrics_path, "w")
    json.dump(final_output, f, indent=4)
    f.close()
    print("Metrics saved successfully.")

if __name__ == "__main__":
    main()

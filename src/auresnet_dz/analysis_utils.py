import numpy as np
import xarray as xr

def calculate_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, and Bias using numpy to avoid sklearn dependency."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    diff = y_pred - y_true
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)
    
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Bias": float(bias)
    }

def get_error_map(y_true, y_pred):
    """Returns the absolute error map."""
    return np.abs(y_pred - y_true)

def get_performance_gain(base_error, model_error):
    """Where AI wins: positive values mean AI is better (lower error)."""
    return base_error - model_error

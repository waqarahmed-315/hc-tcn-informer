import numpy as np

def MAE(pred, true):
    return float(np.mean(np.abs(pred - true)))

def MSE(pred, true):
    return float(np.mean((pred - true) ** 2))

def RMSE(pred, true):
    return float(np.sqrt(MSE(pred, true)))

def STABILIZED_MAPE(pred_original, true_original, eps=1e-6):
    denom = np.maximum(np.abs(true_original), eps)
    return float(100.0 * np.mean(np.abs(pred_original - true_original) / denom))

def standardized_metrics(pred, true):
    return {"mse_std": MSE(pred, true), "mae_std": MAE(pred, true), "rmse_std": RMSE(pred, true)}

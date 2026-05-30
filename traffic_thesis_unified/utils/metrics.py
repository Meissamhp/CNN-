import math
import numpy as np
import torch


def calculate_metrics(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().reshape(-1)
    else:
        y_true = np.array(y_true).reshape(-1)

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy().reshape(-1)
    else:
        y_pred = np.array(y_pred).reshape(-1)

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    errors = y_true - y_pred

    mse = np.mean(errors ** 2)
    mae = np.mean(np.abs(errors))
    rmse = math.sqrt(mse)
    std_error = np.std(errors)

    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "STD_ERROR": float(std_error)
    }
# utils.py

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def regression_metrics(y_true, y_pred, tol: float = 0.10):
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    # Relative error, with small-value protection
    abs_true = np.abs(y_true)
    denom = np.maximum(abs_true, 1.0)  # avoid division by very small numbers
    rel_err = np.abs(y_true - y_pred) / denom
    acc_10 = np.mean(rel_err <= tol)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "acc_10": acc_10,
    }


def plot_variant_series(
    dates,
    true_series,
    pred_series=None,
    variant_name: str = "feature",
    show: bool = True,
):
    
    plt.figure(figsize=(10, 4))
    plt.plot(dates, true_series, label="True")
    if pred_series is not None:
        plt.plot(dates, pred_series, linestyle="--", label="Predicted")
    plt.title(f"{variant_name} over time")
    plt.xlabel("Time")
    plt.ylabel("Scaled value")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()

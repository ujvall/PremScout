"""Evaluation metrics and formatting helpers for the Transfer Value Predictor."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def format_eur(val_eur: float | int | None) -> str:
    """Format an amount in euros into human-readable €M or €K string."""
    if val_eur is None or np.isnan(val_eur):
        return "N/A"
    abs_v = abs(val_eur)
    sign = "-" if val_eur < 0 else ""
    if abs_v >= 1_000_000:
        return f"{sign}€{abs_v / 1_000_000:.2f}M"
    elif abs_v >= 1_000:
        return f"{sign}€{abs_v / 1_000:.1f}K"
    else:
        return f"{sign}€{abs_v:.0f}"


def calculate_metrics(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> dict:
    """Calculate MAE, RMSE, R2, and MAPE for transfer value regression.

    Args:
        y_true: Ground truth market values in EUR
        y_pred: Predicted market values in EUR

    Returns:
        dict of metric scores (both raw float and formatted strings).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    # Ensure no negative market values in predictions
    yp = np.clip(yp, a_min=10_000, a_max=None)

    mae = mean_absolute_error(yt, yp)
    mse = mean_squared_error(yt, yp)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(yt, yp)

    # Safe MAPE on non-zero targets
    valid_mask = yt >= 10_000
    if valid_mask.sum() > 0:
        mape = float(np.mean(np.abs((yt[valid_mask] - yp[valid_mask]) / yt[valid_mask])) * 100.0)
    else:
        mape = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "mae_formatted": format_eur(mae),
        "rmse_formatted": format_eur(rmse),
        "r2_formatted": f"{r2:.4f}",
        "mape_formatted": f"{mape:.2f}%" if not np.isnan(mape) else "N/A",
    }

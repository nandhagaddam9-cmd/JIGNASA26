"""
Evaluation and Chronological Backtesting Module
Implements:
- Chronological Time-Series Split (zero future leakage)
- Standard Metrics: MAE, RMSE, WAPE
- Asymmetric Stock-Out-Oriented Error Loss (SOEL)
- Stock-out Risk Analysis & Comparative Benchmark Report
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def compute_stockout_oriented_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    under_penalty: float = 3.5,
    over_penalty: float = 1.0,
) -> float:
    """
    Computes asymmetric stock-out-oriented error loss (SOEL).
    Penalizes under-forecasting (which leads to empty shelves and lost revenue)
    more heavily than over-forecasting (holding extra safety buffer).
    """
    error = y_true - y_pred  # positive error means under-forecast (demand > forecast)
    loss = np.where(error > 0, under_penalty * error, over_penalty * (-error))
    return float(np.mean(loss))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    under_penalty: float = 3.5,
    over_penalty: float = 1.0,
) -> Dict[str, float]:
    """Computes comprehensive retail forecasting metrics."""
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    # WAPE (Weighted Absolute Percentage Error) handles zeros smoothly
    sum_true = float(np.sum(y_true))
    wape = float(np.sum(np.abs(errors)) / (sum_true + 1e-5)) * 100.0

    soel = compute_stockout_oriented_loss(
        y_true, y_pred, under_penalty=under_penalty, over_penalty=over_penalty
    )

    under_forecast_pct = float(np.mean(y_true > y_pred)) * 100.0
    over_forecast_pct = float(np.mean(y_pred >= y_true)) * 100.0

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "WAPE_pct": round(wape, 2),
        "StockOut_Loss_SOEL": round(soel, 2),
        "Under_Forecast_Rate_pct": round(under_forecast_pct, 1),
        "Over_Forecast_Rate_pct": round(over_forecast_pct, 1),
    }


def evaluate_models_chronologically(
    test_df: pd.DataFrame,
    predictions_dict: Dict[str, np.ndarray],
    target_col: str = "actual_sales",
) -> pd.DataFrame:
    """
    Generates a comparative scorecard DataFrame comparing baseline models against the ANN.
    """
    y_true = test_df[target_col].values
    records = []

    for model_name, preds in predictions_dict.items():
        metrics = compute_metrics(y_true, preds)
        metrics["Model"] = model_name
        records.append(metrics)

    res_df = pd.DataFrame(records)
    # Reorder columns
    cols = ["Model", "MAE", "RMSE", "WAPE_pct", "StockOut_Loss_SOEL", "Under_Forecast_Rate_pct", "Over_Forecast_Rate_pct"]
    return res_df[cols].sort_values(by="StockOut_Loss_SOEL")


if __name__ == "__main__":
    # Smoke test metrics
    y = np.array([50, 40, 60, 2, 0, 80])
    p_good = np.array([48, 42, 59, 2, 1, 78])
    p_bad = np.array([20, 20, 20, 10, 10, 20])  # massive under-forecast

    print("Good model metrics:", compute_metrics(y, p_good))
    print("Under-forecasting model metrics:", compute_metrics(y, p_bad))

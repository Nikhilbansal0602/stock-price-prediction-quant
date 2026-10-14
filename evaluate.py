"""
Evaluation module for financial time series models.
Calculates statistical errors, price forecast metrics, and directional hit rates.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score
from config import REPORTS_DIR

def evaluate_predictions(
    results_df: pd.DataFrame,
    model_name: str = "Model",
    save_report: bool = True,
    reports_dir: Optional[Path] = REPORTS_DIR
) -> Dict[str, Any]:
    """
    Evaluates out-of-sample predictions on both return and price dimensions.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing 'Target_Return', 'Pred_Return',
                                   'Target_Next_Close', 'Pred_Next_Close', 'Target_Direction', 'Pred_Direction'.
        model_name (str): Name of the model being evaluated.
        save_report (bool): Whether to save a markdown/txt report.
        reports_dir (Path): Output directory for reports.

    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics.
    """
    y_true_ret = results_df['Target_Return'].values
    y_pred_ret = results_df['Pred_Return'].values

    y_true_price = results_df['Target_Next_Close'].values
    y_pred_price = results_df['Pred_Next_Close'].values

    y_true_dir = results_df['Target_Direction'].values
    y_pred_dir = results_df['Pred_Direction'].values

    # Return Metrics
    ret_rmse = np.sqrt(mean_squared_error(y_true_ret, y_pred_ret))
    ret_mae = mean_absolute_error(y_true_ret, y_pred_ret)
    corr = np.corrcoef(y_true_ret, y_pred_ret)[0, 1] if len(y_true_ret) > 1 else 0.0

    # Price Metrics
    price_rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    price_mae = mean_absolute_error(y_true_price, y_pred_price)
    price_mape = np.mean(np.abs((y_true_price - y_pred_price) / y_true_price)) * 100.0

    # Directional / Trading Metrics
    hit_rate = accuracy_score(y_true_dir, y_pred_dir) * 100.0
    precision_up = precision_score(y_true_dir, y_pred_dir, zero_division=0) * 100.0
    recall_up = recall_score(y_true_dir, y_pred_dir, zero_division=0) * 100.0

    # Naive baseline (predicting no change / market drift)
    naive_price_mae = mean_absolute_error(y_true_price, results_df['Close'].values)
    naive_hit_rate = max(np.mean(y_true_dir), 1 - np.mean(y_true_dir)) * 100.0

    metrics = {
        "model_name": model_name,
        "sample_size": len(results_df),
        "return_rmse": ret_rmse,
        "return_mae": ret_mae,
        "return_correlation": corr,
        "price_rmse": price_rmse,
        "price_mae": price_mae,
        "price_mape_pct": price_mape,
        "naive_price_mae": naive_price_mae,
        "directional_accuracy_pct": hit_rate,
        "naive_directional_accuracy_pct": naive_hit_rate,
        "precision_up_pct": precision_up,
        "recall_up_pct": recall_up
    }

    report_text = f"""======================================================
 OUT-OF-SAMPLE EVALUATION REPORT: {model_name}
======================================================
Test Sample Size: {metrics['sample_size']} trading days

--- Return Prediction Performance ---
* Log Return RMSE:          {ret_rmse:.6f}
* Log Return MAE:           {ret_mae:.6f}
* Predicted vs True Corr:   {corr:+.4f}

--- Price Forecast Performance ---
* Next-Day Price RMSE:      ${price_rmse:.2f}
* Next-Day Price MAE:       ${price_mae:.2f}
* Next-Day Price MAPE:      {price_mape:.2f}%
* Naive Persistence MAE:    ${naive_price_mae:.2f}

--- Directional Trading Metrics ---
* Directional Accuracy:     {hit_rate:.2f}% (Hit Rate)
* Majority Class Baseline:  {naive_hit_rate:.2f}%
* Up-Market Precision:      {precision_up:.2f}%
* Up-Market Recall:         {recall_up:.2f}%
======================================================
"""
    print(report_text)

    if save_report and reports_dir:
        report_file = reports_dir / f"{model_name}_evaluation.txt"
        with open(report_file, "w") as f:
            f.write(report_text)
        print(f"[Evaluation] Report saved to {report_file}")

    return metrics

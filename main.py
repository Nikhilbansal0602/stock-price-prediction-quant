"""
End-to-End Stock Price Prediction and Quantitative Trading Pipeline.
Orchestrates data ingestion, feature engineering, model training, evaluation, backtesting, and visualization.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

from config import DEFAULT_CONFIG, ProjectConfig, PLOTS_DIR, REPORTS_DIR, MODELS_DIR
from data_loader import fetch_stock_data
from feature_engineering import engineer_features
from model_pipeline import (
    split_time_series_data,
    train_and_evaluate_models,
    generate_test_predictions
)
from evaluate import evaluate_predictions
from backtester import run_backtest
from visualizer import (
    plot_indicators,
    plot_forecast,
    plot_backtest,
    plot_feature_importance
)

def run_pipeline(
    ticker: str = "AAPL",
    start_date: str = "2018-01-01",
    end_date: str = None,
    capital: float = 10000.0,
    force_reload: bool = False,
    generate_plots: bool = True
):
    print("=" * 65)
    print(f" END-TO-END STOCK PREDICTION & QUANT PIPELINE: {ticker} ")
    print("=" * 65)

    config = ProjectConfig(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_capital=capital
    )

    # -------------------------------------------------------------
    # Step 1: Data Ingestion
    # -------------------------------------------------------------
    print("\n[Step 1/6] Ingesting Historical Market Data...")
    raw_df = fetch_stock_data(
        ticker=config.ticker,
        start_date=config.start_date,
        end_date=config.end_date,
        force_reload=force_reload
    )

    # -------------------------------------------------------------
    # Step 2: Feature Engineering (No Look-Ahead Leakage)
    # -------------------------------------------------------------
    print("\n[Step 2/6] Engineering Stationary Technical Features...")
    processed_df, feature_cols = engineer_features(raw_df, config=config, include_target=True)
    print(f"--> Extracted {len(feature_cols)} quantitative features across {len(processed_df)} valid trading days.")

    # -------------------------------------------------------------
    # Step 3: Time Series Train / Validation / Test Splitting
    # -------------------------------------------------------------
    print("\n[Step 3/6] Chronological Time-Series Data Splitting...")
    split_data = split_time_series_data(processed_df, feature_cols, target_col='Target_Return', config=config)
    print(f"--> Train set:      {len(split_data['train_df'])} bars ({split_data['train_df'].index[0].date()} to {split_data['train_df'].index[-1].date()})")
    print(f"--> Validation set: {len(split_data['val_df'])} bars ({split_data['val_df'].index[0].date()} to {split_data['val_df'].index[-1].date()})")
    print(f"--> Test set (OOS): {len(split_data['test_df'])} bars ({split_data['test_df'].index[0].date()} to {split_data['test_df'].index[-1].date()})")

    # -------------------------------------------------------------
    # Step 4: Model Training & Validation
    # -------------------------------------------------------------
    print("\n[Step 4/6] Training & Selecting Candidate Models...")
    trained_models, best_model_name = train_and_evaluate_models(split_data, save_models=True)
    champion_model = trained_models[best_model_name]

    # Generate Out-Of-Sample Predictions
    results_df = generate_test_predictions(
        model=champion_model,
        scaler=split_data['scaler'],
        test_df=split_data['test_df'],
        feature_cols=feature_cols
    )

    # -------------------------------------------------------------
    # Step 5: Statistical & Price Forecast Evaluation
    # -------------------------------------------------------------
    print("\n[Step 5/6] Evaluating Out-of-Sample Performance...")
    eval_metrics = evaluate_predictions(results_df, model_name=best_model_name, save_report=True)

    # -------------------------------------------------------------
    # Step 6: Algorithmic Strategy Backtesting
    # -------------------------------------------------------------
    print("\n[Step 6/6] Running Realistic Strategy Backtest...")
    bt_df, bt_metrics = run_backtest(results_df, config=config, save_report=True)

    # -------------------------------------------------------------
    # Visualizations & Diagnostics
    # -------------------------------------------------------------
    if generate_plots:
        print("\n[Visualizer] Generating Visual Diagnostics...")
        clean_ticker = ticker.upper()
        plot_indicators(processed_df.iloc[-500:], clean_ticker, save_path=PLOTS_DIR / f"{clean_ticker}_indicators.png")
        plot_forecast(results_df, clean_ticker, best_model_name, save_path=PLOTS_DIR / f"{clean_ticker}_forecast.png")
        plot_backtest(bt_df, clean_ticker, save_path=PLOTS_DIR / f"{clean_ticker}_backtest.png")
        plot_feature_importance(champion_model, feature_cols, save_path=PLOTS_DIR / f"{clean_ticker}_features.png")

    print("\n" + "=" * 65)
    print(" PIPELINE EXECUTION COMPLETE ")
    print("=" * 65)
    print(f"Champion Model:       {best_model_name}")
    print(f"Price Forecast MAE:   ${eval_metrics['price_mae']:.2f} ({eval_metrics['price_mape_pct']:.2f}% MAPE)")
    print(f"Directional Hit Rate: {eval_metrics['directional_accuracy_pct']:.2f}% (Baseline: {eval_metrics['naive_directional_accuracy_pct']:.2f}%)")
    print(f"Strategy Return:      {bt_metrics['strategy_total_return_pct']:+.2f}% vs Benchmark {bt_metrics['benchmark_total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:         {bt_metrics['strategy_sharpe_ratio']:.2f} vs Benchmark {bt_metrics['benchmark_sharpe_ratio']:.2f}")
    print(f"Saved Models:         {MODELS_DIR}")
    print(f"Saved Reports:        {REPORTS_DIR}")
    print(f"Saved Plots:          {PLOTS_DIR}")
    print("=" * 65)

    return eval_metrics, bt_metrics

def main():
    parser = argparse.ArgumentParser(description="End-to-End Stock Price Prediction & Trading Strategy")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date YYYY-MM-DD (default: 2018-01-01)")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: latest)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial trading capital in USD (default: 10000.0)")
    parser.add_argument("--force-reload", action="store_true", help="Force re-download of data from Yahoo Finance")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")

    args = parser.parse_args()

    run_pipeline(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        capital=args.capital,
        force_reload=args.force_reload,
        generate_plots=not args.no_plots
    )

if __name__ == "__main__":
    main()

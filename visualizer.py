"""
Visualization module for generating publication-quality charts:
- Technical indicators dashboard
- Actual vs. Predicted stock prices
- Backtest equity curves & drawdowns
- Feature importance rankings
"""
from pathlib import Path
from typing import List, Optional, Any
import matplotlib
matplotlib.use('Agg')  # Headless backend to prevent GUI thread hanging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import PLOTS_DIR

# Clean, professional plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_indicators(df: pd.DataFrame, ticker: str, save_path: Optional[Path] = None):
    """Plots multi-panel OHLCV and technical indicators."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1.5, 1.5]})
    
    # 1. Price and Moving Averages / Bollinger Bands
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='Close Price', color='#1f77b4', linewidth=1.5)
    if 'Dist_SMA_10' in df.columns:
        ax1.plot(df.index, df['Close'] / (df['Dist_SMA_10'] + 1), label='SMA 10', color='#ff7f0e', alpha=0.7)
    if 'Dist_SMA_50' in df.columns:
        ax1.plot(df.index, df['Close'] / (df['Dist_SMA_50'] + 1), label='SMA 50', color='#2ca02c', alpha=0.7)
    ax1.set_title(f"{ticker} Technical Indicators Overview", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc='upper left')

    # 2. Volume
    ax2 = axes[1]
    ax2.bar(df.index, df['Volume'], color='#7f7f7f', alpha=0.5, label='Volume')
    ax2.set_ylabel("Volume")
    ax2.legend(loc='upper left')

    # 3. MACD
    ax3 = axes[2]
    if 'MACD_Norm' in df.columns and 'MACD_Signal_Norm' in df.columns:
        ax3.plot(df.index, df['MACD_Norm'] * 100, label='MACD %', color='#1f77b4', linewidth=1.2)
        ax3.plot(df.index, df['MACD_Signal_Norm'] * 100, label='Signal %', color='#ff7f0e', linewidth=1.2)
        ax3.bar(df.index, df['MACD_Hist_Norm'] * 100, label='Hist %', color='#2ca02c', alpha=0.4)
    ax3.set_ylabel("MACD (% of Price)")
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.6)
    ax3.legend(loc='upper left')

    # 4. RSI
    ax4 = axes[3]
    if 'RSI_14' in df.columns:
        ax4.plot(df.index, df['RSI_14'], label='RSI 14', color='#9467bd', linewidth=1.2)
        ax4.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax4.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax4.set_ylabel("RSI")
    ax4.set_ylim(10, 90)
    ax4.legend(loc='upper left')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
        print(f"[Visualizer] Technical chart saved to {save_path}")
    plt.close(fig)

def plot_forecast(
    results_df: pd.DataFrame,
    ticker: str,
    model_name: str,
    save_path: Optional[Path] = None
):
    """Plots actual next-day close prices vs model predicted next-day close prices."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Actual vs Predicted
    ax1.plot(results_df.index, results_df['Target_Next_Close'], label='Actual Next-Day Close', color='#1f77b4', linewidth=1.8)
    ax1.plot(results_df.index, results_df['Pred_Next_Close'], label=f'{model_name} Predicted Close', color='#e377c2', linestyle='--', linewidth=1.5)
    ax1.set_title(f"Out-of-Sample Price Forecast: {ticker} ({model_name})", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc='upper left')

    # Price Residuals
    residuals = results_df['Pred_Next_Close'] - results_df['Target_Next_Close']
    ax2.plot(results_df.index, residuals, color='#d62728', linewidth=1.0, label='Forecast Error ($)')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.7)
    ax2.set_ylabel("Error ($)")
    ax2.set_xlabel("Date")
    ax2.legend(loc='upper left')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
        print(f"[Visualizer] Forecast comparison chart saved to {save_path}")
    plt.close(fig)

def plot_backtest(
    bt_df: pd.DataFrame,
    ticker: str,
    save_path: Optional[Path] = None
):
    """Plots strategy equity curve, underwater drawdowns, and position exposure."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1.5, 0.8]})

    # 1. Equity Curve
    ax1.plot(bt_df.index, bt_df['Strategy_Equity'], label='ML Quant Strategy (Net of Costs)', color='#2ca02c', linewidth=2.0)
    ax1.plot(bt_df.index, bt_df['Benchmark_Equity'], label=f'{ticker} Buy & Hold Benchmark', color='#1f77b4', linewidth=1.5, linestyle=':')
    ax1.set_title(f"Out-of-Sample Backtest: Strategy Equity vs Benchmark ({ticker})", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc='upper left')

    # 2. Drawdowns
    ax2.plot(bt_df.index, bt_df['Strategy_Drawdown'] * 100, label='Strategy Drawdown %', color='#d62728', linewidth=1.2)
    ax2.plot(bt_df.index, bt_df['Benchmark_Drawdown'] * 100, label='Benchmark Drawdown %', color='#7f7f7f', linestyle=':', alpha=0.7)
    ax2.fill_between(bt_df.index, bt_df['Strategy_Drawdown'] * 100, 0, color='#d62728', alpha=0.15)
    ax2.set_ylabel("Drawdown %")
    ax2.legend(loc='lower left')

    # 3. Position Allocation
    ax3.step(bt_df.index, bt_df['Position'], where='post', color='#9467bd', label='Market Exposure (1=Long, 0=Cash)')
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0, 1])
    ax3.set_ylabel("Position")
    ax3.set_xlabel("Date")
    ax3.legend(loc='upper left')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
        print(f"[Visualizer] Backtest equity chart saved to {save_path}")
    plt.close(fig)

def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
    save_path: Optional[Path] = None
):
    """Extracts and plots top feature importances."""
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)

    if importances is None:
        return

    feat_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    top_feats = feat_series.tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_feats.plot(kind='barh', ax=ax, color='#1f77b4', edgecolor='black', alpha=0.8)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance / Absolute Coefficient")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
        print(f"[Visualizer] Feature importance chart saved to {save_path}")
    plt.close(fig)

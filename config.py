import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Base Project Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"

for d in [DATA_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

@dataclass
class ProjectConfig:
    # Asset Settings
    ticker: str = "AAPL"
    start_date: str = "2018-01-01"
    end_date: str = None  # None defaults to today's date

    # Feature Engineering Settings
    sma_windows: List[int] = field(default_factory=lambda: [10, 20, 50, 200])
    ema_spans: List[int] = field(default_factory=lambda: [12, 20, 26])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    lag_returns: List[int] = field(default_factory=lambda: [1, 2, 3, 5])

    # Dataset Splitting (Time Series Sequential Split)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Trading Strategy Backtesting Parameters
    initial_capital: float = 10000.0
    transaction_cost_pct: float = 0.001  # 0.10% per transaction (slippage + fee)
    long_threshold: float = 0.0005       # Minimum expected return to enter long position
    short_threshold: float = -0.0005     # Threshold for short (if enabled)
    allow_short: bool = False            # Default: Long-only or Cash
    risk_free_rate_annual: float = 0.04  # 4% annual risk-free rate for Sharpe ratio

    # Random Seed
    random_state: int = 42

DEFAULT_CONFIG = ProjectConfig()

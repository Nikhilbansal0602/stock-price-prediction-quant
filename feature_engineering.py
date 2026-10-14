"""
Feature engineering module for quantitative financial time series.
Generates stationary technical indicators and non-leaking forward targets.
"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from config import ProjectConfig, DEFAULT_CONFIG

def engineer_features(
    df: pd.DataFrame,
    config: ProjectConfig = DEFAULT_CONFIG,
    include_target: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Computes stationary technical indicators and targets without look-ahead bias.

    Parameters:
        df (pd.DataFrame): Raw OHLCV DataFrame with DatetimeIndex.
        config (ProjectConfig): Project configuration parameters.
        include_target (bool): If True, generates forward target columns (t+1).

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - Processed DataFrame with features (and targets if include_target=True).
            - List of feature column names (excluding raw prices and targets).
    """
    data = df.copy()

    # Ensure required columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in data.columns:
            raise KeyError(f"Missing required column: {col}")

    # ==========================================
    # 1. Log Returns & Momentum Lags
    # ==========================================
    data['Return_1d'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Return_5d'] = np.log(data['Close'] / data['Close'].shift(5))
    data['Return_10d'] = np.log(data['Close'] / data['Close'].shift(10))
    data['Return_20d'] = np.log(data['Close'] / data['Close'].shift(20))

    # Intraday dynamic
    data['Intraday_Return'] = np.log(data['Close'] / data['Open'])
    data['High_Low_Ratio'] = (data['High'] - data['Low']) / data['Close']

    # Lagged returns (lag 1 is return from t-1 to t, lag 2 is t-2 to t-1, etc.)
    for lag in config.lag_returns:
        data[f'Return_Lag_{lag}'] = data['Return_1d'].shift(lag)

    # ==========================================
    # 2. Moving Averages (Normalized for Stationarity)
    # ==========================================
    # Note: Raw SMAs are non-stationary prices ($100 vs $200).
    # Normalizing as (Close / SMA - 1.0) creates stationary percentage deviations.
    for w in config.sma_windows:
        sma = data['Close'].rolling(window=w).mean()
        data[f'Dist_SMA_{w}'] = (data['Close'] / sma) - 1.0

    # Fast vs Slow moving average spreads
    sma_10 = data['Close'].rolling(window=10).mean()
    sma_50 = data['Close'].rolling(window=50).mean()
    data['Spread_SMA_10_50'] = (sma_10 / sma_50) - 1.0

    for span in config.ema_spans:
        ema = data['Close'].ewm(span=span, adjust=False).mean()
        data[f'Dist_EMA_{span}'] = (data['Close'] / ema) - 1.0

    # ==========================================
    # 3. Standard Wilder's RSI (14-day)
    # ==========================================
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's exponential smoothing
    alpha_rsi = 1.0 / config.rsi_period
    avg_gain = gain.ewm(alpha=alpha_rsi, min_periods=config.rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha_rsi, min_periods=config.rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    data['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))

    # ==========================================
    # 4. MACD & Normalized Signals
    # ==========================================
    ema_fast = data['Close'].ewm(span=config.macd_fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=config.macd_slow, adjust=False).mean()
    macd_raw = ema_fast - ema_slow
    macd_signal = macd_raw.ewm(span=config.macd_signal, adjust=False).mean()
    
    # Normalized by Close so it is scale-invariant across years
    data['MACD_Norm'] = macd_raw / data['Close']
    data['MACD_Signal_Norm'] = macd_signal / data['Close']
    data['MACD_Hist_Norm'] = (macd_raw - macd_signal) / data['Close']

    # ==========================================
    # 5. Bollinger Bands (20-day, 2 std)
    # ==========================================
    bb_mid = data['Close'].rolling(window=config.bb_window).mean()
    bb_std = data['Close'].rolling(window=config.bb_window).std()
    bb_upper = bb_mid + (bb_std * config.bb_std)
    bb_lower = bb_mid - (bb_std * config.bb_std)

    data['BB_PctB'] = (data['Close'] - bb_lower) / ((bb_upper - bb_lower) + 1e-10)
    data['BB_Width'] = (bb_upper - bb_lower) / (bb_mid + 1e-10)

    # ==========================================
    # 6. Volatility & ATR (Average True Range)
    # ==========================================
    # Annualized 20-day historical volatility
    data['Hist_Vol_20d'] = data['Return_1d'].rolling(window=20).std() * np.sqrt(252)

    # True Range
    prev_close = data['Close'].shift(1)
    tr1 = data['High'] - data['Low']
    tr2 = (data['High'] - prev_close).abs()
    tr3 = (data['Low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / config.atr_period, min_periods=config.atr_period, adjust=False).mean()
    data['ATR_Pct'] = atr / data['Close']

    # ==========================================
    # 7. Volume Indicators
    # ==========================================
    vol_sma_20 = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio_20'] = data['Volume'] / (vol_sma_20 + 1e-10)
    data['Volume_Log_Change'] = np.log((data['Volume'] + 1) / (data['Volume'].shift(1) + 1))

    # ==========================================
    # 8. Calendar / Cyclical Features
    # ==========================================
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month

    # ==========================================
    # 9. Target Formulation (Strictly Forward-Looking)
    # ==========================================
    if include_target:
        # Next day's log return: ln(Close_{t+1} / Close_t)
        data['Target_Next_Close'] = data['Close'].shift(-1)
        data['Target_Return'] = np.log(data['Target_Next_Close'] / data['Close'])
        data['Target_Direction'] = (data['Target_Return'] > 0).astype(int)

    # Identify feature columns (strictly inputs to models)
    excluded_cols = {
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Target_Next_Close', 'Target_Return', 'Target_Direction'
    }
    feature_cols = [c for c in data.columns if c not in excluded_cols]

    # Drop NaNs created by rolling indicators and the forward target
    data.dropna(inplace=True)

    return data, feature_cols

if __name__ == "__main__":
    from data_loader import fetch_stock_data
    df = fetch_stock_data("AAPL", start_date="2020-01-01")
    features_df, feats = engineer_features(df)
    print(f"Engineered {len(feats)} features:")
    print(feats)
    print(features_df[feats].tail())

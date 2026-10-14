"""
Data loading module for fetching, caching, and preprocessing historical stock data.
"""
from pathlib import Path
from typing import Optional
import pandas as pd
import yfinance as yf
from config import DATA_DIR

def fetch_stock_data(
    ticker: str,
    start_date: str = "2018-01-01",
    end_date: Optional[str] = None,
    cache_dir: Optional[Path] = DATA_DIR,
    force_reload: bool = False
) -> pd.DataFrame:
    """
    Fetches daily OHLCV stock data from Yahoo Finance with local caching.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'NVDA').
        start_date (str): Start date string 'YYYY-MM-DD'.
        end_date (str, optional): End date string 'YYYY-MM-DD'. Defaults to latest.
        cache_dir (Path, optional): Directory to store/load cached CSV.
        force_reload (bool): If True, ignores cache and re-downloads data.

    Returns:
        pd.DataFrame: DataFrame containing Open, High, Low, Close, Volume indexed by Date.
    """
    clean_ticker = ticker.upper().strip()
    end_str = end_date if end_date else "latest"
    cache_file = cache_dir / f"{clean_ticker}_{start_date}_{end_str}.csv" if cache_dir else None

    if cache_file and cache_file.exists() and not force_reload:
        print(f"[DataLoader] Loading cached data for {clean_ticker} from {cache_file.name}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    print(f"[DataLoader] Downloading {clean_ticker} data from Yahoo Finance ({start_date} to {end_str})...")
    
    # Try downloading with modern yfinance options
    try:
        df = yf.download(
            clean_ticker,
            start=start_date,
            end=end_date,
            progress=False,
            multi_level_index=False
        )
    except TypeError:
        # Fallback for versions that don't support multi_level_index argument
        df = yf.download(clean_ticker, start=start_date, end=end_date, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No price data returned for ticker '{clean_ticker}'. Check spelling or market availability.")

    # Flatten MultiIndex columns if present (e.g. ('Close', 'AAPL'))
    if isinstance(df.columns, pd.MultiIndex):
        # Look for standard OHLCV names in level 0
        if any(col in df.columns.get_level_values(0) for col in ['Close', 'Open', 'High']):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)

    # Required columns
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Downloaded data missing expected columns: {missing_cols}. Available: {list(df.columns)}")

    df = df[expected_cols].copy()

    # Clean and validate types
    for col in expected_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows where all prices are NaN or 0
    df = df[df['Close'] > 0].dropna()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Save to cache
    if cache_file:
        df.to_csv(cache_file)
        print(f"[DataLoader] Cached data saved to {cache_file}")

    print(f"[DataLoader] Successfully loaded {len(df)} trading days for {clean_ticker}.")
    return df

if __name__ == "__main__":
    test_df = fetch_stock_data("AAPL", start_date="2020-01-01")
    print(test_df.tail())

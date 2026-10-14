"""
Backtesting engine for simulating quantitative trading strategies based on ML predictions.
Accounts for realistic execution, transaction costs, and computes standard quant performance metrics.
"""
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from config import REPORTS_DIR, ProjectConfig, DEFAULT_CONFIG

def run_backtest(
    results_df: pd.DataFrame,
    config: ProjectConfig = DEFAULT_CONFIG,
    save_report: bool = True,
    reports_dir: Optional[Path] = REPORTS_DIR
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulates a signal-driven trading strategy against Buy-and-Hold benchmark.

    Parameters:
        results_df (pd.DataFrame): DataFrame with 'Target_Return' and 'Pred_Return'.
        config (ProjectConfig): Strategy & backtest configurations.
        save_report (bool): If True, writes a summary markdown report.
        reports_dir (Path): Output directory for reports.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - DataFrame with daily portfolio value, returns, and drawdowns.
            - Dictionary of performance metrics.
    """
    bt = results_df.copy()

    # Strategy Signal: 1 (Long) if predicted return > threshold, else 0 (Cash)
    bt['Signal'] = np.where(bt['Pred_Return'] > config.long_threshold, 1, 0)
    
    # Position tracking & trade costs
    bt['Position'] = bt['Signal']
    bt['Trade_Executed'] = bt['Position'].diff().abs().fillna(bt['Position'])
    
    # Strategy Return before and after transaction costs
    bt['Strategy_Gross_Return'] = bt['Position'] * bt['Target_Return']
    bt['Cost'] = bt['Trade_Executed'] * config.transaction_cost_pct
    bt['Strategy_Net_Return'] = bt['Strategy_Gross_Return'] - bt['Cost']

    # Benchmark: 100% Buy and Hold
    bt['Benchmark_Return'] = bt['Target_Return']

    # Cumulative wealth trajectories
    bt['Strategy_Equity'] = config.initial_capital * np.exp(bt['Strategy_Net_Return'].cumsum())
    bt['Benchmark_Equity'] = config.initial_capital * np.exp(bt['Benchmark_Return'].cumsum())

    # Drawdown computation
    bt['Strategy_Peak'] = bt['Strategy_Equity'].cummax()
    bt['Strategy_Drawdown'] = (bt['Strategy_Equity'] - bt['Strategy_Peak']) / bt['Strategy_Peak']

    bt['Benchmark_Peak'] = bt['Benchmark_Equity'].cummax()
    bt['Benchmark_Drawdown'] = (bt['Benchmark_Equity'] - bt['Benchmark_Peak']) / bt['Benchmark_Peak']

    # Performance Statistics
    n_days = len(bt)
    years = n_days / 252.0 if n_days > 0 else 1.0

    strat_total_return = (bt['Strategy_Equity'].iloc[-1] / config.initial_capital) - 1.0
    bench_total_return = (bt['Benchmark_Equity'].iloc[-1] / config.initial_capital) - 1.0

    strat_cagr = (1.0 + strat_total_return) ** (1.0 / years) - 1.0 if years > 0 and strat_total_return > -1 else 0.0
    bench_cagr = (1.0 + bench_total_return) ** (1.0 / years) - 1.0 if years > 0 and bench_total_return > -1 else 0.0

    strat_daily_std = bt['Strategy_Net_Return'].std()
    strat_ann_vol = strat_daily_std * np.sqrt(252) if strat_daily_std > 0 else 1e-6
    bench_ann_vol = bt['Benchmark_Return'].std() * np.sqrt(252)

    # Sharpe Ratio (Daily excess return annualized)
    daily_rf = (1.0 + config.risk_free_rate_annual) ** (1.0 / 252) - 1.0
    excess_strat_daily = bt['Strategy_Net_Return'] - daily_rf
    sharpe_ratio = (excess_strat_daily.mean() / (strat_daily_std + 1e-10)) * np.sqrt(252)

    excess_bench_daily = bt['Benchmark_Return'] - daily_rf
    bench_sharpe = (excess_bench_daily.mean() / (bt['Benchmark_Return'].std() + 1e-10)) * np.sqrt(252)

    max_drawdown = bt['Strategy_Drawdown'].min()
    bench_max_drawdown = bt['Benchmark_Drawdown'].min()
    calmar_ratio = strat_cagr / abs(max_drawdown) if abs(max_drawdown) > 0 else 0.0

    # Trade Statistics
    total_trades = int(bt['Trade_Executed'].sum())
    active_days = int((bt['Position'] > 0).sum())
    winning_days = int((bt['Strategy_Net_Return'] > 0).sum())
    win_rate = (winning_days / active_days * 100.0) if active_days > 0 else 0.0

    metrics = {
        "initial_capital": config.initial_capital,
        "final_strategy_equity": bt['Strategy_Equity'].iloc[-1],
        "final_benchmark_equity": bt['Benchmark_Equity'].iloc[-1],
        "strategy_total_return_pct": strat_total_return * 100.0,
        "benchmark_total_return_pct": bench_total_return * 100.0,
        "strategy_cagr_pct": strat_cagr * 100.0,
        "benchmark_cagr_pct": bench_cagr * 100.0,
        "strategy_annual_volatility_pct": strat_ann_vol * 100.0,
        "benchmark_annual_volatility_pct": bench_ann_vol * 100.0,
        "strategy_sharpe_ratio": sharpe_ratio,
        "benchmark_sharpe_ratio": bench_sharpe,
        "strategy_max_drawdown_pct": max_drawdown * 100.0,
        "benchmark_max_drawdown_pct": bench_max_drawdown * 100.0,
        "calmar_ratio": calmar_ratio,
        "total_trades": total_trades,
        "win_rate_pct": win_rate,
        "days_in_market_pct": (active_days / n_days) * 100.0
    }

    report_text = f"""======================================================
 STRATEGY BACKTEST RESULTS (Out-of-Sample)
======================================================
Trading Duration:         {n_days} days (~{years:.1f} years)
Initial Capital:          ${config.initial_capital:,.2f}
Transaction Fee Model:    {config.transaction_cost_pct*100:.2f}% per execution

--- Portfolio Performance ---
* Final Strategy Value:   ${metrics['final_strategy_equity']:,.2f} ({metrics['strategy_total_return_pct']:+.2f}%)
* Final Benchmark Value:  ${metrics['final_benchmark_equity']:,.2f} ({metrics['benchmark_total_return_pct']:+.2f}%)
* Strategy CAGR:          {metrics['strategy_cagr_pct']:.2f}% (vs Benchmark {metrics['benchmark_cagr_pct']:.2f}%)
* Annualized Volatility:  {metrics['strategy_annual_volatility_pct']:.2f}% (vs Benchmark {metrics['benchmark_annual_volatility_pct']:.2f}%)

--- Risk-Adjusted Ratios ---
* Sharpe Ratio:           {metrics['strategy_sharpe_ratio']:.2f} (vs Benchmark {metrics['benchmark_sharpe_ratio']:.2f})
* Max Drawdown:           {metrics['strategy_max_drawdown_pct']:.2f}% (vs Benchmark {metrics['benchmark_max_drawdown_pct']:.2f}%)
* Calmar Ratio:           {metrics['calmar_ratio']:.2f}

--- Trade Activity ---
* Total Trades:           {metrics['total_trades']}
* Days Invested:          {metrics['days_in_market_pct']:.1f}%
* Daily Win Rate:         {metrics['win_rate_pct']:.1f}%
======================================================
"""
    print(report_text)

    if save_report and reports_dir:
        report_file = reports_dir / "backtest_summary.txt"
        with open(report_file, "w") as f:
            f.write(report_text)
        print(f"[Backtester] Report saved to {report_file}")

    return bt, metrics

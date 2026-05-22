"""
metrics.py
----------
Performance and risk metrics for the pairs trading strategy.

All metrics are computed from a daily PnL series. The caller is responsible
for passing the correct slice (in-sample or out-of-sample). Metrics are never
computed on a mix of in-sample and out-of-sample data  -  doing so would obscure
the true out-of-sample performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

TRADING_DAYS = 252


@dataclass
class PerformanceMetrics:
    """All performance and risk metrics for a single backtest period."""
    total_return: float
    annualised_return: float
    annualised_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    n_trades: int
    avg_holding_days: float
    period_label: str = ""

    def to_series(self) -> pd.Series:
        def fmt_dollar(v: float) -> str:
            return f"${v:,.0f}"

        return pd.Series({
            "Period": self.period_label,
            "Total Return ($)": fmt_dollar(self.total_return),
            "Ann. Return ($/yr)": fmt_dollar(self.annualised_return),
            "Ann. Vol ($/yr)": fmt_dollar(self.annualised_vol),
            "Sharpe": f"{self.sharpe_ratio:.2f}",
            "Sortino": f"{self.sortino_ratio:.2f}",
            "Max Drawdown ($)": fmt_dollar(self.max_drawdown),
            "Calmar": f"{self.calmar_ratio:.2f}",
            "Win Rate": f"{self.win_rate:.1%}",
            "Avg Win ($)": fmt_dollar(self.avg_win),
            "Avg Loss ($)": fmt_dollar(self.avg_loss),
            "Profit Factor": f"{self.profit_factor:.2f}",
            "# Trades": str(self.n_trades),
            "Avg Hold (days)": f"{self.avg_holding_days:.1f}",
        })


def compute_metrics(
    daily_pnl: pd.Series,
    trades: Optional[list] = None,
    period_label: str = "",
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Compute the full performance metric suite from a daily PnL series.

    Parameters
    ----------
    daily_pnl   : pd.Series of daily net PnL (dollar or percentage, consistent).
    trades      : Optional list of Trade objects for trade-level stats.
    period_label: "In-Sample" or "Out-of-Sample" for display.
    risk_free_rate: Daily risk-free rate (default 0).
    """
    pnl = daily_pnl.fillna(0.0)
    n_days = len(pnl)

    if n_days == 0:
        return PerformanceMetrics(
            total_return=0, annualised_return=0, annualised_vol=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0, calmar_ratio=0,
            win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
            n_trades=0, avg_holding_days=0, period_label=period_label,
        )

    equity = pnl.cumsum()
    total_return = equity.iloc[-1]
    n_years = n_days / TRADING_DAYS
    annualised_return = total_return / n_years if n_years > 0 else 0.0

    daily_vol = pnl.std(ddof=1)
    annualised_vol = daily_vol * np.sqrt(TRADING_DAYS)

    # Sharpe ratio (annualised, assuming zero risk-free rate by default)
    excess_daily = pnl - risk_free_rate
    if daily_vol > 0:
        sharpe = excess_daily.mean() / daily_vol * np.sqrt(TRADING_DAYS)
    else:
        sharpe = 0.0

    # Sortino ratio (penalises only downside volatility)
    downside = pnl[pnl < 0]
    downside_vol = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if downside_vol > 0:
        sortino = excess_daily.mean() / downside_vol * np.sqrt(TRADING_DAYS)
    else:
        sortino = 0.0

    # Maximum drawdown
    cumulative = equity
    rolling_max = cumulative.cummax()
    drawdown = cumulative - rolling_max
    max_drawdown = drawdown.min()  # most negative value

    # Calmar ratio (annualised return / |max drawdown|)
    calmar = annualised_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    # Trade-level stats (from Trade objects if provided)
    if trades:
        trade_pnls = [t.net_pnl for t in trades]
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        profit_factor = (
            sum(wins) / abs(sum(losses))
            if losses and sum(losses) != 0 else 0.0
        )
        n_trades = len(trade_pnls)
        avg_holding_days = np.mean([t.holding_days for t in trades])
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0.0
        n_trades = 0
        avg_holding_days = 0.0

    return PerformanceMetrics(
        total_return=total_return,
        annualised_return=annualised_return,
        annualised_vol=annualised_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        n_trades=n_trades,
        avg_holding_days=avg_holding_days,
        period_label=period_label,
    )


def metrics_table(
    in_sample: PerformanceMetrics,
    out_of_sample: PerformanceMetrics,
) -> pd.DataFrame:
    """
    Side-by-side metrics table: in-sample vs out-of-sample.
    """
    is_s = in_sample.to_series()
    oos_s = out_of_sample.to_series()
    df = pd.DataFrame({
        "In-Sample": is_s,
        "Out-of-Sample": oos_s,
    })
    df.index.name = "Metric"
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_equity_curve(
    is_equity: pd.Series,
    oos_equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Portfolio Equity Curve",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot cumulative PnL for in-sample and out-of-sample periods, with an
    optional benchmark (e.g., ASX 200 buy-and-hold normalised to same scale).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    ax.plot(is_equity.index, is_equity.values, label="In-Sample", color="steelblue", lw=1.5)
    ax.plot(oos_equity.index, oos_equity.values, label="Out-of-Sample", color="darkorange", lw=1.5)
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, label="ASX 200 (B&H)", color="grey",
                lw=1.0, linestyle="--", alpha=0.7)

    # Mark the in/out-of-sample boundary
    if len(is_equity) > 0 and len(oos_equity) > 0:
        ax.axvline(x=oos_equity.index[0], color="red", linestyle=":", lw=1.0,
                   label="IS/OOS split")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_drawdown(
    equity: pd.Series,
    title: str = "Drawdown",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot the drawdown series (always non-positive)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    rolling_max = equity.cummax()
    drawdown = equity - rolling_max
    ax.fill_between(drawdown.index, drawdown.values, 0, color="crimson", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color="crimson", lw=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown ($)")
    ax.grid(True, alpha=0.3)
    return ax


def plot_rolling_sharpe(
    daily_pnl: pd.Series,
    window: int = 126,
    title: str = "Rolling 6-Month Sharpe",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot rolling Sharpe ratio with a 126-day (6-month) window."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    roll_mean = daily_pnl.rolling(window=window, min_periods=window // 2).mean()
    roll_std = daily_pnl.rolling(window=window, min_periods=window // 2).std(ddof=1)
    rolling_sharpe = (roll_mean / roll_std) * np.sqrt(TRADING_DAYS)

    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="teal", lw=1.2)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.axhline(1, color="green", lw=0.6, linestyle=":", alpha=0.7, label="Sharpe=1")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Sharpe")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def plot_per_pair_contribution(
    pair_pnls: dict[str, pd.Series],
    title: str = "Per-Pair PnL Contribution",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Bar chart of total PnL contribution per pair."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    labels = list(pair_pnls.keys())
    totals = [s.sum() for s in pair_pnls.values()]
    colors = ["steelblue" if t >= 0 else "crimson" for t in totals]

    ax.bar(labels, totals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Total Net PnL ($)")
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def compute_benchmark_return(
    prices: pd.DataFrame,
    benchmark_ticker: str = "^AXJO",
) -> pd.Series:
    """
    Compute cumulative PnL of a buy-and-hold position in the benchmark,
    normalised to start at 0 (dollar PnL on $1 invested).

    If the benchmark ticker is not in the DataFrame, returns None.
    """
    if benchmark_ticker not in prices.columns:
        return None
    bm = prices[benchmark_ticker].dropna()
    returns = bm.pct_change().fillna(0.0)
    return returns.cumsum()  # approximate cumulative return

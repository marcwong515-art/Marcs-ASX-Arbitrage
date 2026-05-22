"""
backtester.py
-------------
Walk-forward pairs trading backtester. Deliberately written without any
external backtesting library  -  every PnL calculation is explicit and auditable.

Key design decisions:
  1. Rolling hedge ratio: refit every REFIT_DAYS using only the preceding
     HEDGE_WINDOW trading days. No future data is used.
  2. Dollar-neutral sizing: long $N in Y, short $N * beta in X, so net dollar
     exposure ≈ 0 and the position profits from spread convergence only.
  3. Target volatility sizing: N is chosen such that the spread position's
     expected annualised return volatility equals TARGET_VOL * portfolio_capital.
     Formula: N = TARGET_VOL * capital / (spread_return_vol_daily * sqrt(252))
  4. Transaction costs: ROUND_TRIP_BPS applied at entry AND exit on both legs.
     Cost per trade (round trip) = 2 legs × 2 sides × ROUND_TRIP_BPS / 2.
  5. Borrow cost: short leg accrues BORROW_BPS_ANNUAL / 252 per day.
  6. No lookahead: all rolling computations use data strictly before the current bar.

Cost model (per completed round-trip trade):
  Entry:  2 legs × COST_ONE_WAY × N
  Exit:   2 legs × COST_ONE_WAY × N
  Borrow: N × BORROW_DAILY × holding_days

Daily PnL for long-spread position (long Y, short beta*X):
  r_spread = (Y_t - Y_{t-1}) / Y_{t-1} - beta * (X_t - X_{t-1}) / X_{t-1}
  gross_pnl_t = N * r_spread * position_t
  net_pnl_t   = gross_pnl_t - cost_t (entry/exit costs at relevant bars)
                             - N * BORROW_DAILY (while position is open)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.pair_selection import SelectedPair
from src.signals import HEDGE_WINDOW, SignalSeries, compute_signals

# ---------------------------------------------------------------------------
# Cost parameters  -  explicitly declared, never hidden
# ---------------------------------------------------------------------------
ROUND_TRIP_BPS = 10       # 10 bps round-trip per leg (entry + exit combined)
BORROW_BPS_ANNUAL = 50    # 50 bps per annum borrow cost on the short leg
TRADING_DAYS_PER_YEAR = 252
TARGET_VOL_ANNUAL = 0.10  # 10% annualised portfolio volatility target per pair

# Derived cost constants
COST_ONE_WAY = ROUND_TRIP_BPS / 2 / 10_000   # 5 bps as a fraction of notional
BORROW_DAILY = BORROW_BPS_ANNUAL / 10_000 / TRADING_DAYS_PER_YEAR


@dataclass
class Trade:
    """Record of a single completed round-trip trade."""
    ticker_y: str
    ticker_x: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int              # +1 = long spread, -1 = short spread
    notional: float             # dollar exposure per leg Y at entry
    gross_pnl: float
    cost: float                 # total costs: entry + exit + borrow
    net_pnl: float
    holding_days: int
    exit_reason: str            # "mean_reversion", "stop_loss", "time_stop", "end_of_data"


@dataclass
class PairBacktestResult:
    """All outputs from backtesting a single pair."""
    ticker_y: str
    ticker_x: str
    daily_pnl: pd.Series        # net daily PnL in dollar terms
    equity_curve: pd.Series     # cumulative PnL (not including initial capital)
    positions: pd.Series        # +1, -1, 0 at each bar
    signals: Optional[SignalSeries]
    trades: list[Trade]
    n_trades: int
    total_net_pnl: float
    total_cost: float


@dataclass
class PortfolioBacktestResult:
    """Aggregate results across all pairs."""
    pair_results: list[PairBacktestResult]
    daily_pnl: pd.Series        # portfolio-level daily PnL (sum across pairs)
    equity_curve: pd.Series     # cumulative PnL (not including initial capital)
    start_date: pd.Timestamp
    end_date: pd.Timestamp


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def _spread_return_vol(
    y: pd.Series,
    x: pd.Series,
    beta: float,
    window: int = HEDGE_WINDOW,
) -> float:
    """
    Estimate daily volatility of the *spread return* using the past `window` bars.

    Spread return r_t = (Y_t/Y_{t-1} - 1) - beta * (X_t/X_{t-1} - 1)

    Returns daily vol as a fraction (dimensionless).
    No lookahead: called at entry time with data only up to [entry - 1].
    """
    y_ret = y.pct_change().dropna()
    x_ret = x.pct_change().dropna()
    spread_ret = y_ret - beta * x_ret
    if len(spread_ret) < 5:
        return 0.01  # 1% fallback
    return float(spread_ret.tail(window).std(ddof=1))


def _target_notional(
    spread_daily_vol: float,
    capital: float,
    target_vol: float = TARGET_VOL_ANNUAL,
) -> float:
    """
    Dollar notional N such that the spread position has approximately
    target_vol annualised volatility.

    Derivation:
      daily PnL std = N * spread_daily_vol
      annual PnL std = N * spread_daily_vol * sqrt(252)
      set equal to target_vol * capital:
        N = target_vol * capital / (spread_daily_vol * sqrt(252))

    Caps:
      - Minimum notional: 1% of capital (avoid zero-size)
      - Maximum notional: 100% of capital per pair (no leverage beyond 1x)
    """
    if spread_daily_vol < 1e-8:
        return capital * 0.01

    n = (target_vol * capital) / (spread_daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR))
    n = max(n, capital * 0.01)
    n = min(n, capital * 1.0)
    return n


# ---------------------------------------------------------------------------
# Core per-pair PnL engine
# ---------------------------------------------------------------------------

def _compute_daily_pnl(
    y: pd.Series,
    x: pd.Series,
    signals: SignalSeries,
    capital: float,
    target_vol: float = TARGET_VOL_ANNUAL,
) -> tuple[pd.Series, list[Trade]]:
    """
    Translate a SignalSeries into daily net PnL in dollar terms.

    Parameters
    ----------
    y, x      : Aligned price series (same index as signals).
    signals   : Output of compute_signals()  -  contains z_score, position, etc.
    capital   : Dollar capital allocated to this pair.
    target_vol: Annualised volatility target for sizing.

    Returns
    -------
    daily_pnl : pd.Series of net dollar PnL (after costs) indexed by date.
    trades    : list[Trade], one per completed round-trip.

    No lookahead:
      - Spread vol is estimated from data up to bar t-1 (via .iloc[...t] slices).
      - Position is determined by z-score, which uses rolling window ending at t.
    """
    n = len(signals.dates)
    daily_pnl_arr = np.zeros(n)
    trades: list[Trade] = []

    current_pos = 0
    current_notional = 0.0
    current_beta = 1.0
    entry_idx = None
    entry_date = None
    gross_pnl_accumulator = 0.0
    cost_accumulator = 0.0
    bars_held = 0

    pos_arr = signals.position.values
    z_arr = signals.z_score.values
    beta_arr = signals.beta.values
    dates = signals.dates
    y_vals = y.values
    x_vals = x.values

    for t in range(n):
        if t == 0:
            continue  # no previous bar, skip

        beta_t = beta_arr[t] if not np.isnan(beta_arr[t]) else 1.0
        new_pos = int(pos_arr[t])

        # --- Daily gross PnL for existing position ---
        if current_pos != 0:
            y_ret = (y_vals[t] - y_vals[t - 1]) / max(abs(y_vals[t - 1]), 1e-6)
            x_ret = (x_vals[t] - x_vals[t - 1]) / max(abs(x_vals[t - 1]), 1e-6)

            # Spread return: long Y, short beta*X
            r_spread = y_ret - current_beta * x_ret
            gross_t = current_notional * current_pos * r_spread

            # Daily borrow on the short leg notional
            borrow_t = current_notional * BORROW_DAILY

            daily_pnl_arr[t] += gross_t - borrow_t
            gross_pnl_accumulator += gross_t
            cost_accumulator += borrow_t
            bars_held += 1

        # --- Exit: position changed from non-zero to flat (or reversed) ---
        if current_pos != 0 and new_pos != current_pos:
            # Close current position
            exit_cost = 2 * current_notional * COST_ONE_WAY  # both legs, exit side
            daily_pnl_arr[t] -= exit_cost
            cost_accumulator += exit_cost

            # Determine exit reason from z-score
            z_t = z_arr[t] if not np.isnan(z_arr[t]) else 0.0
            if abs(z_t) > 3.5:
                reason = "stop_loss"
            elif bars_held >= int(2 * signals.dates.shape[0]):  # approximate
                reason = "time_stop"
            else:
                reason = "mean_reversion"

            trades.append(Trade(
                ticker_y=str(y.name or "Y"),
                ticker_x=str(x.name or "X"),
                entry_date=entry_date,
                exit_date=dates[t],
                direction=current_pos,
                notional=current_notional,
                gross_pnl=gross_pnl_accumulator,
                cost=cost_accumulator,
                net_pnl=gross_pnl_accumulator - cost_accumulator,
                holding_days=bars_held,
                exit_reason=reason,
            ))

            current_pos = 0
            current_notional = 0.0
            entry_idx = None
            entry_date = None
            gross_pnl_accumulator = 0.0
            cost_accumulator = 0.0
            bars_held = 0

        # --- Entry: position moved from flat to non-zero ---
        if current_pos == 0 and new_pos != 0:
            # Size the position using spread vol from recent past (no lookahead)
            lookback_start = max(0, t - HEDGE_WINDOW)
            y_past = y.iloc[lookback_start:t]
            x_past = x.iloc[lookback_start:t]
            s_vol = _spread_return_vol(y_past, x_past, beta_t, window=HEDGE_WINDOW)
            current_notional = _target_notional(s_vol, capital, target_vol)
            current_pos = new_pos
            current_beta = beta_t
            entry_idx = t
            entry_date = dates[t]
            gross_pnl_accumulator = 0.0
            bars_held = 0

            # Entry cost: both legs, entry side
            entry_cost = 2 * current_notional * COST_ONE_WAY
            daily_pnl_arr[t] -= entry_cost
            cost_accumulator = entry_cost

    # Close any position still open at end of data
    if current_pos != 0 and entry_idx is not None:
        exit_cost = 2 * current_notional * COST_ONE_WAY
        daily_pnl_arr[-1] -= exit_cost
        cost_accumulator += exit_cost

        trades.append(Trade(
            ticker_y=str(y.name or "Y"),
            ticker_x=str(x.name or "X"),
            entry_date=entry_date,
            exit_date=dates[-1],
            direction=current_pos,
            notional=current_notional,
            gross_pnl=gross_pnl_accumulator,
            cost=cost_accumulator,
            net_pnl=gross_pnl_accumulator - cost_accumulator,
            holding_days=bars_held,
            exit_reason="end_of_data",
        ))

    return pd.Series(daily_pnl_arr, index=signals.dates), trades


# ---------------------------------------------------------------------------
# Public backtest functions
# ---------------------------------------------------------------------------

def backtest_pair(
    pair: SelectedPair,
    prices: pd.DataFrame,
    half_life_days: float,
    capital: float,
    target_vol_annual: float = TARGET_VOL_ANNUAL,
) -> PairBacktestResult:
    """
    Backtest a single pair over the provided price history.

    The caller controls which price slice is passed (in-sample or OOS).
    This function applies no IS/OOS logic  -  it processes whatever it receives,
    which prevents any accidental data snooping.

    No lookahead: signals.compute_signals() uses rolling windows ending at t-1
    for the hedge ratio, and the z-score window includes t (the current bar).
    Sizing uses rolling vol ending at t-1.
    """
    ticker_y = pair.analysis.ticker_y
    ticker_x = pair.analysis.ticker_x

    if ticker_y not in prices.columns or ticker_x not in prices.columns:
        raise ValueError(f"Tickers {ticker_y} or {ticker_x} not found in price data")

    y = prices[ticker_y].dropna().rename(ticker_y)
    x = prices[ticker_x].dropna().rename(ticker_x)

    common = y.index.intersection(x.index)
    y = y.loc[common]
    x = x.loc[common]

    if len(common) < HEDGE_WINDOW * 2:
        empty = pd.Series(0.0, index=common)
        return PairBacktestResult(
            ticker_y=ticker_y, ticker_x=ticker_x,
            daily_pnl=empty, equity_curve=empty,
            positions=empty, signals=None, trades=[],
            n_trades=0, total_net_pnl=0.0, total_cost=0.0,
        )

    sigs = compute_signals(y, x, half_life_days=half_life_days, window=HEDGE_WINDOW)

    daily_pnl, trades = _compute_daily_pnl(y, x, sigs, capital, target_vol_annual)

    total_cost = sum(t.cost for t in trades)

    return PairBacktestResult(
        ticker_y=ticker_y, ticker_x=ticker_x,
        daily_pnl=daily_pnl,
        equity_curve=daily_pnl.cumsum(),
        positions=sigs.position,
        signals=sigs,
        trades=trades,
        n_trades=len(trades),
        total_net_pnl=float(daily_pnl.sum()),
        total_cost=total_cost,
    )


def backtest_portfolio(
    pairs: list[SelectedPair],
    prices: pd.DataFrame,
    half_life_days_map: dict[str, float],
    total_capital: float,
    target_vol_annual: float = TARGET_VOL_ANNUAL,
) -> PortfolioBacktestResult:
    """
    Backtest all selected pairs and aggregate to portfolio level.

    Capital is split equally among pairs. Each pair is independently sized
    to its target volatility from its allocated capital.

    Parameters
    ----------
    pairs              : From pair_selection.select_pairs().
    prices             : Price DataFrame (full history or a slice thereof).
    half_life_days_map : {f"{ty}~{tx}": half_life} for each pair.
    total_capital      : Total AUD capital to allocate.
    """
    n_pairs = max(len(pairs), 1)
    capital_per_pair = total_capital / n_pairs

    pair_results: list[PairBacktestResult] = []
    all_pnl: list[pd.Series] = []

    for sp in pairs:
        key = f"{sp.analysis.ticker_y}~{sp.analysis.ticker_x}"
        hl = half_life_days_map.get(key, sp.half_life_days)

        result = backtest_pair(
            sp, prices,
            half_life_days=hl,
            capital=capital_per_pair,
            target_vol_annual=target_vol_annual,
        )
        pair_results.append(result)
        all_pnl.append(result.daily_pnl)

    if all_pnl:
        pnl_df = pd.concat(all_pnl, axis=1).fillna(0.0)
        portfolio_pnl = pnl_df.sum(axis=1)
    else:
        portfolio_pnl = pd.Series(dtype=float)

    equity = portfolio_pnl.cumsum()

    return PortfolioBacktestResult(
        pair_results=pair_results,
        daily_pnl=portfolio_pnl,
        equity_curve=equity,
        start_date=portfolio_pnl.index[0] if len(portfolio_pnl) > 0 else pd.NaT,
        end_date=portfolio_pnl.index[-1] if len(portfolio_pnl) > 0 else pd.NaT,
    )

"""
signals.py
----------
Generates trading signals for a pairs trade using a rolling z-score of the spread.

Key design decisions (each chosen to avoid lookahead bias):

1. Hedge ratio is estimated by rolling 60-day OLS, not a static in-sample fit.
   At each date t, the hedge ratio uses only prices from [t-60, t-1].
   A static hedge ratio fitted on in-sample data, then applied to all of
   out-of-sample, would be a mild form of lookahead — the ratio would "know"
   the full distributional properties of the in-sample period.

2. The z-score normalisation (mean, std) uses the same rolling 60-day window.
   Again, the parameters at date t use only past data.

3. Entry and exit thresholds are fixed constants, not calibrated on any data.
   Calibrating thresholds on in-sample spreads, then applying them out-of-sample,
   is allowed; calibrating on the full dataset would be lookahead.

Signal logic:
  - Compute spread_t = Y_t - beta_t * X_t  (rolling beta from 60-day OLS)
  - z_t = (spread_t - rolling_mean_t) / rolling_std_t
  - Entry long spread:   z_t < -ENTRY_Z  (spread cheap)
  - Entry short spread:  z_t > +ENTRY_Z  (spread rich)
  - Exit:                |z_t| < EXIT_Z
  - Stop loss:           |z_t| > STOP_Z
  - Time stop:           position held > 2 * half_life days
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# ---------------------------------------------------------------------------
# Signal thresholds — fixed, not calibrated on data
# ---------------------------------------------------------------------------
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 3.5
HEDGE_WINDOW = 60   # days for rolling OLS and z-score normalisation


@dataclass
class SignalSeries:
    """All time-series produced by the signal generator for one pair."""
    dates: pd.DatetimeIndex
    spread: pd.Series          # raw spread: Y - beta*X
    z_score: pd.Series         # normalised spread
    beta: pd.Series            # rolling hedge ratio
    position: pd.Series        # +1 (long spread), -1 (short spread), 0 (flat)
    entry_signal: pd.Series    # +1/-1 at entry bars, 0 otherwise
    exit_signal: pd.Series     # 1 at exit bars, 0 otherwise


def rolling_ols_beta(
    y: pd.Series,
    x: pd.Series,
    window: int = HEDGE_WINDOW,
) -> pd.Series:
    """
    Compute the rolling OLS hedge ratio β_t using the past `window` observations.

    β_t = Cov(Y[t-w:t], X[t-w:t]) / Var(X[t-w:t])

    No lookahead: at time t, only data strictly before t is used.
    The first `window` observations produce NaN (insufficient history).
    """
    betas = np.full(len(y), np.nan)

    y_arr = y.values
    x_arr = x.values

    for t in range(window, len(y)):
        # [t-window, t) — excludes index t itself
        y_w = y_arr[t - window: t]
        x_w = x_arr[t - window: t]

        # Skip if any NaN in window
        mask = ~(np.isnan(y_w) | np.isnan(x_w))
        if mask.sum() < window // 2:
            continue

        y_clean = y_w[mask]
        x_clean = x_w[mask]

        cov = np.cov(y_clean, x_clean, ddof=1)
        var_x = cov[1, 1]
        if var_x < 1e-10:
            continue
        betas[t] = cov[0, 1] / var_x

    return pd.Series(betas, index=y.index)


def compute_spread(
    y: pd.Series,
    x: pd.Series,
    beta: pd.Series,
) -> pd.Series:
    """
    Spread = Y_t - beta_t * X_t.

    Uses the rolling beta so that each spread value is computed with a hedge
    ratio estimated purely from past data.
    """
    return y - beta * x


def compute_z_score(
    spread: pd.Series,
    window: int = HEDGE_WINDOW,
) -> pd.Series:
    """
    Z-score of the spread using a rolling window mean and std.

    z_t = (spread_t - mean(spread[t-w:t])) / std(spread[t-w:t])

    ddof=1 for sample standard deviation.
    NaN produced where std is zero or window insufficient.
    No lookahead: parameters at time t use only [t-window, t).
    """
    roll_mean = spread.rolling(window=window, min_periods=window).mean()
    roll_std = spread.rolling(window=window, min_periods=window).std(ddof=1)
    z = (spread - roll_mean) / roll_std
    return z


def generate_positions(
    z_score: pd.Series,
    half_life_days: float,
    entry_z: float = ENTRY_Z,
    exit_z: float = EXIT_Z,
    stop_z: float = STOP_Z,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Convert z-scores into a position series using a state machine.

    State transitions:
      FLAT -> LONG:  z < -entry_z   (spread below average => expect mean reversion up)
      FLAT -> SHORT: z > +entry_z   (spread above average => expect mean reversion down)
      LONG -> FLAT:  z > -exit_z    (target zone reached)
      SHORT -> FLAT: z < +exit_z    (target zone reached)
      ANY -> FLAT:   |z| > stop_z   (stop loss)
      ANY -> FLAT:   held > 2 * half_life days (time stop)

    Returns
    -------
    position     : pd.Series, values in {-1, 0, +1}
    entry_signal : pd.Series, +1/-1 at entry bars
    exit_signal  : pd.Series, 1 at bars where a position is closed
    """
    n = len(z_score)
    z = z_score.values
    positions = np.zeros(n)
    entry_signals = np.zeros(n)
    exit_signals = np.zeros(n)

    current_pos = 0      # current position: +1, -1, or 0
    bars_held = 0        # number of bars since entry
    time_stop = int(2 * half_life_days)

    for t in range(n):
        if np.isnan(z[t]):
            positions[t] = 0
            current_pos = 0
            bars_held = 0
            continue

        if current_pos != 0:
            bars_held += 1

        # --- Exit logic (checked before entry) ---
        if current_pos != 0:
            should_exit = False

            # Time stop
            if bars_held >= time_stop:
                should_exit = True

            # Stop loss
            elif abs(z[t]) > stop_z:
                should_exit = True

            # Mean reversion reached: long closes when z moves above -exit_z
            elif current_pos == 1 and z[t] > -exit_z:
                should_exit = True

            # Mean reversion reached: short closes when z moves below +exit_z
            elif current_pos == -1 and z[t] < exit_z:
                should_exit = True

            if should_exit:
                exit_signals[t] = 1
                current_pos = 0
                bars_held = 0

        # --- Entry logic ---
        if current_pos == 0:
            if z[t] < -entry_z:
                current_pos = 1
                entry_signals[t] = 1
                bars_held = 0
            elif z[t] > entry_z:
                current_pos = -1
                entry_signals[t] = -1
                bars_held = 0

        positions[t] = current_pos

    return (
        pd.Series(positions, index=z_score.index, dtype=float),
        pd.Series(entry_signals, index=z_score.index, dtype=float),
        pd.Series(exit_signals, index=z_score.index, dtype=float),
    )


def compute_signals(
    y: pd.Series,
    x: pd.Series,
    half_life_days: float,
    window: int = HEDGE_WINDOW,
    entry_z: float = ENTRY_Z,
    exit_z: float = EXIT_Z,
    stop_z: float = STOP_Z,
) -> SignalSeries:
    """
    Full signal pipeline for a single pair.

    No lookahead: all computations use rolling/lagged windows only.

    Parameters
    ----------
    y, x          : Price series for the two legs.
    half_life_days: OU half-life used for the time stop.
    window        : Rolling window for OLS and z-score (default 60 days).
    entry_z       : Z-score threshold for entry (default 2.0).
    exit_z        : Z-score threshold for exit (default 0.5).
    stop_z        : Z-score threshold for stop loss (default 3.5).
    """
    aligned = pd.concat([y, x], axis=1).dropna()
    y_a = aligned.iloc[:, 0]
    x_a = aligned.iloc[:, 1]

    beta = rolling_ols_beta(y_a, x_a, window=window)
    spread = compute_spread(y_a, x_a, beta)
    z = compute_z_score(spread, window=window)

    position, entry_signal, exit_signal = generate_positions(
        z, half_life_days=half_life_days,
        entry_z=entry_z, exit_z=exit_z, stop_z=stop_z,
    )

    return SignalSeries(
        dates=y_a.index,
        spread=spread,
        z_score=z,
        beta=beta,
        position=position,
        entry_signal=entry_signal,
        exit_signal=exit_signal,
    )

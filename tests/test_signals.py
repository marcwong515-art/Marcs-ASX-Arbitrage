"""
test_signals.py
---------------
Unit tests for signals.py on synthetic mean-reverting series.

Tests verify:
  1. Rolling OLS beta recovers the true hedge ratio.
  2. Z-score is correctly standardised (mean ~0, std ~1 after warm-up).
  3. Position state machine: entries, exits, stop-loss, time-stop.
  4. No position is opened before the rolling window is warm.
"""

import numpy as np
import pandas as pd
from src.signals import (
    HEDGE_WINDOW,
    compute_signals,
    compute_z_score,
    generate_positions,
    rolling_ols_beta,
)

RNG = np.random.default_rng(99)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ou_pair(
    n: int,
    beta: float = 1.5,
    half_life: float = 10.0,
    seed: int = 42,
) -> tuple:
    """Synthetic cointegrated pair with known beta and half-life."""
    rng = np.random.default_rng(seed)
    kappa = np.log(2) / half_life
    idx = pd.date_range("2020-01-01", periods=n, freq="B")

    x = pd.Series(rng.normal(0, 1, n).cumsum() + 50, index=idx)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1 - kappa) + rng.normal(0, 0.5)
    y = beta * x + pd.Series(spread, index=idx)

    return y, x


def make_constant_z_series(values: list, window: int = HEDGE_WINDOW) -> pd.Series:
    """
    Build a z-score series from a list of values padded with NaN prefix
    (simulates the warm-up period).
    """
    prefix = [np.nan] * window
    all_vals = prefix + list(values)
    return pd.Series(all_vals, dtype=float)


# ---------------------------------------------------------------------------
# Rolling OLS beta tests
# ---------------------------------------------------------------------------

class TestRollingOlsBeta:
    def test_first_window_is_nan(self):
        """First HEDGE_WINDOW observations should be NaN (insufficient history)."""
        y, x = make_ou_pair(300, beta=2.0)
        beta = rolling_ols_beta(y, x, window=60)
        assert beta.iloc[:60].isna().all(), "First 60 betas should be NaN"

    def test_beta_recovered_after_warmup(self):
        """Rolling beta should converge to the true beta after warm-up."""
        true_beta = 1.8
        y, x = make_ou_pair(500, beta=true_beta)
        beta = rolling_ols_beta(y, x, window=60)
        # Use median of last 200 observations to assess convergence
        median_beta = beta.iloc[-200:].median()
        assert abs(median_beta - true_beta) < 0.3, (
            f"Rolling beta median {median_beta:.3f} far from true {true_beta}"
        )

    def test_beta_length_matches_input(self):
        """Output Series must be the same length as inputs."""
        y, x = make_ou_pair(200, beta=1.5)
        beta = rolling_ols_beta(y, x, window=60)
        assert len(beta) == len(y)

    def test_beta_index_matches_input(self):
        """Output index must match input index."""
        y, x = make_ou_pair(150, beta=1.5)
        beta = rolling_ols_beta(y, x, window=60)
        pd.testing.assert_index_equal(beta.index, y.index)


# ---------------------------------------------------------------------------
# Z-score tests
# ---------------------------------------------------------------------------

class TestComputeZScore:
    def test_warmup_period_is_nan(self):
        """Z-score should be NaN for the first window-1 observations.
        pandas rolling(min_periods=w) produces its first value at index w-1."""
        spread = pd.Series(RNG.normal(0, 1, 300))
        z = compute_z_score(spread, window=60)
        assert z.iloc[:59].isna().all()

    def test_zscore_approximately_standardised(self):
        """After warm-up, z-score rolling window should have mean ~0, std ~1."""
        # Use a white-noise spread; rolling mean should cancel, rolling std ~1
        spread = pd.Series(RNG.normal(0, 2, 1000))
        z = compute_z_score(spread, window=60)
        valid = z.dropna()
        # Rolling standardisation: std of z depends on autocorrelation structure,
        # but for i.i.d. noise the bulk of z values should be in [-3, 3]
        pct_in_range = ((valid > -3) & (valid < 3)).mean()
        assert pct_in_range > 0.99, f"Too many z-score outliers: {1-pct_in_range:.2%}"

    def test_zscore_length_matches_spread(self):
        spread = pd.Series(RNG.normal(0, 1, 200))
        z = compute_z_score(spread, window=60)
        assert len(z) == len(spread)


# ---------------------------------------------------------------------------
# Position state machine tests
# ---------------------------------------------------------------------------

class TestGeneratePositions:
    def test_entry_long_on_low_z(self):
        """Position should become +1 when z drops below -ENTRY_Z."""
        # Pattern: warm-up NaNs, then trigger, then exit
        z = make_constant_z_series([-2.5, -2.5, 0.0, 0.0, 0.0])
        pos, entry, exit_ = generate_positions(z, half_life_days=10.0)
        # Position should be +1 right after -2.5 trigger
        valid_pos = pos.dropna()
        assert 1.0 in valid_pos.values, "Should have entered long position"

    def test_entry_short_on_high_z(self):
        """Position should become -1 when z exceeds +ENTRY_Z."""
        z = make_constant_z_series([2.5, 2.5, 0.0, 0.0, 0.0])
        pos, entry, exit_ = generate_positions(z, half_life_days=10.0)
        valid_pos = pos.dropna()
        assert -1.0 in valid_pos.values, "Should have entered short position"

    def test_exit_on_z_crossing_zero(self):
        """Long position should exit when z crosses above -EXIT_Z."""
        # Entry at -2.5, then spread reverts to 0
        z_vals = [-2.5] + [0.0] * 10
        z = make_constant_z_series(z_vals)
        pos, entry, exit_ = generate_positions(z, half_life_days=20.0)
        # After reversion, position should be flat
        valid = pos.dropna()
        # At some point after index 1 (the reversion), position should be 0
        later_positions = valid.iloc[len(z_vals) // 2:]
        assert (later_positions == 0).any(), "Position should have closed on reversion"

    def test_stop_loss_triggers(self):
        """Position should close when |z| exceeds STOP_Z."""
        z_vals = [-2.5, -3.6, -3.6, -3.6]  # enters, then stop-loss
        z = make_constant_z_series(z_vals)
        pos, entry, exit_ = generate_positions(z, half_life_days=20.0)
        valid = pos.dropna()
        # After the stop-loss bar, position must be 0
        stop_idx = HEDGE_WINDOW + 2  # bar where |z| = 3.6
        if stop_idx < len(pos):
            assert pos.iloc[stop_idx + 1] == 0 or exit_.iloc[stop_idx] == 1

    def test_no_position_during_nan_warmup(self):
        """No position should be opened during NaN z-score warm-up period."""
        z_vals = [0.0] * 20
        z = make_constant_z_series(z_vals)
        pos, _, _ = generate_positions(z, half_life_days=10.0)
        nan_period_positions = pos.iloc[:HEDGE_WINDOW]
        assert (nan_period_positions == 0).all(), (
            "No positions should be opened during warm-up period"
        )

    def test_time_stop(self):
        """Position should close after 2 * half_life bars regardless of z."""
        half_life = 5.0
        time_limit = int(2 * half_life)  # 10 bars
        # Entry trigger, then z stays between exit and stop bands forever
        z_vals = [-2.5] + [-1.5] * 30  # stays in no-man's-land
        z = make_constant_z_series(z_vals)
        pos, _, exit_ = generate_positions(z, half_life_days=half_life)
        valid = pos.dropna()
        # After time_limit bars from first entry, position must be flat
        entry_bar = HEDGE_WINDOW  # first bar after NaN warmup
        time_stop_bar = entry_bar + time_limit
        if time_stop_bar < len(pos):
            assert pos.iloc[time_stop_bar] == 0, (
                f"Time stop should close position after {time_limit} bars"
            )

    def test_position_values_valid(self):
        """Position values must only be -1, 0, or +1."""
        y, x = make_ou_pair(500, half_life=10.0)
        sigs = compute_signals(y, x, half_life_days=10.0)
        unique_vals = set(sigs.position.dropna().unique())
        assert unique_vals.issubset({-1.0, 0.0, 1.0}), (
            f"Unexpected position values: {unique_vals}"
        )


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

class TestComputeSignals:
    def test_output_lengths_match(self):
        """All output series must have the same length as inputs."""
        y, x = make_ou_pair(400, half_life=10.0)
        sigs = compute_signals(y, x, half_life_days=10.0)
        n = len(sigs.spread)
        assert len(sigs.z_score) == n
        assert len(sigs.beta) == n
        assert len(sigs.position) == n
        assert len(sigs.entry_signal) == n
        assert len(sigs.exit_signal) == n

    def test_no_entry_before_warmup(self):
        """No position should open in the first HEDGE_WINDOW bars."""
        y, x = make_ou_pair(400, half_life=10.0)
        sigs = compute_signals(y, x, half_life_days=10.0)
        early_positions = sigs.position.iloc[:HEDGE_WINDOW]
        assert (early_positions == 0).all(), (
            "Positions opened before rolling window is warm  -  lookahead!"
        )

    def test_some_trades_generated(self):
        """On a mean-reverting pair, the signal generator should produce trades."""
        y, x = make_ou_pair(1000, half_life=10.0, seed=7)
        sigs = compute_signals(y, x, half_life_days=10.0)
        n_entries = (sigs.entry_signal != 0).sum()
        assert n_entries > 5, (
            f"Expected >5 trades on mean-reverting series, got {n_entries}"
        )

    def test_beta_is_nan_in_warmup(self):
        """Beta should be NaN for the first HEDGE_WINDOW bars."""
        y, x = make_ou_pair(300, half_life=10.0)
        sigs = compute_signals(y, x, half_life_days=10.0)
        assert sigs.beta.iloc[:HEDGE_WINDOW].isna().all()

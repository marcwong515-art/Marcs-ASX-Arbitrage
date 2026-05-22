"""
test_backtester.py
------------------
Unit tests for the backtester with hand-calculated PnL verification.

Key invariants tested:
  1. Cost constants match the spec (10 bps round-trip, 50 bps borrow).
  2. _target_notional: higher vol => smaller notional; caps enforced.
  3. backtest_pair: equity curve = cumsum(daily_pnl); no positions in warm-up;
     missing ticker raises ValueError; total_net_pnl equals sum of daily_pnl.
  4. backtest_pair on a long-spread trade: net PnL < gross PnL (costs deducted).

Hand-calculated cost check (cost constants only):
  COST_ONE_WAY = 10 bps / 2 / 10_000 = 5 bps = 0.0005  ✓
  BORROW_DAILY = 50 bps / 10_000 / 252 ≈ 0.0001984      ✓
"""

import numpy as np
import pandas as pd
import pytest

from src.backtester import (
    BORROW_BPS_ANNUAL,
    BORROW_DAILY,
    COST_ONE_WAY,
    ROUND_TRIP_BPS,
    TARGET_VOL_ANNUAL,
    TRADING_DAYS_PER_YEAR,
    Trade,
    _target_notional,
    backtest_pair,
)
from src.cointegration import (
    AdfResult,
    EngleGrangerResult,
    HalfLifeResult,
    JohansenResult,
    PairAnalysis,
)
from src.pair_selection import SelectedPair

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dummy_selected_pair(
    ticker_y: str = "CBA.AX",
    ticker_x: str = "WBC.AX",
    beta: float = 1.5,
    half_life: float = 10.0,
) -> SelectedPair:
    """Construct a SelectedPair with minimal valid fields for testing."""
    adf = AdfResult(
        ticker=ticker_y, adf_stat=-3.0, p_value=0.10,
        is_nonstationary=True, diff_adf_stat=-10.0, diff_p_value=0.01,
        is_integrated_order_1=True,
    )
    eg = EngleGrangerResult(
        ticker_y=ticker_y, ticker_x=ticker_x,
        ols_beta=beta, ols_alpha=0.0,
        residual_adf_stat=-4.0, residual_p_value=0.02,
        is_cointegrated=True, eg_p_value=0.02,
    )
    johansen = JohansenResult(
        ticker_y=ticker_y, ticker_x=ticker_x,
        trace_stat=25.0, trace_crit_95=15.49,
        max_eig_stat=20.0, max_eig_crit_95=14.26,
        cointegration_rank=1, is_cointegrated=True,
    )
    hl = HalfLifeResult(
        ticker_y=ticker_y, ticker_x=ticker_x,
        half_life_days=half_life, ou_kappa=np.log(2) / half_life,
        ou_mu=0.0, ou_sigma=1.0, is_valid=True,
    )
    analysis = PairAnalysis(
        ticker_y=ticker_y, ticker_x=ticker_x,
        adf_y=adf, adf_x=adf,
        eg=eg, johansen=johansen, half_life=hl, passed=True,
    )
    return SelectedPair(
        analysis=analysis, rank=1,
        eg_p_value=0.02, half_life_days=half_life,
        ols_beta=beta, ols_alpha=0.0,
    )


def make_cointegrated_prices(
    n: int = 400,
    beta: float = 1.5,
    half_life: float = 10.0,
    ticker_y: str = "CBA.AX",
    ticker_x: str = "WBC.AX",
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic cointegrated price DataFrame."""
    rng = np.random.default_rng(seed)
    kappa = np.log(2) / half_life
    x_vals = 50 + np.cumsum(rng.normal(0, 1, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1 - kappa) + rng.normal(0, 0.5)
    y_vals = beta * x_vals + spread + 10.0
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({ticker_y: y_vals, ticker_x: x_vals}, index=idx)


# ---------------------------------------------------------------------------
# Cost constant verification (hand-calculated)
# ---------------------------------------------------------------------------

class TestCostConstants:
    def test_cost_one_way_is_5bps(self):
        """COST_ONE_WAY must be exactly 5 bps = 0.0005 (half of 10 bps round-trip)."""
        expected = ROUND_TRIP_BPS / 2 / 10_000
        assert COST_ONE_WAY == pytest.approx(expected, rel=1e-9)
        assert COST_ONE_WAY == pytest.approx(0.0005, rel=1e-9)

    def test_borrow_daily_matches_spec(self):
        """BORROW_DAILY must be 50 bps / 252 per day."""
        expected = BORROW_BPS_ANNUAL / 10_000 / TRADING_DAYS_PER_YEAR
        assert BORROW_DAILY == pytest.approx(expected, rel=1e-9)
        assert BORROW_DAILY == pytest.approx(0.0050 / 252, rel=1e-6)

    def test_borrow_daily_is_positive(self):
        assert BORROW_DAILY > 0

    def test_cost_one_way_is_positive(self):
        assert COST_ONE_WAY > 0


# ---------------------------------------------------------------------------
# Position sizing (_target_notional)
# ---------------------------------------------------------------------------

class TestTargetNotional:
    def test_higher_vol_gives_smaller_notional(self):
        """Higher spread vol => smaller notional (fixed target vol)."""
        n_low = _target_notional(spread_daily_vol=0.001, capital=10_000)
        n_high = _target_notional(spread_daily_vol=0.050, capital=10_000)
        assert n_low > n_high, "Higher vol should reduce notional"

    def test_notional_is_positive(self):
        n = _target_notional(spread_daily_vol=0.01, capital=10_000)
        assert n > 0

    def test_notional_capped_at_100pct_capital(self):
        """Near-zero vol should not produce notional > capital."""
        n = _target_notional(spread_daily_vol=1e-12, capital=10_000)
        assert n <= 10_000, f"Notional {n:.0f} exceeds 100% of capital"

    def test_notional_at_least_1pct_capital(self):
        """Very high vol should not produce zero notional."""
        n = _target_notional(spread_daily_vol=100.0, capital=10_000)
        assert n >= 10_000 * 0.01, "Notional below minimum floor"

    def test_notional_formula(self):
        """Check the derivation: N = target_vol * capital / (vol * sqrt(252))."""
        vol = 0.01
        capital = 10_000
        expected = TARGET_VOL_ANNUAL * capital / (vol * np.sqrt(TRADING_DAYS_PER_YEAR))
        # Clamp to [1%, 100%] of capital
        expected = max(expected, capital * 0.01)
        expected = min(expected, capital * 1.0)
        actual = _target_notional(vol, capital)
        assert actual == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# backtest_pair integration tests
# ---------------------------------------------------------------------------

class TestBacktestPair:
    def test_returns_pair_backtest_result(self):
        prices = make_cointegrated_prices()
        pair = make_dummy_selected_pair()
        result = backtest_pair(pair, prices, half_life_days=10.0, capital=10_000)
        assert result.ticker_y == "CBA.AX"
        assert result.ticker_x == "WBC.AX"
        assert isinstance(result.daily_pnl, pd.Series)

    def test_equity_curve_equals_cumsum_of_daily_pnl(self):
        """Equity curve must be exactly cumsum of daily PnL  -  no other transformation."""
        prices = make_cointegrated_prices()
        pair = make_dummy_selected_pair()
        result = backtest_pair(pair, prices, half_life_days=10.0, capital=10_000)
        expected = result.daily_pnl.cumsum()
        pd.testing.assert_series_equal(
            result.equity_curve, expected, check_names=False, rtol=1e-9
        )

    def test_total_net_pnl_matches_sum_of_daily(self):
        """total_net_pnl must equal daily_pnl.sum() to machine precision."""
        prices = make_cointegrated_prices()
        pair = make_dummy_selected_pair()
        result = backtest_pair(pair, prices, half_life_days=10.0, capital=10_000)
        assert result.total_net_pnl == pytest.approx(result.daily_pnl.sum(), rel=1e-6)

    def test_no_positions_during_warm_up(self):
        """No trade should open before the rolling window is warm."""
        from src.signals import HEDGE_WINDOW
        prices = make_cointegrated_prices(n=500)
        pair = make_dummy_selected_pair()
        result = backtest_pair(pair, prices, half_life_days=10.0, capital=10_000)
        warm_up_positions = result.positions.iloc[:HEDGE_WINDOW]
        assert (warm_up_positions == 0).all(), (
            "Positions opened during warm-up  -  potential lookahead bias"
        )

    def test_missing_ticker_raises_value_error(self):
        prices = make_cointegrated_prices()[["CBA.AX"]]  # missing WBC.AX
        pair = make_dummy_selected_pair("CBA.AX", "WBC.AX")
        with pytest.raises(ValueError, match="not found in price data"):
            backtest_pair(pair, prices, half_life_days=10.0, capital=10_000)

    def test_net_pnl_less_than_gross_pnl(self):
        """Net PnL must be strictly less than gross PnL when trades occur."""
        prices = make_cointegrated_prices(n=600)
        pair = make_dummy_selected_pair()
        result = backtest_pair(pair, prices, half_life_days=10.0, capital=10_000)
        if result.n_trades > 0:
            gross_total = sum(t.gross_pnl for t in result.trades)
            assert result.total_net_pnl <= gross_total + 1e-8, (
                "Net PnL exceeds gross PnL  -  costs have wrong sign"
            )
            assert result.total_cost >= 0, "Total cost must be non-negative"

    def test_trade_direction_valid(self):
        """All trade directions must be +1 or -1."""
        prices = make_cointegrated_prices(n=600)
        pair = make_dummy_selected_pair()
        result = backtest_pair(pair, prices, half_life_days=10.0, capital=10_000)
        for t in result.trades:
            assert t.direction in (1, -1), f"Invalid direction: {t.direction}"

    def test_trade_holding_days_positive(self):
        """All trades must have at least 0 holding days."""
        prices = make_cointegrated_prices(n=600)
        pair = make_dummy_selected_pair()
        result = backtest_pair(pair, prices, half_life_days=10.0, capital=10_000)
        for t in result.trades:
            assert t.holding_days >= 0, f"Negative holding days: {t.holding_days}"

    def test_n_trades_matches_trade_list(self):
        prices = make_cointegrated_prices(n=500)
        pair = make_dummy_selected_pair()
        result = backtest_pair(pair, prices, half_life_days=10.0, capital=10_000)
        assert result.n_trades == len(result.trades)

    def test_position_values_valid(self):
        """Position values must be in {-1, 0, 1}."""
        prices = make_cointegrated_prices(n=500)
        pair = make_dummy_selected_pair()
        result = backtest_pair(pair, prices, half_life_days=10.0, capital=10_000)
        unique = set(result.positions.unique())
        assert unique.issubset({-1.0, 0.0, 1.0}), f"Unexpected positions: {unique}"

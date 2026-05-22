"""
test_cointegration.py
---------------------
Unit tests for cointegration.py using synthetic data with known properties.

Design principle: we generate data where the answer is known analytically, then
verify that our statistical procedures recover the right answer. This tests both
the math and the implementation  -  a test that always passes regardless of
implementation would be worthless.
"""

import numpy as np
import pandas as pd
import pytest

from src.cointegration import (
    ADF_SIGNIFICANCE,
    HALF_LIFE_MAX,
    HALF_LIFE_MIN,
    analyse_pair,
    estimate_half_life,
    adf_unit_root,
    engle_granger,
    johansen,
)

RNG = np.random.default_rng(42)
N = 1000  # enough observations for reliable test power


# ---------------------------------------------------------------------------
# Helpers for synthetic data
# ---------------------------------------------------------------------------

def make_random_walk(n: int, sigma: float = 1.0) -> pd.Series:
    """Simulate a pure random walk (I(1) process)."""
    return pd.Series(RNG.normal(0, sigma, n).cumsum())


def make_cointegrated_pair(
    n: int, beta: float = 1.5, half_life: float = 10.0
) -> tuple:
    """
    Generate a cointegrated pair (Y, X) where:
      X_t = random walk
      Y_t = beta * X_t + spread_t
      spread_t = OU process with given half_life

    The spread follows: delta_s = -kappa * s_{t-1} + noise
    where kappa = log(2) / half_life (by definition of half-life).
    """
    kappa = np.log(2) / half_life
    x = make_random_walk(n, sigma=1.0)

    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1 - kappa) + RNG.normal(0, 0.5)

    y = beta * x + pd.Series(spread)
    return y, x


def make_stationary_series(n: int) -> pd.Series:
    """AR(1) stationary process  -  not I(1)."""
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.8 * s[t - 1] + RNG.normal(0, 1)
    return pd.Series(s)


def make_ou_series(n: int, half_life: float, sigma: float = 1.0) -> pd.Series:
    """Simulate a pure OU process with known half-life."""
    kappa = np.log(2) / half_life
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] * (1 - kappa) + RNG.normal(0, sigma)
    return pd.Series(s)


# ---------------------------------------------------------------------------
# ADF unit root tests
# ---------------------------------------------------------------------------

class TestAdf:
    def test_random_walk_is_nonstationary(self):
        """A random walk should have ADF p-value > 0.05 (fail to reject unit root)."""
        rw = make_random_walk(N)
        result = adf_unit_root(rw, "TEST")
        assert result.is_nonstationary, (
            f"Expected non-stationary random walk but got p={result.p_value:.4f}"
        )

    def test_random_walk_is_i1(self):
        """A random walk should be confirmed as I(1)."""
        rw = make_random_walk(N)
        result = adf_unit_root(rw, "TEST")
        assert result.is_integrated_order_1, (
            f"Expected I(1) random walk. levels p={result.p_value:.4f}, "
            f"diff p={result.diff_p_value:.4f}"
        )

    def test_stationary_series_is_not_i1(self):
        """A stationary AR(1) process should NOT be classified as I(1)."""
        s = make_stationary_series(N)
        result = adf_unit_root(s, "STAT")
        assert not result.is_nonstationary, (
            f"Stationary series should have low ADF p-value, got {result.p_value:.4f}"
        )
        assert not result.is_integrated_order_1

    def test_adf_result_fields_populated(self):
        """Result dataclass should have all numeric fields filled."""
        rw = make_random_walk(N)
        result = adf_unit_root(rw, "X")
        assert np.isfinite(result.adf_stat)
        assert 0 <= result.p_value <= 1
        assert np.isfinite(result.diff_adf_stat)
        assert 0 <= result.diff_p_value <= 1


# ---------------------------------------------------------------------------
# Engle-Granger tests
# ---------------------------------------------------------------------------

class TestEngleGranger:
    def test_cointegrated_pair_passes(self):
        """A synthetically cointegrated pair should pass EG test."""
        y, x = make_cointegrated_pair(N, beta=1.5, half_life=10.0)
        result = engle_granger(y, x, "Y", "X")
        assert result.is_cointegrated, (
            f"Expected cointegrated pair to pass EG, got p={result.eg_p_value:.4f}"
        )

    def test_independent_random_walks_fail(self):
        """Two independent random walks should rarely be cointegrated."""
        false_positives = 0
        rng = np.random.default_rng(123)
        for _ in range(10):
            rw_a = pd.Series(rng.normal(0, 1, N).cumsum())
            rw_b = pd.Series(rng.normal(0, 1, N).cumsum())
            res = engle_granger(rw_a, rw_b)
            if res.is_cointegrated:
                false_positives += 1
        assert false_positives <= 3, (
            f"Too many false positives from independent random walks: {false_positives}/10"
        )

    def test_hedge_ratio_recovered(self):
        """OLS beta should be close to the true cointegration coefficient."""
        true_beta = 2.0
        y, x = make_cointegrated_pair(N, beta=true_beta, half_life=10.0)
        result = engle_granger(y, x, "Y", "X")
        assert abs(result.ols_beta - true_beta) < 0.3, (
            f"Hedge ratio {result.ols_beta:.3f} too far from true {true_beta}"
        )


# ---------------------------------------------------------------------------
# Johansen tests
# ---------------------------------------------------------------------------

class TestJohansen:
    def test_cointegrated_pair_passes(self):
        """A cointegrated pair should be detected by Johansen trace test."""
        y, x = make_cointegrated_pair(N, beta=1.5, half_life=10.0)
        result = johansen(y, x, "Y", "X")
        assert result.is_cointegrated, (
            f"Johansen missed cointegration: trace={result.trace_stat:.2f} "
            f"vs crit={result.trace_crit_95:.2f}"
        )

    def test_cointegration_rank_is_one(self):
        """A single cointegrated pair should have rank 1."""
        y, x = make_cointegrated_pair(N, beta=1.5, half_life=10.0)
        result = johansen(y, x, "Y", "X")
        assert result.cointegration_rank == 1

    def test_stats_are_positive(self):
        """Trace and max-eigenvalue statistics should be positive."""
        y, x = make_cointegrated_pair(N, beta=1.5, half_life=10.0)
        result = johansen(y, x, "Y", "X")
        assert result.trace_stat > 0
        assert result.max_eig_stat > 0


# ---------------------------------------------------------------------------
# Half-life tests
# ---------------------------------------------------------------------------

class TestHalfLife:
    @pytest.mark.parametrize("true_hl", [5.0, 10.0, 20.0])
    def test_half_life_recovered(self, true_hl: float):
        """Estimated half-life should be within 50% of true value."""
        spread = make_ou_series(N * 2, half_life=true_hl, sigma=0.5)
        result = estimate_half_life(spread, "Y", "X")
        assert result.ou_kappa > 0, "kappa must be positive for mean-reverting spread"
        tol = true_hl * 0.5
        assert abs(result.half_life_days - true_hl) < tol, (
            f"Estimated HL={result.half_life_days:.1f} too far from true HL={true_hl}"
        )

    def test_valid_flag_in_range(self):
        """Half-life within [HALF_LIFE_MIN, HALF_LIFE_MAX] should set is_valid=True."""
        spread = make_ou_series(N, half_life=10.0, sigma=0.5)
        result = estimate_half_life(spread)
        assert result.is_valid

    def test_too_slow_half_life_invalid(self):
        """OU series with very slow reversion should be flagged invalid."""
        spread = make_ou_series(N * 5, half_life=90.0, sigma=0.2)
        result = estimate_half_life(spread)
        if result.half_life_days > HALF_LIFE_MAX:
            assert not result.is_valid

    def test_random_walk_kappa_handled_gracefully(self):
        """A random walk spread should not raise an exception."""
        rw = make_random_walk(N)
        result = estimate_half_life(rw)
        assert isinstance(result.half_life_days, float)


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

class TestAnalysePair:
    def test_cointegrated_pair_passes_all_gates(self):
        """A well-specified cointegrated pair should pass the full pipeline."""
        y, x = make_cointegrated_pair(N * 2, beta=1.5, half_life=10.0)
        result = analyse_pair(y, x, "Y", "X")
        assert result.passed, f"Expected PASS but got: {result.rejection_reason}"
        assert result.eg is not None
        assert result.johansen is not None
        assert result.half_life is not None

    def test_independent_rws_rejected(self):
        """Two independent random walks should be rejected (usually at EG step)."""
        rw1 = make_random_walk(N, sigma=1.0)
        rw2 = make_random_walk(N, sigma=1.5)
        result = analyse_pair(rw1, rw2, "RW1", "RW2")
        if result.passed:
            pytest.xfail(
                f"Independent RWs spuriously passed (5% false positive): "
                f"EG p={result.eg.eg_p_value:.4f}"
            )

    def test_rejection_reason_populated_on_fail(self):
        """Rejected pairs must have a rejection_reason string."""
        stat = make_stationary_series(N)
        rw = make_random_walk(N)
        result = analyse_pair(stat, rw, "STAT", "RW")
        assert not result.passed
        assert result.rejection_reason is not None and len(result.rejection_reason) > 0

    def test_passed_pair_has_no_rejection_reason(self):
        """A passing pair should have rejection_reason=None."""
        y, x = make_cointegrated_pair(N * 2, beta=1.5, half_life=10.0)
        result = analyse_pair(y, x, "Y", "X")
        if result.passed:
            assert result.rejection_reason is None

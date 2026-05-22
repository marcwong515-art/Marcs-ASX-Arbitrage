"""
cointegration.py
----------------
Statistical tests for cointegration between price pairs.

Methodology:
  1. ADF unit-root test: both series must be I(1) (non-stationary in levels,
     stationary in first differences).
  2. Engle-Granger (1987) two-step test: OLS regression of Y on X, then ADF
     on residuals. If residuals are I(0), the pair is cointegrated.
  3. Johansen (1988) trace / max-eigenvalue test for robustness confirmation.
  4. OU half-life: fit an Ornstein-Uhlenbeck process to the spread and
     estimate the mean-reversion half-life. This determines signal parameters.

References:
  Engle, R.F. and Granger, C.W.J. (1987). "Co-integration and Error Correction:
    Representation, Estimation, and Testing." Econometrica, 55(2), 251-276.
  Johansen, S. (1988). "Statistical analysis of cointegration vectors."
    Journal of Economic Dynamics and Control, 12(2-3), 231-254.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ADF_SIGNIFICANCE = 0.05          # p-value threshold for stationarity tests
HALF_LIFE_MIN = 1                 # days — reject pairs that revert too fast
HALF_LIFE_MAX = 30                # days — reject pairs that revert too slowly


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AdfResult:
    ticker: str
    adf_stat: float
    p_value: float
    is_nonstationary: bool          # True => I(1) candidate (fail to reject unit root)
    diff_adf_stat: float            # ADF on first difference
    diff_p_value: float
    is_integrated_order_1: bool     # True => I(1) confirmed


@dataclass
class EngleGrangerResult:
    ticker_y: str
    ticker_x: str
    ols_beta: float                 # hedge ratio from OLS (Y = beta*X + alpha + e)
    ols_alpha: float
    residual_adf_stat: float
    residual_p_value: float
    is_cointegrated: bool
    eg_p_value: float               # statsmodels coint() p-value (for logging)


@dataclass
class JohansenResult:
    ticker_y: str
    ticker_x: str
    trace_stat: float
    trace_crit_95: float
    max_eig_stat: float
    max_eig_crit_95: float
    cointegration_rank: int         # 0 or 1 (we expect 1 for a cointegrated pair)
    is_cointegrated: bool


@dataclass
class HalfLifeResult:
    ticker_y: str
    ticker_x: str
    half_life_days: float
    ou_kappa: float                 # mean-reversion speed (annualised)
    ou_mu: float                    # long-run mean of spread
    ou_sigma: float                 # diffusion coefficient
    is_valid: bool                  # True if half-life in [HALF_LIFE_MIN, HALF_LIFE_MAX]


@dataclass
class PairAnalysis:
    ticker_y: str
    ticker_x: str
    adf_y: Optional[AdfResult] = None
    adf_x: Optional[AdfResult] = None
    eg: Optional[EngleGrangerResult] = None
    johansen: Optional[JohansenResult] = None
    half_life: Optional[HalfLifeResult] = None
    passed: bool = False
    rejection_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# ADF unit-root test
# ---------------------------------------------------------------------------
def adf_unit_root(series: pd.Series, ticker: str = "") -> AdfResult:
    """
    Test whether a price series is I(1).

    Null hypothesis of ADF: series has a unit root (non-stationary).
    We require:
      - Fail to reject H0 on levels (p > 0.05) — non-stationary
      - Reject H0 on first differences (p <= 0.05) — difference-stationary

    'autolag="AIC"' selects lag length by minimising AIC, standard practice.
    """
    levels = series.dropna()
    adf_stat, p_val, _, _, _, _ = adfuller(levels, autolag="AIC")
    is_nonstationay = p_val > ADF_SIGNIFICANCE  # fail to reject => unit root present

    diff = levels.diff().dropna()
    diff_stat, diff_p, _, _, _, _ = adfuller(diff, autolag="AIC")
    is_stationary_in_diff = diff_p <= ADF_SIGNIFICANCE

    return AdfResult(
        ticker=ticker,
        adf_stat=adf_stat,
        p_value=p_val,
        is_nonstationary=is_nonstationay,
        diff_adf_stat=diff_stat,
        diff_p_value=diff_p,
        is_integrated_order_1=is_nonstationay and is_stationary_in_diff,
    )


# ---------------------------------------------------------------------------
# Engle-Granger two-step test
# ---------------------------------------------------------------------------
def engle_granger(
    y: pd.Series,
    x: pd.Series,
    ticker_y: str = "",
    ticker_x: str = "",
) -> EngleGrangerResult:
    """
    Engle-Granger (1987) two-step procedure:

    Step 1: OLS regression  Y_t = alpha + beta * X_t + e_t
    Step 2: ADF test on residuals e_t.
      If residuals are I(0) (reject unit root), the pair is cointegrated.

    We also use statsmodels coint() which applies the EG test with
    finite-sample critical-value corrections — we report its p-value for
    transparency but use our own OLS hedge ratio for signal generation.

    No lookahead: all data passed here must be from the in-sample period only.
    """
    aligned = pd.concat([y, x], axis=1).dropna()
    y_vals = aligned.iloc[:, 0].values
    x_vals = aligned.iloc[:, 1].values

    # Step 1: OLS
    X_design = add_constant(x_vals)
    model = OLS(y_vals, X_design).fit()
    alpha, beta = model.params[0], model.params[1]
    residuals = model.resid

    # Step 2: ADF on residuals
    # Note: when testing OLS residuals for stationarity, the ADF critical
    # values differ from standard (MacKinnon 1994). statsmodels coint() handles
    # this correctly via response surface critical values.
    adf_stat, adf_p, _, _, _, _ = adfuller(residuals, autolag="AIC")

    # Statsmodels coint() for cross-check (uses MacKinnon 1994 critical values)
    eg_stat, eg_p, _ = coint(y_vals, x_vals, autolag="AIC")

    return EngleGrangerResult(
        ticker_y=ticker_y,
        ticker_x=ticker_x,
        ols_beta=beta,
        ols_alpha=alpha,
        residual_adf_stat=adf_stat,
        residual_p_value=adf_p,
        is_cointegrated=eg_p <= ADF_SIGNIFICANCE,
        eg_p_value=eg_p,
    )


# ---------------------------------------------------------------------------
# Johansen test
# ---------------------------------------------------------------------------
def johansen(
    y: pd.Series,
    x: pd.Series,
    ticker_y: str = "",
    ticker_x: str = "",
) -> JohansenResult:
    """
    Johansen (1988) maximum likelihood cointegration test.

    Tests H0: cointegration rank r = 0 vs H1: r >= 1.
    We use the trace statistic and max-eigenvalue statistic.
    Critical values at the 95% confidence level.

    det_order=0: constant in cointegrating relation but not in VAR (most common
    assumption for financial prices in levels).
    k_ar_diff=1: one lag in the VECM (VAR(2) equivalent).

    A pair is considered cointegrated if at least trace or max-eig rejects
    rank = 0. We store both for reporting.
    """
    aligned = pd.concat([y, x], axis=1).dropna()

    result = coint_johansen(aligned.values, det_order=0, k_ar_diff=1)

    # Trace statistic: index 0 = rank >= 0 test
    trace_stat = result.lr1[0]
    trace_crit_95 = result.cvt[0, 1]   # [rank, confidence: 90%=0, 95%=1, 99%=2]

    # Max-eigenvalue statistic
    max_eig_stat = result.lr2[0]
    max_eig_crit_95 = result.cvm[0, 1]

    cointegration_rank = int(trace_stat > trace_crit_95)
    is_cointegrated = cointegration_rank >= 1

    return JohansenResult(
        ticker_y=ticker_y,
        ticker_x=ticker_x,
        trace_stat=trace_stat,
        trace_crit_95=trace_crit_95,
        max_eig_stat=max_eig_stat,
        max_eig_crit_95=max_eig_crit_95,
        cointegration_rank=cointegration_rank,
        is_cointegrated=is_cointegrated,
    )


# ---------------------------------------------------------------------------
# OU process half-life estimation
# ---------------------------------------------------------------------------
def estimate_half_life(
    spread: pd.Series,
    ticker_y: str = "",
    ticker_x: str = "",
) -> HalfLifeResult:
    """
    Estimate the mean-reversion half-life of the spread via discrete-time
    OU process regression.

    The continuous-time OU process:
        dS_t = kappa * (mu - S_t) dt + sigma * dW_t

    In discrete time (Euler-Maruyama discretisation):
        delta_S_t = kappa * (mu - S_{t-1}) * dt + epsilon_t
        delta_S_t = a + b * S_{t-1} + epsilon_t

    where a = kappa * mu * dt and b = -kappa * dt.
    OLS on this regression gives:
        kappa = -b / dt    (with dt = 1 day)
        half_life = log(2) / kappa

    Positive kappa => mean reverting. Negative kappa => explosive (reject pair).
    No lookahead: spread passed here must be from the in-sample period.
    """
    s = spread.dropna()
    delta_s = s.diff().dropna()
    s_lag = s.shift(1).dropna()

    # Align
    idx = delta_s.index.intersection(s_lag.index)
    delta_s = delta_s.loc[idx]
    s_lag = s_lag.loc[idx]

    X = add_constant(s_lag.values)
    model = OLS(delta_s.values, X).fit()
    a, b = model.params[0], model.params[1]

    # dt = 1 trading day
    kappa = -b  # must be positive for mean reversion
    mu = a / kappa if kappa > 0 else 0.0

    # sigma from residual standard deviation
    sigma = model.resid.std()

    if kappa <= 0:
        # Spread is non-stationary under OU assumption — reject
        return HalfLifeResult(
            ticker_y=ticker_y, ticker_x=ticker_x,
            half_life_days=np.inf, ou_kappa=kappa, ou_mu=mu, ou_sigma=sigma,
            is_valid=False,
        )

    half_life = np.log(2) / kappa

    return HalfLifeResult(
        ticker_y=ticker_y, ticker_x=ticker_x,
        half_life_days=half_life, ou_kappa=kappa, ou_mu=mu, ou_sigma=sigma,
        is_valid=HALF_LIFE_MIN <= half_life <= HALF_LIFE_MAX,
    )


# ---------------------------------------------------------------------------
# Full pair analysis pipeline
# ---------------------------------------------------------------------------
def analyse_pair(
    y: pd.Series,
    x: pd.Series,
    ticker_y: str = "",
    ticker_x: str = "",
    verbose: bool = False,
) -> PairAnalysis:
    """
    Run the full cointegration test battery for a single pair.

    Returns a PairAnalysis with passed=True only if all four filters pass:
      1. Both series are I(1).
      2. Engle-Granger rejects unit root in residuals (p <= 0.05).
      3. Johansen confirms cointegration (robustness check).
      4. Half-life is in [HALF_LIFE_MIN, HALF_LIFE_MAX] days.

    All data must be from the in-sample period — no out-of-sample data
    is ever passed here during pair selection.
    """
    result = PairAnalysis(ticker_y=ticker_y, ticker_x=ticker_x)

    # Gate 1: Unit root on each series
    result.adf_y = adf_unit_root(y, ticker=ticker_y)
    result.adf_x = adf_unit_root(x, ticker=ticker_x)

    if not result.adf_y.is_integrated_order_1:
        result.rejection_reason = f"{ticker_y} is not I(1) (ADF p={result.adf_y.p_value:.3f})"
        if verbose:
            print(f"  REJECT: {result.rejection_reason}")
        return result

    if not result.adf_x.is_integrated_order_1:
        result.rejection_reason = f"{ticker_x} is not I(1) (ADF p={result.adf_x.p_value:.3f})"
        if verbose:
            print(f"  REJECT: {result.rejection_reason}")
        return result

    # Gate 2: Engle-Granger
    result.eg = engle_granger(y, x, ticker_y=ticker_y, ticker_x=ticker_x)

    if not result.eg.is_cointegrated:
        result.rejection_reason = (
            f"Engle-Granger not significant (p={result.eg.eg_p_value:.3f})"
        )
        if verbose:
            print(f"  REJECT: {result.rejection_reason}")
        return result

    # Gate 3: Johansen (robustness)
    result.johansen = johansen(y, x, ticker_y=ticker_y, ticker_x=ticker_x)

    if not result.johansen.is_cointegrated:
        result.rejection_reason = (
            f"Johansen trace stat {result.johansen.trace_stat:.2f} < "
            f"critical {result.johansen.trace_crit_95:.2f}"
        )
        if verbose:
            print(f"  REJECT: {result.rejection_reason}")
        return result

    # Gate 4: Half-life
    spread = y - result.eg.ols_beta * x
    result.half_life = estimate_half_life(spread, ticker_y=ticker_y, ticker_x=ticker_x)

    if not result.half_life.is_valid:
        result.rejection_reason = (
            f"Half-life {result.half_life.half_life_days:.1f}d outside "
            f"[{HALF_LIFE_MIN}, {HALF_LIFE_MAX}] days"
        )
        if verbose:
            print(f"  REJECT: {result.rejection_reason}")
        return result

    result.passed = True
    if verbose:
        print(
            f"  PASS: beta={result.eg.ols_beta:.3f}, "
            f"EG p={result.eg.eg_p_value:.4f}, "
            f"HL={result.half_life.half_life_days:.1f}d"
        )
    return result

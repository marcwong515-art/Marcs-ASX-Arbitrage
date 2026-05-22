# Marc's ASX Arbitrage  -  Statistical Pairs Trading

A rigorous implementation of statistical arbitrage pairs trading on ASX 50 equities,
built for a quantitative trading club application. The focus is methodological
correctness over impressive-looking results: honest in-sample / out-of-sample
separation, explicit cost modelling, and every assumption documented.

---

## What is statistical arbitrage and why pairs trading?

Statistical arbitrage exploits *relative* mispricings between assets rather than
directional views on any single asset. Pairs trading is the simplest instance:
if two stocks share a common fundamental driver (e.g., two banks competing in
the same retail lending market), their prices should move together over time.
When they temporarily diverge beyond what fundamentals explain, a mean-reverting
trade  -  long the cheaper leg, short the expensive leg  -  can profit from the
convergence.

The strategy is market-neutral by construction: the dollar exposure to the
broad market is approximately zero, so returns are driven by the *spread*
between the two legs, not by market direction. This makes it appealing as
a source of returns that is (in theory) uncorrelated with equity beta.

The academic foundation is **cointegration** (Engle & Granger, 1987; Johansen,
1988): two I(1) price series are cointegrated if a linear combination of them
is stationary. That stationary linear combination is the spread we trade.

---

## Methodology

### 1. Universe

ASX 50 constituents (hardcoded snapshot as of 2024-Q4), grouped by GICS sector.
Only same-sector pairs are tested  -  enforcing sector homogeneity reduces the
probability of spurious cointegration from unrelated macro exposures.

### 2. In-sample / out-of-sample split

| Period | Dates | Purpose |
|--------|-------|---------|
| In-sample | 2015-01-01 – 2020-12-31 | Pair selection and model fitting |
| Out-of-sample | 2021-01-01 – present | Evaluation only |

**The out-of-sample period is never touched during pair selection.**
Any pair selected using knowledge of the 2021–present period would constitute
*data snooping*  -  the researcher would be selecting pairs that happened to work
well in the evaluation period, producing artificially high out-of-sample metrics.
This is one of the most common errors in quantitative research.

### 3. Cointegration test battery

For each same-sector pair $(Y, X)$ on in-sample data:

**Gate 1  -  ADF unit root.** Both series must be I(1): non-stationary in levels,
stationary in first differences. Tested with `statsmodels.adfuller`, lag length
selected by AIC.

**Gate 2  -  Engle-Granger (1987) two-step.**

Step 1 (OLS regression):
$$Y_t = \alpha + \beta X_t + e_t$$

Step 2 (ADF on residuals):
$$\Delta e_t = a + b \cdot e_{t-1} + \sum_{i=1}^{k} c_i \Delta e_{t-i} + \varepsilon_t$$

If the residuals $e_t$ are I(0) (ADF p-value $\leq 0.05$), the pair is
cointegrated. We use `statsmodels.coint()` which applies MacKinnon (1994)
finite-sample critical values.

**Gate 3  -  Johansen (1988) trace test.** The Johansen maximum-likelihood
test provides a robustness check on the EG result. We use `det_order=0`
(constant in the cointegrating relation) and `k_ar_diff=1`. Pairs pass if
the trace statistic exceeds the 95% critical value.

**Gate 4  -  OU half-life.** The spread $S_t = Y_t - \hat{\beta} X_t$ is fit
to a discrete-time Ornstein-Uhlenbeck process:

$$\Delta S_t = a + b \cdot S_{t-1} + \varepsilon_t$$

where $\kappa = -b$ (must be positive for mean reversion) and:

$$\text{half-life} = \frac{\ln 2}{\kappa}$$

Pairs are rejected if the half-life is outside $[1, 30]$ trading days.
Too short: the spread reverts before we can trade it. Too long: the signal is
too slow to generate meaningful risk-adjusted returns and the stop-loss horizon
becomes unwieldy.

### 4. Signal generation

At each bar $t$, using only data up to bar $t-1$:

1. Estimate $\hat{\beta}_t$ from rolling 60-day OLS on $(Y, X)$.
2. Compute spread: $S_t = Y_t - \hat{\beta}_t X_t$.
3. Compute z-score: $z_t = (S_t - \mu_{60}) / \sigma_{60}$
   where $\mu_{60}$ and $\sigma_{60}$ are rolling 60-day mean and standard deviation.

Signal rules (state machine, no concurrent positions):

| Condition | Action |
|-----------|--------|
| $z_t < -2.0$ | Enter long spread (long $Y$, short $X$) |
| $z_t > +2.0$ | Enter short spread (short $Y$, long $X$) |
| $\|z_t\| < 0.5$ | Exit (mean reversion reached) |
| $\|z_t\| > 3.5$ | Exit (stop loss) |
| Held $> 2 \times \text{HL}$ bars | Exit (time stop) |

The rolling hedge ratio ensures no in-sample beta is applied to out-of-sample
prices (which would be a form of lookahead bias).

### 5. Walk-forward backtester

- Hedge ratio is refit on a rolling 60-day window at each bar.
- Position sizing: dollar-neutral ($N$ long $Y$, $N \cdot \hat{\beta}$ short $X$),
  where $N$ is scaled to target 10% annualised portfolio volatility:
  $$N = \frac{\sigma_{\text{target}} \cdot C}{\hat{\sigma}_{\text{spread}} \cdot \sqrt{252}}$$
- Transaction costs: **10 bps round-trip per leg** (5 bps each side, modelling
  commission plus half the bid-ask spread).
- Borrow cost on short leg: **50 bps per annum**, accrued daily.
- No lookahead: every computation at bar $t$ uses only data from bars $< t$.

---

## Results summary

Pair selection identified **1 cointegrated pair** from 121 same-sector candidates:
**CSL.AX / RMD.AX** (Healthcare, EG p=0.017, OU half-life=23.5 days).

Screening funnel: 121 → 108 (both I(1)) → 7 (EG p<0.05) → 3 (Johansen) → **1** (half-life 1–30d).

| Metric | In-Sample (2015–2020) | Out-of-Sample (2021–present) |
|--------|----------------------|------------------------------|
| Total Return | −$223,398 | +$14,040 |
| Ann. Return | −$37,013/yr | +$2,598/yr |
| Sharpe Ratio | **−0.57** | **+0.04** |
| Sortino Ratio | −0.49 | +0.04 |
| Max Drawdown | −$381,009 | −$154,363 |
| Win Rate | 33.3% | 47.4% |
| Profit Factor | 0.53 | 1.05 |
| # Trades | 24 | 19 |
| Avg Holding Period | 24.4 days | 31.6 days |

*All figures on $1,000,000 notional, 10 bps round-trip costs, 50 bps borrow.*

**Honest assessment**: the in-sample performance is negative  -  the CSL/RMD spread
did not behave consistently over 2015–2020. The out-of-sample is marginally positive
(Sharpe +0.04, profit factor 1.05), which is not investable on its own. The value of
this study lies in the methodology, not the strategy P&L: the rigorous cointegration
filters are working correctly (121 candidates collapsed to 1), and the IS/OOS split
prevents data snooping. A stronger ASX pairs universe would require either a larger
universe, relaxed sector constraints, or a shorter IS window.

---

## Limitations

**Survivorship bias.** The universe is the *current* ASX 50 membership. Stocks
delisted or removed before 2024 are absent. Survivors tend to be more stable
businesses with more persistent price relationships, so real-world results
would likely be weaker than reported here.

**No intraday execution model.** PnL is computed on daily closing prices,
assuming we can transact at the close. Market impact, execution timing,
and partial fills are ignored. For strategies trading multi-hundred-thousand-dollar
notionals, market impact would materially increase costs.

**Assumed borrow availability.** The 50 bps borrow cost is a rough estimate for
liquid large-caps. Hard-to-borrow stocks can cost 200–500 bps or more and
may be unavailable entirely.

**Static universe.** ASX 50 membership changes approximately twice a year.
A live strategy would need to handle additions (new pairs to test) and
deletions (open positions that can no longer be maintained).

**Single cointegrating vector.** The EG test assumes one cointegrating vector
between $Y$ and $X$. Sector co-movements are governed by multiple common
factors; this is a deliberate simplification.

**Structural breaks.** COVID-19 (2020) and the subsequent rate tightening cycle
altered credit spreads, commodity prices, and equity correlations. Pairs that
cointegrated strongly in 2015–2020 may have experienced permanent structural
breaks  -  which is precisely why the out-of-sample evaluation matters.

**No ML or data mining.** This project is explicitly classical econometrics.
There is no hyperparameter search across signal thresholds, no feature
engineering, and no model selection beyond the cointegration tests described.
The entry/exit thresholds (2.0, 0.5, 3.5 sigma) are standard values from the
literature (Gatev, Goetzmann & Rouwenhorst, 2006).

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Fetch and cache data

```bash
python3 src/data_loader.py
```

This downloads daily adjusted closes for the ASX 50 universe from Yahoo Finance
and caches them to `data/asx50_prices.parquet`. Subsequent runs load from cache.

### 3. Run the test suite

```bash
python3 -m pytest tests/ -v
```

All 57 tests should pass. The tests use synthetic data with known properties
so you can verify correctness without fetching live data.

### 4. Run the notebooks in order

```bash
cd notebooks
jupyter notebook
```

Run in order:
1. `01_universe_and_pairs.ipynb`  -  pair selection funnel
2. `02_cointegration_analysis.ipynb`  -  deep-dive on top pair
3. `03_backtest_results.ipynb`  -  equity curves, metrics, honest discussion

Notebook 1 saves the selected pairs to `data/selected_pairs.pkl`; Notebooks 2
and 3 load from that file.

---

## Repository structure

```
asx-pairs-trading/
├── src/
│   ├── data_loader.py        # yfinance fetch + parquet cache, IS/OOS split
│   ├── cointegration.py      # ADF, Engle-Granger, Johansen, OU half-life
│   ├── pair_selection.py     # four-gate screening pipeline
│   ├── signals.py            # rolling beta, z-score, position state machine
│   ├── backtester.py         # walk-forward backtester with explicit costs
│   └── metrics.py            # Sharpe, Sortino, drawdown, plots
├── notebooks/
│   ├── 01_universe_and_pairs.ipynb
│   ├── 02_cointegration_analysis.ipynb
│   └── 03_backtest_results.ipynb
├── tests/
│   ├── test_cointegration.py # 20 tests on synthetic cointegrated series
│   ├── test_signals.py       # 18 tests including state machine edge cases
│   └── test_backtester.py    # 19 tests with hand-calculated cost verification
├── data/                     # gitignored  -  parquet cache lives here
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## References

- Engle, R.F. and Granger, C.W.J. (1987). "Co-integration and Error Correction:
  Representation, Estimation, and Testing." *Econometrica*, 55(2), 251–276.
- Johansen, S. (1988). "Statistical analysis of cointegration vectors."
  *Journal of Economic Dynamics and Control*, 12(2–3), 231–254.
- MacKinnon, J.G. (1994). "Approximate asymptotic distribution functions for
  unit-root and cointegration tests." *Journal of Business & Economic Statistics*,
  12(2), 167–176.
- Gatev, E., Goetzmann, W.N., and Rouwenhorst, K.G. (2006). "Pairs trading:
  Performance of a relative-value arbitrage rule." *Review of Financial Studies*,
  19(3), 797–827.

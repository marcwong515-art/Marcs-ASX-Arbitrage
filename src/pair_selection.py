"""
pair_selection.py
-----------------
Screens the ASX 50 universe to find cointegrated pairs suitable for
statistical arbitrage.

Pipeline (each step narrows the candidate set):
  Stage 0 — Sector filter:   only test pairs within the same GICS sector.
  Stage 1 — ADF filter:      both tickers must be I(1).
  Stage 2 — Engle-Granger:   residuals must be stationary (p <= 0.05).
  Stage 3 — Johansen:        trace statistic must exceed 95% critical value.
  Stage 4 — Half-life:       OU half-life must be in [1, 30] trading days.
  Stage 5 — Ranking:         surviving pairs ranked by EG p-value, then half-life.
                             Top 5 selected for backtesting.

CRITICAL: This module receives only in-sample price data. Out-of-sample data
is never passed here. Allowing any out-of-sample signal to influence pair
selection would constitute look-ahead bias (data snooping).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.cointegration import PairAnalysis, analyse_pair
from src.data_loader import TICKER_SECTOR

logger = logging.getLogger(__name__)

TOP_N_PAIRS = 5  # pairs selected for backtesting


@dataclass
class SelectionFunnel:
    """Tracks how many pairs survive each filter stage."""
    stage_0_sector_pairs: int = 0
    stage_1_both_i1: int = 0
    stage_2_eg_pass: int = 0
    stage_3_johansen_pass: int = 0
    stage_4_halflife_pass: int = 0
    selected: int = 0

    def summary(self) -> str:
        lines = [
            "=== Pair Selection Funnel ===",
            f"  Stage 0  Same-sector pairs:      {self.stage_0_sector_pairs}",
            f"  Stage 1  Both I(1) (ADF):        {self.stage_1_both_i1}",
            f"  Stage 2  Engle-Granger (p<0.05): {self.stage_2_eg_pass}",
            f"  Stage 3  Johansen confirmed:      {self.stage_3_johansen_pass}",
            f"  Stage 4  Half-life [1,30] days:   {self.stage_4_halflife_pass}",
            f"  Selected for backtest:             {self.selected}",
        ]
        return "\n".join(lines)


@dataclass
class SelectedPair:
    analysis: PairAnalysis
    rank: int
    eg_p_value: float
    half_life_days: float
    ols_beta: float
    ols_alpha: float


def _same_sector_pairs(
    tickers: list[str],
    sector_map: dict[str, str],
) -> list[tuple[str, str]]:
    """
    Return all (y, x) ordered pairs where both tickers share a GICS sector.
    We test both orderings (y,x) and (x,y) because EG is not symmetric —
    the hedge ratio changes with direction. We deduplicate after selection.
    """
    pairs = []
    by_sector: dict[str, list[str]] = {}
    for t in tickers:
        sector = sector_map.get(t)
        if sector is None:
            continue
        by_sector.setdefault(sector, []).append(t)

    for sector, members in by_sector.items():
        for a, b in itertools.combinations(members, 2):
            pairs.append((a, b))  # we test one direction; EG tries Y~X only
    return pairs


def select_pairs(
    prices_insample: pd.DataFrame,
    sector_map: Optional[dict[str, str]] = None,
    top_n: int = TOP_N_PAIRS,
    verbose: bool = True,
) -> tuple[list[SelectedPair], SelectionFunnel, list[PairAnalysis]]:
    """
    Run the full pair selection pipeline on in-sample prices.

    Parameters
    ----------
    prices_insample : DataFrame of shape (dates, tickers), in-sample ONLY.
    sector_map      : {ticker: sector}. Defaults to TICKER_SECTOR from data_loader.
    top_n           : Number of pairs to select.
    verbose         : Print progress if True.

    Returns
    -------
    selected        : List of SelectedPair (length <= top_n), ranked best first.
    funnel          : SelectionFunnel with count at each stage.
    all_analyses    : Full list of PairAnalysis objects (passing and rejected),
                      useful for the screening funnel notebook.

    No lookahead guarantee: prices_insample must be sliced by the caller
    using split_in_out_of_sample() before calling this function.
    """
    if sector_map is None:
        sector_map = TICKER_SECTOR

    available = [t for t in prices_insample.columns if t in sector_map]
    funnel = SelectionFunnel()
    all_analyses: list[PairAnalysis] = []

    candidate_pairs = _same_sector_pairs(available, sector_map)
    funnel.stage_0_sector_pairs = len(candidate_pairs)

    if verbose:
        print(f"\n[pair_selection] Universe: {len(available)} tickers "
              f"with {len(candidate_pairs)} same-sector candidate pairs\n")

    passing: list[PairAnalysis] = []

    for i, (ticker_y, ticker_x) in enumerate(candidate_pairs):
        y = prices_insample[ticker_y].dropna()
        x = prices_insample[ticker_x].dropna()

        # Need overlapping history of at least 252 trading days (1 year)
        common = y.index.intersection(x.index)
        if len(common) < 252:
            reason = f"insufficient overlap ({len(common)} days)"
            a = PairAnalysis(ticker_y=ticker_y, ticker_x=ticker_x,
                             passed=False, rejection_reason=reason)
            all_analyses.append(a)
            if verbose:
                print(f"  [{i+1}/{len(candidate_pairs)}] {ticker_y}/{ticker_x} "
                      f"SKIP: {reason}")
            continue

        y_aligned = y.loc[common]
        x_aligned = x.loc[common]

        if verbose:
            print(f"  [{i+1}/{len(candidate_pairs)}] "
                  f"{ticker_y}/{ticker_x} testing ...", end=" ")

        analysis = analyse_pair(
            y_aligned, x_aligned,
            ticker_y=ticker_y, ticker_x=ticker_x,
            verbose=False,
        )
        all_analyses.append(analysis)

        # Update funnel counts based on how far the pair got
        if analysis.adf_y is not None and analysis.adf_x is not None:
            if (analysis.adf_y.is_integrated_order_1
                    and analysis.adf_x.is_integrated_order_1):
                funnel.stage_1_both_i1 += 1

        if analysis.eg is not None:
            if analysis.eg.is_cointegrated:
                funnel.stage_2_eg_pass += 1

        if analysis.johansen is not None:
            if analysis.johansen.is_cointegrated:
                funnel.stage_3_johansen_pass += 1

        if analysis.half_life is not None:
            if analysis.half_life.is_valid:
                funnel.stage_4_halflife_pass += 1

        if analysis.passed:
            passing.append(analysis)
            if verbose:
                print(
                    f"PASS  EG p={analysis.eg.eg_p_value:.4f}  "
                    f"HL={analysis.half_life.half_life_days:.1f}d  "
                    f"beta={analysis.eg.ols_beta:.3f}"
                )
        else:
            if verbose:
                print(f"REJECT: {analysis.rejection_reason}")

    if verbose:
        print(f"\n[pair_selection] {len(passing)} pairs passed all filters.")

    # Rank by EG p-value (ascending), then half-life (ascending = faster reversion)
    passing.sort(
        key=lambda a: (a.eg.eg_p_value, a.half_life.half_life_days)
    )

    top = passing[:top_n]
    funnel.selected = len(top)

    selected_pairs = [
        SelectedPair(
            analysis=a,
            rank=rank + 1,
            eg_p_value=a.eg.eg_p_value,
            half_life_days=a.half_life.half_life_days,
            ols_beta=a.eg.ols_beta,
            ols_alpha=a.eg.ols_alpha,
        )
        for rank, a in enumerate(top)
    ]

    if verbose:
        print(f"\n{funnel.summary()}\n")
        print("=== Selected Pairs ===")
        for sp in selected_pairs:
            print(
                f"  #{sp.rank}  {sp.analysis.ticker_y}/{sp.analysis.ticker_x}  "
                f"EG p={sp.eg_p_value:.4f}  HL={sp.half_life_days:.1f}d  "
                f"beta={sp.ols_beta:.3f}"
            )

    return selected_pairs, funnel, all_analyses


def funnel_dataframe(all_analyses: list[PairAnalysis]) -> pd.DataFrame:
    """
    Convert the list of PairAnalysis results into a tidy DataFrame for
    display in the notebook screening funnel table.
    """
    rows = []
    for a in all_analyses:
        row = {
            "ticker_y": a.ticker_y,
            "ticker_x": a.ticker_x,
            "passed": a.passed,
            "rejection_reason": a.rejection_reason or "",
        }
        if a.adf_y is not None:
            row["adf_y_p"] = round(a.adf_y.p_value, 4)
            row["adf_y_i1"] = a.adf_y.is_integrated_order_1
        if a.adf_x is not None:
            row["adf_x_p"] = round(a.adf_x.p_value, 4)
            row["adf_x_i1"] = a.adf_x.is_integrated_order_1
        if a.eg is not None:
            row["eg_p_value"] = round(a.eg.eg_p_value, 4)
            row["eg_passed"] = a.eg.is_cointegrated
            row["ols_beta"] = round(a.eg.ols_beta, 4)
        if a.johansen is not None:
            row["johansen_trace"] = round(a.johansen.trace_stat, 2)
            row["johansen_crit"] = round(a.johansen.trace_crit_95, 2)
            row["johansen_passed"] = a.johansen.is_cointegrated
        if a.half_life is not None:
            row["half_life_days"] = round(a.half_life.half_life_days, 1)
            row["hl_valid"] = a.half_life.is_valid
        rows.append(row)

    return pd.DataFrame(rows)

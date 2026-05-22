"""
data_loader.py
--------------
Fetches daily adjusted close prices for the ASX 50 universe from yfinance
and caches them to parquet. Re-fetches only if the cache is absent.

Universe: ASX 50 constituents as of 2024, grouped by GICS sector so that
downstream pair selection can enforce sector homogeneity.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Hardcoded ASX 50 universe with GICS sector labels.
# Source: ASX website (accessed 2024-Q4). Survivorship bias note: this is a
# snapshot of *current* constituents, so stocks that were delisted or
# removed before 2024 are not represented. This is a known limitation —
# see README Limitations section.
# ---------------------------------------------------------------------------
ASX50_UNIVERSE: dict[str, list[str]] = {
    "Financials": [
        "CBA.AX",   # Commonwealth Bank
        "WBC.AX",   # Westpac
        "ANZ.AX",   # ANZ Group
        "NAB.AX",   # National Australia Bank
        "MQG.AX",   # Macquarie Group
        "SUN.AX",   # Suncorp Group
        "QBE.AX",   # QBE Insurance
        "IAG.AX",   # Insurance Australia Group
        "AMP.AX",   # AMP
        "ASX.AX",   # ASX Limited
    ],
    "Materials": [
        "BHP.AX",   # BHP Group
        "RIO.AX",   # Rio Tinto
        "FMG.AX",   # Fortescue
        "NCM.AX",   # Newcrest Mining (merged into Newmont 2023; use NST as replacement)
        "NST.AX",   # Northern Star Resources
        "S32.AX",   # South32
        "AWC.AX",   # Alumina
        "OZL.AX",   # OZ Minerals (delisted 2023; included for history)
        "MIN.AX",   # Mineral Resources
        "LYC.AX",   # Lynas Rare Earths
    ],
    "Energy": [
        "WDS.AX",   # Woodside Energy
        "STO.AX",   # Santos
        "ORG.AX",   # Origin Energy
        "AGL.AX",   # AGL Energy
        "WPL.AX",   # Woodside Petroleum (pre-merger ticker, keep for legacy data)
    ],
    "ConsumerDiscretionary": [
        "WES.AX",   # Wesfarmers
        "WOW.AX",   # Woolworths Group (note: defensive but classified discretionary here via GIC)
        "COL.AX",   # Coles Group
        "JBH.AX",   # JB Hi-Fi
        "HVN.AX",   # Harvey Norman
        "TWE.AX",   # Treasury Wine Estates
    ],
    "ConsumerStaples": [
        "COH.AX",   # Cochlear
        "A2M.AX",   # a2 Milk
        "BKL.AX",   # Blackmores
    ],
    "Healthcare": [
        "CSL.AX",   # CSL Limited
        "RMD.AX",   # ResMed
        "SHL.AX",   # Sonic Healthcare
        "RHC.AX",   # Ramsay Health Care
        "FPH.AX",   # Fisher & Paykel Healthcare
    ],
    "Industrials": [
        "TCL.AX",   # Transurban Group
        "SYD.AX",   # Sydney Airport (delisted 2022; historical data available)
        "AZJ.AX",   # Aurizon Holdings
        "QAN.AX",   # Qantas Airways
        "ALX.AX",   # Atlas Arteria
    ],
    "RealEstate": [
        "GMG.AX",   # Goodman Group
        "SCG.AX",   # Scentre Group
        "DXS.AX",   # Dexus
        "MGR.AX",   # Mirvac Group
        "GPT.AX",   # GPT Group
    ],
    "Technology": [
        "XRO.AX",   # Xero
        "WTC.AX",   # WiseTech Global
        "CPU.AX",   # Computershare
        "APX.AX",   # Appen
    ],
    "Utilities": [
        "APA.AX",   # APA Group
    ],
    "Communication": [
        "TLS.AX",   # Telstra
        "TPG.AX",   # TPG Telecom
    ],
}

# Flat list of all tickers for bulk download
ALL_TICKERS: list[str] = [t for tickers in ASX50_UNIVERSE.values() for t in tickers]

# Reverse lookup: ticker -> sector
TICKER_SECTOR: dict[str, str] = {
    t: sector
    for sector, tickers in ASX50_UNIVERSE.items()
    for t in tickers
}

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_FILE = DATA_DIR / "asx50_prices.parquet"

START_DATE = "2015-01-01"
# End date is open (None) so yfinance fetches up to the most recent trading day.
END_DATE: Optional[str] = None


def load_prices(
    force_refresh: bool = False,
    start: str = START_DATE,
    end: Optional[str] = END_DATE,
    cache_path: Path = CACHE_FILE,
) -> pd.DataFrame:
    """
    Return a DataFrame of daily adjusted close prices, shape (dates, tickers).

    Loads from parquet cache if it exists; otherwise downloads from yfinance
    and writes the cache. Set force_refresh=True to re-download.

    No lookahead: the cache file stores raw price history. The caller is
    responsible for slicing in-sample vs out-of-sample periods.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not force_refresh and cache_path.exists():
        prices = pd.read_parquet(cache_path)
        print(f"[data_loader] Loaded {prices.shape} from cache: {cache_path}")
        return prices

    print(f"[data_loader] Downloading {len(ALL_TICKERS)} tickers from yfinance ...")
    raw = yf.download(
        tickers=ALL_TICKERS,
        start=start,
        end=end,
        auto_adjust=True,   # gives adjusted close directly in 'Close' column
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns (field, ticker) when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker edge case (shouldn't happen here)
        prices = raw[["Close"]]
        prices.columns = ALL_TICKERS[:1]

    # Keep only tickers that have at least some data
    prices = prices.dropna(axis=1, how="all")

    # Forward-fill small gaps (e.g. staggered ASX holidays), then drop leading NaN rows
    prices = prices.ffill().dropna(how="all")

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"

    prices.to_parquet(cache_path)
    print(f"[data_loader] Cached {prices.shape} to {cache_path}")
    return prices


def get_sector_map() -> dict[str, str]:
    """Return {ticker: sector} for the full universe."""
    return TICKER_SECTOR.copy()


def get_universe_by_sector() -> dict[str, list[str]]:
    """Return {sector: [tickers]} for the full universe."""
    return {k: list(v) for k, v in ASX50_UNIVERSE.items()}


def split_in_out_of_sample(
    prices: pd.DataFrame,
    in_sample_end: str = "2020-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split price history into in-sample and out-of-sample DataFrames.

    in_sample  : 2015-01-01 – in_sample_end (used for pair selection + model fitting)
    out_of_sample: in_sample_end+1 – most recent date (used ONLY for evaluation)

    The out-of-sample period is NEVER used for pair selection decisions. Violating
    this would constitute data snooping / lookahead bias.
    """
    cutoff = pd.Timestamp(in_sample_end)
    in_sample = prices.loc[prices.index <= cutoff].copy()
    out_of_sample = prices.loc[prices.index > cutoff].copy()
    return in_sample, out_of_sample


if __name__ == "__main__":
    prices = load_prices()
    print(prices.tail())
    print(f"\nTickers with data: {list(prices.columns)}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

"""Batch fetcher for financial statements (income, balance sheet, cashflow).

Used to compute Piotroski / Altman / Magic Formula / Beneish factors.
Pulls 2 years of annual statements per ticker via yfinance, with threading
and a persistent disk cache to avoid hitting the API repeatedly.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

_CACHE_DIR = os.path.expanduser("~/.market-hub/quant_cache")


def _cache_path(ticker: str) -> str:
    return os.path.join(_CACHE_DIR, f"{ticker.upper()}.json")


def _load_cache(ticker: str, max_age_seconds: int) -> Optional[Dict]:
    p = _cache_path(ticker)
    if not os.path.exists(p):
        return None
    age = time.time() - os.path.getmtime(p)
    if age > max_age_seconds:
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(ticker: str, data: Dict) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_cache_path(ticker), "w") as f:
            json.dump(data, f, default=str)
    except Exception:
        pass


def _safe_get(stmt: pd.DataFrame, labels, col_idx: int = 0) -> Optional[float]:
    """Read a value from a yfinance statement DataFrame, trying multiple label spellings."""
    if stmt is None or not isinstance(stmt, pd.DataFrame) or stmt.empty:
        return None
    if col_idx >= len(stmt.columns):
        return None
    if isinstance(labels, str):
        labels = [labels]
    for label in labels:
        if label in stmt.index:
            try:
                val = stmt.loc[label].iloc[col_idx]
                if pd.isna(val):
                    continue
                return float(val)
            except (ValueError, TypeError):
                continue
    return None


def _extract_year(income, balance, cashflow, idx: int) -> Dict:
    """Pull one fiscal year's worth of fields. idx=0 is most recent."""
    d = {}
    # Income statement
    d["net_income"] = _safe_get(income, ["Net Income", "Net Income Common Stockholders",
                                          "Net Income From Continuing Operation Net Minority Interest"], idx)
    d["total_revenue"] = _safe_get(income, ["Total Revenue", "Operating Revenue"], idx)
    d["gross_profit"] = _safe_get(income, ["Gross Profit"], idx)
    d["ebit"] = _safe_get(income, ["EBIT", "Operating Income", "Total Operating Income As Reported"], idx)
    d["sga"] = _safe_get(income, ["Selling General And Administration",
                                    "Selling General And Administrative"], idx)
    d["depreciation"] = _safe_get(income, ["Reconciled Depreciation",
                                            "Depreciation And Amortization In Income Statement"], idx)

    # Balance sheet
    d["total_assets"] = _safe_get(balance, ["Total Assets"], idx)
    d["total_liabilities"] = _safe_get(balance, ["Total Liabilities Net Minority Interest",
                                                  "Total Liab"], idx)
    d["current_assets"] = _safe_get(balance, ["Current Assets"], idx)
    d["current_liabilities"] = _safe_get(balance, ["Current Liabilities"], idx)
    d["lt_debt"] = _safe_get(balance, ["Long Term Debt"], idx)
    d["total_debt"] = _safe_get(balance, ["Total Debt"], idx)
    d["retained_earnings"] = _safe_get(balance, ["Retained Earnings"], idx)
    d["shares_out"] = _safe_get(balance, ["Ordinary Shares Number", "Share Issued"], idx)
    d["cash"] = _safe_get(balance, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments",
                                     "Cash"], idx)
    d["receivables"] = _safe_get(balance, ["Accounts Receivable", "Receivables"], idx)
    d["net_fixed_assets"] = _safe_get(balance, ["Net PPE", "Property Plant Equipment Net"], idx)

    # Working capital
    if d["current_assets"] is not None and d["current_liabilities"] is not None:
        d["working_capital"] = d["current_assets"] - d["current_liabilities"]
        d["nwc"] = d["working_capital"]
    else:
        d["working_capital"] = None
        d["nwc"] = None

    # Cash flow
    d["cfo"] = _safe_get(cashflow, ["Operating Cash Flow",
                                     "Total Cash From Operating Activities",
                                     "Cash Flow From Continuing Operating Activities"], idx)
    return d


def fetch_one(ticker: str, max_age_hours: int = 24) -> Dict:
    """Fetch 2 years of annual statements for one ticker. Returns curr/prev/market_cap."""
    cached = _load_cache(ticker, max_age_seconds=max_age_hours * 3600)
    if cached:
        return cached

    try:
        t = yf.Ticker(ticker)
        income = t.income_stmt
        balance = t.balance_sheet
        cashflow = t.cashflow
        try:
            mcap = float(t.fast_info["marketCap"])
        except Exception:
            mcap = 0.0

        out = {
            "ticker": ticker,
            "curr": _extract_year(income, balance, cashflow, 0),
            "prev": _extract_year(income, balance, cashflow, 1),
            "market_cap": mcap,
        }
        _save_cache(ticker, out)
        return out
    except Exception as e:
        return {"ticker": ticker, "error": str(e), "curr": {}, "prev": {}, "market_cap": 0.0}


def fetch_batch(tickers: List[str], max_workers: int = 12,
                 max_age_hours: int = 24) -> Dict[str, Dict]:
    """Fetch statements for many tickers in parallel."""
    results: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one, t, max_age_hours): t for t in tickers}
        for fut in futures:
            try:
                d = fut.result()
            except Exception:
                d = {"ticker": futures[fut], "error": "fetch failed", "curr": {}, "prev": {}, "market_cap": 0.0}
            results[d["ticker"]] = d
    return results

"""Finnhub insider transactions summary.

Aggregates open-market buys/sells by corporate insiders (officers, directors,
10% holders) over a recent window. Strong evidence base — insiders' purchase
portfolios have historically beat the market by ~6–10%/yr.

Free Finnhub tier: 60 calls/min. Per-symbol calls; this module paces them.
"""
import datetime as dt
import os
import json
import time
from typing import Dict, List, Optional

import requests

FINNHUB_BASE = "https://finnhub.io/api/v1"
_CACHE_DIR = os.path.expanduser("~/.market-hub/insider_cache")


def _cache_path(ticker: str, days: int) -> str:
    return os.path.join(_CACHE_DIR, f"{ticker.upper()}_{days}d.json")


def _load_cache(ticker: str, days: int, max_age_seconds: int) -> Optional[Dict]:
    p = _cache_path(ticker, days)
    if not os.path.exists(p):
        return None
    if time.time() - os.path.getmtime(p) > max_age_seconds:
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(ticker: str, days: int, data: Dict) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_cache_path(ticker, days), "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def insider_summary(symbol: str, key: str, days_back: int = 90,
                     max_age_hours: int = 12) -> Dict:
    """Return aggregate insider activity for a symbol over the last N days.

    Output:
      net_usd: buys_usd - sells_usd  (positive = net buying)
      buys_usd, sells_usd: dollar volumes
      n_buys, n_sells: count of transactions
      window_days: lookback used
    """
    empty = {"net_usd": None, "buys_usd": 0, "sells_usd": 0,
             "n_buys": 0, "n_sells": 0, "window_days": days_back}
    if not key:
        return empty

    cached = _load_cache(symbol, days_back, max_age_hours * 3600)
    if cached:
        return cached

    today = dt.date.today()
    frm = today - dt.timedelta(days=days_back)
    try:
        r = requests.get(
            f"{FINNHUB_BASE}/stock/insider-transactions",
            params={"symbol": symbol.upper(), "from": str(frm),
                    "to": str(today), "token": key},
            timeout=15,
        )
        if r.status_code == 429:
            time.sleep(5)
            r = requests.get(
                f"{FINNHUB_BASE}/stock/insider-transactions",
                params={"symbol": symbol.upper(), "from": str(frm),
                        "to": str(today), "token": key},
                timeout=15,
            )
        r.raise_for_status()
        items = (r.json() or {}).get("data", []) or []
    except Exception:
        return empty

    buys_usd = 0.0
    sells_usd = 0.0
    n_buys = 0
    n_sells = 0
    for tx in items:
        try:
            change = float(tx.get("change") or 0)
            price = float(tx.get("transactionPrice") or 0)
        except (ValueError, TypeError):
            continue
        notional = abs(change) * price
        if change > 0 and notional > 0:
            buys_usd += notional
            n_buys += 1
        elif change < 0 and notional > 0:
            sells_usd += notional
            n_sells += 1

    out = {
        "net_usd": buys_usd - sells_usd,
        "buys_usd": buys_usd,
        "sells_usd": sells_usd,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "window_days": days_back,
    }
    _save_cache(symbol, days_back, out)
    return out


def insider_summary_batch(symbols: List[str], key: str, days_back: int = 90,
                            pace_seconds: float = 1.05) -> Dict[str, Dict]:
    """Sequential per-symbol fetch with pacing for the 60-call/min free tier.

    Cache hits are returned immediately; only uncached symbols hit the API
    at ~1 per second. For 50 uncached tickers expect ~50–55 seconds.
    """
    out: Dict[str, Dict] = {}
    for i, sym in enumerate(symbols):
        # Try the cache first — no pacing needed for hits
        cached = _load_cache(sym, days_back, 12 * 3600)
        if cached is not None:
            out[sym] = cached
            continue
        out[sym] = insider_summary(sym, key, days_back=days_back)
        time.sleep(pace_seconds)
    return out

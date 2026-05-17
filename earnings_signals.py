"""Finnhub earnings-surprise signals.

Computes:
- `eps_surprise_pct`     : most recent quarter's surprise % (actual vs estimate)
- `eps_surprise_streak`  : count of consecutive positive surprises (most recent first)
- `eps_surprise_avg_4q`  : mean surprise % over last 4 quarters
- `eps_surprise_z`       : a simple z-score using the std of recent surprises

Free Finnhub tier limit is 60 calls/min — this module paces calls and caches
results on disk for 12h.
"""
import json
import os
import time
from typing import Dict, List, Optional

import requests

FINNHUB_BASE = "https://finnhub.io/api/v1"
_CACHE_DIR = os.path.expanduser("~/.market-hub/earnings_cache")


def _cache_path(ticker: str) -> str:
    return os.path.join(_CACHE_DIR, f"{ticker.upper()}.json")


def _load_cache(ticker: str, max_age_seconds: int) -> Optional[Dict]:
    p = _cache_path(ticker)
    if not os.path.exists(p):
        return None
    if time.time() - os.path.getmtime(p) > max_age_seconds:
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
            json.dump(data, f)
    except Exception:
        pass


def _summarize(items: List[Dict]) -> Dict:
    """Turn the raw Finnhub /stock/earnings list into our signal columns.

    Items come newest-first from Finnhub. Each row has:
      actual, estimate, surprise, surprisePercent, period, quarter, year
    """
    if not items:
        return {"eps_surprise_pct": None, "eps_surprise_streak": 0,
                "eps_surprise_avg_4q": None, "eps_surprise_z": None,
                "n_quarters": 0}

    # Finnhub usually returns newest-first; ensure that order by parsing period
    def _key(r):
        return r.get("period") or f"{r.get('year', 0)}-{r.get('quarter', 0):02d}"
    sorted_items = sorted(items, key=_key, reverse=True)

    # Most recent quarter's surprise %
    latest = sorted_items[0]
    latest_pct = latest.get("surprisePercent")
    try:
        latest_pct = float(latest_pct) if latest_pct is not None else None
    except (TypeError, ValueError):
        latest_pct = None

    # Streak: consecutive positive surprises starting from most recent
    streak = 0
    for r in sorted_items:
        sp = r.get("surprisePercent")
        try:
            sp = float(sp) if sp is not None else None
        except (TypeError, ValueError):
            sp = None
        if sp is not None and sp > 0:
            streak += 1
        else:
            break

    # 4-quarter average + simple z-score
    last4 = []
    for r in sorted_items[:4]:
        sp = r.get("surprisePercent")
        try:
            sp = float(sp)
            last4.append(sp)
        except (TypeError, ValueError):
            continue
    avg_4q = sum(last4) / len(last4) if last4 else None

    z = None
    if len(last4) >= 2:
        mu = avg_4q or 0
        var = sum((x - mu) ** 2 for x in last4) / max(len(last4) - 1, 1)
        sd = var ** 0.5
        if sd > 0 and latest_pct is not None:
            z = round((latest_pct - mu) / sd, 2)

    return {
        "eps_surprise_pct": round(latest_pct, 2) if latest_pct is not None else None,
        "eps_surprise_streak": streak,
        "eps_surprise_avg_4q": round(avg_4q, 2) if avg_4q is not None else None,
        "eps_surprise_z": z,
        "n_quarters": len(sorted_items),
    }


def earnings_summary(symbol: str, key: str, limit: int = 8,
                      max_age_hours: int = 12) -> Dict:
    """Pull the last `limit` quarters of EPS surprises for `symbol` from Finnhub.

    Returns a summary dict (see `_summarize`) — empty fields when key/data missing.
    """
    empty = {"eps_surprise_pct": None, "eps_surprise_streak": 0,
             "eps_surprise_avg_4q": None, "eps_surprise_z": None,
             "n_quarters": 0}
    if not key:
        return empty

    cached = _load_cache(symbol, max_age_hours * 3600)
    if cached:
        return cached

    try:
        r = requests.get(
            f"{FINNHUB_BASE}/stock/earnings",
            params={"symbol": symbol.upper(), "limit": limit, "token": key},
            timeout=15,
        )
        if r.status_code == 429:
            time.sleep(5)
            r = requests.get(
                f"{FINNHUB_BASE}/stock/earnings",
                params={"symbol": symbol.upper(), "limit": limit, "token": key},
                timeout=15,
            )
        r.raise_for_status()
        items = r.json() or []
        if not isinstance(items, list):
            items = []
    except Exception:
        return empty

    out = _summarize(items)
    _save_cache(symbol, out)
    return out


def earnings_summary_batch(symbols: List[str], key: str,
                            pace_seconds: float = 1.05) -> Dict[str, Dict]:
    """Sequential per-symbol fetch with pacing for the 60-call/min free tier.

    Disk cache hits return immediately; only uncached symbols hit the API.
    For ~50 uncached tickers expect ~50–55 seconds.
    """
    out: Dict[str, Dict] = {}
    for sym in symbols:
        cached = _load_cache(sym, 12 * 3600)
        if cached is not None:
            out[sym] = cached
            continue
        out[sym] = earnings_summary(sym, key)
        time.sleep(pace_seconds)
    return out

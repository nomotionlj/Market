"""Hyperliquid public-API adapter.

HL is fully on-chain so liquidation-relevant data is real, not modeled.
Available without auth via POST https://api.hyperliquid.xyz/info :

  - metaAndAssetCtxs : universe meta + per-coin OI, funding, mark, prev funding
  - recentTrades     : last N trades per coin (used for trade-flow / liq spotting)
  - candleSnapshot   : OHLCV bars
  - l2Book           : order-book depth

We expose helpers to fetch these and shape them into the same column names
the rest of the app uses (so the crypto-derivatives panels can show HL data
alongside OKX).
"""
from __future__ import annotations

import datetime as dt
import time
from typing import Dict, List, Optional

import pandas as pd
import requests

HL_INFO = "https://api.hyperliquid.xyz/info"


# Bar codes accepted by HL candleSnapshot
_HL_INTERVALS = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "8h": "8h", "12h": "12h",
    "1d": "1d", "3d": "3d", "1w": "1w", "1mo": "1M",
}


def _post(body: Dict, timeout: float = 12.0) -> object:
    r = requests.post(HL_INFO, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Universe + per-coin context (OI, funding, mark)
# ---------------------------------------------------------------------------

def asset_contexts() -> pd.DataFrame:
    """Return one row per HL perp with mid, mark, OI, funding, prev funding."""
    data = _post({"type": "metaAndAssetCtxs"})
    if not isinstance(data, list) or len(data) < 2:
        return pd.DataFrame()
    universe = (data[0] or {}).get("universe") or []
    ctxs = data[1] or []
    rows = []
    for u, c in zip(universe, ctxs):
        try:
            rows.append({
                "coin": u.get("name"),
                "max_leverage": u.get("maxLeverage"),
                "is_delisted": bool(u.get("isDelisted", False)),
                "mark": float(c.get("markPx") or 0) or None,
                "mid": float(c.get("midPx") or 0) or None,
                "oracle": float(c.get("oraclePx") or 0) or None,
                "open_interest": float(c.get("openInterest") or 0) or None,
                "day_volume_usd": float(c.get("dayNtlVlm") or 0) or None,
                "funding": float(c.get("funding") or 0) or None,       # latest hourly rate
                "premium": float(c.get("premium") or 0) or None,
                "prev_day_px": float(c.get("prevDayPx") or 0) or None,
            })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Funding on HL is hourly; annualize for display
    df["funding_apr"] = df["funding"] * 24 * 365 * 100  # %
    return df


def coin_context(coin: str = "BTC") -> Dict:
    """Convenience: one-row dict for `coin` from asset_contexts."""
    df = asset_contexts()
    if df.empty:
        return {}
    sub = df[df["coin"] == coin.upper()]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Recent trades (for flow + heuristic liquidation spotting)
# ---------------------------------------------------------------------------

def recent_trades(coin: str = "BTC", limit: int = 200) -> pd.DataFrame:
    data = _post({"type": "recentTrades", "coin": coin.upper()})
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    rows = []
    for t in data[:limit]:
        try:
            rows.append({
                "time": pd.to_datetime(int(t.get("time")), unit="ms"),
                "side": "buy" if t.get("side") == "B" else "sell",
                "price": float(t.get("px")),
                "size": float(t.get("sz")),
                "users": t.get("users") or [],
            })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["notional"] = df["price"] * df["size"]
    return df.sort_values("time", ascending=False).reset_index(drop=True)


def recent_large_trades(coin: str = "BTC", limit: int = 200,
                         min_notional: float = 100_000) -> pd.DataFrame:
    """Filter recent trades to ones likely meaningful (≥ $min_notional)."""
    df = recent_trades(coin=coin, limit=limit)
    if df.empty:
        return df
    return df[df["notional"] >= min_notional].reset_index(drop=True)


# ---------------------------------------------------------------------------
# OHLCV candles
# ---------------------------------------------------------------------------

def klines(coin: str = "BTC", interval: str = "1h", lookback_hours: int = 48) -> pd.DataFrame:
    bar = _HL_INTERVALS.get(interval, interval)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - lookback_hours * 3600 * 1000
    body = {
        "type": "candleSnapshot",
        "req": {"coin": coin.upper(), "interval": bar,
                "startTime": start_ms, "endTime": end_ms},
    }
    data = _post(body)
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    rows = []
    for c in data:
        try:
            rows.append({
                "time": pd.to_datetime(int(c.get("t")), unit="ms"),
                "open": float(c.get("o")),
                "high": float(c.get("h")),
                "low": float(c.get("l")),
                "close": float(c.get("c")),
                "volume": float(c.get("v")),
            })
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helper for the vision panel: short text snapshot for AI context
# ---------------------------------------------------------------------------

def coin_snapshot_text(coin: str = "BTC") -> str:
    """Compact text summary of live HL state for use in vision prompts."""
    try:
        ctx = coin_context(coin)
    except Exception as e:
        return f"(HL context fetch failed: {e})"
    if not ctx:
        return f"(HL context for {coin} unavailable)"
    parts = [f"Hyperliquid live data for {coin}:"]
    if ctx.get("mark") is not None:
        parts.append(f"  mark=${ctx['mark']:,.2f}")
    if ctx.get("mid") is not None:
        parts.append(f"  mid=${ctx['mid']:,.2f}")
    if ctx.get("open_interest") is not None:
        parts.append(f"  open_interest={ctx['open_interest']:,.0f} {coin}")
    if ctx.get("day_volume_usd") is not None:
        parts.append(f"  24h_volume=${ctx['day_volume_usd']/1e6:,.1f}M")
    if ctx.get("funding") is not None:
        parts.append(f"  funding (hourly)={ctx['funding']*100:+.4f}%  "
                      f"(APR ≈ {ctx['funding_apr']:+.1f}%)")
    if ctx.get("premium") is not None:
        parts.append(f"  premium={ctx['premium']*100:+.4f}%")
    return "\n".join(parts)

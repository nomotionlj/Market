"""Hyperliquid leaderboard — top traders by PnL / ROI / account value.

Source: https://stats-data.hyperliquid.xyz/Mainnet/leaderboard
This is the same public dataset the official HL frontend uses to render
its leaderboard page. Refreshes server-side periodically.

The payload contains ~36k traders with all-time + windowed performance:
  - day / week / month / allTime windows
  - each window has: pnl (USD), roi (fraction), vlm (volume USD)
  - plus accountValue (current portfolio $) and ethAddress

The module is cached for 5 min so repeated UI hits don't hammer the endpoint.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import pandas as pd
import requests

LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
_CACHE_FILE = os.path.expanduser("~/.market-hub/hl_leaderboard.json")
_CACHE_TTL = 5 * 60  # seconds


VALID_WINDOWS = ("day", "week", "month", "allTime")


def _load_cache() -> Optional[Dict]:
    if not os.path.exists(_CACHE_FILE):
        return None
    if time.time() - os.path.getmtime(_CACHE_FILE) > _CACHE_TTL:
        return None
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(data: Dict) -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        with open(_CACHE_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def fetch_leaderboard(use_cache: bool = True) -> List[Dict]:
    """Return the raw `leaderboardRows` list from HL's public stats endpoint."""
    if use_cache:
        cached = _load_cache()
        if cached is not None:
            return cached.get("leaderboardRows") or []
    try:
        r = requests.get(LEADERBOARD_URL, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        # On failure, fall back to any stale cache we have
        if os.path.exists(_CACHE_FILE):
            try:
                with open(_CACHE_FILE) as f:
                    return json.load(f).get("leaderboardRows") or []
            except Exception:
                pass
        return []
    _save_cache(data)
    return data.get("leaderboardRows") or []


def _pnl(row: Dict, window: str) -> float:
    for w, vals in row.get("windowPerformances", []) or []:
        if w == window:
            try:
                return float(vals.get("pnl", 0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _roi(row: Dict, window: str) -> float:
    for w, vals in row.get("windowPerformances", []) or []:
        if w == window:
            try:
                return float(vals.get("roi", 0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _vlm(row: Dict, window: str) -> float:
    for w, vals in row.get("windowPerformances", []) or []:
        if w == window:
            try:
                return float(vals.get("vlm", 0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _account_value(row: Dict) -> float:
    try:
        return float(row.get("accountValue") or 0)
    except (TypeError, ValueError):
        return 0.0


def _enrich_consistency(row: Dict) -> Dict:
    """Compute consistency proxies from the 4-window perf data."""
    perf = {w: (_pnl(row, w), _roi(row, w), _vlm(row, w))
            for w in VALID_WINDOWS}
    pnls = [perf[w][0] for w in VALID_WINDOWS]
    rois = [perf[w][1] for w in VALID_WINDOWS]
    vlms = [perf[w][2] for w in VALID_WINDOWS]

    pos_windows = sum(1 for p in pnls if p > 0)
    alltime_vlm = vlms[3]
    alltime_pnl = pnls[3]

    # Net edge in basis points per dollar traded — a real measure of skill
    net_edge_bps = (alltime_pnl / alltime_vlm * 10_000) if alltime_vlm > 0 else 0

    # Are the windows reasonably proportional, or is alltime dominated by an
    # ancient bet? Compute "recency ratio" = month_pnl / (alltime_pnl/12) — if
    # they're still making money at the same rate, it's near 1.0+.
    recency = (pnls[2] * 12 / alltime_pnl) if alltime_pnl > 0 else 0

    return {
        "pos_windows": pos_windows,
        "net_edge_bps": net_edge_bps,
        "recency_ratio": recency,
    }


def _consistency_score(row: Dict) -> float:
    """Composite 0..1 score: high = consistent winner with size.

    Weights:
      0.30 — positive windows (4/4 = full credit)
      0.20 — net edge bps (saturates at 25bps = full credit)
      0.20 — log-volume (size of operation)
      0.15 — log-account-value (size of stake)
      0.15 — recency ratio (still winning, not just past)
    """
    import math
    av = _account_value(row)
    enriched = _enrich_consistency(row)

    pos = enriched["pos_windows"] / 4.0                    # 0..1
    edge = min(max(enriched["net_edge_bps"], 0) / 25, 1)   # 0..1 at 25 bps
    vol = math.log10(max(_vlm(row, "allTime"), 1)) / 11    # 0..1 at $100B vol
    size = math.log10(max(av, 1)) / 10                     # 0..1 at $10B AV
    rec = min(max(enriched["recency_ratio"], 0), 2) / 2    # 0..1 at >=2× pace

    return round(0.30*pos + 0.20*edge + 0.20*vol + 0.15*size + 0.15*rec, 4)


def top_traders(n: int = 100, sort_by: str = "account_value",
                  window: str = "allTime",
                  min_account_value: float = 0,
                  min_pos_windows: int = 0,
                  min_alltime_vlm: float = 0,
                  use_cache: bool = True) -> pd.DataFrame:
    """Top-N traders. Adds consistency-based sorts + filters.

    sort_by:
      - 'account_value', 'pnl', 'roi', 'vlm'  (the originals)
      - 'consistency'       : composite score (positive-windows × edge × size)
      - 'net_edge_bps'      : pure $PnL per $volume (skill, not luck)
      - 'pos_windows'       : how many of the 4 windows are profitable

    min_pos_windows: filter out traders profitable in fewer than N windows.
    min_alltime_vlm: hide one-off lucky bets (require real trading volume).
    """
    rows = fetch_leaderboard(use_cache=use_cache)
    if not rows:
        return pd.DataFrame()

    if min_account_value > 0:
        rows = [r for r in rows if _account_value(r) >= min_account_value]
    if min_alltime_vlm > 0:
        rows = [r for r in rows if _vlm(r, "allTime") >= min_alltime_vlm]
    if min_pos_windows > 0:
        rows = [r for r in rows
                if _enrich_consistency(r)["pos_windows"] >= min_pos_windows]

    if sort_by == "account_value":
        rows = sorted(rows, key=_account_value, reverse=True)
    elif sort_by == "pnl":
        rows = sorted(rows, key=lambda r: _pnl(r, window), reverse=True)
    elif sort_by == "roi":
        rows = sorted(rows, key=lambda r: _roi(r, window), reverse=True)
    elif sort_by == "vlm":
        rows = sorted(rows, key=lambda r: _vlm(r, window), reverse=True)
    elif sort_by == "consistency":
        rows = sorted(rows, key=_consistency_score, reverse=True)
    elif sort_by == "net_edge_bps":
        rows = sorted(rows,
                        key=lambda r: _enrich_consistency(r)["net_edge_bps"],
                        reverse=True)
    elif sort_by == "pos_windows":
        # Tie-break by all-time PnL so 4/4 winners are ordered by size
        rows = sorted(rows,
                        key=lambda r: (_enrich_consistency(r)["pos_windows"],
                                         _pnl(r, "allTime")),
                        reverse=True)
    else:
        raise ValueError(f"Unknown sort_by: {sort_by!r}")

    top = rows[:n]

    out = []
    for r in top:
        e = _enrich_consistency(r)
        out.append({
            "address": (r.get("ethAddress") or "").lower(),
            "display_name": r.get("displayName") or "",
            "account_value": _account_value(r),
            "day_pnl": _pnl(r, "day"),
            "week_pnl": _pnl(r, "week"),
            "month_pnl": _pnl(r, "month"),
            "alltime_pnl": _pnl(r, "allTime"),
            "day_roi": _roi(r, "day"),
            "week_roi": _roi(r, "week"),
            "month_roi": _roi(r, "month"),
            "alltime_roi": _roi(r, "allTime"),
            "day_vlm": _vlm(r, "day"),
            "week_vlm": _vlm(r, "week"),
            "month_vlm": _vlm(r, "month"),
            "alltime_vlm": _vlm(r, "allTime"),
            "pos_windows": e["pos_windows"],
            "net_edge_bps": e["net_edge_bps"],
            "recency_ratio": e["recency_ratio"],
            "consistency": _consistency_score(r),
        })
    return pd.DataFrame(out)


def top_addresses(n: int = 100, sort_by: str = "account_value",
                    window: str = "allTime",
                    min_account_value: float = 0) -> List[str]:
    """Just the addresses (used as seeds for the liquidation map)."""
    df = top_traders(n=n, sort_by=sort_by, window=window,
                       min_account_value=min_account_value)
    if df.empty:
        return []
    return df["address"].tolist()

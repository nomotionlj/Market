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


def top_traders(n: int = 100, sort_by: str = "account_value",
                  window: str = "allTime",
                  min_account_value: float = 0,
                  use_cache: bool = True) -> pd.DataFrame:
    """Top-N traders sorted by:

      - 'account_value' : current portfolio $ (default — biggest current size)
      - 'pnl'           : PnL in the given window
      - 'roi'           : ROI in the given window
      - 'vlm'           : volume in the given window

    window: 'day' / 'week' / 'month' / 'allTime' (only relevant for pnl/roi/vlm)

    Returns a DataFrame with: address, account_value, day_pnl, week_pnl,
    month_pnl, alltime_pnl, day_roi_pct, week_roi_pct, ..., alltime_vlm.
    """
    rows = fetch_leaderboard(use_cache=use_cache)
    if not rows:
        return pd.DataFrame()

    if min_account_value > 0:
        rows = [r for r in rows if _account_value(r) >= min_account_value]

    if sort_by == "account_value":
        rows = sorted(rows, key=_account_value, reverse=True)
    elif sort_by == "pnl":
        rows = sorted(rows, key=lambda r: _pnl(r, window), reverse=True)
    elif sort_by == "roi":
        rows = sorted(rows, key=lambda r: _roi(r, window), reverse=True)
    elif sort_by == "vlm":
        rows = sorted(rows, key=lambda r: _vlm(r, window), reverse=True)
    else:
        raise ValueError(f"Unknown sort_by: {sort_by!r}")

    top = rows[:n]

    out = []
    for r in top:
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

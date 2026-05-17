"""Track whale-bias-by-coin over time so the UI can chart bias flips.

A "snapshot" = a row per (timestamp, coin) capturing how the top-N whales
are positioned at that moment (long $, short $, net $, # of traders, etc.)

Snapshots are appended to a single CSV at ~/.market-hub/whale_bias_history.csv.
Take them manually via the UI button, or cron the CLI:

    0 * * * * cd /Users/lj/market-hub && .venv/bin/python picker.py bias-snapshot

A few snapshots/day is plenty — bias doesn't move that fast.
"""
from __future__ import annotations

import datetime as dt
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import pandas as pd
import requests

import hl_leaderboard as lb


HL_INFO = "https://api.hyperliquid.xyz/info"
HISTORY_PATH = os.path.expanduser("~/.market-hub/whale_bias_history.csv")

COLUMNS = [
    "timestamp", "top_n", "sort_by", "window", "coin",
    "n_long", "n_short", "long_notional", "short_notional",
    "net_notional", "net_pct", "long_pnl", "short_pnl",
    "avg_long_lev", "avg_short_lev",
]


def _clearinghouse(addr: str) -> Dict:
    try:
        r = requests.post(HL_INFO,
                           json={"type": "clearinghouseState", "user": addr},
                           timeout=12)
        r.raise_for_status()
        return r.json() or {}
    except Exception:
        return {}


def _aggregate_positions(addresses: List[str],
                           max_workers: int = 20) -> pd.DataFrame:
    """Return a positions DataFrame: address, coin, side, notional, pnl, lev."""
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        states = list(ex.map(_clearinghouse, addresses))

    rows: List[Dict] = []
    for addr, state in zip(addresses, states):
        for p in state.get("assetPositions") or []:
            pos = p.get("position") or {}
            try:
                size = float(pos.get("szi") or 0)
                if size == 0:
                    continue
                rows.append({
                    "address": addr,
                    "coin": pos.get("coin"),
                    "side": "long" if size > 0 else "short",
                    "notional": float(pos.get("positionValue") or 0),
                    "pnl": float(pos.get("unrealizedPnl") or 0),
                    "leverage": float((pos.get("leverage") or {}).get("value") or 0)
                                   or None,
                })
            except (TypeError, ValueError):
                continue
    return pd.DataFrame(rows)


def take_snapshot(top_n: int = 100,
                    sort_by: str = "pnl",
                    window: str = "allTime",
                    min_account_value: float = 0,
                    history_path: str = HISTORY_PATH) -> pd.DataFrame:
    """Compute current bias per coin across the top-N whales and append a
    snapshot to the CSV.

    Returns the snapshot rows that were written.
    """
    traders = lb.top_traders(n=top_n, sort_by=sort_by, window=window,
                                min_account_value=min_account_value,
                                use_cache=True)
    if traders.empty:
        return pd.DataFrame(columns=COLUMNS)

    positions = _aggregate_positions(traders["address"].tolist())
    if positions.empty:
        return pd.DataFrame(columns=COLUMNS)

    grp = (positions.assign(notional=positions["notional"].fillna(0),
                              pnl=positions["pnl"].fillna(0))
                     .groupby(["coin", "side"], observed=True)
                     .agg(n=("notional", "size"),
                            notional=("notional", "sum"),
                            pnl=("pnl", "sum"),
                            avg_lev=("leverage", "mean"))
                     .reset_index())
    longs = grp[grp["side"] == "long"].set_index("coin")
    shorts = grp[grp["side"] == "short"].set_index("coin")
    coins = sorted(set(longs.index) | set(shorts.index))

    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = []
    for c in coins:
        l_n = int(longs.at[c, "n"]) if c in longs.index else 0
        l_not = float(longs.at[c, "notional"]) if c in longs.index else 0.0
        l_pnl = float(longs.at[c, "pnl"]) if c in longs.index else 0.0
        l_lev = float(longs.at[c, "avg_lev"]) if c in longs.index else None
        s_n = int(shorts.at[c, "n"]) if c in shorts.index else 0
        s_not = float(shorts.at[c, "notional"]) if c in shorts.index else 0.0
        s_pnl = float(shorts.at[c, "pnl"]) if c in shorts.index else 0.0
        s_lev = float(shorts.at[c, "avg_lev"]) if c in shorts.index else None
        gross = l_not + s_not
        net = l_not - s_not
        net_pct = (net / gross * 100) if gross > 0 else 0
        rows.append({
            "timestamp": ts,
            "top_n": top_n,
            "sort_by": sort_by,
            "window": window,
            "coin": c,
            "n_long": l_n, "n_short": s_n,
            "long_notional": l_not, "short_notional": s_not,
            "net_notional": net, "net_pct": net_pct,
            "long_pnl": l_pnl, "short_pnl": s_pnl,
            "avg_long_lev": l_lev, "avg_short_lev": s_lev,
        })
    out = pd.DataFrame(rows, columns=COLUMNS)

    # Append (create file with header on first write)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    header = not os.path.exists(history_path)
    out.to_csv(history_path, mode="a", header=header, index=False)
    return out


def load_history(coin: Optional[str] = None,
                   hours_back: Optional[int] = 24 * 14,
                   top_n: Optional[int] = None,
                   sort_by: Optional[str] = None,
                   history_path: str = HISTORY_PATH) -> pd.DataFrame:
    """Load snapshot history, optionally filtered."""
    if not os.path.exists(history_path):
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(history_path, parse_dates=["timestamp"])
    except Exception:
        return pd.DataFrame(columns=COLUMNS)
    if df.empty:
        return df
    # Make sure timestamp is timezone-aware
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    if hours_back:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours_back)
        df = df[df["timestamp"] >= cutoff]
    if coin:
        df = df[df["coin"] == coin]
    if top_n:
        df = df[df["top_n"] == top_n]
    if sort_by:
        df = df[df["sort_by"] == sort_by]
    return df.sort_values("timestamp").reset_index(drop=True)


def n_snapshots(history_path: str = HISTORY_PATH) -> int:
    """Number of distinct snapshot timestamps recorded."""
    if not os.path.exists(history_path):
        return 0
    try:
        df = pd.read_csv(history_path, usecols=["timestamp"])
        return df["timestamp"].nunique()
    except Exception:
        return 0


def coin_universe(history_path: str = HISTORY_PATH) -> List[str]:
    if not os.path.exists(history_path):
        return []
    try:
        df = pd.read_csv(history_path, usecols=["coin"])
        return sorted(df["coin"].dropna().unique().tolist())
    except Exception:
        return []

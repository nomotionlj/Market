"""Hyperliquid Whale Tracker — top traders, their positions, their orders.

Sub-tabs:
  🏆 Top Traders     — ranked leaderboard (PnL / AV / ROI / volume)
  📊 Positions       — filterable browser across top-N (coin, side, leverage…)
  ⚡ Open Orders     — TP / SL / limit / trigger orders for top-N
  🎯 Order Clusters  — where whales' stops & TPs are clustered by price

All data comes from HL's public endpoints. No paid feeds.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import hl_leaderboard as lb
import hyperliquid as hl
import whale_bias_history as wbh


HL_INFO = "https://api.hyperliquid.xyz/info"

SORT_LABELS = {
    "Account value (current $)": ("account_value", "allTime"),
    "All-time PnL": ("pnl", "allTime"),
    "Month PnL": ("pnl", "month"),
    "Week PnL": ("pnl", "week"),
    "Day PnL (24h)": ("pnl", "day"),
    "All-time ROI": ("roi", "allTime"),
    "Month ROI": ("roi", "month"),
    "Day ROI (24h)": ("roi", "day"),
    "All-time volume": ("vlm", "allTime"),
    "Month volume": ("vlm", "month"),
}


# ---------------------------------------------------------------------------
# Cached data fetches
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _top_traders(n: int, sort_by: str, window: str,
                   min_av: float) -> pd.DataFrame:
    return lb.top_traders(n=n, sort_by=sort_by, window=window,
                            min_account_value=min_av)


def _post(body: Dict) -> Optional[object]:
    try:
        r = requests.post(HL_INFO, json=body, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def _clearinghouse(addr: str) -> Dict:
    return _post({"type": "clearinghouseState", "user": addr}) or {}


@st.cache_data(ttl=120, show_spinner=False)
def _clearinghouse_xyz(addr: str) -> Dict:
    """Positions on the `xyz` sub-DEX — stock perps (TSLA, NVDA, AAPL, MSTR…),
    commodity perps (GOLD, SILVER, BRENTOIL…), FX perps (EUR, JPY…)."""
    return _post({"type": "clearinghouseState", "user": addr, "dex": "xyz"}) or {}


@st.cache_data(ttl=120, show_spinner=False)
def _frontend_open_orders(addr: str) -> List[Dict]:
    res = _post({"type": "frontendOpenOrders", "user": addr})
    return res if isinstance(res, list) else []


@st.cache_data(ttl=300, show_spinner=False)
def _user_fills(addr: str, limit: int = 2000) -> List[Dict]:
    """Most recent N fills for `addr`. HL caps this at 2000 newest-first.

    For positions opened more than ~2000 fills ago, Age will be — that's
    a hard API limit (HL doesn't expose older fills via this endpoint).
    """
    res = _post({"type": "userFills", "user": addr})
    if not isinstance(res, list):
        return []
    return res[:limit]


def _position_opened_ms(fills: List[Dict], coin: str, side: str) -> Optional[int]:
    """Return the timestamp (ms) of the most recent fill that opened/added
    to a position of (coin, side). 'side' is 'long' or 'short'.

    HL fills carry a `dir` field like "Open Long" / "Close Short" / "Liquidation".
    """
    target = "Open Long" if side == "long" else "Open Short"
    for f in fills:
        if (f.get("coin") == coin) and (f.get("dir") == target):
            try:
                return int(f.get("time") or 0)
            except (TypeError, ValueError):
                continue
    return None


def _fmt_age_ms(age_ms) -> str:
    # Accept None / NaN / strings / pandas NA without crashing
    if age_ms is None:
        return "—"
    try:
        age_ms = float(age_ms)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(age_ms) or age_ms <= 0:
        return "—"
    s = age_ms / 1000.0
    if s < 60:
        return f"{int(s)}s"
    m = s / 60
    if m < 60:
        return f"{int(m)}m"
    h = m / 60
    if h < 24:
        return f"{int(h)}h"
    d = h / 24
    if d < 30:
        return f"{int(d)}d"
    return f"{int(d/30)}mo"


@st.cache_data(ttl=120, show_spinner=False)
def _whale_dataset(addrs_tuple: tuple) -> Dict[str, pd.DataFrame]:
    """Pull positions + orders for every address in parallel. Cached.

    Returns:
      positions : DataFrame keyed by (address, coin)
      orders    : DataFrame of every open order across the batch
    """
    addrs = list(addrs_tuple)
    pos_rows: List[Dict] = []
    ord_rows: List[Dict] = []

    def _bundle(addr: str) -> Tuple[Dict, Dict, List[Dict], List[Dict]]:
        # 4 calls per address in parallel: main perps + xyz stock perps + orders + fills.
        # Each is independently cached so re-renders are cheap.
        return (_clearinghouse(addr),
                _clearinghouse_xyz(addr),
                _frontend_open_orders(addr),
                _user_fills(addr))

    with ThreadPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(_bundle, addrs))

    import time
    now_ms = int(time.time() * 1000)

    for addr, (state, state_xyz, orders, fills) in zip(addrs, results):
        # Positions from MAIN dex + xyz stock dex (label coin with `xyz:` prefix)
        for source_state, source_label in [(state, ""), (state_xyz, "xyz:")]:
            for p in source_state.get("assetPositions") or []:
                pos = p.get("position") or {}
                try:
                    size = float(pos.get("szi") or 0)
                    if size == 0:
                        continue
                    liq = pos.get("liquidationPx")
                    raw_coin = pos.get("coin") or ""
                    # On xyz dex the coin field already includes "xyz:" prefix
                    # for stock perps, but defensively also prefix if missing.
                    if source_label and not raw_coin.startswith("xyz:"):
                        coin = source_label + raw_coin
                    else:
                        coin = raw_coin
                    side = "long" if size > 0 else "short"
                    opened_ms = _position_opened_ms(fills, raw_coin, side)
                    age_ms = (now_ms - opened_ms) if opened_ms else None
                    pos_rows.append({
                        "address": addr,
                        "coin": coin,
                        "side": side,
                        "size": abs(size),
                        "entry": float(pos.get("entryPx") or 0) or None,
                        "notional": float(pos.get("positionValue") or 0) or None,
                        "unrealized_pnl": float(pos.get("unrealizedPnl") or 0),
                        "leverage": (pos.get("leverage") or {}).get("value"),
                        "liq_px": float(liq) if liq not in (None, "", "0") else None,
                        "opened_ms": opened_ms,
                        "age_ms": age_ms,
                    })
                except (TypeError, ValueError):
                    continue
        # Orders
        for o in orders:
            try:
                lim = float(o.get("limitPx") or 0) or None
                trig = float(o.get("triggerPx") or 0) or None
                ord_rows.append({
                    "address": addr,
                    "coin": o.get("coin"),
                    "side": "buy" if o.get("side") == "B" else "sell",
                    "order_type": o.get("orderType", "?"),
                    "limit_px": lim,
                    "trigger_px": trig,
                    "size": float(o.get("sz") or 0),
                    "is_trigger": bool(o.get("isTrigger", False)),
                    "is_position_tpsl": bool(o.get("isPositionTpsl", False)),
                    "reduce_only": bool(o.get("reduceOnly", False)),
                    "trigger_condition": o.get("triggerCondition", ""),
                    "tif": o.get("tif", ""),
                    "timestamp": o.get("timestamp"),
                })
            except (TypeError, ValueError):
                continue

    return {
        "positions": pd.DataFrame(pos_rows),
        "orders": pd.DataFrame(ord_rows),
    }


@st.cache_data(ttl=120, show_spinner=False)
def _bias_history_cached(coin: str, hours_back: int,
                          top_n: Optional[int],
                          sort_by: Optional[str]) -> pd.DataFrame:
    """Cached wrapper around wbh.load_history. The CSV file read isn't free."""
    return wbh.load_history(coin=coin, hours_back=hours_back,
                              top_n=top_n, sort_by=sort_by)


@st.cache_data(ttl=120, show_spinner=False)
def _bias_history_meta() -> Tuple[int, List[str]]:
    return wbh.n_snapshots(), wbh.coin_universe()


@st.cache_data(ttl=30, show_spinner=False)
def _live_marks() -> Dict[str, float]:
    """Cache HL mark prices for all coins (used to compute distance-from-price)."""
    try:
        df = hl.asset_contexts()
        out = {}
        for _, r in df.iterrows():
            if pd.notna(r.get("coin")) and pd.notna(r.get("mark")):
                out[str(r["coin"]).upper()] = float(r["mark"])
        return out
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_usd(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == 0:
        return "—"
    sign = "−" if v < 0 else ""
    a = abs(v)
    if a >= 1e9:
        return f"{sign}${a / 1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:.2f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:.1f}K"
    return f"{sign}${a:,.0f}"


def _fmt_pct(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v*100:+.1f}%"


def _abbrev_addr(a: str) -> str:
    if not a:
        return "—"
    return a[:6] + "…" + a[-4:]


def _classify_order(row: Dict, mark: Optional[float]) -> str:
    """Best-effort label: Entry-limit, Take-Profit, Stop-Loss, Trigger, etc."""
    is_trig = bool(row.get("is_trigger"))
    reduce = bool(row.get("reduce_only"))
    side = row.get("side")
    lim = row.get("limit_px")
    trig = row.get("trigger_px")
    px = trig if (is_trig and trig) else lim

    if not is_trig:
        return "TP Limit" if reduce else "Entry Limit"
    if mark is None or px is None:
        return "Trigger"
    if reduce:
        if side == "buy" and px > mark:
            return "Stop (short)"
        if side == "sell" and px < mark:
            return "Stop (long)"
        if side == "buy" and px < mark:
            return "TP (short)"
        if side == "sell" and px > mark:
            return "TP (long)"
    return "Trigger"


# ---------------------------------------------------------------------------
# Sub-tab renderers
# ---------------------------------------------------------------------------

def _render_top_traders(traders_df: pd.DataFrame, sort_label: str) -> None:
    df = traders_df
    total_av = df["account_value"].sum()
    total_alltime_pnl = df["alltime_pnl"].sum()
    total_day_pnl = df["day_pnl"].sum()
    total_month_vol = df["month_vlm"].sum()
    a1, a2, a3, a4 = st.columns(4)
    a1.metric(f"Top {len(df)} combined AV", _fmt_usd(total_av))
    a2.metric("Combined all-time PnL", _fmt_usd(total_alltime_pnl))
    a3.metric("Combined day PnL", _fmt_usd(total_day_pnl))
    a4.metric("Combined month volume", _fmt_usd(total_month_vol))

    st.markdown(f"##### Top {len(df)} traders · sorted by **{sort_label}**")
    show = pd.DataFrame({
        "#": range(1, len(df) + 1),
        "Address": df["address"].apply(_abbrev_addr),
        "Account $": df["account_value"].apply(_fmt_usd),
        "All-time PnL": df["alltime_pnl"].apply(_fmt_usd),
        "All-time ROI": df["alltime_roi"].apply(_fmt_pct),
        "Month PnL": df["month_pnl"].apply(_fmt_usd),
        "Week PnL": df["week_pnl"].apply(_fmt_usd),
        "Day PnL": df["day_pnl"].apply(_fmt_usd),
        "Month Vol": df["month_vlm"].apply(_fmt_usd),
    })
    st.dataframe(show, hide_index=True, use_container_width=True,
                  height=min(45 + 32 * len(show), 600))


def _render_positions(positions: pd.DataFrame, marks: Dict[str, float],
                        traders_df: pd.DataFrame) -> None:
    if positions.empty:
        st.info("No open positions across the selected top-N.")
        return

    st.markdown("##### Filter positions")
    cols = st.columns(5)
    with cols[0]:
        coins = sorted(positions["coin"].dropna().unique())
        coin_sel = st.multiselect("Coin", coins, default=[], key="wt_pos_coin")
    with cols[1]:
        side_sel = st.radio("Side", ["both", "long", "short"], horizontal=True,
                              key="wt_pos_side")
    with cols[2]:
        min_lev = st.number_input("Min leverage ×", 0.0, 50.0, 0.0, step=1.0,
                                    key="wt_pos_lev")
    with cols[3]:
        min_not = st.number_input("Min notional ($K)", 0.0, 100000.0, 0.0, step=10.0,
                                    key="wt_pos_not")
    with cols[4]:
        only_pnl = st.radio("PnL", ["any", "in profit", "in loss"], horizontal=True,
                              key="wt_pos_pnl")

    f = positions.copy()
    if coin_sel:
        f = f[f["coin"].isin(coin_sel)]
    if side_sel != "both":
        f = f[f["side"] == side_sel]
    if min_lev > 0:
        f = f[f["leverage"].fillna(0) >= min_lev]
    if min_not > 0:
        f = f[f["notional"].fillna(0) >= min_not * 1000]
    if only_pnl == "in profit":
        f = f[f["unrealized_pnl"] > 0]
    elif only_pnl == "in loss":
        f = f[f["unrealized_pnl"] < 0]
    if f.empty:
        st.warning("No positions match the filters.")
        return

    f = f.sort_values("notional", ascending=False).copy()

    # Header aggregates
    total_long = f.loc[f["side"] == "long", "notional"].fillna(0).sum()
    total_short = f.loc[f["side"] == "short", "notional"].fillna(0).sum()
    total_pnl = f["unrealized_pnl"].sum()
    net = total_long - total_short
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Filtered positions", f"{len(f)}")
    h2.metric("Total long $", _fmt_usd(total_long))
    h3.metric("Total short $", _fmt_usd(total_short))
    h4.metric(
        "Net bias",
        f"{_fmt_usd(abs(net))} {'long' if net>0 else 'short'}",
        delta=f"PnL {_fmt_usd(total_pnl)}",
    )

    def _fmt_px(v):
        """Format a price: dollar sign + thousands separators + sensible decimals.
        $1234.56  ·  $0.000123  ·  $1,234,567.89."""
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        try:
            v = float(v)
        except (TypeError, ValueError):
            return "—"
        a = abs(v)
        if a >= 100:
            return f"${v:,.2f}"
        if a >= 1:
            return f"${v:,.4f}".rstrip("0").rstrip(".")
        if a >= 0.01:
            return f"${v:,.6f}".rstrip("0").rstrip(".")
        return f"${v:.8f}".rstrip("0").rstrip(".") or "$0"

    show = pd.DataFrame({
        "Trader": f["address"].apply(_abbrev_addr),
        "Coin": f["coin"],
        "Side": f["side"].map({"long": "🟢 long", "short": "🔴 short"}),
        "Age": f.get("age_ms", pd.Series(dtype=float)).apply(_fmt_age_ms)
                  if "age_ms" in f.columns else "—",
        "Size": f["size"].apply(lambda v: f"{v:,.4f}"),
        "Entry $": f["entry"].apply(_fmt_px),
        "Notional": f["notional"].apply(_fmt_usd),
        "Unrl PnL": f["unrealized_pnl"].apply(_fmt_usd),
        "Lev": f["leverage"].apply(lambda v: f"{int(v)}×" if pd.notna(v) else "—"),
        "Liq Px": f["liq_px"].apply(_fmt_px),
    })
    st.dataframe(show, hide_index=True, use_container_width=True,
                  height=min(45 + 32 * len(show), 620))


def _render_orders(orders: pd.DataFrame, marks: Dict[str, float]) -> None:
    if orders.empty:
        st.info("No open orders across the selected top-N.")
        return

    # Pre-compute labels + distance-from-mark
    orders = orders.copy()
    orders["mark"] = orders["coin"].str.upper().map(marks)
    orders["effective_px"] = orders.apply(
        lambda r: r["trigger_px"] if (r["is_trigger"] and pd.notna(r["trigger_px"]))
        else r["limit_px"], axis=1,
    )
    orders["dist_pct"] = (orders["effective_px"] - orders["mark"]) / orders["mark"] * 100
    orders["label"] = orders.apply(lambda r: _classify_order(r.to_dict(), r.get("mark")),
                                       axis=1)
    orders["notional_est"] = (orders["effective_px"].fillna(0) * orders["size"].fillna(0))

    st.markdown("##### Filter orders")
    cols = st.columns(5)
    with cols[0]:
        coins = sorted(orders["coin"].dropna().unique())
        coin_sel = st.multiselect("Coin", coins, default=[], key="wt_ord_coin")
    with cols[1]:
        type_options = sorted(orders["label"].dropna().unique())
        type_sel = st.multiselect("Type", type_options, default=[], key="wt_ord_type")
    with cols[2]:
        side_sel = st.radio("Side", ["both", "buy", "sell"], horizontal=True,
                              key="wt_ord_side")
    with cols[3]:
        max_dist = st.slider(
            "|Dist from mark| %", 0.0, 100.0, 100.0, step=1.0,
            key="wt_ord_dist",
            help="Hide orders more than X% away from current mark.",
        )
    with cols[4]:
        min_notional = st.number_input(
            "Min notional ($K)", 0.0, 10000.0, 0.0, step=10.0,
            key="wt_ord_notional",
        )

    f = orders.copy()
    if coin_sel:
        f = f[f["coin"].isin(coin_sel)]
    if type_sel:
        f = f[f["label"].isin(type_sel)]
    if side_sel != "both":
        f = f[f["side"] == side_sel]
    f = f[f["dist_pct"].abs().fillna(999) <= max_dist]
    if min_notional > 0:
        f = f[f["notional_est"] >= min_notional * 1000]
    if f.empty:
        st.warning("No orders match the filters.")
        return

    f = f.sort_values("notional_est", ascending=False)

    # Header summary
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Orders", f"{len(f)}")
    n_trig = int(f["is_trigger"].sum())
    h2.metric("Trigger orders", f"{n_trig}", help="Stop / take-profit triggers")
    h3.metric("Reduce-only", f"{int(f['reduce_only'].sum())}",
                help="Exit orders (not entries)")
    h4.metric("Total est notional", _fmt_usd(f["notional_est"].sum()))

    show = pd.DataFrame({
        "Trader": f["address"].apply(_abbrev_addr),
        "Coin": f["coin"],
        "Type": f["label"],
        "Side": f["side"].map({"buy": "🟢 buy", "sell": "🔴 sell"}),
        "Trigger $": f["trigger_px"].apply(
            lambda v: f"${v:,.4g}" if pd.notna(v) and v > 0 else "—"),
        "Limit $": f["limit_px"].apply(
            lambda v: f"${v:,.4g}" if pd.notna(v) and v > 0 else "—"),
        "Size": f["size"].apply(lambda v: f"{v:,.4f}"),
        "Est notional": f["notional_est"].apply(_fmt_usd),
        "Mark $": f["mark"].apply(lambda v: f"${v:,.4g}" if pd.notna(v) else "—"),
        "Dist %": f["dist_pct"].apply(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"),
        "Reduce only": f["reduce_only"].map({True: "✓", False: ""}),
    })
    st.dataframe(show, hide_index=True, use_container_width=True,
                  height=min(45 + 32 * len(show), 620))


def _render_by_asset(positions: pd.DataFrame, marks: Dict[str, float]) -> None:
    """Aggregate every top-N trader's positions by coin. Shows which assets
    whales are net long vs net short, with size and conviction.
    """
    if positions.empty:
        st.info("No open positions across the selected top-N.")
        return

    # Aggregate per (coin, side) — include median position age if available
    pos = positions.assign(
        notional=positions["notional"].fillna(0),
        pnl=positions["unrealized_pnl"].fillna(0),
    )
    agg_kw = {
        "n": ("notional", "size"),
        "notional": ("notional", "sum"),
        "pnl": ("pnl", "sum"),
        "avg_lev": ("leverage", "mean"),
    }
    if "age_ms" in pos.columns:
        agg_kw["median_age_ms"] = ("age_ms", "median")
    grp = pos.groupby(["coin", "side"], observed=True).agg(**agg_kw).reset_index()

    # Pivot to one row per coin with long_* and short_* columns
    longs = grp[grp["side"] == "long"].set_index("coin")
    shorts = grp[grp["side"] == "short"].set_index("coin")
    coins = sorted(set(longs.index) | set(shorts.index))
    rows = []
    for c in coins:
        l_n = int(longs.at[c, "n"]) if c in longs.index else 0
        l_not = float(longs.at[c, "notional"]) if c in longs.index else 0.0
        l_pnl = float(longs.at[c, "pnl"]) if c in longs.index else 0.0
        l_lev = float(longs.at[c, "avg_lev"]) if c in longs.index else None
        l_age = (float(longs.at[c, "median_age_ms"])
                  if (c in longs.index and "median_age_ms" in longs.columns
                      and pd.notna(longs.at[c, "median_age_ms"])) else None)
        s_n = int(shorts.at[c, "n"]) if c in shorts.index else 0
        s_not = float(shorts.at[c, "notional"]) if c in shorts.index else 0.0
        s_pnl = float(shorts.at[c, "pnl"]) if c in shorts.index else 0.0
        s_lev = float(shorts.at[c, "avg_lev"]) if c in shorts.index else None
        s_age = (float(shorts.at[c, "median_age_ms"])
                  if (c in shorts.index and "median_age_ms" in shorts.columns
                      and pd.notna(shorts.at[c, "median_age_ms"])) else None)
        gross = l_not + s_not
        net = l_not - s_not
        net_pct = (net / gross * 100) if gross > 0 else 0
        rows.append({
            "coin": c,
            "n_long": l_n, "n_short": s_n, "n_total": l_n + s_n,
            "long_notional": l_not, "short_notional": s_not,
            "gross_notional": gross, "net_notional": net, "net_pct": net_pct,
            "long_pnl": l_pnl, "short_pnl": s_pnl,
            "total_pnl": l_pnl + s_pnl,
            "avg_long_lev": l_lev, "avg_short_lev": s_lev,
            "median_long_age_ms": l_age, "median_short_age_ms": s_age,
            "mark": marks.get(str(c).upper()),
        })
    sentiment = pd.DataFrame(rows)
    if sentiment.empty:
        st.info("No data to aggregate.")
        return

    # ── Filters / sort ──
    fcols = st.columns(4)
    with fcols[0]:
        sort_opt = st.selectbox(
            "Rank by",
            ["Net bias $ (most long → most short)",
             "Net bias $ (most short → most long)",
             "Net bias % (most one-sided)",
             "Gross notional (most traded)",
             "Total $ long",
             "Total $ short",
             "# of traders positioned"],
            index=0, key="wt_asset_sort",
        )
    with fcols[1]:
        min_traders = st.number_input(
            "Min traders (any side)", 1, 100, 2, step=1, key="wt_asset_min_n",
            help="Hide coins with too few whales positioned (noisy).",
        )
    with fcols[2]:
        min_gross_m = st.number_input(
            "Min gross notional ($M)", 0.0, 10000.0, 0.0, step=0.5,
            key="wt_asset_min_gross",
        )
    with fcols[3]:
        only_with_mark = st.checkbox(
            "Only coins with live mark price", value=True,
            key="wt_asset_only_mark",
            help="Hides delisted / inactive perps.",
        )

    f = sentiment.copy()
    f = f[f["n_total"] >= min_traders]
    f = f[f["gross_notional"] >= min_gross_m * 1e6]
    if only_with_mark:
        f = f[f["mark"].notna()]
    if f.empty:
        st.warning("No coins match the filters.")
        return

    if sort_opt.startswith("Net bias $ (most long"):
        f = f.sort_values("net_notional", ascending=False)
    elif sort_opt.startswith("Net bias $ (most short"):
        f = f.sort_values("net_notional", ascending=True)
    elif sort_opt.startswith("Net bias %"):
        f["abs_net_pct"] = f["net_pct"].abs()
        f = f.sort_values("abs_net_pct", ascending=False).drop(columns=["abs_net_pct"])
    elif sort_opt.startswith("Gross notional"):
        f = f.sort_values("gross_notional", ascending=False)
    elif sort_opt.startswith("Total $ long"):
        f = f.sort_values("long_notional", ascending=False)
    elif sort_opt.startswith("Total $ short"):
        f = f.sort_values("short_notional", ascending=False)
    else:
        f = f.sort_values("n_total", ascending=False)

    # ── Top-line metrics ──
    total_long = f["long_notional"].sum()
    total_short = f["short_notional"].sum()
    net_overall = total_long - total_short
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Coins shown", f"{len(f)}")
    h2.metric("All longs $", _fmt_usd(total_long))
    h3.metric("All shorts $", _fmt_usd(total_short))
    h4.metric(
        "Aggregate bias",
        f"{_fmt_usd(abs(net_overall))} {'long' if net_overall>0 else 'short'}",
    )

    # ── Bar chart (top 25 by current sort) ──
    chart_df = f.head(25).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=chart_df["long_notional"],
        y=chart_df["coin"],
        orientation="h",
        name="Long $",
        marker=dict(color="#26a69a"),
        hovertemplate="%{y}: long $%{x:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=-chart_df["short_notional"],
        y=chart_df["coin"],
        orientation="h",
        name="Short $",
        marker=dict(color="#e74c3c"),
        hovertemplate="%{y}: short $%{customdata:,.0f}<extra></extra>",
        customdata=chart_df["short_notional"],
    ))
    fig.add_vline(x=0, line_color="#888", line_width=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=max(420, 32 * len(chart_df) + 80),
        margin=dict(l=10, r=10, t=40, b=40),
        title=f"<b>Top {len(chart_df)} coins by {sort_opt}</b>"
              f" — left bar = aggregate short $, right bar = aggregate long $",
        barmode="overlay",
        bargap=0.2,
        xaxis=dict(title="$ notional", gridcolor="rgba(255,255,255,0.05)",
                     zeroline=True, zerolinecolor="#888"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                     xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Detail table ──
    def _lean_bar(net_pct: float, width: int = 14) -> str:
        # Visual bias indicator: red...●...green
        pos = int(round((net_pct + 100) / 200 * (width - 1)))
        pos = max(0, min(width - 1, pos))
        bar = ["─"] * width
        center = width // 2
        bar[center] = "│"
        bar[pos] = "●"
        return "".join(bar)

    show = pd.DataFrame({
        "Coin": f["coin"],
        "# Long": f["n_long"],
        "# Short": f["n_short"],
        "Long $": f["long_notional"].apply(_fmt_usd),
        "Short $": f["short_notional"].apply(_fmt_usd),
        "Net $": f["net_notional"].apply(
            lambda v: f"+{_fmt_usd(v)}" if v >= 0 else f"−{_fmt_usd(abs(v))}"
        ),
        "Net %": f["net_pct"].apply(lambda v: f"{v:+.0f}%"),
        "Lean": f["net_pct"].apply(_lean_bar),
        "Avg L lev": f["avg_long_lev"].apply(
            lambda v: f"{v:.1f}×" if pd.notna(v) else "—"),
        "Avg S lev": f["avg_short_lev"].apply(
            lambda v: f"{v:.1f}×" if pd.notna(v) else "—"),
        "Median L age": f.get("median_long_age_ms",
                                  pd.Series(dtype=float)).apply(
            lambda v: _fmt_age_ms(int(v)) if pd.notna(v) else "—"),
        "Median S age": f.get("median_short_age_ms",
                                  pd.Series(dtype=float)).apply(
            lambda v: _fmt_age_ms(int(v)) if pd.notna(v) else "—"),
        "Combined PnL": f["total_pnl"].apply(_fmt_usd),
        "Mark $": f["mark"].apply(lambda v: f"${v:,.4g}" if pd.notna(v) else "—"),
    })
    st.dataframe(show, hide_index=True, use_container_width=True,
                  height=min(45 + 32 * len(show), 640))

    # ── Highlights ──
    st.markdown("---")
    most_long = f.sort_values("net_notional", ascending=False).head(5)
    most_short = f.sort_values("net_notional", ascending=True).head(5)
    most_one_sided = f.assign(abs_pct=f["net_pct"].abs()) \
                         .sort_values("abs_pct", ascending=False).head(5)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### 🟢 Most NET LONG")
        for _, r in most_long.iterrows():
            st.markdown(
                f"- **{r['coin']}** &nbsp; "
                f"<span style='color:#26a69a;'>+{_fmt_usd(r['net_notional'])}</span> "
                f"<span style='color:#888;'>({r['n_long']}L / {r['n_short']}S)</span>",
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown("##### 🔴 Most NET SHORT")
        for _, r in most_short.iterrows():
            if r["net_notional"] >= 0:
                continue
            st.markdown(
                f"- **{r['coin']}** &nbsp; "
                f"<span style='color:#e74c3c;'>−{_fmt_usd(abs(r['net_notional']))}</span> "
                f"<span style='color:#888;'>({r['n_long']}L / {r['n_short']}S)</span>",
                unsafe_allow_html=True,
            )
    with c3:
        st.markdown("##### 🎯 Most ONE-SIDED")
        for _, r in most_one_sided.iterrows():
            color = "#26a69a" if r["net_pct"] >= 0 else "#e74c3c"
            st.markdown(
                f"- **{r['coin']}** &nbsp; "
                f"<span style='color:{color};'>{r['net_pct']:+.0f}%</span> "
                f"<span style='color:#888;'>({r['n_long']}L / {r['n_short']}S · "
                f"{_fmt_usd(r['gross_notional'])} gross)</span>",
                unsafe_allow_html=True,
            )


def _render_bias_history(top_n: int, sort_by: str, window: str) -> None:
    """Timeseries chart of net long / short / bias % per coin over snapshots."""
    n_snaps, coins_known = _bias_history_meta()

    bb1, bb2, bb3 = st.columns([1, 1, 4])
    with bb1:
        if st.button("📸 Take snapshot now", key="wt_bias_snap_now",
                       type="primary"):
            with st.spinner("Aggregating + writing snapshot..."):
                try:
                    out = wbh.take_snapshot(top_n=top_n, sort_by=sort_by,
                                             window=window)
                    if out.empty:
                        st.warning("Snapshot returned no data.")
                    else:
                        st.success(f"Wrote {len(out)} coin rows to history.")
                        n_snaps += 1
                except Exception as e:
                    st.error(f"Snapshot failed: {e}")
    with bb2:
        st.metric("Snapshots stored", f"{n_snaps}")
    with bb3:
        st.caption(
            "History stored at `~/.market-hub/whale_bias_history.csv`. "
            "Take snapshots manually here, or run "
            "`python picker.py bias-snapshot` on a cron (e.g. hourly) for "
            "continuous tracking."
        )

    if n_snaps < 2:
        st.info(
            "Need at least 2 snapshots to chart bias drift. "
            "Click **Take snapshot now**, wait an hour or so, take another, "
            "and the chart will populate."
        )
        return

    # ── Filters ──
    fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
    with fc1:
        coin = st.selectbox(
            "Coin", coins_known,
            index=(coins_known.index("BTC") if "BTC" in coins_known else 0),
            key="wt_bias_coin",
        )
    with fc2:
        hours = st.select_slider(
            "Window", options=[6, 12, 24, 48, 72, 168, 336],
            value=72, key="wt_bias_hours",
            format_func=lambda h: f"{h}h" if h < 168 else f"{h//168}w",
        )
    with fc3:
        match_settings = st.checkbox(
            "Only this top-N / sort filter",
            value=False, key="wt_bias_match",
            help=f"If on, only show snapshots taken with top_n={top_n}, "
                 f"sort_by={sort_by!r}, window={window!r}.",
        )
    with fc4:
        if st.button("🔄 Reload", key="wt_bias_reload"):
            st.rerun()

    df = _bias_history_cached(
        coin=coin, hours_back=hours,
        top_n=top_n if match_settings else None,
        sort_by=sort_by if match_settings else None,
    )
    if df.empty:
        st.info(f"No history for {coin} in the last {hours}h "
                 f"(with the current filters).")
        return

    # ── Chart 1: Long $ vs Short $ stacked area ──
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["long_notional"],
        mode="lines", name="Long $",
        line=dict(color="#26a69a", width=2),
        fill="tozeroy",
        fillcolor="rgba(38,166,154,0.25)",
        hovertemplate="%{x|%b %-d %H:%M} · long $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=-df["short_notional"],
        mode="lines", name="Short $ (negative axis)",
        line=dict(color="#e74c3c", width=2),
        fill="tozeroy",
        fillcolor="rgba(231,76,60,0.25)",
        customdata=df["short_notional"],
        hovertemplate="%{x|%b %-d %H:%M} · short $%{customdata:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#888", line_width=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=380, margin=dict(l=10, r=10, t=40, b=30),
        title=f"<b>{coin}</b> whale positioning over time — long vs short notional",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="$ notional",
                     gridcolor="rgba(255,255,255,0.05)",
                     zeroline=True, zerolinecolor="#888"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05,
                     xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Chart 2: Bias % over time (-100 short, +100 long) ──
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["timestamp"], y=df["net_pct"],
        mode="lines+markers",
        line=dict(color="#f1c40f", width=2),
        marker=dict(size=6),
        hovertemplate="%{x|%b %-d %H:%M} · bias %{y:+.1f}%<extra></extra>",
        showlegend=False,
    ))
    fig2.add_hline(y=0, line_color="#888", line_width=1, line_dash="dash")
    fig2.add_hrect(y0=50, y1=100, fillcolor="rgba(38,166,154,0.10)",
                    line_width=0, annotation_text="strong long",
                    annotation_position="top right")
    fig2.add_hrect(y0=-100, y1=-50, fillcolor="rgba(231,76,60,0.10)",
                    line_width=0, annotation_text="strong short",
                    annotation_position="bottom right")
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=320, margin=dict(l=10, r=10, t=40, b=30),
        title=f"<b>{coin}</b> bias % — +100 = pure long, −100 = pure short",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Net bias %", range=[-110, 110],
                     gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Detect bias flips ──
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["prev_net"] = df["net_notional"].shift(1)
    flips = df[(df["prev_net"].notna()) &
                  (((df["prev_net"] >= 0) & (df["net_notional"] < 0)) |
                   ((df["prev_net"] < 0) & (df["net_notional"] >= 0)))]
    if not flips.empty:
        st.markdown("##### ⚡ Bias flips in this window")
        flip_lines = []
        for _, r in flips.iterrows():
            direction = ("long → short" if r["prev_net"] >= 0 else "short → long")
            color = "#e74c3c" if r["prev_net"] >= 0 else "#26a69a"
            t = r["timestamp"]
            flip_lines.append(
                f"- {t.strftime('%b %-d %H:%M')} UTC  &nbsp;·&nbsp; "
                f"<span style='color:{color};font-weight:600;'>{direction}</span>  "
                f"&nbsp;·&nbsp; net moved to "
                f"{('+' if r['net_notional']>=0 else '−')}"
                f"${abs(r['net_notional'])/1e6:.1f}M"
            )
        st.markdown("\n".join(flip_lines), unsafe_allow_html=True)
    else:
        st.caption("_No directional flips inside this window — bias has been consistent._")


def _render_clusters(orders: pd.DataFrame, marks: Dict[str, float]) -> None:
    """Render: per-coin histogram of trigger prices (stops & TPs)."""
    if orders.empty:
        st.info("No open orders across the selected top-N.")
        return

    coins_available = sorted(orders.loc[orders["is_trigger"], "coin"]
                                  .dropna().unique())
    if not coins_available:
        st.info("No trigger orders found across the selected top-N.")
        return

    pick = st.selectbox("Coin", coins_available, index=0, key="wt_cluster_coin")
    mark = marks.get(pick.upper())
    if not mark:
        st.warning(f"No mark price for {pick}.")
        return

    triggers = orders[(orders["coin"] == pick) & orders["is_trigger"] &
                       orders["trigger_px"].notna()].copy()
    if triggers.empty:
        st.info(f"No trigger orders on {pick}.")
        return

    # Best-effort split: stops vs TPs based on side relative to mark + reduce_only
    def _is_stop(r):
        if not r["reduce_only"]:
            return False
        if r["side"] == "buy" and r["trigger_px"] > mark:
            return True   # closing a short via buy-stop above
        if r["side"] == "sell" and r["trigger_px"] < mark:
            return True   # closing a long via sell-stop below
        return False

    def _is_tp(r):
        if not r["reduce_only"]:
            return False
        if r["side"] == "buy" and r["trigger_px"] < mark:
            return True   # taking profit on a short
        if r["side"] == "sell" and r["trigger_px"] > mark:
            return True   # taking profit on a long
        return False

    triggers["is_stop"] = triggers.apply(_is_stop, axis=1)
    triggers["is_tp"] = triggers.apply(_is_tp, axis=1)
    triggers["notional"] = triggers["trigger_px"] * triggers["size"]

    # X-axis range ±50% of mark
    lo, hi = mark * 0.5, mark * 1.5
    visible = triggers[(triggers["trigger_px"] >= lo) &
                          (triggers["trigger_px"] <= hi)]

    fig = go.Figure()
    if not visible.empty:
        stops_below = visible[visible["is_stop"] & (visible["trigger_px"] < mark)]
        stops_above = visible[visible["is_stop"] & (visible["trigger_px"] > mark)]
        tps_below = visible[visible["is_tp"] & (visible["trigger_px"] < mark)]
        tps_above = visible[visible["is_tp"] & (visible["trigger_px"] > mark)]

        if not stops_below.empty:
            fig.add_trace(go.Bar(
                x=stops_below["trigger_px"], y=stops_below["notional"],
                name="Stops (long exits below)",
                marker=dict(color="rgba(231,76,60,0.85)"),
                hovertemplate="$%{x:,.0f} · STOP $%{y:,.0f}<extra></extra>",
            ))
        if not stops_above.empty:
            fig.add_trace(go.Bar(
                x=stops_above["trigger_px"], y=stops_above["notional"],
                name="Stops (short exits above)",
                marker=dict(color="rgba(231,76,60,0.85)"),
                hovertemplate="$%{x:,.0f} · STOP $%{y:,.0f}<extra></extra>",
                showlegend=False,
            ))
        if not tps_above.empty:
            fig.add_trace(go.Bar(
                x=tps_above["trigger_px"], y=tps_above["notional"],
                name="Take-profits (long exits above)",
                marker=dict(color="rgba(38,166,154,0.85)"),
                hovertemplate="$%{x:,.0f} · TP $%{y:,.0f}<extra></extra>",
            ))
        if not tps_below.empty:
            fig.add_trace(go.Bar(
                x=tps_below["trigger_px"], y=tps_below["notional"],
                name="Take-profits (short exits below)",
                marker=dict(color="rgba(38,166,154,0.85)"),
                hovertemplate="$%{x:,.0f} · TP $%{y:,.0f}<extra></extra>",
                showlegend=False,
            ))

    fig.add_vline(x=mark, line_dash="dash", line_color="#f1c40f",
                   annotation_text=f"Mark ${mark:,.2f}",
                   annotation_position="top",
                   annotation_font_color="#f1c40f")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=520, margin=dict(l=10, r=10, t=40, b=40),
        title=f"<b>{pick}</b> — whale stops & take-profits clustered by price",
        xaxis=dict(title="Trigger price ($)",
                     gridcolor="rgba(255,255,255,0.05)", range=[lo, hi]),
        yaxis_title="$ notional",
        legend=dict(orientation="h", yanchor="bottom", y=1.05,
                     xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        bargap=0.02,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quick text summary
    stop_below_total = triggers[triggers["is_stop"] &
                                   (triggers["trigger_px"] < mark)]["notional"].sum()
    stop_above_total = triggers[triggers["is_stop"] &
                                   (triggers["trigger_px"] > mark)]["notional"].sum()
    tp_above_total = triggers[triggers["is_tp"] &
                                 (triggers["trigger_px"] > mark)]["notional"].sum()
    tp_below_total = triggers[triggers["is_tp"] &
                                 (triggers["trigger_px"] < mark)]["notional"].sum()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Long stops below", _fmt_usd(stop_below_total),
                help="Long positions get stopped out → forced sells if price drops")
    s2.metric("Short stops above", _fmt_usd(stop_above_total),
                help="Short positions get stopped out → forced buys if price rises")
    s3.metric("Long TPs above", _fmt_usd(tp_above_total),
                help="Long take-profits → planned sells above")
    s4.metric("Short TPs below", _fmt_usd(tp_below_total),
                help="Short take-profits → planned buys below")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.markdown(
        "Top traders on Hyperliquid (live from "
        "`stats-data.hyperliquid.xyz/Mainnet/leaderboard`). Inspect positions, "
        "TP/SL/limit orders, and cross-trader cluster zones — all from public on-chain data."
    )

    # ── Controls ──
    cc1, cc2, cc3, cc4 = st.columns([2, 1, 1, 1])
    with cc1:
        sort_label = st.selectbox(
            "Rank by", list(SORT_LABELS.keys()), index=1, key="wt_sort",
        )
        sort_by, window = SORT_LABELS[sort_label]
    with cc2:
        top_n = st.select_slider(
            "Top N", options=[25, 50, 100, 200, 500, 1000],
            value=100, key="wt_top_n",
        )
    with cc3:
        min_av_m = st.number_input(
            "Min account ($M)", 0.0, 1000.0, 0.0, step=0.5,
            key="wt_min_av",
        )
    with cc4:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("🔄 Refresh", key="wt_refresh"):
            _top_traders.clear()
            _clearinghouse.clear()
            _frontend_open_orders.clear()
            _whale_dataset.clear()
            _live_marks.clear()
            st.rerun()

    with st.spinner("Fetching HL leaderboard..."):
        try:
            traders_df = _top_traders(top_n, sort_by, window, min_av_m * 1e6)
        except Exception as e:
            st.error(f"Leaderboard fetch failed: {e}")
            return

    if traders_df.empty:
        st.warning("No traders matched.")
        return

    # ── Sub-section nav (radio not tabs → only the active body runs) ──
    SUBS = ["🏆 Top Traders", "🐋 By Asset", "📊 Positions",
            "⚡ Open Orders", "🎯 Order Clusters", "📈 Bias History"]
    sub = st.radio(
        "Sub-section", SUBS, horizontal=True,
        label_visibility="collapsed", key="wt_subsection",
    )

    def _load_dataset():
        if not st.session_state.get("wt_loaded"):
            return None, None
        with st.spinner(f"Pulling positions+orders for {len(traders_df)} addresses..."):
            ds = _whale_dataset(tuple(traders_df["address"].tolist()))
            marks = _live_marks()
        return ds, marks

    def _need_load_button(label_action: str, key: str):
        if st.button(f"Load {label_action} for these traders",
                       key=key, type="primary"):
            st.session_state["wt_loaded"] = True
            st.rerun()
        st.info("Click to fetch the top-N traders' positions + orders. "
                 "~5–15s on first run, cached for 2 min after.")

    if sub == "🏆 Top Traders":
        _render_top_traders(traders_df, sort_label)

    elif sub == "🐋 By Asset":
        if not st.session_state.get("wt_loaded"):
            _need_load_button("positions", "wt_load_asset")
        else:
            ds, marks = _load_dataset()
            _render_by_asset(ds["positions"], marks)

    elif sub == "📊 Positions":
        if not st.session_state.get("wt_loaded"):
            _need_load_button("positions + orders", "wt_load_pos")
        else:
            ds, marks = _load_dataset()
            _render_positions(ds["positions"], marks, traders_df)

    elif sub == "⚡ Open Orders":
        if not st.session_state.get("wt_loaded"):
            _need_load_button("orders", "wt_load_ord")
        else:
            ds, marks = _load_dataset()
            _render_orders(ds["orders"], marks)

    elif sub == "🎯 Order Clusters":
        if not st.session_state.get("wt_loaded"):
            _need_load_button("order clusters", "wt_load_clu")
        else:
            ds, marks = _load_dataset()
            _render_clusters(ds["orders"], marks)

    elif sub == "📈 Bias History":
        _render_bias_history(top_n=top_n, sort_by=sort_by, window=window)

"""Sector + S&P 500 heatmaps and value-screen.

Data sources (free):
- Wikipedia for the S&P 500 constituent list + GICS sector mapping
- yfinance for prices (bulk batched download)
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import yfinance as yf

import indicators as ind


# Finviz-style colorscale: red → near-black → green (no yellow)
FINVIZ_COLORSCALE = [
    [0.00, "#8B0000"],   # deep red
    [0.25, "#B22222"],
    [0.45, "#3a1a1a"],
    [0.50, "#1a1a1a"],   # neutral / no change
    [0.55, "#1a3a1a"],
    [0.75, "#228B22"],
    [1.00, "#008B00"],   # deep green
]

DARK_BG = "#0e1117"


# 11 SPDR Select Sector ETFs (one per GICS sector)
SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLE":  "Energy",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLU":  "Utilities",
    "XLC":  "Communication Services",
}


@lru_cache(maxsize=1)
def get_sp500_tickers() -> pd.DataFrame:
    """Return DataFrame with columns: Symbol, Security, GICS Sector, GICS Sub-Industry."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    df = pd.read_html(StringIO(r.text))[0]
    # yfinance uses dashes, not dots (BRK.B → BRK-B)
    df["Symbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
    keep = ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]
    df = df[[c for c in keep if c in df.columns]]
    return df.reset_index(drop=True)


def _bulk_close(tickers, period="5d"):
    """Download 'Close' for many tickers, return DataFrame of close, DataFrame of volume."""
    data = yf.download(
        " ".join(tickers), period=period, auto_adjust=True,
        progress=False, group_by="ticker", threads=True,
    )
    closes = {}
    volumes = {}
    for t in tickers:
        try:
            c = data[t]["Close"].dropna()
            v = data[t]["Volume"].dropna()
            if len(c) >= 2:
                closes[t] = c
                volumes[t] = v
        except Exception:
            continue
    close_df = pd.DataFrame(closes)
    vol_df = pd.DataFrame(volumes)
    return close_df, vol_df


def get_sector_data(period: str = "5d") -> pd.DataFrame:
    """% change for the 11 sector ETFs over the given period."""
    tickers = list(SECTOR_ETFS.keys())
    close, vol = _bulk_close(tickers, period=period)
    rows = []
    for t in tickers:
        if t not in close.columns:
            continue
        c = close[t].dropna()
        if len(c) < 2:
            continue
        chg_1d = (c.iloc[-1] / c.iloc[-2] - 1) * 100
        chg_period = (c.iloc[-1] / c.iloc[0] - 1) * 100
        rows.append({
            "ticker": t,
            "sector": SECTOR_ETFS[t],
            "price": c.iloc[-1],
            "change_1d": chg_1d,
            "change_period": chg_period,
        })
    return pd.DataFrame(rows).sort_values("change_period", ascending=False).reset_index(drop=True)


@lru_cache(maxsize=1)
def _market_caps_cached() -> tuple:
    """Fetch market caps for all S&P 500 tickers in parallel. Cached for the session."""
    meta = get_sp500_tickers()
    tickers = meta["Symbol"].tolist()

    def _get(t):
        try:
            return t, float(yf.Ticker(t).fast_info["marketCap"])
        except Exception:
            return t, None

    with ThreadPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(_get, tickers))
    return tuple(results)  # tuple so it's hashable for lru_cache


def _market_caps() -> dict:
    return {t: m for t, m in _market_caps_cached() if m}


@lru_cache(maxsize=1)
def _fundamentals_cached() -> tuple:
    """Fetch fundamentals (P/E, P/B, ROE, etc.) for all S&P 500 tickers.
    yf.Ticker(t).info is slow (~0.5–1s/ticker) so this runs ~40–60s with threading.
    Cached for the session.
    """
    meta = get_sp500_tickers()
    tickers = meta["Symbol"].tolist()

    def _get(t):
        try:
            info = yf.Ticker(t).info
            return (
                t,
                info.get("trailingPE"),
                info.get("forwardPE"),
                info.get("priceToBook"),
                info.get("returnOnEquity"),
                info.get("debtToEquity"),
                info.get("profitMargins"),
                info.get("revenueGrowth"),
                info.get("earningsGrowth"),
                info.get("dividendYield"),
                info.get("marketCap"),
                info.get("enterpriseToEbitda"),
            )
        except Exception:
            return (t, None, None, None, None, None, None, None, None, None, None, None)

    with ThreadPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(_get, tickers))
    return tuple(results)


def get_fundamentals_df() -> pd.DataFrame:
    """Return a DataFrame indexed by ticker with fundamental fields."""
    rows = _fundamentals_cached()
    cols = ["ticker", "pe", "fwd_pe", "pb", "roe", "debt_equity",
            "profit_margin", "rev_growth", "earnings_growth",
            "div_yield", "market_cap", "ev_ebitda"]
    df = pd.DataFrame(list(rows), columns=cols)
    return df


def get_sp500_heatmap_data(period: str = "5d") -> pd.DataFrame:
    """Return per-stock % change + market-cap + dollar-volume for all S&P 500 names."""
    meta = get_sp500_tickers()
    tickers = meta["Symbol"].tolist()
    close, vol = _bulk_close(tickers, period=period)
    mcaps = _market_caps()

    rows = []
    for t in tickers:
        if t not in close.columns:
            continue
        c = close[t].dropna()
        v = vol[t].dropna() if t in vol.columns else None
        if len(c) < 2:
            continue
        chg_1d = (c.iloc[-1] / c.iloc[-2] - 1) * 100
        chg_period = (c.iloc[-1] / c.iloc[0] - 1) * 100
        dollar_vol = float((c * v).mean()) if v is not None and len(v) else 0
        rows.append({
            "ticker": t,
            "price": c.iloc[-1],
            "change_1d": chg_1d,
            "change_period": chg_period,
            "dollar_volume": dollar_vol,
            "market_cap": mcaps.get(t, 0) or 0,
        })
    df = pd.DataFrame(rows)
    return df.merge(meta, left_on="ticker", right_on="Symbol", how="left")


def sector_treemap(period: str = "5d") -> go.Figure:
    df = get_sector_data(period=period).copy()
    if df.empty:
        return go.Figure()
    color_col = "change_period" if period != "1d" else "change_1d"

    # Weight tiles by total sector market cap so big sectors get bigger boxes.
    # SPDR ETF sector labels use slightly different names than GICS — map them.
    spdr_to_gics = {
        "Technology": "Information Technology",
        # the rest match exactly:
        "Financials": "Financials",
        "Health Care": "Health Care",
        "Consumer Discretionary": "Consumer Discretionary",
        "Consumer Staples": "Consumer Staples",
        "Energy": "Energy",
        "Industrials": "Industrials",
        "Materials": "Materials",
        "Real Estate": "Real Estate",
        "Utilities": "Utilities",
        "Communication Services": "Communication Services",
    }
    try:
        sp_meta = get_sp500_tickers()
        mcaps = _market_caps()
        if mcaps:
            mcap_df = pd.DataFrame([
                {"Symbol": t, "market_cap": m} for t, m in mcaps.items()
            ])
            sector_mcap = sp_meta.merge(mcap_df, on="Symbol", how="left")
            sector_totals = sector_mcap.groupby("GICS Sector")["market_cap"].sum()
            df["weight"] = df["sector"].map(spdr_to_gics).map(sector_totals).fillna(0)
            if df["weight"].sum() == 0:
                df["weight"] = 1
        else:
            df["weight"] = 1
    except Exception:
        df["weight"] = 1

    # Pre-format display strings
    df["chg_1d_str"] = df["change_1d"].map(lambda x: f"{x:+.2f}%")
    df["chg_period_str"] = df["change_period"].map(lambda x: f"{x:+.2f}%")
    df["price_str"] = df["price"].map(lambda x: f"${x:,.2f}")
    df["display_pct"] = df[color_col].map(lambda x: f"{x:+.2f}%")

    fig = px.treemap(
        df,
        path=["sector"],  # flat — no parent band
        values="weight",
        color=color_col,
        color_continuous_scale=FINVIZ_COLORSCALE,
        color_continuous_midpoint=0,
        custom_data=["ticker", "chg_1d_str", "chg_period_str", "price_str", "display_pct"],
        range_color=[-3, 3],
    )
    fig.update_traces(
        textfont=dict(size=20, color="white", family="Arial Black"),
        texttemplate="<b>%{label}</b><br>%{customdata[0]}<br>%{customdata[4]}",
        hovertemplate=("<b>%{label}</b><br>"
                       "Ticker: %{customdata[0]}<br>"
                       "Price: %{customdata[3]}<br>"
                       "1d: %{customdata[1]}<br>"
                       f"{period}: " + "%{customdata[2]}<extra></extra>"),
        marker=dict(line=dict(width=2, color=DARK_BG)),
        root=dict(color=DARK_BG),
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color="white"),
        coloraxis_showscale=False,
    )
    return fig


def sp500_treemap(period: str = "1d", size_by: str = "market_cap") -> go.Figure:
    df = get_sp500_heatmap_data(period=period).copy()
    if df.empty:
        return go.Figure()

    color_col = "change_period" if period != "1d" else "change_1d"

    # Prefer market_cap; fall back to dollar_volume if mostly missing
    if size_by == "market_cap" and df["market_cap"].fillna(0).gt(0).sum() < len(df) * 0.5:
        size_by = "dollar_volume"

    size_values = df[size_by].fillna(0).clip(lower=1)

    # Pre-format display strings
    df["chg_1d_str"] = df["change_1d"].map(lambda x: f"{x:+.2f}%")
    df["chg_period_str"] = df["change_period"].map(lambda x: f"{x:+.2f}%")
    df["price_str"] = df["price"].map(lambda x: f"${x:,.2f}")
    df["display_pct"] = df[color_col].map(lambda x: f"{x:+.2f}%")

    fig = px.treemap(
        df,
        path=[px.Constant("S&P 500"), "GICS Sector", "GICS Sub-Industry", "ticker"],
        values=size_values,
        color=color_col,
        color_continuous_scale=FINVIZ_COLORSCALE,
        color_continuous_midpoint=0,
        range_color=[-3, 3],
        custom_data=["Security", "chg_1d_str", "chg_period_str", "price_str", "display_pct"],
    )
    fig.update_traces(
        textfont=dict(size=13, color="white", family="Arial Black"),
        texttemplate="<b>%{label}</b><br>%{customdata[4]}",
        hovertemplate=(
            "<b>%{label}</b> — %{customdata[0]}<br>"
            "Price: %{customdata[3]}<br>"
            "1d: %{customdata[1]}<br>"
            f"{period}: " + "%{customdata[2]}<extra></extra>"
        ),
        marker=dict(line=dict(width=1, color=DARK_BG)),
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=820,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color="white"),
        coloraxis_showscale=False,
        uniformtext=dict(minsize=10, mode="hide"),  # hide text on tiles too small to read
    )
    return fig


RSI_BANDS = {
    "any":         (0,    100),
    "oversold":    (0,    30),
    "bearish":     (30,   50),
    "bullish":     (50,   70),
    "overbought":  (70,   100),
}


def value_screen(top_n: int = 50,
                 rsi_band: str = "any",
                 rsi_timeframe: str = "daily",
                 min_drawdown_pct: float = 0,
                 require_below_200d: bool = False,
                 max_pe: Optional[float] = None,
                 max_pb: Optional[float] = None,
                 min_roe_pct: Optional[float] = None,
                 max_debt_equity: Optional[float] = None,
                 max_ev_ebitda: Optional[float] = None,
                 use_fundamentals: bool = True) -> pd.DataFrame:
    """Multi-factor value screen on the S&P 500.

    Technical filters:
      - rsi_band: 'oversold' (<30) / 'bearish' (30-50) / 'bullish' (50-70) /
                  'overbought' (>70) / 'any'
      - rsi_timeframe: 'daily' (14d RSI) or 'weekly' (14w RSI)
      - min_drawdown_pct: only show stocks at least N% off their 52-week high
      - require_below_200d: only show stocks trading below the 200-day SMA

    Fundamental filters (use_fundamentals=True; ~60s on first call, cached):
      - max_pe, max_pb, min_roe_pct, max_debt_equity, max_ev_ebitda

    Returns a DataFrame ranked by composite value score (lower P/E + higher ROE
    + bigger drawdown = higher score).
    """
    meta = get_sp500_tickers()
    tickers = meta["Symbol"].tolist()

    # 2y daily so weekly RSI(14) has enough history (14w * 5 = 70 trading days min)
    data = yf.download(
        " ".join(tickers), period="2y", auto_adjust=True,
        progress=False, group_by="ticker", threads=True,
    )

    band_lo, band_hi = RSI_BANDS.get(rsi_band, RSI_BANDS["any"])

    rows = []
    for t in tickers:
        try:
            c = data[t]["Close"].dropna()
            if len(c) < 200:
                continue
            high_52w = c.iloc[-252:].max() if len(c) >= 252 else c.max()
            current = c.iloc[-1]
            drawdown = (current / high_52w - 1) * 100
            if drawdown > -min_drawdown_pct:
                continue

            # RSI on the requested timeframe
            if rsi_timeframe == "weekly":
                weekly = c.resample("W").last().dropna()
                if len(weekly) < 20:
                    continue
                rsi_now = float(ind.rsi(weekly, 14).iloc[-1])
            else:
                rsi_now = float(ind.rsi(c, 14).iloc[-1])
            if pd.isna(rsi_now):
                continue
            if not (band_lo <= rsi_now <= band_hi):
                continue

            sma_200 = float(c.rolling(200).mean().iloc[-1])
            below_200 = current < sma_200
            if require_below_200d and not below_200:
                continue

            rows.append({
                "ticker": t,
                "price": float(current),
                "52w_high": float(high_52w),
                "drawdown_%": float(drawdown),
                "rsi": rsi_now,
                "rsi_tf": rsi_timeframe,
                "below_200d": bool(below_200),
                "sma_200": sma_200,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.merge(meta, left_on="ticker", right_on="Symbol", how="left")

    # Fundamentals overlay
    if use_fundamentals:
        try:
            fund = get_fundamentals_df()
            df = df.merge(fund, on="ticker", how="left")
        except Exception:
            pass

        # Filter on fundamentals if requested
        if max_pe is not None and "pe" in df.columns:
            df = df[(df["pe"].isna()) | (df["pe"] <= max_pe) & (df["pe"] > 0)]
        if max_pb is not None and "pb" in df.columns:
            df = df[(df["pb"].isna()) | ((df["pb"] <= max_pb) & (df["pb"] > 0))]
        if min_roe_pct is not None and "roe" in df.columns:
            df = df[(df["roe"].isna()) | (df["roe"] * 100 >= min_roe_pct)]
        if max_debt_equity is not None and "debt_equity" in df.columns:
            df = df[(df["debt_equity"].isna()) | (df["debt_equity"] <= max_debt_equity)]
        if max_ev_ebitda is not None and "ev_ebitda" in df.columns:
            df = df[(df["ev_ebitda"].isna()) | ((df["ev_ebitda"] <= max_ev_ebitda) & (df["ev_ebitda"] > 0))]

        # Composite score: lower P/E good, higher ROE good, deeper drawdown good
        def _score(r):
            s = 0.0
            if pd.notna(r.get("pe")) and r["pe"] > 0:
                s += max(0, 30 - r["pe"]) / 30  # 0..1, P/E≤0 ignored
            if pd.notna(r.get("roe")):
                s += min(max(r["roe"], 0), 0.5) * 2  # 0..1 capped at 50% ROE
            if pd.notna(r.get("drawdown_%")):
                s += min(abs(r["drawdown_%"]) / 50, 1)  # 0..1 capped at 50% dd
            if pd.notna(r.get("debt_equity")):
                s += max(0, 1 - r["debt_equity"] / 200)  # less leverage = better
            return round(s, 3)

        df["score"] = df.apply(_score, axis=1)
        df = df.sort_values("score", ascending=False)
    else:
        df = df.sort_values("drawdown_%")

    return df.head(top_n).reset_index(drop=True)

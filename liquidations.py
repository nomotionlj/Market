"""Crypto derivatives data: Open Interest, Funding, Liquidations, Long/Short ratio.

Primary source: OKX public v5 API (no key, accessible from US IPs).
OKX exposes a public liquidation-orders endpoint, which is rare among major
exchanges — most either gate it behind a key (Coinglass) or block US (Binance/Bybit).

Optional fallback: Coinglass v3 (free tier with key) for cross-exchange aggregations.
"""
from typing import Optional

import pandas as pd
import requests


OKX_BASE = "https://www.okx.com"
COINGLASS_BASE = "https://open-api-v3.coinglass.com"


# OKX bar codes for /market/candles
_OKX_BAR = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1H", "4h": "4H", "12h": "12H", "1d": "1D", "1w": "1W",
}
# OKX period codes for /rubik/stat endpoints (only 5m / 1H / 8H / 1D are valid).
# We map 4h → 1H so the rubik panels still load when the user picks 4h klines.
_OKX_RUBIK_PERIOD = {
    "5m": "5m", "1h": "1H", "4h": "1H", "1d": "1D",
}


def _okx_swap(symbol: str) -> str:
    """Convert e.g. 'BTC' or 'BTCUSDT' to OKX swap inst id 'BTC-USDT-SWAP'."""
    s = symbol.upper().replace("-", "").replace("/", "")
    if s.endswith("USDT"):
        base = s[:-4]
    elif s.endswith("USD"):
        base = s[:-3]
    else:
        base = s
    return f"{base}-USDT-SWAP"


def _okx_ccy(symbol: str) -> str:
    """Convert e.g. 'BTC-USDT-SWAP' or 'BTCUSDT' to base currency 'BTC'."""
    s = symbol.upper().replace("-", "").replace("/", "")
    if s.endswith("USDTSWAP"):
        return s[:-8]
    if s.endswith("USDT"):
        return s[:-4]
    if s.endswith("USD"):
        return s[:-3]
    return s


# ---------------------------------------------------------------------------
# OKX — public, no key required
# ---------------------------------------------------------------------------

def okx_klines(symbol: str = "BTC-USDT-SWAP", interval: str = "1h",
               limit: int = 300) -> pd.DataFrame:
    """OHLCV klines from OKX perp."""
    inst = symbol if "-" in symbol else _okx_swap(symbol)
    url = f"{OKX_BASE}/api/v5/market/candles"
    params = {"instId": inst, "bar": _OKX_BAR.get(interval, "1H"),
              "limit": min(limit, 300)}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    body = r.json()
    rows = body.get("data") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "ts", "open", "high", "low", "close", "volume", "vol_ccy", "vol_ccy_quote", "confirm",
    ])
    df["time"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    return df.sort_values("time").reset_index(drop=True)[
        ["time", "open", "high", "low", "close", "volume"]
    ]


def okx_open_interest_history(symbol: str = "BTC", interval: str = "1h",
                                limit: int = 200) -> pd.DataFrame:
    """OI history aggregated across OKX swaps for the currency.
    Returns: time, open_interest (in coin), open_interest_value (USD).
    """
    ccy = _okx_ccy(symbol)
    url = f"{OKX_BASE}/api/v5/rubik/stat/contracts/open-interest-volume"
    params = {"ccy": ccy, "period": _OKX_RUBIK_PERIOD.get(interval, "1H")}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    body = r.json()
    rows = body.get("data") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "open_interest", "volume"])
    df["time"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms")
    df["open_interest"] = pd.to_numeric(df["open_interest"])
    df["volume"] = pd.to_numeric(df["volume"])
    # OKX returns oldest-first or newest-first depending on the endpoint;
    # always take the most recent `limit` rows after sorting ascending.
    return df.sort_values("time").reset_index(drop=True).tail(limit).reset_index(drop=True)[
        ["time", "open_interest", "volume"]
    ]


def okx_funding_rate_history(symbol: str = "BTC-USDT-SWAP", limit: int = 100) -> pd.DataFrame:
    """Funding rate history (every 8h on most OKX perps)."""
    inst = symbol if "-" in symbol else _okx_swap(symbol)
    url = f"{OKX_BASE}/api/v5/public/funding-rate-history"
    params = {"instId": inst, "limit": min(limit, 100)}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    body = r.json()
    rows = body.get("data") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(pd.to_numeric(df["fundingTime"]), unit="ms")
    df["funding_rate"] = pd.to_numeric(df["fundingRate"]) * 100  # to %
    return df.sort_values("time").reset_index(drop=True)[["time", "funding_rate"]]


def okx_long_short_ratio(symbol: str = "BTC", interval: str = "1h",
                          limit: int = 200) -> pd.DataFrame:
    """Account-based long/short ratio across OKX swaps for the currency."""
    ccy = _okx_ccy(symbol)
    url = f"{OKX_BASE}/api/v5/rubik/stat/contracts/long-short-account-ratio"
    params = {"ccy": ccy, "period": _OKX_RUBIK_PERIOD.get(interval, "1H")}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    body = r.json()
    rows = body.get("data") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "long_short_ratio"])
    df["time"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms")
    df["long_short_ratio"] = pd.to_numeric(df["long_short_ratio"])
    return df.sort_values("time").reset_index(drop=True).tail(limit).reset_index(drop=True)[
        ["time", "long_short_ratio"]
    ]


def okx_liquidations(symbol: str = "BTC", limit: int = 100) -> pd.DataFrame:
    """Recent liquidation orders for a perp underlying from OKX (public, no key).
    `symbol` may be 'BTC', 'BTC-USDT', or 'BTC-USDT-SWAP' — only base-quote is used.
    Returns one row per liquidated position with: time, liq_side ('long'|'short'),
    size, price, notional.
    """
    # Normalize to OKX uly format e.g. 'BTC-USDT'
    s = symbol.upper()
    if s.endswith("-SWAP"):
        uly = s[:-5]
    elif "-" in s:
        uly = s
    else:
        ccy = _okx_ccy(s)
        uly = f"{ccy}-USDT"
    url = f"{OKX_BASE}/api/v5/public/liquidation-orders"
    params = {"instType": "SWAP", "uly": uly, "state": "filled",
              "limit": min(limit, 100)}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    body = r.json()
    data = body.get("data") or []
    rows = []
    for item in data:
        details = item.get("details") or []
        for d in details:
            rows.append({
                "time": pd.to_datetime(pd.to_numeric(d.get("ts")), unit="ms"),
                "side": d.get("side"),         # 'buy' = short liq, 'sell' = long liq (counterparty)
                "size": pd.to_numeric(d.get("sz", 0)),
                "price": pd.to_numeric(d.get("bkPx", 0)),
                "loss": pd.to_numeric(d.get("bkLoss", 0)),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Convert side to long_liq/short_liq label.
    # OKX: side='sell' means a long was liquidated (had to sell out).
    df["liq_side"] = df["side"].map({"sell": "long", "buy": "short"})
    df["notional"] = df["size"] * df["price"]
    return df.sort_values("time", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Coinglass — requires free API key (https://www.coinglass.com/sign/up)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Bull/Bear bias from derivatives signals
# ---------------------------------------------------------------------------

def derivatives_bias(klines: pd.DataFrame,
                      funding: pd.DataFrame,
                      ls_ratio: pd.DataFrame,
                      oi: pd.DataFrame,
                      liquidations: pd.DataFrame) -> dict:
    """Combine funding rate + long/short ratio + open interest change + price
    + liquidations into a directional bias.

    Each signal contributes a score in [-1, +1]. A 'naive' interpretation
    treats positive funding / rising OI / crowded shorts as bullish; the
    function also exposes a 'contrarian' label flag when positioning looks
    crowded (extreme L/S or extreme funding).
    """
    out = {
        "funding_score": None, "funding_note": "",
        "ls_score": None, "ls_note": "",
        "oi_score": None, "oi_note": "",
        "liq_score": None, "liq_note": "",
        "total_score": 0, "verdict": "Neutral", "color": "#888",
        "is_crowded": False,
    }
    n = 0

    # ---- Funding rate (last available) ----
    if not funding.empty:
        f = float(funding["funding_rate"].iloc[-1])  # already in %
        out["funding_note"] = f"{f:+.4f}% per 8h"
        # Most assets fund at ±0.01% on stable; ±0.05% is high; ±0.1% extreme
        if f > 0.05:
            out["funding_score"] = -0.5  # crowded longs (contrarian bearish)
            out["funding_note"] += " — crowded longs"
            out["is_crowded"] = True
        elif f > 0.01:
            out["funding_score"] = 0.3  # mildly bullish positioning
            out["funding_note"] += " — bullish bias"
        elif f < -0.05:
            out["funding_score"] = 0.5  # crowded shorts (contrarian bullish)
            out["funding_note"] += " — crowded shorts"
            out["is_crowded"] = True
        elif f < -0.01:
            out["funding_score"] = -0.3
            out["funding_note"] += " — bearish bias"
        else:
            out["funding_score"] = 0
            out["funding_note"] += " — neutral"
        n += 1

    # ---- Long/Short ratio (last available) ----
    if not ls_ratio.empty:
        r = float(ls_ratio["long_short_ratio"].iloc[-1])
        out["ls_note"] = f"{r:.2f}"
        if r > 2.5:
            out["ls_score"] = -0.6
            out["ls_note"] += " — heavily long-biased (contrarian bearish)"
            out["is_crowded"] = True
        elif r > 1.5:
            out["ls_score"] = -0.2
            out["ls_note"] += " — long-biased"
        elif r < 0.4:
            out["ls_score"] = 0.6
            out["ls_note"] += " — heavily short-biased (contrarian bullish)"
            out["is_crowded"] = True
        elif r < 0.7:
            out["ls_score"] = 0.2
            out["ls_note"] += " — short-biased"
        else:
            out["ls_score"] = 0
            out["ls_note"] += " — balanced"
        n += 1

    # ---- OI vs price change ----
    if not oi.empty and not klines.empty and len(oi) >= 4 and len(klines) >= 4:
        oi_chg = float(oi["open_interest"].iloc[-1] / oi["open_interest"].iloc[0] - 1)
        px_chg = float(klines["close"].iloc[-1] / klines["close"].iloc[0] - 1)
        out["oi_note"] = f"OI {oi_chg*100:+.1f}% · px {px_chg*100:+.1f}%"
        # Classic OI-vs-price interpretation
        if oi_chg > 0.02 and px_chg > 0.01:
            out["oi_score"] = 0.7
            out["oi_note"] += " → new longs (bull trend strong)"
        elif oi_chg > 0.02 and px_chg < -0.01:
            out["oi_score"] = -0.7
            out["oi_note"] += " → new shorts (bear trend strong)"
        elif oi_chg < -0.02 and px_chg > 0.01:
            out["oi_score"] = 0.2
            out["oi_note"] += " → short covering (weak bull)"
        elif oi_chg < -0.02 and px_chg < -0.01:
            out["oi_score"] = -0.2
            out["oi_note"] += " → long unwind (weak bear)"
        else:
            out["oi_score"] = 0
            out["oi_note"] += " → no clear regime"
        n += 1

    # ---- Liquidation skew ----
    if not liquidations.empty:
        long_liq = liquidations.loc[liquidations["liq_side"] == "long", "notional"].sum()
        short_liq = liquidations.loc[liquidations["liq_side"] == "short", "notional"].sum()
        total = long_liq + short_liq
        out["liq_note"] = f"L ${long_liq/1e6:.1f}M · S ${short_liq/1e6:.1f}M"
        if total > 0:
            short_pct = short_liq / total
            # Lots of shorts liquidated → price was rising (bullish momentum, may exhaust)
            if short_pct > 0.7:
                out["liq_score"] = 0.4
                out["liq_note"] += " — shorts squeezed (bull momentum)"
            elif short_pct < 0.3:
                out["liq_score"] = -0.4
                out["liq_note"] += " — longs flushed (bear momentum)"
            else:
                out["liq_score"] = 0
                out["liq_note"] += " — balanced"
            n += 1

    # ---- Total ----
    components = [out[f"{k}_score"] for k in ("funding", "ls", "oi", "liq")
                  if out[f"{k}_score"] is not None]
    if components:
        score = sum(components) / len(components)
    else:
        score = 0
    out["total_score"] = round(score, 2)

    # Verdict thresholds
    if score >= 0.4:
        out["verdict"] = "Strong Bull"
        out["color"] = "#1f7a1f"
    elif score >= 0.15:
        out["verdict"] = "Bullish"
        out["color"] = "#3aa83a"
    elif score <= -0.4:
        out["verdict"] = "Strong Bear"
        out["color"] = "#a02020"
    elif score <= -0.15:
        out["verdict"] = "Bearish"
        out["color"] = "#cc4444"
    else:
        out["verdict"] = "Neutral"
        out["color"] = "#888"

    return out


def coinglass_liquidations(symbol: str = "BTC", key: Optional[str] = None,
                            interval: str = "h1") -> pd.DataFrame:
    """Aggregated cross-exchange liquidation history.
    interval: m1, m5, m15, m30, h1, h4, h12, d1
    """
    if not key:
        return pd.DataFrame([{"error": "Coinglass API key required (free at coinglass.com)."}])
    url = f"{COINGLASS_BASE}/api/futures/liquidation/aggregated-history"
    headers = {"accept": "application/json", "CG-API-KEY": key}
    params = {"symbol": symbol.upper(), "interval": interval}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])

    data = body.get("data") if isinstance(body, dict) else None
    if not data:
        msg = body.get("msg", "No data") if isinstance(body, dict) else "No data"
        return pd.DataFrame([{"error": msg}])

    df = pd.DataFrame(data)
    if df.empty:
        return df
    ts_col = "time" if "time" in df.columns else "t"
    df["time"] = pd.to_datetime(pd.to_numeric(df[ts_col]), unit="ms")
    for src, dst in [("longLiquidationUsd", "long_liq"),
                     ("shortLiquidationUsd", "short_liq"),
                     ("longVolUsd", "long_liq"),
                     ("shortVolUsd", "short_liq")]:
        if src in df.columns and dst not in df.columns:
            df[dst] = pd.to_numeric(df[src])
    if "long_liq" in df.columns and "short_liq" in df.columns:
        df["total_liq"] = df["long_liq"] + df["short_liq"]
    return df.sort_values("time").reset_index(drop=True)

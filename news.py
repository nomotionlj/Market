"""News + ticker name lookups.

Sources (all free):
- SEC company_tickers.json — maps US ticker → company name
- Finnhub /news, /company-news — market and per-ticker news
- Stocktwits public stream — community sentiment per ticker
- Google News RSS — broad web search per ticker (no key)
"""
import datetime as dt
import xml.etree.ElementTree as ET
from functools import lru_cache

import pandas as pd
import requests

FINNHUB_BASE = "https://finnhub.io/api/v1"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_HEADERS = {"User-Agent": "market-hub local-research contact@example.com"}


# ---------------------------------------------------------------------------
# Ticker -> company name
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _ticker_map() -> dict:
    """One-time fetch of SEC's full ticker map. ~13k US-listed companies."""
    try:
        r = requests.get(SEC_TICKERS_URL, headers=SEC_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        return {row["ticker"].upper(): row["title"] for row in data.values()}
    except Exception:
        return {}


def name_for(ticker: str) -> str:
    if not ticker:
        return ""
    return _ticker_map().get(ticker.upper(), "")


def add_names(df: pd.DataFrame, ticker_col: str = "symbol", new_col: str = "name") -> pd.DataFrame:
    if df.empty or ticker_col not in df.columns:
        return df
    m = _ticker_map()
    df = df.copy()
    df[new_col] = df[ticker_col].astype(str).str.upper().map(m).fillna("")
    return df


# ---------------------------------------------------------------------------
# Finnhub news
# ---------------------------------------------------------------------------

def market_news(key: str, category: str = "general", limit: int = 30) -> pd.DataFrame:
    """category: general / forex / crypto / merger"""
    r = requests.get(f"{FINNHUB_BASE}/news",
                     params={"category": category, "token": key}, timeout=15)
    r.raise_for_status()
    items = r.json() or []
    df = pd.DataFrame(items[:limit])
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"], unit="s", errors="coerce")
    return df


def company_news(key: str, symbol: str, days_back: int = 14, limit: int = 50) -> pd.DataFrame:
    today = dt.date.today()
    frm = today - dt.timedelta(days=days_back)
    r = requests.get(f"{FINNHUB_BASE}/company-news",
                     params={"symbol": symbol.upper(),
                             "from": str(frm), "to": str(today),
                             "token": key}, timeout=15)
    r.raise_for_status()
    items = r.json() or []
    df = pd.DataFrame(items[:limit])
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"], unit="s", errors="coerce")
    return df.sort_values("datetime", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stocktwits — community sentiment + recent posts (no key needed)
# ---------------------------------------------------------------------------

def stocktwits(symbol: str, limit: int = 30) -> pd.DataFrame:
    """Public stream of recent Stocktwits posts for a ticker."""
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol.upper()}.json"
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])
    data = r.json()
    msgs = data.get("messages", []) or []
    rows = []
    for m in msgs[:limit]:
        ent = (m.get("entities") or {}).get("sentiment") or {}
        rows.append({
            "time": m.get("created_at"),
            "user": (m.get("user") or {}).get("username"),
            "followers": (m.get("user") or {}).get("followers", 0),
            "sentiment": ent.get("basic"),  # 'Bullish' / 'Bearish' / None
            "body": m.get("body"),
            "likes": (m.get("likes") or {}).get("total", 0),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def stocktwits_sentiment(df: pd.DataFrame) -> dict:
    """Summarize bullish/bearish ratio from a stocktwits stream."""
    if df.empty or "sentiment" not in df.columns:
        return {"bullish": 0, "bearish": 0, "neutral": 0, "ratio": None}
    bull = (df["sentiment"] == "Bullish").sum()
    bear = (df["sentiment"] == "Bearish").sum()
    neut = df["sentiment"].isna().sum()
    ratio = bull / (bull + bear) if (bull + bear) > 0 else None
    return {"bullish": int(bull), "bearish": int(bear), "neutral": int(neut), "ratio": ratio}


# ---------------------------------------------------------------------------
# Google News RSS — broad web search (no key)
# ---------------------------------------------------------------------------

def google_news(query: str, limit: int = 20) -> pd.DataFrame:
    url = "https://news.google.com/rss/search"
    params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    try:
        r = requests.get(url, params=params, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return pd.DataFrame()
    rows = []
    for item in root.findall(".//item")[:limit]:
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        pub = item.findtext("pubDate") or ""
        source_el = item.find("source")
        source = source_el.text if source_el is not None else ""
        try:
            ts = pd.to_datetime(pub, errors="coerce")
        except Exception:
            ts = pd.NaT
        rows.append({"time": ts, "source": source, "headline": title, "url": link})
    return pd.DataFrame(rows)

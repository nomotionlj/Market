"""Forward-looking calendars: economic events, earnings, IPOs.

Primary source: Finnhub. Secondary: FRED for upcoming US release dates.
"""
import datetime as dt
from typing import List, Optional

import pandas as pd
import requests

FINNHUB_BASE = "https://finnhub.io/api/v1"
FRED_BASE = "https://api.stlouisfed.org/fred"


def _fh(path: str, key: str, **params) -> dict:
    params["token"] = key
    r = requests.get(f"{FINNHUB_BASE}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def economic_calendar(key: str, days_ahead: int = 14, days_back: int = 1,
                      countries: Optional[List[str]] = None,
                      min_impact: str = "low") -> pd.DataFrame:
    """Upcoming + recent economic events. impact filter: low/medium/high."""
    today = dt.date.today()
    frm = today - dt.timedelta(days=days_back)
    to = today + dt.timedelta(days=days_ahead)
    data = _fh("/calendar/economic", key, **{"from": str(frm), "to": str(to)})
    events = data.get("economicCalendar", []) or []
    df = pd.DataFrame(events)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if countries:
        df = df[df["country"].isin(countries)]
    impact_rank = {"low": 0, "medium": 1, "high": 2}
    df["_imp"] = df["impact"].str.lower().map(impact_rank).fillna(-1)
    df = df[df["_imp"] >= impact_rank.get(min_impact, 0)]
    df = df.drop(columns=["_imp"]).sort_values("time").reset_index(drop=True)
    return df


def earnings_calendar(key: str, days_ahead: int = 14, days_back: int = 1,
                      symbols: Optional[List[str]] = None) -> pd.DataFrame:
    today = dt.date.today()
    frm = today - dt.timedelta(days=days_back)
    to = today + dt.timedelta(days=days_ahead)
    data = _fh("/calendar/earnings", key, **{"from": str(frm), "to": str(to)})
    events = data.get("earningsCalendar", []) or []
    df = pd.DataFrame(events)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if symbols:
        df = df[df["symbol"].isin([s.upper() for s in symbols])]
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def ipo_calendar(key: str, days_ahead: int = 30, days_back: int = 7) -> pd.DataFrame:
    today = dt.date.today()
    frm = today - dt.timedelta(days=days_back)
    to = today + dt.timedelta(days=days_ahead)
    data = _fh("/calendar/ipo", key, **{"from": str(frm), "to": str(to)})
    events = data.get("ipoCalendar", []) or []
    df = pd.DataFrame(events)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def fred_upcoming_releases(key: str, days_ahead: int = 30) -> pd.DataFrame:
    today = dt.date.today()
    to = today + dt.timedelta(days=days_ahead)
    r = requests.get(f"{FRED_BASE}/releases/dates", params={
        "api_key": key, "file_type": "json",
        "include_release_dates_with_no_data": "false",
        "realtime_start": str(today), "realtime_end": str(to),
        "sort_order": "asc", "order_by": "release_date",
        "limit": "1000",
    }, timeout=20)
    r.raise_for_status()
    rd = r.json().get("release_dates", []) or []
    df = pd.DataFrame(rd)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[["date", "release_name", "release_id"]].sort_values("date").reset_index(drop=True)

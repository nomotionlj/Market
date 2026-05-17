"""Display formatters: turn raw API data into human-readable strings.

Used by app.py to render dataframes for the UI without showing
empty boxes, raw IDs, or unitless numbers.
"""
import math

import pandas as pd

EM_DASH = "—"


def pretty_num(value, unit: str = "") -> str:
    """Format a number with its unit appended. Returns '—' for None/NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return EM_DASH
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value) if value else EM_DASH

    if unit == "$":
        return pretty_money(v)
    if unit in ("%", "% MoM", "% YoY"):
        return f"{v:.2f}%"
    if unit == "K":
        return f"{v:,.1f}K"
    if unit == "M":
        return f"{v:,.1f}M"
    if unit == "B":
        return f"{v:,.1f}B"

    # numeric, no unit: use compact form for big numbers
    if abs(v) >= 1e9:
        return f"{v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e3 and abs(v) == int(v):
        return f"{int(v):,}"
    if abs(v) >= 100:
        return f"{v:,.1f}"
    return f"{v:,.3f}".rstrip("0").rstrip(".")


def pretty_money(value) -> str:
    """Format a dollar amount as $1.23B / $456M / $7,890."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return EM_DASH
    try:
        v = float(value)
    except (TypeError, ValueError):
        return EM_DASH
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e12:
        return f"{sign}${a / 1e12:.2f}T"
    if a >= 1e9:
        return f"{sign}${a / 1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:.2f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:.1f}K"
    return f"{sign}${a:,.0f}"


def pretty_dt(ts) -> str:
    """Format a datetime as 'Mon May 5, 2:00 PM ET'-ish (concise)."""
    if pd.isna(ts):
        return EM_DASH
    try:
        return pd.Timestamp(ts).strftime("%a %b %-d, %-I:%M %p")
    except Exception:
        return str(ts)


def pretty_date(ts) -> str:
    if pd.isna(ts):
        return EM_DASH
    try:
        return pd.Timestamp(ts).strftime("%a %b %-d, %Y")
    except Exception:
        return str(ts)


def format_econ_calendar(df: pd.DataFrame, drop_country: bool = False) -> pd.DataFrame:
    """Format Finnhub economic calendar rows for display."""
    if df.empty:
        return df
    out = pd.DataFrame()
    out["When"] = df["time"].apply(pretty_dt)
    if not drop_country and "country" in df.columns:
        out["Country"] = df["country"]
    out["Event"] = df["event"]
    out["Impact"] = df["impact"].str.capitalize() if "impact" in df.columns else EM_DASH
    out["Forecast"] = df.apply(lambda r: pretty_num(r.get("estimate"), r.get("unit", "")), axis=1)
    out["Previous"] = df.apply(lambda r: pretty_num(r.get("prev"), r.get("unit", "")), axis=1)
    has_actual = df["actual"].notna().any() if "actual" in df.columns else False
    if has_actual:
        out["Actual"] = df.apply(lambda r: pretty_num(r.get("actual"), r.get("unit", "")), axis=1)
    return out


def format_earnings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    hour_map = {"bmo": "Before open", "amc": "After close", "dmh": "Mid-day", "": EM_DASH}
    out = pd.DataFrame()
    out["Date"] = df["date"].apply(pretty_date)
    if "name" in df.columns:
        out["Symbol"] = df.apply(
            lambda r: f"{r['symbol']} ({r['name']})" if r.get("name") else r["symbol"],
            axis=1,
        )
    else:
        out["Symbol"] = df["symbol"]
    out["When"] = df["hour"].fillna("").map(lambda x: hour_map.get(x, x or EM_DASH))
    if "quarter" in df.columns and "year" in df.columns:
        out["Period"] = df.apply(
            lambda r: f"Q{int(r['quarter'])} {int(r['year'])}"
            if pd.notna(r.get("quarter")) and pd.notna(r.get("year")) else EM_DASH,
            axis=1,
        )
    out["EPS forecast"] = df["epsEstimate"].apply(lambda v: pretty_num(v))
    has_eps_actual = "epsActual" in df.columns and df["epsActual"].notna().any()
    if has_eps_actual:
        out["EPS actual"] = df["epsActual"].apply(lambda v: pretty_num(v))
    out["Revenue forecast"] = df["revenueEstimate"].apply(pretty_money)
    has_rev_actual = "revenueActual" in df.columns and df["revenueActual"].notna().any()
    if has_rev_actual:
        out["Revenue actual"] = df["revenueActual"].apply(pretty_money)
    return out


def format_ipos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame()
    out["Date"] = df["date"].apply(pretty_date)
    out["Symbol"] = df.apply(
        lambda r: f"{r['symbol']} ({r['name']})"
        if r.get("symbol") and r.get("name") else (r.get("name") or r.get("symbol") or EM_DASH),
        axis=1,
    )
    out["Exchange"] = df["exchange"].fillna(EM_DASH).replace("None", EM_DASH).replace("", EM_DASH)
    out["Price range"] = df["price"].fillna(EM_DASH).replace("None", EM_DASH).replace("", EM_DASH)
    if "numberOfShares" in df.columns:
        out["Shares"] = df["numberOfShares"].apply(
            lambda v: f"{int(v):,}" if pd.notna(v) and v else EM_DASH
        )
    if "totalSharesValue" in df.columns:
        out["Deal size"] = df["totalSharesValue"].apply(pretty_money)
    if "status" in df.columns:
        out["Status"] = df["status"].fillna(EM_DASH).str.capitalize()
    return out


def format_fred_releases(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame()
    out["Date"] = df["date"].apply(pretty_date)
    out["Release"] = df["release_name"]
    return out


def format_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Format aggregated holdings (issuer-level) for display."""
    if df.empty:
        return df
    out = pd.DataFrame()
    if "ticker" in df.columns:
        out["Ticker"] = df["ticker"].fillna("").replace("", EM_DASH)
    out["Issuer"] = df["issuer"]
    if "value" in df.columns:
        out["Value"] = df["value"].apply(pretty_money)
    if "shares" in df.columns:
        out["Shares"] = df["shares"].apply(
            lambda v: f"{int(v):,}" if pd.notna(v) and v else EM_DASH
        )
    if "%_of_portfolio" in df.columns:
        out["% of Portfolio"] = df["%_of_portfolio"].apply(lambda v: f"{v:.2f}%")
    return out


def format_changes(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Format quarter-over-quarter holdings changes. kind: 'new'/'increased'/'reduced'."""
    if df.empty:
        return df
    out = pd.DataFrame()
    if "ticker" in df.columns:
        out["Ticker"] = df["ticker"].fillna("").replace("", EM_DASH)
    out["Issuer"] = df["issuer"]
    if kind == "new":
        out["Shares bought"] = df["shares_curr"].apply(
            lambda v: f"{int(v):,}" if pd.notna(v) else EM_DASH
        )
        out["Position value"] = df["value_curr"].apply(pretty_money)
    elif kind == "increased":
        out["Shares added"] = df["share_change"].apply(
            lambda v: f"+{int(v):,}" if pd.notna(v) else EM_DASH
        )
        out["Change"] = df["pct_change"].apply(
            lambda v: f"+{v*100:.1f}%" if pd.notna(v) and v != float("inf") else EM_DASH
        )
        out["Position value"] = df["value_curr"].apply(pretty_money)
    else:  # reduced / exited
        out["Shares sold"] = df["share_change"].apply(
            lambda v: f"{int(v):,}" if pd.notna(v) else EM_DASH
        )
        out["Change"] = df["pct_change"].apply(
            lambda v: f"{v*100:.1f}%" if pd.notna(v) and v != float("inf") else EM_DASH
        )
        out["Status"] = df["status"].str.capitalize()
    return out

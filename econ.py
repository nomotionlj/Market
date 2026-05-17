"""Economic indicators from FRED (no API key needed for the CSV download endpoint)."""
from io import StringIO

import pandas as pd
import requests

# Curated list of major US economic indicators
INDICATORS = {
    "CPI (YoY %)": {"id": "CPIAUCSL", "transform": "yoy_pct"},
    "Core CPI (YoY %)": {"id": "CPILFESL", "transform": "yoy_pct"},
    "PCE (YoY %)": {"id": "PCEPI", "transform": "yoy_pct"},
    "Core PCE (YoY %)": {"id": "PCEPILFE", "transform": "yoy_pct"},
    "Unemployment Rate": {"id": "UNRATE", "transform": "raw"},
    "Nonfarm Payrolls (Δ, thousands)": {"id": "PAYEMS", "transform": "monthly_diff"},
    "Fed Funds Rate (effective)": {"id": "DFF", "transform": "raw"},
    "Real GDP (QoQ annualized %)": {"id": "A191RL1Q225SBEA", "transform": "raw"},
    "10Y Treasury": {"id": "DGS10", "transform": "raw"},
    "2Y Treasury": {"id": "DGS2", "transform": "raw"},
    "10Y-2Y Spread": {"id": "T10Y2Y", "transform": "raw"},
    "ISM Manufacturing PMI": {"id": "MANEMP", "transform": "raw"},  # proxy
    "Retail Sales (YoY %)": {"id": "RSAFS", "transform": "yoy_pct"},
    "Initial Jobless Claims": {"id": "ICSA", "transform": "raw"},
    "M2 Money Supply": {"id": "M2SL", "transform": "raw"},
    "VIX": {"id": "VIXCLS", "transform": "raw"},
}


def fetch_series(series_id: str) -> pd.DataFrame:
    """Fetch a FRED series via fredgraph.csv (no API key required)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    # Modern CSV uses 'observation_date' but legacy was 'DATE'
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    df = df.rename(columns={date_col: "date", series_id: "value"})
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"]).reset_index(drop=True)


def transform_series(df: pd.DataFrame, transform: str) -> pd.DataFrame:
    df = df.copy()
    if transform == "yoy_pct":
        df["value"] = df["value"].pct_change(12) * 100
    elif transform == "monthly_diff":
        df["value"] = df["value"].diff()
    return df.dropna(subset=["value"]).reset_index(drop=True)


def get_indicator(name: str) -> pd.DataFrame:
    cfg = INDICATORS[name]
    df = fetch_series(cfg["id"])
    df = transform_series(df, cfg["transform"])
    return df


def latest_snapshot() -> pd.DataFrame:
    """Pull most recent value + previous value for every indicator."""
    rows = []
    for name in INDICATORS:
        try:
            df = get_indicator(name)
            if len(df) >= 2:
                last = df.iloc[-1]
                prev = df.iloc[-2]
                rows.append({
                    "indicator": name,
                    "date": last["date"].strftime("%Y-%m-%d"),
                    "value": round(last["value"], 3),
                    "previous": round(prev["value"], 3),
                    "change": round(last["value"] - prev["value"], 3),
                })
        except Exception as e:
            rows.append({"indicator": name, "date": None, "value": None,
                         "previous": None, "change": None, "error": str(e)})
    return pd.DataFrame(rows)

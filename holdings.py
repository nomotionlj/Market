"""SEC EDGAR 13F holdings fetcher.

13F-HR filings are filed quarterly by institutional managers with >$100M AUM.
Filings are public, free, and lag ~45 days after quarter end.
"""
import json
import os
import re
import time
from io import BytesIO
from typing import Dict, List
from xml.etree import ElementTree as ET

import pandas as pd
import requests

# SEC requires a descriptive User-Agent with contact info.
HEADERS = {"User-Agent": "market-hub local-research contact@example.com"}

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
_CUSIP_CACHE_FILE = os.path.expanduser("~/.market-hub/cusip_ticker_cache.json")

FAMOUS_FUNDS = {
    "Berkshire Hathaway (Buffett)": "0001067983",
    "Pershing Square (Ackman)": "0001336528",
    "Scion Asset Management (Burry)": "0001649339",
    "Bridgewater Associates": "0001350694",
    "Renaissance Technologies": "0001037389",
    "Citadel Advisors": "0001423053",
    "Soros Fund Management": "0001029160",
    "Tiger Global": "0001167483",
    "Appaloosa (Tepper)": "0001656456",
    "Greenlight Capital (Einhorn)": "0001079114",
    "Baupost Group (Klarman)": "0001061768",
    "Duquesne Family Office (Druckenmiller)": "0001536411",
    "Third Point (Loeb)": "0001040273",
    "Euro Pacific Asset Management": "0001403438",
}


def _pad_cik(cik: str) -> str:
    return str(cik).lstrip("0").zfill(10)


def search_company(query: str, max_results: int = 10):
    """Search EDGAR for companies/filers by name. Returns list of (name, cik)."""
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {"action": "getcompany", "company": query, "type": "13F", "dateb": "", "owner": "include", "count": "40"}
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    # Parse table with CIK links
    matches = re.findall(r'CIK=(\d+).*?>([^<]+)</a>', r.text)
    seen = set()
    results = []
    for cik, name in matches:
        cik = cik.zfill(10)
        if cik in seen:
            continue
        seen.add(cik)
        results.append({"name": name.strip(), "cik": cik})
        if len(results) >= max_results:
            break
    return results


def get_filings(cik: str, form_type: str = "13F-HR") -> pd.DataFrame:
    """Get list of recent filings of given type for a CIK."""
    cik_padded = _pad_cik(cik)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()
    df = pd.DataFrame(recent)
    df = df[df["form"].str.startswith(form_type, na=False)].copy()
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    df["reportDate"] = pd.to_datetime(df["reportDate"])
    df["entity_name"] = data.get("name", "")
    df["cik"] = cik_padded
    return df.sort_values("filingDate", ascending=False).reset_index(drop=True)


def _filing_doc_urls(cik_padded: str, accession: str):
    accession_clean = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik_padded)}/{accession_clean}"
    index_url = f"{base}/{accession}-index.htm"
    return base, index_url


def get_holdings(cik: str, accession: str) -> pd.DataFrame:
    """Download and parse the information table XML for a given 13F filing."""
    cik_padded = _pad_cik(cik)
    accession_clean = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik_padded)}/{accession_clean}"
    # Find the information table XML — list directory JSON
    idx_url = f"{base}/index.json"
    r = requests.get(idx_url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    items = r.json().get("directory", {}).get("item", [])
    info_xml_name = None
    for it in items:
        name = it.get("name", "")
        if name.lower().endswith(".xml") and "primary_doc" not in name.lower():
            info_xml_name = name
            break
    if info_xml_name is None:
        # fallback: any xml
        for it in items:
            if it.get("name", "").lower().endswith(".xml"):
                info_xml_name = it["name"]
                break
    if info_xml_name is None:
        return pd.DataFrame()
    xml_url = f"{base}/{info_xml_name}"
    time.sleep(0.1)  # be polite
    r = requests.get(xml_url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return _parse_information_table(r.content)


def _parse_information_table(xml_bytes: bytes) -> pd.DataFrame:
    # Strip namespaces for simpler parsing
    text = xml_bytes.decode("utf-8", errors="ignore")
    text = re.sub(r'\sxmlns(:\w+)?="[^"]+"', "", text, count=0)
    text = re.sub(r"<(/?)\w+:", r"<\1", text)
    root = ET.fromstring(text)
    rows = []
    for it in root.findall(".//infoTable"):
        def g(tag):
            el = it.find(f".//{tag}")
            return el.text.strip() if el is not None and el.text else None
        rows.append({
            "issuer": g("nameOfIssuer"),
            "class": g("titleOfClass"),
            "cusip": g("cusip"),
            "value": float(g("value") or 0),
            "shares": float(g("sshPrnamt") or 0),
            "share_type": g("sshPrnamtType"),
            "put_call": g("putCall"),
            "discretion": g("investmentDiscretion"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Note: Post-2022 filings report value in actual dollars. Pre-2022 in thousands.
    # We just return as-is and let caller decide; column is raw value from filing.
    df = df.sort_values("value", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CUSIP → Ticker lookup (OpenFIGI, free, no key required for ≤25 req / 6s)
# ---------------------------------------------------------------------------

def _load_cusip_cache() -> Dict[str, str]:
    try:
        with open(_CUSIP_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cusip_cache(cache: Dict[str, str]) -> None:
    try:
        os.makedirs(os.path.dirname(_CUSIP_CACHE_FILE), exist_ok=True)
        with open(_CUSIP_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def cusip_to_ticker_batch(cusips: List[str], api_key: str = "") -> Dict[str, str]:
    """Batch CUSIP → US ticker via OpenFIGI. Persistent disk cache.
    Without an API key: 25 requests / 6 seconds, 5 ids per request.
    With a key: 250 / 6 seconds, 100 per request.
    """
    cusips = [c for c in {c.strip() for c in cusips if c} if c]
    if not cusips:
        return {}
    cache = _load_cusip_cache()
    out = {c: cache[c] for c in cusips if c in cache}
    todo = [c for c in cusips if c not in cache]
    if not todo:
        return out

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key
        chunk_size, sleep_s = 100, 0.25
    else:
        chunk_size, sleep_s = 5, 0.30  # ≈20 req/6s — under the 25 limit

    for i in range(0, len(todo), chunk_size):
        chunk = todo[i:i + chunk_size]
        body = [{"idType": "ID_CUSIP", "idValue": c, "marketSecDes": "Equity",
                 "exchCode": "US"} for c in chunk]
        try:
            r = requests.post(OPENFIGI_URL, headers=headers,
                              data=json.dumps(body), timeout=15)
            if r.status_code == 429:
                time.sleep(7)
                r = requests.post(OPENFIGI_URL, headers=headers,
                                  data=json.dumps(body), timeout=15)
            r.raise_for_status()
            results = r.json()
        except Exception:
            results = [{"data": []} for _ in chunk]
        for c, res in zip(chunk, results):
            data = res.get("data") if isinstance(res, dict) else None
            if data:
                # Prefer composite/common-stock entries with a US-style ticker
                tk = ""
                for entry in data:
                    t = (entry.get("ticker") or "").strip()
                    if t and " " not in t:  # filter out warrants etc with spaces
                        tk = t
                        break
                cache[c] = tk
                out[c] = tk
            else:
                cache[c] = ""
                out[c] = ""
        time.sleep(sleep_s)

    _save_cusip_cache(cache)
    return out


def add_tickers(holdings_df: pd.DataFrame, api_key: str = "") -> pd.DataFrame:
    """Add a 'ticker' column to a holdings DataFrame using CUSIP lookup."""
    if holdings_df.empty or "cusip" not in holdings_df.columns:
        return holdings_df.assign(ticker="") if not holdings_df.empty else holdings_df
    cusips = holdings_df["cusip"].dropna().astype(str).str.zfill(9).unique().tolist()
    mapping = cusip_to_ticker_batch(cusips, api_key=api_key)
    df = holdings_df.copy()
    df["ticker"] = df["cusip"].astype(str).str.zfill(9).map(mapping).fillna("")
    return df


def compare_holdings(prev: pd.DataFrame, curr: pd.DataFrame) -> pd.DataFrame:
    """Compare two holdings snapshots, return changes by issuer (aggregated).
    Keeps one CUSIP per issuer so callers can do downstream ticker lookup
    on just the rows they intend to display.
    """
    def _agg(df, prefix):
        if df is None or df.empty:
            return pd.DataFrame(columns=["issuer", "cusip",
                                          f"shares_{prefix}", f"value_{prefix}"])
        return df.groupby("issuer", as_index=False).agg(
            **{f"shares_{prefix}": ("shares", "sum"),
               f"value_{prefix}":  ("value",  "sum"),
               "cusip":            ("cusip",  "first")})

    if prev is None or prev.empty:
        out = _agg(curr, "curr")
        out["shares_prev"] = 0
        out["value_prev"] = 0
    elif curr is None or curr.empty:
        out = _agg(prev, "prev")
        out["shares_curr"] = 0
        out["value_curr"] = 0
    else:
        a = _agg(prev, "prev")
        b = _agg(curr, "curr")
        # merge — issuer is the join key, cusip lives in both
        out = a.merge(b, on="issuer", how="outer", suffixes=("_p", ""))
        # prefer the curr cusip; fall back to prev cusip
        out["cusip"] = out["cusip"].fillna(out.get("cusip_p"))
        if "cusip_p" in out.columns:
            out = out.drop(columns=["cusip_p"])
        out = out.fillna({"shares_prev": 0, "value_prev": 0,
                          "shares_curr": 0, "value_curr": 0})
    out["share_change"] = out["shares_curr"] - out["shares_prev"]
    out["pct_change"] = out.apply(
        lambda r: (r["share_change"] / r["shares_prev"]) if r["shares_prev"] else float("inf") if r["shares_curr"] else 0,
        axis=1,
    )
    out["status"] = out.apply(
        lambda r: "NEW" if r["shares_prev"] == 0 and r["shares_curr"] > 0
        else "EXITED" if r["shares_curr"] == 0 and r["shares_prev"] > 0
        else "INCREASED" if r["share_change"] > 0
        else "REDUCED" if r["share_change"] < 0
        else "UNCHANGED",
        axis=1,
    )
    return out.sort_values("value_curr", ascending=False).reset_index(drop=True)

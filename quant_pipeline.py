"""Streamlit-free core of the Quant Screen.

Used by both the interactive UI (quant_screen_ui.py) and the headless
CLI runner (picker.py / cron). All caching here is pure on-disk caching
via the underlying modules — no st.cache_data decorators.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

import earnings_signals as es
import heatmap as hm
import insider as ins
import quant_factors as qf
import quant_fetch


# ---------------------------------------------------------------------------
# Goal presets — single source of truth, imported by UI and picker
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict] = {
    "Custom": {
        "rsi_band": "any", "rsi_tf": "daily",
        "min_dd": 0, "below_200": False,
        "max_pe": 25.0, "max_pb": 5.0, "min_roe": 10.0, "max_de": 200.0, "max_ev": 20.0,
        "weights": None,
        "fetch_quant": True, "fetch_insider": False, "fetch_earnings": True,
    },
    "Long-term Value": {
        "rsi_band": "any", "rsi_tf": "weekly",
        "min_dd": 5, "below_200": False,
        "max_pe": 20.0, "max_pb": 3.0, "min_roe": 12.0, "max_de": 150.0, "max_ev": 15.0,
        "weights": {"f_score": 0.18, "altman_z": 0.10, "earnings_yield": 0.15,
                    "roic": 0.15, "accruals": 0.07, "asset_growth": 0.05,
                    "eps_surprise": 0.05, "eps_beat_streak": 0.05,
                    "low_vol": 0.05, "dist_52wh": 0.05,
                    "mom_12_1": 0.05, "insider_buy": 0.05},
        "fetch_quant": True, "fetch_insider": True, "fetch_earnings": True,
    },
    "Momentum Quality": {
        "rsi_band": "bullish", "rsi_tf": "daily",
        "min_dd": 0, "below_200": False,
        "max_pe": 50.0, "max_pb": 10.0, "min_roe": 15.0, "max_de": 200.0, "max_ev": 30.0,
        "weights": {"mom_12_1": 0.22, "dist_52wh": 0.15, "eps_surprise": 0.12,
                    "eps_beat_streak": 0.08, "f_score": 0.10, "roic": 0.13,
                    "earnings_yield": 0.08, "altman_z": 0.07, "accruals": 0.05},
        "fetch_quant": True, "fetch_insider": False, "fetch_earnings": True,
    },
    "Dividend Income": {
        "rsi_band": "any", "rsi_tf": "daily",
        "min_dd": 0, "below_200": False,
        "max_pe": 25.0, "max_pb": 5.0, "min_roe": 10.0, "max_de": 150.0, "max_ev": 20.0,
        "weights": {"f_score": 0.20, "altman_z": 0.20, "earnings_yield": 0.15,
                    "roic": 0.10, "accruals": 0.05, "asset_growth": 0.05,
                    "low_vol": 0.20, "eps_beat_streak": 0.05},
        "fetch_quant": True, "fetch_insider": False, "fetch_earnings": True,
        "min_div_yield": 2.5,
    },
    "Deep Value": {
        "rsi_band": "oversold", "rsi_tf": "daily",
        "min_dd": 25, "below_200": True,
        "max_pe": 12.0, "max_pb": 1.5, "min_roe": 5.0, "max_de": 200.0, "max_ev": 10.0,
        "weights": {"earnings_yield": 0.22, "f_score": 0.20, "altman_z": 0.18,
                    "roic": 0.10, "accruals": 0.10, "insider_buy": 0.15,
                    "asset_growth": 0.05},
        "fetch_quant": True, "fetch_insider": True, "fetch_earnings": True,
    },
    "Magic Formula (Greenblatt)": {
        "rsi_band": "any", "rsi_tf": "daily",
        "min_dd": 0, "below_200": False,
        "max_pe": 100.0, "max_pb": 100.0, "min_roe": 0.0, "max_de": 1000.0, "max_ev": 100.0,
        "weights": {"earnings_yield": 0.5, "roic": 0.5},
        "fetch_quant": True, "fetch_insider": False, "fetch_earnings": False,
    },
}


# ---------------------------------------------------------------------------
# Per-ticker price-momentum factor (parallel-friendly)
# ---------------------------------------------------------------------------

def _compute_price_factors(ticker: str, period: str = "2y") -> Dict:
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            return {}
        return qf.price_momentum(hist["Close"])
    except Exception:
        return {}


def enrich_with_quant(tickers: List[str],
                       fetch_insider: bool = False,
                       fetch_earnings: bool = True,
                       finnhub_key: str = "",
                       max_workers: int = 12) -> pd.DataFrame:
    """Compute Piotroski / Altman / Beneish / Magic Formula / accruals /
    asset-growth / momentum / SUE / insider for a list of tickers.
    Returns a DataFrame keyed by `ticker`.
    """
    if not tickers:
        return pd.DataFrame()

    # 1. Statements (parallel + 24h disk cache)
    stmts = quant_fetch.fetch_batch(tickers, max_workers=max_workers)

    rows = []
    for tk in tickers:
        s = stmts.get(tk, {})
        curr = s.get("curr", {})
        prev = s.get("prev", {})
        mc = s.get("market_cap", 0) or 0

        f = qf.piotroski_f_score(curr, prev)
        z = qf.altman_z_score({**curr, "market_cap": mc})
        m = qf.beneish_m_score(curr, prev)
        mf = qf.magic_formula({**curr, "market_cap": mc})
        rows.append({
            "ticker": tk,
            "f_score": f["f_score"],
            "altman_z": z,
            "altman_zone": qf.altman_zone(z),
            "beneish_m": m,
            "earnings_yield": mf["earnings_yield"],
            "roic": mf["roic"],
            "accruals": qf.sloan_accruals(curr),
            "asset_growth": qf.asset_growth(curr, prev),
        })
    df = pd.DataFrame(rows)

    # 2. Price momentum (parallel)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        price_factors = list(ex.map(_compute_price_factors, tickers))
    pf_df = pd.DataFrame(price_factors)
    pf_df["ticker"] = tickers
    df = df.merge(pf_df, on="ticker", how="left")

    # 3. Earnings surprises (sequential, paced + 12h cache)
    if fetch_earnings and finnhub_key:
        earn_data = es.earnings_summary_batch(tickers, finnhub_key)
        earn_df = pd.DataFrame([
            {"ticker": tk, **earn_data.get(tk, {})} for tk in tickers
        ])
        # Rename `eps_surprise_streak` to match `composite_quant_score` expectation
        earn_df = earn_df.rename(columns={"eps_surprise_streak": "eps_beat_streak"})
        df = df.merge(earn_df, on="ticker", how="left")

    # 4. Insider data (sequential, 12h cache)
    if fetch_insider and finnhub_key:
        insider_data = ins.insider_summary_batch(tickers, finnhub_key, days_back=90)
        ins_df = pd.DataFrame([
            {"ticker": tk, **insider_data.get(tk, {})} for tk in tickers
        ])
        df = df.merge(ins_df.rename(columns={"net_usd": "insider_net_usd"}),
                       on="ticker", how="left")
    return df


# ---------------------------------------------------------------------------
# End-to-end pipeline: preset → final ranked DataFrame
# ---------------------------------------------------------------------------

def run_screen(preset_name: str = "Long-term Value",
                top_n_quant: int = 30,
                final_top_n: Optional[int] = None,
                finnhub_key: str = "",
                base_universe_size: int = 200) -> pd.DataFrame:
    """Run the full Quant Screen pipeline end-to-end.

    Returns a DataFrame ranked by composite Quant Score, deepest-quant first.
    `final_top_n` (if set) trims the returned rows after ranking.
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name!r}. "
                          f"Available: {list(PRESETS)}")
    P = PRESETS[preset_name]

    # Step 1: basic screen on full S&P 500
    base = hm.value_screen(
        top_n=base_universe_size,
        rsi_band=P["rsi_band"],
        rsi_timeframe=P["rsi_tf"],
        min_drawdown_pct=P["min_dd"],
        require_below_200d=P["below_200"],
        use_fundamentals=True,
        max_pe=P["max_pe"],
        max_pb=P["max_pb"],
        min_roe_pct=P["min_roe"],
        max_debt_equity=P["max_de"],
        max_ev_ebitda=P["max_ev"],
    )
    if base.empty:
        return base

    # Preset-specific extras
    if "min_div_yield" in P and "div_yield" in base.columns:
        base = base[base["div_yield"].fillna(0) * 100 >= P["min_div_yield"]]
    if base.empty:
        return base

    candidates = base.head(top_n_quant).copy()

    # Step 2: deep quant overlay
    if P["fetch_quant"]:
        quant_df = enrich_with_quant(
            tickers=candidates["ticker"].tolist(),
            fetch_insider=P["fetch_insider"],
            fetch_earnings=P.get("fetch_earnings", True),
            finnhub_key=finnhub_key,
        )
        if not quant_df.empty:
            candidates = candidates.merge(quant_df, on="ticker", how="left")

            # Composite Quant Score
            candidates["quant_score"] = candidates.apply(
                lambda r: qf.composite_quant_score(r.to_dict(), weights=P["weights"]),
                axis=1,
            )
            candidates = candidates.sort_values(
                "quant_score", ascending=False
            ).reset_index(drop=True)

    if final_top_n is not None:
        candidates = candidates.head(final_top_n).reset_index(drop=True)
    return candidates

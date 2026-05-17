"""Renderable Streamlit Quant Screen.

Usage in any Streamlit page:

    import quant_screen_ui
    quant_screen_ui.render(finnhub_key=FINNHUB_KEY)

Goal presets + multi-factor (Piotroski / Altman / Magic Formula / Beneish /
momentum / 52WH / insider) overlay on the existing S&P 500 Value Screen.
"""
from typing import Dict, Optional

import pandas as pd
import streamlit as st
import yfinance as yf

import heatmap as hm
import indicators as ind
import insider as ins
import quant_factors as qf
import quant_fetch


# ---------------------------------------------------------------------------
# Goal presets — set defaults for filters + composite weights
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict] = {
    "Custom": {
        "rsi_band": "any", "rsi_tf": "daily",
        "min_dd": 0, "below_200": False,
        "max_pe": 25.0, "max_pb": 5.0, "min_roe": 10.0, "max_de": 200.0, "max_ev": 20.0,
        "weights": None,
        "fetch_quant": True, "fetch_insider": False,
    },
    "Long-term Value": {
        "rsi_band": "any", "rsi_tf": "weekly",
        "min_dd": 5, "below_200": False,
        "max_pe": 20.0, "max_pb": 3.0, "min_roe": 12.0, "max_de": 150.0, "max_ev": 15.0,
        "weights": {"f_score": 0.25, "altman_z": 0.15, "earnings_yield": 0.20,
                    "roic": 0.20, "low_vol": 0.05, "dist_52wh": 0.05,
                    "mom_12_1": 0.05, "insider_buy": 0.05},
        "fetch_quant": True, "fetch_insider": True,
    },
    "Momentum Quality": {
        "rsi_band": "bullish", "rsi_tf": "daily",
        "min_dd": 0, "below_200": False,
        "max_pe": 50.0, "max_pb": 10.0, "min_roe": 15.0, "max_de": 200.0, "max_ev": 30.0,
        "weights": {"mom_12_1": 0.30, "dist_52wh": 0.20, "f_score": 0.15,
                    "roic": 0.15, "earnings_yield": 0.10, "altman_z": 0.10},
        "fetch_quant": True, "fetch_insider": False,
    },
    "Dividend Income": {
        "rsi_band": "any", "rsi_tf": "daily",
        "min_dd": 0, "below_200": False,
        "max_pe": 25.0, "max_pb": 5.0, "min_roe": 10.0, "max_de": 150.0, "max_ev": 20.0,
        "weights": {"f_score": 0.25, "altman_z": 0.20, "earnings_yield": 0.20,
                    "roic": 0.15, "low_vol": 0.20},
        "fetch_quant": True, "fetch_insider": False,
        "min_div_yield": 2.5,
    },
    "Deep Value": {
        "rsi_band": "oversold", "rsi_tf": "daily",
        "min_dd": 25, "below_200": True,
        "max_pe": 12.0, "max_pb": 1.5, "min_roe": 5.0, "max_de": 200.0, "max_ev": 10.0,
        "weights": {"earnings_yield": 0.25, "f_score": 0.25, "altman_z": 0.20,
                    "roic": 0.15, "insider_buy": 0.15},
        "fetch_quant": True, "fetch_insider": True,
    },
    "Magic Formula (Greenblatt)": {
        "rsi_band": "any", "rsi_tf": "daily",
        "min_dd": 0, "below_200": False,
        "max_pe": 100.0, "max_pb": 100.0, "min_roe": 0.0, "max_de": 1000.0, "max_ev": 100.0,
        "weights": {"earnings_yield": 0.5, "roic": 0.5},
        "fetch_quant": True, "fetch_insider": False,
    },
}


# ---------------------------------------------------------------------------
# Helper: enrich screen results with quant factors + insider data
# ---------------------------------------------------------------------------

def _compute_price_factors(ticker: str, period: str = "2y") -> Dict:
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            return {}
        return qf.price_momentum(hist["Close"])
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _enrich_with_quant(tickers_tuple: tuple, fetch_insider: bool,
                        fetch_earnings: bool, finnhub_key: str,
                        max_workers: int = 12) -> pd.DataFrame:
    """Thin Streamlit-cached wrapper around quant_pipeline.enrich_with_quant."""
    import quant_pipeline as qpipe
    return qpipe.enrich_with_quant(
        tickers=list(tickers_tuple),
        fetch_insider=fetch_insider,
        fetch_earnings=fetch_earnings,
        finnhub_key=finnhub_key,
        max_workers=max_workers,
    )


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render(finnhub_key: str = "") -> None:
    st.markdown(
        "Multi-factor S&P 500 screen with deep quant overlay: **Piotroski F-Score**, "
        "**Altman Z-Score**, **Beneish M-Score**, **Magic Formula** (EBIT/EV + ROIC), "
        "**12-1 momentum**, **52-week-high distance**, and **insider buying**. "
        "Pick a goal preset or tune manually."
    )

    # ── Goal preset ──
    preset_name = st.selectbox(
        "Goal preset", list(PRESETS.keys()), index=1,
        help="Sets filters and factor weights for that style. Switch to 'Custom' to fully tune.",
    )
    P = PRESETS[preset_name]

    # ── Technical filters ──
    st.markdown("**Technical filters**")
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        rsi_band = st.selectbox(
            "RSI band", ["any", "oversold", "bearish", "bullish", "overbought"],
            index=["any", "oversold", "bearish", "bullish", "overbought"].index(P["rsi_band"]),
            key="qq_rsi_band",
        )
    with tc2:
        rsi_tf = st.radio("RSI timeframe", ["daily", "weekly"],
                            index=0 if P["rsi_tf"] == "daily" else 1,
                            horizontal=True, key="qq_rsi_tf")
    with tc3:
        min_dd = st.slider("Min % off 52-wk high", 0, 60, P["min_dd"], key="qq_min_dd")
    with tc4:
        below_200 = st.checkbox("Below 200d SMA", value=P["below_200"], key="qq_below_200")

    # ── Fundamental filters ──
    st.markdown("**Fundamental filters** (yfinance — cached)")
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    with fc1:
        max_pe = st.number_input("Max P/E", 0.0, 200.0, P["max_pe"], step=1.0, key="qq_max_pe")
    with fc2:
        max_pb = st.number_input("Max P/B", 0.0, 50.0, P["max_pb"], step=0.5, key="qq_max_pb")
    with fc3:
        min_roe = st.number_input("Min ROE %", -50.0, 100.0, P["min_roe"], step=1.0, key="qq_min_roe")
    with fc4:
        max_de = st.number_input("Max Debt/Equity", 0.0, 1000.0, P["max_de"], step=10.0,
                                  help="yfinance reports D/E as a percentage (200 = 2.0×).",
                                  key="qq_max_de")
    with fc5:
        max_ev = st.number_input("Max EV/EBITDA", 0.0, 100.0, P["max_ev"], step=1.0, key="qq_max_ev")

    # ── Deep quant toggle ──
    st.markdown("**Deep quant overlay**")
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        fetch_quant = st.checkbox("Piotroski / Altman / Magic / Beneish + accruals",
                                    value=P["fetch_quant"], key="qq_fetch_quant",
                                    help="Pulls 2y of financial statements per stock. "
                                         "First run ~60-120s, cached for 24h.")
    with qc2:
        fetch_earnings = st.checkbox(
            "EPS surprise / beat streak (Finnhub)",
            value=P.get("fetch_earnings", True), key="qq_fetch_earnings",
            help="Pulls last 8 quarters of EPS surprises per ticker. "
                 "~30s for 30 tickers, cached 12h.",
            disabled=not finnhub_key,
        )
        if not finnhub_key and P.get("fetch_earnings", True):
            st.caption("⚠ FINNHUB_KEY missing — earnings unavailable.")
    with qc3:
        fetch_insider = st.checkbox("Insider buying (Finnhub)",
                                     value=P["fetch_insider"], key="qq_fetch_insider",
                                     help="One Finnhub call per ticker, paced for the 60/min "
                                          "free tier. ~50s for 50 tickers.",
                                     disabled=not finnhub_key)
        if not finnhub_key and P["fetch_insider"]:
            st.caption("⚠ FINNHUB_KEY missing — insider data unavailable.")
    with qc4:
        top_n_quant = st.number_input("Top N to deep-analyze", 5, 100, 30, step=5,
                                       key="qq_top_n",
                                       help="Apply quant overlay only to top N from the basic screen.")

    run = st.button("Run quant screen", type="primary", key="qq_run")

    if not run:
        return

    # ── Step 1: basic value screen on full S&P 500 ──
    spinner_msg = ("Running basic S&P 500 screen (technicals + fundamentals)..."
                   if rsi_tf == "daily" else
                   "Running basic S&P 500 screen (weekly RSI requires 2y of data)...")
    with st.spinner(spinner_msg):
        try:
            base = hm.value_screen(
                top_n=200,
                rsi_band=rsi_band,
                rsi_timeframe=rsi_tf,
                min_drawdown_pct=min_dd,
                require_below_200d=below_200,
                use_fundamentals=True,
                max_pe=max_pe, max_pb=max_pb, min_roe_pct=min_roe,
                max_debt_equity=max_de, max_ev_ebitda=max_ev,
            )
        except Exception as e:
            st.error(f"Basic screen failed: {e}")
            return

    if base.empty:
        st.warning("No stocks matched the basic filters.")
        return

    # Apply preset-specific extras
    if "min_div_yield" in P and "div_yield" in base.columns:
        base = base[base["div_yield"].fillna(0) * 100 >= P["min_div_yield"]]

    if base.empty:
        st.warning("Filters returned no matches after preset post-filters.")
        return

    candidates = base.head(top_n_quant).copy()
    st.success(f"Basic screen returned **{len(base)}** stocks. Deep-analyzing top **{len(candidates)}**.")

    # ── Step 2: quant overlay ──
    quant_df = pd.DataFrame()
    if fetch_quant:
        with st.spinner(f"Fetching factor data for {len(candidates)} tickers..."):
            quant_df = _enrich_with_quant(
                tickers_tuple=tuple(candidates["ticker"].tolist()),
                fetch_insider=fetch_insider,
                fetch_earnings=fetch_earnings,
                finnhub_key=finnhub_key,
            )
            candidates = candidates.merge(quant_df, on="ticker", how="left")

    # ── Step 3: composite quant score ──
    if fetch_quant and not quant_df.empty:
        weights = P.get("weights")
        candidates["quant_score"] = candidates.apply(
            lambda r: qf.composite_quant_score(r.to_dict(), weights=weights),
            axis=1,
        )
        candidates = candidates.sort_values("quant_score", ascending=False).reset_index(drop=True)

    # ── Step 4: display ──
    show = candidates.copy()
    show["Price"] = show["price"].apply(lambda v: f"${v:,.2f}")
    show["Drawdown"] = show["drawdown_%"].apply(lambda v: f"{v:+.1f}%")
    rsi_label = f"RSI({rsi_tf[0]}14)"
    show[rsi_label] = show["rsi"].apply(lambda v: f"{v:.1f}")
    show["P/E"] = show["pe"].apply(lambda v: f"{v:.1f}" if pd.notna(v) and v > 0 else "—")
    show["ROE"] = show["roe"].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—")

    cols = ["ticker", "Security", "GICS Sector", "Price", "Drawdown", rsi_label, "P/E", "ROE"]

    if fetch_quant:
        show["F"] = show["f_score"].apply(lambda v: f"{int(v)}/9" if pd.notna(v) else "—")
        show["Z"] = show["altman_z"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
        show["Zone"] = show["altman_zone"].fillna("—")
        show["M"] = show["beneish_m"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        show["EarnYld"] = show["earnings_yield"].apply(
            lambda v: f"{v*100:.1f}%" if pd.notna(v) and v > 0 else "—")
        show["ROIC"] = show["roic"].apply(
            lambda v: f"{v*100:.1f}%" if pd.notna(v) and v > 0 else "—")
        show["Accruals"] = show.get("accruals", pd.Series(dtype=float)).apply(
            lambda v: f"{v*100:+.1f}%" if pd.notna(v) else "—")
        show["AssetGrowth"] = show.get("asset_growth", pd.Series(dtype=float)).apply(
            lambda v: f"{v*100:+.1f}%" if pd.notna(v) else "—")
        show["Mom 12-1"] = show.get("mom_12_1", pd.Series(dtype=float)).apply(
            lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
        show["%off 52WH"] = show.get("dist_52wh", pd.Series(dtype=float)).apply(
            lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
        show["Vol 1y"] = show.get("vol_1y", pd.Series(dtype=float)).apply(
            lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
        show["Quant Score"] = show["quant_score"].apply(lambda v: f"{v:.3f}")
        cols = ["ticker", "Security", "GICS Sector", "Quant Score",
                "F", "Z", "Zone", "M", "EarnYld", "ROIC",
                "Accruals", "AssetGrowth",
                "Mom 12-1", "%off 52WH", "Vol 1y",
                "Price", "Drawdown", rsi_label, "P/E", "ROE"]

    if fetch_earnings and "eps_surprise_pct" in show.columns:
        show["EPS Surp"] = show["eps_surprise_pct"].apply(
            lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
        show["Beat Streak"] = show["eps_beat_streak"].apply(
            lambda v: f"{int(v)}" if pd.notna(v) else "—")
        # Insert after Quant Score
        if "Quant Score" in cols:
            insert_at = cols.index("Quant Score") + 1
            cols.insert(insert_at, "EPS Surp")
            cols.insert(insert_at + 1, "Beat Streak")

    if fetch_insider and "insider_net_usd" in show.columns:
        def _ins_fmt(v):
            if pd.isna(v) or v == 0:
                return "—"
            sign = "+" if v > 0 else "−"
            return f"{sign}${abs(v) / 1e6:.1f}M"
        show["Insider 90d"] = show["insider_net_usd"].apply(_ins_fmt)
        cols.insert(cols.index("Quant Score") + 1, "Insider 90d") \
            if "Quant Score" in cols else cols.append("Insider 90d")

    st.dataframe(
        show[cols].rename(columns={"ticker": "Symbol", "GICS Sector": "Sector"}),
        use_container_width=True, hide_index=True, height=620,
    )
    st.caption(
        f"{len(candidates)} stocks shown. Preset: **{preset_name}**. "
        f"RSI: {rsi_tf} 14-period. "
        + ("Quant factors loaded. " if fetch_quant else "")
        + ("Insider data: 90-day window. " if fetch_insider else "")
        + ("⚠ Altman Z is unreliable for banks/insurance — interpret financials' Z with care."
           if fetch_quant else "")
    )

    # Stash for AI panel
    st.session_state["last_screen"] = candidates.head(20).to_dict(orient="records")

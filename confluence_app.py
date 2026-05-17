"""Standalone Streamlit page: the Confluence Analyzer + live HL panel.

Run on its own port:

    .venv/bin/streamlit run confluence_app.py --server.port 8503 \\
        --server.address 0.0.0.0 --server.headless true
"""
import pandas as pd
import streamlit as st

import hyperliquid as hl
import vision_ui


st.set_page_config(page_title="Market Hub — Confluence", layout="wide")
st.title("Confluence Analyzer")
st.caption(
    "Multi-model vision analysis of liquidation heatmaps and charts. "
    "HL on-chain data is treated as exact; CoinGlass-style modeled maps are "
    "weighted lower. Backed by live Hyperliquid funding / OI."
)

tab_conf, tab_hl = st.tabs(["Confluence", "Hyperliquid Live"])

with tab_conf:
    coin_default = st.session_state.get("conf_coin_default", "BTC")
    vision_ui.render(coin_default=coin_default)

with tab_hl:
    st.markdown("Live Hyperliquid market state — funding, open interest, "
                 "premium, recent trade flow.")

    cc1, cc2 = st.columns([1, 5])
    with cc1:
        coin = st.text_input("Coin", value="BTC", key="hl_coin").upper().strip()
    with cc2:
        if st.button("Refresh", key="hl_refresh"):
            st.rerun()

    try:
        ctx = hl.coin_context(coin)
    except Exception as e:
        st.error(f"HL fetch failed: {e}")
        ctx = {}

    if not ctx:
        st.warning(f"No HL data for {coin}.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{coin} Mark",
                    f"${ctx.get('mark') or 0:,.2f}",
                    delta=(f"{(ctx['mark']/ctx['prev_day_px']-1)*100:+.2f}% 24h"
                           if ctx.get("mark") and ctx.get("prev_day_px") else None))
        m2.metric("Open Interest",
                    f"{(ctx.get('open_interest') or 0):,.0f} {coin}")
        m3.metric("Funding (hourly)",
                    f"{(ctx.get('funding') or 0)*100:+.4f}%",
                    delta=f"APR ≈ {ctx.get('funding_apr') or 0:+.1f}%")
        m4.metric("Premium", f"{(ctx.get('premium') or 0)*100:+.4f}%")

    st.markdown("---")
    st.markdown("##### Top 20 perps on Hyperliquid (by 24h volume)")
    try:
        df = hl.asset_contexts()
    except Exception as e:
        df = pd.DataFrame()
        st.error(f"HL fetch failed: {e}")
    if not df.empty:
        df_show = df.dropna(subset=["day_volume_usd"]).copy()
        df_show = df_show.sort_values("day_volume_usd", ascending=False).head(20)
        out = pd.DataFrame({
            "Coin": df_show["coin"],
            "Mark": df_show["mark"].apply(lambda v: f"${v:,.4g}"),
            "24h Vol": df_show["day_volume_usd"].apply(lambda v: f"${v/1e6:,.1f}M"),
            "Open Interest": df_show["open_interest"].apply(
                lambda v: f"{v:,.0f}" if pd.notna(v) else "—"),
            "Funding (hr)": df_show["funding"].apply(
                lambda v: f"{v*100:+.4f}%" if pd.notna(v) else "—"),
            "Funding APR": df_show["funding_apr"].apply(
                lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"),
            "Max Lev": df_show["max_leverage"].apply(
                lambda v: f"{int(v)}×" if pd.notna(v) else "—"),
        })
        st.dataframe(out, hide_index=True, use_container_width=True,
                      height=min(45 + 32 * len(out), 520))

    st.markdown("---")
    st.markdown(f"##### Recent large trades — {coin}  (≥ $100K notional)")
    try:
        trades = hl.recent_large_trades(coin, limit=200, min_notional=100_000)
    except Exception as e:
        trades = pd.DataFrame()
        st.error(f"HL recent-trades failed: {e}")
    if trades.empty:
        st.caption("No large recent trades.")
    else:
        ts = trades.head(30).copy()
        ts["When"] = ts["time"].dt.strftime("%H:%M:%S")
        ts["Side"] = ts["side"].map({"buy": "🟢 buy", "sell": "🔴 sell"})
        ts["Price"] = ts["price"].apply(lambda v: f"${v:,.2f}")
        ts["Size"] = ts["size"].apply(lambda v: f"{v:,.4f}")
        ts["Notional"] = ts["notional"].apply(lambda v: f"${v:,.0f}")
        st.dataframe(ts[["When", "Side", "Price", "Size", "Notional"]],
                      hide_index=True, use_container_width=True, height=480)

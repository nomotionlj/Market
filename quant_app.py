"""Tabbed Quant app: Run Screen + Snapshots.

This is the recommended entry-point now that the snapshot system exists.
Run on its own port:

    .venv/bin/streamlit run quant_app.py --server.port 8502 \\
        --server.address 0.0.0.0 --server.headless true
"""
import streamlit as st

import quant_screen_ui
import snapshots_ui
import trading_ui

st.set_page_config(page_title="Market Hub — Quant", layout="wide")
st.title("Quant")
st.caption("S&P 500 stock picker · Piotroski / Altman / Beneish / Magic Formula / "
            "momentum / 52WH / insider buying.")

FINNHUB_KEY = st.secrets.get("FINNHUB_KEY", "") if hasattr(st, "secrets") else ""

tab_screen, tab_snaps, tab_trade = st.tabs(["Run Screen", "Snapshots", "Trading"])

with tab_screen:
    quant_screen_ui.render(finnhub_key=FINNHUB_KEY)

with tab_snaps:
    snapshots_ui.render()

with tab_trade:
    trading_ui.render()

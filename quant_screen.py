"""Standalone Streamlit app for the Quant Screen.

Run on its own port without touching app.py:

    .venv/bin/streamlit run quant_screen.py --server.port 8502 \\
        --server.address 0.0.0.0 --server.headless true
"""
import streamlit as st

import quant_screen_ui

st.set_page_config(page_title="Market Hub — Quant Screen", layout="wide")
st.title("Quant Screen")
st.caption("Multi-factor S&P 500 stock picker — Piotroski / Altman / Beneish / Magic Formula / "
            "momentum / 52WH / insider buying.")

FINNHUB_KEY = st.secrets.get("FINNHUB_KEY", "") if hasattr(st, "secrets") else ""
quant_screen_ui.render(finnhub_key=FINNHUB_KEY)

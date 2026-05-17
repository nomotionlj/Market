"""Unified Market Hub — every feature behind one URL.

Three top-level sections (selected via radio for LAZY rendering — only the
active section's code runs on each rerun):

  • Market Hub  — heatmap, news, crypto derivatives, economic data, backtester, 13F
  • Quant       — multi-factor screen, snapshots, paper trading
  • Confluence  — multi-model vision analysis of liquidation maps + live HL

Why radio instead of st.tabs at the top level: Streamlit's st.tabs renders
EVERY tab body on every rerun (the inactive ones are just hidden via CSS),
so all underlying API calls fire on every interaction. With a radio, only
the selected branch executes.

Run on port 8501:
    .venv/bin/streamlit run hub.py --server.port 8501 \\
        --server.address 0.0.0.0 --server.headless true
"""
import re
import sys
from pathlib import Path

import streamlit as st


# Must be FIRST Streamlit call — only hub owns page config
st.set_page_config(
    page_title="Market Hub",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Allow imports from this directory
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Secrets (single source of truth — passed into sub-renderers as needed)
# ---------------------------------------------------------------------------
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY", "") if hasattr(st, "secrets") else ""
FRED_KEY = st.secrets.get("FRED_KEY", "") if hasattr(st, "secrets") else ""

# ---------------------------------------------------------------------------
# Cache the app.py source so we don't re-read + recompile 1.4K lines every rerun
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _compiled_app_py():
    """Read app.py once, strip its outer page-config/title, return compiled code."""
    app_path = Path(__file__).parent / "app.py"
    src = app_path.read_text()
    src = re.sub(r"^\s*st\.set_page_config\([^)]*\)\s*$", "",
                  src, flags=re.MULTILINE)
    src = re.sub(r'^\s*st\.title\(\s*"Market Hub"\s*\)\s*$', "",
                  src, flags=re.MULTILINE)
    return compile(src, str(app_path), "exec")


# ---------------------------------------------------------------------------
# Header + top-level navigation (radio, NOT tabs — for lazy rendering)
# ---------------------------------------------------------------------------
st.markdown(
    "<div style='display:flex;align-items:baseline;gap:14px;margin-bottom:6px;'>"
    "<span style='font-size:2.2em;font-weight:700;'>Market Hub</span>"
    "<span style='color:#888;font-size:0.9em;'>"
    "Stocks · Crypto · AI Confluence</span></div>",
    unsafe_allow_html=True,
)

SECTIONS = ["📈  Market Hub", "🎯  Quant", "🤖  AI Confluence"]
section = st.radio(
    "Section",
    SECTIONS,
    horizontal=True,
    label_visibility="collapsed",
    key="hub_section",
)
st.markdown("---")


# ---------------------------------------------------------------------------
# Section 1 — original app.py content, nested
# ---------------------------------------------------------------------------
if section == SECTIONS[0]:
    # Small 13F filing-deadline banner above the original app.py content.
    try:
        import filing_schedule as _fs  # noqa: E402
        _sched = _fs.schedule_summary()
        _days = _sched["days_to_current_deadline"]
        _urg_color = ("#e74c3c" if _days <= 7 else
                       "#f39c12" if _days <= 30 else
                       "#26a69a")
        st.markdown(
            f"<div style='display:flex;gap:18px;align-items:center;"
            f"padding:10px 14px;margin-bottom:6px;background:#16191f;"
            f"border:1px solid #2a2e36;border-radius:8px;font-size:0.9em;'>"
            f"<span style='color:#888;font-size:0.78em;text-transform:uppercase;"
            f"letter-spacing:0.05em;'>13F filings</span>"
            f"<span><b>Latest closed:</b> "
            f"<span style='color:#aaa;'>{_sched['latest_filed_quarter']}</span> "
            f"<span style='color:#666;'>(deadline was "
            f"{_sched['latest_filed_deadline'].strftime('%b %-d')}, "
            f"{_sched['days_since_latest_deadline']}d ago)</span></span>"
            f"<span><b>Next deadline:</b> "
            f"<span style='color:{_urg_color};font-weight:600;'>"
            f"{_sched['current_window_quarter']} due "
            f"{_sched['current_window_deadline'].strftime('%b %-d, %Y')} "
            f"&nbsp;·&nbsp; {_days}d</span></span>"
            f"<span style='color:#666;font-size:0.85em;margin-left:auto;'>"
            f"45-day SEC rule · refile cycle: Feb 14 · May 15 · Aug 14 · Nov 14"
            f"</span></div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    try:
        code = _compiled_app_py()
    except FileNotFoundError:
        st.error("app.py not found alongside hub.py")
        code = None

    if code is not None:
        ns = {"__name__": "__main__",
              "__file__": str(Path(__file__).parent / "app.py")}
        exec(code, ns)


# ---------------------------------------------------------------------------
# Section 2 — Quant pipeline
# ---------------------------------------------------------------------------
elif section == SECTIONS[1]:
    import quant_screen_ui  # noqa: E402
    import snapshots_ui  # noqa: E402
    import trading_ui  # noqa: E402

    q_screen, q_snaps, q_trade = st.tabs(["Run Screen", "Snapshots", "Trading"])
    with q_screen:
        quant_screen_ui.render(finnhub_key=FINNHUB_KEY)
    with q_snaps:
        snapshots_ui.render()
    with q_trade:
        trading_ui.render()


# ---------------------------------------------------------------------------
# Section 3 — AI Confluence
# ---------------------------------------------------------------------------
elif section == SECTIONS[2]:
    import hyperliquid as _hl  # noqa: E402
    import pandas as _pd  # noqa: E402
    import vision_ui_v2 as vision_ui  # noqa: E402  — auto-HL-map + uploads

    # Cache HL data so typing/refreshing doesn't hit the API every keystroke.
    # 30s for per-coin context (mark / OI / funding), 60s for the universe table,
    # 30s for the recent-large-trades window. The "Refresh" button clears the cache.
    @st.cache_data(ttl=30, show_spinner=False)
    def _hl_ctx(coin: str):
        return _hl.coin_context(coin)

    @st.cache_data(ttl=60, show_spinner=False)
    def _hl_universe():
        return _hl.asset_contexts()

    @st.cache_data(ttl=30, show_spinner=False)
    def _hl_large_trades(coin: str, min_notional: float):
        return _hl.recent_large_trades(coin, limit=200, min_notional=min_notional)

    import whale_tracker_ui  # noqa: E402

    # Use radio (not tabs) so only the active sub-section's body runs.
    # st.tabs renders every body on every rerun — even hidden ones — which
    # meant Confluence's auto-HL-map rebuild was firing while you were on
    # other sub-tabs. Killing the spinner-that-never-stops.
    CONF_SUBS = ["Confluence", "Whale Tracker", "Hyperliquid Live"]
    conf_sub = st.radio(
        "Sub-section", CONF_SUBS,
        horizontal=True, label_visibility="collapsed",
        key="conf_subsection",
    )

    if conf_sub == "Confluence":
        vision_ui.render()
    elif conf_sub == "Whale Tracker":
        whale_tracker_ui.render()
    elif conf_sub == "Hyperliquid Live":
        st.markdown(
            "Live Hyperliquid market state — funding, open interest, premium, "
            "recent trade flow. All data is real (on-chain), not modeled."
        )

        hc1, hc2 = st.columns([1, 5])
        with hc1:
            hl_coin = st.text_input("Coin", value="BTC", key="hl_coin").upper().strip()
        with hc2:
            if st.button("Refresh", key="hl_refresh"):
                # Bust the HL caches so the next render hits the API
                _hl_ctx.clear()
                _hl_universe.clear()
                _hl_large_trades.clear()
                st.rerun()

        try:
            ctx = _hl_ctx(hl_coin)
        except Exception as e:
            st.error(f"HL fetch failed: {e}")
            ctx = {}

        if not ctx:
            st.warning(f"No HL data for {hl_coin}.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            mark = ctx.get("mark") or 0
            prev = ctx.get("prev_day_px") or 0
            delta = (f"{(mark/prev-1)*100:+.2f}% 24h"
                     if mark and prev else None)
            m1.metric(f"{hl_coin} Mark", f"${mark:,.2f}", delta=delta)
            m2.metric("Open Interest",
                      f"{(ctx.get('open_interest') or 0):,.0f} {hl_coin}")
            m3.metric("Funding (hourly)",
                      f"{(ctx.get('funding') or 0)*100:+.4f}%",
                      delta=f"APR ≈ {ctx.get('funding_apr') or 0:+.1f}%")
            m4.metric("Premium", f"{(ctx.get('premium') or 0)*100:+.4f}%")

        st.markdown("---")
        st.markdown("##### Top 20 perps on Hyperliquid (by 24h volume)")
        try:
            df = _hl_universe()
        except Exception as e:
            df = _pd.DataFrame()
            st.error(f"HL fetch failed: {e}")
        if not df.empty:
            df_show = df.dropna(subset=["day_volume_usd"]).copy()
            df_show = df_show.sort_values("day_volume_usd", ascending=False).head(20)
            out = _pd.DataFrame({
                "Coin": df_show["coin"],
                "Mark": df_show["mark"].apply(lambda v: f"${v:,.4g}"),
                "24h Vol": df_show["day_volume_usd"].apply(lambda v: f"${v/1e6:,.1f}M"),
                "Open Interest": df_show["open_interest"].apply(
                    lambda v: f"{v:,.0f}" if _pd.notna(v) else "—"),
                "Funding (hr)": df_show["funding"].apply(
                    lambda v: f"{v*100:+.4f}%" if _pd.notna(v) else "—"),
                "Funding APR": df_show["funding_apr"].apply(
                    lambda v: f"{v:+.1f}%" if _pd.notna(v) else "—"),
                "Max Lev": df_show["max_leverage"].apply(
                    lambda v: f"{int(v)}×" if _pd.notna(v) else "—"),
            })
            st.dataframe(out, hide_index=True, use_container_width=True,
                          height=min(45 + 32 * len(out), 520))

        st.markdown("---")
        st.markdown(f"##### Recent large trades — {hl_coin}  (≥ $100K notional)")
        try:
            trades = _hl_large_trades(hl_coin, 100_000)
        except Exception as e:
            trades = _pd.DataFrame()
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

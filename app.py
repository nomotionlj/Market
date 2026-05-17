"""Market Hub — local Streamlit app.

Tabs:
1. Indicator Backtester — measure accuracy of MA crossovers / RSI / MACD
2. Fund Holdings — track 13F filings (Buffett, Burry, Ackman, etc.)
3. Economic Indicators — latest US macro data via FRED
"""
import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import ai_panel as ai
import backtest as bt
import calendar_data as cal
import charts as ch
import econ
import format_utils as fmt
import heatmap as hm
import holdings as hl
import indicators as ind
import liquidations as lq
import news as nw

st.set_page_config(page_title="Market Hub", layout="wide")
st.title("Market Hub")

# Read API keys from secrets (no error if missing — features degrade gracefully)
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY", "") if hasattr(st, "secrets") else ""
FRED_KEY = st.secrets.get("FRED_KEY", "") if hasattr(st, "secrets") else ""
ANTHROPIC_KEY = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
GROQ_KEY = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
OPENROUTER_KEY = st.secrets.get("OPENROUTER_API_KEY", "") if hasattr(st, "secrets") else ""

tab_bt, tab_heat, tab_econ_cal, tab_earnings, tab_funds, tab_econ, tab_news, tab_crypto = st.tabs([
    "Backtester",
    "Heatmap",
    "Economic Calendar",
    "Earnings & IPOs",
    "Fund Holdings (13F)",
    "Economic Indicators",
    "News",
    "Crypto Derivatives",
])


# =============================================================================
# Backtester
# =============================================================================
with tab_bt:
    st.subheader("Indicator Accuracy Backtester")
    st.caption("Pull OHLC from Yahoo Finance and measure how well an indicator's signals predict forward returns.")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="SPY",
                               help="Stocks: AAPL, SPY. Crypto: BTC-USD, ETH-USD. FX: EURUSD=X.")
    with c2:
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo", "1h", "4h"], index=0)
    with c3:
        years = st.slider("Lookback (years)", 1, 25, 10)
    with c4:
        forward_bars = st.number_input("Forward bars (accuracy)", 1, 50, 5)

    indicator_choice = st.selectbox(
        "Indicator",
        ["MA Crossover", "RSI", "MACD",
         "Anchored VWAP", "Order Blocks + FVG",
         "Break of Structure", "Liquidity Sweep"],
    )

    params = {}
    if indicator_choice == "MA Crossover":
        a, b, c = st.columns(3)
        with a:
            params["fast"] = st.number_input("Fast period", 2, 400, 50)
        with b:
            params["slow"] = st.number_input("Slow period", 3, 800, 200)
        with c:
            params["ma_type"] = st.selectbox("MA type", ["SMA", "EMA"])
    elif indicator_choice == "RSI":
        a, b, c = st.columns(3)
        with a:
            params["period"] = st.number_input("RSI period", 2, 100, 14)
        with b:
            params["lower"] = st.number_input("Oversold (long)", 1, 49, 30)
        with c:
            params["upper"] = st.number_input("Overbought (short)", 51, 99, 70)
    elif indicator_choice == "MACD":
        a, b, c = st.columns(3)
        with a:
            params["fast"] = st.number_input("Fast EMA", 2, 100, 12)
        with b:
            params["slow"] = st.number_input("Slow EMA", 3, 200, 26)
        with c:
            params["signal_period"] = st.number_input("Signal EMA", 2, 100, 9)
    elif indicator_choice == "Anchored VWAP":
        a, b, c = st.columns(3)
        with a:
            params["anchor"] = st.selectbox(
                "Anchor", ["swing_low", "swing_high", "first", "date"],
                help="swing_low/high: most recent confirmed swing point. first: regular VWAP. date: pick a date.",
            )
        with b:
            params["swing_lookback"] = st.number_input("Swing lookback", 3, 50, 10)
        with c:
            params["anchor_date"] = st.text_input("Anchor date (YYYY-MM-DD)", value="",
                                                   disabled=params["anchor"] != "date")
    elif indicator_choice == "Order Blocks + FVG":
        a, b, c, d = st.columns(4)
        with a:
            params["use_obs"] = st.checkbox("Use Order Blocks", value=True)
        with b:
            params["use_fvgs"] = st.checkbox("Use FVGs", value=True)
        with c:
            params["displacement_atr_mult"] = st.number_input(
                "Displacement (×ATR)", 0.5, 5.0, 1.5, step=0.1)
        with d:
            params["swing_lookback"] = st.number_input("Swing lookback", 3, 30, 5,
                                                       key="ob_swing_lookback")
    elif indicator_choice == "Break of Structure":
        a, b = st.columns(2)
        with a:
            params["swing_lookback"] = st.number_input("Swing lookback", 3, 30, 5,
                                                       key="bos_swing_lookback")
        with b:
            params["require_sweep"] = st.checkbox(
                "Require liquidity sweep first (CHoCH)", value=False,
                help="Stricter: a wick must take prior opposite swing before BOS confirms.")
    elif indicator_choice == "Liquidity Sweep":
        params["swing_lookback"] = st.number_input("Swing lookback", 3, 30, 5,
                                                   key="ls_swing_lookback")

    long_only = st.checkbox("Long-only (ignore short signals)", value=True)

    run = st.button("Run backtest", type="primary")

    if run:
        end = dt.date.today()
        # yfinance has hard limits for intraday lookback; clamp gracefully
        if interval in ("1h", "4h"):
            start = end - dt.timedelta(days=min(years * 365, 720))
        else:
            start = end - dt.timedelta(days=years * 365)
        with st.spinner(f"Downloading {symbol} ({interval})..."):
            try:
                # yfinance doesn't support 4h natively — resample from 1h
                if interval == "4h":
                    raw = yf.download(symbol, start=start, end=end, interval="1h",
                                      auto_adjust=True, progress=False)
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    df = raw.resample("4h").agg({"Open": "first", "High": "max",
                                                  "Low": "min", "Close": "last",
                                                  "Volume": "sum"}).dropna()
                else:
                    df = yf.download(symbol, start=start, end=end, interval=interval,
                                     auto_adjust=True, progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()

        if df is None or df.empty or "Close" not in df.columns:
            st.error("No data returned. Check symbol or shorten lookback.")
            st.stop()

        close = df["Close"].dropna()
        if len(close) < 50:
            st.error(f"Only {len(close)} bars returned — too few to backtest.")
            st.stop()

        # compute indicator
        if indicator_choice == "MA Crossover":
            sig_df = ind.ma_crossover_signals(close, params["fast"], params["slow"], params["ma_type"])
        elif indicator_choice == "RSI":
            sig_df = ind.rsi_signals(close, params["period"], params["lower"], params["upper"])
        elif indicator_choice == "MACD":
            sig_df = ind.macd_signals(close, params["fast"], params["slow"], params["signal_period"])
        elif indicator_choice == "Anchored VWAP":
            sig_df = ind.avwap_signals(
                df, anchor=params["anchor"],
                swing_lookback=int(params["swing_lookback"]),
                anchor_date=params["anchor_date"] or None,
            )
        elif indicator_choice == "Order Blocks + FVG":
            sig_df = ind.ob_fvg_signals(
                df,
                use_obs=params["use_obs"],
                use_fvgs=params["use_fvgs"],
                displacement_atr_mult=float(params["displacement_atr_mult"]),
                swing_lookback=int(params["swing_lookback"]),
            )
        elif indicator_choice == "Break of Structure":
            sig_df = ind.bos_signals(
                df,
                swing_lookback=int(params["swing_lookback"]),
                require_sweep=params["require_sweep"],
            )
        else:  # Liquidity Sweep
            sig_df = ind.liquidity_sweep_signals(
                df, swing_lookback=int(params["swing_lookback"])
            )

        metrics = bt.compute_metrics(close, sig_df["signal"], forward_bars=forward_bars,
                                     long_only=long_only)

        # ---- metrics
        st.markdown("### Results")
        m = st.columns(4)
        fa = metrics["forward_accuracy"]
        m[0].metric(f"Forward {forward_bars}-bar Accuracy",
                    f"{fa*100:.1f}%" if pd.notna(fa) else "n/a",
                    help=f"% of {metrics['n_signals']} signal events where direction matched the next {forward_bars}-bar return.")
        m[1].metric("Win Rate (per trade)",
                    f"{metrics['win_rate']*100:.1f}%" if pd.notna(metrics['win_rate']) else "n/a",
                    help=f"{metrics['n_trades']} trades")
        m[2].metric("Strategy Return", f"{metrics['total_return_strategy']*100:.1f}%",
                    delta=f"vs B&H {metrics['total_return_buyhold']*100:.1f}%")
        m[3].metric("Sharpe (annualized)",
                    f"{metrics['sharpe']:.2f}" if pd.notna(metrics['sharpe']) else "n/a",
                    help=f"Max DD: {metrics['max_drawdown']*100:.1f}%")

        # ---- price chart with signals
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
        fig.add_trace(go.Scatter(x=close.index, y=close, name="Close", line=dict(color="#888")), row=1, col=1)

        if indicator_choice == "MA Crossover":
            fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df["fast"], name=f"{params['ma_type']} {params['fast']}", line=dict(color="#1f77b4")), row=1, col=1)
            fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df["slow"], name=f"{params['ma_type']} {params['slow']}", line=dict(color="#ff7f0e")), row=1, col=1)
            entries = sig_df[sig_df["event"] == "golden"].index
            exits = sig_df[sig_df["event"] == "death"].index
        elif indicator_choice == "RSI":
            fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df["rsi"], name="RSI", line=dict(color="#1f77b4")), row=2, col=1)
            fig.add_hline(y=params["upper"], line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=params["lower"], line_dash="dash", line_color="green", row=2, col=1)
            entries = sig_df[sig_df["event"] == "buy"].index
            exits = sig_df[sig_df["event"] == "sell"].index
        elif indicator_choice == "MACD":
            fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df["macd"], name="MACD", line=dict(color="#1f77b4")), row=2, col=1)
            fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df["signal_line"], name="Signal", line=dict(color="#ff7f0e")), row=2, col=1)
            fig.add_trace(go.Bar(x=sig_df.index, y=sig_df["hist"], name="Hist", marker_color="rgba(150,150,150,0.5)"), row=2, col=1)
            entries = sig_df[sig_df["event"] == "bull_cross"].index
            exits = sig_df[sig_df["event"] == "bear_cross"].index
        elif indicator_choice == "Anchored VWAP":
            fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df["avwap"], name="AVWAP",
                                     line=dict(color="#9467bd", width=2)), row=1, col=1)
            anchor_idx = int(sig_df["anchor_idx"].iloc[-1])
            anchor_ts = close.index[anchor_idx]
            anchor_price = float(close.iloc[anchor_idx])
            # Vertical "anchor" marker as a scatter trace (avoids plotly add_vline
            # datetime-arithmetic bug in newer pandas).
            fig.add_trace(go.Scatter(
                x=[anchor_ts, anchor_ts],
                y=[float(close.min()) * 0.98, float(close.max()) * 1.02],
                mode="lines",
                line=dict(color="#9467bd", dash="dot", width=1),
                name="Anchor",
                hovertext=f"Anchor: {anchor_ts.strftime('%Y-%m-%d')} @ ${anchor_price:.2f}",
                hoverinfo="text",
                showlegend=False,
            ), row=1, col=1)
            entries = sig_df[sig_df["event"] == "cross_up"].index
            exits = sig_df[sig_df["event"] == "cross_down"].index
        elif indicator_choice == "Order Blocks + FVG":
            # draw zones as horizontal rectangles
            if params["use_obs"]:
                obs = ind.order_blocks(df,
                                        displacement_atr_mult=float(params["displacement_atr_mult"]),
                                        swing_lookback=int(params["swing_lookback"]))
                for _, r in obs.iterrows():
                    color = "rgba(46,204,113,0.18)" if r["kind"] == "bull" else "rgba(231,76,60,0.18)"
                    fig.add_shape(type="rect", x0=r["time"], x1=close.index[-1],
                                  y0=r["bottom"], y1=r["top"], fillcolor=color,
                                  line=dict(width=0), row=1, col=1)
            if params["use_fvgs"]:
                fvgs = ind.fair_value_gaps(df)
                for _, r in fvgs.iterrows():
                    color = "rgba(46,204,113,0.10)" if r["kind"] == "bull" else "rgba(231,76,60,0.10)"
                    fig.add_shape(type="rect", x0=r["time"], x1=close.index[-1],
                                  y0=r["bottom"], y1=r["top"], fillcolor=color,
                                  line=dict(width=0), row=1, col=1)
            entries = sig_df[sig_df["event"] == "long_tap"].index
            exits = sig_df[sig_df["event"] == "short_tap"].index
        elif indicator_choice == "Break of Structure":
            entries = sig_df[sig_df["event"] == "bull_bos"].index
            exits = sig_df[sig_df["event"] == "bear_bos"].index
        else:  # Liquidity Sweep
            entries = sig_df[sig_df["event"] == "sweep_low"].index
            exits = sig_df[sig_df["event"] == "sweep_high"].index

        fig.add_trace(go.Scatter(x=entries, y=close.reindex(entries),
                                 mode="markers", name="Long entry",
                                 marker=dict(symbol="triangle-up", size=10, color="green")), row=1, col=1)
        if not long_only:
            fig.add_trace(go.Scatter(x=exits, y=close.reindex(exits),
                                     mode="markers", name="Short entry",
                                     marker=dict(symbol="triangle-down", size=10, color="red")), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=exits, y=close.reindex(exits),
                                     mode="markers", name="Exit",
                                     marker=dict(symbol="x", size=8, color="gray")), row=1, col=1)

        fig.update_layout(height=620, hovermode="x", margin=dict(l=10, r=10, t=30, b=10))
        ch.add_crosshair(fig)
        st.plotly_chart(fig, use_container_width=True)

        # ---- equity curve
        ec = pd.DataFrame({
            "Strategy": metrics["equity_curve"],
            "Buy & Hold": metrics["buyhold_curve"],
        })
        st.markdown("### Equity Curve")
        st.plotly_chart(ch.line_chart(ec, height=320), use_container_width=True)

        # ---- trade log
        if metrics["trades"]:
            st.markdown("### Trade Log")
            tdf = pd.DataFrame(metrics["trades"])
            tdf["pnl_pct"] = (tdf["pnl_pct"] * 100).round(2)
            st.dataframe(tdf, use_container_width=True, height=300)


# =============================================================================
# Heatmap (sector + S&P 500) + Value Screen
# =============================================================================
with tab_heat:
    st.subheader("Market Heatmap")
    st.caption("Live % moves across sectors and S&P 500 stocks. Color = % change. Size of S&P boxes = average dollar volume.")

    sub_sec, sub_sp, sub_val = st.tabs(["Sectors", "S&P 500", "Value Screen"])

    period = st.session_state.get("hm_period", "5d")

    # ---- Sector heatmap
    with sub_sec:
        c1, c2 = st.columns([1, 4])
        with c1:
            sec_period = st.selectbox("Period", ["5d", "1mo", "3mo", "6mo", "ytd", "1y"],
                                      index=0, key="sec_period")
        if st.button("Refresh sectors"):
            st.cache_data.clear()

        @st.cache_data(ttl=300)
        def _sec_data(p):
            return hm.get_sector_data(period=p)

        @st.cache_data(ttl=300)
        def _sec_fig(p):
            return hm.sector_treemap(period=p)

        with st.spinner("Loading sector ETFs..."):
            try:
                sec_df = _sec_data(sec_period)
                fig = _sec_fig(sec_period)
            except Exception as e:
                st.error(f"Failed: {e}")
                sec_df = pd.DataFrame()
                fig = None

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        if not sec_df.empty:
            tbl = sec_df.copy()
            tbl["Price"] = tbl["price"].apply(lambda v: f"${v:,.2f}")
            tbl["1d %"] = tbl["change_1d"].apply(lambda v: f"{v:+.2f}%")
            tbl[f"{sec_period} %"] = tbl["change_period"].apply(lambda v: f"{v:+.2f}%")
            st.dataframe(
                tbl[["ticker", "sector", "Price", "1d %", f"{sec_period} %"]]
                   .rename(columns={"ticker": "ETF", "sector": "Sector"}),
                use_container_width=True, hide_index=True, height=440,
            )

    # ---- S&P 500 heatmap
    with sub_sp:
        c1, c2, c3 = st.columns([1, 1, 3])
        with c1:
            sp_period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "ytd", "1y"],
                                     index=1, key="sp_period")
        with c2:
            sort_by = st.selectbox("Top movers", ["Gainers", "Losers"], key="sp_sort")

        @st.cache_data(ttl=300)
        def _sp_data(p):
            return hm.get_sp500_heatmap_data(period=p)

        @st.cache_data(ttl=300)
        def _sp_fig(p):
            return hm.sp500_treemap(period=p)

        with st.spinner("Loading S&P 500 (~10–15s)..."):
            try:
                sp_df = _sp_data(sp_period)
                fig = _sp_fig(sp_period)
            except Exception as e:
                st.error(f"Failed: {e}")
                sp_df = pd.DataFrame()
                fig = None

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        if not sp_df.empty:
            color_col = "change_1d" if sp_period == "1d" else "change_period"
            ascending = (sort_by == "Losers")
            top = sp_df.sort_values(color_col, ascending=ascending).head(20).copy()
            top["Price"] = top["price"].apply(lambda v: f"${v:,.2f}")
            top["1d %"] = top["change_1d"].apply(lambda v: f"{v:+.2f}%")
            top[f"{sp_period} %"] = top["change_period"].apply(lambda v: f"{v:+.2f}%")
            top["Avg $ Volume"] = top["dollar_volume"].apply(fmt.pretty_money)
            st.markdown(f"### Top 20 {sort_by} ({sp_period})")
            st.dataframe(
                top[["ticker", "Security", "GICS Sector", "Price", "1d %", f"{sp_period} %", "Avg $ Volume"]]
                   .rename(columns={"ticker": "Symbol", "GICS Sector": "Sector"}),
                use_container_width=True, hide_index=True, height=420,
            )

    # ---- Value Screen
    with sub_val:
        st.markdown(
            "Multi-factor S&P 500 screen: technicals (RSI / drawdown / 200d SMA) "
            "+ fundamentals (P/E, P/B, ROE, debt/equity, EV/EBITDA), ranked by a "
            "composite value score. Not investment advice."
        )

        st.markdown("**Technical filters**")
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            rsi_band = st.selectbox(
                "RSI band",
                ["any", "oversold", "bearish", "bullish", "overbought"],
                index=0,
                help="oversold <30 · bearish 30–50 · bullish 50–70 · overbought >70",
            )
        with tc2:
            rsi_tf = st.radio("RSI timeframe", ["daily", "weekly"], index=0,
                               horizontal=True,
                               help="Daily = RSI(14) on daily closes. "
                                    "Weekly = RSI(14) on weekly closes (slower, less noise).")
        with tc3:
            min_dd = st.slider("Min % off 52-wk high", 0, 60, 0,
                                help="0 = no constraint")
        with tc4:
            below_200 = st.checkbox("Below 200d SMA", value=False)

        st.markdown("**Fundamental filters** (yfinance — first run takes ~60s, cached)")
        use_fund = st.checkbox("Apply fundamentals", value=True,
                                help="Uncheck to skip fundamentals fetch — much faster.")
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        with fc1:
            max_pe = st.number_input("Max P/E", 0.0, 200.0, 25.0, step=1.0,
                                      disabled=not use_fund)
        with fc2:
            max_pb = st.number_input("Max P/B", 0.0, 50.0, 5.0, step=0.5,
                                      disabled=not use_fund)
        with fc3:
            min_roe = st.number_input("Min ROE %", -50.0, 100.0, 10.0, step=1.0,
                                       disabled=not use_fund)
        with fc4:
            max_de = st.number_input("Max Debt/Equity", 0.0, 1000.0, 200.0, step=10.0,
                                      help="yfinance reports D/E as a percentage (200 = 2.0×).",
                                      disabled=not use_fund)
        with fc5:
            max_ev = st.number_input("Max EV/EBITDA", 0.0, 100.0, 20.0, step=1.0,
                                      disabled=not use_fund)

        run = st.button("Run screen", type="primary")

        if run:
            spinner_msg = ("Downloading 2y of price data + fundamentals for 500 stocks "
                           "(~60s on first run, cached after)..." if use_fund else
                           "Downloading 2y of price data for 500 stocks (~20s)...")
            with st.spinner(spinner_msg):
                try:
                    @st.cache_data(ttl=1800)
                    def _value(rsi_band, rsi_tf, min_dd, below, use_fund,
                                max_pe, max_pb, min_roe, max_de, max_ev):
                        return hm.value_screen(
                            top_n=100,
                            rsi_band=rsi_band,
                            rsi_timeframe=rsi_tf,
                            min_drawdown_pct=min_dd,
                            require_below_200d=below,
                            use_fundamentals=use_fund,
                            max_pe=max_pe if use_fund else None,
                            max_pb=max_pb if use_fund else None,
                            min_roe_pct=min_roe if use_fund else None,
                            max_debt_equity=max_de if use_fund else None,
                            max_ev_ebitda=max_ev if use_fund else None,
                        )

                    vdf = _value(rsi_band, rsi_tf, min_dd, below_200, use_fund,
                                  max_pe, max_pb, min_roe, max_de, max_ev)
                except Exception as e:
                    st.error(f"Screen failed: {e}")
                    vdf = pd.DataFrame()

            if vdf.empty:
                st.warning("No stocks matched the criteria. Try loosening the filters.")
            else:
                rsi_label = f"RSI({rsi_tf[0]}14)"
                show = vdf.copy()
                show["Price"] = show["price"].apply(lambda v: f"${v:,.2f}")
                show["52w High"] = show["52w_high"].apply(lambda v: f"${v:,.2f}")
                show["Drawdown"] = show["drawdown_%"].apply(lambda v: f"{v:+.1f}%")
                show[rsi_label] = show["rsi"].apply(lambda v: f"{v:.1f}")
                show["Below 200d"] = show["below_200d"].apply(lambda v: "Yes" if v else "No")

                cols = ["ticker", "Security", "GICS Sector", "Price", "52w High",
                        "Drawdown", rsi_label, "Below 200d"]
                if use_fund and "score" in show.columns:
                    show["Score"] = show["score"].apply(lambda v: f"{v:.2f}")
                    show["P/E"] = show["pe"].apply(lambda v: f"{v:.1f}" if pd.notna(v) and v > 0 else "—")
                    show["P/B"] = show["pb"].apply(lambda v: f"{v:.1f}" if pd.notna(v) and v > 0 else "—")
                    show["ROE"] = show["roe"].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—")
                    show["D/E"] = show["debt_equity"].apply(lambda v: f"{v/100:.2f}×" if pd.notna(v) else "—")
                    show["EV/EBITDA"] = show["ev_ebitda"].apply(lambda v: f"{v:.1f}" if pd.notna(v) and v > 0 else "—")
                    show["Div Yld"] = show["div_yield"].apply(
                        lambda v: f"{v*100:.2f}%" if pd.notna(v) and v > 0 else "—")
                    cols = ["ticker", "Security", "GICS Sector", "Score",
                            "Price", "Drawdown", rsi_label,
                            "P/E", "P/B", "ROE", "D/E", "EV/EBITDA", "Div Yld"]

                st.dataframe(
                    show[cols].rename(columns={"ticker": "Symbol", "GICS Sector": "Sector"}),
                    use_container_width=True, hide_index=True, height=600,
                )
                st.caption(f"{len(vdf)} stocks matched. RSI computed on {rsi_tf} closes (RSI 14-period).")

                # Stash latest screen results in session state for the AI panel section
                st.session_state["last_screen"] = vdf.head(20).to_dict(orient="records")

        # ---- AI Panel: multi-model consensus ----
        st.markdown("---")
        st.markdown("### 🤖 Ask AI Panel")
        configured = [name for name, key in
                       [("Claude", ANTHROPIC_KEY), ("GPT", OPENAI_KEY),
                        ("Gemini", GEMINI_KEY), ("Groq", GROQ_KEY),
                        ("OpenRouter", OPENROUTER_KEY)]
                       if key]
        if not configured:
            st.info(
                "Add at least one provider API key to `.streamlit/secrets.toml` to enable.\n\n"
                "**Free options** (recommended):\n"
                "- **Gemini** — `GEMINI_API_KEY` from "
                "[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)\n"
                "- **Groq** — `GROQ_API_KEY` from "
                "[console.groq.com/keys](https://console.groq.com/keys) (Llama 3.3 70B, very fast)\n"
                "- **OpenRouter** — `OPENROUTER_API_KEY` from "
                "[openrouter.ai/keys](https://openrouter.ai/keys) (DeepSeek + others, `:free` models)\n\n"
                "All three are different model families → a genuine free consensus."
            )
        else:
            st.caption(f"Configured providers: **{', '.join(configured)}**. "
                        f"Sends the **top 20** of your most recent screen to each model in parallel "
                        f"and computes consensus.")
            ask = st.button("Ask AI panel", type="secondary",
                             disabled=("last_screen" not in st.session_state))
            if "last_screen" not in st.session_state:
                st.caption("⚠️ Run a screen above first.")
            if ask and "last_screen" in st.session_state:
                with st.spinner(f"Querying {', '.join(configured)} in parallel..."):
                    per_model = ai.multi_model_panel(
                        st.session_state["last_screen"],
                        claude_key=ANTHROPIC_KEY,
                        openai_key=OPENAI_KEY,
                        gemini_key=GEMINI_KEY,
                        groq_key=GROQ_KEY,
                        openrouter_key=OPENROUTER_KEY,
                    )
                # Show errors for any provider that failed
                err_msgs = []
                for name, results in per_model.items():
                    if results and isinstance(results[0], dict) and "_error" in results[0]:
                        err_msgs.append(f"**{name}**: {results[0]['_error']}")
                if err_msgs:
                    st.error("Some providers errored:\n\n" + "\n\n".join(err_msgs))

                consensus = ai.consensus_table(per_model, st.session_state["last_screen"])
                if consensus.empty:
                    st.warning("No usable rankings returned.")
                else:
                    # Color verdict cells
                    verdict_color = {
                        "Strong Buy": "#1f7a1f", "Buy": "#3aa83a", "Hold": "#888",
                        "Avoid": "#a02020", "Value Trap": "#5a0a0a", "—": "#333",
                    }

                    def _verdict_pill(v):
                        c = verdict_color.get(v, "#444")
                        return (f'<span style="background:{c};color:white;padding:2px 8px;'
                                f'border-radius:10px;font-size:0.85em;">{v}</span>')

                    rows_html = []
                    rows_html.append('<table style="width:100%;border-collapse:collapse;">')
                    headers = ["Ticker", "Security"]
                    for name in configured:
                        headers.append(f"{name}")
                    headers += ["Avg rank", "Consensus", "Thesis"]
                    rows_html.append(
                        "<tr>" +
                        "".join(f'<th style="text-align:left;padding:6px;border-bottom:1px solid #333;'
                                f'color:#aaa;font-size:0.85em;">{h}</th>' for h in headers) +
                        "</tr>"
                    )
                    for _, r in consensus.iterrows():
                        cells = [
                            f'<b>{r["ticker"]}</b>',
                            f'<span style="color:#aaa;">{r["Security"]}</span>',
                        ]
                        for name in configured:
                            v = r.get(f"{name} verdict", "—")
                            rk = r.get(f"{name} rank")
                            rk_str = f" #{int(rk)}" if pd.notna(rk) else ""
                            cells.append(_verdict_pill(v) + f' <span style="color:#888;font-size:0.8em;">{rk_str}</span>')
                        cells.append(f'{r["avg_rank"]:.1f}' if pd.notna(r["avg_rank"]) else "—")
                        cells.append(_verdict_pill(r["consensus"]))
                        thesis = r.get("thesis", "")
                        risk = r.get("key_risk", "")
                        cells.append(
                            f'<div style="font-size:0.85em;color:#ddd;">{thesis}</div>'
                            f'<div style="font-size:0.78em;color:#999;margin-top:2px;">⚠ {risk}</div>'
                            if thesis else "—"
                        )
                        rows_html.append(
                            "<tr>" +
                            "".join(f'<td style="padding:8px 6px;border-bottom:1px solid #222;vertical-align:top;">{c}</td>'
                                    for c in cells) +
                            "</tr>"
                        )
                    rows_html.append("</table>")
                    st.markdown("".join(rows_html), unsafe_allow_html=True)

                    # Strong-consensus picks call-out
                    strong = consensus[
                        (consensus["agreement"] >= max(2, len(configured) - 1)) &
                        (consensus["consensus"].isin(["Strong Buy", "Buy"]))
                    ]
                    if not strong.empty:
                        st.markdown("##### ✅ Consensus picks")
                        st.markdown(
                            ", ".join(f"**{t}**" for t in strong["ticker"].head(10))
                        )


# =============================================================================
# Economic Calendar (Finnhub + FRED)
# =============================================================================
with tab_econ_cal:
    st.subheader("Economic Calendar")
    st.caption("Upcoming + recent economic events. Source: Finnhub (forecast / actual / prior).")

    if not FINNHUB_KEY:
        st.warning("No FINNHUB_KEY in `.streamlit/secrets.toml`.")
    else:
        # ---- Key releases highlight (curated, ignores impact filter) ----
        KEY_RELEASE_PATTERNS = [
            "Non Farm Payrolls", "Nonfarm Payrolls", "Unemployment Rate",
            "Initial Jobless Claims", "ADP Employment",
            "CPI", "Core CPI", "PCE", "Core PCE", "PPI", "Core PPI",
            "Retail Sales MoM", "Retail Sales Ex Autos",
            "GDP Growth", "GDP QoQ",
            "Fed Interest Rate", "FOMC", "Fed Chair",
            "ISM Manufacturing PMI", "ISM Services PMI",
            "JOLTs Job Openings",
            "Michigan Consumer Sentiment",
        ]

        @st.cache_data(ttl=900)
        def _key_releases(k):
            d = cal.economic_calendar(k, days_ahead=30, days_back=0,
                                      countries=["US"], min_impact="low")
            if d.empty:
                return d
            pat = "|".join(KEY_RELEASE_PATTERNS)
            d = d[d["event"].str.contains(pat, case=False, na=False, regex=True)]
            # Dedupe near-identical entries (e.g. "CPI" vs "CPI s.a")
            d = d.sort_values(["time", "impact"], ascending=[True, False])
            return d.reset_index(drop=True)

        with st.spinner("Loading key US releases..."):
            try:
                key_df = _key_releases(FINNHUB_KEY)
            except Exception as e:
                st.error(f"Finnhub error: {e}")
                key_df = pd.DataFrame()

        st.markdown("### Key US Releases (next 30 days)")
        if key_df.empty:
            st.info("No major US releases scheduled.")
        else:
            st.dataframe(
                fmt.format_econ_calendar(key_df, drop_country=True),
                use_container_width=True, height=360, hide_index=True,
            )

        st.markdown("---")
        st.markdown("### Full Calendar")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        with c1:
            days_back = st.number_input("Days back", 0, 30, 1, key="ec_back")
        with c2:
            days_ahead = st.number_input("Days ahead", 1, 60, 14, key="ec_fwd")
        with c3:
            min_impact = st.selectbox("Min impact", ["low", "medium", "high"], index=1, key="ec_imp")
        with c4:
            country_filter = st.multiselect(
                "Countries (blank = all)",
                ["US", "EU", "GB", "JP", "CN", "DE", "FR", "CA", "AU", "CH", "IN"],
                default=["US"],
                key="ec_countries",
            )

        @st.cache_data(ttl=900)
        def _econ_cal(k, da, db, c, mi):
            return cal.economic_calendar(k, days_ahead=da, days_back=db,
                                         countries=c or None, min_impact=mi)

        with st.spinner("Loading calendar..."):
            try:
                df = _econ_cal(FINNHUB_KEY, days_ahead, days_back, country_filter, min_impact)
            except Exception as e:
                st.error(f"Finnhub error: {e}")
                df = pd.DataFrame()

        if df.empty:
            st.info("No events for the selected filters.")
        else:
            today = pd.Timestamp(dt.date.today())
            upcoming = df[df["time"] >= today]
            past = df[df["time"] < today]
            single_country = len(country_filter) == 1

            st.markdown(f"### Upcoming ({len(upcoming)})")
            st.dataframe(
                fmt.format_econ_calendar(upcoming, drop_country=single_country),
                use_container_width=True, height=420, hide_index=True,
            )

            with st.expander(f"Recent ({len(past)}) — past {days_back} day(s)"):
                st.dataframe(
                    fmt.format_econ_calendar(past, drop_country=single_country),
                    use_container_width=True, height=300, hide_index=True,
                )



# =============================================================================
# Earnings & IPOs (Finnhub)
# =============================================================================
with tab_earnings:
    st.subheader("Earnings & IPO Calendars")
    if not FINNHUB_KEY:
        st.warning("No FINNHUB_KEY in `.streamlit/secrets.toml`.")
    else:
        sub_e, sub_i = st.tabs(["Earnings", "IPOs"])

        with sub_e:
            c1, c2, c3 = st.columns([1, 1, 3])
            with c1:
                e_back = st.number_input("Days back", 0, 30, 1, key="e_back")
            with c2:
                e_fwd = st.number_input("Days ahead", 1, 60, 14, key="e_fwd")
            with c3:
                e_syms = st.text_input("Filter symbols (comma-separated, blank = all)", value="", key="e_syms")

            symbols = [s.strip().upper() for s in e_syms.split(",") if s.strip()] or None

            @st.cache_data(ttl=900)
            def _earn(k, da, db, syms):
                return cal.earnings_calendar(k, days_ahead=da, days_back=db, symbols=syms)

            with st.spinner("Loading earnings..."):
                try:
                    edf = _earn(FINNHUB_KEY, e_fwd, e_back, tuple(symbols) if symbols else None)
                except Exception as ex:
                    st.error(f"Finnhub error: {ex}")
                    edf = pd.DataFrame()

            if edf.empty:
                st.info("No earnings in selected window/filter.")
            else:
                edf = nw.add_names(edf, ticker_col="symbol")
                st.dataframe(fmt.format_earnings(edf),
                             use_container_width=True, height=560, hide_index=True)
                st.caption(f"{len(edf)} earnings events.")

        with sub_i:
            i_back = st.number_input("Days back", 0, 60, 7, key="i_back")
            i_fwd = st.number_input("Days ahead", 1, 90, 30, key="i_fwd")

            @st.cache_data(ttl=1800)
            def _ipo(k, da, db):
                return cal.ipo_calendar(k, days_ahead=da, days_back=db)

            with st.spinner("Loading IPOs..."):
                try:
                    idf = _ipo(FINNHUB_KEY, i_fwd, i_back)
                except Exception as ex:
                    st.error(f"Finnhub error: {ex}")
                    idf = pd.DataFrame()

            if idf.empty:
                st.info("No IPOs scheduled in this window.")
            else:
                st.dataframe(fmt.format_ipos(idf),
                             use_container_width=True, height=500, hide_index=True)


# =============================================================================
# Fund Holdings (13F)
# =============================================================================
with tab_funds:
    st.subheader("Institutional Holdings (13F-HR)")
    st.caption("Quarterly filings via SEC EDGAR. ~45-day lag is the legal reality. Crypto is generally NOT in 13F (only certain ETFs like IBIT, FBTC).")

    col1, col2 = st.columns([1, 2])
    with col1:
        mode = st.radio("Pick fund", ["From famous list", "Search by name", "Enter CIK"])
    with col2:
        cik = None
        if mode == "From famous list":
            choice = st.selectbox("Fund", list(hl.FAMOUS_FUNDS.keys()))
            cik = hl.FAMOUS_FUNDS[choice]
        elif mode == "Search by name":
            q = st.text_input("Company / fund name", value="")
            if q:
                try:
                    results = hl.search_company(q)
                    if results:
                        labels = [r["name"] for r in results]
                        pick = st.selectbox("Match", labels)
                        cik = results[labels.index(pick)]["cik"]
                    else:
                        st.warning("No matches.")
                except Exception as e:
                    st.error(f"Search failed: {e}")
        else:
            cik = st.text_input("CIK (advanced)", value="0001067983",
                                help="The 10-digit SEC identifier for the filer.")

    if cik:
        try:
            with st.spinner("Loading filings list..."):
                filings = hl.get_filings(cik, "13F-HR")
        except Exception as e:
            st.error(f"Failed to load filings: {e}")
            filings = pd.DataFrame()

        # Use only original 13F-HR filings (skip amendments) for the period picker
        primary = filings[filings["form"] == "13F-HR"].sort_values("reportDate", ascending=False).reset_index(drop=True)

        if primary.empty:
            st.warning("No 13F-HR filings found for this fund.")
        else:
            st.success(f"**{primary.iloc[0]['entity_name']}** — {len(primary)} quarterly filings on file")

            # Build clean period labels: "Q4 2025" instead of dates + accession numbers
            def quarter_label(d):
                q = (d.month - 1) // 3 + 1
                return f"Q{q} {d.year}"

            options = [quarter_label(row.reportDate) for row in primary.itertuples()]
            c1, c2 = st.columns(2)
            with c1:
                latest_idx = c1.selectbox("Quarter to view", options, index=0)
            with c2:
                prev_default = 1 if len(options) > 1 else 0
                prev_idx = c2.selectbox("Compare to", options, index=prev_default)
            i_curr = options.index(latest_idx)
            i_prev = options.index(prev_idx)

            if st.button("Load holdings", type="primary"):
                with st.spinner("Downloading filings from SEC..."):
                    curr = hl.get_holdings(cik, primary.iloc[i_curr]["accessionNumber"])
                    prev = hl.get_holdings(cik, primary.iloc[i_prev]["accessionNumber"]) if i_prev != i_curr else pd.DataFrame()

                if curr.empty:
                    st.error("Could not parse current filing.")
                else:
                    total_value = curr["value"].sum()
                    n_positions = curr["issuer"].nunique()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Portfolio Value", fmt.pretty_money(total_value))
                    m2.metric("Positions", f"{n_positions}")
                    m3.metric("Filed", primary.iloc[i_curr]["filingDate"].strftime("%b %d, %Y"))

                    st.markdown(f"### Top Positions — {options[i_curr]}")
                    agg = curr.groupby("issuer", as_index=False).agg(
                        value=("value", "sum"), shares=("shares", "sum"),
                        cusip=("cusip", "first"),
                    ).sort_values("value", ascending=False)
                    agg["%_of_portfolio"] = (agg["value"] / agg["value"].sum() * 100).round(2)

                    # Look up tickers for the top 50 positions only (rate-limit friendly).
                    top_agg = agg.head(50).copy()
                    with st.spinner("Resolving CUSIP → tickers..."):
                        try:
                            top_cusips = top_agg["cusip"].dropna().astype(str).str.zfill(9).tolist()
                            mapping = hl.cusip_to_ticker_batch(top_cusips)
                            top_agg["ticker"] = top_agg["cusip"].astype(str).str.zfill(9).map(mapping).fillna("")
                        except Exception:
                            top_agg["ticker"] = ""
                    st.dataframe(fmt.format_holdings(top_agg),
                                 use_container_width=True, height=400, hide_index=True)

                    if not prev.empty:
                        st.markdown(f"### Changes: {options[i_prev]} → {options[i_curr]}")
                        diff = hl.compare_holdings(prev, curr)

                        new_top = diff[diff["status"] == "NEW"].sort_values(
                            "value_curr", ascending=False).head(20)
                        inc_top = diff[diff["status"] == "INCREASED"].sort_values(
                            "value_curr", ascending=False).head(20)
                        red_top = diff[diff["status"].isin(["REDUCED", "EXITED"])].sort_values(
                            "share_change", ascending=True).head(20)

                        # Look up tickers ONLY for the rows we'll actually display
                        with st.spinner("Resolving tickers for changes..."):
                            try:
                                cusips_to_lookup = pd.concat([
                                    new_top["cusip"], inc_top["cusip"], red_top["cusip"],
                                ]).dropna().astype(str).str.zfill(9).unique().tolist()
                                ch_map = hl.cusip_to_ticker_batch(cusips_to_lookup)

                                def _attach(d):
                                    if d.empty:
                                        return d
                                    d = d.copy()
                                    d["ticker"] = d["cusip"].astype(str).str.zfill(9).map(ch_map).fillna("")
                                    return d

                                new_top = _attach(new_top)
                                inc_top = _attach(inc_top)
                                red_top = _attach(red_top)
                            except Exception:
                                pass

                        cc1, cc2, cc3 = st.columns(3)
                        cc1.markdown("**New positions**")
                        cc1.dataframe(fmt.format_changes(new_top, "new"),
                                       use_container_width=True, height=300, hide_index=True)
                        cc2.markdown("**Increased**")
                        cc2.dataframe(fmt.format_changes(inc_top, "increased"),
                                       use_container_width=True, height=300, hide_index=True)
                        cc3.markdown("**Reduced / Exited**")
                        cc3.dataframe(fmt.format_changes(red_top, "reduced"),
                                       use_container_width=True, height=300, hide_index=True)


# =============================================================================
# Economic Indicators
# =============================================================================
with tab_econ:
    st.subheader("US Economic Indicators (FRED)")
    st.caption("Latest values for major macro releases. Free, no API key. For an upcoming-event calendar, get a FRED API key (free) and we'll add it.")

    if st.button("Refresh snapshot"):
        st.cache_data.clear()

    @st.cache_data(ttl=3600)
    def _snap():
        return econ.latest_snapshot()

    with st.spinner("Loading FRED data..."):
        snap = _snap()
    st.dataframe(snap, use_container_width=True, height=560)

    st.markdown("### Chart an indicator")
    pick = st.selectbox("Indicator", list(econ.INDICATORS.keys()))
    if pick:
        try:
            df = econ.get_indicator(pick)
            df = df.set_index("date")
            series = df["value"].rename(pick)
            st.plotly_chart(ch.line_chart(series, title=pick, height=420),
                            use_container_width=True)
            st.caption(f"FRED series: `{econ.INDICATORS[pick]['id']}` · transform: `{econ.INDICATORS[pick]['transform']}`")
        except Exception as e:
            st.error(f"Failed to fetch: {e}")


# =============================================================================
# News
# =============================================================================
def _news_card(headline: str, url: str, source: str, when: str,
               summary: str = "", image: str = "") -> str:
    """Return an HTML news-card string. Used inside st.markdown(unsafe_allow_html=True)."""
    img_html = (
        f'<img src="{image}" '
        'style="width:120px;height:80px;object-fit:cover;border-radius:6px;'
        'flex:none;background:#1a1a1a;" '
        'onerror="this.style.display=\'none\'"/>'
    ) if image else ""
    summary_html = (
        f'<div style="color:#aaa;font-size:0.85em;margin-top:4px;'
        'overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;'
        '-webkit-box-orient:vertical;">{}</div>'
    ).format(summary) if summary else ""
    return f'''
<div style="display:flex;gap:12px;padding:10px;margin-bottom:8px;
            background:#16191f;border:1px solid #2a2e36;border-radius:8px;">
  {img_html}
  <div style="flex:1;min-width:0;">
    <a href="{url}" target="_blank" style="color:#e6e6e6;text-decoration:none;
       font-weight:600;font-size:0.95em;line-height:1.3;">{headline}</a>
    <div style="color:#888;font-size:0.78em;margin-top:4px;">{source} · {when}</div>
    {summary_html}
  </div>
</div>'''


with tab_news:
    st.subheader("News & Sentiment")
    st.caption("Headlines from Finnhub + Google News, plus Stocktwits community sentiment.")

    sub_market, sub_ticker = st.tabs(["Market News", "By Ticker"])

    # ---- Market news ----
    with sub_market:
        if not FINNHUB_KEY:
            st.warning("No FINNHUB_KEY in `.streamlit/secrets.toml`.")
        else:
            ncc1, ncc2 = st.columns([3, 2])
            with ncc1:
                cat = st.radio("Category", ["general", "forex", "crypto", "merger"],
                               horizontal=True, key="market_cat")
            with ncc2:
                macro_only = st.toggle("Macro only", value=True, key="macro_only",
                                        help="Filter to Fed/inflation/jobs/GDP/rates and central-bank stories.")

            @st.cache_data(ttl=600)
            def _mkt_news(k, c):
                return nw.market_news(k, category=c, limit=80)

            with st.spinner("Loading headlines..."):
                try:
                    mdf = _mkt_news(FINNHUB_KEY, cat)
                except Exception as e:
                    st.error(f"Finnhub error: {e}")
                    mdf = pd.DataFrame()

            # Macro filter — keyword match across headline + summary + source + tags
            if macro_only and not mdf.empty:
                MACRO_KEYWORDS = [
                    # US macro
                    "fed", "fomc", "powell", "rate cut", "rate hike", "interest rate",
                    "inflation", "cpi", "ppi", "pce", "core cpi",
                    "jobs", "payroll", "nfp", "unemployment", "labor",
                    "gdp", "recession", "growth", "consumer", "retail sales",
                    "ism", "pmi", "manufacturing", "services",
                    "treasury", "yield", "bond", "10-year", "10y",
                    "tariff", "trade", "trump", "white house", "fiscal", "deficit",
                    # central banks / global macro
                    "ecb", "boe", "bank of england", "boj", "bank of japan", "pboc",
                    "central bank", "yen", "dollar", "currency",
                    # markets-defining
                    "spy", "s&p 500", "nasdaq", "vix", "earnings season",
                ]
                pattern = "|".join([k.replace(" ", r"\s+") for k in MACRO_KEYWORDS])
                hay = (
                    mdf["headline"].fillna("").astype(str) + " | " +
                    mdf.get("summary", pd.Series([""] * len(mdf))).fillna("").astype(str) + " | " +
                    mdf.get("category", pd.Series([""] * len(mdf))).fillna("").astype(str)
                )
                mdf = mdf[hay.str.contains(pattern, case=False, regex=True, na=False)]

            if mdf.empty:
                st.info("No headlines match. Toggle macro filter or try another category.")
            else:
                cards = []
                for _, r in mdf.iterrows():
                    when = r["datetime"].strftime("%a %b %-d, %-I:%M %p") if pd.notna(r.get("datetime")) else ""
                    cards.append(_news_card(
                        headline=r.get("headline", ""),
                        url=r.get("url", "#"),
                        source=r.get("source", ""),
                        when=when,
                        summary=(r.get("summary") or "")[:240],
                        image=r.get("image", "") or "",
                    ))
                st.markdown("".join(cards), unsafe_allow_html=True)

    # ---- Ticker-specific news ----
    with sub_ticker:
        symbol = st.text_input("Ticker", value="NVDA", key="news_ticker").upper().strip()
        days = st.slider("Days back", 1, 30, 7, key="news_days")

        if symbol:
            company = nw.name_for(symbol)
            if company:
                st.markdown(f"### {symbol} — {company}")
            else:
                st.markdown(f"### {symbol}")

            cn_col, st_col = st.columns([2, 1])

            # Finnhub company news
            with cn_col:
                st.markdown("#### Headlines (Finnhub)")
                if not FINNHUB_KEY:
                    st.warning("No Finnhub key.")
                else:
                    @st.cache_data(ttl=600)
                    def _co_news(k, s, d):
                        return nw.company_news(k, s, days_back=d, limit=30)

                    try:
                        cdf = _co_news(FINNHUB_KEY, symbol, days)
                    except Exception as e:
                        st.error(f"Finnhub error: {e}")
                        cdf = pd.DataFrame()

                    if cdf.empty:
                        st.info("No company news in window.")
                    else:
                        cards = []
                        for _, r in cdf.head(15).iterrows():
                            when = r["datetime"].strftime("%a %b %-d, %-I:%M %p") if pd.notna(r.get("datetime")) else ""
                            cards.append(_news_card(
                                headline=r.get("headline", ""),
                                url=r.get("url", "#"),
                                source=r.get("source", ""),
                                when=when,
                                summary=(r.get("summary") or "")[:200],
                                image=r.get("image", "") or "",
                            ))
                        st.markdown("".join(cards), unsafe_allow_html=True)

                # Google News RSS — broader web coverage
                st.markdown("#### Web Coverage (Google News)")

                @st.cache_data(ttl=600)
                def _gnews(q):
                    return nw.google_news(q, limit=15)

                q = f"{symbol} {company}" if company else symbol
                try:
                    gdf = _gnews(q)
                except Exception as e:
                    st.error(f"Google News error: {e}")
                    gdf = pd.DataFrame()

                if gdf.empty:
                    st.info("No web results.")
                else:
                    cards = []
                    for _, r in gdf.head(10).iterrows():
                        when = r["time"].strftime("%a %b %-d") if pd.notna(r.get("time")) else ""
                        cards.append(_news_card(
                            headline=r.get("headline", ""),
                            url=r.get("url", "#"),
                            source=r.get("source", ""),
                            when=when,
                        ))
                    st.markdown("".join(cards), unsafe_allow_html=True)

            # Stocktwits sentiment + posts
            with st_col:
                st.markdown("#### Stocktwits Sentiment")

                @st.cache_data(ttl=300)
                def _stwits(s):
                    return nw.stocktwits(s, limit=30)

                sdf = _stwits(symbol)
                if sdf.empty or "error" in sdf.columns:
                    st.info("Stocktwits unavailable for this ticker.")
                else:
                    sent = nw.stocktwits_sentiment(sdf)
                    bull = sent["bullish"]
                    bear = sent["bearish"]
                    ratio = sent["ratio"]
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Bullish", bull)
                    cc2.metric("Bearish", bear)
                    cc3.metric("Bull %",
                               f"{ratio*100:.0f}%" if ratio is not None else "—")

                    st.markdown("**Recent posts**")
                    posts = []
                    for _, r in sdf.head(15).iterrows():
                        sentiment = r.get("sentiment") or ""
                        bar_color = "#1f7a1f" if sentiment == "Bullish" else "#a02020" if sentiment == "Bearish" else "#444"
                        body = (r.get("body") or "").replace("\n", " ").replace("<", "&lt;")[:200]
                        when = r["time"].strftime("%b %-d %H:%M") if pd.notna(r.get("time")) else ""
                        user = r.get("user", "") or ""
                        posts.append(f'''
<div style="display:flex;gap:8px;padding:8px 10px;margin-bottom:6px;
            background:#16191f;border-left:3px solid {bar_color};border-radius:4px;">
  <div style="flex:1;min-width:0;">
    <div style="color:#bbb;font-size:0.78em;">@{user} · {when}</div>
    <div style="color:#e6e6e6;font-size:0.88em;margin-top:3px;">{body}</div>
  </div>
</div>''')
                    st.markdown("".join(posts), unsafe_allow_html=True)


# =============================================================================
# Crypto Derivatives — OI, funding, long/short ratio, real liquidations
# =============================================================================
with tab_crypto:
    st.subheader("Crypto Derivatives")
    st.caption("Open interest, funding rates, long/short ratio, and live liquidations from OKX (no key).")

    cc1, cc2 = st.columns([2, 1])
    with cc1:
        crypto_sym = st.text_input("Underlying", value="BTC",
                                   help="Base coin: BTC, ETH, SOL, etc. Quote = USDT.").upper().strip()
    with cc2:
        crypto_int = st.selectbox("Interval", ["5m", "1h", "4h", "1d"], index=1)

    if not crypto_sym:
        st.stop()

    @st.cache_data(ttl=120)
    def _ck(sym, interval):
        return lq.okx_klines(f"{sym}-USDT-SWAP", interval=interval, limit=200)

    @st.cache_data(ttl=120)
    def _coi(sym, interval):
        return lq.okx_open_interest_history(sym, interval=interval, limit=200)

    @st.cache_data(ttl=300)
    def _cfund(sym):
        return lq.okx_funding_rate_history(f"{sym}-USDT-SWAP", limit=100)

    @st.cache_data(ttl=180)
    def _clsr(sym, interval):
        return lq.okx_long_short_ratio(sym, interval=interval, limit=200)

    @st.cache_data(ttl=60)
    def _cliq(sym):
        return lq.okx_liquidations(sym, limit=100)

    with st.spinner(f"Loading {crypto_sym} derivatives..."):
        try:
            kdf = _ck(crypto_sym, crypto_int)
            oi = _coi(crypto_sym, crypto_int)
            funding = _cfund(crypto_sym)
            lsr = _clsr(crypto_sym, crypto_int)
            liq = _cliq(crypto_sym)
        except Exception as e:
            st.error(f"OKX request failed: {e}")
            st.stop()

    # ---- Summary metrics ----
    m = st.columns(4)
    if not kdf.empty:
        last_px = kdf["close"].iloc[-1]
        chg = (kdf["close"].iloc[-1] / kdf["close"].iloc[0] - 1) * 100
        m[0].metric(f"{crypto_sym}-USDT", f"${last_px:,.2f}", delta=f"{chg:+.2f}% ({len(kdf)} bars)")
    if not funding.empty:
        m[1].metric("Last Funding", f"{funding['funding_rate'].iloc[-1]:+.4f}%",
                    help="Annualized ≈ funding × 3 × 365")
    if not lsr.empty:
        m[2].metric("Long/Short Ratio", f"{lsr['long_short_ratio'].iloc[-1]:.2f}",
                    help=">1 means more longs than shorts")
    if not liq.empty:
        total_long = liq.loc[liq["liq_side"] == "long", "notional"].sum()
        total_short = liq.loc[liq["liq_side"] == "short", "notional"].sum()
        m[3].metric("Recent Liquidations",
                    f"${(total_long + total_short) / 1e6:.1f}M",
                    delta=f"L ${total_long/1e6:.1f}M · S ${total_short/1e6:.1f}M",
                    help=f"Last {len(liq)} liquidation events")

    # ---- Bias panel (combines funding + L/S + OI + liquidations) ----
    bias = lq.derivatives_bias(kdf, funding, lsr, oi, liq)
    bias_html = f'''
<div style="background:#16191f;border:1px solid #2a2e36;border-radius:10px;
            padding:14px 18px;margin:10px 0 6px;">
  <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
    <div>
      <div style="color:#888;font-size:0.78em;text-transform:uppercase;letter-spacing:0.05em;">
        Derivatives Bias
      </div>
      <div style="display:flex;align-items:baseline;gap:10px;">
        <span style="font-size:1.6em;font-weight:700;color:{bias['color']};">
          {bias['verdict']}
        </span>
        <span style="color:#aaa;font-size:0.9em;">score {bias['total_score']:+.2f}</span>
        {('<span style="background:#553;color:#ffe680;padding:1px 8px;border-radius:8px;'
          'font-size:0.75em;">CROWDED</span>') if bias['is_crowded'] else ''}
      </div>
    </div>
    <div style="flex:1;min-width:300px;display:grid;grid-template-columns:auto 1fr;
                gap:4px 14px;font-size:0.84em;color:#ccc;">
      <div style="color:#888;">Funding</div><div>{bias['funding_note'] or '—'}</div>
      <div style="color:#888;">L/S Ratio</div><div>{bias['ls_note'] or '—'}</div>
      <div style="color:#888;">OI vs Price</div><div>{bias['oi_note'] or '—'}</div>
      <div style="color:#888;">Liquidations</div><div>{bias['liq_note'] or '—'}</div>
    </div>
  </div>
</div>
'''
    st.markdown(bias_html, unsafe_allow_html=True)

    st.markdown("---")

    # ---- Price + OI chart with liquidation overlay ----
    if not kdf.empty:
        # Both subplots share x-axis. Constrain to the overlap of klines + OI windows
        # so neither panel is squashed.
        t_start = kdf["time"].min()
        t_end = kdf["time"].max()
        if not oi.empty:
            t_start = max(t_start, oi["time"].min())
            t_end = min(t_end, oi["time"].max())

        kdf_view = kdf[(kdf["time"] >= t_start) & (kdf["time"] <= t_end)]
        oi_view = oi[(oi["time"] >= t_start) & (oi["time"] <= t_end)] if not oi.empty else oi

        fig_oi = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.72, 0.28], vertical_spacing=0.04,
                                subplot_titles=("Price + Liquidations", "Open Interest"))
        fig_oi.add_trace(go.Candlestick(
            x=kdf_view["time"], open=kdf_view["open"], high=kdf_view["high"],
            low=kdf_view["low"], close=kdf_view["close"], name=crypto_sym,
            showlegend=False, increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ), row=1, col=1)

        # Overlay liquidation markers — only show meaningful sizes, simpler visual
        if not liq.empty:
            liq_view = liq[(liq["time"] >= t_start) & (liq["time"] <= t_end)].copy()
            if not liq_view.empty:
                # Filter to top-N by notional, then by quantile threshold to hide noise
                threshold = max(50_000, liq_view["notional"].quantile(0.6))
                liq_view = liq_view[liq_view["notional"] >= threshold]

            if not liq_view.empty:
                # Smoother sizing: square root of notional (in $M), capped
                import numpy as _np
                sizes = (_np.sqrt(liq_view["notional"] / 1e6) * 5 + 6).clip(lower=6, upper=24)
                longs = liq_view[liq_view["liq_side"] == "long"]
                shorts = liq_view[liq_view["liq_side"] == "short"]
                if len(longs):
                    fig_oi.add_trace(go.Scatter(
                        x=longs["time"], y=longs["price"],
                        mode="markers", name="Long liq",
                        marker=dict(symbol="circle", color="rgba(239,83,80,0.55)",
                                    size=sizes.loc[longs.index],
                                    line=dict(width=1, color="rgba(239,83,80,1.0)")),
                        hovertemplate="<b>LONG liquidated</b><br>%{x|%b %-d %H:%M}<br>"
                                      "Price: $%{y:,.2f}<br>Notional: $%{customdata:,.0f}<extra></extra>",
                        customdata=longs["notional"],
                    ), row=1, col=1)
                if len(shorts):
                    fig_oi.add_trace(go.Scatter(
                        x=shorts["time"], y=shorts["price"],
                        mode="markers", name="Short liq",
                        marker=dict(symbol="circle", color="rgba(38,166,154,0.55)",
                                    size=sizes.loc[shorts.index],
                                    line=dict(width=1, color="rgba(38,166,154,1.0)")),
                        hovertemplate="<b>SHORT liquidated</b><br>%{x|%b %-d %H:%M}<br>"
                                      "Price: $%{y:,.2f}<br>Notional: $%{customdata:,.0f}<extra></extra>",
                        customdata=shorts["notional"],
                    ), row=1, col=1)

        if not oi_view.empty:
            fig_oi.add_trace(go.Scatter(
                x=oi_view["time"], y=oi_view["open_interest"], name="OI",
                line=dict(color="#9467bd", width=1.5), fill="tozeroy",
                fillcolor="rgba(148,103,189,0.15)",
                showlegend=False,
            ), row=2, col=1)
        fig_oi.update_layout(height=620, hovermode="x unified",
                             margin=dict(l=10, r=10, t=40, b=10),
                             xaxis_rangeslider_visible=False,
                             legend=dict(orientation="h", yanchor="bottom",
                                         y=1.02, xanchor="right", x=1))
        ch.add_crosshair(fig_oi)
        st.plotly_chart(fig_oi, use_container_width=True)

    # ---- Funding + L/S ratio ----
    fl1, fl2 = st.columns(2)
    with fl1:
        st.markdown("##### Funding Rate History")
        if funding.empty:
            st.info("No funding history.")
        else:
            ffig = go.Figure()
            colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in funding["funding_rate"]]
            ffig.add_trace(go.Bar(x=funding["time"], y=funding["funding_rate"],
                                   marker_color=colors, name="Funding %"))
            ffig.add_hline(y=0, line_color="#888", line_width=1)
            ffig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                yaxis_title="%", showlegend=False)
            ch.add_crosshair(ffig)
            st.plotly_chart(ffig, use_container_width=True)
    with fl2:
        st.markdown("##### Long/Short Ratio (top traders)")
        if lsr.empty:
            st.info("No long/short data.")
        else:
            lfig = go.Figure()
            lfig.add_trace(go.Scatter(x=lsr["time"], y=lsr["long_short_ratio"],
                                       line=dict(color="#1f77b4"), name="L/S"))
            lfig.add_hline(y=1, line_dash="dash", line_color="#888")
            lfig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                yaxis_title="ratio", showlegend=False)
            ch.add_crosshair(lfig)
            st.plotly_chart(lfig, use_container_width=True)

    # ---- Liquidation table ----
    st.markdown("##### Recent Liquidations (OKX, BTC-USDT/ETH-USDT/etc. perp)")
    if liq.empty:
        st.info("No liquidations in the recent window.")
    else:
        show = liq.head(50).copy()
        show["Time"] = show["time"].dt.strftime("%b %-d %H:%M:%S")
        show["Side"] = show["liq_side"].map(lambda x: "🔻 Long" if x == "long" else "🔺 Short")
        show["Price"] = show["price"].map(lambda x: f"${x:,.2f}")
        show["Size"] = show["size"].map(lambda x: f"{x:,.4f}")
        show["Notional"] = show["notional"].map(lambda x: f"${x:,.0f}")
        st.dataframe(show[["Time", "Side", "Price", "Size", "Notional"]],
                     hide_index=True, use_container_width=True, height=420)

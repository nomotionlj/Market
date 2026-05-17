"""Trading tab — Alpaca paper-trading rebalance UI.

Three sections:
1. Account & positions overview
2. Rebalance preview (build a Plan against the latest or selected snapshot)
3. Execute (with mode-aware confirmation gate)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

import broker
import quant_pipeline as qp


SCREENS_DIR = Path(__file__).parent / "screens"
INDEX_FILE = SCREENS_DIR / "index.json"
LIVE_CONFIRM_PHRASE = "I UNDERSTAND THIS IS REAL MONEY"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _secrets_dict() -> Dict:
    keys = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_LIVE",
            "DISCORD_WEBHOOK_URL", "NTFY_TOPIC", "SLACK_WEBHOOK_URL",
            "SMTP_HOST", "SMTP_USER", "SMTP_PASS", "SMTP_TO"]
    return {k: st.secrets.get(k, "") for k in keys if hasattr(st, "secrets")}


def _load_index() -> list:
    if not INDEX_FILE.exists():
        return []
    try:
        return json.loads(INDEX_FILE.read_text())
    except Exception:
        return []


def _load_snapshot(filename: str) -> Optional[dict]:
    fpath = SCREENS_DIR / filename
    if not fpath.exists():
        return None
    try:
        return json.loads(fpath.read_text())
    except Exception:
        return None


def _pos_to_df(positions: list) -> pd.DataFrame:
    if not positions:
        return pd.DataFrame()
    rows = []
    for p in positions:
        rows.append({
            "Symbol": p.get("symbol"),
            "Qty": float(p.get("qty") or 0),
            "Avg cost": float(p.get("avg_entry_price") or 0),
            "Last": float(p.get("current_price") or 0),
            "Mkt value": float(p.get("market_value") or 0),
            "Unrealized $": float(p.get("unrealized_pl") or 0),
            "Unrealized %": float(p.get("unrealized_plpc") or 0) * 100,
        })
    df = pd.DataFrame(rows).sort_values("Mkt value", ascending=False)
    return df


def _format_pos_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["Avg cost"] = out["Avg cost"].apply(lambda v: f"${v:,.2f}")
    out["Last"] = out["Last"].apply(lambda v: f"${v:,.2f}")
    out["Mkt value"] = out["Mkt value"].apply(lambda v: f"${v:,.2f}")
    out["Unrealized $"] = out["Unrealized $"].apply(
        lambda v: f"{'+' if v>=0 else '−'}${abs(v):,.2f}")
    out["Unrealized %"] = out["Unrealized %"].apply(lambda v: f"{v:+.2f}%")
    out["Qty"] = out["Qty"].apply(lambda v: f"{v:,.4f}")
    return out


def _plan_to_df(plan: broker.Plan) -> pd.DataFrame:
    if not plan.actions:
        return pd.DataFrame()
    rows = []
    for a in plan.actions:
        rows.append({
            "Side": "🔻 SELL" if a.side == "sell" else "🟢 BUY",
            "Symbol": a.symbol,
            "Notional": f"${a.notional:,.2f}",
            "Cur value": f"${a.current_value:,.2f}",
            "Target value": f"${a.target_value:,.2f}",
            "Last": f"${a.last_price:,.2f}" if a.last_price else "—",
            "Note": a.note,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render() -> None:
    secrets = _secrets_dict()
    if not secrets.get("ALPACA_API_KEY") or not secrets.get("ALPACA_SECRET_KEY"):
        st.warning(
            "Add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` to "
            "`.streamlit/secrets.toml` to enable trading."
        )
        with st.expander("How to get free paper-trading keys"):
            st.markdown(
                """1. Sign up at https://app.alpaca.markets (free)
2. Toggle the top-left selector to **Paper**
3. Right sidebar → **API Keys** → **Generate** (paper)
4. Copy both values into `.streamlit/secrets.toml`:

```toml
ALPACA_API_KEY = "PK..."
ALPACA_SECRET_KEY = "..."
ALPACA_LIVE = "false"
```

5. Restart Streamlit.

Paper accounts start with $100k of fake cash. No bank link, no SSN, no risk.
"""
            )
        return

    # ---- Mode selector (always defaults to PAPER) ----
    mode_col, _ = st.columns([1, 3])
    with mode_col:
        secrets_default_live = str(secrets.get("ALPACA_LIVE", "")).lower() in ("1", "true", "yes")
        mode_label = st.radio(
            "Mode", ["📄 Paper", "💵 Live"],
            index=1 if secrets_default_live else 0,
            horizontal=True, key="trade_mode",
        )
    use_live = mode_label.startswith("💵")
    if use_live:
        st.error("⚠ **LIVE mode** — orders will execute against your real Alpaca account.")

    # ---- Build client ----
    try:
        client = broker.make_client(secrets, live_override=use_live)
    except broker.AlpacaError as e:
        st.error(f"Could not create Alpaca client: {e}")
        return

    # ---- Account + positions ----
    st.markdown("### Account")
    col_refresh, _ = st.columns([1, 5])
    if col_refresh.button("🔄 Refresh", key="trade_refresh"):
        st.rerun()

    try:
        acct = client.account()
        positions = client.positions()
    except broker.AlpacaError as e:
        st.error(f"Alpaca error: {e}")
        return

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Equity", f"${float(acct.get('equity') or 0):,.2f}")
    a2.metric("Cash", f"${float(acct.get('cash') or 0):,.2f}")
    a3.metric("Buying power", f"${float(acct.get('buying_power') or 0):,.2f}")
    a4.metric("Positions", f"{len(positions)}")

    if positions:
        st.markdown("##### Holdings")
        df = _pos_to_df(positions)
        st.dataframe(_format_pos_df(df), hide_index=True,
                      use_container_width=True, height=min(45 + 32 * len(df), 360))
    else:
        st.caption("No open positions.")

    # ---- Rebalance ----
    st.markdown("---")
    st.markdown("### Rebalance plan")

    index = _load_index()
    if not index:
        st.info("No snapshots yet. Run a screen first (Run Screen tab).")
        return

    rc1, rc2, rc3, rc4 = st.columns([2, 1, 1, 1])
    with rc1:
        snap_options = [(e["file"],
                         f"{e['timestamp'][:19].replace('T',' ')} · "
                         f"{e['preset']} · n={e['n_returned']}")
                        for e in index]
        labels = [label for _, label in snap_options]
        picked_label = st.selectbox("Target snapshot", labels, index=0,
                                       key="trade_snapshot")
        picked_file = dict((label, fname) for fname, label in snap_options)[picked_label]
    with rc2:
        top_n = st.number_input("Hold top-N", 5, 50, 20, step=5, key="trade_top_n")
    with rc3:
        cash_reserve_pct = st.slider("Cash reserve %", 0.0, 25.0, 2.0, step=0.5,
                                       key="trade_cash") / 100.0
    with rc4:
        threshold_pct = st.slider("Drift threshold %", 0.0, 50.0, 10.0, step=1.0,
                                    key="trade_threshold",
                                    help="Skip rebalance if a position's drift is less than this fraction of its target.") / 100.0

    rc5, rc6, rc7 = st.columns([1, 1, 1])
    liquidate = rc5.checkbox("Sell positions outside target list", value=True,
                              key="trade_liquidate")
    min_order = rc6.number_input("Min order $", 1.0, 1000.0, 1.0, step=1.0,
                                   key="trade_min_order")
    max_sector = rc7.slider(
        "Max sector %", 5.0, 100.0, 30.0, step=5.0, key="trade_max_sector",
        help="No GICS sector will exceed this fraction of the portfolio. "
             "Set to 100 to disable.",
    ) / 100.0

    if st.button("Build plan", key="trade_build"):
        snap = _load_snapshot(picked_file)
        if not snap:
            st.error("Could not load snapshot.")
            return
        picks_data = (snap.get("picks") or [])[: int(top_n)]
        targets = [p["ticker"] for p in picks_data]
        ticker_sectors = {p["ticker"]: (p.get("sector") or "Unknown")
                           for p in picks_data if p.get("ticker")}

        kwargs = dict(
            client=client,
            target_tickers=targets,
            cash_reserve_pct=cash_reserve_pct,
            rebalance_threshold_pct=threshold_pct,
            min_order_usd=float(min_order),
            liquidate_outside_targets=liquidate,
            snapshot_file=picked_file,
            preset=snap.get("preset", ""),
        )
        if max_sector < 1.0 - 1e-9:
            kwargs["ticker_sectors"] = ticker_sectors
            kwargs["default_sector_cap"] = max_sector

        with st.spinner("Fetching account, positions, and quotes..."):
            try:
                plan = broker.build_equal_weight_plan(**kwargs)
            except broker.AlpacaError as e:
                st.error(f"Failed to build plan: {e}")
                return
        st.session_state["last_plan"] = plan
        st.session_state["last_plan_mode"] = client.mode

    plan = st.session_state.get("last_plan")
    plan_mode = st.session_state.get("last_plan_mode")

    if plan and plan_mode == client.mode:
        st.markdown("##### Plan summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Buys", f"{len(plan.buys)}",
                   f"${plan.total_buy_notional:,.0f}")
        s2.metric("Sells", f"{len(plan.sells)}",
                   f"-${plan.total_sell_notional:,.0f}")
        s3.metric("Net cash impact",
                   f"${(plan.total_sell_notional - plan.total_buy_notional):+,.0f}")
        s4.metric("Targets", f"{len(plan.target_universe)}")

        if plan.warnings:
            for w in plan.warnings:
                st.warning(f"⚠ {w}")

        # Sector breakdown card
        try:
            breakdown = broker.sector_breakdown(plan)
        except Exception:
            breakdown = []
        if breakdown:
            with st.expander(f"🏷️  Sector allocation ({len(breakdown)} sectors)",
                              expanded=False):
                rows = []
                for b in breakdown:
                    rows.append({
                        "Sector": b["sector"],
                        "Target %": f"{b['target_pct']*100:.1f}%",
                        "Target $": f"${b['target_value']:,.0f}",
                        "Stocks": f"{len(b['tickers'])}",
                        "Tickers": ", ".join(b["tickers"][:8])
                                    + ("…" if len(b["tickers"]) > 8 else ""),
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                              use_container_width=True)

        if plan.actions:
            st.dataframe(_plan_to_df(plan), hide_index=True,
                          use_container_width=True,
                          height=min(45 + 32 * len(plan.actions), 480))
        else:
            st.success("Already balanced — no orders needed.")
            return

        # ---- Execute gate ----
        st.markdown("---")
        st.markdown(f"##### Execute ({plan.mode})")
        if plan.mode == "LIVE":
            st.error(
                f"This will submit {len(plan.actions)} **real** market orders. "
                f"Type the phrase below exactly to enable the submit button."
            )
            phrase_input = st.text_input(
                f"Type: {LIVE_CONFIRM_PHRASE}",
                key="trade_confirm_phrase", placeholder=LIVE_CONFIRM_PHRASE,
            )
            confirm_ok = phrase_input.strip() == LIVE_CONFIRM_PHRASE
            btn_label = "🔴 Submit LIVE orders"
        else:
            confirm_ok = st.checkbox(
                "I've reviewed the plan and want to submit these PAPER orders.",
                key="trade_confirm_paper",
            )
            btn_label = "📄 Submit paper orders"

        if st.button(btn_label, type="primary", disabled=not confirm_ok,
                       key="trade_submit"):
            with st.spinner(f"Submitting {len(plan.actions)} orders..."):
                res = broker.execute(client, plan)
            if res.n_err == 0:
                st.success(f"✓ {res.n_ok} order(s) submitted successfully.")
            else:
                st.warning(f"{res.n_ok} ok, {res.n_err} errors:")
                for sym, msg in res.errors:
                    st.error(f"{sym}: {msg}")
            # Clear plan so user must re-build before re-submitting
            st.session_state.pop("last_plan", None)
            st.session_state.pop("last_plan_mode", None)
            st.session_state.pop("trade_confirm_paper", None)
            st.session_state.pop("trade_confirm_phrase", None)
    elif plan and plan_mode != client.mode:
        st.info(f"Plan was built in {plan_mode} mode — switch back or re-build.")

    # ---- Open orders preview ----
    st.markdown("---")
    st.markdown("##### Open orders")
    try:
        open_o = client.open_orders()
    except broker.AlpacaError as e:
        st.caption(f"(could not fetch: {e})")
        open_o = []
    if not open_o:
        st.caption("No open orders.")
    else:
        rows = [{
            "Time": o.get("submitted_at", "")[:19].replace("T", " "),
            "Symbol": o.get("symbol"),
            "Side": o.get("side"),
            "Type": o.get("type"),
            "Qty/Notional": (o.get("qty") or f"${o.get('notional','')}"),
            "Status": o.get("status"),
        } for o in open_o]
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                      use_container_width=True)
        if st.button("Cancel all open orders", key="trade_cancel_all"):
            try:
                client.cancel_all_orders()
                st.success("Cancellation request sent.")
            except broker.AlpacaError as e:
                st.error(f"Cancel failed: {e}")

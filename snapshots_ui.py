"""Snapshot viewer for the Quant Screen.

Reads JSON files written by picker.py and renders:
- A table of recent snapshots
- A detail view of one snapshot's picks
- A buy/sell diff between two snapshots
- A button to trigger a new snapshot run inline
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

import notifier
import quant_pipeline as qp

SCREENS_DIR = Path(__file__).parent / "screens"
INDEX_FILE = SCREENS_DIR / "index.json"


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


def _picks_to_df(picks: list) -> pd.DataFrame:
    if not picks:
        return pd.DataFrame()
    df = pd.DataFrame(picks)
    if "quant_score" in df.columns:
        df = df.sort_values("quant_score", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def _format_picks_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame()
    out["#"] = df.get("rank", df.index + 1)
    out["Symbol"] = df.get("ticker", "")
    out["Security"] = df.get("security", "")
    out["Sector"] = df.get("sector", "")
    if "quant_score" in df.columns:
        out["Score"] = df["quant_score"].apply(
            lambda v: f"{v:.3f}" if pd.notna(v) else "—"
        )
    if "f_score" in df.columns:
        out["F"] = df["f_score"].apply(
            lambda v: f"{int(v)}/9" if pd.notna(v) else "—"
        )
    if "altman_z" in df.columns:
        out["Z"] = df["altman_z"].apply(
            lambda v: f"{v:.1f}" if pd.notna(v) else "—"
        )
    if "earnings_yield" in df.columns:
        out["EarnYld"] = df["earnings_yield"].apply(
            lambda v: f"{v*100:.1f}%" if pd.notna(v) and v > 0 else "—"
        )
    if "roic" in df.columns:
        out["ROIC"] = df["roic"].apply(
            lambda v: f"{v*100:.1f}%" if pd.notna(v) and v > 0 else "—"
        )
    if "accruals" in df.columns:
        out["Accruals"] = df["accruals"].apply(
            lambda v: f"{v*100:+.1f}%" if pd.notna(v) else "—"
        )
    if "asset_growth" in df.columns:
        out["AssetGrowth"] = df["asset_growth"].apply(
            lambda v: f"{v*100:+.1f}%" if pd.notna(v) else "—"
        )
    if "eps_surprise_pct" in df.columns:
        out["EPS Surp"] = df["eps_surprise_pct"].apply(
            lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"
        )
    if "eps_beat_streak" in df.columns:
        out["Beat Streak"] = df["eps_beat_streak"].apply(
            lambda v: f"{int(v)}" if pd.notna(v) else "—"
        )
    if "mom_12_1" in df.columns:
        out["Mom 12-1"] = df["mom_12_1"].apply(
            lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"
        )
    if "dist_52wh" in df.columns:
        out["%off 52WH"] = df["dist_52wh"].apply(
            lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"
        )
    if "insider_net_usd" in df.columns:
        def _fmt_ins(v):
            if pd.isna(v) or v == 0:
                return "—"
            sign = "+" if v > 0 else "−"
            return f"{sign}${abs(v) / 1e6:.1f}M"
        out["Insider 90d"] = df["insider_net_usd"].apply(_fmt_ins)
    if "price" in df.columns:
        out["Price"] = df["price"].apply(
            lambda v: f"${v:,.2f}" if pd.notna(v) else "—"
        )
    return out


def _config_for_alerts() -> dict:
    """Pull alert credentials from st.secrets so the UI can dispatch directly."""
    keys = ["DISCORD_WEBHOOK_URL", "NTFY_TOPIC", "NTFY_SERVER",
            "SLACK_WEBHOOK_URL",
            "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS",
            "SMTP_TO", "SMTP_FROM"]
    return {k: st.secrets.get(k, "") for k in keys if hasattr(st, "secrets")}


def render() -> None:
    st.markdown(
        "Saved snapshots from the headless `picker.py` runner. "
        "Use these for cron-scheduled weekly screens, or trigger a new one inline."
    )

    # ---- Alert configuration banner ----
    cfg = _config_for_alerts()
    channels = notifier.configured_channels(cfg)
    if channels:
        st.success(f"📢 Alerts configured: **{', '.join(channels)}**")
    else:
        with st.expander("📢 Set up alerts (Discord / ntfy / Slack / Email)",
                          expanded=False):
            st.markdown(
                """Add any of these to `.streamlit/secrets.toml` to receive buy/sell alerts:

**Discord (easiest)** — Server → Channel settings → Integrations → Webhooks → Copy URL
```toml
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."
```

**ntfy.sh (free push to phone, no signup)** — install the ntfy app, subscribe to a unique string
```toml
NTFY_TOPIC = "lj-quant-7x9k2random"
```

**Email via Gmail** — enable 2FA, then create an "App Password" at myaccount.google.com/apppasswords
```toml
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = "587"
SMTP_USER = "you@gmail.com"
SMTP_PASS = "your-16-char-app-password"
SMTP_TO = "you@gmail.com"
```
"""
            )

    # ---- Run-now button ----
    with st.expander("▶ Run new snapshot now", expanded=False):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            preset_run = st.selectbox(
                "Preset", list(qp.PRESETS.keys()), index=1, key="snap_run_preset",
            )
        with c2:
            top_n_run = st.number_input(
                "Top N", 5, 50, 20, step=5, key="snap_run_top_n",
            )
        with c3:
            send_alert = st.checkbox("Send alert", value=bool(channels),
                                      key="snap_run_alert", disabled=not channels,
                                      help="Sends through every configured channel after the run.")
        if st.button("Run + save snapshot", key="snap_run_btn"):
            log_box = st.empty()
            log_box.info("Running… first run with deep quant takes ~60–120s.")
            cmd = [sys.executable, str(Path(__file__).parent / "picker.py"),
                   "run", "--preset", preset_run, "--top-n", str(int(top_n_run))]
            if not send_alert:
                cmd.append("--no-alert")
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600,
                    cwd=str(Path(__file__).parent),
                )
                output = (result.stdout or "") + (result.stderr or "")
                if result.returncode == 0:
                    log_box.success(f"Done.\n```\n{output[-2000:]}\n```")
                else:
                    log_box.error(f"Picker exited with code {result.returncode}\n"
                                   f"```\n{output[-2000:]}\n```")
            except subprocess.TimeoutExpired:
                log_box.error("Picker timed out after 10 minutes.")
            except Exception as e:
                log_box.error(f"Failed to invoke picker: {e}")

    st.markdown("---")

    # ---- Recent snapshots index ----
    index = _load_index()
    if not index:
        st.info(
            "No snapshots yet. Run one above, or via CLI:\n```\n"
            "cd /Users/lj/market-hub && .venv/bin/python picker.py run\n```"
        )
        return

    st.markdown(f"### Recent snapshots ({len(index)})")
    idx_rows = []
    for entry in index[:30]:
        idx_rows.append({
            "When": entry["timestamp"][:19].replace("T", " "),
            "Preset": entry["preset"],
            "N": entry["n_returned"],
            "Top picks": ", ".join(entry["tickers"][:6])
                         + ("…" if len(entry["tickers"]) > 6 else ""),
            "File": entry["file"],
        })
    st.dataframe(pd.DataFrame(idx_rows), hide_index=True, use_container_width=True,
                  height=min(45 + 32 * min(len(idx_rows), 12), 420))

    st.markdown("---")

    # ---- View one snapshot in full ----
    st.markdown("### View snapshot")
    options = [(e["file"], f"{e['timestamp'][:19].replace('T',' ')} · "
                            f"{e['preset']} · n={e['n_returned']}")
               for e in index]
    file_lookup = {label: filename for filename, label in options}
    selected_label = st.selectbox(
        "Pick a snapshot",
        [label for _, label in options],
        index=0,
        key="snap_select",
    )
    sel = _load_snapshot(file_lookup[selected_label])
    if sel:
        st.caption(f"**{sel['preset']}** · {sel['timestamp']} · "
                    f"{sel['n_returned']} picks")
        df = _picks_to_df(sel.get("picks", []))
        st.dataframe(_format_picks_table(df), hide_index=True,
                      use_container_width=True, height=520)

    # ---- Alert preview / send ----
    st.markdown("---")
    st.markdown("### 📢 Alert preview")
    if not channels:
        st.caption("Configure at least one channel above to enable alert dispatch.")
    else:
        # Build alert from currently selected snapshot vs. its predecessor
        cur_file = file_lookup[selected_label]
        all_files = [e["file"] for e in index]
        try:
            i = all_files.index(cur_file)
            prev_file = all_files[i + 1] if i + 1 < len(all_files) else None
        except ValueError:
            prev_file = None
        prev_snap = _load_snapshot(prev_file) if prev_file else None
        sel = sel or {}
        sel_with_filename = dict(sel)
        sel_with_filename["_filename"] = cur_file
        alert = notifier.alert_from_snapshots(sel_with_filename, prev_snap)

        st.code(alert.to_plain_text(max_top=10), language="text")
        ac1, ac2, ac3 = st.columns([1, 1, 2])
        with ac1:
            if st.button("Send this alert", key="snap_send_alert"):
                with st.spinner("Sending..."):
                    results = notifier.dispatch_alert(alert, cfg)
                for ch, (ok, info) in results.items():
                    (st.success if ok else st.error)(f"{ch}: {info}")
        with ac2:
            if st.button("Send synthetic test", key="snap_send_test"):
                with st.spinner("Sending test..."):
                    test_alert = notifier.Alert(
                        title="Quant test alert",
                        preset="Long-term Value",
                        timestamp=alert.timestamp,
                        added=["WMT", "JPM"], removed=["NVR"],
                        held=["AAPL", "MSFT"],
                        top_picks=[{"ticker": "WMT", "score": 0.91, "security": "Walmart"}],
                    )
                    results = notifier.dispatch_alert(test_alert, cfg)
                for ch, (ok, info) in results.items():
                    (st.success if ok else st.error)(f"{ch}: {info}")
        with ac3:
            st.caption(f"Will dispatch to: **{', '.join(channels)}**")

    # ---- Buy/sell diff ----
    if len(index) >= 2:
        st.markdown("---")
        st.markdown("### Buy/sell delta")
        c1, c2 = st.columns(2)
        with c1:
            new_label = st.selectbox(
                "Latest (or any newer)",
                [label for _, label in options],
                index=0,
                key="diff_new",
            )
        with c2:
            old_label = st.selectbox(
                "Compare against",
                [label for _, label in options],
                index=1,
                key="diff_old",
            )
        new_snap = _load_snapshot(file_lookup[new_label])
        old_snap = _load_snapshot(file_lookup[old_label])
        if new_snap and old_snap:
            new_tickers = {p["ticker"] for p in new_snap.get("picks", [])}
            old_tickers = {p["ticker"] for p in old_snap.get("picks", [])}
            added = sorted(new_tickers - old_tickers)
            dropped = sorted(old_tickers - new_tickers)
            held = sorted(new_tickers & old_tickers)

            d1, d2, d3 = st.columns(3)
            with d1:
                st.metric("➕ ADD", len(added))
                if added:
                    st.markdown("\n".join(f"- **{t}**" for t in added))
                else:
                    st.caption("(none)")
            with d2:
                st.metric("➖ REMOVE", len(dropped))
                if dropped:
                    st.markdown("\n".join(f"- **{t}**" for t in dropped))
                else:
                    st.caption("(none)")
            with d3:
                st.metric("= HOLD", len(held))
                if held:
                    with st.expander(f"{len(held)} held"):
                        st.markdown(", ".join(f"`{t}`" for t in held))

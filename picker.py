"""Headless stock-picker runner.

Runs a Quant Screen preset and writes the top-N picks to a JSON snapshot.
Designed for cron scheduling.

Usage:
    # Run with default preset (Long-term Value), save top 20
    python picker.py

    # Pick a different preset and depth
    python picker.py --preset "Deep Value" --top-n 15

    # See past runs
    python picker.py --list
    python picker.py --show 2026-05-04T09-30-00

    # Cron entry — every Monday 9:30am ET (15:30 UTC, adjust for your TZ):
    #   30 9 * * 1 cd /Users/lj/market-hub && .venv/bin/python picker.py >> /tmp/picker.log 2>&1

Snapshot files:
    screens/<ISO-timestamp>.json   one per run
    screens/current.json           symlink (or copy) to the latest
    screens/index.json             rolling list of all snapshot filenames
"""
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

from typing import Optional

import pandas as pd

# Allow running both as `python picker.py` and `python -m picker`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import broker
import notifier
import quant_pipeline as qp
import whale_bias_history as wbh

SCREENS_DIR = Path(__file__).parent / "screens"
INDEX_FILE = SCREENS_DIR / "index.json"
CURRENT_FILE = SCREENS_DIR / "current.json"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_finnhub_key() -> str:
    """Pull FINNHUB_KEY from .streamlit/secrets.toml (same source as the app)."""
    try:
        secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
        if not secrets_path.exists():
            return ""
        for line in secrets_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("FINNHUB_KEY"):
                # naive parse: FINNHUB_KEY = "..."
                _, _, rhs = line.partition("=")
                return rhs.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _df_to_picks(df: pd.DataFrame, top_n: int) -> list:
    """Convert the ranked DataFrame into a list of clean dicts for JSON."""
    picks = []
    keep = [
        "ticker", "Security", "GICS Sector", "GICS Sub-Industry", "quant_score",
        "f_score", "altman_z", "altman_zone", "beneish_m",
        "earnings_yield", "roic", "accruals", "asset_growth",
        "eps_surprise_pct", "eps_beat_streak",
        "eps_surprise_avg_4q", "eps_surprise_z",
        "mom_12_1", "dist_52wh", "vol_1y", "insider_net_usd",
        "price", "drawdown_%", "rsi", "pe", "pb", "roe",
        "debt_equity", "ev_ebitda", "div_yield", "market_cap",
    ]
    for i, row in df.head(top_n).iterrows():
        pick = {"rank": int(i) + 1}
        for col in keep:
            if col not in row.index:
                continue
            v = row[col]
            if pd.isna(v):
                pick[col] = None
            elif hasattr(v, "item"):
                pick[col] = v.item()
            else:
                pick[col] = v
        # Friendly aliases
        pick["security"] = pick.pop("Security", None)
        pick["sector"] = pick.pop("GICS Sector", None)
        picks.append(pick)
    return picks


def write_snapshot(df: pd.DataFrame, preset: str, top_n: int,
                    output_dir: Path = SCREENS_DIR) -> Path:
    """Save a snapshot to disk and refresh the rolling index. Returns the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now()
    fname = ts.strftime("%Y-%m-%dT%H-%M-%S") + ".json"
    fpath = output_dir / fname

    snapshot = {
        "timestamp": ts.isoformat(),
        "timestamp_utc": ts.astimezone(dt.timezone.utc).isoformat(),
        "preset": preset,
        "filters": qp.PRESETS.get(preset, {}),
        "top_n": top_n,
        "n_returned": len(df),
        "picks": _df_to_picks(df, top_n),
    }
    fpath.write_text(json.dumps(snapshot, indent=2, default=str))

    # Update current.json (plain copy, works on every filesystem)
    CURRENT_FILE.write_text(fpath.read_text())

    # Update rolling index
    index = []
    if INDEX_FILE.exists():
        try:
            index = json.loads(INDEX_FILE.read_text())
        except Exception:
            index = []
    index.append({
        "file": fname,
        "timestamp": ts.isoformat(),
        "preset": preset,
        "top_n": top_n,
        "n_returned": len(df),
        "tickers": [p["ticker"] for p in snapshot["picks"]],
    })
    index = sorted(index, key=lambda x: x["timestamp"], reverse=True)
    INDEX_FILE.write_text(json.dumps(index, indent=2))
    return fpath


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _previous_snapshot(latest_filename: str) -> dict:
    """Return the snapshot immediately preceding `latest_filename` from the index."""
    if not INDEX_FILE.exists():
        return {}
    try:
        index = json.loads(INDEX_FILE.read_text())
    except Exception:
        return {}
    # index is sorted newest-first; find the entry one position after `latest_filename`.
    files = [e["file"] for e in index]
    if latest_filename not in files:
        return {}
    i = files.index(latest_filename)
    if i + 1 >= len(files):
        return {}
    prev_path = SCREENS_DIR / files[i + 1]
    if not prev_path.exists():
        return {}
    try:
        return json.loads(prev_path.read_text())
    except Exception:
        return {}


def _build_and_dispatch_alert(snapshot: dict, snapshot_filename: str,
                                quiet_unchanged: bool) -> int:
    """Construct an alert from the snapshot and send it through every configured channel.
    Returns the number of channels that succeeded."""
    snapshot["_filename"] = snapshot_filename
    previous = _previous_snapshot(snapshot_filename)

    alert = notifier.alert_from_snapshots(snapshot, previous or None)
    config = notifier.load_secrets()
    channels = notifier.configured_channels(config)

    if not channels:
        print("  (no alert channels configured — set DISCORD_WEBHOOK_URL / NTFY_TOPIC / "
              "SLACK_WEBHOOK_URL / SMTP_* in .streamlit/secrets.toml)")
        return 0

    print(f"  Dispatching alert through: {', '.join(channels)}")
    results = notifier.dispatch_alert(alert, config, suppress_unchanged=quiet_unchanged)
    n_ok = 0
    for ch, (ok, info) in results.items():
        marker = "✓" if ok else "✗"
        print(f"    {marker} {ch}: {info}")
        if ok and ch != "_suppressed":
            n_ok += 1
    return n_ok


def cmd_run(args) -> int:
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] "
          f"Running screen: preset={args.preset!r}, top_n={args.top_n}")
    finnhub_key = _load_finnhub_key()
    if not finnhub_key:
        print("  ⚠ No FINNHUB_KEY in .streamlit/secrets.toml — insider data disabled")

    df = qp.run_screen(
        preset_name=args.preset,
        top_n_quant=max(args.top_n, 30),
        final_top_n=args.top_n,
        finnhub_key=finnhub_key,
    )
    if df.empty:
        print("  No stocks matched the screen.")
        return 1

    fpath = write_snapshot(df, preset=args.preset, top_n=args.top_n)
    print(f"  ✓ Saved snapshot: {fpath.name}")
    print(f"  ✓ Updated current.json")
    print()
    print(f"  Top {args.top_n} picks:")
    for i, row in df.head(args.top_n).iterrows():
        score = row.get("quant_score", float("nan"))
        score_str = f"{score:.3f}" if pd.notna(score) else "—"
        print(f"    {int(i)+1:>2}. {row['ticker']:<6} {score_str}  "
              f"{row.get('Security', '')[:40]}")

    if not args.no_alert:
        print()
        snapshot = json.loads(fpath.read_text())
        _build_and_dispatch_alert(
            snapshot, fpath.name, quiet_unchanged=args.quiet_unchanged,
        )
    return 0


def cmd_test_alert(args) -> int:
    """Send a synthetic alert through every configured channel."""
    print("Sending test alert through every configured channel...")
    config = notifier.load_secrets()
    channels = notifier.configured_channels(config)
    if not channels:
        print("  No channels configured. Add at least one of "
              "DISCORD_WEBHOOK_URL / NTFY_TOPIC / SLACK_WEBHOOK_URL / SMTP_* "
              "to .streamlit/secrets.toml.")
        return 1
    print(f"  Channels: {', '.join(channels)}")

    alert = notifier.Alert(
        title="Quant picks · +2 / -1 (TEST)",
        preset="Long-term Value",
        timestamp=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        added=["WMT", "JPM"],
        removed=["NVR"],
        held=["AAPL", "MSFT", "GOOG", "BRK-B"],
        top_picks=[
            {"ticker": "WMT", "score": 0.912, "security": "Walmart Inc."},
            {"ticker": "AAPL", "score": 0.870, "security": "Apple Inc."},
            {"ticker": "MSFT", "score": 0.850, "security": "Microsoft Corp."},
        ],
        snapshot_file="(test)",
    )
    results = notifier.dispatch_alert(alert, config)
    n_ok = 0
    for ch, (ok, info) in results.items():
        marker = "✓" if ok else "✗"
        print(f"    {marker} {ch}: {info}")
        if ok and ch != "_suppressed":
            n_ok += 1
    return 0 if n_ok > 0 else 2


LIVE_CONFIRM_PHRASE = "I UNDERSTAND THIS IS REAL MONEY"


def _resolve_snapshot(target: Optional[str]) -> Optional[dict]:
    """Resolve `target` (None, 'current', date prefix, or filename) to a snapshot dict."""
    if not target or target == "current":
        if (SCREENS_DIR / "current.json").exists():
            return json.loads((SCREENS_DIR / "current.json").read_text())
        return None
    if target.endswith(".json"):
        path = SCREENS_DIR / target
    else:
        candidates = sorted(SCREENS_DIR.glob(f"{target}*.json"), reverse=True)
        if not candidates:
            return None
        path = candidates[0]
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _parse_sector_caps(spec: Optional[str]) -> dict:
    """Parse --sector-cap "Information Technology=0.25,Financials=0.20" into a dict."""
    if not spec:
        return {}
    out = {}
    for part in spec.split(","):
        if "=" not in part:
            continue
        sec, _, val = part.partition("=")
        sec = sec.strip()
        try:
            v = float(val.strip())
            if 0 < v <= 1:
                out[sec] = v
        except ValueError:
            continue
    return out


def cmd_rebalance(args) -> int:
    """Build (and optionally execute) an equal-weight rebalance against a snapshot."""
    secrets = notifier.load_secrets()
    if not secrets.get("ALPACA_API_KEY") or not secrets.get("ALPACA_SECRET_KEY"):
        print("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .streamlit/secrets.toml.")
        print("Get free paper-trading keys at https://app.alpaca.markets")
        return 1

    snapshot = _resolve_snapshot(args.snapshot)
    if not snapshot:
        print(f"Could not load snapshot: {args.snapshot!r}")
        return 1

    picks = snapshot.get("picks") or []
    targets = [p["ticker"] for p in picks][: args.top_n]
    ticker_sectors = {p["ticker"]: (p.get("sector") or "Unknown")
                       for p in picks if p.get("ticker")}
    if not targets:
        print("Snapshot contains no picks.")
        return 1

    if args.live:
        print()
        print("⚠⚠⚠  LIVE MODE — orders will execute against your real Alpaca account.")
        print(f"     Type the phrase exactly to confirm: {LIVE_CONFIRM_PHRASE}")
        try:
            entered = input("> ").strip()
        except EOFError:
            entered = ""
        if entered != LIVE_CONFIRM_PHRASE:
            print("Confirmation phrase did not match. Aborting.")
            return 2

    try:
        client = broker.make_client(secrets, live_override=bool(args.live))
    except broker.AlpacaError as e:
        print(f"Could not create client: {e}")
        return 1

    print(f"Mode: {client.mode}   Targets: {len(targets)} from {snapshot.get('preset','?')!r}")
    sector_caps_kwargs = {}
    parsed_caps = _parse_sector_caps(getattr(args, "sector_cap", ""))
    if parsed_caps:
        sector_caps_kwargs["sector_caps"] = parsed_caps
        print(f"Per-sector overrides: {parsed_caps}")
    if args.max_sector_pct is not None and args.max_sector_pct < 1.0:
        sector_caps_kwargs["default_sector_cap"] = args.max_sector_pct
        print(f"Default sector cap: {args.max_sector_pct*100:.0f}%")
    if sector_caps_kwargs:
        sector_caps_kwargs["ticker_sectors"] = ticker_sectors

    print(f"Building equal-weight plan ({args.cash_reserve*100:.1f}% cash reserve)...")
    try:
        plan = broker.build_equal_weight_plan(
            client,
            target_tickers=targets,
            cash_reserve_pct=args.cash_reserve,
            min_order_usd=args.min_order,
            rebalance_threshold_pct=args.threshold,
            liquidate_outside_targets=not args.no_liquidate,
            snapshot_file=snapshot.get("_filename", "current"),
            preset=snapshot.get("preset", ""),
            **sector_caps_kwargs,
        )
    except broker.AlpacaError as e:
        print(f"Failed to build plan: {e}")
        return 1

    # Print sector breakdown if caps were applied
    if sector_caps_kwargs:
        print()
        print("Sector breakdown after caps:")
        for b in broker.sector_breakdown(plan):
            print(f"  {b['sector']:<28} {b['target_pct']*100:>5.1f}%  "
                  f"({len(b['tickers'])} stk)  {', '.join(b['tickers'][:6])}"
                  + ("…" if len(b['tickers']) > 6 else ""))

    print()
    print(plan.render_text())
    print()

    if not plan.actions:
        return 0

    if args.dry_run:
        print("[dry-run] No orders sent. Re-run without --dry-run to execute.")
        return 0

    if args.live and not args.yes:
        print(f"⚠ LIVE — final check. Press ENTER to submit, or Ctrl-C to abort.")
        try:
            input()
        except KeyboardInterrupt:
            print("Aborted.")
            return 2

    print(f"Submitting {len(plan.actions)} orders...")
    res = broker.execute(client, plan)
    print(f"  ✓ {res.n_ok} ok    ✗ {res.n_err} errors")
    for sym, msg in res.errors:
        print(f"    ✗ {sym}: {msg}")

    # Optional: send a Discord/ntfy alert summarizing the rebalance
    if not args.no_alert:
        config = secrets
        channels = notifier.configured_channels(config)
        if channels:
            alert = notifier.Alert(
                title=f"Rebalance · {plan.mode}",
                preset=plan.preset or "Custom",
                timestamp=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                added=[a.symbol for a in plan.buys],
                removed=[a.symbol for a in plan.sells],
                held=targets,
                top_picks=[{"ticker": a.symbol,
                             "score": a.signed_notional,
                             "security": f"{a.side.upper()} ${a.notional:,.0f}"}
                            for a in plan.actions[:10]],
                snapshot_file=plan.snapshot_file,
            )
            notifier.dispatch_alert(alert, config)

    return 0 if res.n_err == 0 else 2


def cmd_account(args) -> int:
    """Show account + positions."""
    secrets = notifier.load_secrets()
    try:
        client = broker.make_client(secrets, live_override=bool(args.live))
    except broker.AlpacaError as e:
        print(f"Error: {e}")
        return 1

    try:
        acct = client.account()
        positions = client.positions()
    except broker.AlpacaError as e:
        print(f"API error: {e}")
        return 1

    print(f"Mode: {client.mode}")
    print(f"  Equity:        ${float(acct.get('equity') or 0):,.2f}")
    print(f"  Cash:          ${float(acct.get('cash') or 0):,.2f}")
    print(f"  Buying power:  ${float(acct.get('buying_power') or 0):,.2f}")
    print(f"  Status:        {acct.get('status')}")
    print()
    print(f"Positions ({len(positions)}):")
    for p in sorted(positions, key=lambda x: -float(x.get("market_value") or 0)):
        sym = p["symbol"]
        qty = float(p.get("qty") or 0)
        mv = float(p.get("market_value") or 0)
        ul = float(p.get("unrealized_plpc") or 0) * 100
        print(f"  {sym:<6} {qty:>10.4f} sh   ${mv:>12,.2f}   {ul:+.2f}%")
    return 0


def cmd_bias_snapshot(args) -> int:
    """Append a whale-bias-by-coin snapshot to ~/.market-hub/whale_bias_history.csv."""
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] "
          f"Taking whale-bias snapshot (top-{args.top_n} by {args.sort_by}/{args.window})")
    df = wbh.take_snapshot(top_n=args.top_n,
                             sort_by=args.sort_by,
                             window=args.window)
    if df.empty:
        print("  No data captured (no positions across top-N).")
        return 1
    n_long = int((df["net_notional"] > 0).sum())
    n_short = int((df["net_notional"] < 0).sum())
    total_long = float(df["long_notional"].sum())
    total_short = float(df["short_notional"].sum())
    print(f"  ✓ Wrote {len(df)} coin rows.")
    print(f"  Net long coins: {n_long}  ·  Net short coins: {n_short}")
    print(f"  Aggregate $:  long ${total_long/1e6:.0f}M  ·  short ${total_short/1e6:.0f}M")
    # Top 3 net long + net short
    top_long = df.nlargest(3, "net_notional")
    top_short = df.nsmallest(3, "net_notional")
    print("  Top net long :", ", ".join(
        f"{r['coin']} +${r['net_notional']/1e6:.1f}M" for _, r in top_long.iterrows()
    ))
    print("  Top net short:", ", ".join(
        f"{r['coin']} -${abs(r['net_notional'])/1e6:.1f}M"
        for _, r in top_short.iterrows() if r["net_notional"] < 0
    ))
    return 0


def cmd_alert(args) -> int:
    """Re-dispatch the latest snapshot's alert (e.g. you missed it)."""
    if not INDEX_FILE.exists():
        print("No snapshots to alert on.")
        return 1
    index = json.loads(INDEX_FILE.read_text())
    if not index:
        print("No snapshots to alert on.")
        return 1
    latest_file = index[0]["file"]
    snap_path = SCREENS_DIR / latest_file
    if not snap_path.exists():
        print(f"Latest snapshot file missing: {snap_path}")
        return 1
    print(f"Re-sending alert for {latest_file}")
    snapshot = json.loads(snap_path.read_text())
    n_ok = _build_and_dispatch_alert(
        snapshot, latest_file, quiet_unchanged=args.quiet_unchanged,
    )
    return 0 if n_ok > 0 else 2


def cmd_list(args) -> int:
    if not INDEX_FILE.exists():
        print("No snapshots yet.")
        return 1
    index = json.loads(INDEX_FILE.read_text())
    if not index:
        print("No snapshots yet.")
        return 1
    print(f"{len(index)} snapshot(s):")
    for entry in index[: args.limit]:
        print(f"  {entry['file']}  preset={entry['preset']!r}  "
              f"n={entry['n_returned']}  picks={entry['tickers'][:5]}...")
    return 0


def cmd_show(args) -> int:
    """Show one snapshot. Accepts a date prefix or filename."""
    if not SCREENS_DIR.exists():
        print("No snapshots directory.")
        return 1
    target = args.snapshot
    if target.endswith(".json"):
        path = SCREENS_DIR / target
    else:
        # match prefix
        candidates = sorted(SCREENS_DIR.glob(f"{target}*.json"), reverse=True)
        if not candidates:
            print(f"No snapshot matches {target!r}")
            return 1
        path = candidates[0]
    if not path.exists():
        print(f"Not found: {path}")
        return 1
    print(path.read_text())
    return 0


def cmd_diff(args) -> int:
    """Show the buy/sell delta between two snapshots."""
    if not INDEX_FILE.exists():
        print("No snapshots to diff.")
        return 1
    index = json.loads(INDEX_FILE.read_text())
    if len(index) < 2:
        print("Need at least 2 snapshots to diff.")
        return 1
    new_entry = index[0]
    old_entry = index[1]
    old = set(old_entry["tickers"])
    new = set(new_entry["tickers"])
    added = sorted(new - old)
    dropped = sorted(old - new)
    held = sorted(new & old)
    print(f"Latest:   {new_entry['file']}  ({new_entry['preset']})")
    print(f"Previous: {old_entry['file']}  ({old_entry['preset']})")
    print()
    print(f"  + ADD ({len(added)}):     {', '.join(added) or '—'}")
    print(f"  − REMOVE ({len(dropped)}): {', '.join(dropped) or '—'}")
    print(f"  = HOLD ({len(held)}):    {', '.join(held)}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Quant Screen headless runner")
    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run the screen and save a snapshot (default)")
    p_run.add_argument("--preset", default="Long-term Value",
                       choices=list(qp.PRESETS.keys()),
                       help="Goal preset (default: 'Long-term Value')")
    p_run.add_argument("--top-n", type=int, default=20,
                       help="Number of picks to save (default: 20)")
    p_run.add_argument("--no-alert", action="store_true",
                       help="Skip the post-run alert dispatch.")
    p_run.add_argument("--quiet-unchanged", action="store_true",
                       help="Don't send an alert if the buy/sell list didn't change.")
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="List recent snapshots")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Print a single snapshot's JSON")
    p_show.add_argument("snapshot", help="Filename or date prefix")
    p_show.set_defaults(func=cmd_show)

    p_diff = sub.add_parser("diff", help="Buy/sell delta between latest two snapshots")
    p_diff.set_defaults(func=cmd_diff)

    p_test = sub.add_parser("test-alert",
                              help="Send a synthetic alert through every configured channel.")
    p_test.set_defaults(func=cmd_test_alert)

    p_bias = sub.add_parser("bias-snapshot",
                              help="Append a whale-bias-by-coin snapshot to "
                                   "~/.market-hub/whale_bias_history.csv.")
    p_bias.add_argument("--top-n", type=int, default=100)
    p_bias.add_argument("--sort-by", default="pnl",
                          choices=["account_value", "pnl", "roi", "vlm"])
    p_bias.add_argument("--window", default="allTime",
                          choices=["day", "week", "month", "allTime"])
    p_bias.set_defaults(func=cmd_bias_snapshot)

    p_alrt = sub.add_parser("alert",
                              help="Re-dispatch the latest snapshot's alert.")
    p_alrt.add_argument("--quiet-unchanged", action="store_true",
                         help="Skip if buy/sell didn't change.")
    p_alrt.set_defaults(func=cmd_alert)

    p_acct = sub.add_parser("account", help="Show Alpaca account + positions")
    p_acct.add_argument("--live", action="store_true",
                         help="Use LIVE endpoint (default: paper).")
    p_acct.set_defaults(func=cmd_account)

    p_reb = sub.add_parser("rebalance",
                              help="Equal-weight rebalance Alpaca to a snapshot's picks.")
    p_reb.add_argument("--snapshot", default="current",
                       help="'current' (default), date prefix, or filename.")
    p_reb.add_argument("--top-n", type=int, default=20,
                       help="How many of the snapshot's picks to hold equally weighted.")
    p_reb.add_argument("--cash-reserve", type=float, default=0.02,
                       help="Fraction of equity to keep in cash (default 0.02 = 2%%).")
    p_reb.add_argument("--threshold", type=float, default=0.10,
                       help="Skip rebalance if drift fraction is below this (default 0.10 = 10%%).")
    p_reb.add_argument("--min-order", type=float, default=1.0,
                       help="Skip orders smaller than this dollar amount.")
    p_reb.add_argument("--no-liquidate", action="store_true",
                       help="Do NOT sell positions outside the target list (default is to sell them).")
    p_reb.add_argument("--dry-run", action="store_true",
                       help="Preview the plan only — do not submit any orders.")
    p_reb.add_argument("--live", action="store_true",
                       help="Submit to the LIVE endpoint. Requires confirmation phrase.")
    p_reb.add_argument("--yes", action="store_true",
                       help="Skip the final ENTER confirmation in --live mode (use with care).")
    p_reb.add_argument("--no-alert", action="store_true",
                       help="Don't send a notification after the rebalance.")
    p_reb.add_argument("--max-sector-pct", type=float, default=None,
                       help="Max fraction of portfolio per GICS sector "
                            "(e.g. 0.30 = 30%%). Omit to disable.")
    p_reb.add_argument("--sector-cap", type=str, default="",
                       help='Per-sector overrides, comma-separated. Example: '
                            '"Information Technology=0.25,Financials=0.20"')
    p_reb.set_defaults(func=cmd_rebalance)

    # Default to `run` if no subcommand given
    args = p.parse_args()
    if not args.cmd:
        ns = p_run.parse_args([])
        return cmd_run(ns)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)

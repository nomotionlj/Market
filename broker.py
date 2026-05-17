"""Alpaca paper/live trading wrapper.

SAFETY-BY-DEFAULT:
- Defaults to PAPER endpoint (paper-api.alpaca.markets). Live requires
  explicit `live=True` AND the right base URL.
- All rebalance flows produce a `Plan` object first; nothing is sent
  to the API until the caller explicitly calls `execute()`.
- The `execute()` function fails closed on any individual order error
  and returns a per-order report.

Get free paper-trading keys at https://app.alpaca.markets (click "Paper").
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests


PAPER_BASE = "https://paper-api.alpaca.markets/v2"
LIVE_BASE = "https://api.alpaca.markets/v2"
DATA_BASE = "https://data.alpaca.markets/v2"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class AlpacaError(Exception):
    pass


@dataclass
class AlpacaClient:
    api_key: str
    api_secret: str
    live: bool = False
    timeout: float = 15.0

    @property
    def base_url(self) -> str:
        return LIVE_BASE if self.live else PAPER_BASE

    @property
    def mode(self) -> str:
        return "LIVE" if self.live else "PAPER"

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

    def _req(self, method: str, path: str, *, base: Optional[str] = None,
             params: Optional[dict] = None, json_body: Optional[dict] = None) -> dict:
        url = (base or self.base_url).rstrip("/") + path
        try:
            r = requests.request(method, url, headers=self._headers,
                                  params=params, json=json_body,
                                  timeout=self.timeout)
        except requests.RequestException as e:
            raise AlpacaError(f"Network error: {e}") from e
        if r.status_code == 401 or r.status_code == 403:
            raise AlpacaError(f"Auth failed ({r.status_code}). Check API keys "
                              f"and that you're using the right base URL "
                              f"(paper vs live).")
        if r.status_code >= 400:
            raise AlpacaError(f"HTTP {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except ValueError:
            return {}

    # --- Account / portfolio queries ---

    def account(self) -> Dict:
        return self._req("GET", "/account")

    def positions(self) -> List[Dict]:
        data = self._req("GET", "/positions")
        return data if isinstance(data, list) else []

    def clock(self) -> Dict:
        return self._req("GET", "/clock")

    def latest_price(self, symbol: str) -> Optional[float]:
        """Latest trade price via the data API. Falls back to None on failure."""
        try:
            data = self._req("GET", f"/stocks/{symbol}/trades/latest", base=DATA_BASE)
            return float(((data.get("trade") or {}).get("p")) or 0) or None
        except Exception:
            return None

    def latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        if not symbols:
            return {}
        try:
            data = self._req("GET", "/stocks/trades/latest", base=DATA_BASE,
                              params={"symbols": ",".join(symbols)})
            trades = data.get("trades") or {}
            return {s: float(t.get("p") or 0) for s, t in trades.items() if t}
        except Exception:
            # Fall back to per-symbol queries
            out: Dict[str, float] = {}
            for s in symbols:
                p = self.latest_price(s)
                if p:
                    out[s] = p
                time.sleep(0.05)
            return out

    # --- Orders ---

    def submit_order(self, symbol: str, qty: float, side: str,
                      type_: str = "market",
                      time_in_force: str = "day",
                      notional: Optional[float] = None) -> Dict:
        body: Dict[str, object] = {
            "symbol": symbol,
            "side": side,
            "type": type_,
            "time_in_force": time_in_force,
        }
        if notional is not None:
            body["notional"] = round(float(notional), 2)
        else:
            # Alpaca accepts fractional qty for fractionable equities
            body["qty"] = str(round(float(qty), 6))
        return self._req("POST", "/orders", json_body=body)

    def open_orders(self) -> List[Dict]:
        data = self._req("GET", "/orders", params={"status": "open"})
        return data if isinstance(data, list) else []

    def cancel_all_orders(self) -> List[Dict]:
        return self._req("DELETE", "/orders") or []  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Rebalance plan
# ---------------------------------------------------------------------------

@dataclass
class Action:
    symbol: str
    side: str                       # "buy" or "sell"
    notional: float                 # always positive USD
    current_value: float = 0.0
    target_value: float = 0.0
    current_qty: float = 0.0
    last_price: Optional[float] = None
    note: str = ""

    @property
    def signed_notional(self) -> float:
        return self.notional if self.side == "buy" else -self.notional


@dataclass
class Plan:
    mode: str                       # PAPER / LIVE
    target_universe: List[str]
    cash_reserve_pct: float
    equity: float
    cash: float
    buying_power: float
    actions: List[Action] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    snapshot_file: str = ""
    preset: str = ""

    @property
    def buys(self) -> List[Action]:
        return [a for a in self.actions if a.side == "buy"]

    @property
    def sells(self) -> List[Action]:
        return [a for a in self.actions if a.side == "sell"]

    @property
    def total_buy_notional(self) -> float:
        return sum(a.notional for a in self.buys)

    @property
    def total_sell_notional(self) -> float:
        return sum(a.notional for a in self.sells)

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "preset": self.preset,
            "snapshot_file": self.snapshot_file,
            "equity": self.equity,
            "cash": self.cash,
            "buying_power": self.buying_power,
            "cash_reserve_pct": self.cash_reserve_pct,
            "n_targets": len(self.target_universe),
            "n_buys": len(self.buys),
            "n_sells": len(self.sells),
            "total_buy_notional": round(self.total_buy_notional, 2),
            "total_sell_notional": round(self.total_sell_notional, 2),
            "actions": [a.__dict__ for a in self.actions],
            "skipped": self.skipped,
            "warnings": self.warnings,
        }

    def render_text(self) -> str:
        lines = [
            f"Rebalance plan ({self.mode})",
            f"  Equity: ${self.equity:,.2f}   Cash: ${self.cash:,.2f}   "
            f"Buying power: ${self.buying_power:,.2f}",
            f"  Targets: {len(self.target_universe)}   "
            f"Cash reserve: {self.cash_reserve_pct*100:.1f}%",
            "",
        ]
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
            lines.append("")
        if not self.actions:
            lines.append("  Already balanced — no orders to send.")
            return "\n".join(lines)
        lines.append(f"  Sells ({len(self.sells)}):")
        for a in self.sells:
            lines.append(f"    SELL {a.symbol:<6} ${a.notional:>10,.2f}   "
                         f"({a.note})" if a.note else
                         f"    SELL {a.symbol:<6} ${a.notional:>10,.2f}")
        lines.append("")
        lines.append(f"  Buys ({len(self.buys)}):")
        for a in self.buys:
            lines.append(f"    BUY  {a.symbol:<6} ${a.notional:>10,.2f}"
                         + (f"   ({a.note})" if a.note else ""))
        lines.append("")
        lines.append(f"  Net: SELL ${self.total_sell_notional:,.2f}   "
                     f"BUY ${self.total_buy_notional:,.2f}")
        if self.skipped:
            lines.append(f"  Skipped: {', '.join(self.skipped)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plan builder — equal-weight to a target list of tickers
# ---------------------------------------------------------------------------

def _apply_sector_caps(target_tickers: List[str],
                         ticker_sectors: Optional[Dict[str, str]],
                         sector_caps: Optional[Dict[str, float]],
                         default_sector_cap: Optional[float]) -> Tuple[Dict[str, float], List[str]]:
    """Compute per-ticker target weights honoring sector caps.

    Returns:
      (weights: ticker → fraction of investable [sums to ≤ 1.0],
       notes:   list of human-readable adjustment notes)

    Strategy:
      1. Start equal-weight (1/N each).
      2. For each sector that breaches its cap, scale its tickers down so the
         sector total = cap. Distribute the freed capital pro-rata to the
         remaining (uncapped, headroom-having) sectors.
      3. Iterate up to a few times — converges fast with ~10 sectors.
    """
    n = len(target_tickers)
    if n == 0:
        return {}, []
    if not sector_caps and not default_sector_cap:
        return {t: 1.0 / n for t in target_tickers}, []
    if not ticker_sectors:
        # Caps requested but no sector data — fall back to equal-weight + warn
        return ({t: 1.0 / n for t in target_tickers},
                ["Sector caps requested but no ticker→sector map provided — "
                 "falling back to equal-weight"])

    # Default per-ticker weight = 1/N
    weights = {t: 1.0 / n for t in target_tickers}

    def _sector_of(t: str) -> str:
        return ticker_sectors.get(t, "Unknown")

    def _cap_for(sector: str) -> float:
        if sector_caps and sector in sector_caps:
            return float(sector_caps[sector])
        return float(default_sector_cap) if default_sector_cap is not None else 1.0

    notes: List[str] = []

    for _ in range(8):  # converge within a handful of passes
        # Compute current per-sector totals
        sector_total: Dict[str, float] = {}
        for t, w in weights.items():
            sec = _sector_of(t)
            sector_total[sec] = sector_total.get(sec, 0.0) + w

        # Find any sector over its cap
        over = {}
        excess = 0.0
        for sec, tot in sector_total.items():
            cap = _cap_for(sec)
            if tot > cap + 1e-9:
                over[sec] = tot - cap
                excess += tot - cap

        if not over:
            break  # done

        # Scale each over-cap sector's tickers down to the cap
        for sec, _diff in over.items():
            cap = _cap_for(sec)
            tot = sector_total[sec]
            scale = cap / tot if tot > 0 else 0.0
            for t in target_tickers:
                if _sector_of(t) == sec:
                    weights[t] *= scale

        # Recompute headroom in non-capped sectors and redistribute pro-rata
        room: Dict[str, float] = {}
        room_total = 0.0
        for sec, tot in sector_total.items():
            if sec in over:
                continue
            cap = _cap_for(sec)
            r = max(cap - (sum(weights[t] for t in target_tickers if _sector_of(t) == sec)), 0.0)
            if r > 0:
                room[sec] = r
                room_total += r

        if room_total <= 0:
            # No headroom to absorb the excess — leave it uninvested (cash will rise)
            notes.append(f"Excess {excess*100:.1f}% from capped sectors had no "
                          f"headroom — that fraction will sit in cash.")
            break

        # Distribute proportionally among non-capped sectors' tickers (equal-weight inside sector)
        for sec, r in room.items():
            give = excess * (r / room_total)
            sec_tickers = [t for t in target_tickers if _sector_of(t) == sec]
            if not sec_tickers:
                continue
            per = give / len(sec_tickers)
            for t in sec_tickers:
                weights[t] += per

    # Build readable notes per capped sector
    sector_total_final: Dict[str, float] = {}
    for t, w in weights.items():
        sec = _sector_of(t)
        sector_total_final[sec] = sector_total_final.get(sec, 0.0) + w
    for sec, tot in sorted(sector_total_final.items(), key=lambda x: -x[1]):
        cap = _cap_for(sec)
        if cap < 1.0 - 1e-9:
            notes.append(f"{sec}: {tot*100:.1f}% (cap {cap*100:.0f}%)")

    return weights, notes


def build_equal_weight_plan(client: AlpacaClient,
                             target_tickers: List[str],
                             cash_reserve_pct: float = 0.02,
                             min_order_usd: float = 1.0,
                             rebalance_threshold_pct: float = 0.10,
                             liquidate_outside_targets: bool = True,
                             snapshot_file: str = "",
                             preset: str = "",
                             ticker_sectors: Optional[Dict[str, str]] = None,
                             sector_caps: Optional[Dict[str, float]] = None,
                             default_sector_cap: Optional[float] = None) -> Plan:
    """Build (but do not execute) an equal-weight rebalance plan.

    Args:
      target_tickers: tickers to hold equally-weighted
      cash_reserve_pct: 0.02 = keep 2% in cash (avoid buying-power overrun)
      min_order_usd: skip orders smaller than this
      rebalance_threshold_pct: don't trade if drift is below this fraction
                               of the target weight (reduces churn)
      liquidate_outside_targets: sell anything currently held that's not in targets
      ticker_sectors: optional ticker→sector map (required if using caps)
      sector_caps: optional dict like {"Information Technology": 0.30}
      default_sector_cap: optional cap applied to every sector that doesn't
                          have an explicit override (e.g. 0.30 = 30%)

    Returns a Plan that you can preview, then call `execute(client, plan)`.
    """
    target_tickers = [t.upper() for t in target_tickers if t]
    target_set = set(target_tickers)

    acct = client.account()
    equity = float(acct.get("equity") or 0)
    cash = float(acct.get("cash") or 0)
    bp = float(acct.get("buying_power") or 0)

    if equity <= 0:
        return Plan(mode=client.mode, target_universe=target_tickers,
                     cash_reserve_pct=cash_reserve_pct,
                     equity=equity, cash=cash, buying_power=bp,
                     warnings=["Account equity is zero — fund the account first."])

    investable = equity * (1 - cash_reserve_pct)
    if not target_tickers:
        return Plan(mode=client.mode, target_universe=[],
                     cash_reserve_pct=cash_reserve_pct,
                     equity=equity, cash=cash, buying_power=bp,
                     warnings=["No targets supplied."])

    # Normalize sector keys (trim whitespace) for both caps and the map
    if ticker_sectors:
        ticker_sectors = {t.upper(): (s or "Unknown").strip()
                           for t, s in ticker_sectors.items()}
    if sector_caps:
        sector_caps = {k.strip(): v for k, v in sector_caps.items()}

    weights_by_ticker, sector_notes = _apply_sector_caps(
        target_tickers=target_tickers,
        ticker_sectors=ticker_sectors,
        sector_caps=sector_caps,
        default_sector_cap=default_sector_cap,
    )
    target_value_by_ticker: Dict[str, float] = {
        t: investable * w for t, w in weights_by_ticker.items()
    }

    # Current positions keyed by symbol
    positions = client.positions()
    cur_by_sym: Dict[str, Dict] = {p["symbol"]: p for p in positions}

    # Latest prices for everything we might trade
    syms_to_quote = sorted(set(target_tickers) | set(cur_by_sym.keys()))
    prices = client.latest_prices(syms_to_quote)

    actions: List[Action] = []
    warnings: List[str] = []
    skipped: List[str] = []

    # --- Sell anything not in targets ---
    if liquidate_outside_targets:
        for sym, pos in cur_by_sym.items():
            if sym in target_set:
                continue
            mv = float(pos.get("market_value") or 0)
            qty = float(pos.get("qty") or 0)
            if abs(mv) < min_order_usd:
                continue
            actions.append(Action(
                symbol=sym, side="sell", notional=abs(mv),
                current_value=mv, target_value=0.0,
                current_qty=qty, last_price=prices.get(sym),
                note="not in target list",
            ))

    # --- Buy/sell within target list to hit target weight ---
    for sym in target_tickers:
        cur = cur_by_sym.get(sym)
        cur_value = float(cur.get("market_value") or 0) if cur else 0.0
        cur_qty = float(cur.get("qty") or 0) if cur else 0.0
        last_price = prices.get(sym)
        target_value = target_value_by_ticker.get(sym, 0.0)

        if not last_price:
            skipped.append(f"{sym} (no quote)")
            continue

        diff = target_value - cur_value
        # Threshold to avoid micro-rebalances
        if target_value > 0:
            drift_frac = abs(diff) / target_value
        else:
            drift_frac = 1.0 if abs(diff) > 0 else 0.0

        if drift_frac < rebalance_threshold_pct or abs(diff) < min_order_usd:
            continue

        if diff > 0:
            actions.append(Action(
                symbol=sym, side="buy", notional=diff,
                current_value=cur_value, target_value=target_value,
                current_qty=cur_qty, last_price=last_price,
                note=f"top up {drift_frac*100:.0f}% under",
            ))
        else:
            actions.append(Action(
                symbol=sym, side="sell", notional=-diff,
                current_value=cur_value, target_value=target_value,
                current_qty=cur_qty, last_price=last_price,
                note=f"trim {drift_frac*100:.0f}% over",
            ))

    # --- Sanity warnings ---
    total_buy = sum(a.notional for a in actions if a.side == "buy")
    total_sell = sum(a.notional for a in actions if a.side == "sell")
    available = cash + total_sell
    if total_buy > available + 1:  # small slack for float rounding
        warnings.append(
            f"Buys ${total_buy:,.2f} exceed available cash + sell proceeds "
            f"${available:,.2f}. Consider raising cash reserve or pre-running sells."
        )

    if client.mode == "LIVE":
        warnings.append("LIVE mode — orders will affect a real brokerage account.")

    # Surface the sector cap adjustment notes (informational, not error)
    for n in sector_notes:
        warnings.append(n)

    # Order: sells first so cash settles before buys
    actions.sort(key=lambda a: (0 if a.side == "sell" else 1, a.symbol))

    plan = Plan(
        mode=client.mode,
        target_universe=target_tickers,
        cash_reserve_pct=cash_reserve_pct,
        equity=equity, cash=cash, buying_power=bp,
        actions=actions, skipped=skipped, warnings=warnings,
        snapshot_file=snapshot_file, preset=preset,
    )
    # Stash the resolved per-ticker target-value map for UI display
    plan._target_values = target_value_by_ticker
    plan._ticker_sectors = ticker_sectors or {}
    plan._weights = weights_by_ticker
    return plan


def sector_breakdown(plan: Plan) -> List[Dict]:
    """Return a list of {sector, tickers, target_value, target_pct} sorted desc."""
    target_values = getattr(plan, "_target_values", {}) or {}
    ticker_sectors = getattr(plan, "_ticker_sectors", {}) or {}
    if not target_values:
        return []
    invest = sum(target_values.values()) or 1.0
    by_sector: Dict[str, Dict] = {}
    for t, v in target_values.items():
        sec = ticker_sectors.get(t, "Unknown") or "Unknown"
        bucket = by_sector.setdefault(sec, {"sector": sec, "tickers": [],
                                              "target_value": 0.0})
        bucket["tickers"].append(t)
        bucket["target_value"] += v
    out = []
    for b in by_sector.values():
        b["target_pct"] = b["target_value"] / invest
        b["tickers"].sort()
        out.append(b)
    out.sort(key=lambda x: -x["target_pct"])
    return out


# ---------------------------------------------------------------------------
# Execute the plan
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    plan: Plan
    submitted: List[Dict] = field(default_factory=list)   # alpaca order objs
    errors: List[Tuple[str, str]] = field(default_factory=list)  # (symbol, msg)

    @property
    def n_ok(self) -> int:
        return len(self.submitted)

    @property
    def n_err(self) -> int:
        return len(self.errors)


def execute(client: AlpacaClient, plan: Plan,
             pause_seconds: float = 0.15) -> ExecutionResult:
    """Submit each action as a notional market order. Sells first.

    This is the only function in this module that mutates real (or paper)
    account state. Always preview the Plan first.
    """
    result = ExecutionResult(plan=plan)

    for a in plan.actions:
        try:
            order = client.submit_order(
                symbol=a.symbol,
                qty=0,                              # ignored when notional is set
                side=a.side,
                type_="market",
                time_in_force="day",
                notional=round(a.notional, 2),
            )
            result.submitted.append(order)
        except AlpacaError as e:
            result.errors.append((a.symbol, str(e)))
        time.sleep(pause_seconds)

    return result


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(secrets: Dict) -> Dict:
    """Pull Alpaca-related fields from a flat secrets dict (st.secrets / load_secrets)."""
    return {
        "ALPACA_API_KEY": secrets.get("ALPACA_API_KEY", ""),
        "ALPACA_SECRET_KEY": secrets.get("ALPACA_SECRET_KEY", ""),
        "ALPACA_LIVE": str(secrets.get("ALPACA_LIVE", "")).lower() in ("1", "true", "yes"),
    }


def make_client(secrets: Dict, *, live_override: Optional[bool] = None) -> AlpacaClient:
    """Construct an AlpacaClient from a secrets dict.

    `live_override`:
      - None  → respect ALPACA_LIVE in secrets
      - False → force PAPER (safest)
      - True  → force LIVE (caller must have already gotten user confirmation)
    """
    cfg = load_config(secrets)
    if not cfg["ALPACA_API_KEY"] or not cfg["ALPACA_SECRET_KEY"]:
        raise AlpacaError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in secrets.")
    use_live = cfg["ALPACA_LIVE"] if live_override is None else live_override
    return AlpacaClient(
        api_key=cfg["ALPACA_API_KEY"],
        api_secret=cfg["ALPACA_SECRET_KEY"],
        live=bool(use_live),
    )

"""Quant factor computations for the value screen.

Pure functions — no I/O. Inputs are dicts of pre-fetched fundamentals.

Includes:
- Piotroski F-Score (0-9): nine binary fundamental signals
- Altman Z-Score: bankruptcy risk
- Beneish M-Score: earnings manipulation flag
- Magic Formula: Greenblatt EBIT/EV + ROIC
- Price momentum (12-1 month) and 52-week-high distance
- Composite quant score combining all of the above
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _safe_div(a, b) -> Optional[float]:
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return None
    try:
        return float(a) / float(b)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Piotroski F-Score (Piotroski 2000 — "value with quality" filter)
# ---------------------------------------------------------------------------

def piotroski_f_score(curr: Dict, prev: Dict) -> Dict:
    """Nine binary signals (each = 1 if the test passes, else 0).

    Required fields on curr/prev dicts (None if missing):
      net_income, total_assets, cfo, lt_debt, current_assets,
      current_liabilities, shares_out, gross_profit, total_revenue

    Returns: {"f_score": int 0-9, "signals": {name: 0|1}}
    Stocks with F ≥ 7 are typically considered fundamentally strong.
    """
    s = {}

    # 1. Positive net income
    s["pos_net_income"] = 1 if (curr.get("net_income") or 0) > 0 else 0

    # 2. Positive operating cash flow
    s["pos_cfo"] = 1 if (curr.get("cfo") or 0) > 0 else 0

    # 3. ROA increased YoY
    roa_c = _safe_div(curr.get("net_income"), curr.get("total_assets"))
    roa_p = _safe_div(prev.get("net_income"), prev.get("total_assets"))
    s["roa_up"] = 1 if (roa_c is not None and roa_p is not None and roa_c > roa_p) else 0

    # 4. CFO > Net Income (quality of earnings)
    s["cfo_gt_ni"] = 1 if (curr.get("cfo") or 0) > (curr.get("net_income") or 0) else 0

    # 5. Long-term debt decreased YoY
    lt_c, lt_p = curr.get("lt_debt"), prev.get("lt_debt")
    if lt_c is not None and lt_p is not None:
        s["debt_down"] = 1 if lt_c < lt_p else 0
    else:
        s["debt_down"] = 0

    # 6. Current ratio increased YoY
    cr_c = _safe_div(curr.get("current_assets"), curr.get("current_liabilities"))
    cr_p = _safe_div(prev.get("current_assets"), prev.get("current_liabilities"))
    s["cr_up"] = 1 if (cr_c is not None and cr_p is not None and cr_c > cr_p) else 0

    # 7. No new shares issued (≤ 1% increase tolerated)
    sh_c, sh_p = curr.get("shares_out"), prev.get("shares_out")
    if sh_c is not None and sh_p is not None and sh_p > 0:
        s["no_dilution"] = 1 if sh_c <= sh_p * 1.01 else 0
    else:
        s["no_dilution"] = 0

    # 8. Gross margin increased YoY
    gm_c = _safe_div(curr.get("gross_profit"), curr.get("total_revenue"))
    gm_p = _safe_div(prev.get("gross_profit"), prev.get("total_revenue"))
    s["gm_up"] = 1 if (gm_c is not None and gm_p is not None and gm_c > gm_p) else 0

    # 9. Asset turnover increased YoY
    at_c = _safe_div(curr.get("total_revenue"), curr.get("total_assets"))
    at_p = _safe_div(prev.get("total_revenue"), prev.get("total_assets"))
    s["at_up"] = 1 if (at_c is not None and at_p is not None and at_c > at_p) else 0

    return {"f_score": sum(s.values()), "signals": s}


# ---------------------------------------------------------------------------
# Altman Z-Score (bankruptcy risk)
# ---------------------------------------------------------------------------

def altman_z_score(d: Dict) -> Optional[float]:
    """Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*MVE/TL + 1.0*Sales/TA

    Z > 3.0  : Safe zone
    1.8–3.0 : Grey zone
    Z < 1.8  : Distress zone (high bankruptcy risk)

    d: working_capital, retained_earnings, ebit, market_cap,
       total_liabilities, total_revenue, total_assets
    """
    ta = d.get("total_assets")
    tl = d.get("total_liabilities")
    if not ta or not tl or ta <= 0 or tl <= 0:
        return None
    try:
        wc_ta = (d.get("working_capital") or 0) / ta
        re_ta = (d.get("retained_earnings") or 0) / ta
        ebit_ta = (d.get("ebit") or 0) / ta
        mve_tl = (d.get("market_cap") or 0) / tl
        sales_ta = (d.get("total_revenue") or 0) / ta
        return round(1.2 * wc_ta + 1.4 * re_ta + 3.3 * ebit_ta +
                     0.6 * mve_tl + 1.0 * sales_ta, 2)
    except Exception:
        return None


def altman_zone(z: Optional[float]) -> str:
    if z is None:
        return "—"
    if z >= 3.0:
        return "Safe"
    if z >= 1.8:
        return "Grey"
    return "Distress"


# ---------------------------------------------------------------------------
# Beneish M-Score (earnings manipulation indicator)
#
# M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
#     + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
#
# M > -2.22 → likely manipulation. Higher = worse.
# ---------------------------------------------------------------------------

def beneish_m_score(curr: Dict, prev: Dict) -> Optional[float]:
    """Requires curr/prev with: total_revenue, receivables, gross_profit,
    total_assets, current_assets, net_fixed_assets, depreciation, sga,
    total_liabilities, total_debt, net_income, cfo."""
    try:
        # DSRI: Days Sales in Receivables Index
        dsri = _safe_div(
            _safe_div(curr.get("receivables"), curr.get("total_revenue")),
            _safe_div(prev.get("receivables"), prev.get("total_revenue")),
        )

        # GMI: Gross Margin Index (decreasing margin = bad)
        gm_c = _safe_div(curr.get("gross_profit"), curr.get("total_revenue"))
        gm_p = _safe_div(prev.get("gross_profit"), prev.get("total_revenue"))
        gmi = _safe_div(gm_p, gm_c)

        # AQI: Asset Quality Index — non-current assets less PP&E to TA
        def _aqi(d):
            ta, ca, nfa = d.get("total_assets"), d.get("current_assets"), d.get("net_fixed_assets")
            if ta is None or ta == 0 or ca is None or nfa is None:
                return None
            return (ta - ca - nfa) / ta
        aqi_c, aqi_p = _aqi(curr), _aqi(prev)
        aqi = _safe_div(aqi_c, aqi_p)

        # SGI: Sales Growth Index
        sgi = _safe_div(curr.get("total_revenue"), prev.get("total_revenue"))

        # DEPI: Depreciation Index — drop in dep rate looks like manipulation
        def _dep_rate(d):
            dep, nfa = d.get("depreciation"), d.get("net_fixed_assets")
            if not dep or not nfa:
                return None
            return dep / (dep + nfa)
        depi = _safe_div(_dep_rate(prev), _dep_rate(curr))

        # SGAI: SG&A Index
        sgai = _safe_div(
            _safe_div(curr.get("sga"), curr.get("total_revenue")),
            _safe_div(prev.get("sga"), prev.get("total_revenue")),
        )

        # TATA: Total Accruals to Total Assets = (NI - CFO) / TA
        tata = _safe_div(
            (curr.get("net_income") or 0) - (curr.get("cfo") or 0),
            curr.get("total_assets"),
        )

        # LVGI: Leverage Index
        lvgi = _safe_div(
            _safe_div(curr.get("total_liabilities"), curr.get("total_assets")),
            _safe_div(prev.get("total_liabilities"), prev.get("total_assets")),
        )

        comps = [dsri, gmi, aqi, sgi, depi, sgai, tata, lvgi]
        if any(c is None for c in comps):
            return None

        m = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
             + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
        return round(m, 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sloan Accruals — earnings quality
# Accruals = (Net Income − CFO) / Total Assets
# Lower (more negative) is better; high positive accruals signal aggressive
# earnings recognition that often unwinds in future periods.
# ---------------------------------------------------------------------------

def sloan_accruals(d: Dict) -> Optional[float]:
    """Total accruals scaled by total assets. d: net_income, cfo, total_assets."""
    ni = d.get("net_income")
    cfo = d.get("cfo")
    ta = d.get("total_assets")
    if ni is None or cfo is None or not ta:
        return None
    try:
        return (float(ni) - float(cfo)) / float(ta)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------------------
# Asset Growth — Cooper, Gulen, Schill (2008)
# (Assets_t − Assets_{t-1}) / Assets_{t-1}
# Firms with very high asset growth tend to underperform (capex/issuance burn).
# ---------------------------------------------------------------------------

def asset_growth(curr: Dict, prev: Dict) -> Optional[float]:
    """Year-over-year change in total assets. Returns a fraction (0.20 = +20%)."""
    a_c = curr.get("total_assets")
    a_p = prev.get("total_assets")
    if not a_c or not a_p:
        return None
    try:
        return (float(a_c) - float(a_p)) / float(a_p)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------------------
# Greenblatt Magic Formula
# ---------------------------------------------------------------------------

def magic_formula(d: Dict) -> Dict:
    """Earnings yield = EBIT / EV.  ROIC = EBIT / (NWC + Net Fixed Assets).

    d: ebit, market_cap, total_debt, cash, nwc (or current_assets/current_liabilities),
       net_fixed_assets
    """
    ebit = d.get("ebit")
    mc = d.get("market_cap") or 0
    debt = d.get("total_debt") or 0
    cash = d.get("cash") or 0
    ev = mc + debt - cash if mc else None

    nwc = d.get("nwc")
    if nwc is None:
        ca, cl = d.get("current_assets"), d.get("current_liabilities")
        if ca is not None and cl is not None:
            nwc = ca - cl
    nfa = d.get("net_fixed_assets")

    out = {"earnings_yield": None, "roic": None, "ev": ev}
    if ebit is not None and ev and ev > 0:
        out["earnings_yield"] = ebit / ev
    if ebit is not None and nwc is not None and nfa is not None:
        denom = nwc + nfa
        if denom and denom > 0:
            out["roic"] = ebit / denom
    return out


# ---------------------------------------------------------------------------
# Price-based factors (computed from a Close series)
# ---------------------------------------------------------------------------

def price_momentum(close: pd.Series) -> Dict:
    """Returns 12-1 momentum, 52-week-high distance, 52-week-low distance, 1y vol."""
    if close is None or len(close) < 252:
        return {"mom_12_1": None, "dist_52wh": None, "dist_52wl": None, "vol_1y": None}
    p_curr = float(close.iloc[-1])
    p_t21 = float(close.iloc[-21])
    p_t252 = float(close.iloc[-252])
    high_252 = float(close.iloc[-252:].max())
    low_252 = float(close.iloc[-252:].min())
    rets = close.pct_change().dropna().iloc[-252:]
    vol_1y = float(rets.std() * np.sqrt(252)) if len(rets) > 20 else None

    return {
        "mom_12_1": (p_t21 / p_t252 - 1) * 100 if p_t252 > 0 else None,
        "dist_52wh": (p_curr / high_252 - 1) * 100 if high_252 > 0 else None,
        "dist_52wl": (p_curr / low_252 - 1) * 100 if low_252 > 0 else None,
        "vol_1y": vol_1y * 100 if vol_1y is not None else None,
    }


# ---------------------------------------------------------------------------
# Composite Quant Score (z-sum across available factors, normalized to [0,1])
# ---------------------------------------------------------------------------

def composite_quant_score(row: Dict, weights: Optional[Dict[str, float]] = None) -> float:
    """Combine factors into a single 0–1 score. Higher = better.

    Each factor is normalized to [0, 1] using a saturation function. Missing
    factors are skipped. Weights can be customized; default is equal-weight.
    """
    components = {}

    # Piotroski (0–9)
    f = row.get("f_score")
    if f is not None:
        components["f_score"] = f / 9.0

    # Altman Z (saturating at z=5)
    z = row.get("altman_z")
    if z is not None:
        components["altman_z"] = min(max(z / 5.0, 0), 1)

    # Beneish M-Score: lower is better. M < -2.22 is clean. Map to [0,1].
    m = row.get("beneish_m")
    if m is not None:
        # M=-3 → 1.0,  M=-2.22 → 0.65,  M=-1 → 0.2,  M=0 → 0
        components["beneish_m"] = max(0, min(1, (-m - 1) / 2))

    # Magic Formula earnings yield (cap at 25%)
    ey = row.get("earnings_yield")
    if ey is not None and ey > 0:
        components["earnings_yield"] = min(ey / 0.25, 1)

    # ROIC (cap at 50%)
    roic = row.get("roic")
    if roic is not None and roic > 0:
        components["roic"] = min(roic / 0.50, 1)

    # 12-1 momentum: -30% → 0, +50% → 1
    mom = row.get("mom_12_1")
    if mom is not None:
        components["mom_12_1"] = min(max((mom + 30) / 80, 0), 1)

    # 52-week-high distance: -50% → 0, 0% → 1
    dh = row.get("dist_52wh")
    if dh is not None:
        components["dist_52wh"] = min(max((dh + 50) / 50, 0), 1)

    # 1y volatility: lower is better. <20% → 1, >80% → 0
    v = row.get("vol_1y")
    if v is not None:
        components["low_vol"] = min(max((80 - v) / 60, 0), 1)

    # Insider net buying ($)
    ins = row.get("insider_net_usd")
    if ins is not None:
        if ins > 0:
            components["insider_buy"] = min(ins / 1e6, 1)  # $1M+ saturates
        else:
            components["insider_buy"] = 0

    # Sloan accruals — lower is better. -0.10 (very clean) → 1.0, +0.10 → 0
    acc = row.get("accruals")
    if acc is not None:
        components["accruals"] = min(max((-acc + 0.10) / 0.20, 0), 1)

    # Asset growth — lower is better. <0 → 1.0, >+50% → 0
    ag = row.get("asset_growth")
    if ag is not None:
        components["asset_growth"] = min(max(1 - ag / 0.50, 0), 1)

    # Earnings surprise (latest quarter, % surprise). 0 → 0.5, +20% → 1.0, -20% → 0
    sue = row.get("eps_surprise_pct")
    if sue is not None:
        components["eps_surprise"] = min(max(0.5 + sue / 40.0, 0), 1)

    # Earnings beat streak — count of consecutive positive surprises (max ~8)
    streak = row.get("eps_beat_streak")
    if streak is not None:
        components["eps_beat_streak"] = min(streak / 8.0, 1)

    if not components:
        return 0.0

    # Apply weights (or equal-weight)
    if weights:
        total_w = 0.0
        score = 0.0
        for k, v in components.items():
            w = weights.get(k, 0)
            score += v * w
            total_w += w
        return round(score / total_w, 3) if total_w > 0 else 0.0
    return round(sum(components.values()) / len(components), 3)

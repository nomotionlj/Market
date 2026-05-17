"""SEC 13F filing schedule helpers.

13F-HR rules:
  - Filed quarterly by institutional managers with >$100M AUM
  - Reports holdings as of the LAST DAY of each calendar quarter
  - Must be filed within 45 calendar days after quarter end

Quarter-end dates and filing deadlines:
  Q1: Mar 31  →  filing due May 15
  Q2: Jun 30  →  filing due Aug 14
  Q3: Sep 30  →  filing due Nov 14
  Q4: Dec 31  →  filing due Feb 14 (next year)

This module exposes:
  - next_quarter_end(today)
  - next_filing_deadline(today)
  - last_filing_deadline(today)
  - schedule_summary(today)   →  dict with all the dates + days-until
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, Optional


QUARTER_ENDS = [(3, 31), (6, 30), (9, 30), (12, 31)]


def _date(year: int, month: int, day: int) -> dt.date:
    return dt.date(year, month, day)


def _filing_deadline(quarter_end: dt.date) -> dt.date:
    return quarter_end + dt.timedelta(days=45)


def _quarter_label(d: dt.date) -> str:
    q = (d.month - 1) // 3 + 1
    return f"Q{q} {d.year}"


def all_quarter_ends_around(today: dt.date) -> Dict[str, dt.date]:
    """Return last_quarter_end and next_quarter_end relative to `today`."""
    candidates = []
    for y in (today.year - 1, today.year, today.year + 1):
        for m, d in QUARTER_ENDS:
            candidates.append(_date(y, m, d))
    last_q = max(d for d in candidates if d < today)
    next_q = min(d for d in candidates if d >= today)
    return {"last_quarter_end": last_q, "next_quarter_end": next_q}


def schedule_summary(today: Optional[dt.date] = None) -> Dict:
    """Comprehensive schedule view for UI display."""
    today = today or dt.date.today()
    qends = all_quarter_ends_around(today)
    last_qe = qends["last_quarter_end"]
    next_qe = qends["next_quarter_end"]

    last_deadline = _filing_deadline(last_qe)
    next_deadline = _filing_deadline(next_qe)

    # If today is AFTER the most recent deadline, that quarter's filings are
    # already out — the "current open window" is the next quarter.
    if today > last_deadline:
        current_window_qe = next_qe
        current_window_deadline = next_deadline
        # The most recently completed filing cycle
        latest_filed_quarter = last_qe
        latest_filed_deadline = last_deadline
    else:
        # We're still inside the window for the last quarter
        current_window_qe = last_qe
        current_window_deadline = last_deadline
        # The previously-completed filing cycle: 1 step further back
        prior_qe_candidates = [
            _date(y, m, d) for y in (today.year - 2, today.year - 1, today.year)
            for m, d in QUARTER_ENDS if _date(y, m, d) < last_qe
        ]
        latest_filed_quarter = max(prior_qe_candidates)
        latest_filed_deadline = _filing_deadline(latest_filed_quarter)

    return {
        "today": today,
        "current_window_quarter": _quarter_label(current_window_qe),
        "current_window_quarter_end": current_window_qe,
        "current_window_deadline": current_window_deadline,
        "days_to_current_deadline": (current_window_deadline - today).days,
        "latest_filed_quarter": _quarter_label(latest_filed_quarter),
        "latest_filed_quarter_end": latest_filed_quarter,
        "latest_filed_deadline": latest_filed_deadline,
        "days_since_latest_deadline": (today - latest_filed_deadline).days,
        # Calendar of the next four quarter deadlines
        "upcoming_deadlines": _build_upcoming(today, n=4),
    }


def _build_upcoming(today: dt.date, n: int = 4) -> list:
    """Next n quarter-end → filing-deadline pairs from today."""
    out = []
    for y in (today.year, today.year + 1, today.year + 2):
        for m, d in QUARTER_ENDS:
            qe = _date(y, m, d)
            dl = _filing_deadline(qe)
            if dl >= today:
                out.append({
                    "quarter": _quarter_label(qe),
                    "quarter_end": qe,
                    "filing_deadline": dl,
                    "days_until_deadline": (dl - today).days,
                })
            if len(out) >= n:
                return out
    return out

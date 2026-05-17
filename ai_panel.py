"""Multi-model AI panel: pluggable providers + consensus.

Supports five providers. The Value Screen UI runs whichever have a key set,
in parallel, and aggregates a consensus view.

  - Anthropic Claude  (PAID — best quality)
  - OpenAI GPT        (PAID)
  - Google Gemini     (FREE tier — 15 RPM, 1M tokens/day)
  - Groq              (FREE — Llama 3.3 70B / Mixtral, very fast)
  - OpenRouter        (FREE models via :free suffix, paid models too)

Each provider call returns a list of dicts:
  [{"ticker": "AAPL", "rank": 1, "verdict": "Buy",
    "thesis": "...", "key_risk": "..."}, ...]
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import pandas as pd
import requests


# Default model IDs (May 2026)
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-chat-v3.1:free"

VALID_VERDICTS = ["Strong Buy", "Buy", "Hold", "Avoid", "Value Trap"]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(candidates: List[Dict]) -> str:
    """Identical prompt sent to all three providers for a fair consensus."""
    lines = []
    for c in candidates:
        bits = [f"{c['ticker']} ({c.get('Security', '')})"]
        bits.append(f"sector={c.get('GICS Sector', 'n/a')}")
        if c.get("price") is not None:
            bits.append(f"px=${c['price']:.2f}")
        if c.get("drawdown_%") is not None:
            bits.append(f"dd={c['drawdown_%']:+.1f}%")
        if c.get("rsi") is not None:
            bits.append(f"rsi={c['rsi']:.1f}")
        for f, label in [("pe", "P/E"), ("pb", "P/B"), ("ev_ebitda", "EV/EBITDA")]:
            v = c.get(f)
            if v is not None and not pd.isna(v) and v > 0:
                bits.append(f"{label}={v:.1f}")
        if c.get("roe") is not None and not pd.isna(c["roe"]):
            bits.append(f"ROE={c['roe']*100:.1f}%")
        if c.get("debt_equity") is not None and not pd.isna(c["debt_equity"]):
            bits.append(f"D/E={c['debt_equity']/100:.2f}x")
        if c.get("rev_growth") is not None and not pd.isna(c["rev_growth"]):
            bits.append(f"rev_growth={c['rev_growth']*100:+.1f}%")
        if c.get("div_yield") is not None and not pd.isna(c["div_yield"]):
            bits.append(f"div_yld={c['div_yield']*100:.2f}%")
        lines.append("- " + " · ".join(bits))

    candidate_block = "\n".join(lines)

    return f"""You are a quantitative equity analyst. Rank the following S&P 500 \
stocks from a value-screen result by attractiveness as long-term value picks. \
Use the supplied technicals and fundamentals; do NOT invent data.

Candidates (most-recent metrics):
{candidate_block}

Output ONLY a JSON array, no commentary, no markdown fences. Each element:
{{
  "ticker": "AAPL",
  "rank": 1,
  "verdict": "Strong Buy" | "Buy" | "Hold" | "Avoid" | "Value Trap",
  "thesis": "<≤20 words on why this is/isn't value>",
  "key_risk": "<≤15 words on main risk>"
}}

rank=1 is your best pick. Verdicts must be one of the five listed strings exactly."""


# ---------------------------------------------------------------------------
# JSON extraction (LLMs sometimes wrap in code fences despite instructions)
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> List[Dict]:
    if not text:
        return []
    # Strip code fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    # Find first '[' to last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    chunk = text[start:end + 1]
    try:
        data = json.loads(chunk)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass
    return []


# ---------------------------------------------------------------------------
# Provider calls — all return list of dicts (or empty on error)
# ---------------------------------------------------------------------------

def call_claude(candidates: List[Dict], api_key: str,
                model: str = DEFAULT_CLAUDE_MODEL) -> List[Dict]:
    if not api_key:
        return []
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": _build_prompt(candidates)}],
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        body = r.json()
        text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        return _extract_json_array(text)
    except Exception as e:
        return [{"_error": f"Claude: {e}"}]


def _call_openai_compatible(candidates: List[Dict], api_key: str, base_url: str,
                              model: str, label: str,
                              json_object_mode: bool = True) -> List[Dict]:
    """Shared implementation for any OpenAI-compatible Chat Completions endpoint.
    Used for OpenAI, Groq, OpenRouter, DeepSeek, Together AI, etc."""
    if not api_key:
        return []
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if "openrouter.ai" in base_url:
        headers["HTTP-Referer"] = "https://github.com/market-hub"
        headers["X-Title"] = "Market Hub"

    prompt = _build_prompt(candidates)
    if json_object_mode:
        prompt += '\n\nReturn the array under a top-level key "rankings".'

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    if json_object_mode:
        body["response_format"] = {"type": "json_object"}

    try:
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        try:
            obj = json.loads(text)
            arr = obj.get("rankings") if isinstance(obj, dict) else obj
            if isinstance(arr, list):
                return [d for d in arr if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass
        return _extract_json_array(text)
    except Exception as e:
        return [{"_error": f"{label}: {e}"}]


def call_openai(candidates: List[Dict], api_key: str,
                model: str = DEFAULT_OPENAI_MODEL) -> List[Dict]:
    return _call_openai_compatible(candidates, api_key,
                                    "https://api.openai.com/v1",
                                    model, "OpenAI", json_object_mode=True)


def call_groq(candidates: List[Dict], api_key: str,
              model: str = DEFAULT_GROQ_MODEL) -> List[Dict]:
    """Groq — free tier, OpenAI-compatible, hosts Llama 3.3 70B and Mixtral."""
    return _call_openai_compatible(candidates, api_key,
                                    "https://api.groq.com/openai/v1",
                                    model, "Groq", json_object_mode=True)


def call_openrouter(candidates: List[Dict], api_key: str,
                     model: str = DEFAULT_OPENROUTER_MODEL) -> List[Dict]:
    """OpenRouter — free models via `:free` suffix (DeepSeek, Llama, etc.)."""
    # Many free OpenRouter models don't support response_format=json_object.
    return _call_openai_compatible(candidates, api_key,
                                    "https://openrouter.ai/api/v1",
                                    model, "OpenRouter", json_object_mode=False)


def call_gemini(candidates: List[Dict], api_key: str,
                model: str = DEFAULT_GEMINI_MODEL) -> List[Dict]:
    if not api_key:
        return []
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": _build_prompt(candidates)}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()
        cands = data.get("candidates", [])
        if not cands:
            return []
        parts = cands[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        # responseMimeType=application/json forces a JSON array directly
        try:
            arr = json.loads(text)
            if isinstance(arr, dict):
                # Some Gemini versions wrap in an object
                for v in arr.values():
                    if isinstance(v, list):
                        return v
                return []
            if isinstance(arr, list):
                return [d for d in arr if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass
        return _extract_json_array(text)
    except Exception as e:
        return [{"_error": f"Gemini: {e}"}]


# ---------------------------------------------------------------------------
# Multi-provider parallel call + consensus
# ---------------------------------------------------------------------------

def multi_model_panel(candidates: List[Dict],
                       claude_key: str = "",
                       openai_key: str = "",
                       gemini_key: str = "",
                       groq_key: str = "",
                       openrouter_key: str = "") -> Dict[str, List[Dict]]:
    """Call all configured providers in parallel. Skips any with no key.
    Returns dict of provider_name → list of ranking dicts (or [{'_error': ...}]).
    """
    jobs = {}
    if claude_key:
        jobs["Claude"] = (call_claude, candidates, claude_key)
    if openai_key:
        jobs["GPT"] = (call_openai, candidates, openai_key)
    if gemini_key:
        jobs["Gemini"] = (call_gemini, candidates, gemini_key)
    if groq_key:
        jobs["Groq"] = (call_groq, candidates, groq_key)
    if openrouter_key:
        jobs["OpenRouter"] = (call_openrouter, candidates, openrouter_key)
    if not jobs:
        return {}

    results = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futures = {ex.submit(fn, c, k): name for name, (fn, c, k) in jobs.items()}
        for fut in futures:
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as e:
                results[name] = [{"_error": str(e)}]
    return results


_VERDICT_SCORE = {"Strong Buy": 5, "Buy": 4, "Hold": 3, "Avoid": 2, "Value Trap": 1}


def consensus_table(per_model: Dict[str, List[Dict]],
                     candidates: List[Dict]) -> pd.DataFrame:
    """Aggregate the per-model outputs into a consensus DataFrame.

    Columns: ticker, security, [Claude/GPT/Gemini verdict + rank columns],
             agreement (1–3), avg_rank, avg_score, consensus_verdict
    """
    tickers = [c["ticker"] for c in candidates]
    sec_map = {c["ticker"]: c.get("Security", "") for c in candidates}

    rows = []
    for t in tickers:
        row = {"ticker": t, "Security": sec_map.get(t, "")}
        verdicts = []
        ranks = []
        scores = []
        for name, results in per_model.items():
            # find this ticker's entry in the model's response
            entry = next((r for r in results if isinstance(r, dict)
                          and r.get("ticker", "").upper() == t.upper()), None)
            if entry is None or "_error" in entry:
                row[f"{name} verdict"] = "—"
                row[f"{name} rank"] = None
                continue
            v = entry.get("verdict", "")
            row[f"{name} verdict"] = v if v in _VERDICT_SCORE else "—"
            row[f"{name} rank"] = entry.get("rank")
            if v in _VERDICT_SCORE:
                verdicts.append(v)
                scores.append(_VERDICT_SCORE[v])
            r = entry.get("rank")
            if isinstance(r, (int, float)):
                ranks.append(r)

        # Consensus stats
        row["agreement"] = len(verdicts)
        row["avg_rank"] = round(sum(ranks) / len(ranks), 1) if ranks else None
        row["avg_score"] = round(sum(scores) / len(scores), 2) if scores else None

        # Consensus verdict: take the most common verdict; tiebreak by avg_score
        if verdicts:
            from collections import Counter
            counts = Counter(verdicts)
            top = counts.most_common(1)[0]
            if top[1] >= 2:
                row["consensus"] = top[0]
            else:
                # all different — use the average-score-bucket
                avg = row["avg_score"] or 3
                if avg >= 4.5:
                    row["consensus"] = "Strong Buy"
                elif avg >= 3.5:
                    row["consensus"] = "Buy"
                elif avg >= 2.5:
                    row["consensus"] = "Hold"
                elif avg >= 1.5:
                    row["consensus"] = "Avoid"
                else:
                    row["consensus"] = "Value Trap"
        else:
            row["consensus"] = "—"

        # Pull a thesis from any model (priority: Claude → GPT → Gemini)
        for name in ("Claude", "GPT", "Gemini"):
            results = per_model.get(name, [])
            entry = next((r for r in results if isinstance(r, dict)
                          and r.get("ticker", "").upper() == t.upper()), None)
            if entry and entry.get("thesis"):
                row["thesis"] = entry["thesis"]
                row["key_risk"] = entry.get("key_risk", "")
                break
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Sort: highest agreement first, then best avg_score
    df = df.sort_values(by=["avg_score", "agreement"],
                        ascending=[False, False],
                        na_position="last").reset_index(drop=True)
    return df

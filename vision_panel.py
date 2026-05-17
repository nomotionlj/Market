"""Multi-model vision analysis for liquidation-map / chart confluence.

Sends 1–N images (with optional per-image label and source-type) plus a
structured prompt to Claude / GPT / Gemini in parallel. Parses each model's
JSON response. Aggregates a consensus view of price zones flagged by ≥2 models.

The prompt distinguishes EXACT (on-chain Hyperliquid) sources from MODELED
(CoinGlass / Coinalyze / Hyblock) so the AI weights them correctly.
"""
from __future__ import annotations

import base64
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Image payload
# ---------------------------------------------------------------------------

@dataclass
class ImageInput:
    name: str                  # e.g. "BTC 12h HL"
    source: str                # "HL_exact" | "modeled" | "other"
    data: bytes                # raw PNG/JPEG bytes
    media_type: str = "image/png"

    @property
    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = """You are a crypto derivatives analyst. The user is trying to \
identify where BTC may move over the next 0-3 days by analyzing liquidation \
heatmaps and related charts across multiple timeframes and data sources.

CRITICAL distinction between sources:
- **HL_exact** : Hyperliquid liquidation maps (Hyperdash, LiqFlow, etc.). \
HL positions are fully on-chain so these maps show the EXACT cumulative \
liquidation USD at exact price levels — not an estimate. Bright bands \
represent precisely-known forced-buy/sell levels. Weight these heavily.
- **modeled**  : CoinGlass / Coinalyze / Hyblock heatmaps. These infer \
clusters from public OI + leverage assumptions on exchanges that DO NOT \
publish positions. They are estimates, not exact. Use as corroboration.
- **other**    : price action, order book, funding, OI, anything else.

For EACH image, you are told its source type and label.

Output ONLY a single JSON object (no markdown fences) with this schema:
{
  "summary": "one or two sentences on what you see",
  "current_price_estimate": <number, your best read of current BTC price>,
  "per_image": [
    {
      "name": "...",
      "source": "HL_exact" | "modeled" | "other",
      "key_zones_above": [{"low": <number>, "high": <number>, "side": "short_liq" | "long_liq", "intensity": "high" | "med" | "low", "note": "..."}],
      "key_zones_below": [...]
    }
  ],
  "consensus_zones_above": [
    {
      "low": <number>, "high": <number>,
      "side": "short_liq" | "long_liq" | "mixed",
      "exact_count": <int>,    # how many HL_exact images flag this zone
      "modeled_count": <int>,  # how many modeled images flag this zone
      "agreement_count": <int>,# total images flagging it
      "note": "why this zone matters"
    }
  ],
  "consensus_zones_below": [...],
  "bias": "upside_sweep" | "downside_sweep" | "trap_both_sides" | "neutral",
  "bias_rationale": "...",
  "invalidation": "what price/event would invalidate this read"
}

Rules:
- Quote EXACT dollar values from HL_exact images when possible.
- Only call something a 'consensus_zone' if at least 2 images flag it.
- Prioritize HL_exact agreement over modeled-only agreement.
- Do not invent zones not visible in the supplied images."""


def build_user_text(images: List[ImageInput], extra_context: str = "") -> str:
    """Construct the per-call text that lists each image's name + source."""
    lines = ["Images supplied (in order):"]
    for i, im in enumerate(images, 1):
        lines.append(f"  {i}. {im.name}  [source={im.source}]")
    if extra_context.strip():
        lines.append("")
        lines.append("Additional context from live data:")
        lines.append(extra_context.strip())
    lines.append("")
    lines.append("Analyze them per the schema above. Return ONLY JSON.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> Optional[Dict]:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------

def call_claude_vision(images: List[ImageInput], prompt: str,
                        user_text: str, api_key: str,
                        model: str = DEFAULT_CLAUDE_MODEL,
                        timeout: float = 90.0) -> Tuple[Optional[Dict], str]:
    if not api_key:
        return None, "no key"
    content: List[Dict] = []
    for im in images:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": im.media_type, "data": im.b64},
        })
    content.append({"type": "text", "text": user_text})

    body = {
        "model": model,
        "max_tokens": 4000,
        "system": prompt,
        "messages": [{"role": "user", "content": content}],
    }
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body, timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        text = "".join(p.get("text", "") for p in data.get("content", [])
                       if p.get("type") == "text")
        return _extract_json_object(text), text
    except Exception as e:
        return None, f"error: {e}"


def call_openai_vision(images: List[ImageInput], prompt: str,
                        user_text: str, api_key: str,
                        model: str = DEFAULT_OPENAI_MODEL,
                        timeout: float = 90.0) -> Tuple[Optional[Dict], str]:
    if not api_key:
        return None, "no key"
    user_content: List[Dict] = []
    for im in images:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{im.media_type};base64,{im.b64}"},
        })
    user_content.append({"type": "text", "text": user_text})

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body, timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return _extract_json_object(text), text
    except Exception as e:
        return None, f"error: {e}"


def call_gemini_vision(images: List[ImageInput], prompt: str,
                        user_text: str, api_key: str,
                        model: str = DEFAULT_GEMINI_MODEL,
                        timeout: float = 90.0) -> Tuple[Optional[Dict], str]:
    if not api_key:
        return None, "no key"
    parts: List[Dict] = [{"text": prompt + "\n\n" + user_text}]
    for im in images:
        parts.append({
            "inline_data": {"mime_type": im.media_type, "data": im.b64},
        })

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": api_key},
            json=body, timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        cands = data.get("candidates", [])
        if not cands:
            return None, "no candidates"
        ps = cands[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in ps)
        return _extract_json_object(text), text
    except Exception as e:
        return None, f"error: {e}"


# ---------------------------------------------------------------------------
# Multi-model dispatch + consensus
# ---------------------------------------------------------------------------

def multi_model_vision(images: List[ImageInput],
                        prompt: str,
                        user_text: str,
                        claude_key: str = "",
                        openai_key: str = "",
                        gemini_key: str = "") -> Dict[str, Dict]:
    """Call every configured vision model in parallel.

    Returns: {provider: {"parsed": dict|None, "raw": str}}
    """
    jobs = {}
    if claude_key:
        jobs["Claude"] = (call_claude_vision, claude_key)
    if openai_key:
        jobs["GPT"] = (call_openai_vision, openai_key)
    if gemini_key:
        jobs["Gemini"] = (call_gemini_vision, gemini_key)
    if not jobs:
        return {}

    results: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        future_map = {
            ex.submit(fn, images, prompt, user_text, key): name
            for name, (fn, key) in jobs.items()
        }
        for fut in future_map:
            name = future_map[fut]
            try:
                parsed, raw = fut.result()
            except Exception as e:
                parsed, raw = None, f"exception: {e}"
            results[name] = {"parsed": parsed, "raw": raw}
    return results


def configured_models(secrets: Dict) -> List[str]:
    out = []
    if secrets.get("ANTHROPIC_API_KEY"):
        out.append("Claude")
    if secrets.get("OPENAI_API_KEY"):
        out.append("GPT")
    if secrets.get("GEMINI_API_KEY"):
        out.append("Gemini")
    return out


# ---------------------------------------------------------------------------
# Consensus aggregation across models
#
# Each model returns its own consensus_zones_above / consensus_zones_below.
# We merge overlapping zones across models (by price range) into a final
# "cross-model" consensus.
# ---------------------------------------------------------------------------

def _overlap_pct(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if hi <= lo:
        return 0.0
    span = max(a[1] - a[0], b[1] - b[0])
    return (hi - lo) / span if span > 0 else 0.0


def cross_model_zones(per_model: Dict[str, Dict],
                       side: str = "above",
                       overlap_threshold: float = 0.30) -> List[Dict]:
    """Merge zones from each model's output into a cross-model consensus.

    side = "above" or "below"
    """
    key = f"consensus_zones_{side}"
    bucket: List[Dict] = []
    for model_name, payload in per_model.items():
        parsed = payload.get("parsed")
        if not parsed:
            continue
        zones = parsed.get(key) or []
        for z in zones:
            try:
                lo, hi = float(z.get("low")), float(z.get("high"))
            except (TypeError, ValueError):
                continue
            if hi < lo:
                lo, hi = hi, lo
            new_z = {
                "low": lo, "high": hi,
                "side": z.get("side"),
                "models": {model_name},
                "exact_count": int(z.get("exact_count") or 0),
                "modeled_count": int(z.get("modeled_count") or 0),
                "notes": [f"{model_name}: {z.get('note','')}".strip(": ")],
            }
            placed = False
            for ex in bucket:
                if _overlap_pct((lo, hi), (ex["low"], ex["high"])) >= overlap_threshold:
                    # merge — widen range to union, accumulate counts
                    ex["low"] = min(ex["low"], lo)
                    ex["high"] = max(ex["high"], hi)
                    ex["models"].add(model_name)
                    ex["exact_count"] = max(ex["exact_count"], new_z["exact_count"])
                    ex["modeled_count"] = max(ex["modeled_count"], new_z["modeled_count"])
                    ex["notes"].extend(new_z["notes"])
                    placed = True
                    break
            if not placed:
                bucket.append(new_z)

    # Score and sort
    for b in bucket:
        b["models"] = sorted(b["models"])
        # Sort: exact data first, then most models agreeing, then widest
        b["score"] = (b["exact_count"] * 3 + len(b["models"]) * 2
                       + b["modeled_count"])
    bucket.sort(key=lambda x: -x["score"])
    return bucket


def consensus_bias(per_model: Dict[str, Dict]) -> Tuple[str, Dict[str, int]]:
    """Return (overall_bias, vote_counts) across the per-model biases."""
    from collections import Counter
    counts: Counter = Counter()
    for payload in per_model.values():
        parsed = payload.get("parsed")
        if not parsed:
            continue
        b = parsed.get("bias")
        if b:
            counts[b] += 1
    if not counts:
        return "—", {}
    top = counts.most_common(1)[0][0]
    return top, dict(counts)

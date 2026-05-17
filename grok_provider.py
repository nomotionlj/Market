"""xAI (Grok) provider — text + vision.

xAI exposes an OpenAI-compatible Chat Completions API at https://api.x.ai/v1.
Auth: `Authorization: Bearer <XAI_API_KEY>`.

This module mirrors the per-provider call signatures used by ai_panel.py and
vision_panel.py so it can be wired into multi_model_panel / multi_model_vision
with a few-line patch (see DEPLOY notes below).

Pricing: NOT free. X Premium+ (chat UI) does NOT include API access — you must
top up at https://console.x.ai with credits separately. ~$5 min.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


XAI_BASE = "https://api.x.ai/v1"
DEFAULT_GROK_TEXT_MODEL = "grok-4-fast-reasoning"   # cheap, fast, reasoning-tuned
DEFAULT_GROK_VISION_MODEL = "grok-4"                # has vision built in


# ---------------------------------------------------------------------------
# JSON extraction (shared with the other providers)
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> List[Dict]:
    if not text:
        return []
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass
    return []


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
# Text: call shape matches ai_panel's other provider calls.
# Used by multi_model_panel as the "Grok" branch.
# Signature: (candidates: List[Dict], api_key: str, model: str = ...) -> List[Dict]
# ---------------------------------------------------------------------------

def call_grok(candidates: List[Dict], api_key: str,
               model: str = DEFAULT_GROK_TEXT_MODEL) -> List[Dict]:
    if not api_key:
        return []

    # Import locally to avoid a hard dependency on ai_panel at import time
    try:
        from ai_panel import _build_prompt
    except Exception:
        # Fallback minimal prompt if ai_panel isn't importable
        def _build_prompt(c):  # type: ignore[no-redef]
            return ("Rank these S&P 500 candidates by long-term value. Return a "
                    "JSON array of {ticker, rank, verdict, thesis, key_risk}. "
                    "Verdict ∈ {Strong Buy, Buy, Hold, Avoid, Value Trap}.\n"
                    f"Candidates: {json.dumps(c)}")

    prompt = _build_prompt(candidates)
    prompt += '\n\nReturn the array under a top-level key "rankings".'

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    try:
        r = requests.post(
            f"{XAI_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=body, timeout=60,
        )
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
        return [{"_error": f"Grok: {e}"}]


# ---------------------------------------------------------------------------
# Vision: call shape matches vision_panel's other provider calls.
# Used by multi_model_vision as the "Grok" branch.
# Signature: (images, prompt, user_text, api_key, model, timeout) -> (parsed, raw)
# ---------------------------------------------------------------------------

def call_grok_vision(images, prompt: str, user_text: str, api_key: str,
                      model: str = DEFAULT_GROK_VISION_MODEL,
                      timeout: float = 90.0) -> Tuple[Optional[Dict], str]:
    if not api_key:
        return None, "no key"

    # `images` is a list of vision_panel.ImageInput.
    user_content: List[Dict] = []
    for im in images:
        # xAI accepts the same image_url data-URL format as OpenAI
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
            f"{XAI_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=body, timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return _extract_json_object(text), text
    except Exception as e:
        return None, f"error: {e}"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_key(api_key: str) -> Tuple[bool, str]:
    """Quick auth + model availability check. Returns (ok, info)."""
    if not api_key:
        return False, "no key supplied"
    try:
        r = requests.post(
            f"{XAI_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": DEFAULT_GROK_TEXT_MODEL,
                  "messages": [{"role": "user", "content": "Reply with just OK."}],
                  "max_tokens": 5},
            timeout=15,
        )
        if r.status_code == 200:
            try:
                txt = r.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                txt = ""
            return True, f"reply: {txt[:40]}"
        if r.status_code == 401 or r.status_code == 403:
            return False, "auth failed (key invalid or no credit on account)"
        if r.status_code == 429:
            return False, "rate-limited (key works, retry shortly)"
        return False, f"HTTP {r.status_code}: {r.text[:250]}"
    except Exception as e:
        return False, f"exception: {e}"

"""Streamlit UI for the Confluence Analyzer.

Upload N liquidation-map / chart screenshots, tag each with its source type
(HL_exact vs modeled vs other), and ship them to Claude / GPT / Gemini in
parallel. Display each model's reading and a cross-model consensus view of
the zones where ≥2 models agree.
"""
from __future__ import annotations

import io
from typing import Dict, List

import pandas as pd
import streamlit as st

import hyperliquid as hl
import vision_panel as vp


SOURCE_TYPES = ["HL_exact", "modeled", "other"]


def _secrets() -> Dict:
    keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]
    return {k: st.secrets.get(k, "") for k in keys if hasattr(st, "secrets")}


def _guess_source(name: str) -> str:
    n = (name or "").lower()
    hl_hits = ("hyperdash", "hyperliquid", "liqflow", "liqflow.app",
                "hl ", "hl_", "_hl", "icefless")
    modeled_hits = ("coinglass", "coinalyze", "hyblock", "tensorcharts", "bookmap")
    if any(h in n for h in hl_hits):
        return "HL_exact"
    if any(h in n for h in modeled_hits):
        return "modeled"
    return "other"


def _zone_table(zones: List[Dict], side: str) -> pd.DataFrame:
    rows = []
    for z in zones:
        try:
            lo, hi = float(z.get("low")), float(z.get("high"))
        except (TypeError, ValueError):
            continue
        rows.append({
            "Side": side.upper(),
            "Low":  f"${lo:,.0f}",
            "High": f"${hi:,.0f}",
            "Type": z.get("side", "—"),
            "Models": ", ".join(z.get("models", []) or []),
            "Exact": z.get("exact_count", 0),
            "Modeled": z.get("modeled_count", 0),
            "Score": z.get("score", 0),
            "Notes": "  •  ".join(z.get("notes", []) or [])[:200],
        })
    return pd.DataFrame(rows)


def render(coin_default: str = "BTC") -> None:
    secrets = _secrets()
    models = vp.configured_models(secrets)
    if not models:
        st.error(
            "No vision-capable AI keys configured. Add at least one of "
            "`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` to "
            "`.streamlit/secrets.toml`."
        )
        return

    st.markdown(
        f"Drop liquidation-map / chart screenshots below. Each model in your "
        f"panel ({', '.join(models)}) will read them in parallel and report "
        f"price zones above and below current price. The cross-model consensus "
        f"highlights zones where ≥ 2 models agree. **HL on-chain maps are weighted "
        f"more heavily than modeled (CoinGlass-style) maps.**"
    )

    # --- Coin selector + live HL context ---
    cc1, cc2, cc3 = st.columns([1, 1, 4])
    with cc1:
        coin = st.text_input("Coin", value=coin_default, key="conf_coin").upper().strip()
    with cc2:
        include_hl_context = st.checkbox(
            "Inject live HL context",
            value=True, key="conf_hl_ctx",
            help="Adds current HL mark / OI / funding for the coin to the prompt.",
        )
    with cc3:
        if include_hl_context and coin:
            with st.spinner("Fetching HL state..."):
                try:
                    hl_ctx_text = hl.coin_snapshot_text(coin)
                except Exception as e:
                    hl_ctx_text = f"(HL fetch failed: {e})"
            st.code(hl_ctx_text, language="text")
        else:
            hl_ctx_text = ""

    # --- File uploader ---
    st.markdown("##### Upload screenshots")
    uploads = st.file_uploader(
        "PNG or JPEG. Drop several at once.",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="conf_uploads",
    )

    images: List[vp.ImageInput] = []
    if uploads:
        st.markdown(f"##### Tag each image ({len(uploads)})")
        cols_per_row = 3
        for i in range(0, len(uploads), cols_per_row):
            row = uploads[i:i + cols_per_row]
            cols = st.columns(len(row))
            for col, f in zip(cols, row):
                with col:
                    st.image(f, caption=f.name, use_container_width=True)
                    default_src = _guess_source(f.name)
                    default_idx = SOURCE_TYPES.index(default_src) if default_src in SOURCE_TYPES else 2
                    src = st.selectbox(
                        f"Source — {f.name[:30]}",
                        SOURCE_TYPES, index=default_idx,
                        key=f"conf_src_{i}_{f.name}",
                        help=("HL_exact = Hyperdash, LiqFlow, etc. (real on-chain data). "
                              "modeled = CoinGlass-style estimate."),
                    )
                    label = st.text_input(
                        "Label (optional)",
                        value=f.name.rsplit(".", 1)[0][:40],
                        key=f"conf_lbl_{i}_{f.name}",
                    )
                    raw = f.getvalue()
                    media = "image/jpeg" if f.name.lower().endswith((".jpg", ".jpeg")) else "image/png"
                    images.append(vp.ImageInput(
                        name=label or f.name, source=src, data=raw, media_type=media,
                    ))

    # --- Prompt + run ---
    with st.expander("Prompt template (editable)", expanded=False):
        prompt = st.text_area("System prompt sent to every model",
                                value=vp.DEFAULT_PROMPT, height=320,
                                key="conf_prompt")

    rc1, rc2 = st.columns([1, 3])
    with rc1:
        run = st.button("🔬 Run confluence analysis",
                          type="primary",
                          disabled=not images,
                          key="conf_run")
    with rc2:
        if not images:
            st.caption("Upload at least one image to run.")
        else:
            st.caption(f"Will send {len(images)} image(s) to: **{', '.join(models)}**.")

    if not run:
        return

    user_text = vp.build_user_text(images, extra_context=hl_ctx_text)

    progress = st.empty()
    progress.info(f"Querying {', '.join(models)} in parallel...")
    per_model = vp.multi_model_vision(
        images=images,
        prompt=prompt,
        user_text=user_text,
        claude_key=secrets.get("ANTHROPIC_API_KEY", ""),
        openai_key=secrets.get("OPENAI_API_KEY", ""),
        gemini_key=secrets.get("GEMINI_API_KEY", ""),
    )
    progress.empty()

    if not per_model:
        st.error("No model responded.")
        return

    # --- Consensus first (the headline) ---
    overall, vote_counts = vp.consensus_bias(per_model)
    above = vp.cross_model_zones(per_model, side="above")
    below = vp.cross_model_zones(per_model, side="below")

    st.markdown("---")
    st.markdown("### 🎯 Cross-model consensus")

    cb1, cb2, cb3 = st.columns([1, 1, 2])
    with cb1:
        bias_color = {"upside_sweep": "#2ecc71",
                       "downside_sweep": "#e74c3c",
                       "trap_both_sides": "#f39c12",
                       "neutral": "#888"}.get(overall, "#888")
        st.markdown(
            f"<div style='padding:14px;background:#16191f;border-radius:10px;"
            f"border:1px solid #2a2e36;'>"
            f"<div style='color:#888;font-size:0.78em;text-transform:uppercase;"
            f"letter-spacing:0.05em;'>Overall bias</div>"
            f"<div style='font-size:1.5em;font-weight:700;color:{bias_color};'>"
            f"{overall.replace('_',' ').title()}</div>"
            f"<div style='color:#aaa;font-size:0.85em;margin-top:4px;'>"
            f"votes: {vote_counts}</div></div>",
            unsafe_allow_html=True,
        )
    with cb2:
        st.metric("Zones ABOVE", len(above))
        st.metric("Zones BELOW", len(below))
    with cb3:
        # pull invalidation from any model with one
        inv_lines = []
        for name, payload in per_model.items():
            p = payload.get("parsed") or {}
            if p.get("invalidation"):
                inv_lines.append(f"**{name}**: {p['invalidation']}")
        if inv_lines:
            st.markdown("**Invalidation conditions:**")
            for l in inv_lines:
                st.markdown(f"- {l}")

    # Zone tables
    if not above.empty if isinstance(above, pd.DataFrame) else above:
        st.markdown("##### ⬆ Zones above current price")
        st.dataframe(_zone_table(above, "above"), hide_index=True,
                      use_container_width=True)
    if below:
        st.markdown("##### ⬇ Zones below current price")
        st.dataframe(_zone_table(below, "below"), hide_index=True,
                      use_container_width=True)

    # --- Per-model details ---
    st.markdown("---")
    st.markdown("### 🤖 Per-model readings")
    tabs = st.tabs(list(per_model.keys()))
    for tab, (name, payload) in zip(tabs, per_model.items()):
        with tab:
            parsed = payload.get("parsed")
            raw = payload.get("raw", "")
            if not parsed:
                st.error("Model did not return valid JSON.")
                with st.expander("Raw response"):
                    st.code(raw[:5000])
                continue
            st.markdown(f"**Summary:** {parsed.get('summary','—')}")
            cur = parsed.get("current_price_estimate")
            bias = parsed.get("bias", "—")
            ra = parsed.get("bias_rationale", "")
            mcols = st.columns(3)
            mcols[0].metric("Current price (read)",
                             f"${float(cur):,.0f}" if cur else "—")
            mcols[1].metric("Bias", str(bias).replace("_", " "))
            mcols[2].metric("Invalidation", "see above" if parsed.get("invalidation") else "—")
            if ra:
                st.caption(f"_Rationale: {ra}_")

            # per-image zones
            for img in parsed.get("per_image", []) or []:
                with st.expander(f"📷 {img.get('name','image')}  "
                                  f"[{img.get('source','?')}]"):
                    above_z = img.get("key_zones_above") or []
                    below_z = img.get("key_zones_below") or []
                    if above_z:
                        st.markdown("**Above:**")
                        for z in above_z:
                            st.markdown(
                                f"- ${z.get('low'):,} – ${z.get('high'):,}   "
                                f"`{z.get('side','')}` · {z.get('intensity','')}"
                                + (f" — _{z['note']}_" if z.get("note") else "")
                            )
                    if below_z:
                        st.markdown("**Below:**")
                        for z in below_z:
                            st.markdown(
                                f"- ${z.get('low'):,} – ${z.get('high'):,}   "
                                f"`{z.get('side','')}` · {z.get('intensity','')}"
                                + (f" — _{z['note']}_" if z.get("note") else "")
                            )

            with st.expander("Raw JSON"):
                st.code(raw[:8000])

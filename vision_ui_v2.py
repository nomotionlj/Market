"""Confluence panel v2 — auto-fetched HL liquidation map + optional uploads.

Differences vs vision_ui.py:
- Builds a real HL liquidation map from on-chain positions (hl_liqmap module)
  for the selected coin and displays it immediately (no upload needed).
- Sends the auto-map AS AN IMAGE (PNG via kaleido) to the vision models so the
  same multi-model consensus flow works without screenshots.
- Uploads are still accepted as supplementary inputs (CoinGlass, custom charts,
  whatever) and are tagged the same way (HL_exact / modeled / other).
- Same per-model + cross-model consensus UI as v1.
"""
from __future__ import annotations

from io import BytesIO
from typing import Dict, List

import pandas as pd
import streamlit as st

import hl_liqmap as hlm
import hyperliquid as hl
import vision_panel as vp


SOURCE_TYPES = ["HL_exact", "modeled", "other"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _secrets() -> Dict:
    keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY"]
    return {k: st.secrets.get(k, "") for k in keys if hasattr(st, "secrets")}


def _guess_source(name: str) -> str:
    n = (name or "").lower()
    if any(h in n for h in ("hyperdash", "hyperliquid", "liqflow", "icefless",
                              "hl ", "hl_", "_hl", "auto_hl", "auto-hl")):
        return "HL_exact"
    if any(h in n for h in ("coinglass", "coinalyze", "hyblock", "tensorcharts")):
        return "modeled"
    return "other"


def _zone_table(zones: List[Dict]) -> pd.DataFrame:
    rows = []
    for z in zones:
        try:
            lo, hi = float(z.get("low")), float(z.get("high"))
        except (TypeError, ValueError):
            continue
        rows.append({
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


# ---------------------------------------------------------------------------
# Cached HL map build (5-min TTL — keeps cost down + UI snappy)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def _hl_snapshot_text(coin: str) -> str:
    """Cached HL live-context block (mark/OI/funding). Was hitting the API on every keystroke."""
    try:
        return hl.coin_snapshot_text(coin)
    except Exception as e:
        return f"(HL fetch failed: {e})"


@st.cache_data(ttl=300, show_spinner=False)
def _build_liqmap(coin: str, trades_per_coin: int):
    """Returns the raw payload from hl_liqmap.build_liqmap (figure NOT pickled-safe,
    so we return the dataframes + scalars and rebuild the figure outside the cache).
    """
    res = hlm.build_liqmap(coin=coin, trades_per_coin=trades_per_coin)
    return {
        "cum_df": res["cum_df"],
        "agg_df": res.get("agg_df"),
        "current_price": res["current_price"],
        "n_users": res["n_users"],
        "n_positions": res["n_positions"],
    }


@st.cache_data(ttl=300, show_spinner=False)
def _liqmap_png(coin: str, trades_per_coin: int) -> bytes:
    """Build the figure and render to PNG bytes for the vision call. Cached."""
    res = hlm.build_liqmap(coin=coin, trades_per_coin=trades_per_coin)
    return hlm.figure_to_png_bytes(res["figure"], width=1400, height=600, scale=2)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

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
        f"Live HL liquidation map auto-builds from real on-chain positions. "
        f"Drop additional screenshots (CoinGlass, etc.) below for cross-source "
        f"confluence. Models in panel: **{', '.join(models)}** — "
        f"HL on-chain maps are weighted more heavily than modeled (CoinGlass-style)."
    )

    # ── Coin + map controls ──
    cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 3])
    with cc1:
        coin = st.text_input("Coin", value=coin_default,
                              key="confv2_coin").upper().strip() or "BTC"
    with cc2:
        trades_per = st.select_slider(
            "Address sample",
            options=[100, 200, 300, 500],
            value=200, key="confv2_sample",
            help="Recent trades per discovery coin used to find active addresses. "
                 "Higher = wider sample, slower.",
        )
    with cc3:
        if st.button("🔄 Refresh map", key="confv2_refresh"):
            _build_liqmap.clear()
            _liqmap_png.clear()
            st.rerun()
    with cc4:
        ctx_text = _hl_snapshot_text(coin)
        st.code(ctx_text, language="text")

    # ── Auto liquidation map ──
    st.markdown("##### 🔴🟢 Live HL liquidation map (on-chain, no upload)")
    with st.spinner(f"Aggregating positions for {coin}..."):
        try:
            data = _build_liqmap(coin, trades_per)
        except Exception as e:
            st.error(f"HL map build failed: {e}")
            data = None

    auto_image: vp.ImageInput | None = None
    if data is not None:
        cumdf = data["cum_df"]
        aggdf = data.get("agg_df")
        current_price = data["current_price"]
        n_users = data["n_users"]
        n_positions = data["n_positions"]
        fig = hlm.render_liqmap_figure(cumdf, current_price=current_price, coin=coin,
                                          n_users_scanned=n_users,
                                          n_positions=n_positions,
                                          agg_df=aggdf)
        st.plotly_chart(fig, use_container_width=True)

        coverage_caption = (
            f"Coverage: {n_users} addresses scanned from recent trade flow + "
            f"seed whales · {n_positions} {coin} positions with valid liquidation "
            f"prices. Partial: misses traders who haven't traded recently and "
            f"aren't in the seed list. All shown data is real on-chain."
        )
        st.caption(coverage_caption)

        # Render the same chart to PNG bytes for the vision call
        try:
            png = _liqmap_png(coin, trades_per)
            auto_image = vp.ImageInput(
                name=f"AUTO {coin} HL liquidation map ({n_users} addr, {n_positions} pos)",
                source="HL_exact",
                data=png,
                media_type="image/png",
            )
        except Exception as e:
            st.warning(f"Could not render map to PNG for AI panel: {e}")

    # ── Optional uploads ──
    with st.expander("➕ Add supplementary screenshots (optional)", expanded=False):
        uploads = st.file_uploader(
            "PNG or JPEG. Drop CoinGlass / Hyperdash / custom charts to cross-check.",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="confv2_uploads",
        )
        upload_images: List[vp.ImageInput] = []
        if uploads:
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
                            key=f"confv2_src_{i}_{f.name}",
                        )
                        label = st.text_input(
                            "Label",
                            value=f.name.rsplit(".", 1)[0][:40],
                            key=f"confv2_lbl_{i}_{f.name}",
                        )
                        media = ("image/jpeg" if f.name.lower().endswith((".jpg", ".jpeg"))
                                  else "image/png")
                        upload_images.append(vp.ImageInput(
                            name=label or f.name, source=src,
                            data=f.getvalue(), media_type=media,
                        ))

    # ── Combined image list ──
    images: List[vp.ImageInput] = []
    if auto_image is not None:
        images.append(auto_image)
    images.extend(upload_images if 'upload_images' in locals() else [])

    # ── Prompt + run ──
    with st.expander("Prompt template (editable)", expanded=False):
        prompt = st.text_area("System prompt sent to every model",
                                value=vp.DEFAULT_PROMPT, height=320,
                                key="confv2_prompt")

    rc1, rc2 = st.columns([1, 3])
    with rc1:
        run = st.button("🔬 Run confluence analysis",
                          type="primary",
                          disabled=not images,
                          key="confv2_run")
    with rc2:
        st.caption(
            f"Will send **{len(images)}** image(s) to {', '.join(models)}: "
            f"{'1 auto HL map' if auto_image else 'no auto map'}"
            + (f" + {len(images)-1} upload(s)" if (len(images)-1) > 0 else "")
        )

    if not run:
        return

    user_text = vp.build_user_text(images, extra_context=ctx_text)

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
        inv_lines = []
        for name, payload in per_model.items():
            p = payload.get("parsed") or {}
            if p.get("invalidation"):
                inv_lines.append(f"**{name}**: {p['invalidation']}")
        if inv_lines:
            st.markdown("**Invalidation conditions:**")
            for l in inv_lines:
                st.markdown(f"- {l}")

    if above:
        st.markdown("##### ⬆ Zones above current price")
        st.dataframe(_zone_table(above), hide_index=True, use_container_width=True)
    if below:
        st.markdown("##### ⬇ Zones below current price")
        st.dataframe(_zone_table(below), hide_index=True, use_container_width=True)

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

            for img in parsed.get("per_image", []) or []:
                with st.expander(f"📷 {img.get('name','image')}  "
                                  f"[{img.get('source','?')}]"):
                    for side_key, label in [("key_zones_above", "Above"),
                                              ("key_zones_below", "Below")]:
                        zs = img.get(side_key) or []
                        if zs:
                            st.markdown(f"**{label}:**")
                            for z in zs:
                                st.markdown(
                                    f"- ${z.get('low'):,} – ${z.get('high'):,}   "
                                    f"`{z.get('side','')}` · {z.get('intensity','')}"
                                    + (f" — _{z['note']}_" if z.get("note") else "")
                                )
            with st.expander("Raw JSON"):
                st.code(raw[:8000])

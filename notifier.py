"""Multi-channel alert dispatcher for the quant screener.

Channels (each optional — configured in .streamlit/secrets.toml):

  DISCORD_WEBHOOK_URL  — free; create one in any Discord server's Channel Settings
  NTFY_TOPIC           — free; pick any unique string and subscribe via ntfy.sh app
  SLACK_WEBHOOK_URL    — free; create at api.slack.com/apps
  SMTP_HOST/PORT/USER/PASS/TO — free with Gmail app password

Each `send_*` function returns (success: bool, info: str). The high-level
`dispatch_alert` calls every configured channel and returns a per-channel report.
"""
from __future__ import annotations

import json
import smtplib
import ssl
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# Alert payload
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """Structured alert. Renderers transform it for each channel."""
    title: str                          # short headline
    preset: str                         # e.g. "Long-term Value"
    timestamp: str                      # ISO timestamp of the snapshot
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    held: List[str] = field(default_factory=list)
    top_picks: List[Dict] = field(default_factory=list)  # [{ticker, score, security}]
    snapshot_file: str = ""

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed)

    @property
    def color_hex(self) -> int:
        """Discord embed color: green if pure adds, red if pure removes, blue if mixed/none."""
        if self.added and not self.removed:
            return 0x2ecc71
        if self.removed and not self.added:
            return 0xe74c3c
        if self.added or self.removed:
            return 0xf39c12  # orange — mixed
        return 0x3498db      # blue — no changes

    # -- text renderers --

    def to_plain_text(self, max_top: int = 10) -> str:
        lines = [
            f"🤖 {self.title}",
            f"Preset: {self.preset}",
            f"Time:   {self.timestamp}",
            "",
        ]
        if self.has_changes:
            lines.append(f"➕ ADD ({len(self.added)}): "
                          f"{', '.join(self.added) if self.added else '—'}")
            lines.append(f"➖ REMOVE ({len(self.removed)}): "
                          f"{', '.join(self.removed) if self.removed else '—'}")
        else:
            lines.append("No changes since the previous snapshot.")
        lines.append(f"= HOLD ({len(self.held)})")
        if self.top_picks:
            lines.append("")
            lines.append(f"Top {min(max_top, len(self.top_picks))} picks:")
            for p in self.top_picks[:max_top]:
                score = p.get("score")
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
                sec = (p.get("security") or "")[:32]
                lines.append(f"  {p['ticker']:<6} {score_str}  {sec}")
        return "\n".join(lines)

    def to_markdown(self, max_top: int = 10) -> str:
        lines = [
            f"## 🤖 {self.title}",
            f"**Preset:** {self.preset}  ·  **Time:** {self.timestamp}",
            "",
        ]
        if self.has_changes:
            if self.added:
                lines.append(f"**➕ ADD ({len(self.added)}):** "
                              + ", ".join(f"`{t}`" for t in self.added))
            if self.removed:
                lines.append(f"**➖ REMOVE ({len(self.removed)}):** "
                              + ", ".join(f"`{t}`" for t in self.removed))
        else:
            lines.append("_No changes since the previous snapshot._")
        lines.append(f"**= HOLD:** {len(self.held)}")
        if self.top_picks:
            lines.append("")
            lines.append(f"**Top {min(max_top, len(self.top_picks))} picks:**")
            for p in self.top_picks[:max_top]:
                score = p.get("score")
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
                sec = p.get("security") or ""
                lines.append(f"- `{p['ticker']}` **{score_str}** — {sec}")
        return "\n".join(lines)

    def to_html(self, max_top: int = 10) -> str:
        rows = []
        for p in self.top_picks[:max_top]:
            score = p.get("score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
            sec = p.get("security") or ""
            rows.append(
                f'<tr><td style="padding:4px 10px;font-family:monospace;">'
                f'<b>{p["ticker"]}</b></td>'
                f'<td style="padding:4px 10px;">{score_str}</td>'
                f'<td style="padding:4px 10px;color:#555;">{sec}</td></tr>'
            )
        added_html = (", ".join(f"<code>{t}</code>" for t in self.added)
                      if self.added else "—")
        removed_html = (", ".join(f"<code>{t}</code>" for t in self.removed)
                        if self.removed else "—")
        change_section = (
            f'<p>➕ <b>ADD ({len(self.added)}):</b> {added_html}</p>'
            f'<p>➖ <b>REMOVE ({len(self.removed)}):</b> {removed_html}</p>'
        ) if self.has_changes else (
            '<p style="color:#888;"><i>No changes since the previous snapshot.</i></p>'
        )
        return f"""
<div style="font-family:-apple-system,Helvetica,Arial,sans-serif;color:#222;">
  <h2 style="margin-bottom:0;">🤖 {self.title}</h2>
  <p style="color:#888;margin-top:4px;">
    <b>Preset:</b> {self.preset} &nbsp;·&nbsp; <b>Time:</b> {self.timestamp}
  </p>
  {change_section}
  <p>= <b>HOLD:</b> {len(self.held)}</p>
  <h3>Top picks</h3>
  <table style="border-collapse:collapse;">
    <thead><tr style="border-bottom:1px solid #ddd;">
      <th style="text-align:left;padding:4px 10px;">Symbol</th>
      <th style="text-align:left;padding:4px 10px;">Score</th>
      <th style="text-align:left;padding:4px 10px;">Security</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>
"""


# ---------------------------------------------------------------------------
# Channel implementations
# ---------------------------------------------------------------------------

def send_discord(alert: Alert, webhook_url: str) -> Tuple[bool, str]:
    if not webhook_url:
        return False, "no webhook configured"
    fields = []
    if alert.added:
        fields.append({
            "name": f"➕ ADD ({len(alert.added)})",
            "value": ", ".join(f"`{t}`" for t in alert.added)[:1024],
            "inline": False,
        })
    if alert.removed:
        fields.append({
            "name": f"➖ REMOVE ({len(alert.removed)})",
            "value": ", ".join(f"`{t}`" for t in alert.removed)[:1024],
            "inline": False,
        })
    if alert.top_picks:
        top_lines = []
        for p in alert.top_picks[:10]:
            score = p.get("score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
            sec = (p.get("security") or "")[:30]
            top_lines.append(f"`{p['ticker']:<5}` **{score_str}**  {sec}")
        fields.append({
            "name": f"Top {len(top_lines)} picks",
            "value": "\n".join(top_lines)[:1024],
            "inline": False,
        })
    fields.append({"name": "Hold", "value": f"{len(alert.held)} positions",
                   "inline": True})
    fields.append({"name": "Snapshot", "value": alert.snapshot_file or "—",
                   "inline": True})

    payload = {
        "username": "Quant Screen",
        "embeds": [{
            "title": alert.title,
            "description": (f"**Preset:** {alert.preset}\n"
                             f"**Time:** {alert.timestamp}"),
            "color": alert.color_hex,
            "fields": fields,
        }],
    }
    try:
        r = requests.post(webhook_url, json=payload, timeout=15)
        if r.status_code in (200, 204):
            return True, f"OK ({r.status_code})"
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"exception: {e}"


def send_ntfy(alert: Alert, topic: str,
               server: str = "https://ntfy.sh") -> Tuple[bool, str]:
    """Send via ntfy.sh — free push to your phone (install ntfy app, subscribe to topic)."""
    if not topic:
        return False, "no topic configured"
    url = f"{server.rstrip('/')}/{topic}"
    body = alert.to_plain_text(max_top=8)
    headers = {
        "Title": alert.title,
        "Priority": "default",
        "Tags": "chart_with_upwards_trend",
    }
    if alert.has_changes:
        headers["Priority"] = "high"
        headers["Tags"] = "rotating_light,chart_with_upwards_trend"
    try:
        r = requests.post(url, data=body.encode("utf-8"),
                           headers=headers, timeout=15)
        if r.status_code == 200:
            return True, "OK"
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"exception: {e}"


def send_slack(alert: Alert, webhook_url: str) -> Tuple[bool, str]:
    if not webhook_url:
        return False, "no webhook configured"
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": alert.title}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Preset:* {alert.preset}"},
            {"type": "mrkdwn", "text": f"*Time:* {alert.timestamp}"},
        ]},
    ]
    if alert.added:
        blocks.append({"type": "section", "text": {
            "type": "mrkdwn",
            "text": f"*➕ ADD ({len(alert.added)}):* "
                    + ", ".join(f"`{t}`" for t in alert.added),
        }})
    if alert.removed:
        blocks.append({"type": "section", "text": {
            "type": "mrkdwn",
            "text": f"*➖ REMOVE ({len(alert.removed)}):* "
                    + ", ".join(f"`{t}`" for t in alert.removed),
        }})
    if alert.top_picks:
        lines = []
        for p in alert.top_picks[:10]:
            score = p.get("score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
            lines.append(f"`{p['ticker']:<5}` *{score_str}*  {p.get('security','')[:30]}")
        blocks.append({"type": "section", "text": {
            "type": "mrkdwn", "text": "*Top picks:*\n" + "\n".join(lines),
        }})
    payload = {"text": alert.title, "blocks": blocks}
    try:
        r = requests.post(webhook_url, json=payload, timeout=15)
        if r.status_code == 200:
            return True, "OK"
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"exception: {e}"


def send_email(alert: Alert, smtp_host: str, smtp_port: int, smtp_user: str,
                smtp_pass: str, smtp_to: str,
                smtp_from: Optional[str] = None) -> Tuple[bool, str]:
    if not (smtp_host and smtp_port and smtp_user and smtp_pass and smtp_to):
        return False, "missing SMTP config"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Quant] {alert.title}"
    msg["From"] = smtp_from or smtp_user
    msg["To"] = smtp_to
    msg.attach(MIMEText(alert.to_plain_text(), "plain"))
    msg.attach(MIMEText(alert.to_html(), "html"))

    try:
        if smtp_port == 465:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=ctx, timeout=30) as s:
                s.login(smtp_user, smtp_pass)
                s.sendmail(msg["From"], [smtp_to], msg.as_string())
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(smtp_user, smtp_pass)
                s.sendmail(msg["From"], [smtp_to], msg.as_string())
        return True, f"sent to {smtp_to}"
    except Exception as e:
        return False, f"exception: {e}"


# ---------------------------------------------------------------------------
# High-level dispatch
# ---------------------------------------------------------------------------

def dispatch_alert(alert: Alert, config: Dict,
                    suppress_unchanged: bool = False) -> Dict[str, Tuple[bool, str]]:
    """Send the alert through every channel for which credentials are present.

    `config` should be a dict (e.g. st.secrets) with these optional keys:
        DISCORD_WEBHOOK_URL
        NTFY_TOPIC, NTFY_SERVER (optional override)
        SLACK_WEBHOOK_URL
        SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_TO, SMTP_FROM

    Returns: {channel_name: (success, info)} for every attempted channel.
    """
    if suppress_unchanged and not alert.has_changes:
        return {"_suppressed": (True, "no changes — alert suppressed")}

    results: Dict[str, Tuple[bool, str]] = {}

    if config.get("DISCORD_WEBHOOK_URL"):
        results["discord"] = send_discord(alert, config["DISCORD_WEBHOOK_URL"])
    if config.get("NTFY_TOPIC"):
        results["ntfy"] = send_ntfy(
            alert, config["NTFY_TOPIC"],
            server=config.get("NTFY_SERVER", "https://ntfy.sh"),
        )
    if config.get("SLACK_WEBHOOK_URL"):
        results["slack"] = send_slack(alert, config["SLACK_WEBHOOK_URL"])
    if all(config.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS", "SMTP_TO")):
        try:
            port = int(config.get("SMTP_PORT", 587))
        except (TypeError, ValueError):
            port = 587
        results["email"] = send_email(
            alert,
            smtp_host=config["SMTP_HOST"], smtp_port=port,
            smtp_user=config["SMTP_USER"], smtp_pass=config["SMTP_PASS"],
            smtp_to=config["SMTP_TO"],
            smtp_from=config.get("SMTP_FROM"),
        )

    if not results:
        results["_none"] = (False, "no channels configured")
    return results


def configured_channels(config: Dict) -> List[str]:
    """Return list of human-readable channel names that have credentials."""
    out = []
    if config.get("DISCORD_WEBHOOK_URL"):
        out.append("Discord")
    if config.get("NTFY_TOPIC"):
        out.append(f"ntfy ({config['NTFY_TOPIC']})")
    if config.get("SLACK_WEBHOOK_URL"):
        out.append("Slack")
    if all(config.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS", "SMTP_TO")):
        out.append(f"Email → {config['SMTP_TO']}")
    return out


# ---------------------------------------------------------------------------
# Helpers for picker.py
# ---------------------------------------------------------------------------

def alert_from_snapshots(latest: Dict, previous: Optional[Dict]) -> Alert:
    """Construct an Alert from two snapshot dicts (as written by picker.write_snapshot)."""
    new_picks = latest.get("picks", []) or []
    old_picks = (previous or {}).get("picks", []) or []
    new_t = [p["ticker"] for p in new_picks]
    old_t = [p["ticker"] for p in old_picks]

    added = [t for t in new_t if t not in old_t]
    removed = [t for t in old_t if t not in new_t]
    held = [t for t in new_t if t in old_t]

    top_picks = [
        {"ticker": p["ticker"],
         "score": p.get("quant_score"),
         "security": p.get("security") or p.get("Security", "")}
        for p in new_picks
    ]

    title_bits = []
    if added:
        title_bits.append(f"+{len(added)}")
    if removed:
        title_bits.append(f"-{len(removed)}")
    title_suffix = f" · {' / '.join(title_bits)}" if title_bits else " · no changes"
    title = f"Quant picks{title_suffix}"

    return Alert(
        title=title,
        preset=latest.get("preset", "Custom"),
        timestamp=latest.get("timestamp", "")[:19].replace("T", " "),
        added=added, removed=removed, held=held,
        top_picks=top_picks,
        snapshot_file=latest.get("_filename", ""),
    )


def load_secrets() -> Dict:
    """Read .streamlit/secrets.toml so the CLI can use the same config as the UI."""
    from pathlib import Path
    secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return {}
    config: Dict = {}
    try:
        for raw in secrets_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if v:
                config[k] = v
    except Exception:
        pass
    return config

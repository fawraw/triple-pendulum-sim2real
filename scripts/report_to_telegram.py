"""Read training/probe results from /workspace/ and POST a summary to Telegram.

Used on cloud pods that can't reach the LAN-only n8n webhook. Designed to
run as a one-shot from a RunPod pod whose dockerStartCmd is:

  bash -c '
    until getent hosts github.com >/dev/null 2>&1; do sleep 2; done
    git clone https://github.com/fawraw/triple-pendulum-sim2real.git /tmp/repo
    python3 /tmp/repo/scripts/report_to_telegram.py
    # then podStop
  '

Requires env vars (set in pod env):
  TELEGRAM_FALLBACK_BOT_TOKEN
  TELEGRAM_FALLBACK_CHAT_ID  (or TELEGRAM_CHAT_ID)
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

WORKSPACE = Path(os.environ.get("TP_WORKSPACE", "/workspace"))
TG_TOKEN  = os.environ.get("TELEGRAM_FALLBACK_BOT_TOKEN", "")
TG_CHAT   = os.environ.get("TELEGRAM_FALLBACK_CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID", "")
MAX_MSG   = 3800   # leave headroom under Telegram's 4096 char limit


def tg_send(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT:
        print(f"[report] no Telegram creds, dumping to stdout:\n{text}")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    # Telegram limit: 4096 chars. Chunk if needed.
    for i in range(0, len(text), MAX_MSG):
        chunk = text[i:i+MAX_MSG]
        data = urllib.parse.urlencode({
            "chat_id": TG_CHAT,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }).encode()
        try:
            with urllib.request.urlopen(url, data=data, timeout=15) as resp:
                print(f"[report] tg HTTP {resp.status}")
        except Exception as exc:
            print(f"[report] tg send failed: {exc}")


def html_esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def collect_results() -> list[dict]:
    results_dir = WORKSPACE / "triple-pendulum-sim2real" / "results"
    files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for f in files[:5]:  # last 5 results
        try:
            data = json.loads(f.read_text())
            out.append({"path": str(f), "name": f.name, **data})
        except Exception as e:
            out.append({"path": str(f), "name": f.name, "error": str(e)})
    return out


def collect_done_blocks() -> list[str]:
    """Find each 'DONE in ... EP0..7 / overall' block in bootstrap.log.
    Uses line-anchored multiline matching so two consecutive training blocks
    don't bleed into each other via the non-greedy .*?"""
    log_path = WORKSPACE / "bootstrap.log"
    if not log_path.exists():
        return []
    text = log_path.read_text(errors="replace")
    blocks = []
    # Collect lines from each "DONE in X s." until the next "DONE in" or EOF.
    segments = re.split(r"(?=^DONE in [\d\.]+s\.)", text, flags=re.MULTILINE)
    for seg in segments:
        if not seg.startswith("DONE in"):
            continue
        # Truncate at first occurrence of a new top-level log section marker
        block = seg[:800].rstrip()
        if len(seg) > 800:
            block += "..."
        blocks.append(block)
    return blocks


def fmt_result(r: dict) -> str:
    if "error" in r:
        return f"<b>{html_esc(r['name'])}</b>: error {html_esc(r['error'])}"
    metrics = r.get("metrics", {})
    overall = metrics.get("overall_success_rate")
    overall_pct = f"{overall*100:.0f}%" if overall is not None else "?"
    lines = [f"<b>{html_esc(r.get('milestone','?'))}</b> <code>{html_esc(r['name'])}</code>",
             f"  overall: <b>{overall_pct}</b>"]
    for ep in range(8):
        sr = metrics.get(f"ep{ep}_success_rate")
        ln = metrics.get(f"ep{ep}_length_mean")
        if sr is not None:
            lines.append(f"  EP{ep}: {sr*100:>3.0f}%  len={ln:.0f}")
    return "\n".join(lines)


def main() -> int:
    pieces = ["📊 <b>Training results dump</b>"]

    results = collect_results()
    if results:
        pieces.append("")
        pieces.append(f"<b>Recent results JSONs ({len(results)})</b>")
        for r in results:
            pieces.append("")
            pieces.append(fmt_result(r))

    blocks = collect_done_blocks()
    if blocks:
        pieces.append("")
        pieces.append(f"<b>Bootstrap.log DONE blocks ({len(blocks)})</b>")
        for b in blocks[-3:]:  # last 3 only
            pieces.append("<pre>" + html_esc(b) + "</pre>")

    msg = "\n".join(pieces)
    print(msg)
    tg_send(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

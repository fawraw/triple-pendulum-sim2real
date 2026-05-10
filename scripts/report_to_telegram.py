"""Read training/probe results from /workspace/ and POST a human-friendly
IT-expert summary to Telegram.

Designed to run on RunPod pods that can't reach the LAN-only n8n.
Requires env vars: TELEGRAM_FALLBACK_BOT_TOKEN, TELEGRAM_FALLBACK_CHAT_ID
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
MAX_MSG   = 3800

# ─── EP metadata: code name, IT-friendly description ─────────────────────────
EP_INFO = {
    0: ("DDD", "🟢 All arms hanging",   "Gravity does the work. Trivial baseline."),
    1: ("DDU", "🟡 Top arm up",         "Top arm inverted, base + middle hang free."),
    2: ("DUD", "🟡 Middle arm up",      "Middle arm inverted. Base hangs, top hangs."),
    3: ("DUU", "🟠 Top 2 arms up",      "Top 2 arms inverted on a freely-hanging base."),
    4: ("UDD", "🔴 Only top-tip up",    "Hardest 1-link case: tip inverted, 2 floppy links below."),
    5: ("UDU", "🔴 Top + base up",      "Non-adjacent arms inverted. Middle link is a joker."),
    6: ("UUD", "🔴 Top 2 up, base free","2 upper arms inverted, base segment hangs + shakes."),
    7: ("UUU", "🟠 All arms up",        "Full stack: 3 levels of inversion simultaneously."),
}

THRESHOLD = 0.75  # overall_success_rate goal

# ─── Telegram send ─────────────────────────────────────────────────────────────
def tg(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    for i in range(0, len(text), MAX_MSG):
        chunk = text[i:i+MAX_MSG]
        data = urllib.parse.urlencode({
            "chat_id": TG_CHAT, "text": chunk,
            "parse_mode": "HTML", "disable_web_page_preview": "true",
        }).encode()
        try:
            with urllib.request.urlopen(url, data=data, timeout=20) as r:
                print(f"[tg] HTTP {r.status}")
        except Exception as e:
            print(f"[tg] fail: {e}")

def esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ─── Progress bar ──────────────────────────────────────────────────────────────
def bar(pct: float, width: int = 8) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)

# ─── Verdict emoji ─────────────────────────────────────────────────────────────
def verdict(overall: float | None) -> str:
    if overall is None:     return "❓"
    if overall >= THRESHOLD: return "✅"
    if overall >= 0.60:     return "🟡"
    return "🔴"

# ─── Format one result file ────────────────────────────────────────────────────
def fmt_result(data: dict) -> str:
    metrics   = data.get("metrics", {})
    milestone = data.get("milestone", "?")
    run_name  = data.get("run_name", "?")[:35]
    overall   = metrics.get("overall_success_rate")
    overall_pct = f"{overall*100:.0f}%" if overall is not None else "?"

    v = verdict(overall)
    lines = [
        f"{v} <b>{esc(milestone)}</b>  <code>{esc(run_name)}</code>",
        f"   Overall: <b>{overall_pct}</b> / 75% goal  {bar(overall*100 if overall else 0)}",
        "",
        "   <b>Per-configuration breakdown</b>",
        "   (Think: 3 robot arms, each can point ↑UP or ↓DOWN)",
        "   Policy must keep the ↑ arms stable — ↓ arms hang freely.",
        "",
    ]

    baseline = {4: 0.0, 5: 0.70, 6: 0.0, 7: 0.80}  # M3b CPU baseline
    any_hard_improved = False

    for ep in range(8):
        sr = metrics.get(f"ep{ep}_success_rate")
        ln = metrics.get(f"ep{ep}_length_mean")
        if sr is None:
            continue
        code, label, desc = EP_INFO[ep]
        pct = int(sr * 100)
        b = bar(pct)
        ep_v = "✅" if sr >= 0.80 else ("🟡" if sr >= 0.50 else "🔴")

        # delta vs baseline
        base = baseline.get(ep)
        delta = ""
        if base is not None:
            diff = sr - base
            if diff > 0.05:
                delta = f" ▲{diff*100:.0f}%"
                if ep in (4, 6):
                    any_hard_improved = True
            elif diff < -0.05:
                delta = f" ▼{abs(diff)*100:.0f}%"

        lines.append(
            f"   {ep_v} EP{ep} <b>{pct:>3}%</b> {b}{delta}  "
            f"<i>{esc(label)}</i>"
        )
        if ep in (4, 6):
            lines.append(f"        └ {esc(desc)}")

    lines.append("")

    # Blocking analysis
    ep4 = metrics.get("ep4_success_rate", 0.0)
    ep6 = metrics.get("ep6_success_rate", 0.0)
    if ep4 < 0.5 and ep6 < 0.5:
        lines += [
            "   <b>🔍 What's blocking</b>",
            "   EP4 and EP6 both stuck near 0%.",
            "   These need the TOP arm inverted while the BASE segment(s)",
            "   hang and swing freely. The cart struggles to stabilise the",
            "   top when the bottom wobbles unpredictably.",
            "   → Current fix attempt: adaptive reward (penalises the",
            "     inverted arm 5× more, ignores the hanging ones).",
        ]
    elif ep4 < 0.5:
        lines += [
            "   <b>🔍 EP4 still blocking</b>",
            "   Top-only inversion: 2 floppy links below make it hard.",
        ]
    elif ep6 < 0.5:
        lines += [
            "   <b>🔍 EP6 still blocking</b>",
            "   2 arms up, free-hanging base destabilises the stack.",
        ]
    else:
        lines.append("   <b>🎉 EP4 and EP6 both > 50%! Fix validated!</b>")

    return "\n".join(lines)

# ─── Extract DONE blocks from bootstrap.log ────────────────────────────────────
def collect_done_blocks() -> list[str]:
    log_path = WORKSPACE / "bootstrap.log"
    if not log_path.exists():
        return []
    text = log_path.read_text(errors="replace")
    segments = re.split(r"(?=^DONE in [\d\.]+s\.)", text, flags=re.MULTILINE)
    blocks = []
    for seg in segments:
        if not seg.startswith("DONE in"):
            continue
        blocks.append(seg[:800].rstrip() + ("..." if len(seg) > 800 else ""))
    return blocks

# ─── Collect result files ──────────────────────────────────────────────────────
def collect_results() -> list[dict]:
    results_dir = WORKSPACE / "triple-pendulum-sim2real" / "results"
    if not results_dir.exists():
        return []
    files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for f in files[:5]:
        try:
            out.append({"name": f.name, **json.loads(f.read_text())})
        except Exception as e:
            out.append({"name": f.name, "error": str(e)})
    return out

# ─── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    results = collect_results()
    done_blocks = collect_done_blocks()

    pieces = ["🤖 <b>Triple Pendulum Training Report</b>"]
    pieces.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if results:
        for r in results:
            if "error" in r:
                pieces.append(f"⚠️ {esc(r['name'])}: {esc(r['error'])}")
                continue
            pieces.append("")
            pieces.append(fmt_result(r))
            pieces.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    elif done_blocks:
        # Fallback: parse bootstrap log
        pieces.append("")
        pieces.append("📋 <b>Bootstrap log summary</b>")
        for b in done_blocks[-2:]:
            pieces.append("<pre>" + esc(b[:500]) + "</pre>")
    else:
        pieces.append("")
        pieces.append("⚠️ No result files found on network volume.")
        pieces.append(f"Looked in: <code>{WORKSPACE}/triple-pendulum-sim2real/results/</code>")

    msg = "\n".join(pieces)
    print(msg)
    tg(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

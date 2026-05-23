#!/usr/bin/env python3
"""RunPod safety auditor — list running pods, warn on long-running, optionally
force-stop pods older than --max-age-hours.

Cron-friendly: exits 0 normally, 1 if any pod was force-stopped (so cron can
email/alert). Designed to be run hourly on CT 1018 as a safety net against
runaway pods (root cause of the 2026-05-12 → 2026-05-23 11-day runaway: a
fetch pod whose self-stop via GraphQL podStop silently 403'd, leaving it
running until manually noticed — ~$71 wasted).

Usage:
    RUNPOD_API_KEY=... python3 scripts/runpod_audit.py             # report only
    RUNPOD_API_KEY=... python3 scripts/runpod_audit.py --max-age-hours 12 --kill
    RUNPOD_API_KEY=... python3 scripts/runpod_audit.py --whitelist tp-train-prod
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone


def list_pods(api_key: str) -> list[dict]:
    req = urllib.request.Request(
        "https://rest.runpod.io/v1/pods",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    return data if isinstance(data, list) else data.get("data", [])


def stop_pod(api_key: str, pod_id: str) -> bool:
    req = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{pod_id}/stop",
        headers={"Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=20)
        return True
    except Exception:
        return False


def parse_started(s: str | None) -> datetime | None:
    if not s:
        return None
    # Format observed: "2026-05-12 07:42:10.076 +0000 UTC"
    try:
        s = s.replace(" +0000 UTC", "+00:00").replace(" UTC", "")
        # Trim microseconds extra digits if any
        if "." in s:
            head, dot, tail = s.partition(".")
            tail_digits = ""
            i = 0
            while i < len(tail) and tail[i].isdigit():
                tail_digits += tail[i]
                i += 1
            s = head + "." + tail_digits[:6] + tail[i:]
        return datetime.fromisoformat(s.replace(" ", "T"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age-hours", type=float, default=24.0,
                    help="warn (or kill with --kill) on pods older than this")
    ap.add_argument("--kill", action="store_true",
                    help="actually force-stop offending pods (otherwise report only)")
    ap.add_argument("--whitelist", action="append", default=[],
                    help="pod name (exact) to exempt; can be specified multiple times")
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY env var required", file=sys.stderr)
        return 2

    pods = list_pods(api_key)
    now = datetime.now(timezone.utc)
    running = [p for p in pods if p.get("desiredStatus") == "RUNNING"]

    killed = 0
    print(f"RunPod audit @ {now.strftime('%Y-%m-%d %H:%M:%S')} UTC — "
          f"{len(running)}/{len(pods)} pods running")
    if not running:
        return 0

    for p in running:
        name = p.get("name", "?")
        pid = p.get("id", "?")
        started = parse_started(p.get("lastStartedAt"))
        cost = float(p.get("costPerHr", 0) or 0)
        age_h = (now - started).total_seconds() / 3600 if started else None
        spent = age_h * cost if age_h else 0
        flag = ""
        if name in args.whitelist:
            flag = "WHITELISTED"
        elif age_h is not None and age_h > args.max_age_hours:
            flag = f"OLD (>{args.max_age_hours}h)"
        age_str = f"{age_h:.1f}h" if age_h is not None else "?"
        print(f"  [{flag:>15}] {name:30} {pid}  age={age_str:>7}  "
              f"${cost:.2f}/h  spent=${spent:.2f}")
        if args.kill and flag.startswith("OLD"):
            if stop_pod(api_key, pid):
                print(f"    → STOPPED {pid}")
                killed += 1
            else:
                print(f"    → FAILED to stop {pid}")

    if killed:
        print(f"\n{killed} pods stopped")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

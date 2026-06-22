#!/usr/bin/env python3
"""Sync a triple-pendulum n8n workflow JSON to the live n8n instance.

Why this exists
---------------
The two workflows under ``n8n/`` ship with ``YOUR_*`` placeholders instead of
real secrets (so the repo stays secret-free). Deploying or rotating secrets
therefore requires substituting the placeholders with the live values and
pushing the result into n8n. docs/runbook.md referenced this script for years
but it did not exist; the secret-rotation procedure was a manual, error-prone
``python3 -m`` block.

IMPORTANT (learned 2026-06-22): editing ``workflow_entity.nodes`` directly in
the sqlite DB does NOT update the running n8n (its in-memory/compiled copy is
authoritative). Use n8n's own import path so the change actually takes effect:

    # 1. render the filled workflow (secrets pulled from the environment)
    PIPELINE_SECRET=... LAUNCHER_SECRET=... TELEGRAM_BOT_TOKEN=... \\
    RUNPOD_API_KEY=... RUNPOD_POD_ID=... \\
      python3 scripts/sync_n8n_workflow.py n8n/triple_pendulum_pipeline.json \\
        --out /tmp/filled.json
    # 2. import via the n8n CLI on the n8n host, then restart so webhooks
    #    re-register with the new code:
    sudo HOME=/ n8n import:workflow --input=/tmp/filled.json
    sudo systemctl restart n8n
    shred -u /tmp/filled.json

Or push straight to the REST API when an API key is configured:

    N8N_API_URL=http://10.1.4.226:5678 N8N_API_KEY=... PIPELINE_SECRET=... ... \\
      python3 scripts/sync_n8n_workflow.py n8n/triple_pendulum_pipeline.json --push

Placeholders are of the form ``YOUR_<NAME>`` and are filled from the env var
``<NAME>`` (e.g. ``YOUR_PIPELINE_SECRET`` <- ``$PIPELINE_SECRET``).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

PLACEHOLDER_RE = re.compile(r"YOUR_([A-Z0-9_]+)")


def find_placeholders(text: str) -> set[str]:
    """Return the set of env-var names referenced as YOUR_<NAME> in text."""
    return {m.group(1) for m in PLACEHOLDER_RE.finditer(text)}


def substitute(text: str, env: dict[str, str]) -> tuple[str, list[str]]:
    """Replace every YOUR_<NAME> with env[NAME].

    Returns (filled_text, missing) where `missing` lists the NAMEs that had no
    value in `env` (those placeholders are left untouched).
    """
    missing: list[str] = []

    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name in env and env[name] != "":
            return env[name]
        if name not in missing:
            missing.append(name)
        return m.group(0)

    return PLACEHOLDER_RE.sub(repl, text), missing


def _push(api_url: str, api_key: str, workflow: dict) -> None:
    wid = workflow.get("id")
    body = json.dumps(workflow).encode()
    if wid:
        url = f"{api_url.rstrip('/')}/api/v1/workflows/{wid}"
        method = "PUT"
    else:
        url = f"{api_url.rstrip('/')}/api/v1/workflows"
        method = "POST"
    req = urllib.request.Request(
        url, data=body, method=method,
        headers={"Content-Type": "application/json", "X-N8N-API-KEY": api_key},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(f"{method} {url} -> HTTP {resp.status}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workflow", help="path to the workflow JSON template (with YOUR_* placeholders)")
    p.add_argument("--out", help="write the filled workflow JSON to this path")
    p.add_argument("--push", action="store_true", help="push to the n8n REST API (needs N8N_API_URL + N8N_API_KEY)")
    p.add_argument("--allow-missing", action="store_true", help="do not fail if some placeholders have no env value")
    args = p.parse_args(argv)

    raw = open(args.workflow, encoding="utf-8").read()
    needed = find_placeholders(raw)
    filled, missing = substitute(raw, dict(os.environ))

    if missing and not args.allow_missing:
        print(f"ERROR: missing env values for: {', '.join(sorted(missing))}", file=sys.stderr)
        print(f"(referenced placeholders: {', '.join('YOUR_'+n for n in sorted(needed))})", file=sys.stderr)
        return 2

    workflow = json.loads(filled)  # validate it is still valid JSON

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(workflow, f, indent=2)
        os.chmod(args.out, 0o600)
        print(f"wrote {args.out} (0600); filled {len(needed) - len(missing)}/{len(needed)} placeholders")
    if args.push:
        api_url = os.environ.get("N8N_API_URL")
        api_key = os.environ.get("N8N_API_KEY")
        if not api_url or not api_key:
            print("ERROR: --push needs N8N_API_URL and N8N_API_KEY", file=sys.stderr)
            return 2
        _push(api_url, api_key, workflow)
    if not args.out and not args.push:
        print(f"(dry run) {len(needed) - len(missing)}/{len(needed)} placeholders filled; "
              f"missing: {sorted(missing) or 'none'}. Use --out or --push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

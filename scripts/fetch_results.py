"""Fetch training results from the RunPod network volume via a GitHub gist.

Usage:
    python3 scripts/fetch_results.py              # last 5 results
    python3 scripts/fetch_results.py --last 10    # last 10
    python3 scripts/fetch_results.py --milestone M3b_v3  # filter by milestone

How it works:
  1. Creates a private GitHub gist (placeholder).
  2. Spawns a one-shot RunPod pod that reads /workspace/.../results/*.json
     and PATCHes the gist with the data.
  3. Polls until the pod exits (~30-60 seconds).
  4. Reads the gist, deletes it, prints the results.

Requirements (set as env vars or in credentials/12_triple_pendulum.md):
  RUNPOD_API_KEY     - RunPod API key
  GH_TOKEN           - GitHub personal access token (gist:write scope)
  RUNPOD_NETWORK_VOL - network volume ID (default: 60peck51dg)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

def _read_cred(key: str, cred_file: str = "", default: str = "") -> str:
    """Read from env or credentials file."""
    val = os.environ.get(key, "")
    if val:
        return val
    if cred_file:
        try:
            text = (ROOT.parent / "credentials" / cred_file).read_text()
            for line in text.splitlines():
                if key in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        return parts[1].strip()
        except Exception:
            pass
    return default

RUNPOD_API_KEY   = _read_cred("RUNPOD_API_KEY",   "12_triple_pendulum.md")
NETWORK_VOLUME   = _read_cred("RUNPOD_NETWORK_VOL", default="60peck51dg")
GPU_TYPE         = "NVIDIA RTX A5000"
IMAGE            = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# GitHub token: try gh CLI first, then env
def _gh_token() -> str:
    tok = os.environ.get("GH_TOKEN", "")
    if tok:
        return tok
    try:
        result = subprocess.run(
            [str(Path.home() / "bin" / "gh"), "auth", "token"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        pass
    return ""

# ── GitHub gist helpers ──────────────────────────────────────────────────────
def _gh_request(method: str, url: str, body: dict | None = None, token: str = "") -> dict:
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Authorization": f"token {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)

def create_gist(token: str) -> str:
    resp = _gh_request("POST", "https://api.github.com/gists", {
        "description": "tp-results (auto-deleted after read)",
        "public": False,
        "files": {"results.json": {"content": "pending"}},
    }, token)
    return resp["id"]

def read_gist(gist_id: str, token: str) -> str:
    resp = _gh_request("GET", f"https://api.github.com/gists/{gist_id}", token=token)
    return resp.get("files", {}).get("results.json", {}).get("content", "")

def delete_gist(gist_id: str, token: str) -> None:
    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}",
        headers={"Authorization": f"token {token}"},
        method="DELETE",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass

# ── RunPod helpers ───────────────────────────────────────────────────────────
# All RunPod calls use the REST API.  GraphQL `podStop`/`podTerminate` silently
# return 403 in some auth contexts and were the root cause of the 11-day
# `tp-fetch-results` runaway in 2026-05-12 → 2026-05-23.

def spawn_pod(gist_id: str, gh_token: str, last: int, milestone: str) -> str:
    milestone_filter = f"and d.get('milestone') == '{milestone}'" if milestone else ""
    cmd = f"""set +e
for i in 1 2 3 4 5; do getent hosts api.github.com >/dev/null 2>&1 && break; sleep 3; done
python3 -c "
import json, urllib.request
from pathlib import Path

ws = Path('/workspace/triple-pendulum-sim2real/results')
files = sorted(ws.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
out = []
for f in files:
    try:
        d = json.loads(f.read_text())
        m = d.get('metrics', {{}})
        if True {milestone_filter}:
            row = {{'file': f.name[:60], 'milestone': d.get('milestone'), 'run': (d.get('run_name') or '')[:40], 'ts': d.get('timestamp','')}}
            for ep in range(8):
                sr = m.get(f'ep{{ep}}_success_rate')
                if sr is not None: row[f'ep{{ep}}'] = round(sr * 100)
            row['overall'] = round((m.get('overall_success_rate') or 0) * 100, 1)
            out.append(row)
            if len(out) >= {last}: break
    except Exception as e:
        out.append({{'file': f.name, 'error': str(e)}})

patch = json.dumps({{'files': {{'results.json': {{'content': json.dumps(out, indent=2)}}}}}}).encode()
req = urllib.request.Request(
    'https://api.github.com/gists/{gist_id}',
    data=patch,
    headers={{'Authorization': 'token {gh_token}', 'Content-Type': 'application/json'}},
    method='PATCH')
with urllib.request.urlopen(req, timeout=15) as r:
    print('updated gist:', r.status)
"
sleep 3
# Self-stop via REST API. GraphQL podStop sometimes returns 403 in this
# auth context — root cause of the 2026-05-12 → 2026-05-23 11-day runaway.
curl -sf -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \\
  -H 'Authorization: Bearer {RUNPOD_API_KEY}' >/dev/null || true
"""
    payload = {
        "name": "tp-fetch-results",
        "gpuTypeIds": [GPU_TYPE],
        "gpuCount": 1,
        "containerDiskInGb": 5,
        "volumeMountPath": "/workspace",
        "networkVolumeId": NETWORK_VOLUME,
        "computeType": "GPU",
        "cloudType": "SECURE",
        "imageName": IMAGE,
        "env": {"RUNPOD_API_KEY": RUNPOD_API_KEY},
        "dockerStartCmd": ["bash", "-c", cmd],
    }
    req = urllib.request.Request(
        "https://rest.runpod.io/v1/pods",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        resp = json.load(r)
        return resp["id"]

def _rest_get_pod(pod_id: str) -> dict:
    req = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{pod_id}",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)

def wait_pod_exit(pod_id: str, timeout: int = 300) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            p = _rest_get_pod(pod_id)
            status = p.get("desiredStatus", "?")
            print(f"  [{int(time.time()-start):>3}s] {status}", end="\r", flush=True)
            if status == "EXITED":
                print()
                return True
        except Exception:
            pass
        time.sleep(8)
    print()
    return False

def terminate_pod(pod_id: str) -> None:
    """Force-stop via REST API (GraphQL podTerminate returns 403 in some auth contexts)."""
    try:
        req = urllib.request.Request(
            f"https://rest.runpod.io/v1/pods/{pod_id}/stop",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=15)
    except Exception:
        pass

# ── Pretty print ─────────────────────────────────────────────────────────────
EP_LABELS = ["DDD ↓↓↓", "UDD ↑↓↓", "DUD ↓↑↓", "UUD ↑↑↓",
             "DDU ↓↓↑", "UDU ↑↓↑", "DUU ↓↑↑", "UUU ↑↑↑"]

def bar(pct: float, w: int = 8) -> str:
    f = round(pct / 100 * w)
    return "█" * f + "░" * (w - f)

def pretty_print(results: list[dict]) -> None:
    for r in results:
        milestone = r.get("milestone", "?")
        run = r.get("run", "?")
        overall = r.get("overall", 0)
        v = "✅" if overall >= 75 else ("🟡" if overall >= 60 else "🔴")
        print(f"\n{v} {milestone}  |  {run}")
        print(f"   overall: {overall:.1f}% / 75% target  {bar(overall)}")
        for ep in range(8):
            pct = r.get(f"ep{ep}")
            if pct is not None:
                ev = "✅" if pct >= 80 else ("🟡" if pct >= 50 else "🔴")
                print(f"   {ev} EP{ep} {EP_LABELS[ep]:10} {pct:>3}% {bar(pct, 6)}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch RunPod training results via GitHub gist")
    parser.add_argument("--last", type=int, default=5, help="Number of results to fetch")
    parser.add_argument("--milestone", default="", help="Filter by milestone (e.g. M3b_v3)")
    args = parser.parse_args()

    if not RUNPOD_API_KEY:
        print("ERROR: RUNPOD_API_KEY not set", file=sys.stderr)
        return 1

    gh_token = _gh_token()
    if not gh_token:
        print("ERROR: GitHub token not found (set GH_TOKEN or login with gh CLI)", file=sys.stderr)
        return 1

    print(f"Creating gist...")
    gist_id = create_gist(gh_token)
    print(f"Gist created: {gist_id}")

    print(f"Spawning pod to read results...")
    pod_id = spawn_pod(gist_id, gh_token, args.last, args.milestone)
    print(f"Pod: {pod_id}")

    print("Waiting for pod to read & exit (~30-60s)...")
    exited = wait_pod_exit(pod_id, timeout=240)
    if not exited:
        print("Timeout — terminating pod")
        terminate_pod(pod_id)

    print("Reading gist...")
    content = read_gist(gist_id, gh_token)
    delete_gist(gist_id, gh_token)
    print("Gist deleted.")

    try:
        results = json.loads(content)
        if not results:
            print("No results found on volume.")
            return 0
        pretty_print(results)
        print()
        return 0
    except json.JSONDecodeError:
        print(f"Could not parse results:\n{content[:500]}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

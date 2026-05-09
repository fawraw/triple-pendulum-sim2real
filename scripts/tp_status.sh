#!/usr/bin/env bash
# Triple Pendulum pipeline status — single command for "where are we"
# Usage: ./scripts/tp_status.sh
#
# Hits the launcher, MLflow, and (if configured) n8n. No secrets needed for
# read-only endpoints.

set -uo pipefail

LAUNCHER="${TP_LAUNCHER_URL:-http://10.1.4.232:8765}"
MLFLOW="${MLFLOW_TRACKING_URI:-http://10.1.4.230:5000}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
dim()  { printf '\033[2m%s\033[0m\n' "$*"; }

bold "=== Launcher ($LAUNCHER) ==="
status=$(curl -s --max-time 5 "$LAUNCHER/status" 2>/dev/null)
if [ -z "$status" ]; then
    echo "  unreachable"
else
    echo "$status" | python3 -c "
import json, sys
d = json.load(sys.stdin)
count = d.get('count', '?')
print(f'  active processes: {count}')
for s in d.get('sessions', [])[:3]:
    print('  - ' + s[:120])
" 2>/dev/null || echo "$status"
fi

echo ""
bold "=== MLflow ($MLFLOW) ==="
runs=$(curl -s --max-time 5 -X POST -H "Content-Type: application/json" \
    "$MLFLOW/api/2.0/mlflow/runs/search" \
    -d '{"experiment_ids":["1","2"],"max_results":3,"order_by":["start_time DESC"]}' 2>/dev/null)
if [ -z "$runs" ]; then
    echo "  unreachable"
else
    echo "$runs" | python3 -c '
import json, sys, datetime
d = json.load(sys.stdin)
for r in d.get("runs", [])[:3]:
    info = r["info"]
    metrics = {m["key"]: m["value"] for m in r["data"].get("metrics", [])}
    name = info.get("run_name", "?")[:40]
    status = info.get("status", "?")
    ts = metrics.get("timesteps")
    rew = metrics.get("rollout_ep_rew_mean")
    overall = metrics.get("final_overall_success_rate")
    started = datetime.datetime.fromtimestamp(int(info["start_time"])/1000).strftime("%m-%d %H:%M")
    parts = [f"started={started}", f"status={status}"]
    if ts is not None:
        parts.append(f"steps={int(ts):,}")
    if rew is not None:
        parts.append(f"reward={rew:.1f}")
    if overall is not None:
        parts.append(f"overall={overall:.2%}")
    print(f"  {name}")
    print(f"    " + "  ".join(parts))
' 2>/dev/null || echo "$runs"
fi

echo ""
bold "=== Recent results JSON ==="
ssh -o ConnectTimeout=3 root@10.1.4.232 \
    "ls -1t /opt/triple-pendulum/repo/results/*.json 2>/dev/null | head -3" 2>/dev/null \
    | while read -r f; do
        echo "  $(basename "$f")"
    done || echo "  (CT 1018 unreachable)"

#!/usr/bin/env bash
# RunPod / generic GPU pod bootstrap script.
#
# Designed to run as the pod's startup command (Container Start Command in
# RunPod, or `docker run ... runpod_bootstrap.sh` locally).
#
# Required env vars (set in RunPod template or `--env`):
#   TP_STAGE_MODULE         e.g. training.train_m4_transitions
#   TP_STAGE_CONFIG         e.g. training/configs/m4_transitions_tqc.yaml
#
# Optional env vars:
#   TP_REPO_URL             default: https://github.com/fawraw/triple-pendulum-sim2real.git
#   TP_REPO_BRANCH          default: main
#   TP_REPO_DIR             default: /workspace/triple-pendulum-sim2real
#   MLFLOW_TRACKING_URI     default: http://10.1.4.230:5000  (only reachable inside Lab Perso VPN)
#   N8N_PIPELINE_WEBHOOK    n8n callback URL (optional, set in RunPod template)
#   N8N_PIPELINE_SECRET     shared secret for the webhook
#   TP_AUTO_SHUTDOWN        default: 1 — set to 0 to keep the pod alive after training
#   RUNPOD_API_KEY          required if TP_AUTO_SHUTDOWN=1, used to call the RunPod API
#   RUNPOD_POD_ID           required if TP_AUTO_SHUTDOWN=1 (RunPod injects this automatically)

set -uo pipefail   # NOT -e: we want the trap to fire on errors, not silent exit

# Mirror everything to a persistent log on the network volume so we can
# inspect AFTER a crash (the pod-level SSH only works when the container
# is alive — but if the bootstrap fails we lose SSH access).
mkdir -p /workspace 2>/dev/null || true
exec > >(tee -a /workspace/bootstrap.log) 2>&1

REPO_URL="${TP_REPO_URL:-https://github.com/fawraw/triple-pendulum-sim2real.git}"
REPO_BRANCH="${TP_REPO_BRANCH:-main}"
REPO_DIR="${TP_REPO_DIR:-/workspace/triple-pendulum-sim2real}"
MODULE="${TP_STAGE_MODULE:-}"
CONFIG="${TP_STAGE_CONFIG:-}"
AUTO_SHUTDOWN="${TP_AUTO_SHUTDOWN:-1}"

# CRITICAL: keep the container alive on ANY exit (including errors) when
# AUTO_SHUTDOWN=0, so the operator can SSH in and inspect bootstrap.log.
# Without this, a `set -e` style failure would terminate the container and
# make post-mortem impossible.
keep_alive_on_exit() {
    rc=$?
    if [ "${AUTO_SHUTDOWN:-1}" != "1" ]; then
        echo ""
        echo "[bootstrap] === EXIT (rc=$rc) at $(date -u +%FT%TZ) ==="
        echo "[bootstrap] AUTO_SHUTDOWN=0; keeping container alive for SSH."
        echo "[bootstrap] tail -f /workspace/bootstrap.log to follow."
        exec sleep infinity
    fi
}
trap keep_alive_on_exit EXIT

echo "[bootstrap] === START $(date -u +%FT%TZ) ==="
echo "[bootstrap] AUTO_SHUTDOWN=$AUTO_SHUTDOWN  MODULE=$MODULE  CONFIG=$CONFIG"

if [ -z "$MODULE" ] || [ -z "$CONFIG" ]; then
    echo "ERROR: TP_STAGE_MODULE and TP_STAGE_CONFIG must be set."
    echo "  e.g. TP_STAGE_MODULE=training.train_m4_transitions"
    echo "       TP_STAGE_CONFIG=training/configs/m4_transitions_tqc.yaml"
    exit 1
fi

echo "=== Triple Pendulum bootstrap ==="
echo "  repo:    $REPO_URL ($REPO_BRANCH)"
echo "  module:  $MODULE"
echo "  config:  $CONFIG"
echo "  GPU:     $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")' 2>/dev/null || echo unknown)"
echo ""

# 0. Ensure OS deps are present (needed for headless MuJoCo + git clone).
# Idempotent: skip apt-get if all packages already installed (custom Dockerfile case).
need_apt=0
for pkg in git tmux ffmpeg libgl1 libosmesa6 libglfw3 curl ca-certificates; do
    dpkg -s "$pkg" >/dev/null 2>&1 || need_apt=1
done
if [ "$need_apt" = "1" ]; then
    echo "[bootstrap] installing OS deps via apt-get (one-time, ~30s)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq \
        git tmux ffmpeg \
        libgl1 libglu1-mesa libosmesa6 libegl1 libglfw3 \
        ca-certificates curl jq
fi

# 1. Clone or pull
if [ -d "$REPO_DIR/.git" ]; then
    echo "[bootstrap] pulling latest from $REPO_BRANCH"
    cd "$REPO_DIR"
    git fetch origin "$REPO_BRANCH"
    git reset --hard "origin/$REPO_BRANCH"
else
    echo "[bootstrap] cloning $REPO_URL"
    git clone --branch "$REPO_BRANCH" --depth 50 "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi
git log --oneline -1

# 2. Verify dependencies (the Dockerfile already installed them, but a stale
# image may need a delta install).
pip install -q -r requirements.txt

# 3. Run training
echo ""
echo "=== Starting training ==="
LOG_FILE="/workspace/training.log"
set +e
PYTHONUNBUFFERED=1 python -m "$MODULE" --config "$CONFIG" 2>&1 | tee "$LOG_FILE"
TRAIN_RC=${PIPESTATUS[0]}
set -e
echo ""
echo "=== Training exit code: $TRAIN_RC ==="

# 4. Auto-shutdown the pod (RunPod API), or stay alive for SSH inspection.
# We auto-shutdown even on failure so a crashed run doesn't burn $/hour idle.
if [ "$AUTO_SHUTDOWN" = "1" ]; then
    if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
        echo "[bootstrap] requesting pod shutdown via RunPod API"
        curl -s -X POST "https://api.runpod.io/graphql" \
            -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id } }\"}" \
            >/dev/null && echo "[bootstrap] shutdown requested" \
            || echo "[bootstrap] WARN shutdown request failed; pod will stay alive"
    else
        echo "[bootstrap] AUTO_SHUTDOWN=1 but RUNPOD_API_KEY/RUNPOD_POD_ID missing; staying alive."
        echo "[bootstrap] Stop manually: https://www.runpod.io/console/pods"
    fi
    exit $TRAIN_RC
fi

# AUTO_SHUTDOWN=0: the EXIT trap installed at the top will keep the container
# alive via `exec sleep infinity`, so the operator can SSH in. We just fall
# off the end of the script naturally.
echo ""
echo "[bootstrap] training exited with code $TRAIN_RC."
echo "[bootstrap] AUTO_SHUTDOWN=0 — EXIT trap will keep container alive."

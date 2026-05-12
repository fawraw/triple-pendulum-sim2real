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
#   TP_REPO_BRANCH          default: main  (used if TP_REPO_REF unset)
#   TP_REPO_REF             commit SHA / tag to pin to (overrides TP_REPO_BRANCH)
#   TP_REPO_DIR             default: /workspace/triple-pendulum-sim2real
#   MLFLOW_TRACKING_URI     default: file:/workspace/mlruns (network volume, persistent)
#   N8N_PIPELINE_WEBHOOK    n8n callback URL (optional, set in RunPod template)
#   N8N_PIPELINE_SECRET     shared secret for the webhook
#   TELEGRAM_FALLBACK_BOT_TOKEN  fallback notification token (recommended for cloud)
#   TELEGRAM_FALLBACK_CHAT_ID    chat id for fallback
#   TP_AUTO_SHUTDOWN        default: 1 — set to 0 to keep the pod alive after training
#   TP_IDLE_SHUTDOWN_MIN    default: 30 — minutes of GPU<5% before forced shutdown
#                                         (when AUTO_SHUTDOWN=0; safety against forgotten pods)
#   RUNPOD_API_KEY          required for any auto-shutdown path
#   RUNPOD_POD_ID           required (RunPod injects this automatically)
#   TP_MAX_RUNTIME_MIN      default: 480 (8h) — hard wall-clock kill for the training process.
#                                         Prevents per_ep_eval from running forever when GPU
#                                         inference keeps util >5% (fooling the idle watchdog).

set -uo pipefail   # NOT -e: we want the trap to fire on errors, not silent exit

# Mirror everything to a persistent log on the network volume so we can
# inspect AFTER a crash (the pod-level SSH only works when the container
# is alive — but if the bootstrap fails we lose SSH access).
mkdir -p /workspace 2>/dev/null || true
exec > >(tee -a /workspace/bootstrap.log) 2>&1

REPO_URL="${TP_REPO_URL:-https://github.com/fawraw/triple-pendulum-sim2real.git}"
REPO_BRANCH="${TP_REPO_BRANCH:-main}"
REPO_REF="${TP_REPO_REF:-}"
REPO_DIR="${TP_REPO_DIR:-/workspace/triple-pendulum-sim2real}"
MODULE="${TP_STAGE_MODULE:-}"
CONFIG="${TP_STAGE_CONFIG:-}"
AUTO_SHUTDOWN="${TP_AUTO_SHUTDOWN:-1}"
IDLE_SHUTDOWN_MIN="${TP_IDLE_SHUTDOWN_MIN:-30}"
MAX_RUNTIME_MIN="${TP_MAX_RUNTIME_MIN:-480}"  # 8h default wall-clock limit

# Defaults that the rest of the codebase expects to be set.
# - MUJOCO_GL=osmesa: required for headless rendering on a fresh PyTorch image
#   (without it, MuJoCo tries GLFW and crashes).
# - MLFLOW_TRACKING_URI on the network volume so runs survive pod death.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:/workspace/mlruns}"

# Idle-pod cost guard: when AUTO_SHUTDOWN=0 (interactive/debug mode), an
# operator may forget the pod over a weekend at $0.30/hr. Spawn a watchdog
# that triggers a podStop after IDLE_SHUTDOWN_MIN consecutive minutes of
# GPU<5%. Disabled if RUNPOD_API_KEY is unset (no way to call the API).
spawn_idle_watchdog() {
    # Always spawn watchdog regardless of AUTO_SHUTDOWN — even with AUTO_SHUTDOWN=1,
    # if per_ep_eval gets stuck the pod would run forever without a safety net.
    # Use a longer threshold (2h) when AUTO_SHUTDOWN=1 to give eval time to finish.
    if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
        return
    fi
    # When AUTO_SHUTDOWN=1, use 2h threshold; otherwise use TP_IDLE_SHUTDOWN_MIN
    local effective_threshold=${IDLE_SHUTDOWN_MIN}
    if [ "${AUTO_SHUTDOWN:-1}" = "1" ]; then
        effective_threshold=120  # 2h safety net for stuck per_ep_eval
    fi
    (
        idle=0
        check_every=60
        threshold_seconds=$((effective_threshold * 60))
        while true; do
            sleep "$check_every"
            util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
            util=${util:-0}
            if [ "$util" -lt 5 ] 2>/dev/null; then
                idle=$((idle + check_every))
            else
                idle=0
            fi
            if [ "$idle" -ge "$threshold_seconds" ]; then
                echo "[idle-watchdog] GPU idle ${IDLE_SHUTDOWN_MIN}min — requesting podStop" >> /workspace/bootstrap.log
                curl -s -X POST "https://api.runpod.io/graphql" \
                    -H "Authorization: Bearer $RUNPOD_API_KEY" \
                    -H "Content-Type: application/json" \
                    -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id } }\"}" \
                    >> /workspace/bootstrap.log 2>&1
                exit 0
            fi
        done
    ) &
    echo "[bootstrap] spawned idle watchdog (PID $!) — shutdown if GPU<5% for ${IDLE_SHUTDOWN_MIN}min"
}

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
        # Use sleep + wait (not exec sleep) so SIGTERM can flush logs cleanly.
        # SIGTERM trap installed below.
        sleep infinity &
        wait $!
    fi
}
trap keep_alive_on_exit EXIT
# Flush logs on SIGTERM (RunPod graceful shutdown sends SIGTERM).
trap 'echo "[bootstrap] SIGTERM at $(date -u +%FT%TZ); flushing logs"; sync; exit 0' TERM INT

echo "[bootstrap] === START $(date -u +%FT%TZ) ==="
echo "[bootstrap] AUTO_SHUTDOWN=$AUTO_SHUTDOWN  MODULE=$MODULE  CONFIG=$CONFIG"
echo "[bootstrap] MUJOCO_GL=$MUJOCO_GL  MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"

if [ -z "$MODULE" ] || [ -z "$CONFIG" ]; then
    echo "ERROR: TP_STAGE_MODULE and TP_STAGE_CONFIG must be set."
    echo "  e.g. TP_STAGE_MODULE=training.train_m4_transitions"
    echo "       TP_STAGE_CONFIG=training/configs/m4_transitions_tqc.yaml"
    exit 1
fi

echo "=== Triple Pendulum bootstrap ==="
echo "  repo:    $REPO_URL ($REPO_BRANCH${REPO_REF:+ @$REPO_REF})"
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

# 1. Wait for DNS to be ready, then clone or pull.
# (Container DNS can take 5-30s after start; git was failing immediately on
# fresh pods with "Could not resolve host: github.com".)
echo "[bootstrap] waiting for DNS..."
for i in $(seq 1 30); do
    getent hosts github.com >/dev/null 2>&1 && break
    [ $i -eq 30 ] && echo "[bootstrap] WARN DNS still not resolving github.com after 60s"
    sleep 2
done

if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    if [ -n "$REPO_REF" ]; then
        echo "[bootstrap] fetching pinned ref: $REPO_REF"
        git fetch --tags --depth 50 origin "$REPO_REF" 2>/dev/null || git fetch --depth 50 origin "$REPO_BRANCH"
        git reset --hard "$REPO_REF"
    else
        echo "[bootstrap] pulling latest from $REPO_BRANCH"
        git fetch --depth 50 origin "$REPO_BRANCH"
        git reset --hard "origin/$REPO_BRANCH"
    fi
else
    echo "[bootstrap] cloning $REPO_URL"
    git clone --branch "$REPO_BRANCH" --depth 50 "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    if [ -n "$REPO_REF" ]; then
        git fetch --tags --depth 50 origin "$REPO_REF" 2>/dev/null || true
        git reset --hard "$REPO_REF"
    fi
fi
echo "[bootstrap] resolved commit: $(git rev-parse HEAD)"
git log --oneline -1

# 2. Install Python deps. Two complications on the runpod/pytorch image:
#
#   a) blinker is apt-installed as python3-blinker (distutils-managed). pip
#      refuses to uninstall distutils-tracked packages, so the entire install
#      aborts when mlflow → Flask → blinker pulls a blinker upgrade. Fix:
#      pre-install blinker via pip (--ignore-installed --no-deps) so it
#      becomes pip-managed. Pin range to avoid future blinker 2.x breakage.
#
#   b) The image ships torch 2.4.x + CUDA 12.4 user-space, matched to the
#      host driver (CUDA 12.7). Using `pip install --ignore-installed -r
#      requirements.txt` would force-reinstall torch and pip would resolve
#      `torch>=2.2` to the latest (torch 2.11, CUDA 13) — INCOMPATIBLE with
#      the host driver. Solution: regular pip install (no --ignore-installed)
#      with default upgrade-strategy=only-if-needed, so torch 2.4.1 already
#      installed satisfies the >=2.2 constraint and is NOT touched.
pip install --upgrade pip
echo "[bootstrap] pre-installing blinker (>=1.6.2,<2) via pip to bypass apt distutils conflict"
pip install --ignore-installed --no-deps "blinker>=1.6.2,<2"
echo "[bootstrap] installing requirements.txt (default upgrade strategy keeps image's torch 2.4.1)"
pip install -r requirements.txt
python -c "
import torch, mlflow, sb3_contrib, mujoco
cuda_ok = torch.cuda.is_available()
print(f'deps OK | torch {torch.__version__} (cuda={cuda_ok}) | mlflow {mlflow.__version__} | sb3_contrib {sb3_contrib.__version__} | mujoco {mujoco.__version__}')
assert cuda_ok, f'CUDA not available — driver mismatch (torch {torch.__version__} expects newer driver?)'
"

# Spawn idle-pod watchdog now that everything is healthy.
spawn_idle_watchdog

# 3. Run training (with hard wall-clock limit to prevent per_ep_eval from running forever
# when GPU inference keeps utilization >5% and fools the idle watchdog).
echo ""
echo "=== Starting training (wall-clock limit: ${MAX_RUNTIME_MIN}min) ==="
LOG_FILE="/workspace/training.log"
set +e
PYTHONUNBUFFERED=1 timeout "${MAX_RUNTIME_MIN}m" \
    python -m "$MODULE" --config "$CONFIG" 2>&1 | tee "$LOG_FILE"
TRAIN_RC=${PIPESTATUS[0]}
set -e
if [ "$TRAIN_RC" = "124" ]; then
    echo "[bootstrap] WARN: training killed by ${MAX_RUNTIME_MIN}min wall-clock timeout (per_ep_eval stuck?)"
fi
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
# alive via `sleep infinity & wait`. Idle watchdog will force-stop after
# IDLE_SHUTDOWN_MIN if GPU stays idle, so cost is bounded.
echo ""
echo "[bootstrap] training exited with code $TRAIN_RC."
echo "[bootstrap] AUTO_SHUTDOWN=0 — EXIT trap will keep container alive (idle watchdog will stop pod after ${IDLE_SHUTDOWN_MIN}min idle)."

#!/usr/bin/env bash
# Bootstrap a Python venv with MuJoCo + SB3 for this project.
# Requires Python >= 3.10 (TQC needs sb3-contrib which needs >= 3.10).
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v python3.11 >/dev/null 2>&1 && ! command -v python3.10 >/dev/null 2>&1; then
    echo "Need Python 3.10+ . Install via: brew install python@3.11"
    exit 1
fi

PY="$(command -v python3.11 || command -v python3.10)"
echo "Using $PY"

if [ ! -d .venv ]; then
    "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "OK. Activate with: source .venv/bin/activate"
echo "Quick sanity check: python -m sim.envs.triple_pendulum_env"

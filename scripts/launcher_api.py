"""
HTTP launcher service for the triple-pendulum training pipeline.
Runs on CT 1018 (training machine). n8n POSTs to /launch to start the
next training stage without needing SSH access.

Setup:
    export LAUNCHER_SECRET=<same secret as in n8n config>
    python scripts/launcher_api.py

Or as a systemd service — see docs/launcher_api.service.

Endpoints:
    GET  /status  — list active tmux training sessions
    POST /launch  — start a new training session
      body: {"secret": "...", "module": "training.train_m3_all_eps",
             "config": "training/configs/m3c_all_eps_tqc.yaml"}
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

REPO = os.environ.get("TRIPLE_PENDULUM_REPO", "/opt/triple-pendulum/repo")
VENV = os.environ.get("TRIPLE_PENDULUM_VENV", "/opt/triple-pendulum/.venv")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://10.1.4.230:5000")
N8N_WEBHOOK = os.environ.get("N8N_PIPELINE_WEBHOOK", "")
N8N_SECRET = os.environ.get("N8N_PIPELINE_SECRET", "")
SECRET = os.environ.get("LAUNCHER_SECRET", "")
PORT = int(os.environ.get("LAUNCHER_PORT", "8765"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("launcher")

_PLACEHOLDER = "YOUR_LAUNCHER_SECRET"

if not SECRET or SECRET == _PLACEHOLDER:
    log.error("LAUNCHER_SECRET is not set or is a placeholder — refusing to start.")
    sys.exit(1)

ALLOWED_MODULES = {
    "training.train_m2_upright",
    "training.train_m3_all_eps",
    "training.train_m4_transitions",
}


def _active_train_sessions() -> list[str]:
    """Detect any running training process, regardless of tmux session naming.
    Looks for python processes running training.train_m* modules."""
    result = subprocess.run(
        ["pgrep", "-af", r"python.*-m training\.train_m"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.strip().splitlines() if line]


def _valid_config(config: str) -> bool:
    if not config or not config.endswith(".yaml"):
        return False
    try:
        resolved = Path(REPO, config).resolve()
        return resolved.is_relative_to(Path(REPO).resolve())
    except Exception:
        return False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)

    def _json(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw)

    def do_GET(self):
        if self.path != "/status":
            self._json(404, {"error": "not found"})
            return
        sessions = _active_train_sessions()
        self._json(200, {"sessions": sessions, "count": len(sessions)})

    def do_POST(self):
        if self.path != "/launch":
            self._json(404, {"error": "not found"})
            return
        try:
            data = self._read_body()
        except Exception:
            self._json(400, {"error": "invalid JSON"})
            return

        incoming = data.get("secret", "")
        if not hmac.compare_digest(incoming, SECRET):
            log.warning("rejected launch: wrong secret from %s", self.address_string())
            self._json(403, {"error": "forbidden"})
            return

        module = data.get("module", "")
        config = data.get("config", "")

        if module not in ALLOWED_MODULES:
            self._json(400, {"error": f"module '{module}' not in allowed list"})
            return
        if not _valid_config(config):
            self._json(400, {"error": "invalid config path"})
            return

        active = _active_train_sessions()
        if active:
            log.warning("launch rejected: session already running: %s", active[0])
            self._json(409, {"error": "a training session is already running", "sessions": active})
            return

        session = f"train_{int(time.time())}"
        log_file = f"/tmp/{session}.log"
        cmd = " && ".join([
            f"cd {REPO}",
            f"source {VENV}/bin/activate",
            (
                f"MUJOCO_GL=osmesa "
                f"MLFLOW_TRACKING_URI={MLFLOW_URI} "
                f"N8N_PIPELINE_WEBHOOK={N8N_WEBHOOK} "
                f"N8N_PIPELINE_SECRET={N8N_SECRET} "
                f"LAUNCHER_SECRET={SECRET} "
                f"PYTHONUNBUFFERED=1 "
                f"python -m {module} --config {config} "
                f"2>&1 | tee {log_file}"
            ),
        ]) + f"; echo TERMINAL:FINISHED >> {log_file}"

        subprocess.Popen(["tmux", "new-session", "-d", "-s", session, cmd])
        log.info("launched session=%s module=%s config=%s", session, module, config)
        self._json(200, {
            "ok": True,
            "session": session,
            "log": log_file,
            "module": module,
            "config": config,
        })


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main() -> None:
    log.info("Launcher API listening on 0.0.0.0:%d", PORT)
    log.info("REPO=%s  VENV=%s  MLFLOW=%s", REPO, VENV, MLFLOW_URI)
    ThreadedHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

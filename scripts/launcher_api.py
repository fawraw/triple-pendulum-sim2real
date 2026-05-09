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

import json
import logging
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

REPO = os.environ.get("TRIPLE_PENDULUM_REPO", "/opt/triple-pendulum/repo")
VENV = os.environ.get("TRIPLE_PENDULUM_VENV", "/opt/triple-pendulum/.venv")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://10.1.4.230:5000")
N8N_WEBHOOK = os.environ.get("N8N_PIPELINE_WEBHOOK", "")
N8N_SECRET = os.environ.get("N8N_PIPELINE_SECRET", "")
SECRET = os.environ.get("LAUNCHER_SECRET", "YOUR_LAUNCHER_SECRET")
PORT = int(os.environ.get("LAUNCHER_PORT", "8765"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("launcher")

ALLOWED_MODULES = {
    "training.train_m2_upright",
    "training.train_m3_all_eps",
    "training.train_m4_transitions",
}


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
        result = subprocess.run(["tmux", "list-sessions"], capture_output=True, text=True)
        sessions = [s for s in result.stdout.strip().splitlines() if "train_" in s]
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

        if data.get("secret") != SECRET:
            log.warning("rejected launch: wrong secret from %s", self.address_string())
            self._json(403, {"error": "forbidden"})
            return

        module = data.get("module", "")
        config = data.get("config", "")

        if module not in ALLOWED_MODULES:
            self._json(400, {"error": f"module '{module}' not in allowed list"})
            return
        if not config or ".." in config or config.startswith("/"):
            self._json(400, {"error": "invalid config path"})
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


def main() -> None:
    log.info("Launcher API listening on 0.0.0.0:%d", PORT)
    log.info("REPO=%s  VENV=%s  MLFLOW=%s", REPO, VENV, MLFLOW_URI)
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

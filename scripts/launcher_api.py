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
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

# Mutex protecting the active-session check + tmux spawn from TOCTOU races
# (two near-simultaneous webhooks from n8n retry, etc.).
_LAUNCH_LOCK = threading.Lock()
LOG_DIR = Path(os.environ.get("TP_LAUNCHER_LOG_DIR", "/var/log/tp-launcher"))

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
    """Detect any running training process. Match python invocations of
    training.train_m{module} with --config — tighter than 'train_m' so that
    importing the module from a REPL or evaluating a checkpoint does not
    register as 'training in progress'."""
    try:
        result = subprocess.run(
            ["pgrep", "-af", r"python.*-m\s+training\.train_m\w+\s+--config"],
            capture_output=True, text=True, timeout=5,
        )
    except subprocess.TimeoutExpired:
        log.warning("pgrep timed out — assuming no active sessions")
        return []
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

    def _kill_active_session(self):
        """Find any tmux session running training and kill it."""
        # Find pids of training processes
        result = subprocess.run(
            ["pgrep", "-af", r"python.*-m\s+training\.train_m\w+\s+--config"],
            capture_output=True, text=True, timeout=5,
        )
        pids = []
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                if line:
                    pids.append(line.split()[0])

        # Kill any tmux session whose name matches train_*
        tmux_result = subprocess.run(
            ["tmux", "list-sessions"], capture_output=True, text=True, timeout=5,
        )
        killed_sessions = []
        if tmux_result.returncode == 0:
            for line in tmux_result.stdout.strip().splitlines():
                # tmux output format: "session_name: 1 windows ..."
                if ":" in line:
                    name = line.split(":", 1)[0]
                    if name.startswith("train_") or name in {"m3b", "m2", "m3c", "m4"}:
                        subprocess.run(["tmux", "kill-session", "-t", name],
                                       capture_output=True, timeout=5)
                        killed_sessions.append(name)

        # Hard-kill any leftover python training processes
        killed_pids = []
        for pid in pids:
            try:
                subprocess.run(["kill", "-TERM", pid], capture_output=True, timeout=2)
                killed_pids.append(pid)
            except Exception:
                pass

        return {"killed_sessions": killed_sessions, "killed_pids": killed_pids}

    def do_POST(self):
        if self.path == "/kill":
            try:
                data = self._read_body()
            except Exception:
                self._json(400, {"error": "invalid JSON"})
                return
            incoming = data.get("secret", "")
            if not hmac.compare_digest(incoming, SECRET):
                log.warning("rejected kill: wrong secret from %s", self.address_string())
                self._json(403, {"error": "forbidden"})
                return
            try:
                result = self._kill_active_session()
                log.info("kill result: %s", result)
                self._json(200, {"ok": True, **result})
            except Exception as exc:
                log.error("kill failed: %s", exc)
                self._json(500, {"error": str(exc)})
            return

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
        # Surface "config file missing" cleanly to n8n. Without this, the
        # launch succeeds, the python process inside tmux raises FileNotFoundError
        # and dies — but n8n never sees the failure and reports a successful
        # launch.
        if not (Path(REPO) / config).is_file():
            self._json(400, {"error": f"config file does not exist: {config}"})
            return

        # The lock + active check + tmux spawn must be atomic to prevent two
        # concurrent webhook retries from racing past the active-check.
        with _LAUNCH_LOCK:
            active = _active_train_sessions()
            if active:
                log.warning("launch rejected: session already running: %s", active[0])
                self._json(409, {"error": "a training session is already running", "sessions": active})
                return

            session = f"train_{int(time.time())}"
            try:
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                log_file = str(LOG_DIR / f"{session}.log")
            except OSError:
                # Fallback: keep going with /tmp if the log dir cannot be created
                # — better than refusing to launch a multi-day training.
                log_file = f"/tmp/{session}.log"
                log.warning("LOG_DIR unwritable, falling back to %s", log_file)

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

            # Verify tmux session was actually created. Without this, a missing
            # tmux binary or a quoting bug means n8n gets 200 OK + nothing runs.
            time.sleep(2)
            check = subprocess.run(
                ["tmux", "has-session", "-t", session],
                capture_output=True, text=True, timeout=5,
            )
            if check.returncode != 0:
                log.error("tmux has-session check failed for %s: %s", session, check.stderr)
                self._json(500, {"error": "tmux session did not start", "session": session})
                return

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

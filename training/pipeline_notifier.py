"""
Notifier called at the end of every training script.
Writes a JSON result file for persistent storage and POSTs to the n8n
orchestrator webhook so the next pipeline stage can be launched automatically.

Usage (at the end of a training script):
    from training.pipeline_notifier import notify
    notify(stage="M3b", run_name=run_name, run_id=run_id,
           metrics=per_ep, config=cfg_path)

Environment variables:
    N8N_PIPELINE_WEBHOOK  : n8n webhook URL (required for orchestration)
    N8N_PIPELINE_SECRET   : shared secret included in the POST body
    TELEGRAM_FALLBACK_BOT_TOKEN  : if set, fallback notification path
                                   when the n8n webhook fails after retries
    TELEGRAM_FALLBACK_CHAT_ID    : chat id for the fallback notification
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import mlflow as _mlflow
except ImportError:
    _mlflow = None

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

WEBHOOK_RETRY_DELAYS = (1, 4, 16)  # seconds — exponential-ish backoff


def _safe_mlflow_tag(key: str, value: str) -> None:
    """Tag the active MLflow run, swallowing any error — late-run network
    blips on set_tag must not abort an 8h training."""
    if _mlflow is None:
        return
    try:
        _mlflow.set_tag(key, value[:500] if isinstance(value, str) else value)
    except Exception:
        pass


def _writable_results_dir() -> Path:
    """Return a writable directory for result snapshots. Falls back to a
    sub-folder under the system tempdir if RESULTS_DIR cannot be created
    (read-only fs, permission denied) — better than crashing AFTER 5M steps
    of compute."""
    try:
        RESULTS_DIR.mkdir(exist_ok=True)
        probe = RESULTS_DIR / ".write_probe"
        probe.touch()
        probe.unlink()
        return RESULTS_DIR
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / "triple_pendulum_results"
        fallback.mkdir(exist_ok=True)
        print(f"[pipeline] WARN results dir unavailable ({exc}); using {fallback}", flush=True)
        _safe_mlflow_tag("pipeline_results_dir_fallback", str(fallback))
        return fallback


def _telegram_fallback(stage: str, run_name: str, run_id: str, metrics: dict, error: str) -> None:
    """Last-resort notification when n8n is unreachable. Posts a short
    summary directly to Telegram so the operator finds out within seconds
    instead of hours later."""
    token = os.environ.get("TELEGRAM_FALLBACK_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_FALLBACK_CHAT_ID", "")
    if not token or not chat_id:
        print("[pipeline] no Telegram fallback configured (TELEGRAM_FALLBACK_BOT_TOKEN/CHAT_ID)")
        return

    overall = metrics.get("overall_success_rate", "n/a")
    text = (
        f"⚠️ Triple Pendulum {stage} finished but n8n webhook failed.\n"
        f"run: {run_name}\n"
        f"run_id: {run_id}\n"
        f"overall_success_rate: {overall}\n"
        f"webhook error: {error[:200]}"
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=10) as resp:
            print(f"[pipeline] telegram fallback: HTTP {resp.status}")
    except Exception as exc:
        print(f"[pipeline] telegram fallback failed: {exc}")


def notify(
    stage: str,
    run_name: str,
    run_id: str,
    metrics: dict,
    config: str,
    *,
    webhook_url: str | None = None,
    secret: str | None = None,
) -> None:
    """Write a JSON result snapshot and trigger the n8n webhook with
    retry + Telegram fallback. Never raises; always returns."""
    results_dir = _writable_results_dir()

    token = secret or os.environ.get("N8N_PIPELINE_SECRET", "")
    payload = {
        "milestone": stage,
        "run_name": run_name,
        "run_id": run_id,
        "config": str(config),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
        "pipeline_secret": token,
    }

    result_path = results_dir / f"{run_name}.json"
    disk_payload = {k: v for k, v in payload.items() if k != "pipeline_secret"}
    try:
        result_path.write_text(json.dumps(disk_payload, indent=2))
        print(f"[pipeline] results saved: {result_path}")
    except OSError as exc:
        print(f"[pipeline] WARN failed to write results snapshot: {exc}", flush=True)
        _safe_mlflow_tag("pipeline_results_write_error", str(exc))

    url = webhook_url or os.environ.get("N8N_PIPELINE_WEBHOOK", "")
    if not url:
        print("[pipeline] N8N_PIPELINE_WEBHOOK not set — skipping orchestrator call.")
        return

    body = json.dumps(payload).encode()
    last_error = ""
    for attempt, delay in enumerate(WEBHOOK_RETRY_DELAYS, start=1):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                print(f"[pipeline] n8n webhook: HTTP {resp.status} (attempt {attempt})")
                _safe_mlflow_tag("pipeline_notify_status", f"ok_attempt_{attempt}")
                return
        except Exception as exc:
            last_error = str(exc)
            print(f"[pipeline] n8n webhook attempt {attempt} failed: {last_error}", flush=True)
            if attempt < len(WEBHOOK_RETRY_DELAYS):
                time.sleep(delay)

    # All retries exhausted — record the failure prominently and try Telegram.
    print(f"[pipeline] n8n webhook FAILED after {len(WEBHOOK_RETRY_DELAYS)} attempts.", flush=True)
    _safe_mlflow_tag("pipeline_notify_error", last_error[:500])
    _safe_mlflow_tag("pipeline_notify_status", "failed")
    _telegram_fallback(stage, run_name, run_id, metrics, last_error)

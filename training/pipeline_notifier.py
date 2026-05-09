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
    N8N_PIPELINE_SECRET   : shared secret sent in X-Pipeline-Secret header
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

try:
    import mlflow as _mlflow
except ImportError:
    _mlflow = None

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


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
    """Write a JSON result snapshot and optionally trigger the n8n webhook."""
    RESULTS_DIR.mkdir(exist_ok=True)

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

    result_path = RESULTS_DIR / f"{run_name}.json"
    disk_payload = {k: v for k, v in payload.items() if k != "pipeline_secret"}
    result_path.write_text(json.dumps(disk_payload, indent=2))
    print(f"[pipeline] results saved: {result_path}")

    url = webhook_url or os.environ.get("N8N_PIPELINE_WEBHOOK", "")
    if not url:
        print("[pipeline] N8N_PIPELINE_WEBHOOK not set — skipping orchestrator call.")
        return

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Pipeline-Secret": token,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"[pipeline] n8n webhook: HTTP {resp.status}")
    except Exception as exc:
        msg = str(exc)
        print(f"[pipeline] n8n webhook failed (non-fatal): {msg}")
        if _mlflow is not None:
            try:
                _mlflow.set_tag("pipeline_notify_error", msg[:500])
            except Exception:
                pass

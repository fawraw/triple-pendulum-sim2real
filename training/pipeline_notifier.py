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

    payload = {
        "milestone": stage,
        "run_name": run_name,
        "run_id": run_id,
        "config": str(config),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
    }

    result_path = RESULTS_DIR / f"{run_name}.json"
    result_path.write_text(json.dumps(payload, indent=2))
    print(f"[pipeline] results saved: {result_path}")

    url = webhook_url or os.environ.get("N8N_PIPELINE_WEBHOOK", "")
    if not url:
        print("[pipeline] N8N_PIPELINE_WEBHOOK not set — skipping orchestrator call.")
        return

    token = secret or os.environ.get("N8N_PIPELINE_SECRET", "")
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
        print(f"[pipeline] n8n webhook failed (non-fatal): {exc}")

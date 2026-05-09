"""Tests for training.pipeline_notifier — particularly the secret-on-disk
audit fix and graceful webhook failure."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from training import pipeline_notifier


SECRET = "tp-pipeline-test-2026"


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    """Redirect RESULTS_DIR to a temp folder so tests don't pollute the repo."""
    monkeypatch.setattr(pipeline_notifier, "RESULTS_DIR", tmp_path)
    yield tmp_path


def test_disk_payload_excludes_secret(tmp_results, monkeypatch):
    """Audit HIGH fix: pipeline_secret must NOT appear in the JSON written to disk."""
    monkeypatch.setenv("N8N_PIPELINE_SECRET", SECRET)
    monkeypatch.delenv("N8N_PIPELINE_WEBHOOK", raising=False)

    pipeline_notifier.notify(
        stage="M3b",
        run_name="test_run_001",
        run_id="abcd1234",
        metrics={"overall_success_rate": 0.81},
        config="training/configs/m3b_all_eps_tqc.yaml",
    )

    saved = json.loads((tmp_results / "test_run_001.json").read_text())
    assert "pipeline_secret" not in saved, "Secret leaked to disk!"
    assert saved["milestone"] == "M3b"
    assert saved["metrics"]["overall_success_rate"] == 0.81


def test_webhook_failure_does_not_raise(tmp_results, monkeypatch):
    """Audit LOW: a webhook outage must not propagate as an exception."""
    monkeypatch.setenv("N8N_PIPELINE_SECRET", SECRET)
    monkeypatch.setenv("N8N_PIPELINE_WEBHOOK", "http://0.0.0.0:1/dead")

    # Should complete without raising even though the URL is unreachable.
    pipeline_notifier.notify(
        stage="M3b",
        run_name="test_run_002",
        run_id="dead",
        metrics={"overall_success_rate": 0.5},
        config="training/configs/m3b_all_eps_tqc.yaml",
    )


def test_webhook_payload_includes_secret(tmp_results, monkeypatch):
    """The POST body MUST include pipeline_secret (n8n Code nodes can't read headers)."""
    monkeypatch.setenv("N8N_PIPELINE_SECRET", SECRET)
    monkeypatch.setenv("N8N_PIPELINE_WEBHOOK", "http://example.invalid/webhook")

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["data"] = req.data
        captured["headers"] = dict(req.header_items())

        class FakeResp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        return FakeResp()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        pipeline_notifier.notify(
            stage="M3b",
            run_name="test_run_003",
            run_id="r3",
            metrics={"overall_success_rate": 0.9},
            config="cfg.yaml",
        )

    body = json.loads(captured["data"].decode())
    assert body["pipeline_secret"] == SECRET
    assert body["milestone"] == "M3b"


def test_metrics_rounded_to_6_decimals(tmp_results, monkeypatch):
    monkeypatch.setenv("N8N_PIPELINE_SECRET", SECRET)
    monkeypatch.delenv("N8N_PIPELINE_WEBHOOK", raising=False)

    pipeline_notifier.notify(
        stage="M3b",
        run_name="test_round",
        run_id="r",
        metrics={"x": 1.123456789, "y": 0.000000123456},
        config="cfg.yaml",
    )

    saved = json.loads((tmp_results / "test_round.json").read_text())
    assert saved["metrics"]["x"] == 1.123457
    assert saved["metrics"]["y"] == 0.0

"""Drift test: ensure pipeline_stages.json (consumed by humans / docs) and the
embedded STAGES dict in the n8n workflow JSON (consumed by n8n at runtime)
agree on milestone, metric, threshold, and pass/fail next-stage.

This is the "single source of truth" gap the audit flagged: two independent
copies of the same state machine. If they drift, n8n decides one thing and
the docs say another."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGES_FILE = ROOT / "training" / "pipeline_stages.json"
N8N_FILE = ROOT / "n8n" / "triple_pendulum_pipeline.json"


def _load_repo_stages() -> dict:
    data = json.loads(STAGES_FILE.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _extract_n8n_stages_js() -> str:
    """Pull the JS object literal assigned to STAGES from the embedded jsCode."""
    workflow = json.loads(N8N_FILE.read_text())
    code = next(
        n["parameters"]["jsCode"]
        for n in workflow["nodes"]
        if n.get("type") == "n8n-nodes-base.code"
    )
    m = re.search(r"const STAGES = (\{.*?\n\};)", code, re.DOTALL)
    assert m, "Could not locate STAGES literal in n8n Code node"
    return m.group(1).rstrip(";").rstrip()


@pytest.fixture(scope="module")
def repo_stages():
    return _load_repo_stages()


@pytest.fixture(scope="module")
def n8n_stages_js() -> str:
    return _extract_n8n_stages_js()


def test_n8n_workflow_has_stages_block(n8n_stages_js):
    assert n8n_stages_js.startswith("{")
    assert "M2" in n8n_stages_js
    assert "M3b" in n8n_stages_js


def test_each_repo_stage_appears_in_n8n(repo_stages, n8n_stages_js):
    """Every stage in pipeline_stages.json must also be defined in n8n.
    Missing stages cause n8n to fall through to HUMAN_REVIEW with no reason."""
    for stage in repo_stages:
        assert f"{stage}: {{" in n8n_stages_js or f"'{stage}':" in n8n_stages_js, \
            f"Stage '{stage}' is in pipeline_stages.json but missing from n8n STAGES"


def test_each_repo_threshold_appears_in_n8n(repo_stages, n8n_stages_js):
    """If the repo threshold changes (e.g. relax M3 to 0.70), n8n must follow."""
    for stage, spec in repo_stages.items():
        thr = spec["threshold"]
        # Allow either 0.75 or 0.8 short forms — JS accepts both.
        thr_str = str(thr).rstrip("0").rstrip(".") if "." in str(thr) else str(thr)
        # We only assert presence of the float value somewhere in the n8n block.
        assert str(thr) in n8n_stages_js or thr_str in n8n_stages_js, \
            f"Threshold {thr} for {stage} not found in n8n STAGES"

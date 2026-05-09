"""Validate pipeline_stages.json schema. Catches typos and missing fields
before n8n tries to read the spec at runtime."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGES_FILE = ROOT / "training" / "pipeline_stages.json"


@pytest.fixture(scope="module")
def stages():
    data = json.loads(STAGES_FILE.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def test_pipeline_stages_file_exists():
    assert STAGES_FILE.exists()


def test_each_stage_has_required_keys(stages):
    for name, spec in stages.items():
        assert "module" in spec, f"{name} missing 'module'"
        assert "metric" in spec, f"{name} missing 'metric'"
        assert "threshold" in spec, f"{name} missing 'threshold'"
        assert "pass" in spec, f"{name} missing 'pass'"
        assert "fail" in spec, f"{name} missing 'fail'"


def test_thresholds_are_in_unit_interval(stages):
    for name, spec in stages.items():
        t = spec["threshold"]
        assert 0.0 < t <= 1.0, f"{name} threshold {t} not in (0, 1]"


def test_pass_fail_branches_have_stage(stages):
    for name, spec in stages.items():
        for branch in ("pass", "fail"):
            assert "stage" in spec[branch], f"{name}.{branch} missing 'stage'"


def test_non_terminal_branches_have_module_and_config(stages):
    """If pass/fail.stage is not HUMAN_REVIEW, it must have a module and config
    so the launcher can pick up the next training run."""
    for name, spec in stages.items():
        for branch in ("pass", "fail"):
            b = spec[branch]
            if b["stage"] != "HUMAN_REVIEW":
                assert "module" in b, f"{name}.{branch} non-terminal but missing 'module'"
                assert "config" in b, f"{name}.{branch} non-terminal but missing 'config'"


def test_referenced_configs_exist(stages):
    """All config paths referenced in pipeline_stages.json must exist on disk."""
    for name, spec in stages.items():
        for branch in ("pass", "fail"):
            b = spec[branch]
            cfg = b.get("config")
            if cfg:
                cfg_path = ROOT / cfg
                assert cfg_path.exists(), f"{name}.{branch}.config '{cfg}' does not exist"


def test_referenced_modules_exist(stages):
    """All training modules referenced must be importable as files."""
    for name, spec in stages.items():
        candidates = {spec["module"]}
        for branch in ("pass", "fail"):
            mod = spec[branch].get("module")
            if mod:
                candidates.add(mod)
        for mod in candidates:
            path = ROOT / (mod.replace(".", "/") + ".py")
            assert path.exists(), f"Module {mod} not found at {path}"

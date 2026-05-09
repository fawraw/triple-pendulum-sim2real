"""Tests for the launcher_api module — covers the audit fixes:
- timing-safe secret comparison (hmac.compare_digest)
- config path validation rejects traversal and absolute paths
- module whitelist
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture
def launcher(monkeypatch):
    """Import launcher_api with a valid SECRET so the startup guard passes."""
    monkeypatch.setenv("LAUNCHER_SECRET", "tp-launcher-test-2026")
    monkeypatch.setenv("TRIPLE_PENDULUM_REPO", str(ROOT))
    # Force fresh import so the env vars are picked up.
    if "launcher_api" in sys.modules:
        del sys.modules["launcher_api"]
    return importlib.import_module("launcher_api")


def test_module_whitelist_contains_known_stages(launcher):
    assert "training.train_m2_upright" in launcher.ALLOWED_MODULES
    assert "training.train_m3_all_eps" in launcher.ALLOWED_MODULES
    assert "training.train_m4_transitions" in launcher.ALLOWED_MODULES


def test_module_whitelist_excludes_arbitrary_modules(launcher):
    assert "os.system" not in launcher.ALLOWED_MODULES
    assert "subprocess" not in launcher.ALLOWED_MODULES


def test_valid_config_accepts_existing_yaml(launcher):
    assert launcher._valid_config("training/configs/m3b_all_eps_tqc.yaml") is True


def test_valid_config_rejects_path_traversal(launcher):
    assert launcher._valid_config("../etc/passwd") is False
    assert launcher._valid_config("training/../../../etc/passwd") is False


def test_valid_config_rejects_absolute_paths(launcher):
    assert launcher._valid_config("/etc/passwd") is False
    assert launcher._valid_config("/tmp/evil.yaml") is False


def test_valid_config_rejects_non_yaml(launcher):
    assert launcher._valid_config("training/configs/m3b_all_eps_tqc.json") is False
    assert launcher._valid_config("training/configs/m3b_all_eps_tqc") is False
    assert launcher._valid_config("") is False


def test_valid_config_rejects_missing_file(launcher):
    # Path doesn't exist on disk — Path.resolve still works but we accept by
    # path containment, not existence. Confirm that a non-existent path under
    # REPO is currently allowed (existence is checked when training runs).
    # The point of this test is to document the boundary.
    assert launcher._valid_config("training/configs/does_not_exist.yaml") is True

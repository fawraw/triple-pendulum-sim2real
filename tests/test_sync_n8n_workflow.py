"""Tests for scripts/sync_n8n_workflow.py placeholder substitution."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "sync_n8n_workflow", ROOT / "scripts" / "sync_n8n_workflow.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_find_placeholders():
    text = 'const S="YOUR_PIPELINE_SECRET"; const T="YOUR_TELEGRAM_BOT_TOKEN";'
    assert mod.find_placeholders(text) == {"PIPELINE_SECRET", "TELEGRAM_BOT_TOKEN"}


def test_substitute_fills_known_and_reports_missing():
    text = 'a=YOUR_PIPELINE_SECRET b=YOUR_LAUNCHER_SECRET c=YOUR_RUNPOD_API_KEY'
    out, missing = mod.substitute(text, {"PIPELINE_SECRET": "p1", "LAUNCHER_SECRET": "l1"})
    assert "a=p1" in out and "b=l1" in out
    # unknown placeholder is left intact and reported
    assert "YOUR_RUNPOD_API_KEY" in out
    assert missing == ["RUNPOD_API_KEY"]


def test_substitute_empty_value_counts_as_missing():
    out, missing = mod.substitute("x=YOUR_TOKEN", {"TOKEN": ""})
    assert out == "x=YOUR_TOKEN"
    assert missing == ["TOKEN"]


def test_substitute_no_placeholders_is_noop():
    out, missing = mod.substitute('{"nodes": []}', {})
    assert out == '{"nodes": []}'
    assert missing == []

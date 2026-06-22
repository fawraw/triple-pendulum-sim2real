"""Tests for training.eval_utils: the single success-criterion helper."""
import pytest

from training.eval_utils import success_rate, success_threshold


def test_success_threshold_scales_with_budget():
    assert success_threshold(1000, 0.8) == 800
    assert success_threshold(2000, 0.8) == 1600
    assert success_threshold(1000, 0.5) == 500


def test_success_rate_zero_when_all_short():
    assert success_rate([100, 200, 799], max_steps=1000, frac=0.8) == 0.0


def test_success_rate_full_when_all_held():
    assert success_rate([800, 900, 1000], max_steps=1000, frac=0.8) == 1.0


def test_success_rate_partial():
    assert success_rate([100, 800, 200, 900], max_steps=1000, frac=0.8) == 0.5


def test_success_rate_empty_is_zero():
    assert success_rate([], max_steps=1000) == 0.0


def test_success_rate_respects_non_1000_budget():
    # The old hardcoded 800 silently broke for non-1000 budgets; the helper
    # tracks the actual budget. At max=2000 the threshold is 1600, so 1500
    # is a failure -- whereas the old `>= 800` would have called it a success.
    assert success_rate([1500, 1700], max_steps=2000, frac=0.8) == 0.5
    assert success_rate([1500, 1700], max_steps=1000, frac=0.8) == 1.0
    assert success_rate([700], max_steps=2000, frac=0.8) == 0.0

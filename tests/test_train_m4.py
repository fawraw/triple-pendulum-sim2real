"""Tests for training.train_m4_transitions: cold-start guard, success metric,
and ALL_TRANSITIONS coverage."""
import pytest

from training import train_m4_transitions as m4


def test_all_transitions_has_56_pairs():
    assert len(m4.ALL_TRANSITIONS) == 56


def test_all_transitions_excludes_self_pairs():
    for src, dst in m4.ALL_TRANSITIONS:
        assert src != dst, f"self-pair found: ({src}, {dst})"


def test_all_transitions_covers_all_pairs():
    pairs = set(m4.ALL_TRANSITIONS)
    for src in range(8):
        for dst in range(8):
            if src != dst:
                assert (src, dst) in pairs, f"missing transition ({src}, {dst})"


# ---------------------------------------------------------------------------
# _transition_success
# ---------------------------------------------------------------------------

def test_transition_success_zero_when_short():
    # max=1000, lengths all under 500 → 0% success
    assert m4._transition_success([100, 200, 499], max_steps=1000) == 0.0


def test_transition_success_full_when_held():
    # All episodes hit at least 0.5 * max_steps
    assert m4._transition_success([500, 800, 1000], max_steps=1000) == 1.0


def test_transition_success_partial():
    # 2 out of 4 hit threshold
    sr = m4._transition_success([100, 600, 200, 700], max_steps=1000)
    assert sr == 0.5


# ---------------------------------------------------------------------------
# Cold-start guard
# ---------------------------------------------------------------------------

VALID_CFG = {
    "env": {"max_episode_steps": 1000},
    "tqc": {"policy": "MlpPolicy"},
    "total_timesteps": 1_000_000,
}


def test_cold_start_refused_without_pretrained():
    cfg = dict(VALID_CFG)
    cfg["pretrained_policy"] = None
    with pytest.raises(ValueError, match="cold-start"):
        m4._validate_cfg(cfg)


def test_cold_start_refused_when_field_missing():
    cfg = dict(VALID_CFG)  # no pretrained_policy at all
    with pytest.raises(ValueError, match="cold-start"):
        m4._validate_cfg(cfg)


def test_cold_start_allowed_when_explicit_opt_in():
    cfg = dict(VALID_CFG)
    cfg["pretrained_policy"] = None
    cfg["allow_cold_start"] = True
    m4._validate_cfg(cfg)  # no raise


def test_pretrained_validated_to_exist(tmp_path):
    cfg = dict(VALID_CFG)
    cfg["pretrained_policy"] = "checkpoints/does_not_exist/final.zip"
    with pytest.raises(ValueError, match="not found"):
        m4._validate_cfg(cfg)


def test_pretrained_absolute_path_skipped(tmp_path):
    """Absolute paths bypass the relative-to-ROOT existence check (operator
    is responsible for absolute paths)."""
    cfg = dict(VALID_CFG)
    cfg["pretrained_policy"] = "/nonexistent/absolute/path.zip"
    # No raise — absolute paths are not validated against ROOT.
    m4._validate_cfg(cfg)

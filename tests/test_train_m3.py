"""Tests for training.train_m3_all_eps config validation."""
import pytest

from training import train_m3_all_eps as m3


def _cfg(target_mode):
    return {
        "env": {"max_episode_steps": 1000, "target_mode": target_mode},
        "tqc": {"policy": "MlpPolicy"},
        "total_timesteps": 1000,
    }


@pytest.mark.parametrize("mode", ["fixed", "random", "weighted", "transition"])
def test_validate_accepts_all_supported_modes(mode):
    m3._validate_cfg_m3(_cfg(mode))  # no raise


def test_validate_rejects_unknown_mode_and_lists_transition():
    # The error message must enumerate all valid modes (the old message
    # omitted 'transition', which was confusing at debug time).
    with pytest.raises(ValueError, match="transition"):
        m3._validate_cfg_m3(_cfg("bogus"))


def test_validate_rejects_nonpositive_timesteps():
    cfg = _cfg("random")
    cfg["total_timesteps"] = 0
    with pytest.raises(ValueError, match="positive"):
        m3._validate_cfg_m3(cfg)

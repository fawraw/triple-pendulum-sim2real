"""Tests for the TriplePendulumEnv Gymnasium environment."""
import numpy as np
import pytest

from sim.envs.triple_pendulum_env import TriplePendulumEnv, ep_target_angles


@pytest.fixture
def env():
    e = TriplePendulumEnv(target_ep=7, init_mode="near_target",
                          init_noise=0.05, max_episode_steps=50)
    yield e
    e.close()


def test_obs_shape_is_16(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == (16,), f"expected (16,), got {obs.shape}"
    assert obs.dtype == np.float32


def test_obs_shape_matches_observation_space(env):
    """Catches the kind of bug where the env emits N dims but observation_space
    is declared with M (M3/M4 docstrings vs implementation)."""
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, (
        f"obs shape {obs.shape} != observation_space.shape {env.observation_space.shape}"
    )


def test_action_space_is_1d_continuous(env):
    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0


def test_one_hot_target_encoded_in_obs(env):
    obs, _ = env.reset(seed=0)
    one_hot = obs[8:16]
    assert one_hot.sum() == 1.0
    assert one_hot[7] == 1.0  # target_ep=7


def test_step_returns_5tuple(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step([0.0])
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "target_ep" in info


def test_truncation_at_max_steps():
    e = TriplePendulumEnv(target_ep=0, init_mode="near_target",
                          init_noise=0.0, max_episode_steps=5)
    e.reset(seed=0)
    for _ in range(4):
        _, _, _, trunc, _ = e.step([0.0])
        assert not trunc
    _, _, _, trunc, _ = e.step([0.0])
    assert trunc
    e.close()


def test_random_target_mode_resamples_on_reset():
    e = TriplePendulumEnv(target_ep=0, target_mode="random",
                          init_mode="near_target", init_noise=0.05,
                          max_episode_steps=50)
    targets = set()
    for seed in range(30):
        e.reset(seed=seed)
        targets.add(e.target_ep)
    e.close()
    # With 30 seeds and uniform sampling over 8 EPs, we should hit several distinct targets.
    assert len(targets) >= 4, f"target_mode='random' should resample, got only {targets}"


def test_fixed_target_mode_keeps_target():
    e = TriplePendulumEnv(target_ep=3, target_mode="fixed",
                          init_mode="near_target", max_episode_steps=50)
    for seed in range(5):
        e.reset(seed=seed)
        assert e.target_ep == 3
    e.close()


def test_reset_options_overrides_target():
    e = TriplePendulumEnv(target_ep=0, target_mode="fixed",
                          init_mode="near_target", max_episode_steps=50)
    e.reset(seed=0, options={"target_ep": 5})
    assert e.target_ep == 5
    e.close()


# ---------------------------------------------------------------------------
# Equilibrium encoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ep,bits", [
    (0, [np.pi, np.pi, np.pi]),    # DDD
    (1, [0.0, np.pi, np.pi]),       # DDU (link 1 up)
    (2, [np.pi, 0.0, np.pi]),       # DUD
    (4, [np.pi, np.pi, 0.0]),       # UDD (link 3 up)
    (7, [0.0, 0.0, 0.0]),           # UUU
])
def test_ep_target_angles_bit_encoding(ep, bits):
    angles = ep_target_angles(ep)
    np.testing.assert_array_almost_equal(angles, bits)


def test_all_8_eps_distinct():
    encodings = [tuple(ep_target_angles(ep)) for ep in range(8)]
    assert len(set(encodings)) == 8, "8 EPs must produce 8 distinct angle tuples"


# ---------------------------------------------------------------------------
# Fall detection
# ---------------------------------------------------------------------------

def test_fall_when_cart_off_rail(env):
    env.reset(seed=0)
    env.data.qpos[0] = 1.5  # well past 0.95 m
    assert env._is_fallen()


def test_fall_when_link_too_far_from_target(env):
    env.reset(seed=0)
    # Force link 1 to point sideways (~1.5 rad off from target = 0.0)
    env.data.qpos[1] = 1.5
    assert env._is_fallen()


def test_no_fall_at_target():
    e = TriplePendulumEnv(target_ep=7, init_mode="near_target",
                          init_noise=0.0, max_episode_steps=50)
    e.reset(seed=0)
    # qpos[0] is the cart, [1..3] are relative hinge angles starting from
    # 0 rad in init (which is target since EP7 is UUU).
    assert not e._is_fallen()
    e.close()

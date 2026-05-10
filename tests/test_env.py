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


# ---------------------------------------------------------------------------
# Per-link fall threshold (audit 2026-05-10 fix)
# ---------------------------------------------------------------------------

def test_per_link_threshold_up_is_tight():
    """Links targeted UP use the tight 0.6 rad threshold."""
    e = TriplePendulumEnv(target_ep=7, init_mode="near_target",
                          init_noise=0.0, max_episode_steps=50)
    e.reset(seed=0)
    thresholds = e._fall_thresholds()
    # EP7 = all UP → all thresholds should be FALL_THRESHOLD_UP_RAD
    import numpy as np
    np.testing.assert_array_almost_equal(
        thresholds, [e.FALL_THRESHOLD_UP_RAD] * 3
    )
    e.close()


def test_per_link_threshold_down_is_loose():
    """Links targeted DOWN use the loose 1.5 rad threshold."""
    e = TriplePendulumEnv(target_ep=0, init_mode="near_target",
                          init_noise=0.0, max_episode_steps=50)
    e.reset(seed=0)
    thresholds = e._fall_thresholds()
    # EP0 = DDD → all DOWN → all should be FALL_THRESHOLD_DOWN_RAD
    import numpy as np
    np.testing.assert_array_almost_equal(
        thresholds, [e.FALL_THRESHOLD_DOWN_RAD] * 3
    )
    e.close()


def test_ep4_hanging_links_do_not_trigger_fall():
    """EP4 (UDD): link 1 targeted UP (strict), links 2-3 targeted DOWN (loose).
    A hanging link swinging 1.0 rad (< 1.5 rad DOWN threshold) must NOT trip.
    This is the pre-fix bug that caused EP4=0% in M3b."""
    import mujoco, numpy as np
    e = TriplePendulumEnv(target_ep=4, init_mode="near_target",
                          init_noise=0.0, max_episode_steps=50)
    e.reset(seed=0)
    # Force link 2 to swing 1.0 rad (well past old 0.6 global threshold,
    # but below the new DOWN threshold of 1.5 rad).
    e.data.qpos[2] = 1.0  # relative angle of hinge2 (link 2)
    mujoco.mj_forward(e.model, e.data)
    assert not e._is_fallen(), (
        "Hanging link 2 at 1.0 rad should NOT trigger fall for EP4 "
        "(target=DOWN, threshold=1.5 rad). Pre-fix bug reproduced."
    )
    e.close()


def test_fall_grace_steps_delays_termination():
    """With fall_grace_steps=5, _is_fallen() returns False for the first 5
    consecutive over-threshold steps and True on the 6th."""
    import mujoco, numpy as np
    e = TriplePendulumEnv(target_ep=7, init_mode="near_target",
                          init_noise=0.0, max_episode_steps=200,
                          fall_grace_steps=5)
    e.reset(seed=0)
    # Force link 1 over the UP threshold (0.6 rad)
    e.data.qpos[1] = 1.0
    mujoco.mj_forward(e.model, e.data)
    for step in range(1, 7):
        result = e._is_fallen()
        if step <= 5:
            assert not result, f"Step {step}: should still be within grace"
        else:
            assert result, f"Step {step}: should have fallen (grace exceeded)"
    e.close()


def test_fall_counter_resets_on_recovery():
    """If a link recovers (goes back under threshold), counter resets and
    subsequent over-threshold steps restart the grace count."""
    import mujoco, numpy as np
    e = TriplePendulumEnv(target_ep=7, init_mode="near_target",
                          init_noise=0.0, max_episode_steps=200,
                          fall_grace_steps=5)
    e.reset(seed=0)
    # Push link 1 over threshold 4 times
    e.data.qpos[1] = 1.0
    mujoco.mj_forward(e.model, e.data)
    for _ in range(4):
        e._is_fallen()
    assert e._fall_counter == 4
    # Recover — bring back under threshold
    e.data.qpos[1] = 0.0
    mujoco.mj_forward(e.model, e.data)
    e._is_fallen()
    assert e._fall_counter == 0, "Counter must reset when under threshold"
    e.close()


def test_reset_clears_fall_counter():
    """env.reset() must zero out _fall_counter."""
    import mujoco, numpy as np
    e = TriplePendulumEnv(target_ep=7, init_mode="near_target",
                          init_noise=0.0, max_episode_steps=50,
                          fall_grace_steps=10)
    e.reset(seed=0)
    e.data.qpos[1] = 1.0
    mujoco.mj_forward(e.model, e.data)
    for _ in range(5):
        e._is_fallen()
    assert e._fall_counter == 5
    e.reset(seed=1)
    assert e._fall_counter == 0, "reset() must clear _fall_counter"
    e.close()


def test_reward_link1_weighted_5x():
    """Link 1 angular error contributes 5× to ang_cost vs links 2 and 3."""
    import mujoco, numpy as np
    # EP7: all targets at 0. Push only link 1 by ε, everything else at 0.
    e7 = TriplePendulumEnv(target_ep=7, init_mode="near_target",
                           init_noise=0.0, max_episode_steps=50)
    e7.reset(seed=0)
    eps = 0.1
    # err for link 1 only = ε, links 2-3 = 0 → ang_cost = 5*eps²
    e7.data.qpos[1] = eps
    e7.data.qpos[2] = 0.0
    e7.data.qpos[3] = 0.0
    mujoco.mj_forward(e7.model, e7.data)
    r1 = e7._reward(fallen=False)

    # Now push only link 2 by ε → ang_cost = 1*eps² (weight=1)
    e7.reset(seed=0)
    e7.data.qpos[1] = 0.0
    e7.data.qpos[2] = eps
    e7.data.qpos[3] = 0.0
    mujoco.mj_forward(e7.model, e7.data)
    r2 = e7._reward(fallen=False)
    e7.close()

    # |r1| should be ~5× |r2| (link 1 weighted 5×)
    # Exact: r1 ≈ -(5*eps² + vel_cost + cart_cost), r2 ≈ -(1*eps² + ...)
    assert abs(r1) > abs(r2) * 4, (
        f"Link 1 reward contribution should be ~5× link 2; got r1={r1:.4f} r2={r2:.4f}"
    )

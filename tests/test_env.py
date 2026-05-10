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
    Link 2 with abs_angle = π-1.0 has |error|=1.0 rad from target π.
    1.0 rad < 1.5 rad DOWN threshold → must NOT trigger fall.
    This is the pre-fix bug (old threshold 0.6 would have triggered here)."""
    import mujoco, numpy as np
    e = TriplePendulumEnv(target_ep=4, init_mode="near_target",
                          init_noise=0.0, max_episode_steps=50)
    e.reset(seed=0)
    # EP4 targets = ep_target_angles(4) = [π, π, 0]:
    #   bit 0 of 4 = 0 → link 1 DOWN → θ₁_target = π
    #   bit 1 of 4 = 0 → link 2 DOWN → θ₂_target = π
    #   bit 2 of 4 = 1 → link 3 UP   → θ₃_target = 0
    # After near_target reset (noise=0): qpos[1]=π, qpos[2]=0, qpos[3]=-π
    # (qpos[i] are RELATIVE angles: qpos[2] = t2-t1 = π-π = 0)
    # abs_θ₂ = qpos[1] + qpos[2] = π + 0 = π → err[1] = 0
    #
    # To give link 2 an error of 1.0 rad: abs_θ₂ = π-1.0
    # → qpos[2] = abs_θ₂ - qpos[1] = (π-1.0) - π = -1.0
    # abs angles: θ = qpos[1], θ₂ = qpos[1]+qpos[2], θ₃ = qpos[1]+qpos[2]+qpos[3]
    # After reset(EP4,noise=0): qpos[1]=π, qpos[2]=0, qpos[3]=-π
    # Change link 2 by 1.0 rad: qpos[2] = -1.0 → θ₂ = π-1.0 → err[1] = -1.0
    # But this also shifts θ₃! Compensate qpos[3] to keep θ₃=0 (UP target):
    # θ₃ = qpos[1]+qpos[2]+qpos[3] = π + (-1.0) + qpos[3] = 0 → qpos[3] = 1.0-π
    e.data.qpos[2] = -1.0         # θ₂ = π-1.0 → err[1] = -1.0  (< 1.5 DOWN threshold)
    e.data.qpos[3] = 1.0 - np.pi  # θ₃ = 0 → err[2] = 0           (< 0.6 UP threshold)
    mujoco.mj_forward(e.model, e.data)
    err = e._angle_error()
    assert abs(abs(err[1]) - 1.0) < 0.05, f"Expected |err[1]|≈1.0, got {err[1]:.3f}"
    assert abs(err[2]) < 0.05,            f"Expected err[2]≈0, got {err[2]:.3f}"
    assert not e._is_fallen(), (
        f"Link 2 at 1.0 rad error from DOWN target should NOT trigger fall "
        f"(DOWN threshold=1.5 rad). err={np.round(err,3)}."
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
    """Link 1 angular error contributes 5× to ang_cost vs links 2 and 3.

    Key: qpos are RELATIVE angles (θᵢ_abs = qpos[1]+...+qpos[i]).
    To isolate link 1 error: qpos[1]=ε, qpos[2]=-ε → θ₁=ε, θ₂=0, θ₃=0.
    To isolate link 2 error: qpos[1]=0, qpos[2]=ε, qpos[3]=-ε → θ₁=0, θ₂=ε, θ₃=0."""
    import mujoco, numpy as np
    e7 = TriplePendulumEnv(target_ep=7, init_mode="near_target",
                           init_noise=0.0, max_episode_steps=50)
    eps = 0.1

    # Case 1: only link 1 has error ε (θ₁=ε, θ₂=0, θ₃=0)
    e7.reset(seed=0)
    e7.data.qpos[1] = eps    # θ₁ = eps
    e7.data.qpos[2] = -eps   # θ₂ = eps + (-eps) = 0
    e7.data.qpos[3] = 0.0    # θ₃ = 0
    mujoco.mj_forward(e7.model, e7.data)
    err1 = e7._angle_error()
    assert abs(err1[0] - eps) < 0.001, f"Expected err[0]≈{eps}, got {err1}"
    assert abs(err1[1]) < 0.001, f"Expected err[1]≈0, got {err1}"
    r1 = e7._reward(fallen=False)

    # Case 2: only link 2 has error ε (θ₁=0, θ₂=ε, θ₃=0)
    e7.reset(seed=0)
    e7.data.qpos[1] = 0.0
    e7.data.qpos[2] = eps    # θ₂ = 0 + eps = eps
    e7.data.qpos[3] = -eps   # θ₃ = 0 + eps + (-eps) = 0
    mujoco.mj_forward(e7.model, e7.data)
    err2 = e7._angle_error()
    assert abs(err2[0]) < 0.001, f"Expected err2[0]≈0, got {err2}"
    assert abs(err2[1] - eps) < 0.001, f"Expected err2[1]≈{eps}, got {err2}"
    r2 = e7._reward(fallen=False)
    e7.close()

    # ang_cost_1 = 5*eps², ang_cost_2 = eps²  → diff = 4*eps²
    # Verify the difference in rewards ≈ 4*eps² (vel/cart/ctrl costs are equal)
    ang_diff = abs(r1) - abs(r2)
    expected_diff = 4 * eps ** 2
    assert ang_diff > expected_diff * 0.8, (
        f"Expected |r1|-|r2| ≈ 4*eps²={expected_diff:.4f}, got {ang_diff:.4f}. "
        f"r1={r1:.4f}, r2={r2:.4f}"
    )

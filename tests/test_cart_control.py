"""Tests for the cart-control knobs (cart_limit, cart_cost_coef) added to make
the M4 swing-up keep the cart on the rail."""
import mujoco
import numpy as np

from sim.envs.triple_pendulum_env import TriplePendulumEnv


def _env_at_uuu(**kw):
    # Start near UUU so the angle-fall check is not triggered; isolates the cart.
    env = TriplePendulumEnv(target_ep=7, target_mode="fixed", init_mode="near_target",
                            init_noise=0.0, max_episode_steps=2000, **kw)
    env.reset(seed=0)
    return env


def test_cart_limit_default_is_095():
    env = _env_at_uuu()
    assert env.cart_limit == 0.95
    env.data.qpos[0] = 0.90
    assert env._is_fallen() is False
    env.data.qpos[0] = 0.96
    assert env._is_fallen() is True


def test_cart_limit_configurable():
    wide = _env_at_uuu(cart_limit=1.20)
    wide.data.qpos[0] = 1.00
    assert wide._is_fallen() is False      # 1.00 < 1.20, still on rail
    tight = _env_at_uuu(cart_limit=0.50)
    tight.data.qpos[0] = 0.60
    assert tight._is_fallen() is True      # 0.60 > 0.50, off rail


def test_cart_cost_coef_default_and_scaling():
    soft = _env_at_uuu(cart_cost_coef=0.1)
    hard = _env_at_uuu(cart_cost_coef=1.0)
    assert soft.cart_cost_coef == 0.1
    for e in (soft, hard):
        e.data.qpos[0] = 1.0
        mujoco.mj_forward(e.model, e.data)
    # Same off-center cart, larger coef -> more negative reward (stronger centering).
    assert hard._reward(False) < soft._reward(False)


def test_cart_barrier_default_off_and_steep_near_rail():
    off = _env_at_uuu(cart_barrier_coef=0.0)
    on = _env_at_uuu(cart_barrier_coef=50.0, cart_limit=1.10)
    assert off.cart_barrier_coef == 0.0
    # Near the centre the barrier is negligible: reward ~ unchanged vs no barrier.
    mid = _env_at_uuu(cart_barrier_coef=50.0, cart_limit=1.10)
    for e in (off, mid):
        e.data.qpos[0] = 0.2
        mujoco.mj_forward(e.model, e.data)
    assert abs(mid._reward(False) - off._reward(False)) < 0.05
    # Near the rail end the barrier dominates -> reward far more negative.
    on.data.qpos[0] = 1.05  # ~95% of the 1.10 limit
    mujoco.mj_forward(on.model, on.data)
    off.data.qpos[0] = 1.05
    mujoco.mj_forward(off.model, off.data)
    assert on._reward(False) < off._reward(False) - 5.0


def test_xml_allows_motion_past_old_limit():
    # The physical rail was widened so cart_limit is the effective bound.
    env = _env_at_uuu(cart_limit=1.4)
    lo, hi = env.model.jnt_range[env.model.joint("slider").id]
    assert hi >= 1.4 and lo <= -1.4

"""Tests for training.env_utils helpers (seeding, vec env factory)."""
import numpy as np

from training.env_utils import make_vec_env, seed_everything


def test_seed_everything_reproducible():
    seed_everything(123)
    a = np.random.rand(4)
    seed_everything(123)
    b = np.random.rand(4)
    np.testing.assert_array_equal(a, b)


def test_seed_everything_differs_by_seed():
    seed_everything(123)
    a = np.random.rand(4)
    seed_everything(124)
    c = np.random.rand(4)
    assert not np.allclose(a, c)


def test_seed_everything_none_is_noop():
    # None must not raise and must return None (preserves non-deterministic
    # behaviour for configs that omit a seed).
    assert seed_everything(None) is None


def test_make_vec_env_dummy_for_single_env():
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = make_vec_env({"max_episode_steps": 200}, n_envs=1)
    try:
        assert isinstance(env, DummyVecEnv)
        assert env.num_envs == 1
    finally:
        env.close()

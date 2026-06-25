"""Vectorized environment helpers.

Centralizes the env-thunk factory so all training scripts (M2/M3/M4) can
spin up parallel envs the same way. SubprocVecEnv requires the thunk to be
picklable — that means the factory must be a module-level callable, not a
closure. functools.partial fills that gap cleanly.

Speedup expectation: with mujoco + TQC, 8 parallel envs typically yield
3-5x wall-clock throughput on a ~16-core CT, more with a GPU (since the
critic forward+backward stops being the per-step bottleneck).
"""
from __future__ import annotations

from functools import partial

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from sim.envs.triple_pendulum_env import TriplePendulumEnv


def seed_everything(seed: int | None) -> int | None:
    """Seed Python/NumPy/Torch RNGs for reproducible training.

    No-op when seed is None (preserves the historical non-deterministic
    behaviour for configs that omit a seed). Returns the seed it applied so
    callers can log it. The model's own RNG (action sampling, env seeding) is
    seeded separately via TQC(seed=...) / model.set_random_seed(...).
    """
    if seed is None:
        return None
    seed = int(seed)
    set_random_seed(seed)
    return seed


def _make_one_env(env_cfg: dict) -> Monitor:
    """Top-level factory so SubprocVecEnv can pickle it."""
    env = TriplePendulumEnv(
        target_ep=int(env_cfg.get("target_ep", 7)),
        target_mode=str(env_cfg.get("target_mode", "fixed")),
        init_mode=str(env_cfg.get("init_mode", "near_target")),
        init_noise=float(env_cfg.get("init_noise", 0.05)),
        max_episode_steps=int(env_cfg["max_episode_steps"]),
        fall_grace_steps=int(env_cfg.get("fall_grace_steps", 0)),
        start_grace_steps=int(env_cfg.get("start_grace_steps", 0)),
        hard_ep_weight=float(env_cfg.get("hard_ep_weight", 1.0)),
        w_down=float(env_cfg.get("w_down", 1.0)),
        progress_reward_coef=float(env_cfg.get("progress_reward_coef", 0.0)),
        vel_cost_coef=float(env_cfg.get("vel_cost_coef", 0.05)),
        cart_cost_coef=float(env_cfg.get("cart_cost_coef", 0.1)),
        cart_barrier_coef=float(env_cfg.get("cart_barrier_coef", 0.0)),
        cart_limit=float(env_cfg.get("cart_limit", 0.95)),
        start_ep=env_cfg.get("start_ep"),  # M4 transition tasks
        transition_success_tol_rad=float(env_cfg.get("transition_success_tol_rad", 0.2)),
        transition_success_steps=int(env_cfg.get("transition_success_steps", 200)),
        transition_bonus=float(env_cfg.get("transition_bonus", 200.0)),
    )
    return Monitor(env)


def make_vec_env(env_cfg: dict, n_envs: int = 1, *, force_dummy: bool = False) -> VecEnv:
    """Build a vec env. Falls back to DummyVecEnv when n_envs <= 1 or when
    SubprocVecEnv is undesired (debugging, macOS fork quirks)."""
    if n_envs <= 1 or force_dummy:
        return DummyVecEnv([partial(_make_one_env, env_cfg)])
    return SubprocVecEnv([partial(_make_one_env, env_cfg) for _ in range(n_envs)])

"""Milestone 3: train a single conditional TQC policy that stabilizes any
of the 8 equilibrium configurations of the triple pendulum.

Differences from M2:
- The env's target_mode is set to "random" so target_ep is resampled uniformly
  on every reset.
- The policy must learn to read the target one-hot in the observation.
- The eval phase rolls out per-EP success metrics.

Run from project root:

    MUJOCO_GL=osmesa python -m training.train_m3_all_eps \\
        --config training/configs/m3_all_eps_tqc.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import yaml
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.envs.triple_pendulum_env import TriplePendulumEnv  # noqa: E402
from training.mlflow_setup import init_mlflow  # noqa: E402
from training.pipeline_notifier import notify as pipeline_notify  # noqa: E402


def make_env(env_cfg: dict):
    def _thunk():
        env = TriplePendulumEnv(
            target_ep=int(env_cfg.get("target_ep", 7)),
            target_mode=str(env_cfg.get("target_mode", "random")),
            init_mode=str(env_cfg.get("init_mode", "near_target")),
            init_noise=float(env_cfg.get("init_noise", 0.05)),
            max_episode_steps=int(env_cfg["max_episode_steps"]),
        )
        return Monitor(env)
    return _thunk


class MLflowRolloutLogger(BaseCallback):
    def __init__(self, log_freq: int = 4000):
        super().__init__()
        self.log_freq = log_freq
        self._last = 0
        self._t0 = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.log_freq:
            return True
        self._last = self.num_timesteps
        sps = self.num_timesteps / max(1e-3, time.time() - self._t0)
        mlflow.log_metric("timesteps", self.num_timesteps, step=self.num_timesteps)
        mlflow.log_metric("steps_per_s", float(sps), step=self.num_timesteps)
        rewards = [ep["r"] for ep in self.model.ep_info_buffer][-50:]
        if rewards:
            mlflow.log_metric("rollout_ep_rew_mean", float(np.mean(rewards)),
                              step=self.num_timesteps)
            mlflow.log_metric("rollout_ep_rew_min", float(np.min(rewards)),
                              step=self.num_timesteps)
            mlflow.log_metric("rollout_ep_rew_max", float(np.max(rewards)),
                              step=self.num_timesteps)
        return True


def per_ep_eval(model, env_cfg: dict, n_per_ep: int = 10) -> dict:
    """Roll out n_per_ep deterministic episodes per equilibrium and return
    a flat dict of metrics (mean reward and mean episode length per EP)."""
    out = {}
    overall_lengths = []
    overall_rewards = []
    for ep in range(8):
        cfg = dict(env_cfg)
        cfg["target_mode"] = "fixed"
        cfg["target_ep"] = ep
        env = TriplePendulumEnv(
            target_ep=ep,
            target_mode="fixed",
            init_mode=str(cfg.get("init_mode", "near_target")),
            init_noise=float(cfg.get("init_noise", 0.05)),
            max_episode_steps=int(cfg["max_episode_steps"]),
        )
        rewards, lengths = [], []
        for trial in range(n_per_ep):
            obs, _ = env.reset(seed=ep * 1000 + trial)
            ep_r, ep_n = 0.0, 0
            done = False
            trunc = False
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, trunc, _ = env.step(action)
                ep_r += float(r)
                ep_n += 1
            rewards.append(ep_r)
            lengths.append(ep_n)
        out[f"ep{ep}_reward_mean"] = float(np.mean(rewards))
        out[f"ep{ep}_length_mean"] = float(np.mean(lengths))
        out[f"ep{ep}_success_rate"] = float(np.mean([
            l >= int(0.8 * cfg["max_episode_steps"]) for l in lengths
        ]))
        overall_rewards.extend(rewards)
        overall_lengths.extend(lengths)
    out["overall_reward_mean"] = float(np.mean(overall_rewards))
    out["overall_length_mean"] = float(np.mean(overall_lengths))
    out["overall_success_rate"] = float(np.mean([
        l >= int(0.8 * env_cfg["max_episode_steps"]) for l in overall_lengths
    ]))
    return out


def main(cfg_path: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    init_mlflow()
    run_name = f"m3_all_eps_{time.strftime('%Y%m%d_%H%M%S')}"

    env_cfg = cfg["env"]
    total_timesteps = int(cfg["total_timesteps"])

    train_env = DummyVecEnv([make_env(env_cfg)])
    eval_env = DummyVecEnv([make_env(env_cfg)])

    tqc_kwargs = dict(cfg["tqc"])
    policy = tqc_kwargs.pop("policy")
    policy_kwargs = tqc_kwargs.pop("policy_kwargs", {})

    model = TQC(
        policy,
        train_env,
        verbose=0,
        tensorboard_log=str(ROOT / "runs" / run_name),
        policy_kwargs=policy_kwargs,
        **tqc_kwargs,
    )

    cb_cfg = cfg.get("callbacks", {})
    eval_cfg = cfg.get("eval", {})

    rollout_cb = MLflowRolloutLogger(log_freq=int(cb_cfg.get("rollout_log_freq", 4000)))
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(ROOT / "checkpoints" / run_name),
        log_path=str(ROOT / "runs" / run_name / "eval"),
        eval_freq=int(eval_cfg.get("freq", 25000)),
        n_eval_episodes=int(eval_cfg.get("n_episodes", 16)),
        deterministic=bool(eval_cfg.get("deterministic", True)),
    )
    ckpt_cb = CheckpointCallback(
        save_freq=int(cb_cfg.get("checkpoint_freq", 100000)),
        save_path=str(ROOT / "checkpoints" / run_name),
        name_prefix="tqc",
    )

    t0 = time.time()
    with mlflow.start_run(run_name=run_name) as run:
        for k, v in tqc_kwargs.items():
            mlflow.log_param(f"tqc.{k}", v)
        for k, v in env_cfg.items():
            mlflow.log_param(f"env.{k}", v)
        mlflow.log_param("total_timesteps", total_timesteps)
        mlflow.log_param("git_commit",
                         os.popen("git rev-parse HEAD").read().strip() or "unknown")

        print(f"Run ID  : {run.info.run_id}")
        print(f"Run URL : {mlflow.get_tracking_uri()}/#/experiments/"
              f"{run.info.experiment_id}/runs/{run.info.run_id}")

        model.learn(total_timesteps=total_timesteps,
                    callback=[rollout_cb, eval_cb, ckpt_cb],
                    progress_bar=False)

        elapsed = time.time() - t0
        mlflow.log_metric("train_wall_seconds", elapsed)
        save_path = ROOT / "checkpoints" / run_name / "final.zip"
        model.save(str(save_path))
        try:
            mlflow.log_artifact(str(save_path), artifact_path="model")
        except Exception as exc:
            mlflow.set_tag("artifact_path_local", str(save_path))
            mlflow.set_tag("artifact_log_error", repr(exc)[:500])

        per_ep = per_ep_eval(model, env_cfg, n_per_ep=int(eval_cfg.get("final_n_episodes", 80)) // 8)
        for k, v in per_ep.items():
            mlflow.log_metric(f"final_{k}", v)
        print(f"DONE in {elapsed:.0f}s.")
        for ep in range(8):
            print(f"  EP{ep}  rew={per_ep[f'ep{ep}_reward_mean']:>+8.1f}  "
                  f"len={per_ep[f'ep{ep}_length_mean']:>6.1f}  "
                  f"succ={per_ep[f'ep{ep}_success_rate']:.2f}")
        print(f"  overall rew={per_ep['overall_reward_mean']:+.1f}  "
              f"len={per_ep['overall_length_mean']:.1f}  "
              f"succ={per_ep['overall_success_rate']:.2f}")

        stage = str(cfg.get("stage", "M3"))
        pipeline_notify(
            stage=stage,
            run_name=run_name,
            run_id=run.info.run_id,
            metrics=per_ep,
            config=cfg_path,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="training/configs/m3_all_eps_tqc.yaml")
    args = p.parse_args()
    main(args.config)

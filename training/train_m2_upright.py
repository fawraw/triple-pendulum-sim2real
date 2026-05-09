"""Milestone 2: train TQC to stabilize EP7 (UUU) in MuJoCo simulation.

Run from project root:

    source .venv/bin/activate
    MUJOCO_GL=osmesa python -m training.train_m2_upright \\
        --config training/configs/m2_upright_tqc.yaml

By default this logs to a local mlruns/ directory. To log to a remote MLflow
tracking server, export MLFLOW_TRACKING_URI before launching.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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


def make_env(env_cfg: dict):
    def _thunk():
        env = TriplePendulumEnv(
            target_ep=int(env_cfg["target_ep"]),
            init_mode=str(env_cfg.get("init_mode", "near_target")),
            init_noise=float(env_cfg.get("init_noise", 0.05)),
            max_episode_steps=int(env_cfg["max_episode_steps"]),
        )
        return Monitor(env)
    return _thunk


class MLflowRolloutLogger(BaseCallback):
    """Log rolling-window episode return statistics to MLflow."""

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
        rewards = [ep["r"] for ep in self.model.ep_info_buffer][-50:]
        try:
            mlflow.log_metric("timesteps", self.num_timesteps, step=self.num_timesteps)
            mlflow.log_metric("steps_per_s", float(sps), step=self.num_timesteps)
            if rewards:
                mlflow.log_metric("rollout_ep_rew_mean", float(np.mean(rewards)),
                                  step=self.num_timesteps)
                mlflow.log_metric("rollout_ep_rew_min", float(np.min(rewards)),
                                  step=self.num_timesteps)
                mlflow.log_metric("rollout_ep_rew_max", float(np.max(rewards)),
                                  step=self.num_timesteps)
        except Exception as e:
            print(f"[MLflowRolloutLogger] log failed at step {self.num_timesteps}: {e}", flush=True)
        return True


def main(cfg_path: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    init_mlflow()
    run_name = f"m2_upright_{time.strftime('%Y%m%d_%H%M%S')}"

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
        eval_freq=int(eval_cfg.get("freq", 20000)),
        n_eval_episodes=int(eval_cfg.get("n_episodes", 5)),
        deterministic=bool(eval_cfg.get("deterministic", True)),
    )
    ckpt_cb = CheckpointCallback(
        save_freq=int(cb_cfg.get("checkpoint_freq", 50000)),
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
        # Best-effort artifact logging. If the MLflow server is configured
        # without --serve-artifacts, the local artifact root is on a
        # different host and log_artifact will fail; we surface the path
        # via a tag instead so the run is still useful.
        try:
            mlflow.log_artifact(str(save_path), artifact_path="model")
        except Exception as exc:
            mlflow.set_tag("artifact_path_local", str(save_path))
            mlflow.set_tag("artifact_log_error", repr(exc)[:500])

        rewards = []
        for _ in range(int(eval_cfg.get("final_n_episodes", 20))):
            obs = eval_env.reset()
            done = [False]
            ep_r = 0.0
            while not done[0]:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, _ = eval_env.step(action)
                ep_r += float(r[0])
            rewards.append(ep_r)
        mlflow.log_metric("final_eval_reward_mean", float(np.mean(rewards)))
        mlflow.log_metric("final_eval_reward_std", float(np.std(rewards)))
        print(f"DONE in {elapsed:.0f}s. "
              f"Final eval mean={np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="training/configs/m2_upright_tqc.yaml")
    args = p.parse_args()
    main(args.config)

"""Milestone 2: train TQC to stabilize EP7 (UUU) in MuJoCo simulation.

Run from project root:

    source .venv/bin/activate
    python -m training.train_m2_upright --config training/configs/m2_upright_tqc.yaml

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
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Make sure `sim/` is importable when running as a module.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.envs.triple_pendulum_env import TriplePendulumEnv  # noqa: E402
from training.mlflow_setup import init_mlflow  # noqa: E402


def make_env(target_ep: int, max_steps: int, seed: int):
    def _thunk():
        env = TriplePendulumEnv(target_ep=target_ep, max_episode_steps=max_steps)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk


def main(cfg_path: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    init_mlflow()
    run_name = f"m2_upright_{time.strftime('%Y%m%d_%H%M%S')}"

    target_ep = int(cfg["env"]["target_ep"])
    max_steps = int(cfg["env"]["max_episode_steps"])
    total_timesteps = int(cfg["total_timesteps"])

    train_env = DummyVecEnv([make_env(target_ep, max_steps, seed=0)])
    eval_env = DummyVecEnv([make_env(target_ep, max_steps, seed=99)])

    tqc_kwargs = dict(cfg["tqc"])
    policy = tqc_kwargs.pop("policy")
    policy_kwargs = tqc_kwargs.pop("policy_kwargs", {})

    model = TQC(
        policy,
        train_env,
        verbose=1,
        tensorboard_log=str(ROOT / "runs" / run_name),
        policy_kwargs=policy_kwargs,
        **tqc_kwargs,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(ROOT / "checkpoints" / run_name),
        log_path=str(ROOT / "runs" / run_name / "eval"),
        eval_freq=int(cfg["eval"]["freq"]),
        n_eval_episodes=int(cfg["eval"]["n_episodes"]),
        deterministic=bool(cfg["eval"]["deterministic"]),
    )

    with mlflow.start_run(run_name=run_name):
        # Log config + git commit
        mlflow.log_params({f"cfg.{k}": v for k, v in tqc_kwargs.items()})
        mlflow.log_param("env.target_ep", target_ep)
        mlflow.log_param("env.max_episode_steps", max_steps)
        mlflow.log_param("total_timesteps", total_timesteps)
        mlflow.log_param("git_commit", os.popen("git rev-parse HEAD").read().strip() or "unknown")

        model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=True)
        save_path = ROOT / "checkpoints" / run_name / "final.zip"
        model.save(save_path)
        mlflow.log_artifact(str(save_path), artifact_path="model")

        # Final eval
        rewards = []
        for _ in range(20):
            obs = eval_env.reset()
            done = [False]
            ep_r = 0.0
            while not done[0]:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, _info = eval_env.step(action)
                ep_r += float(r[0])
            rewards.append(ep_r)
        mlflow.log_metric("final_eval_reward_mean", float(np.mean(rewards)))
        mlflow.log_metric("final_eval_reward_std", float(np.std(rewards)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="training/configs/m2_upright_tqc.yaml")
    args = p.parse_args()
    main(args.config)

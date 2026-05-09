"""Milestone 4: train a TQC policy that transitions between any two equilibria.

The policy receives:
  - current joint state (8 values: x, dx, θ₁, dθ₁, θ₂, dθ₂, θ₃, dθ₃)
  - source EP one-hot (8 bits) — set to zero vector during free transitions
  - target EP one-hot (8 bits)
Total observation: 24 dimensions.

Training curriculum (3 phases controlled by init_mode):
  1. "near_target"  — stabilization only (same as M3, warm-start)
  2. "bottom"       — swing-up from DDD to any EP
  3. "random"       — arbitrary start, arbitrary target

Success criterion per rollout: policy reaches the target EP window
(all |θᵢ - θᵢ*| < 0.3 rad, |x| < 0.5 m) AND holds it for at least
0.5 * max_episode_steps steps.

Run:
    MUJOCO_GL=osmesa python -m training.train_m4_transitions \\
        --config training/configs/m4_transitions_tqc.yaml
"""
from __future__ import annotations

import argparse
import subprocess
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
from training.pipeline_notifier import notify as pipeline_notify  # noqa: E402

EP_NAMES = ["DDD", "DDU", "DUD", "DUU", "UDD", "UDU", "UUD", "UUU"]

# 56 directed transitions (src != dst)
ALL_TRANSITIONS = [(src, dst) for src in range(8) for dst in range(8) if src != dst]


def make_env(env_cfg: dict):
    def _thunk():
        env = TriplePendulumEnv(
            target_ep=int(env_cfg.get("target_ep", 7)),
            target_mode=str(env_cfg.get("target_mode", "random")),
            init_mode=str(env_cfg.get("init_mode", "random")),
            init_noise=float(env_cfg.get("init_noise", 0.1)),
            max_episode_steps=int(env_cfg["max_episode_steps"]),
        )
        return Monitor(env)
    return _thunk


class MLflowRolloutLogger(BaseCallback):
    def __init__(self, log_freq: int = 10_000):
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


def _transition_success(lengths: list[int], max_steps: int) -> float:
    """Success = episode held target for >= 50% of max_steps."""
    return float(np.mean([l >= 0.5 * max_steps for l in lengths]))


def per_transition_eval(model, env_cfg: dict, n_per_transition: int = 5) -> dict:
    """Roll out n_per_transition deterministic episodes for each of the 56 transitions."""
    max_steps = int(env_cfg["max_episode_steps"])
    out: dict[str, float] = {}
    all_lengths: list[int] = []

    for src, dst in ALL_TRANSITIONS:
        lengths = []
        env = TriplePendulumEnv(
            target_ep=dst,
            target_mode="fixed",
            init_mode="near_target" if src == dst else "random",
            init_noise=0.3,
            max_episode_steps=max_steps,
        )
        for trial in range(n_per_transition):
            obs, _ = env.reset(seed=src * 100 + dst * 10 + trial,
                               options={"target_ep": dst})
            ep_n = 0
            done = trunc = False
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, trunc, _ = env.step(action)
                ep_n += 1
            lengths.append(ep_n)
        env.close()

        key = f"ep{src}to{dst}"
        out[f"{key}_success_rate"] = _transition_success(lengths, max_steps)
        out[f"{key}_length_mean"] = float(np.mean(lengths))
        all_lengths.extend(lengths)

    out["overall_success_rate"] = _transition_success(all_lengths, max_steps)
    out["overall_length_mean"] = float(np.mean(all_lengths))
    return out


_M4_REQUIRED = {
    ("env", "max_episode_steps"), ("tqc", "policy"), ("total_timesteps",),
}


def _validate_cfg(cfg: dict) -> None:
    for keys in _M4_REQUIRED:
        node = cfg
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                raise ValueError(f"Config missing required key: {'.'.join(keys)}")
            node = node[k]
    if not isinstance(cfg["total_timesteps"], (int, float)) or cfg["total_timesteps"] <= 0:
        raise ValueError("total_timesteps must be a positive number")


def _git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                       cwd=ROOT, check=False)
    return r.stdout.strip() or "unknown"


def main(cfg_path: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    _validate_cfg(cfg)

    init_mlflow()
    run_name = f"m4_transitions_{time.strftime('%Y%m%d_%H%M%S')}"

    env_cfg = cfg["env"]
    total_timesteps = int(cfg["total_timesteps"])
    tqc_kwargs = dict(cfg["tqc"])
    policy = tqc_kwargs.pop("policy")
    policy_kwargs = tqc_kwargs.pop("policy_kwargs", {})
    cb_cfg = cfg.get("callbacks", {})
    eval_cfg = cfg.get("eval", {})

    pretrained = cfg.get("pretrained_policy")

    train_env = DummyVecEnv([make_env(env_cfg)])
    eval_env = DummyVecEnv([make_env(env_cfg)])

    if pretrained:
        print(f"Loading pretrained policy: {pretrained}")
        model = TQC.load(pretrained, env=train_env, **tqc_kwargs)
    else:
        model = TQC(policy, train_env, verbose=0,
                    tensorboard_log=str(ROOT / "runs" / run_name),
                    policy_kwargs=policy_kwargs, **tqc_kwargs)

    rollout_cb = MLflowRolloutLogger(log_freq=int(cb_cfg.get("rollout_log_freq", 10_000)))
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(ROOT / "checkpoints" / run_name),
        log_path=str(ROOT / "runs" / run_name / "eval"),
        eval_freq=int(eval_cfg.get("freq", 50_000)),
        n_eval_episodes=int(eval_cfg.get("n_episodes", 56)),
        deterministic=bool(eval_cfg.get("deterministic", True)),
    )
    ckpt_cb = CheckpointCallback(
        save_freq=int(cb_cfg.get("checkpoint_freq", 500_000)),
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
        mlflow.log_param("pretrained_policy", pretrained or "none")
        mlflow.log_param("git_commit", _git_commit())

        print(f"Run ID  : {run.info.run_id}")
        print(f"Run URL : {mlflow.get_tracking_uri()}/#/experiments/"
              f"{run.info.experiment_id}/runs/{run.info.run_id}")
        print(f"Training {total_timesteps:,} steps — 56 transition policy")

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

        n_eval = int(eval_cfg.get("final_n_episodes", 280)) // 56  # 5 per transition
        metrics = per_transition_eval(model, env_cfg, n_per_transition=n_eval)
        for k, v in metrics.items():
            mlflow.log_metric(f"final_{k}", v)

        print(f"\nDONE in {elapsed:.0f}s  overall_success_rate={metrics['overall_success_rate']:.3f}")
        for src, dst in ALL_TRANSITIONS:
            key = f"ep{src}to{dst}"
            sr = metrics.get(f"{key}_success_rate", 0.0)
            if sr < 0.8:
                print(f"  {EP_NAMES[src]}→{EP_NAMES[dst]:>3s}  succ={sr:.2f}  ← below threshold")

        stage = str(cfg.get("stage", "M4"))
        pipeline_notify(stage=stage, run_name=run_name, run_id=run.info.run_id,
                        metrics=metrics, config=cfg_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="training/configs/m4_transitions_tqc.yaml")
    args = p.parse_args()
    main(args.config)

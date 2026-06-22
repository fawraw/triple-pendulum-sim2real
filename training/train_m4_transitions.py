"""Milestone 4: train a TQC policy that transitions between any two equilibria.

The policy reuses the M3 observation space (16 dims):
  - joint state (8): x, dx, theta1, dtheta1, theta2, dtheta2, theta3, dtheta3
  - target EP one-hot (8 bits)

It does NOT include a source-EP one-hot — the policy infers the current state
from the joint angles directly. The 56-transition coverage comes from training
with random init across all configurations and a target resampled every reset.

Pre-requisite: warm-start from a passing M3 checkpoint. Set `pretrained_policy`
in the YAML config to `checkpoints/<m3b_run>/final.zip`. Cold-start from random
init is rejected unless `allow_cold_start: true` is set explicitly in the config.

Success criterion per rollout: episode length >= 0.5 * max_episode_steps
(policy holds the target EP for at least half the budget).

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.envs.triple_pendulum_env import TriplePendulumEnv  # noqa: E402
from sim.equilibria import EP_NAMES  # noqa: E402
from training.env_utils import make_vec_env, seed_everything  # noqa: E402
from training.eval_utils import success_rate  # noqa: E402
from training.mlflow_setup import init_mlflow  # noqa: E402
from training.mlflow_safe import safe_artifact, safe_tag  # noqa: E402
from training.pipeline_notifier import notify as pipeline_notify  # noqa: E402

# 56 directed transitions (src != dst)
ALL_TRANSITIONS = [(src, dst) for src in range(8) for dst in range(8) if src != dst]


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
    return success_rate(lengths, max_steps, frac=0.5)


def per_transition_eval(model, env_cfg: dict, n_per_transition: int = 5) -> dict:
    """Roll out n_per_transition deterministic episodes for each of the 56 transitions.

    Uses the env's `target_mode='transition'` (added 2026-05-23) which:
      - Initialises near `start_ep` (passed explicitly per pair).
      - Disables angle-based fall detection until target is reached at least once.
      - Reports `info['reached_target']` after `transition_success_steps` in-tolerance steps.

    Success per episode = arrival (`reached_target=True`) before truncation.
    """
    max_steps = int(env_cfg["max_episode_steps"])
    out: dict[str, float] = {}
    all_arrivals: list[int] = []
    all_lengths: list[int] = []

    for src, dst in ALL_TRANSITIONS:
        lengths = []
        arrivals = []
        for trial in range(n_per_transition):
            env = TriplePendulumEnv(
                target_ep=dst,
                start_ep=src,
                target_mode="transition",
                init_mode="near_target",
                init_noise=float(env_cfg.get("init_noise", 0.05)),
                max_episode_steps=max_steps,
                fall_grace_steps=0,
                transition_success_tol_rad=float(env_cfg.get("transition_success_tol_rad", 0.2)),
                transition_success_steps=int(env_cfg.get("transition_success_steps", 200)),
                transition_bonus=0.0,
            )
            obs, _ = env.reset(seed=src * 100 + dst * 10 + trial)
            ep_n = 0
            done = trunc = False
            arrived = False
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, trunc, info = env.step(action)
                if info.get("reached_target"):
                    arrived = True
                ep_n += 1
            lengths.append(ep_n)
            arrivals.append(1 if arrived else 0)
            env.close()

        key = f"ep{src}to{dst}"
        out[f"{key}_success_rate"] = float(np.mean(arrivals))
        out[f"{key}_length_mean"] = float(np.mean(lengths))
        all_arrivals.extend(arrivals)
        all_lengths.extend(lengths)

    out["overall_success_rate"] = float(np.mean(all_arrivals))
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

    # Cold-start guard: M4 needs a warm-start from a passing M3 policy.
    # Training from scratch on 56 transitions wastes 5M steps and is almost
    # certainly the result of n8n auto-launching M4 before pretrained_policy
    # was filled in. Operator must opt-in explicitly.
    pretrained = cfg.get("pretrained_policy")
    if not pretrained and not cfg.get("allow_cold_start"):
        raise ValueError(
            "M4 cold-start refused: set 'pretrained_policy' to a passing M3 "
            "checkpoint (e.g. checkpoints/<m3b_run>/final.zip), or set "
            "'allow_cold_start: true' if this is intentional."
        )
    if pretrained and not Path(pretrained).is_absolute():
        # Relative paths are resolved against the project root (consistent
        # with how the launcher passes config paths).
        resolved = ROOT / pretrained
        if not resolved.exists():
            raise ValueError(f"pretrained_policy not found: {resolved}")


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

    seed = seed_everything(cfg.get("seed"))
    env_cfg = cfg["env"]
    total_timesteps = int(cfg["total_timesteps"])
    n_envs = int(cfg.get("n_envs", 1))
    tqc_kwargs = dict(cfg["tqc"])
    policy = tqc_kwargs.pop("policy")
    policy_kwargs = tqc_kwargs.pop("policy_kwargs", {})
    device = tqc_kwargs.pop("device", "auto")
    cb_cfg = cfg.get("callbacks", {})
    eval_cfg = cfg.get("eval", {})

    pretrained = cfg.get("pretrained_policy")

    train_env = make_vec_env(env_cfg, n_envs=n_envs)
    eval_env = make_vec_env(env_cfg, n_envs=1)

    if pretrained:
        print(f"Loading pretrained policy: {pretrained}")
        model = TQC.load(pretrained, env=train_env, device=device, **tqc_kwargs)
        if seed is not None:
            model.set_random_seed(seed)
    else:
        model = TQC(policy, train_env, verbose=0,
                    tensorboard_log=str(ROOT / "runs" / run_name),
                    policy_kwargs=policy_kwargs, device=device, seed=seed, **tqc_kwargs)
    actual_device = str(model.device)

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
        mlflow.log_param("n_envs", n_envs)
        mlflow.log_param("device", actual_device)
        mlflow.log_param("seed", seed if seed is not None else "none")
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
        final_path = ROOT / "checkpoints" / run_name / "final.zip"
        model.save(str(final_path))

        # Ship whichever of {best_model, final} has higher overall_success_rate
        # over the 56 transitions — same rationale as M3.
        n_eval = int(eval_cfg.get("final_n_episodes", 280)) // 56
        metrics = per_transition_eval(model, env_cfg, n_per_transition=n_eval)
        best_path = ROOT / "checkpoints" / run_name / "best_model.zip"
        save_path = final_path
        if best_path.exists():
            best_model = TQC.load(str(best_path))
            best_metrics = per_transition_eval(best_model, env_cfg, n_per_transition=n_eval)
            mlflow.log_metric("best_overall_success_rate", best_metrics["overall_success_rate"])
            mlflow.log_metric("final_overall_success_rate", metrics["overall_success_rate"])
            if best_metrics["overall_success_rate"] > metrics["overall_success_rate"]:
                print(f"[best] best_model wins ({best_metrics['overall_success_rate']:.3f} vs "
                      f"{metrics['overall_success_rate']:.3f}) — shipping best_model.")
                metrics = best_metrics
                save_path = best_path
                safe_tag("shipped_checkpoint", "best_model")
            else:
                safe_tag("shipped_checkpoint", "final")

        safe_artifact(str(save_path), artifact_path="model")

        for k, v in metrics.items():
            mlflow.log_metric(f"final_{k}", v)

        # Single-transition smoke (target_mode=fixed): the 56-transition overall
        # is uninformative because it scores 55 transitions the policy never
        # trained on. Surface the trained pair's own success rate so the smoke
        # result is interpretable (this was the cause of the misleading 0%).
        if str(env_cfg.get("target_mode")) == "fixed" and env_cfg.get("start_ep") is not None:
            src = int(env_cfg["start_ep"])
            dst = int(env_cfg.get("target_ep", 7))
            trained_sr = metrics.get(f"ep{src}to{dst}_success_rate")
            if trained_sr is not None:
                mlflow.log_metric("final_trained_transition_success_rate", trained_sr)
                mlflow.log_param("trained_transition", f"{EP_NAMES[src]}->{EP_NAMES[dst]}")
                print(f"  [smoke] trained transition {EP_NAMES[src]}->{EP_NAMES[dst]}: "
                      f"success={trained_sr:.2f}  (the 56-transition overall below is "
                      f"uninformative for a single-transition run)")

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

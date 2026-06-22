"""BC-only training: pretrain TQC actor on LQR demos, then evaluate (no RL).

Tests Stage 3A hypothesis: BC pre-training alone should improve EP4/EP6
without the destructive RL fine-tune that hurt EP5/EP7 in E1.

Demos collected on ALL 4 hard EPs (4, 5, 6, 7) — balanced exposure helps the
shared-weight network avoid biasing the actor toward EP4/EP6 only.

Run from project root:

    MUJOCO_GL=osmesa python -m training.train_bc_only \\
        --config training/configs/m3b_stage3a_bc_only_local.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.envs.triple_pendulum_env import TriplePendulumEnv  # noqa: E402
from training.eval_utils import success_rate  # noqa: E402
from training.train_bc_then_rl import (  # noqa: E402
    collect_lqr_demos,
    bc_pretrain,
    EQ_QPOS,
)

from sb3_contrib import TQC

# Complete the equilibrium table for ALL 8 EPs (was originally only EP4/EP6).
# Bit convention: bit 0 = link 1 (base), bit 1 = link 2 (mid), bit 2 = link 3 (tip);
# 1 = UP (absolute angle 0), 0 = DOWN (absolute angle pi).
_FULL_EQ = {
    0: np.array([0.0,  np.pi,  0.0,    0.0]),    # DDD
    1: np.array([0.0,  0.0,    np.pi,  0.0]),    # UDD (base up only)
    2: np.array([0.0,  np.pi, -np.pi,  np.pi]),  # DUD (mid up only)
    3: np.array([0.0,  0.0,    0.0,    np.pi]),  # UUD (base+mid up)
    4: np.array([0.0,  np.pi,  0.0,   -np.pi]),  # DDU (tip up only)
    5: np.array([0.0,  0.0,   -np.pi, -np.pi]),  # UDU (base+tip up)
    6: np.array([0.0,  np.pi, -np.pi,  0.0]),    # DUU (mid+tip up)
    7: np.array([0.0,  0.0,    0.0,    0.0]),    # UUU
}
for _k, _v in _FULL_EQ.items():
    EQ_QPOS.setdefault(_k, _v)


def per_ep_eval(model, n_per_ep=10, max_steps=1000, init_noise=0.05):
    out = {}
    overall_lengths = []
    for ep in range(8):
        lengths, rewards = [], []
        for trial in range(n_per_ep):
            env = TriplePendulumEnv(
                target_ep=ep, target_mode="fixed",
                init_mode="near_target", init_noise=init_noise,
                max_episode_steps=max_steps, fall_grace_steps=0,
            )
            obs, _ = env.reset(seed=ep * 1000 + trial)
            ep_r, ep_n = 0.0, 0
            done, trunc = False, False
            while not (done or trunc):
                act, _ = model.predict(obs, deterministic=True)
                obs, r, done, trunc, _ = env.step(act)
                ep_r += float(r)
                ep_n += 1
            lengths.append(ep_n)
            rewards.append(ep_r)
        succ = success_rate(lengths, max_steps, 0.8)
        out[f"EP{ep}"] = {
            "success_rate": float(succ),
            "mean_length": float(np.mean(lengths)),
            "mean_reward": float(np.mean(rewards)),
            "lengths": lengths,
        }
        overall_lengths.extend(lengths)
        print(f"EP{ep}: succ={succ:.2f}  lens={lengths}  mean={np.mean(lengths):.0f}")
    overall = success_rate(overall_lengths, max_steps, 0.8)
    out["overall_success_rate"] = float(overall)
    print(f"\nOverall: {overall*100:.1f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_path = cfg.get("load_model_path", "") or ""

    if load_path:
        resolved = str(ROOT / load_path) if not load_path.startswith("/") else load_path
        print(f"Loading: {resolved}")
        model = TQC.load(resolved, device=device)
        # Eval BEFORE BC (baseline)
        print("\n=== Baseline eval (before BC) ===")
        eval_before = per_ep_eval(model, n_per_ep=10)
    else:
        # From-scratch TQC build for BC pretraining
        from training.env_utils import make_vec_env  # noqa: E402
        print("From-scratch BC: building fresh TQC with policy_kwargs from config")
        tqc_kwargs = dict(cfg["tqc"])
        policy = tqc_kwargs.pop("policy")
        policy_kwargs = tqc_kwargs.pop("policy_kwargs", {})
        # Need a vec env to instantiate TQC; the env_cfg defines obs/action space
        env_cfg_for_init = {**cfg["env"], "target_mode": "fixed", "target_ep": 0}
        tmp_env = make_vec_env(env_cfg_for_init, n_envs=1)
        model = TQC(
            policy, tmp_env, verbose=0,
            policy_kwargs=policy_kwargs,
            device=device,
            **tqc_kwargs,
        )
        eval_before = {"note": "from-scratch (no baseline to eval)"}
        for ep in range(8):
            eval_before[f"EP{ep}"] = {"success_rate": 0.0}
        eval_before["overall_success_rate"] = 0.0

    # Generate LQR demos
    bc_cfg = cfg.get("bc", {})
    n_eps_per = int(bc_cfg.get("n_eps_per", 50))
    eps_list = tuple(bc_cfg.get("eps", [4, 5, 6, 7]))
    print(f"\n=== Collecting LQR demos for EPs {eps_list} ===")
    t0 = time.time()
    # _FULL_EQ populated at module top covers all 8 EPs.
    obs, actions = collect_lqr_demos(cfg["env"], eps_list=eps_list, n_eps_per=n_eps_per)
    print(f"  demos in {time.time()-t0:.1f}s, {len(obs)} transitions")

    # BC pretrain
    n_epochs = int(bc_cfg.get("n_epochs", 30))
    batch = int(bc_cfg.get("batch_size", 256))
    lr = float(bc_cfg.get("lr", 3e-4))
    print(f"\n=== BC pretrain ({n_epochs} epochs, batch {batch}, lr {lr}) ===")
    t0 = time.time()
    bc_pretrain(model, obs, actions, n_epochs=n_epochs, batch_size=batch, lr=lr)
    print(f"  BC done in {time.time()-t0:.1f}s")

    # Save and eval after BC
    bc_ckpt = ROOT / "checkpoints" / "bc_only.zip"
    bc_ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(bc_ckpt))
    print(f"  BC checkpoint saved: {bc_ckpt}")

    print("\n=== Eval AFTER BC ===")
    eval_after = per_ep_eval(model, n_per_ep=10)

    # Summary
    out = {
        "config": args.config,
        "load_model_path": str(load_path),
        "eval_before": eval_before,
        "eval_after": eval_after,
        "delta": {
            f"EP{i}": eval_after[f"EP{i}"]["success_rate"] - eval_before[f"EP{i}"]["success_rate"]
            for i in range(8)
        },
        "overall_delta": eval_after["overall_success_rate"] - eval_before["overall_success_rate"],
    }
    out_path = ROOT / "results" / f"bc_only_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved: {out_path}")
    print(f"\n=== DELTA per EP ===")
    for i in range(8):
        d = out["delta"][f"EP{i}"]
        arrow = "▲" if d > 0 else ("▼" if d < 0 else "=")
        print(f"  EP{i}: {eval_before[f'EP{i}']['success_rate']:.2f} → "
              f"{eval_after[f'EP{i}']['success_rate']:.2f} ({arrow}{abs(d):.2f})")
    print(f"  Overall: {eval_before['overall_success_rate']:.3f} → "
          f"{eval_after['overall_success_rate']:.3f}  ({out['overall_delta']:+.3f})")


if __name__ == "__main__":
    main()

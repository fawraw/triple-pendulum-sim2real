"""BC-only training: pretrain TQC actor on LQR demos, then evaluate (no RL).

Tests Stage 3A hypothesis: BC pre-training alone should improve EP4/EP6
without the destructive RL fine-tune that hurt EP5/EP7 in E1.

Demos collected on ALL 4 hard EPs (4, 5, 6, 7) — balanced exposure helps the
shared-weight network avoid biasing the actor toward EP4/EP6 only.

Run from project root:

    MUJOCO_GL=osmesa python -m training.train_bc_only \\
        --config training/configs/m3b_stage3a_bc_only.yaml
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
from training.train_bc_then_rl import (  # noqa: E402
    collect_lqr_demos,
    bc_pretrain,
    EQ_QPOS,
)

from sb3_contrib import TQC


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
        succ = sum(1 for l in lengths if l >= 800) / n_per_ep
        out[f"EP{ep}"] = {
            "success_rate": float(succ),
            "mean_length": float(np.mean(lengths)),
            "mean_reward": float(np.mean(rewards)),
            "lengths": lengths,
        }
        overall_lengths.extend(lengths)
        print(f"EP{ep}: succ={succ:.2f}  lens={lengths}  mean={np.mean(lengths):.0f}")
    overall = sum(1 for l in overall_lengths if l >= 800) / len(overall_lengths)
    out["overall_success_rate"] = float(overall)
    print(f"\nOverall: {overall*100:.1f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    load_path = cfg["load_model_path"]
    resolved = str(ROOT / load_path) if not load_path.startswith("/") else load_path
    print(f"Loading: {resolved}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TQC.load(resolved, device=device)
    print(f"  device={model.device}")

    # Eval BEFORE BC (baseline)
    print("\n=== Baseline eval (before BC) ===")
    eval_before = per_ep_eval(model, n_per_ep=10)

    # Generate LQR demos
    bc_cfg = cfg.get("bc", {})
    n_eps_per = int(bc_cfg.get("n_eps_per", 50))
    eps_list = tuple(bc_cfg.get("eps", [4, 5, 6, 7]))
    print(f"\n=== Collecting LQR demos for EPs {eps_list} ===")
    t0 = time.time()
    # collect_lqr_demos works for EP4/EP6 (in EQ_QPOS dict). Add EP5/EP7 quadrants:
    # We need to extend EQ_QPOS for all hard EPs we want.
    # EP5 = UDU = (link1=U, link2=D, link3=U): qpos[1]=0, qpos[2]=-pi (relative goes to pi), qpos[3]=-pi (relative back to 0)
    # EP7 = UUU: qpos = [0, 0, 0, 0]
    extra_eqs = {
        5: np.array([0.0, 0.0, -np.pi, -np.pi]),
        7: np.array([0.0, 0.0, 0.0, 0.0]),
    }
    for k, v in extra_eqs.items():
        if k not in EQ_QPOS:
            EQ_QPOS[k] = v
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

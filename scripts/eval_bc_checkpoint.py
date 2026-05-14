"""Evaluate a checkpoint with the same per-EP protocol used by train_m3_all_eps's
`per_ep_eval()`: 10 deterministic rollouts per EP, strict (fall_grace_steps=0),
init_noise=0.05.  Output JSON summary.

Accepts (and ignores) --config so the standard runpod bootstrap can invoke it.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_DEFAULT_ROOT = Path("/workspace/triple-pendulum-sim2real")
ROOT = _DEFAULT_ROOT if _DEFAULT_ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sb3_contrib import TQC  # noqa: E402
from sim.envs.triple_pendulum_env import TriplePendulumEnv  # noqa: E402


CKPT_CANDIDATES = [
    ROOT / "checkpoints" / "_bc_pretrained.zip",
    Path("/workspace/_bc_pretrained.zip"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="ignored (bootstrap compat)")
    ap.add_argument("--checkpoint", default="",
                    help="explicit path; else searches CKPT_CANDIDATES")
    args, _ = ap.parse_known_args()

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        ckpt = next((p for p in CKPT_CANDIDATES if p.exists()), None)
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found, tried: {CKPT_CANDIDATES}")
    print(f"Loading: {ckpt}")
    model = TQC.load(str(ckpt), device="cpu")

    n_per_ep = 10
    out = {"checkpoint": str(ckpt), "n_per_ep": n_per_ep, "per_ep": {}}
    overall_lengths = []
    for ep in range(8):
        lengths = []
        rewards = []
        for trial in range(n_per_ep):
            env = TriplePendulumEnv(
                target_ep=ep, target_mode="fixed",
                init_mode="near_target", init_noise=0.05,
                max_episode_steps=1000, fall_grace_steps=0,
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
        out["per_ep"][f"EP{ep}"] = {
            "success_rate": succ,
            "lengths": lengths,
            "mean_length": float(np.mean(lengths)),
            "mean_reward": float(np.mean(rewards)),
        }
        overall_lengths.extend(lengths)
        print(f"EP{ep}: succ={succ:.2f}  lens={lengths}  mean_len={np.mean(lengths):.0f}")
    overall_succ = sum(1 for l in overall_lengths if l >= 800) / len(overall_lengths)
    out["overall_success_rate"] = overall_succ
    out["overall_mean_length"] = float(np.mean(overall_lengths))
    print(f"\nOverall: {overall_succ*100:.1f}% (n={len(overall_lengths)})")

    out_path = ROOT.parent / "bc_eval.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

"""Render a trained policy: produces an MP4 of one rollout and saves a
final-pose PNG. Intended for quick visual inspection of training runs.

Usage:
    MUJOCO_GL=osmesa python scripts/eval_policy.py \\
        --checkpoint checkpoints/m2/final.zip \\
        --target-ep 7 \\
        --out assets/eval_m2.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np
from sb3_contrib import TQC

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from sim.envs.triple_pendulum_env import TriplePendulumEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target-ep", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--out", default="assets/eval.mp4")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = TriplePendulumEnv(target_ep=args.target_ep, max_episode_steps=args.max_steps)
    obs, _ = env.reset(seed=args.seed)
    model = TQC.load(args.checkpoint, env=None)

    renderer = mujoco.Renderer(env.model, height=720, width=1280)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(str(out_path), fps=args.fps, codec="libx264", quality=8)

    rewards = []
    try:
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = env.step(action)
            rewards.append(float(r))
            renderer.update_scene(env.data, camera="track")
            writer.append_data(renderer.render())
    finally:
        writer.close()

    final_png = out_path.with_suffix(".png")
    renderer.update_scene(env.data, camera="track")
    iio.imwrite(str(final_png), renderer.render())

    print(f"Saved: {out_path}")
    print(f"Saved: {final_png}")
    print(f"Episode steps   : {len(rewards)}")
    print(f"Sum of rewards  : {np.sum(rewards):.2f}")
    print(f"Mean reward     : {np.mean(rewards):.4f}")


if __name__ == "__main__":
    main()

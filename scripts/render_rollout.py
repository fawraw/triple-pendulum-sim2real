"""Render a saved policy as an MP4 video.

Usage:
    MUJOCO_GL=osmesa python scripts/render_rollout.py \\
        --policy checkpoints/m3b_all_eps_<run>/final.zip \\
        --ep 7 \\
        --out results/videos/ep7.mp4

    # Render all 8 EPs back-to-back:
    MUJOCO_GL=osmesa python scripts/render_rollout.py \\
        --policy checkpoints/m3b_all_eps_<run>/final.zip \\
        --all-eps \\
        --out results/videos/all_eps.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
from sb3_contrib import TQC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.envs.triple_pendulum_env import TriplePendulumEnv  # noqa: E402
from sim.equilibria import EP_NAMES  # noqa: E402

FPS = 50


def render_ep(model, ep: int, max_steps: int, seed: int) -> list[np.ndarray]:
    env = TriplePendulumEnv(
        target_ep=ep,
        target_mode="fixed",
        init_mode="near_target",
        init_noise=0.05,
        max_episode_steps=max_steps,
        render_mode="rgb_array",
    )
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    done = trunc = False
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, _ = env.step(action)
        frames.append(env.render())
    env.close()
    return frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="Path to .zip policy file")
    p.add_argument("--ep", type=int, default=None, help="Target EP (0-7)")
    p.add_argument("--all-eps", action="store_true", help="Render all 8 EPs sequentially")
    p.add_argument("--out", default="rollout.mp4", help="Output MP4 path")
    p.add_argument("--steps", type=int, default=1000, help="Max episode steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=FPS)
    args = p.parse_args()

    if args.ep is None and not args.all_eps:
        p.error("Specify --ep <0-7> or --all-eps")

    print(f"Loading policy: {args.policy}")
    model = TQC.load(args.policy)

    eps = list(range(8)) if args.all_eps else [args.ep]

    all_frames: list[np.ndarray] = []
    for ep in eps:
        print(f"  Rendering EP{ep} ({EP_NAMES[ep]})...")
        frames = render_ep(model, ep, args.steps, args.seed + ep)
        all_frames.extend(frames)
        print(f"    {len(frames)} frames")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(all_frames)} frames → {out}")
    imageio.mimsave(str(out), all_frames, fps=args.fps, quality=8)
    print("Done.")


if __name__ == "__main__":
    main()

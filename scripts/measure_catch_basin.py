#!/usr/bin/env python3
"""Measure a stabilizer policy's catch basin for a target equilibrium.

For the two-stage hand-off (swing-up -> stabilizer) to work, the swing-up must
deliver the system into the stabilizer's basin of attraction. This sweeps an
initial angular offset x initial velocity on the target's UP link and reports
the fraction of rollouts the stabilizer recovers (survives >= success_frac of
the budget). The resulting grid is the catch region the swing-up must hit.

Finding (M3 run m3_all_eps_20260509_072849 on UDD, 2026-06-25): reliable only
within ~0.1 rad at near-zero velocity; collapses for offset >= 0.3 or vel >= 2.
=> the swing-up needs a soft, near-zero-velocity delivery, OR train a
wider-basin catcher (larger init_noise + velocity + off-centre cart).

Example:
    MUJOCO_GL=osmesa python scripts/measure_catch_basin.py \
        --stabilizer checkpoints/m3_all_eps_20260509_072849/best_model.zip --ep 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from sb3_contrib import TQC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.envs.triple_pendulum_env import TriplePendulumEnv  # noqa: E402
from sim.equilibria import EP_NAMES, ep_target_angles  # noqa: E402


def measure(stab_path, ep, offsets, vels, n_trials, max_steps, success_frac):
    m = TQC.load(stab_path, device="cpu")
    tgt = ep_target_angles(ep)
    up_links = [i for i in range(3) if abs(tgt[i]) < 1e-6]  # links targeted UP
    perturb = up_links[0] if up_links else 0
    need = int(success_frac * max_steps)
    print(f"Catch basin for {EP_NAMES[ep]} (perturb link{perturb+1}); "
          f"success = survive >= {need}/{max_steps}")
    print("vel\\off " + "".join(f"{o:5.1f}" for o in offsets))
    for vel in vels:
        row = []
        for off in offsets:
            succ = 0
            for trial in range(n_trials):
                env = TriplePendulumEnv(target_ep=ep, target_mode="fixed",
                                        init_mode="near_target", init_noise=0.0,
                                        max_episode_steps=max_steps, fall_grace_steps=0)
                env.reset(seed=trial)
                sgn = 1 if trial % 2 else -1
                abs_ang = tgt.copy()
                abs_ang[perturb] += sgn * off
                env.data.qpos[1] = abs_ang[0]
                env.data.qpos[2] = abs_ang[1] - abs_ang[0]
                env.data.qpos[3] = abs_ang[2] - abs_ang[1]
                env.data.qvel[1 + perturb] = sgn * vel
                mujoco.mj_forward(env.model, env.data)
                obs = env._obs()
                n = 0
                term = trunc = False
                while not (term or trunc):
                    a, _ = m.predict(obs, deterministic=True)
                    obs, _, term, trunc, _ = env.step(a)
                    n += 1
                succ += n >= need
            row.append(succ / n_trials)
        print(f"{vel:4.0f}   " + "".join(f"{x:5.1f}" for x in row))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stabilizer", required=True)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--offsets", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--vels", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0])
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--success-frac", type=float, default=0.8)
    a = p.parse_args()
    measure(a.stabilizer, a.ep, a.offsets, a.vels, a.n, a.max_steps, a.success_frac)


if __name__ == "__main__":
    main()

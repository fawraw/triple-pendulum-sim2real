#!/usr/bin/env python3
"""Evaluate the two-stage hand-off (swing-up -> stabilizer) on a transition.

Drives the env with a HandoffController: a swing-up policy until the state enters
the stabilizer's capture set, then the stabilizer (M3) takes over. Reports, per
trial, whether the target was reached + held, when the hand-off happened, the
cart excursion, and how the control time split between the two policies.

Example (run from project root, on a host with the checkpoints):
    MUJOCO_GL=osmesa python scripts/handoff_eval.py \
        --swingup checkpoints/m4_transitions_20260623_183958/final.zip \
        --stabilizer checkpoints/m3_all_eps_20260509_072849/best_model.zip \
        --src 0 --dst 1 --capture-tol 0.35 --n 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sb3_contrib import TQC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.envs.triple_pendulum_env import TriplePendulumEnv  # noqa: E402
from sim.equilibria import EP_NAMES, ep_target_angles  # noqa: E402
from sim.handoff import HandoffController  # noqa: E402


def run(swingup_path, stab_path, src, dst, capture_tol, capture_vel, n, max_steps,
        cart_limit, hold_frac):
    swingup = TQC.load(swingup_path, device="cpu")
    stab = TQC.load(stab_path, device="cpu")
    target = ep_target_angles(dst)
    hold_needed = int(hold_frac * max_steps)
    print(f"Hand-off {EP_NAMES[src]}->{EP_NAMES[dst]}  capture_tol={capture_tol} "
          f"capture_vel={capture_vel}  n={n}  hold>={hold_needed}/{max_steps}")
    reached_n = held_n = 0
    for trial in range(n):
        env = TriplePendulumEnv(target_ep=dst, start_ep=src, target_mode="transition",
                                init_mode="near_target", init_noise=0.05,
                                max_episode_steps=max_steps, fall_grace_steps=0,
                                transition_success_tol_rad=capture_tol,
                                transition_success_steps=hold_needed,
                                transition_bonus=0.0, cart_limit=cart_limit)
        ctrl = HandoffController(swingup, stab, target,
                                 capture_tol_rad=capture_tol,
                                 capture_vel_rad_s=capture_vel, latch=True)
        obs, _ = env.reset(seed=trial)
        ctrl.reset()
        xs, modes = [], []
        term = trunc = False
        post_handoff_intol = 0
        max_post_handoff_intol = 0
        reached = False
        while not (term or trunc):
            js = env._joint_state()
            action, mode = ctrl.act(obs, [js[2], js[4], js[6]], [js[3], js[5], js[7]])
            obs, _, term, trunc, info = env.step(action)
            xs.append(float(env.data.qpos[0]))
            modes.append(mode)
            if info["reached_target"]:
                reached = True
            if ctrl._captured:
                in_tol = bool(np.all(np.abs(env._angle_error()) < capture_tol))
                post_handoff_intol = post_handoff_intol + 1 if in_tol else 0
                max_post_handoff_intol = max(max_post_handoff_intol, post_handoff_intol)
        held = max_post_handoff_intol >= hold_needed
        reached_n += reached
        held_n += held
        stab_frac = modes.count("stab") / len(modes) if modes else 0.0
        print(f"  t{trial}: handoff@{ctrl.handoff_step} reached={reached} held={held} "
              f"| cartx[{min(xs):+.2f},{max(xs):+.2f}] stab_time={stab_frac:.0%} "
              f"| max_hold_after_handoff={max_post_handoff_intol}/{hold_needed}")
        env.close()
    print(f"SUMMARY {EP_NAMES[src]}->{EP_NAMES[dst]}: reached {reached_n}/{n}, held {held_n}/{n}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--swingup", required=True)
    p.add_argument("--stabilizer", required=True)
    p.add_argument("--src", type=int, default=0)
    p.add_argument("--dst", type=int, default=1)
    p.add_argument("--capture-tol", type=float, default=0.35)
    p.add_argument("--capture-vel", type=float, default=None)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--cart-limit", type=float, default=1.10)
    p.add_argument("--hold-frac", type=float, default=0.05)
    a = p.parse_args()
    run(a.swingup, a.stabilizer, a.src, a.dst, a.capture_tol, a.capture_vel,
        a.n, a.max_steps, a.cart_limit, a.hold_frac)


if __name__ == "__main__":
    main()

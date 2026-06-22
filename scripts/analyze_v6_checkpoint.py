"""Stage 1 behavioral analysis of M3b-v6 cloud 72.5% checkpoint.

For each of EP4, EP6, EP7:
  - 30 deterministic rollouts with different seeds
  - Per rollout: initial qpos/qvel, success/fail, length, first 50 actions

For EP4 specifically (the hardest):
  - Compute LQR feedback K (numerical linearization + Riccati)
  - For each rollout step, compute ||policy_action - LQR_action||
  - Aggregate divergence stats: success vs failure

Output: JSON summary at /workspace/v6_analysis.json (or repo-relative).

Accepts (and ignores) --config so the standard runpod bootstrap can invoke
it as `python -m scripts.analyze_v6_checkpoint --config X` without crashing.
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import sys
import json
from pathlib import Path
import numpy as np
import pickle
from scipy.linalg import solve_continuous_are
import mujoco

_DEFAULT_ROOT = Path("/workspace/triple-pendulum-sim2real")
ROOT = _DEFAULT_ROOT if _DEFAULT_ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sb3_contrib import TQC
from sim.envs.triple_pendulum_env import TriplePendulumEnv
from training.eval_utils import success_threshold


CKPT_DIR = ROOT / "checkpoints" / "m3_all_eps_20260512_204421"
CKPT_BEST = CKPT_DIR / "best_model.zip"
CKPT_FINAL = CKPT_DIR / "final.zip"
CKPT = CKPT_BEST if CKPT_BEST.exists() else CKPT_FINAL


def equilibrium_qpos(ep):
    target_qpos = {
        0: np.array([0.0,  np.pi,  0.0,    0.0]),
        4: np.array([0.0,  np.pi,  0.0,   -np.pi]),
        6: np.array([0.0,  np.pi, -np.pi,  0.0]),
        7: np.array([0.0,  0.0,   0.0,    0.0]),
    }
    return target_qpos.get(ep)


def get_state(data):
    return np.concatenate([data.qpos.copy(), data.qvel.copy()])


def set_state(model, data, x):
    data.qpos[:] = x[:4]
    data.qvel[:] = x[4:]
    mujoco.mj_forward(model, data)


def compute_lqr_K(env, ep):
    """Numerical linearization + continuous Riccati."""
    x_eq = np.concatenate([equilibrium_qpos(ep), np.zeros(4)])
    n_x = 8
    eps = 1e-5
    dt = env.model.opt.timestep
    A_d = np.zeros((n_x, n_x))
    B_d = np.zeros((n_x, 1))
    set_state(env.model, env.data, x_eq)
    env.data.ctrl[0] = 0.0
    mujoco.mj_step(env.model, env.data)
    x_next_eq = get_state(env.data)
    for i in range(n_x):
        x_pert = x_eq.copy(); x_pert[i] += eps
        set_state(env.model, env.data, x_pert)
        env.data.ctrl[0] = 0.0
        mujoco.mj_step(env.model, env.data)
        A_d[:, i] = (get_state(env.data) - x_next_eq) / eps
    set_state(env.model, env.data, x_eq)
    env.data.ctrl[0] = eps
    mujoco.mj_step(env.model, env.data)
    B_d[:, 0] = (get_state(env.data) - x_next_eq) / eps
    A = (A_d - np.eye(n_x)) / dt
    B = B_d / dt
    Q = np.diag([1.0, 50.0, 50.0, 100.0, 0.1, 1.0, 1.0, 1.0])
    R = np.array([[1.0]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, x_eq


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def rollout_with_trace(model, ep, seed, K=None, x_eq=None, max_steps=1000):
    env = TriplePendulumEnv(
        target_ep=ep, target_mode="fixed",
        init_mode="near_target", init_noise=0.05,
        max_episode_steps=max_steps, fall_grace_steps=0,
    )
    obs, _ = env.reset(seed=seed)
    init_state = get_state(env.data)
    actions = []
    lqr_actions = []
    states = []
    for step in range(max_steps):
        act, _ = model.predict(obs, deterministic=True)
        a = float(act[0]) if hasattr(act, "__len__") else float(act)

        if K is not None and x_eq is not None:
            x = get_state(env.data)
            dx = x - x_eq
            for i in [1, 2, 3]:
                dx[i] = wrap(dx[i])
            u = float(-(K @ dx)[0])
            lqr_actions.append(max(-1.0, min(1.0, u)))
        actions.append(a)
        states.append(get_state(env.data).copy())

        obs, r, done, trunc, _ = env.step(act)
        if done or trunc:
            break
    survived = len(actions)
    success = survived >= success_threshold(env.max_episode_steps, 0.8)
    return {
        "seed": seed,
        "survived": survived,
        "success": success,
        "init_state": init_state.tolist(),
        "actions": actions,
        "lqr_actions": lqr_actions if K is not None else None,
        "final_state": get_state(env.data).tolist(),
    }


def analyze_ep(model, ep, n_rollouts=30, compute_lqr=False):
    print(f"\n=== EP{ep} — {n_rollouts} rollouts ===")
    K, x_eq = None, None
    if compute_lqr:
        env_tmp = TriplePendulumEnv(target_ep=ep, target_mode="fixed",
            init_mode="near_target", init_noise=0.0,
            max_episode_steps=1000, fall_grace_steps=0)
        env_tmp.reset(seed=0)
        K, x_eq = compute_lqr_K(env_tmp, ep)
        print(f"  LQR K computed (shape {K.shape})")

    rollouts = []
    for seed in range(n_rollouts):
        r = rollout_with_trace(model, ep, seed=ep*1000+seed, K=K, x_eq=x_eq)
        rollouts.append(r)

    successes = [r for r in rollouts if r["success"]]
    failures = [r for r in rollouts if not r["success"]]
    succ_rate = len(successes) / n_rollouts
    mean_survived = np.mean([r["survived"] for r in rollouts])
    print(f"  Success rate: {len(successes)}/{n_rollouts} = {succ_rate*100:.1f}%")
    print(f"  Mean survived: {mean_survived:.0f}")

    init_states = np.array([r["init_state"] for r in rollouts])
    succ_mask = np.array([r["success"] for r in rollouts])
    summary = {
        "ep": ep, "n_rollouts": n_rollouts,
        "success_rate": succ_rate,
        "mean_survived": float(mean_survived),
        "init_state_stats": {},
        "action_stats": {},
    }
    state_dims = ["cart_x", "th1", "th2", "th3", "vcart", "vth1", "vth2", "vth3"]
    if succ_mask.sum() > 0 and (~succ_mask).sum() > 0:
        for i, name in enumerate(state_dims):
            s_vals = init_states[succ_mask, i]
            f_vals = init_states[~succ_mask, i]
            summary["init_state_stats"][name] = {
                "succ_mean": float(s_vals.mean()), "succ_std": float(s_vals.std()),
                "fail_mean": float(f_vals.mean()), "fail_std": float(f_vals.std()),
                "abs_diff_means": float(abs(s_vals.mean() - f_vals.mean())),
            }

    all_actions_first50 = np.array([r["actions"][:50] + [0.0]*(50-len(r["actions"])) for r in rollouts])
    summary["action_stats"]["all_mean"] = float(all_actions_first50.mean())
    summary["action_stats"]["all_std"]  = float(all_actions_first50.std())
    summary["action_stats"]["sat_lo_pct"] = float((all_actions_first50 <= -0.99).mean() * 100)
    summary["action_stats"]["sat_hi_pct"] = float((all_actions_first50 >=  0.99).mean() * 100)
    if succ_mask.sum() > 0 and (~succ_mask).sum() > 0:
        s_a = np.array([r["actions"][:50] + [0.0]*(50-len(r["actions"])) for r in rollouts if r["success"]])
        f_a = np.array([r["actions"][:50] + [0.0]*(50-len(r["actions"])) for r in rollouts if not r["success"]])
        summary["action_stats"]["succ_mean"] = float(s_a.mean())
        summary["action_stats"]["succ_std"]  = float(s_a.std())
        summary["action_stats"]["fail_mean"] = float(f_a.mean())
        summary["action_stats"]["fail_std"]  = float(f_a.std())

    if compute_lqr:
        lqr_div = []
        for r in rollouts:
            if r["lqr_actions"]:
                n = min(len(r["actions"]), len(r["lqr_actions"]), 100)
                a = np.array(r["actions"][:n])
                la = np.array(r["lqr_actions"][:n])
                lqr_div.append({
                    "success": r["success"],
                    "mean_abs_diff": float(np.mean(np.abs(a - la))),
                    "early_mean_abs_diff": float(np.mean(np.abs(a[:20] - la[:20]))),
                })
        if lqr_div:
            s_div = [d["mean_abs_diff"] for d in lqr_div if d["success"]]
            f_div = [d["mean_abs_diff"] for d in lqr_div if not d["success"]]
            summary["lqr_divergence"] = {
                "all_mean": float(np.mean([d["mean_abs_diff"] for d in lqr_div])),
                "succ_mean": float(np.mean(s_div)) if s_div else None,
                "fail_mean": float(np.mean(f_div)) if f_div else None,
                "early_all_mean": float(np.mean([d["early_mean_abs_diff"] for d in lqr_div])),
            }
    return summary, rollouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="ignored (for bootstrap compat)")
    ap.parse_known_args()

    print(f"Loading: {CKPT}")
    model = TQC.load(str(CKPT), device="cpu")
    print(f"  policy: {model.policy.__class__.__name__}")

    summaries = {}
    all_rollouts = {}
    for ep in [4, 6, 7]:
        s, r = analyze_ep(model, ep, n_rollouts=30, compute_lqr=(ep == 4))
        summaries[f"EP{ep}"] = s
        all_rollouts[f"EP{ep}"] = r

    out = ROOT.parent / "v6_analysis.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"\nSaved: {out}")
    pickle_out = ROOT.parent / "v6_rollouts.pkl"
    with pickle_out.open("wb") as f:
        pickle.dump(all_rollouts, f)
    print(f"Saved: {pickle_out}")


if __name__ == "__main__":
    main()

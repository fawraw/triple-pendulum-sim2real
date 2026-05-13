"""E1: LQR Behavior Cloning pre-training → standard RL fine-tuning.

Stage 1 diagnostic showed v6's actor outputs actions ~4x larger than the LQR
optimum on EP4 (mean abs diff 0.83 in [-1,+1]).  Direct fix: pre-train the
actor via supervised regression against LQR-generated demonstrations on the
hard EPs (4 and 6), then continue standard TQC fine-tuning.

Procedure:
  Phase 0  Collect LQR demos    : 100 episodes per EP on EP4+EP6 with the
                                  numerically-derived LQR controller. ~100K
                                  (obs, lqr_action) pairs.
  Phase 1  BC pretrain          : minimize MSE(actor(obs), lqr_action) for
                                  N_BC_EPOCHS over the demos.  ~30s on GPU.
  Phase 2  RL fine-tune         : standard TQC fine-tune with weighted
                                  target_mode (same as v6 phase 1) from the
                                  BC-pretrained checkpoint.

Run via the standard bootstrap by setting TP_STAGE_MODULE=training.train_bc_then_rl
and TP_STAGE_CONFIG=training/configs/m3b_stage2_e1_bc_rl.yaml.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.linalg import solve_continuous_are
import mujoco
from sb3_contrib import TQC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.envs.triple_pendulum_env import TriplePendulumEnv  # noqa: E402
from training.train_m3_all_eps import main as rl_train_main  # noqa: E402


EQ_QPOS = {
    4: np.array([0.0, np.pi, 0.0, -np.pi]),  # DDU
    6: np.array([0.0, np.pi, -np.pi, 0.0]),  # DUU
}


def _get_state(data):
    return np.concatenate([data.qpos.copy(), data.qvel.copy()])


def _set_state(model, data, x):
    data.qpos[:] = x[:4]
    data.qvel[:] = x[4:]
    mujoco.mj_forward(model, data)


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def compute_lqr_K(env, ep):
    x_eq = np.concatenate([EQ_QPOS[ep], np.zeros(4)])
    n_x = 8
    eps = 1e-5
    dt = env.model.opt.timestep
    A_d = np.zeros((n_x, n_x))
    B_d = np.zeros((n_x, 1))
    _set_state(env.model, env.data, x_eq)
    env.data.ctrl[0] = 0.0
    mujoco.mj_step(env.model, env.data)
    x_next_eq = _get_state(env.data)
    for i in range(n_x):
        x_pert = x_eq.copy(); x_pert[i] += eps
        _set_state(env.model, env.data, x_pert)
        env.data.ctrl[0] = 0.0
        mujoco.mj_step(env.model, env.data)
        A_d[:, i] = (_get_state(env.data) - x_next_eq) / eps
    _set_state(env.model, env.data, x_eq)
    env.data.ctrl[0] = eps
    mujoco.mj_step(env.model, env.data)
    B_d[:, 0] = (_get_state(env.data) - x_next_eq) / eps
    A = (A_d - np.eye(n_x)) / dt
    B = B_d / dt
    Q = np.diag([1.0, 50.0, 50.0, 100.0, 0.1, 1.0, 1.0, 1.0])
    R = np.array([[1.0]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, x_eq


def collect_lqr_demos(env_cfg, eps_list=(4, 6), n_eps_per=100, max_steps=1000):
    """Collect LQR-controlled trajectories on the hard EPs."""
    all_obs, all_actions = [], []
    for ep in eps_list:
        env_tmp = TriplePendulumEnv(
            target_ep=ep, target_mode="fixed",
            init_mode="near_target", init_noise=0.0,
            max_episode_steps=max_steps, fall_grace_steps=0,
        )
        env_tmp.reset(seed=0)
        K, x_eq = compute_lqr_K(env_tmp, ep)
        print(f"[demos] EP{ep}: LQR K = {K.flatten().round(2)}")

        env_run = TriplePendulumEnv(
            target_ep=ep, target_mode="fixed",
            init_mode=str(env_cfg.get("init_mode", "near_target")),
            init_noise=float(env_cfg.get("init_noise", 0.05)),
            max_episode_steps=max_steps, fall_grace_steps=0,
        )
        success_count = 0
        for trial in range(n_eps_per):
            obs, _ = env_run.reset(seed=ep * 10000 + trial)
            survived = 0
            for step in range(max_steps):
                x = _get_state(env_run.data)
                dx = x - x_eq
                for i in [1, 2, 3]:
                    dx[i] = _wrap(dx[i])
                u = float(-(K @ dx)[0])
                u = max(-1.0, min(1.0, u))
                all_obs.append(obs.copy())
                all_actions.append([u])
                obs, _, done, trunc, _ = env_run.step(np.array([u]))
                survived = step + 1
                if done or trunc:
                    break
            if survived >= 800:
                success_count += 1
        print(f"[demos] EP{ep}: {success_count}/{n_eps_per} LQR successes, "
              f"{len(all_obs)} transitions total so far")

    obs_arr = np.array(all_obs, dtype=np.float32)
    act_arr = np.array(all_actions, dtype=np.float32)
    print(f"[demos] total: {len(obs_arr)} (obs={obs_arr.shape}, act={act_arr.shape})")
    return obs_arr, act_arr


def bc_pretrain(model, obs, actions, n_epochs=30, batch_size=256, lr=3e-4):
    """Supervised pre-training of TQC actor on (obs, action) demos.
    Minimizes MSE between actor's deterministic output and LQR action.
    """
    device = model.device
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.tensor(actions, dtype=torch.float32, device=device)
    n = len(obs_t)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=lr)
    print(f"[bc] {n} samples, {n_epochs} epochs, batch {batch_size}, lr {lr}")
    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            b_obs = obs_t[idx]
            b_act = act_t[idx]
            # actor(obs, deterministic=True) returns the squashed mean action.
            pred = model.actor(b_obs, deterministic=True)
            loss = ((pred - b_act) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"[bc] epoch {epoch}: mse={np.mean(losses):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    load_path = cfg.get("load_model_path", "")
    if not load_path:
        raise ValueError("E1 requires load_model_path in config")
    resolved = str(ROOT / load_path) if not load_path.startswith("/") else load_path

    print(f"[bc-rl] Loading warm-start model: {resolved}")
    # Load on cuda if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TQC.load(resolved, device=device)
    print(f"[bc-rl] device={model.device}")

    bc_cfg = cfg.get("bc", {})
    n_eps_per = int(bc_cfg.get("n_eps_per", 100))
    n_bc_epochs = int(bc_cfg.get("n_epochs", 30))
    bc_batch = int(bc_cfg.get("batch_size", 256))
    bc_lr = float(bc_cfg.get("lr", 3e-4))

    print(f"\n[bc-rl] Phase 0: collecting LQR demos (n_eps_per={n_eps_per})")
    t0 = time.time()
    obs, actions = collect_lqr_demos(cfg["env"], eps_list=(4, 6), n_eps_per=n_eps_per)
    print(f"[bc-rl] demos in {time.time()-t0:.1f}s")

    print(f"\n[bc-rl] Phase 1: BC pre-training")
    t0 = time.time()
    bc_pretrain(model, obs, actions, n_epochs=n_bc_epochs, batch_size=bc_batch, lr=bc_lr)
    print(f"[bc-rl] BC done in {time.time()-t0:.1f}s")

    # Save BC-pretrained checkpoint
    bc_ckpt = ROOT / "checkpoints" / "_bc_pretrained.zip"
    bc_ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(bc_ckpt))
    print(f"[bc-rl] BC checkpoint saved: {bc_ckpt}")

    # Update config to point to BC-pretrained model, write tmp, and hand off to RL trainer
    cfg["load_model_path"] = str(bc_ckpt.relative_to(ROOT))
    tmp_cfg = Path(args.config).with_suffix(".bc.yaml")
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg, f)
    print(f"\n[bc-rl] Phase 2: handing off to RL fine-tune with {tmp_cfg}")
    rl_train_main(str(tmp_cfg))


if __name__ == "__main__":
    main()

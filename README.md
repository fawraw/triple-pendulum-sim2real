# Triple Inverted Pendulum, Sim2Real RL

**Goal:** First demonstration of all 56 equilibrium transitions of a physical triple inverted pendulum on a cart, controlled by a sim-to-real reinforcement-learning policy, without precomputed trajectories or system-specific feedforward controllers.

| Bottom equilibrium (DDD) | Top equilibrium (UUU) |
|:---:|:---:|
| ![DDD](assets/pose_DDD.png) | ![UUU](assets/pose_UUU.png) |

## Why this matters

A triple inverted pendulum on a cart has 8 equilibrium configurations (each link Up or Down: 2³). Moving between any two of them, 8 × 7 = **56 transitions**, is the most general control benchmark that exists for this system.

| Author | System | Method | Sim2Real | Equilibria covered |
|---|---|---|---|---|
| Graichen et al. (Automatica 2013) | Triple cart-pole | LQR + 56 precomputed trajectories | n/a | 8 EPs, 56 transitions |
| Baek et al. (EAAI 2024) | Triple cart-pole | Model-free RL on hardware | ❌ trained on hw | 1 (swing-up to top) |
| Cambridge (Robotica 2026) | Underactuated triple | SAC + curriculum | ✅ | 1 (balance at top) |
| MDPI Machines (2025) | Double cart-pole | Sim2Real RL | ✅ | 4 EPs, 12 transitions |
| **This project** | **Triple cart-pole** | **Sim2Real RL (TQC)** | **✅** | **8 EPs, 56 transitions** |

The intersection (triple pendulum, Sim2Real RL, all 56 transitions) has never been demonstrated in the literature as of May 2026.

## Quickstart

```bash
git clone https://github.com/fawraw/triple-pendulum-sim2real.git
cd triple-pendulum-sim2real
./scripts/setup_env.sh                # creates .venv with MuJoCo, Gymnasium, SB3, TQC
source .venv/bin/activate

# Sanity check the environment
MUJOCO_GL=osmesa python -m sim.envs.triple_pendulum_env

# Train milestone 2 (stabilize UUU). Logs to ./mlruns by default.
MUJOCO_GL=osmesa python -m training.train_m2_upright

# Render a deterministic rollout of a trained policy
MUJOCO_GL=osmesa python scripts/eval_policy.py \
    --checkpoint checkpoints/<run_name>/final.zip --out assets/eval.mp4
```

The `setup_env.sh` script needs Python 3.10 or newer. On macOS install
`python@3.11` first via Homebrew. To log experiments to a remote MLflow
tracking server instead of the local `mlruns/` folder, export
`MLFLOW_TRACKING_URI` before training.

See [docs/roadmap.md](docs/roadmap.md) for the full milestone plan and
[docs/literature/state_of_the_art.md](docs/literature/state_of_the_art.md)
for the gap analysis backing the project.

## Approach

1. **Modeling.** High-fidelity MuJoCo XML of the cart-pole-pole-pole system, parameterized by physical constants.
2. **Single-EP stabilization.** Train TQC to stabilize each of the 8 equilibrium points in simulation.
3. **Transition policy.** Train a single conditional policy that takes a target EP as input and reaches it from any starting state.
4. **Domain randomization.** Friction, mass, latency, sensor noise, motor backlash randomized during training to bridge the reality gap.
5. **Hardware build.** ~600 to 900 CHF open-source BOM, V-slot rail, brushless motor, AS5048A magnetic encoders, STM32 real-time loop.
6. **Sim2Real transfer.** Deploy the policy on real hardware. Measure success rate per transition over N trials.
7. **Publication.** arXiv preprint, conference submission (CoRL, ICRA, RSS, NeurIPS Sim2Real workshop), open-source release.

## Tech stack

- Simulation: **MuJoCo** + **Gymnasium**
- RL: **Stable-Baselines3** with **TQC** (Truncated Quantile Critics)
- Backend: **PyTorch**
- Experiment tracking: **MLflow** (self-hosted)
- Real-time comms: **ZeroMQ** between PC (policy) and STM32 (low-level loop)
- Monitoring: **Grafana** + **InfluxDB**
- Publication: **GitHub** + **arXiv** + **Zenodo**

## Status

| Milestone | Status | Date |
|---|---|---|
| 0. Literature gap confirmed | ✅ | 2026-05-08 |
| 1. MuJoCo model, 3 links on cart | ✅ | 2026-05-08 |
| 2. Stable upright in sim (TQC) | 🟡 partial | 2026-05-08 |
| 3. All 8 EPs stabilized in sim | ⬜ | |
| 4. 56 transitions in sim | ⬜ | |
| 5. Domain randomization, robustness | ⬜ | |
| 6. Hardware v1 assembled | ⬜ | |
| 7. First Sim2Real swing-up | ⬜ | |
| 8. All 56 transitions on hardware | ⬜ | |
| 9. arXiv preprint | ⬜ | |
| 10. Conference submission | ⬜ | |

### M2 first results

200K-step TQC run (network 128 by 128, 3 critics, 20 quantiles, init noise 0.05 rad
near UUU). Mean episode length over 20 deterministic eval rollouts: 824 steps
out of 1000 (about 16 seconds at 50 Hz), with a maximum of 1000 steps reached on
several seeds. The acceptance threshold for M2 is 1000 steps over 80% of
rollouts; we are not there yet, but the pipeline is fully validated end to end.

![learning curve](assets/learning_curve_m2.png)

The eval rollout below shows the policy holding the upright configuration for
the full 20-second episode, sliding the cart along the rail to absorb angular
momentum.

![eval final pose](assets/eval_m2_upright.png)

A 20-second video of this rollout is at
[assets/eval_m2_upright.mp4](assets/eval_m2_upright.mp4).

## Repository layout

```
sim/
  models/           MuJoCo XML files
  envs/             Gymnasium environments
training/
  configs/          TQC hyperparameters per milestone
  scripts/          Training launchers
hardware/
  cad/              Mechanical design files (FreeCAD, STEP)
  firmware/         STM32 real-time controller
  bom/              Bill of materials
scripts/            Utilities (eval, video, profiling)
docs/
  literature/       Annotated bibliography of related work
assets/             Figures, videos, demo media
```

## Reproducibility

- Every training run is logged to MLflow with: code commit hash, seed, full config, learning curves, eval video.
- MuJoCo XML files are versioned alongside training scripts.
- Hardware BOM, firmware, CAD, and assembly instructions will be released with the paper.

## Citation (placeholder)

```bibtex
@misc{said2026triplependulum,
  author = {Saïd, Farid},
  title  = {Sim-to-Real Reinforcement Learning for All 56 Equilibrium Transitions of a Triple Inverted Pendulum},
  year   = {2026},
  url    = {https://github.com/fawraw/triple-pendulum-sim2real}
}
```

## License

MIT (code) + CERN-OHL-W v2 (hardware) + CC-BY 4.0 (docs).

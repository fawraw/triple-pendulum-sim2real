# Triple Inverted Pendulum, Sim2Real RL

[![CI](https://github.com/fawraw/triple-pendulum-sim2real/actions/workflows/ci.yml/badge.svg)](https://github.com/fawraw/triple-pendulum-sim2real/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/sim-MuJoCo%203.x-green)](https://mujoco.org/)
[![Wiki](https://img.shields.io/badge/docs-wiki-blueviolet)](https://github.com/fawraw/triple-pendulum-sim2real/wiki)

**Goal:** First demonstration of all 56 equilibrium transitions of a physical triple inverted pendulum on a cart, controlled by a sim-to-real reinforcement-learning policy, without precomputed trajectories or system-specific feedforward controllers.

| Bottom equilibrium (DDD) | Top equilibrium (UUU) |
|:---:|:---:|
| ![DDD](assets/pose_DDD.png) | ![UUU](assets/pose_UUU.png) |

## Why this matters

A triple inverted pendulum on a cart has 8 equilibrium configurations (each link Up or Down: 2³). Moving between any two of them — 8 × 7 = **56 transitions** — is the most general control benchmark for this system.

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

# Run unit tests
pytest

# Train milestone 2 (stabilize UUU). Logs to ./mlruns by default.
MUJOCO_GL=osmesa python -m training.train_m2_upright \
    --config training/configs/m2_upright_tqc.yaml

# Render a deterministic rollout of a trained policy
MUJOCO_GL=osmesa python scripts/render_rollout.py \
    --policy checkpoints/<run_name>/final.zip --ep 7 --out rollout.mp4
```

To log experiments to a remote MLflow server: `export MLFLOW_TRACKING_URI=...`.

For training pipeline details, n8n orchestration, and configs, see the [project wiki](https://github.com/fawraw/triple-pendulum-sim2real/wiki).

## Pipeline architecture

The training stages run unattended on a dedicated host. n8n decides what to do after each stage finishes — advance to the next, retry, or escalate.

```mermaid
flowchart LR
    A["Training script<br/>train_m2/m3/m4_*.py"] -->|writes results.json| B[pipeline_notifier.py]
    B -->|POST webhook| C{{n8n orchestrator}}
    C -->|"metric &ge; threshold"| D["Launcher API<br/>:8765/launch"]
    C -->|"metric &lt; threshold"| E["Launcher API<br/>fallback config"]
    C -->|HUMAN_REVIEW| F[Telegram alert]
    D -->|tmux new-session| A
    E -->|tmux new-session| A
    A -.->|metrics| G[("MLflow<br/>10.1.4.230")]
```

| Stage | Pass criterion | On pass | On fail |
|---|---|---|---|
| M2 | `ep7_success_rate ≥ 0.80` | M3b | HUMAN_REVIEW |
| M3b | `overall_success_rate ≥ 0.75` | M4 | M3c |
| M3c | `overall_success_rate ≥ 0.75` | M4 | HUMAN_REVIEW |
| M4 | `overall_success_rate ≥ 0.80` (over 56 transitions) | HUMAN_REVIEW (M5) | HUMAN_REVIEW |

See [n8n-Orchestration](https://github.com/fawraw/triple-pendulum-sim2real/wiki/n8n-Orchestration) and [Training-Pipeline](https://github.com/fawraw/triple-pendulum-sim2real/wiki/Training-Pipeline) for the full configuration.

## Status

| Milestone | Status | Date |
|---|---|---|
| 0. Literature gap confirmed | ✅ | 2026-05-08 |
| 1. MuJoCo model, 3 links on cart | ✅ | 2026-05-08 |
| 2. Stabilize UUU in sim (TQC) | 🟡 partial | 2026-05-08 |
| 3. All 8 EPs stabilized in sim | 🟡 M3b-v2 training (env fix applied, ETA ~5h, ~$1.50) | 2026-05-10 |
| 4. 56 transitions in sim | ⬜ scaffolded | |
| 5. Domain randomization | ⬜ | |
| 6. Hardware v1 assembled | ⬜ | |
| 7. First Sim2Real swing-up | ⬜ | |
| 8. All 56 transitions on hardware | ⬜ | |
| 9. arXiv preprint | ⬜ | |
| 10. Conference submission | ⬜ | |

### Latest results

**M2 (UUU, 150K steps, [128,128]):** mean episode length 824/1000 over 20 deterministic eval rollouts, peak 1000. Pipeline validated end-to-end.

![learning curve](assets/learning_curve_m2.png)

**M3b (2M steps) — two parallel runs converge to ~67%:**

| EP | Config | CT 1018 (CPU, n_envs=1, grad=1) | RunPod (A5000, n_envs=8, grad=8) |
|:--:|:------:|:------------:|:------------:|
| 0 | DDD | 100% | 100% |
| 1 | DDU | 100% | 100% |
| 2 | DUD | 50% | 100% |
| 3 | DUU | 80% | 100% |
| 4 | UDD | **0%** | **0%** |
| 5 | UDU | 70% | 60% |
| 6 | UUD | **0%** | **0%** |
| 7 | UUU | 80% | 80% |
| **Overall** | | **67.5%** | **68%** |

> **Diagnosis (audit 2026-05-10):** the plateau is **structural, not capacity-bound**. The env's `_is_fallen()` used a single 0.6 rad threshold for all 3 links. EP4 (UDD) and EP6 (UUD) need link 1 vertical while links 2–3 hang — but the cart's stabilizing motion shakes the hanging links naturally past 0.6 rad → false-positive fall → -100 penalty → **policy can't learn**. EP0–3 work (link 1 stable when down). EP7 works (all targets at 0). EP4/EP6 stuck at 0%.
>
> **Fixes applied (`sim/envs/triple_pendulum_env.py`):**
> - Per-link fall threshold: 0.6 rad for links targeted UP, 1.5 rad for links targeted DOWN
> - `fall_grace_steps=20` in training config: 20 consecutive over-threshold steps before termination (~40ms grace period)
> - Reward: `ang_cost = 5*err[0]² + err[1]² + err[2]²` (link 1 weighted 5×)
> - `vel_cost` coefficient: 0.01 → 0.05
>
> **M3b-v2** (2M steps, all fixes, gradient_steps=8, n_envs=8) launched on RunPod A5000. ETA ~5h, ~$1.50.

**Cumulative training cost on RunPod so far:** ~$2 USD (M3b cloud + probes). Local CT 1018 free but slow (~12h/2M steps).

## Tech stack

| Layer | Tool |
|---|---|
| Simulation | MuJoCo 3.x + Gymnasium |
| RL algorithm | TQC (Truncated Quantile Critics) via sb3-contrib |
| Backend | PyTorch 2.4 (CPU on CT 1018, CUDA 12.4 on RunPod cloud) |
| Parallel envs | `SubprocVecEnv` with picklable thunks, n_envs=8 default |
| Experiment tracking | MLflow (self-hosted on CT 1016) |
| Pipeline orchestration | n8n (self-hosted on CT 1003) |
| Cloud GPU | RunPod (RTX A5000, $0.27/hr, network volume `tp-data` 50 GB) |
| Bot interface | Telegram `@TriplePendulumBot` (n8n polling, 11 commands) |
| Real-time control | ZeroMQ between policy PC and STM32 1 kHz loop (planned, M6+) |
| Monitoring | Grafana + InfluxDB (planned) |

## Infrastructure

```
┌────────────┐    n8n webhook     ┌──────────────┐
│ RunPod pod │  ─────────────►    │   n8n        │
│ A5000 GPU  │                    │  CT 1003     │ ──► Telegram bot
└────────────┘                    │              │     @TriplePendulumBot
       │ logs                     └──────────────┘
       ▼
┌────────────┐                    ┌──────────────┐
│ tp-data    │                    │   MLflow     │
│ network    │                    │  CT 1016     │
│ volume     │                    │              │
└────────────┘                    └──────────────┘
       ▲
       │  /workspace
┌────────────┐
│ CT 1018    │  pipeline_notifier ► n8n webhook ► launch next stage
│ launcher   │  (HTTP :8765)
└────────────┘
```

- **Local CPU training**: CT 1018 runs the launcher API; n8n triggers stages via the launcher when previous stage finishes.
- **Cloud GPU training**: RunPod pods clone the repo at boot, run training, push results via webhook. Network volume persists `mlruns/` and `results/` across pods.
- **Telegram bot**: `/status`, `/runpod`, `/launch m3b|m3c|m4`, `/kill`, `/cost`, `/pod_start`, `/pod_stop confirm` — see [n8n/triple_pendulum_bot.json](n8n/triple_pendulum_bot.json).
- **Operational doc**: [docs/runbook.md](docs/runbook.md) covers DNS-not-ready, MLflow zombies, secret rotation, cost guards.

## Cloud training (RunPod)

```bash
# Pre-requisite: RunPod account + network volume "tp-data" + template "tp-train"
# (see runpod/README.md for full setup)

# Launch via REST API (or use Telegram bot /launch m3b)
curl -X POST "https://rest.runpod.io/v1/pods" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "name": "tp-m3b",
      "templateId": "<tp-train template id>",
      "gpuTypeIds": ["NVIDIA RTX A5000"],
      "gpuCount": 1,
      "networkVolumeId": "<tp-data id>",
      "env": {
        "TP_AUTO_SHUTDOWN": "1",
        "TP_STAGE_MODULE": "training.train_m3_all_eps",
        "TP_STAGE_CONFIG": "training/configs/m3b_all_eps_tqc.yaml",
        "TELEGRAM_FALLBACK_BOT_TOKEN": "<your bot token>",
        "TELEGRAM_FALLBACK_CHAT_ID": "<your chat id>"
      }
    }'
```

The pod runs [`scripts/runpod_bootstrap.sh`](scripts/runpod_bootstrap.sh) which:
1. Waits for DNS, installs OS deps if absent.
2. Clones the repo at the latest `main` (or `TP_REPO_REF` if pinned).
3. Pre-installs `blinker` via pip to bypass the apt-distutils conflict.
4. Installs `requirements.txt` (keeps the image's torch 2.4.1+cu124 to match the host driver).
5. Spawns an idle watchdog (force-stops the pod after 30 min of GPU<5% — cost guard).
6. Runs the training stage. On completion, posts results JSON to n8n (with Telegram fallback if the LAN n8n is unreachable from cloud).
7. Auto-shuts down via the RunPod GraphQL API.

**Typical cost:** $0.27–0.40 per stage on A5000 ($0.30 for M3b 2M steps, $0.50–0.80 for M3c 4M).

## Repository layout

```
sim/
  envs/                Gymnasium environments
  models/              MuJoCo XML files
training/
  configs/             TQC hyperparameters per milestone (m2, m3b, m3c, m4)
  train_m{2,3,4}_*.py  Training entrypoints
  pipeline_notifier.py POSTs to n8n + writes results JSON
  pipeline_stages.json Stage transitions (read by n8n)
scripts/
  launcher_api.py      HTTP launcher for n8n to start training
  render_rollout.py    Render saved policy to MP4
  eval_policy.py       Per-EP evaluation
  plot_learning_curve.py
n8n/
  triple_pendulum_pipeline.json   Workflow definition (importable)
hardware/
  bom/                 Bill of materials (planned for M6)
docs/
  roadmap.md           Mirror of the wiki Roadmap (canonical: wiki)
  literature/          Annotated bibliography
tests/                 pytest unit tests (env, notifier, stages, launcher)
assets/                Figures and demo media
```

## Reproducibility

- Every training run is logged to MLflow with code commit hash, seed, full config, learning curves, and final eval metrics.
- MuJoCo XML files are versioned alongside training scripts.
- Pipeline state JSON files in `results/` are committed for permanent record (without secrets — see [`pipeline_notifier.py`](training/pipeline_notifier.py)).
- Hardware BOM, firmware, CAD will be released with the paper.

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

- **Code, configs, scripts:** [MIT](LICENSE) — current scope of this repo.
- **Hardware (BOM, CAD, firmware):** CERN-OHL-W v2 — will be added under `hardware/` and a `LICENSE-HARDWARE` file when M6 ships.
- **Docs (`docs/`, wiki):** CC-BY 4.0 — will be added under `LICENSE-DOCS` when the paper is released.

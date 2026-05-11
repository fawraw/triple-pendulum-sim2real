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

Training stages run unattended. n8n decides advance / retry / escalate. Cloud pods (RunPod) can't reach the LAN n8n, so they fall back to Telegram directly.

```mermaid
flowchart TB
    subgraph cloud["☁️ RunPod Cloud (A5000 GPU, $0.27/hr)"]
        POD["Training pod\ntrain_m3_all_eps.py"]
        VOL[("tp-data volume\n/workspace")]
    end
    subgraph lab["🏠 Lab Perso (LAN 10.1.4.x)"]
        LAUNCHER["Launcher API\nCT 1018 :8765"]
        N8N["n8n orchestrator\nCT 1003"]
        MLFLOW[("MLflow\nCT 1016 :5000")]
    end
    TG["📱 Telegram\n@TriplePendulumBot"]
    CLI["💻 fetch_results.py\n→ gist → read"]

    POD -->|results JSON| VOL
    POD -->|"Telegram fallback\n(n8n LAN-only)"| TG
    N8N -->|POST /launch| LAUNCHER
    LAUNCHER -->|tmux spawn| POD
    TG -->|"/launch /kill /status"| N8N
    CLI -->|"pod reads VOL\n→ GitHub gist"| VOL
```

### Probe → validate → full run

```mermaid
flowchart LR
    A["New fix\nhypothesis"] --> B["Probe 200K steps\nEP-fixed ~50min\n~$0.22"]
    B --> C{EP success\n≥ 50%?}
    C -->|YES ✅| D["Full run 2M steps\n~5h ~$1.50"]
    C -->|NO ❌| E["Refine or\ndig deeper"]
    E --> A
    D --> F["fetch_results.py\n~60s $0.01"]
    F --> G{overall\n≥ 75%?}
    G -->|YES| H["M4 transitions"]
    G -->|NO| I["M3c 4M steps\n[512,512]"]
```

| Stage | Pass criterion | On pass | On fail |
|---|---|---|---|
| M2 | `ep7_success_rate ≥ 0.80` | M3b | HUMAN_REVIEW |
| M3b | `overall_success_rate ≥ 0.75` | M4 | M3c |
| M3c | `overall_success_rate ≥ 0.75` | M4 | HUMAN_REVIEW |
| M4 | `overall_success_rate ≥ 0.80` (56 transitions) | M5 | HUMAN_REVIEW |

See [n8n-Orchestration](https://github.com/fawraw/triple-pendulum-sim2real/wiki/n8n-Orchestration) and [Training-Pipeline](https://github.com/fawraw/triple-pendulum-sim2real/wiki/Training-Pipeline) for the full config.

## Status

| Milestone | Status | Date |
|---|---|---|
| 0. Literature gap confirmed | ✅ | 2026-05-08 |
| 1. MuJoCo model, 3 links on cart | ✅ | 2026-05-08 |
| 2. Stabilize UUU in sim (TQC) | 🟡 partial | 2026-05-08 |
| 3. All 8 EPs stabilized in sim | 🟡 **8 pods parallel** (M3b-v4 A–H, multiple strategies, ETA ~15h CET) | 2026-05-11 |
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
> **Two bugs found and fixed (2026-05-10):**
>
> **Bug 1 — False-positive fall detection:** `_is_fallen()` used a single 0.6 rad threshold for all links. When the DOWN-targeted links swing past 34° (natural physics), the env killed the episode with -100 penalty. Policy learned to do nothing.
>
> ```
> BEFORE: any link > 0.6 rad  →  fall  →  -100
> AFTER:  UP-targeted link > 0.6 rad  →  fall  →  -100
>         DOWN-targeted link > 1.5 rad →  fall  →  -100  (much looser)
> ```
>
> **Bug 2 — Wrong reward focus:** reward always penalised link 1 (base/cart) 5× regardless of target. For EP4 (base DOWN, tip UP), the policy should focus on the tip — but the reward kept demanding perfect control of the hanging base.
>
> ```
> BEFORE:  5×err[0]² + 1×err[1]² + 1×err[2]²   (always base 5×)
> AFTER:   w[0]×err[0]² + w[1]×err[1]² + w[2]²  where w[i]=5 if link i is UP, 1 if DOWN
> ```
>
> **Validation probes** (200K steps EP-fixed, 2026-05-10 evening):
> - EP4-v3 with adaptive reward: **60%** (was 0% across 3 prior runs) ✅
> - EP6 with adaptive reward: **10%** (was 0%) — needs more steps
>
> **M3b-v3** (2M steps, adaptive reward, eval strict, `stage=M3b_v3`) launched on RunPod A5000, ETA ~5h.

### M3 debugging journey

```
Run              Steps  Overall  EP4   EP6   Notes
─────────────────────────────────────────────────────────────────────
M3 baseline      400K   42.5%    0%    0%    First attempt
M3b CPU          2M     67.5%    0%    0%    More steps, same plateau
M3b GPU          2M     68.0%    0%    0%    GPU (8×faster), same plateau
M3b-v2           2M     61.3%    0%    0%    Per-link fix, eval not strict → worse
Probe EP4-v3     200K   n/a      60%   n/a   ← adaptive reward fix validated!
Probe EP6        200K   n/a      n/a   10%   ← non-zero first time
M3b-v3          2M      60%     0%    0%    EP2 regression, EP4/EP6 still 0%
M3b-v4 A–H     2M×7    TBD     TBD   TBD   8 parallel pods, ETA ~15h CET
```

**M3b-v3 post-mortem:** Adaptive reward (UP=5, DOWN=1) introduced a new regression — EP2 dropped from 100% to 40% because removing the base-stability prior (`w_down=1`) hurt EPs where the base links correlate with upper-link stability. EP4/EP6 remain at 0% — the probe showed 60% with fixed-EP training but the full run allocates only 12.5% training time per EP.

**8 parallel pods (M3b-v4) running as of 2026-05-11:**

| Pod | Key changes | Hypothesis |
|---|---|---|
| A | `w_down=2` | EP2 regression: base prior too weak |
| B | Oversample EP4/EP6 ×3 | EP4/EP6: too little training time |
| C | Progress reward + grace | EP4/EP6: no dense gradient in failed episodes |
| D | M3c [512,512] 4M steps | Null: is it a capacity problem? |
| E | **A+B** | Combo of two best individual fixes |
| **F** | **A+B+C (full)** | Kitchen sink — highest chance of passing 75% |
| G | Oversample ×5 + A + C | More aggressive if ×3 isn't enough |
| H | A + C + `vel_cost=0.01` | Restore old vel_cost — tests EP7 regression cause |

**Cumulative training cost on RunPod:** ~$14 USD total (all runs + 8 new pods). CT 1018 free.

*Note on link numbering:* config labels like "UDD" read Top→Bottom (U=tip up, D=middle down, D=base down); code uses bit 0 = link 1 = base (cart-attached). Both are internally consistent — see [[System-Explained]] for details.

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
  launcher_api.py         HTTP launcher API for n8n to start training (:8765)
  runpod_bootstrap.sh     Cloud pod boot script (DNS wait, pip fix, idle watchdog)
  report_to_telegram.py   Exfil results from cloud pods via Telegram (SSH fallback)
  tp_status.sh            All-in-one pipeline status (launcher + MLflow + results)
  render_rollout.py       Render saved policy to MP4
  eval_policy.py          Per-EP evaluation
runpod/
  Dockerfile              CUDA image definition (PyTorch 2.4+MuJoCo+osmesa)
  README.md               Operator setup guide, GPU types, cost table, gotchas
n8n/
  triple_pendulum_pipeline.json  Orchestration workflow (importable)
  triple_pendulum_bot.json       Telegram bot (11 commands, polling-based)
hardware/
  bom/                    Bill of materials (planned for M6)
docs/
  roadmap.md              Mirror of the wiki Roadmap (canonical: wiki)
  runbook.md              Operational troubleshooting (8 failure scenarios)
  launcher_api.service    Systemd unit template (KillMode=process)
  literature/             Annotated bibliography
tests/                    pytest unit tests (env, notifier, stages, launcher, m4)
results/                  Per-run JSON snapshots from pipeline_notifier (committed)
CHANGELOG.md              Dated changelog
.githooks/
  pre-commit              Secret-pattern scanner, chains to global Lab Perso hook
assets/                   Figures and demo media
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

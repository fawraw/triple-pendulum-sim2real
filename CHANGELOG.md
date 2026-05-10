# Changelog

All notable changes to this project. Format: [Keep a Changelog](https://keepachangelog.com/), versioning by date because this is research code with no semver yet.

---

## 2026-05-10 — Audit-driven RL fixes + cloud expansion

This was a heavy session focused on understanding why M3b plateau'd at 67–68% across two independent runs (CPU and GPU), and on operationalizing the cloud training stack.

### Diagnosed: M3b plateau is structural, not capacity

A code-reviewer audit identified that `_is_fallen()` used a single 0.6 rad threshold for all 3 links, regardless of whether the link is targeted Up or Down. For EPs where link 1 is up and links 2–3 hang (EP4 = UDD, EP6 = UUD), the cart's stabilizing motion on link 1 shakes the hanging links naturally past 0.6 rad → false-positive fall → `-100` penalty → policy literally cannot learn those configurations. EPs with all-targets-Up (EP7) or all-targets-Down (EP0–3) work fine because their thresholds are meaningful.

Confirmed by data: across two M3b runs (CPU `gradient_steps=1` and GPU `gradient_steps=8`, otherwise identical hyperparameters), EP4/EP6 stuck at 0% in both. Capacity (gradient density, network size) was not the bottleneck.

### Added — env fixes

- **Per-link fall threshold** (`sim/envs/triple_pendulum_env.py`): `FALL_THRESHOLD_UP_RAD=0.6` for links targeted at 0 rad, `FALL_THRESHOLD_DOWN_RAD=1.5` for links targeted at π. `_fall_thresholds()` helper picks the right one per link based on the current `target_ep`.
- **Reward weighting**: `ang_cost = 5*err[0]² + err[1]² + err[2]²` (link 1 weighted 5× as the structural pivot). `vel_cost` coefficient bumped 0.01 → 0.05.
- **Soft termination** (opt-in via `fall_grace_steps` env param): instead of failing on a single step over threshold, tolerate N consecutive steps. Default 0 (legacy strict, used by eval). Probe v2 and `m3b_all_eps_tqc.yaml` use 20 (~40ms grace at 50Hz). Eval always uses default=0 for fair scoring.
- **7 new `tests/test_env.py` tests** covering all new env features: `test_per_link_threshold_up_is_tight`, `test_per_link_threshold_down_is_loose`, `test_ep4_hanging_links_do_not_trigger_fall` (reproduces and validates the pre-fix bug), `test_fall_grace_steps_delays_termination`, `test_fall_counter_resets_on_recovery`, `test_reset_clears_fall_counter`, `test_reward_link1_weighted_5x`.

### Added — cloud GPU training (RunPod)

- `runpod/Dockerfile` (PyTorch CUDA 12.1 + MuJoCo + osmesa).
- `scripts/runpod_bootstrap.sh`: idempotent, DNS-wait, blinker pre-install (bypasses apt distutils conflict), keeps the image's torch 2.4.1+cu124 to match the host driver. Includes:
  - `trap EXIT` so a script-level failure doesn't kill the container — operator can SSH and inspect.
  - `MUJOCO_GL=osmesa` and `MLFLOW_TRACKING_URI=file:/workspace/mlruns` defaults.
  - Idle watchdog spawned in background (force-stop after `TP_IDLE_SHUTDOWN_MIN` min of GPU<5% — cost guard).
  - SIGTERM trap that flushes logs before exit.
  - `TP_REPO_REF` for pinning to a commit SHA / tag (reproducibility).
- `runpod/README.md`: full operator setup, GPU recommendations, cost table.
- Network volume `tp-data` (50 GB, US-IL-1, $3.50/mo) persists `mlruns/` and `results/` across pods.

### Added — Telegram bot (n8n)

- `n8n/triple_pendulum_bot.json`: Schedule trigger (30s) → `getUpdates` long-poll → dispatch via Code node → reply via HTTP. Uses `staticData.lastOffset` for offset tracking.
- 11 commands: `/status`, `/runpod`, `/ct1018`, `/mlflow [n]`, `/gpu`, `/cost`, `/launch m2|m3b|m3c|m4`, `/kill ct1018|runpod`, `/pod_start`, `/pod_stop confirm`, `/help`.
- HTML parse mode + escape — fixes Markdown injection from underscored session names.
- Silent drop on unauthorized chat_ids (no leak of bot existence).
- `/launch` and `/kill` send an immediate "⏳" ack before the long API call.
- `setMyCommands` registers the menu so commands appear in the Telegram UI auto-completion.

### Added — Launcher API endpoint

- `POST /kill` on `scripts/launcher_api.py`: kills any running training tmux session + sends `SIGTERM` to leftover python processes. Auth via `hmac.compare_digest`. Used by the bot's `/kill ct1018`.

### Added — Operational scripts

- `scripts/report_to_telegram.py`: read `/workspace/results/*.json` + `bootstrap.log` DONE blocks, post HTML summary to Telegram. Used to exfil results from cloud pods when the LAN-only n8n webhook is unreachable.

### Added — Configs

- `training/configs/probe_ep4_tqc.yaml`: 200K-step EP4-fixed probe (validates audit hypothesis #1).
- `training/configs/probe_ep4_v2_tqc.yaml`: same + `fall_grace_steps=20` (audit hypothesis #2).

### Added — Configs (afternoon)

- `training/configs/probe_ep4_v2_tqc.yaml`: 200K EP4-fixed + `fall_grace_steps=20` (validates soft-termination hypothesis in isolation from per-link fix).

### Launched — M3b-v2 (afternoon)

- M3b-v2 launched on RunPod A5000 (pod `r0ghhvrt529gy0`): 2M steps, [256,256], n_envs=8, gradient_steps=8, fall_grace_steps=20, per-link threshold, reward weighting. ETA ~1h20min from 16:00 CET. Expected to break the EP4/EP6=0% plateau that persisted across 3 prior runs.

### Fixed — report_to_telegram.py regex (afternoon)

- `collect_done_blocks()` regex changed from non-greedy `.*?` with `re.DOTALL` (could merge consecutive training runs) to `re.split` on `^DONE in` line boundaries. Each DONE block now correctly isolated to one training run.

### Changed — Existing configs

- `training/configs/m3b_all_eps_tqc.yaml`, `m3c`, `m4_transitions`: `gradient_steps: 1 → 8` to match `n_envs=8` (preserves 1-grad-per-env-transition density).
- `training/configs/m3b_all_eps_tqc.yaml`: added `fall_grace_steps: 20` in env section (applied during training; eval always uses strict default=0).

### Fixed — Operational bugs (multiple in-session live fixes)

| | |
|---|---|
| n8n SSL TLS handshake fail on `n8n.faji.co` | switched pipeline_notifier to HTTP-direct `http://10.1.4.226:5678/...` until NPM cert is fixed |
| n8n Code node `Module 'crypto' is disallowed` | replaced `timingSafeEqual` with pure-JS constant-time `safeEqual()` shim |
| `systemctl restart tp-launcher` killed running training | added `KillMode=process` to the systemd unit so detached tmux sessions survive |
| MLflow zombie RUNNING runs after crashes | manually marked as FAILED via `/api/2.0/mlflow/runs/update`; janitor cron in backlog |
| Bootstrap pip install collided with apt-installed `blinker` | pre-install blinker via pip (`--ignore-installed --no-deps`), then standard pip install |
| Bootstrap pulled torch 2.11+cu13 (driver too old) | rely on the image's pre-installed torch 2.4.1+cu124 by NOT using `--ignore-installed` for the full install |
| DNS not ready at container start | added `until getent hosts github.com; do sleep 2; done` before any network operation |

### Audit history (today, four iterations)

1. **Round 1** — code-level: MLflow callback resilience, secret leak in JSON, launcher hardening, missing `__init__.py`.
2. **Round 2** — operational: PrivateTmp cgroup bug, idempotency races, webhook retry strategy, secret-injection drift.
3. **Round 3** — production: launcher RestartSec budgeting, signal handling, log persistence, cost guards.
4. **Round 4** — RL training: per-link fall threshold (the actual blocker), reward weighting, gradient density misconfig.

### Added — Wiki pages (afternoon)

- `Bot.md` (new): Telegram bot architecture, all 11 commands with backend mapping, auth, gotchas.
- `Cloud-Training.md` (new): RunPod setup, lifecycle diagram, common issues (DNS, blinker, torch CUDA, region lock, SSH flakiness, cost guard), reading results back.
- Updated `Home.md`, `Results.md`, `Roadmap.md`, `Training-Pipeline.md` with current state.

### Added — Skills (afternoon)

- `~/.claude/skills/runpod/` — RunPod cloud GPU management (spawn, stop, GPU util, balance, SSH, gotchas).
- `~/.claude/skills/mlflow/` — MLflow queries (list runs + ETA, per-EP success, cleanup zombies).
- `~/.claude/skills/triple-pendulum/` — Project-specific ops (launch stages, baseline table, bot commands, architecture).

### Evening session (2026-05-10)

- **Adaptive reward weighting** — the morning fix weighted `5×err[0]²` (link 1/base, always). For EP4 (base DOWN, tip UP), this incentivised controlling the hanging base instead of the inverted tip. Fix: `weights = where(target≈0, 5.0, 1.0); ang_cost = sum(weights * err²)`. UP-targeted links get 5×, DOWN-targeted get 1×. See commit `620994c`.
- **Pre-M3b-v3 audit fixes** (commit `05b57b7`):
  - `per_ep_eval()` hardcoded `fall_grace_steps=0` (was inheriting 20 from config → eval not strict)
  - `EvalCallback` uses `eval_env_cfg = {**env_cfg, "fall_grace_steps": 0}`
  - `stage: M3b → M3b_v3` for clean MLflow tracking
- **Probe EP4-v3** (adaptive reward): EP4 **60%** ← breakthrough from persistent 0%
- **Probe EP6** (adaptive reward): EP6 **10%** ← first non-zero
- **M3b-v3 launched** on RunPod A5000 (pod `1d38awsk08hzdj`), commit pinned `1f67901`. Expected overall ~78-82%.
- **`scripts/fetch_results.py`** — automates gist-based result read: spawn pod → read volume → update gist → read → delete. `python3 scripts/fetch_results.py --milestone M3b_v3`
- **Skills** — `training-analyst` (new), `triple-pendulum` (CI check + probe workflow), `runpod`/`mlflow` already complete
- **Wiki** — `System-Explained.md` (new, accessible guide), `Simulation-Model.md` enriched (EP difficulty table with ⭐, reward function table, before/after fix docs)
- **README** — M3 debugging journey table, dual Mermaid diagrams (infrastructure + probe→validate→launch workflow)

Total commits today: ~40. See `git log --since=2026-05-09` for the full list.

### Known gaps (carried over)

- HTTPS for `n8n.faji.co` (NPM cert) — affects only n8n webhook from cloud and Telegram webhooks
- Auto-sync MLflow runs from cloud network volume to CT 1016 MLflow — currently manual via spawned readonly pod
- `pipeline_stages.json` ↔ n8n STAGES dict drift test (committed but not in CI yet)
- Static IP for CT 1018 in UDM-ENT
- Move tracked binary blobs (`assets/*.mp4`, `*.png`) to git-lfs

---

## 2026-05-09 — Pipeline automation + GitHub polish

- n8n orchestration workflow (Schedule + Webhook + Code + Telegram).
- Launcher HTTP service on CT 1018 (`scripts/launcher_api.py`, port 8765).
- Pipeline notifier (`training/pipeline_notifier.py`) — POSTs JSON to n8n, falls back gracefully.
- Wiki initialized (Home, Roadmap, Simulation-Model, Training-Pipeline, n8n-Orchestration, Results, Hardware).
- README revamp with badges, mermaid pipeline diagram, milestone status.
- 5 pytest modules (env, notifier, stages, launcher, M4 transitions).
- GitHub polish: issue/PR templates, CONTRIBUTING.md.
- Pre-commit hook chained to global Lab Perso secret scanner.
- CI workflow runs pytest + smoke training (200 steps).
- M2 (UUU) baseline trained: mean episode length 824/1000.
- M3 baseline (400K steps): EP0/1/2/3 = 100/100/50/80%, EP4–7 = 0–10%. Overall 42.5%.

---

## 2026-05-08 — Project genesis

- Literature gap audit (8 papers reviewed): no prior work covers triple-pendulum + Sim2Real RL + all 56 transitions.
- MuJoCo XML model: cart on 2 m rail, 3 hinged links (mass 0.10 / 0.08 / 0.05 kg, length 0.25 m each), motor gear 15.
- Gymnasium env: 16-dim obs (joint state + target one-hot), 1-dim continuous action, absolute angle convention.
- TQC training scripts (`training/train_m{2,3,4}_*.py`).
- M2 → M3 → M4 → M5 → M6 milestone roadmap.

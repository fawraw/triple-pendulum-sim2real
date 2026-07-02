# Roadmap

> **Canonical version:** the [wiki Roadmap page](https://github.com/fawraw/triple-pendulum-sim2real/wiki/Roadmap) is the source of truth and is updated whenever a milestone advances. This file is kept in sync for offline / source-checkout readers; if they ever diverge, trust the wiki.

Milestones are sequential: each is a prerequisite for the next. Acceptance criteria are quantitative to avoid subjective pass/fail.

## M0: Literature gap audit

**Goal:** Confirm the combination (triple pendulum + Sim2Real RL + all 56 transitions) is unclaimed.

**Done:** May 2026. Eight papers reviewed; no prior work covers the full intersection. See [`docs/literature/state_of_the_art.md`](literature/state_of_the_art.md).

---

## M1: MuJoCo simulation model

**Goal:** A physically accurate cart-pole-pole-pole model that can be stepped headless.

**Acceptance:** `MUJOCO_GL=osmesa python -m sim.envs.triple_pendulum_env` runs without error and prints plausible observations.

**Done:** May 2026.

| Component | Value |
|---|---|
| Cart mass | 0.5 kg |
| Rail length | 2.0 m (slide range: -0.95 to 0.95 m) |
| Link 1 length / mass | 0.25 m / 0.10 kg |
| Link 2 length / mass | 0.25 m / 0.08 kg |
| Link 3 length / mass | 0.25 m / 0.05 kg |
| Motor gear ratio | 15 |
| Control range | -1 to 1 (normalized) |
| Simulation timestep | 0.01 s (100 Hz internal, 50 Hz policy) |

---

## M2: Stabilize EP7 (UUU) in simulation

**Goal:** A TQC policy that keeps all three links pointing up for ≥ 80% of deterministic eval episodes.

**Acceptance:** Mean episode length ≥ 800 / 1000 over 20 deterministic rollouts, with ≥ 16/20 reaching 800+ steps.

**Status:** Partial (May 2026). Best run: mean length 824/1000, peak 1000. Pipeline validated end-to-end.

**Config:** `training/configs/m2_upright_tqc.yaml` — 150K steps, [128, 128], 3 critics, 20 quantiles.

---

## M3: Conditional policy across all 8 EPs

**Goal:** A single TQC policy that reads a target EP one-hot from the observation and stabilizes any of the 8 equilibria.

**Acceptance:** `overall_success_rate ≥ 0.75` over 80 rollouts (10 per EP). This 0.75 is the canonical threshold (matches the code and `pipeline_stages.json`); if any wiki page shows 0.80, the code/pipeline value wins.

**Status:** Closed at 72.5% (M3b-v6 cloud, 2026-05-14) — all 8 EPs non-zero in random mode (baseline was 42.5% at 400K steps). The 0.75 threshold was an internal goal and was not met; 72.5% is accepted as the scientific milestone (every equilibrium stabilized at least once). Caveat: the 72.5% run lived on the RunPod `tp-data` volume and is not in the persistent MLflow (best reproducible run there is 67.5%).

**Pipeline:** M3b (2M steps, [256,256]) → if `overall_success_rate < 0.75` → M3c (4M steps, [512,512]). Automated via n8n.

---

## M4: 56 transitions in simulation

**Goal:** A policy that starts from any equilibrium (or random state) and reaches a commanded target EP.

**Scope:** 8 × 7 = 56 directed transitions. Each tested with 5 deterministic rollouts; success = policy reaches the target EP within `max_episode_steps` and holds it for ≥ 0.5 × `max_episode_steps`.

**Acceptance:** ≥ 80% success rate aggregated over all 56 transitions.

**Status (2026-06-25):** In progress via a **two-stage hand-off** (swing-up -> M3 stabilizer). De-risking on the single transition DDD->UDD found and fixed 3 eval/config bugs that had made prior results uninterpretable (step-1 episode death, random-pair eval, uninformative overall -- see `docs/m4_findings.md`). With a cart barrier, episodes now survive and the swing-up reaches UDD (~0.21 rad) but inconsistently and without holding; the hand-off controller fires but M3's catch basin is only ~0.1 rad / near-zero velocity, so it drops the fast delivery. Next: a wider-basin catcher + a soft swing-up delivery, iterated on GPU.

**Notes:** Requires swing-up in addition to stabilization. Single-transition swing-up configs use `target_mode: fixed` + `init_mode: bottom`; the full 56-transition config uses `target_mode: transition` (matching `per_transition_eval`, which pins each pair via `reset(options=...)`). Env knobs: `cart_cost_coef`, `cart_barrier_coef`, `cart_limit`, `progress_reward_coef`.

---

## M5: Domain randomization

**Goal:** Make the simulation-trained policy robust to physical parameter uncertainty.

**Randomized parameters:** link masses (±20%), link lengths (±5%), motor friction (0–0.1), control latency (0–20 ms), sensor noise (Gaussian, σ = 0.01 rad).

**Acceptance:** Same success criteria as M4, maintained after adding all randomization ranges.

---

## M6: Hardware assembly

**Goal:** A working physical prototype matching the simulation model within 10% on all mechanical parameters.

**Deliverables:**
- V-slot 2040 rail, 1.0 m usable travel
- ODrive-based brushless motor drive
- 3 × AS5048A magnetic encoders (absolute, 14-bit)
- STM32 real-time loop at 1 kHz, ZeroMQ bridge to policy PC at 50 Hz
- Measured vs simulated step response comparison

---

## M7: First Sim2Real result

**Goal:** At least one equilibrium transition demonstrated on hardware without any hardware-specific fine-tuning.

**Acceptance:** EP7 (UUU) stabilization: hold for ≥ 10 s in ≥ 3/5 trials.

---

## M8: All 56 transitions on hardware

**Goal:** All 56 transitions demonstrated on the physical system.

**Acceptance:** ≥ 70% success rate over 5 trials per transition (350 trials total), logged with video evidence.

---

## M9: arXiv preprint

**Scope:** Methods, results, BOM, code release. Target venues: CoRL, ICRA, RSS, or NeurIPS Sim2Real workshop.

---

## M10: Conference submission

Submit the version that is closest to the next deadline, with the workshop track as a fast-path fallback.

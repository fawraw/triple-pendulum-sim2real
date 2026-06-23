# Project strategy: a realistic path to publication

> Framing note, 2026-06-23. Where the project actually stands versus the
> publication goal, the critical path, and the route a solo effort can realistically
> take to a preprint. Honest assessment, not a pitch.

## The claim vs reality

The headline claim (see README) is: *the first demonstration of all 56 equilibrium
transitions of a **physical** triple inverted pendulum on a cart, controlled by a
sim-to-real RL policy, without precomputed trajectories or system-specific
feedforward controllers.* That is milestone **M8**.

Honest milestone status:

| Milestone | State | Distance to the claim |
|---|---|---|
| M0-M1 literature + MuJoCo model | done | -- |
| M2 stabilize UUU (sim) | partial (824/1000) | roughly acquired |
| M3 stabilize all 8 EPs (sim) | closed at **72.5%**, and **not reproducible** (the run lived on the now-gone RunPod volume; best reproducible local run is 67.5%) | necessary baseline, modest |
| **M4 the 56 transitions (sim)** | **not started**: the DDD->UUU smoke scored 0% even on the trained pair (200K steps), swing-up not demonstrated | **the wall** |
| M5 domain randomization | not started | prerequisite for sim2real |
| M6 hardware assembled | not started | real money + weeks of build |
| M7 first sim2real transfer | not started | the make-or-break bet |
| M8 all 56 on hardware | not started | = the claim |

**Bottom line:** the project is at simulation stabilization (~M3). The paper targets a
hardware demonstration (M8). The hardest and most uncertain parts are all ahead, and
the "72.5% scientific milestone" is real but modest -- and not even replayable today.

## Critical path and unknowns

```
M4 (sim, 56 transitions) --> M5 (domain rand) --> M6 (hardware) --> M7 (1 sim2real) --> M8 (56 on hardware)
   ^ everything depends on it    incremental         $$ + time         the real risk        = the claim
   RISK #1: unproven                                 RISK #3           RISK #2
```

- **Risk #1 -- M4 in simulation.** If a single RL policy cannot learn all 56 transitions
  in simulation, there is no paper, whatever the target. It is the front door to
  everything downstream, and the smoke test showed it is not free.
- **Risk #2 -- the sim2real transfer itself.** A triple pendulum is chaotic; sim->real
  transfer on such systems is notoriously fragile. M7 alone can cost months
  (system identification, aggressive domain randomization).
- **Risk #3 -- the hardware.** V-slot rail, ODrive + brushless motor, 3x AS5048A
  encoders, STM32 1 kHz loop, ZeroMQ bridge. Real money plus weeks of build and debug,
  solo.

## Three realistic publication targets

| Target | What it needs | Cost / risk | Novelty |
|---|---|---|---|
| **A. Full claim** (56 on hardware) | M4 -> M8 | very high, months, hardware $$ | maximal (never done) -- CoRL / ICRA / RSS |
| **B. Sim-only** (56 transitions via a single learned RL policy, no precomputed trajectories, in simulation) | M4 + M5 | moderate, **no hardware** | defensible (Graichen 2013 used 56 precomputed trajectories; no prior work learns all 56 with one policy) -- NeurIPS Sim2Real workshop / L4DC |
| **C. Methods / negative** (curriculum + reward design for multi-equilibrium swing-up) | the analysis already underway | low | weak as a standalone contribution |

## Recommendation

**Target B (sim-only) as the first genuinely publishable milestone, and treat the
hardware (A) as a separate go/no-go decided after M4.** Rationale:

1. B is achievable solo with no hardware spend, and it is the prerequisite for A
   anyway -- nothing is lost by targeting it first.
2. B is already an honest, citable contribution (a title without "sim2real":
   *RL of a single policy for all 56 transitions of a triple cart-pole, in simulation*).
3. Hardware (A) is a commitment of a different order (time + money + the sim2real
   risk). Open it only if M4/M5 succeed and the result is motivating.

Whatever the choice, **the immediate gate is M4 in simulation.** Everything depends on
proving it is tractable before investing in a long run.

## Immediate next step

De-risk M4 with two cheap probes (~1h GPU each) before committing to an expensive
full run:

- **Probe A (easy):** a single-link transition (e.g. DUU -> UUU, only link 1 to raise),
  500K steps, `target_mode: transition`, warm-started from the M3 policy.
- **Probe B (hard):** the full DDD -> UUU swing-up, 500K steps.

The comparison decides the direction:

- A learns, B does not -> swing-up difficulty -> build a **difficulty curriculum**
  (1-link -> 2-link -> 3-link swing-up, warm-starting each stage from the previous).
- Neither learns -> the transition reward is insufficient -> add an **energy-based term**
  to drive swing-up, not just an angle-error penalty.

Either way, we learn what M4 needs before paying for a long run.

# Roadmap

The project is organized into 10 milestones. Each milestone is a closed deliverable
that can be evaluated quantitatively and committed as a release tag.

## M0 — Literature gap audit

**Status:** complete (2026-05-08).

Confirmed that no published work covers the intersection of:
1. Triple pendulum on a cart
2. Reinforcement-learning policy (no precomputed trajectories)
3. Sim-to-real transfer (training in simulation, deployment on physical hardware)
4. All 56 transitions between the 8 equilibrium configurations

See [literature/state_of_the_art.md](literature/state_of_the_art.md) for the
annotated bibliography backing this claim.

## M1 — MuJoCo model

**Status:** complete.

Cart on a 2 m linear rail, three serial hinged poles (carbon-fiber-like
parameters), absolute angle bookkeeping in the env, force actuator on the cart,
joint position/velocity sensors. Visual cleanup with a tracking camera.

**Acceptance:** the simulation is stable under integration, and pole/cart
trajectories are physically reasonable when actuated by random control inputs.

## M2 — Stabilize one equilibrium (UUU) in simulation

**Status:** in progress.

Train TQC starting from a state near UUU. Reward is a quadratic combination of
absolute angle errors, link velocities, cart position, and a small control
penalty. The policy must reject perturbations and settle the system back to UUU
within a fixed time budget.

**Acceptance:**
- Mean episode reward >= -200 over 20 deterministic eval rollouts (ep length 1000).
- Tip standard deviation under 5 cm in the second half of an eval rollout.

## M3 — Stabilize all 8 equilibria

Train one conditional policy that stabilizes any of the 8 EPs given the target as
a one-hot input. Curriculum: cycle through targets during training, with init
near the target each time.

**Acceptance:** for each of the 8 EPs, the same policy maintains the system
within a tight tolerance for >= 80% of eval rollouts.

## M4 — Transition control (56 transitions) in simulation

Single policy that, given a target one-hot, moves the system from any starting
EP (or any state) to that target. Curriculum over (start EP, target EP) pairs;
randomize starting state across the 8 wells. May require energy shaping in
the reward to enable the swing-up motion.

**Acceptance:** success rate >= 70% over the full 56-transition matrix in
simulation, success defined as "tip stays within tolerance of target for the
last 200 steps of the episode".

## M5 — Domain randomization & robustness

Randomize during training:
- Mass and length of each link (+/- 10%)
- Friction at each hinge and on the slider
- Motor latency and saturation
- Sensor noise on angles and velocities
- Action delay (1 to 4 control steps)

Validate on a held-out perturbation set; success rate should not drop more than
~15 points relative to the deterministic-physics baseline.

## M6 — Hardware build

V-slot 2040 rail, brushless motor with ODrive, 3 x AS5048A magnetic encoders
(SPI), carbon-fiber rods, 608ZZ bearings, STM32 (or similar) running a
real-time ~1 kHz loop, ZeroMQ link to the policy host. BOM, CAD, firmware,
and assembly notes will be released with the paper.

**Acceptance:** the rig measures all 4 angles and the cart position cleanly,
and a hand-tuned LQR can hold UUU briefly. This is the hardware's bring-up
test, before any RL is involved.

## M7 — Sim2Real transfer (single-EP stabilization)

Deploy the M3 policy on the physical rig, holding each of the 8 EPs in turn.
This is the first sim-to-real evaluation.

**Acceptance:** for each of the 8 EPs, the policy maintains the equilibrium
on hardware for >= 10 seconds in >= 80% of trials.

## M8 — Sim2Real transfer (56 transitions)

Deploy the M5 policy on the physical rig over the full 56-transition matrix.
This is the headline result.

**Acceptance:** success rate >= 60% over the 56 transitions, measured as
"tip in tolerance for the last 2 seconds of the trial".

## M9 — Paper, video, kit

- arXiv preprint with the gap statement, method, ablations, and physical
  results.
- A short demo video (about 90 seconds) suitable for social media.
- BOM, CAD, firmware, weights, training and eval scripts published as a
  reproducible kit.

## M10 — Conference submission

Target venues: CoRL, ICRA, RSS, NeurIPS Sim2Real workshop. Submit the version
that is closest to the next deadline, with the workshop track as a fast-path
fallback.

## Tracking

Every milestone has its own MLflow experiment tag (`m2_upright`, `m3_all_eps`,
etc.). Closed milestones are turned into git tags (`v0.2`, `v0.3`, ...).

# M4 (56 transitions): investigation findings

> Working notes from the M4 de-risking effort (June 2026). What was actually
> wrong, what the data shows, and the data-grounded plan. Honest, not a pitch.

## Three bugs that made every prior M4 result uninterpretable

1. **Episodes died at step 1.** `target_mode=fixed` + `init_mode=near_target`
   with `start_ep != target_ep` (a swing-up) triggers the angle-fall check at
   step 0 (the start is far from the target) so every training episode
   terminated immediately with the -100 penalty. Fix: `init_mode: bottom` for
   swing-ups + a `_validate_cfg` guard.
2. **The eval measured random transitions.** `target_mode="transition"`
   randomises the (start, target) pair on every reset unless it is pinned via
   `reset(options=...)`. `per_transition_eval` never pinned it, so every
   `ep{src}to{dst}` metric (and the smoke/probe verdicts) measured a *random*
   transition. Fix: pin the pair via `options`.
3. (Earlier, audit) the M4 smoke's 56-transition overall is uninformative for a
   single-transition run; the trained pair is logged separately.

With (1) and (2) fixed the M4 eval is finally trustworthy.

## What the swing-up actually does (pinned DDD->UDD)

- Without a cart barrier: the policy degenerates to a **cart slide into the
  rail** (~165 steps), never approaching the target. The quadratic `cart_cost`
  is bounded and does not deter reaching the rail end.
- With a steep barrier (`cart_barrier_coef=50`, `(x/cart_limit)^8`) + dense
  progress reward: episodes survive **~1200-1400 steps** and the swing-up
  **reaches UDD** on some seeds (closest all-link error **0.21 rad**), but
  inconsistently (~1/4) and without holding; the cart still eventually drifts
  to the limit.

## The binding constraint: the catcher's basin is tiny

`scripts/measure_catch_basin.py` on M3 (run 072849) for UDD:

| init vel \ offset | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|
| 0.0 | 1.0 | 0.6 | 0.0 | 0.0 | 0.0 |
| 1.0 | 0.6 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

M3 reliably catches only within **~0.1 rad and near-zero velocity**. The
swing-up delivers ~0.2 rad *mid-swing* (high velocity) -> outside M3's basin ->
the hand-off fires (verified) but M3 drops it within a few steps.

## Plan (data-grounded)

The two-stage architecture is built and works mechanically (`sim/handoff.py`,
`scripts/handoff_eval.py`; the hand-off triggers). It is gated by two things:

1. **Widen the catcher's basin** (highest leverage): train an M3-like stabilizer
   with a harder init distribution (larger `init_noise`, non-zero link
   velocities, off-centre cart) so it tolerates a realistic swing-up delivery.
2. **Improve the swing-up delivery**: reward arriving slow + cart-centred (a
   "soft landing" term), so it lands inside the catcher's basin more often.
3. Budget: swing-up training wants more than 500K steps; iterate on **GPU**
   (RunPod) rather than ~2h CPU runs.

Per-link fall thresholds, `cart_cost_coef`, `cart_barrier_coef`, `cart_limit`,
`progress_reward_coef` and the swing-up `init_mode` are all config knobs now.

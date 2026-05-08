# State of the art (May 2026)

This file tracks all known prior work that touches on triple-pendulum control,
sim-to-real RL on inverted pendulums, or transition control between equilibria.
Update as new papers appear; the gap claim depends on this being current.

## Triple pendulum, classical control

- **Graichen, Treuer, Zeitz, *Automatica* (2013).** "Swing-up control of a triple
  pendulum on a cart with experimental validation." Two-degrees-of-freedom scheme:
  nonlinear feedforward + LQR feedback. Demonstrates **all 56 transitions** between
  the 8 equilibrium points of the triple pendulum. Uses **precomputed trajectories**,
  not learning. *This is the prior art our method generalizes.*

## Triple pendulum, RL

- **Baek, Lee, Lee, Jeon, Han, *Engineering Applications of AI* 128, 107518 (2024).**
  "Reinforcement learning to achieve real-time control of triple inverted pendulum."
  Model-free RL trained **directly on the physical system** (no sim-to-real).
  Off-policy actor-critic + structure-aware Virtual Experience Replay (VER) using
  the geometric symmetry. Scope: **swing-up to one equilibrium** (top). Does not
  cover the 56 transitions.

- **Cambridge, *Robotica* (online March 2026).** "Adaptive curriculum reinforcement
  learning with sim-to-real strategy in balance control of underactuated triple
  pendulum robots." Curriculum SAC + domain randomization, **sim-to-real** transfer
  to a hardware UTPR. Scope: **balance at the upper equilibrium only**, no
  transition control.

## Quadruple pendulum, RL

- **IJCAS (2025).** "Reinforcement Learning to Achieve Real-time Control of a
  Quadruple Inverted Pendulum." TQC + VER, swing-up + balance on hardware. The
  sim-to-real status is not the central claim of the paper; transitions are not
  covered.

## Double pendulum, sim-to-real RL with transitions

- **Lee, Ju, Lee, *MDPI Machines* 13(3):186 (2025).** "Transition Control of a
  Double-Inverted Pendulum System Using Sim2Real Reinforcement Learning." Covers
  the 4 EPs (DD, DU, UD, UU) and **all 12 transitions**. Hardware-centered
  Sim2Real approach. *This is the closest piece of prior art conceptually; we
  extend it from 4 EPs / 12 transitions to 8 EPs / 56 transitions on a triple.*

- **MDPI Mathematics 13(12):1996 (2025).** "Sim-to-Real Reinforcement Learning for
  a Rotary Double-Inverted Pendulum Based on a Mathematical Model."

## Single pendulum, sim-to-real

- **arXiv:2503.11065 (March 2025).** "Low-cost Real-world Implementation of the
  Swing-up Pendulum for Deep Reinforcement Learning Experiments." Single
  pendulum, low-cost open hardware. Useful methodological reference for our
  sim-to-real bridge.

## Multi-arm benchmarks

- **arXiv:2205.06231.** "The Experimental Multi-Arm Pendulum on a Cart: A
  Benchmark System for Chaos, Learning, and Control." Open-hardware benchmark
  (single, double, triple) intended for community use; control demonstrations
  are classical, not RL.

## Gap statement

> No published work demonstrates a **sim-to-real reinforcement-learning policy**
> that achieves **all 56 transitions** between the **8 equilibrium configurations**
> of a **physical triple inverted pendulum on a cart**.

Each adjacent prior work is missing exactly one ingredient:

| Work | Triple? | RL? | Sim2Real? | 56 transitions? |
|---|---|---|---|---|
| Graichen 2013 | ✅ | ❌ (LQR/feedforward) | n/a | ✅ |
| Baek 2024 | ✅ | ✅ | ❌ (on hw) | ❌ (swing-up only) |
| Cambridge 2026 | ✅ | ✅ | ✅ | ❌ (balance only) |
| Lee 2025 (MDPI) | ❌ (double) | ✅ | ✅ | ✅ (12 of double) |

## Watchlist (groups likely to publish a competing result)

- Han Lab (Baek et al.), natural extension of their triple/quadruple work to
  Sim2Real and transitions.
- Lee, Ju, Lee, double-pendulum transitions in 2025; triple is the next step.
- Cambridge / UTPR group, already have Sim2Real on triple; transitions next.

Set up Google Scholar alerts for: `"triple inverted pendulum" reinforcement learning`,
`"transition control" pendulum sim-to-real`, and the author names above. Re-check
arXiv (cs.RO, eess.SY) weekly.

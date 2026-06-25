"""Two-stage swing-up + stabilize hand-off controller.

Rationale (M4 probes, 2026-06-23): a single TQC policy learns to swing a link up
but cannot also settle and hold the full target equilibrium, and the cart drifts
off the rail. Classic remedy (Astrom-Furuta energy-swingup + LQR catch): use one
policy to drive the system *near* the target, then hand control to a stabilizer
that already holds that equilibrium (here the M3 policy, ~100% on the easy EPs).

This controller is policy-agnostic: it takes two objects with the Stable-Baselines3
`predict(obs, deterministic=True) -> (action, state)` interface and switches based
on the current angular distance (and optionally velocity) to the target.
"""
from __future__ import annotations

import numpy as np


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return np.arctan2(np.sin(x), np.cos(x))


class HandoffController:
    """Switch between a swing-up policy and a stabilizer policy.

    The stabilizer takes over once every link is within `capture_tol_rad` of the
    target (and, if `capture_vel_rad_s` is set, every link velocity is below it).
    With `latch=True` (default) the stabilizer keeps control once captured, even
    if the state briefly leaves the capture set -- this is what lets it actually
    *recover* a near-miss instead of flip-flopping back to swing-up.
    """

    def __init__(self, swingup, stabilizer, target_angles,
                 capture_tol_rad: float = 0.3,
                 capture_vel_rad_s: float | None = None,
                 latch: bool = True):
        self.swingup = swingup
        self.stabilizer = stabilizer
        self.target_angles = np.asarray(target_angles, dtype=np.float64)
        self.capture_tol_rad = float(capture_tol_rad)
        self.capture_vel_rad_s = capture_vel_rad_s
        self.latch = bool(latch)
        self._captured = False
        self.handoff_step: int | None = None  # step index at which hand-off first happened
        self._step = 0

    def reset(self) -> None:
        self._captured = False
        self.handoff_step = None
        self._step = 0

    def in_capture_set(self, abs_angles, abs_vels=None) -> bool:
        err = wrap_to_pi(np.asarray(abs_angles, dtype=np.float64) - self.target_angles)
        if not np.all(np.abs(err) < self.capture_tol_rad):
            return False
        if self.capture_vel_rad_s is not None and abs_vels is not None:
            if not np.all(np.abs(np.asarray(abs_vels)) < self.capture_vel_rad_s):
                return False
        return True

    def act(self, obs, abs_angles, abs_vels=None):
        """Return (action, mode) where mode is 'swing' or 'stab'."""
        use_stab = self.in_capture_set(abs_angles, abs_vels)
        if self.latch and self._captured:
            use_stab = True
        if use_stab and not self._captured:
            self._captured = True
            self.handoff_step = self._step
        self._step += 1
        policy = self.stabilizer if use_stab else self.swingup
        action, _ = policy.predict(obs, deterministic=True)
        return action, ("stab" if use_stab else "swing")

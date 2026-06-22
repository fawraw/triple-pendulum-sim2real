"""Canonical equilibrium-point (EP) naming and target angles.

Single source of truth for the 8 equilibria of the triple pendulum. Import
``ep_target_angles``, ``ep_name`` and ``EP_NAMES`` from here rather than
re-deriving them per module (which historically produced three mutually
incompatible naming conventions).

Convention
----------
- An EP id is a 3-bit integer in 0..7. Bit ``i`` (``i in {0,1,2}``) encodes
  link ``i+1``: bit set => that link is targeted UP (absolute angle 0),
  bit clear => DOWN (absolute angle pi).
- The name string is read base -> tip, i.e. ``name[0]`` is link 1 (the
  cart-attached link) and ``name[2]`` is link 3 (the tip). ``U`` = up, ``D`` = down.

So EP1 = ``001`` = link1 up only = "UDD", EP4 = ``100`` = link3 up only = "DDU",
EP6 = ``110`` = links 2+3 up = "DUU", EP0 = "DDD", EP7 = "UUU".
"""

from __future__ import annotations

import numpy as np


def ep_target_angles(ep_id: int) -> np.ndarray:
    """Return target absolute angles [theta1, theta2, theta3] for equilibrium ep_id (0..7).

    theta = 0 means the link points up, theta = pi means it points down.
    """
    bits = [(ep_id >> i) & 1 for i in range(3)]  # bit 0 = link 1, etc.
    return np.array([0.0 if b == 1 else np.pi for b in bits], dtype=np.float64)


def ep_name(ep_id: int) -> str:
    """Return the canonical name (base->tip, 'U'=up/'D'=down) for ep_id (0..7)."""
    bits = [(ep_id >> i) & 1 for i in range(3)]  # bit 0 = link 1 (base)
    return "".join("U" if b == 1 else "D" for b in bits)


# Indexed by EP id 0..7. Read base->tip.
EP_NAMES = [ep_name(i) for i in range(8)]

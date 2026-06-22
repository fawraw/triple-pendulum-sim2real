"""Shared evaluation helpers.

Single definition of the episode-length success criterion, used by every
eval path (M2/M3 stabilisation, M4 transitions, BC scripts). Previously the
threshold was the literal ``800`` hardcoded in four places (silently assuming
max_episode_steps == 1000) plus two different ``frac * max_steps`` expressions
-- so the same word "success" meant three different things and broke whenever
a config used a non-1000 budget.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def success_threshold(max_steps: int, frac: float = 0.8) -> int:
    """Minimum episode length (in steps) that counts as a success."""
    return int(frac * max_steps)


def success_rate(lengths: Sequence[float], max_steps: int, frac: float = 0.8) -> float:
    """Fraction of episodes whose length >= success_threshold(max_steps, frac)."""
    if len(lengths) == 0:
        return 0.0
    thr = success_threshold(max_steps, frac)
    return float(np.mean([length >= thr for length in lengths]))

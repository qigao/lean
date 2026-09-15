from __future__ import annotations

import math
import random


def deterministic_silence_indices(total_units: int, fraction: float, seed: int) -> tuple[int, ...]:
    if type(total_units) is not int or total_units <= 0:
        raise ValueError("total_units must be a positive integer")
    if type(fraction) not in (int, float) or isinstance(fraction, bool):
        raise ValueError("silence fraction must be numeric")
    value = float(fraction)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("silence fraction must be inside (0,1)")
    if type(seed) is not int:
        raise ValueError("silence seed must be an integer")
    count = max(1, int(math.floor(total_units * value)))
    return tuple(sorted(random.Random(seed).sample(range(total_units), count)))

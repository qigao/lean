from __future__ import annotations

from collections.abc import Sequence


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    return sum(a * b for a, b in zip(left, right))


def vector_potential_change(gradient: Sequence[float], direction: Sequence[float]) -> float:
    """First-order potential change represented by a Euclidean inner product."""
    return _dot(gradient, direction)


def vector_structural_conflict(g1: Sequence[float], g2: Sequence[float]) -> bool:
    """Two local motivational gradients conflict when their inner product is negative."""
    return _dot(g1, g2) < 0.0


def vector_tradeoff_direction(g1: Sequence[float], g2: Sequence[float]) -> tuple[float, ...]:
    """Return `-g1`, the witness direction used by the Lean theorem."""
    if not vector_structural_conflict(g1, g2):
        raise ValueError("gradients are not structurally conflicted")
    return tuple(-component for component in g1)

"""Exact rational schema primitives and authored BB conformance cases.

This module deliberately uses :class:`fractions.Fraction` for every rational
field. Production floating-point belief values are outside this contract.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "bb-rational-conformance/v1"
CONTRACT_SOURCE = "conformance/bb_rational_v1.contract"


def contract_hash() -> str:
    """Hash the explicit normalized semantic contract source.

    Git metadata and timestamps are intentionally excluded. Updating exact v1
    semantics requires an intentional edit of the reviewed contract source,
    which in turn changes every generated record's provenance digest.
    """

    root = Path(__file__).resolve().parents[1]
    source = (root / CONTRACT_SOURCE).read_bytes().replace(b"\r\n", b"\n")
    payload = SCHEMA.encode("utf-8") + b"\0" + CONTRACT_SOURCE.encode("utf-8") + b"\0" + source
    return hashlib.sha256(payload).hexdigest()


def rat(value: int | Fraction) -> dict[str, int]:
    """Return the canonical structural JSON encoding of an exact rational."""

    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("rational value must be an int or Fraction")
    normalized = Fraction(value)
    return {"num": normalized.numerator, "den": normalized.denominator}


def parse_rat(value: object) -> Fraction:
    """Parse one strict, already-reduced structural rational value."""

    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise TypeError("rational must be exactly an object with num and den")

    num = value["num"]
    den = value["den"]
    if isinstance(num, bool) or not isinstance(num, int):
        raise TypeError("rational numerator must be an integer")
    if isinstance(den, bool) or not isinstance(den, int):
        raise TypeError("rational denominator must be an integer")
    if den <= 0:
        raise ValueError("rational denominator must be positive")

    normalized = Fraction(num, den)
    if normalized.numerator != num or normalized.denominator != den:
        raise ValueError("rational numerator and denominator must be reduced")
    return normalized


def canonical_json_line(case: dict[str, Any]) -> str:
    """Serialize one canonical compact JSONL record with a final newline."""

    return json.dumps(
        case,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "inclusive-threshold",
        "kind": "propagation",
        "alpha": Fraction(1, 2),
        "threshold": Fraction(1, 2),
        "beliefs": (Fraction(1, 2), Fraction(0)),
        "exposures": (0, 0),
    },
    {
        "case_id": "no-broadcast",
        "kind": "propagation",
        "alpha": Fraction(1, 2),
        "threshold": Fraction(1, 2),
        "beliefs": (Fraction(1, 4), Fraction(1, 8)),
        "exposures": (0, 0),
    },
    {
        "case_id": "nontrivial-mean",
        "kind": "propagation",
        "alpha": Fraction(1, 2),
        "threshold": Fraction(1, 2),
        "beliefs": (Fraction(1), Fraction(1, 2), Fraction(2, 3)),
        "exposures": (0, 0, 0),
    },
    {
        "case_id": "bb-successive-births",
        "kind": "replay",
        "seed_fitness": (Fraction(1), Fraction(1)),
        "seed_edges": ((0, 1),),
        "birth_targets": ((1,), (2,)),
    },
    {
        "case_id": "duplicate-target-error",
        "kind": "error",
        "seed_fitness": (Fraction(1), Fraction(1)),
        "seed_edges": ((0, 1),),
        "targets": (0, 0),
    },
)


def _case(case_id: str) -> dict[str, Any]:
    for case in CASES:
        if case["case_id"] == case_id:
            return case
    raise KeyError(f"unknown conformance case: {case_id}")


def _path_neighbors(n: int, i: int) -> tuple[int, ...]:
    neighbors: list[int] = []
    if i > 0:
        neighbors.append(i - 1)
    if i + 1 < n:
        neighbors.append(i + 1)
    return tuple(neighbors)


def _evaluate_propagation(case: dict[str, Any]) -> dict[str, Any]:
    alpha: Fraction = case["alpha"]
    threshold: Fraction = case["threshold"]
    beliefs: tuple[Fraction, ...] = case["beliefs"]
    exposures: tuple[int, ...] = case["exposures"]
    broadcasting = tuple(threshold <= belief for belief in beliefs)

    next_beliefs: list[Fraction] = []
    next_exposures: list[int] = []
    for i, belief in enumerate(beliefs):
        incoming = tuple(
            j for j in _path_neighbors(len(beliefs), i) if broadcasting[j]
        )
        next_exposures.append(exposures[i] + len(incoming))
        if not incoming:
            next_beliefs.append(belief)
            continue
        mean = sum((beliefs[j] for j in incoming), Fraction(0)) / len(incoming)
        next_beliefs.append((1 - alpha) * belief + alpha * mean)

    return {
        "schema": SCHEMA,
        "case_id": case["case_id"],
        "kind": case["kind"],
        "input": {
            "alpha": rat(alpha),
            "threshold": rat(threshold),
            "beliefs": [rat(value) for value in beliefs],
            "exposures": list(exposures),
        },
        "expected": {
            "broadcasting": list(broadcasting),
            "beliefs": [rat(value) for value in next_beliefs],
            "exposures": next_exposures,
        },
    }


def _canonical_edge(a: int, b: int) -> tuple[int, int]:
    if a == b:
        raise ValueError("self edge is not valid in the BB conformance graph")
    return (a, b) if a < b else (b, a)


def _degrees(node_count: int, edges: set[tuple[int, int]]) -> list[int]:
    result = [0] * node_count
    for a, b in edges:
        result[a] += 1
        result[b] += 1
    return result


def _birth_probability(
    fitness: list[Fraction],
    edges: set[tuple[int, int]],
    targets: tuple[int, ...],
) -> Fraction:
    node_count = len(fitness)
    if any(target < 0 or target >= node_count for target in targets):
        raise ValueError("target out of range")
    if len(set(targets)) != len(targets):
        raise ValueError("duplicateTarget")

    degrees = _degrees(node_count, edges)
    remaining = set(range(node_count))
    probability = Fraction(1)
    for target in targets:
        weights = {i: fitness[i] * degrees[i] for i in remaining}
        total = sum(weights.values(), Fraction(0))
        if total <= 0:
            raise ValueError("nonpositive attachment mass")
        probability *= weights[target] / total
        remaining.remove(target)
    return probability


def _evaluate_successive_births(case: dict[str, Any]) -> dict[str, Any]:
    fitness = list(case["seed_fitness"])
    edges = {_canonical_edge(*edge) for edge in case["seed_edges"]}
    trace_mass = Fraction(1)
    birth_masses: list[Fraction] = []

    for targets in case["birth_targets"]:
        mass = _birth_probability(fitness, edges, targets)
        birth_masses.append(mass)
        trace_mass *= mass
        newborn = len(fitness)
        fitness.append(Fraction(1))
        for target in targets:
            edges.add(_canonical_edge(newborn, target))

    return {
        "schema": SCHEMA,
        "case_id": case["case_id"],
        "kind": case["kind"],
        "input": {
            "seed_fitness": [rat(value) for value in case["seed_fitness"]],
            "seed_edges": [list(edge) for edge in case["seed_edges"]],
            "birth_targets": [list(targets) for targets in case["birth_targets"]],
        },
        "expected": {
            "birth_masses": [rat(value) for value in birth_masses],
            "trace_mass": rat(trace_mass),
            "node_count": len(fitness),
            "edges": [list(edge) for edge in sorted(edges)],
        },
    }


def _evaluate_duplicate_target(case: dict[str, Any]) -> dict[str, Any]:
    fitness = list(case["seed_fitness"])
    edges = {_canonical_edge(*edge) for edge in case["seed_edges"]}
    try:
        _birth_probability(fitness, edges, case["targets"])
    except ValueError as exc:
        if str(exc) != "duplicateTarget":
            raise
        return {
            "schema": SCHEMA,
            "case_id": case["case_id"],
            "kind": case["kind"],
            "input": {
                "seed_fitness": [rat(value) for value in case["seed_fitness"]],
                "seed_edges": [list(edge) for edge in case["seed_edges"]],
                "targets": list(case["targets"]),
            },
            "expected": {
                "error": {
                    "stage": "tick_network",
                    "code": "duplicateTarget",
                    "field": "targets",
                }
            },
        }
    raise AssertionError("duplicate-target-error fixture unexpectedly succeeded")


def evaluate_named(case_id: str) -> dict[str, Any]:
    """Evaluate one authored v1 case with exact arithmetic only."""

    case = _case(case_id)
    if case["kind"] == "propagation":
        return _evaluate_propagation(case)
    if case_id == "bb-successive-births":
        return _evaluate_successive_births(case)
    if case_id == "duplicate-target-error":
        return _evaluate_duplicate_target(case)
    raise AssertionError(f"unhandled conformance case: {case_id}")

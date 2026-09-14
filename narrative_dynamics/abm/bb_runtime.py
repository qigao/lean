"""Checked exact Bianconi--Barabasi seed and birth arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from narrative_dynamics.abm.bb_runtime_contracts import (
    BBRuntimeBirth,
    BBRuntimeError,
    BBRuntimeRawSeed,
    BBRuntimeTopology,
)


def _network_error(
    stage: str,
    code: str,
    field: str,
    *,
    tick_index: int | None = None,
    birth_index: int | None = None,
) -> BBRuntimeError:
    return BBRuntimeError(
        stage,
        code,
        field=field,
        tick_index=tick_index,
        birth_index=birth_index,
        bb_cause=None if code == "invalidType" else code,
    )


def _parse_bb_seed(raw: object) -> BBRuntimeTopology | BBRuntimeError:
    if not isinstance(raw, BBRuntimeRawSeed):
        return _network_error("seed_network", "invalidType", "network")

    n = raw.node_count
    if isinstance(n, bool) or not isinstance(n, int):
        return _network_error("seed_network", "invalidType", "node_count")
    if n < 2:
        return _network_error("seed_network", "invalidNodeCount", "node_count")

    fitness = raw.fitness
    if not isinstance(fitness, tuple):
        return _network_error("seed_network", "invalidType", "fitness")
    if len(fitness) != n:
        return _network_error("seed_network", "fitnessSizeMismatch", "fitness")
    checked_fitness: list[Fraction] = []
    for index, value in enumerate(fitness):
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
            return _network_error(
                "seed_network", "invalidType", f"fitness[{index}]",
            )
        checked_fitness.append(Fraction(value))
    if any(value <= 0 for value in checked_fitness):
        return _network_error("seed_network", "nonpositiveFitness", "fitness")

    edges = raw.edges
    if not isinstance(edges, tuple):
        return _network_error("seed_network", "invalidType", "edges")
    canonical_edges: list[tuple[int, int]] = []
    for edge_index, edge in enumerate(edges):
        if not isinstance(edge, tuple):
            return _network_error(
                "seed_network", "invalidType", f"edges[{edge_index}]",
            )
        if len(edge) != 2:
            return _network_error("seed_network", "invalidEdge", "edges")
        u, v = edge
        for endpoint_index, endpoint in enumerate((u, v)):
            if isinstance(endpoint, bool) or not isinstance(endpoint, int):
                return _network_error(
                    "seed_network",
                    "invalidType",
                    f"edges[{edge_index}][{endpoint_index}]",
                )
        if not (0 <= u < n and 0 <= v < n) or u == v:
            return _network_error("seed_network", "invalidEdge", "edges")
        canonical_edges.append((min(u, v), max(u, v)))

    if len(set(canonical_edges)) != len(canonical_edges):
        return _network_error("seed_network", "duplicateEdge", "edges")

    adjacency = [set() for _ in range(n)]
    for u, v in canonical_edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    visited: set[int] = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency[node] - visited)
    if len(visited) != n:
        return _network_error("seed_network", "disconnectedSeed", "edges")

    return BBRuntimeTopology(
        tuple(checked_fitness),
        tuple(sorted(canonical_edges)),
    )


def _check_m(
    m: object,
    n: int,
    *,
    stage: str,
    tick_index: int | None = None,
    birth_index: int | None = None,
) -> int | BBRuntimeError:
    indices = (
        {"tick_index": tick_index, "birth_index": birth_index}
        if stage == "tick_network" else {}
    )
    if isinstance(m, bool) or not isinstance(m, int):
        return BBRuntimeError(stage, "invalidType", field="m", **indices)
    if not 1 <= m <= n:
        code = "initialM" if stage == "initial_m" else "invalidM"
        return BBRuntimeError(
            stage,
            code,
            field="m",
            bb_cause=code if stage == "tick_network" else None,
            **indices,
        )
    return m


@dataclass(frozen=True)
class _CheckedBBBirth:
    fitness: Fraction
    targets: tuple[int, ...]
    tick_mass: Fraction


def _check_birth(
    topology: BBRuntimeTopology,
    m: object,
    raw: BBRuntimeBirth,
    *,
    tick_index: int,
    birth_index: int,
) -> _CheckedBBBirth | BBRuntimeError:
    checked_m = _check_m(
        m,
        topology.node_count,
        stage="tick_network",
        tick_index=tick_index,
        birth_index=birth_index,
    )
    if isinstance(checked_m, BBRuntimeError):
        return checked_m

    new_fitness = raw.fitness
    if isinstance(new_fitness, bool) or not isinstance(new_fitness, (int, Fraction)):
        return _network_error(
            "tick_network", "invalidType", "fitness",
            tick_index=tick_index, birth_index=birth_index,
        )
    new_fitness = Fraction(new_fitness)
    if new_fitness <= 0:
        return _network_error(
            "tick_network", "nonpositiveFitness", "fitness",
            tick_index=tick_index, birth_index=birth_index,
        )

    targets = raw.targets
    if not isinstance(targets, tuple):
        return _network_error(
            "tick_network", "invalidType", "targets",
            tick_index=tick_index, birth_index=birth_index,
        )
    if len(targets) != checked_m:
        return _network_error(
            "tick_network", "targetCountMismatch", "targets",
            tick_index=tick_index, birth_index=birth_index,
        )
    for index, target in enumerate(targets):
        if isinstance(target, bool) or not isinstance(target, int):
            return _network_error(
                "tick_network", "invalidType", f"targets[{index}]",
                tick_index=tick_index, birth_index=birth_index,
            )
    if any(target < 0 or target >= topology.node_count for target in targets):
        return _network_error(
            "tick_network", "targetOutOfRange", "targets",
            tick_index=tick_index, birth_index=birth_index,
        )
    if len(set(targets)) != len(targets):
        return _network_error(
            "tick_network", "duplicateTarget", "targets",
            tick_index=tick_index, birth_index=birth_index,
        )

    degree = [0] * topology.node_count
    for u, v in topology.edges:
        degree[u] += 1
        degree[v] += 1
    weights = tuple(f * d for f, d in zip(topology.fitness, degree))
    remaining = sum(weights, Fraction(0))
    mass = Fraction(1)
    for target in targets:
        mass *= weights[target] / remaining
        remaining -= weights[target]
    return _CheckedBBBirth(new_fitness, targets, mass)


def _apply_birth(
    topology: BBRuntimeTopology,
    checked: _CheckedBBBirth,
) -> BBRuntimeTopology:
    newborn = topology.node_count
    return BBRuntimeTopology(
        topology.fitness + (checked.fitness,),
        tuple(sorted(
            topology.edges + tuple((t, newborn) for t in checked.targets)
        )),
    )

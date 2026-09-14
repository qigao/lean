"""Checked exact Bianconi--Barabasi seed and birth arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from narrative_dynamics.abm.bb_runtime_contracts import (
    BBRuntimeAgent,
    BBRuntimeBirth,
    BBRuntimeConfig,
    BBRuntimeError,
    BBRuntimeFrame,
    BBRuntimeNewborn,
    BBRuntimeRawSeed,
    BBRuntimeTick,
    BBRuntimeTopology,
    BBRuntimeTransition,
)
from narrative_dynamics.abm.contracts import (
    NetworkABMModel,
    NetworkAgentSpec,
    NetworkAgentState,
    PopulationState,
    SocialEdge,
    SocialNetwork,
)
from narrative_dynamics.abm.contracts import _probability, _text
from narrative_dynamics.abm.simulation import simulate_round


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


def _agent_error(
    stage: str,
    code: str,
    field: str,
    *,
    agent_index: int | None,
    tick_index: int | None,
    birth_index: int | None,
) -> BBRuntimeError:
    return BBRuntimeError(
        stage,
        code,
        field=field,
        agent_index=agent_index,
        tick_index=tick_index,
        birth_index=birth_index,
    )


def _parse_agent(
    raw: object,
    *,
    newborn: bool,
    used_ids: frozenset[str],
    stage: str,
    agent_index: int | None = None,
    tick_index: int | None = None,
    birth_index: int | None = None,
) -> tuple[NetworkAgentSpec, NetworkAgentState] | BBRuntimeError:
    expected = BBRuntimeNewborn if newborn else BBRuntimeAgent
    if not isinstance(raw, expected):
        return _agent_error(
            stage, "invalidType", "agent",
            agent_index=agent_index,
            tick_index=tick_index,
            birth_index=birth_index,
        )

    normalized: dict[str, float] = {}
    for field in ("receptivity", "broadcast_threshold", "belief"):
        value = getattr(raw, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _agent_error(
                stage, "invalidType", field,
                agent_index=agent_index,
                tick_index=tick_index,
                birth_index=birth_index,
            )
        try:
            normalized[field] = _probability(value, label=field)
        except (ValueError, OverflowError):
            return _agent_error(
                stage, "invalidAgentValue", field,
                agent_index=agent_index,
                tick_index=tick_index,
                birth_index=birth_index,
            )

    exposure_count = 0
    if not newborn:
        exposure = raw.exposure_count
        if isinstance(exposure, bool) or not isinstance(exposure, int):
            return _agent_error(
                stage, "invalidType", "exposure_count",
                agent_index=agent_index,
                tick_index=tick_index,
                birth_index=birth_index,
            )
        if exposure < 0:
            return _agent_error(
                stage, "invalidExposure", "exposure_count",
                agent_index=agent_index,
                tick_index=tick_index,
                birth_index=birth_index,
            )
        exposure_count = exposure

    agent_id = raw.agent_id
    if not isinstance(agent_id, str):
        return _agent_error(
            stage, "invalidType", "agent_id",
            agent_index=agent_index,
            tick_index=tick_index,
            birth_index=birth_index,
        )
    try:
        agent_id = _text(agent_id, label="agent_id")
    except ValueError:
        return _agent_error(
            stage, "invalidText", "agent_id",
            agent_index=agent_index,
            tick_index=tick_index,
            birth_index=birth_index,
        )
    if agent_id in used_ids:
        return _agent_error(
            stage, "duplicateAgentId", "agent_id",
            agent_index=agent_index,
            tick_index=tick_index,
            birth_index=birth_index,
        )

    role = raw.role
    if not isinstance(role, str):
        return _agent_error(
            stage, "invalidType", "role",
            agent_index=agent_index,
            tick_index=tick_index,
            birth_index=birth_index,
        )
    try:
        role = _text(role, label="role")
    except ValueError:
        return _agent_error(
            stage, "invalidText", "role",
            agent_index=agent_index,
            tick_index=tick_index,
            birth_index=birth_index,
        )

    adoption = raw.adoption_threshold
    if isinstance(adoption, bool) or not isinstance(adoption, (int, float)):
        return _agent_error(
            stage, "invalidType", "adoption_threshold",
            agent_index=agent_index,
            tick_index=tick_index,
            birth_index=birth_index,
        )
    try:
        adoption = _probability(adoption, label="adoption_threshold")
    except (ValueError, OverflowError):
        return _agent_error(
            stage, "invalidAdoptionThreshold", "adoption_threshold",
            agent_index=agent_index,
            tick_index=tick_index,
            birth_index=birth_index,
        )

    profile = NetworkAgentSpec(
        agent_id,
        role,
        normalized["receptivity"],
        adoption,
        normalized["broadcast_threshold"],
    )
    state = NetworkAgentState(
        agent_id,
        normalized["belief"],
        exposure_count,
        normalized["belief"] >= normalized["broadcast_threshold"],
    )
    return profile, state


def _model_for(
    config: BBRuntimeConfig,
    agent_ids: tuple[str, ...],
    topology: BBRuntimeTopology,
    profiles: tuple[NetworkAgentSpec, ...],
) -> NetworkABMModel:
    channels = tuple(
        SocialEdge(agent_ids[source], agent_ids[target], "bb", 1.0, True)
        for u, v in topology.edges
        for source, target in ((u, v), (v, u))
    )
    return NetworkABMModel(
        config.model_id, config.version, profiles,
        SocialNetwork(agent_ids, channels),
    )


def _tick_error(
    prior: BBRuntimeFrame,
    code: str,
    field: str,
) -> BBRuntimeError:
    return BBRuntimeError(
        "tick",
        code,
        field=field,
        tick_index=prior.tick_count,
        birth_index=prior.birth_count,
    )


def _advance_tick(
    prior: BBRuntimeFrame,
    raw: object,
) -> BBRuntimeTransition | BBRuntimeError:
    if not isinstance(raw, BBRuntimeTick):
        return _tick_error(prior, "invalidType", "tick")
    if not isinstance(raw.kind, str):
        return _tick_error(prior, "invalidType", "kind")
    if raw.kind not in ("idle", "birth"):
        return _tick_error(prior, "invalidTickKind", "kind")
    if raw.kind == "idle" and raw.birth is not None:
        return _tick_error(prior, "unexpectedBirthData", "birth")
    if raw.kind == "birth" and raw.birth is None:
        return _tick_error(prior, "missingBirthData", "birth")
    if raw.kind == "birth" and not isinstance(raw.birth, BBRuntimeBirth):
        return _tick_error(prior, "invalidType", "birth")

    if raw.kind == "birth":
        checked = _check_birth(
            prior.topology,
            prior.config.m,
            raw.birth,
            tick_index=prior.tick_count,
            birth_index=prior.birth_count,
        )
        if isinstance(checked, BBRuntimeError):
            return checked
        parsed = _parse_agent(
            raw.birth.agent,
            newborn=True,
            used_ids=frozenset(prior.agent_ids),
            stage="tick_agent",
            tick_index=prior.tick_count,
            birth_index=prior.birth_count,
        )
        if isinstance(parsed, BBRuntimeError):
            return parsed
        newborn_profile, newborn_state = parsed

        topology = _apply_birth(prior.topology, checked)
        agent_ids = prior.agent_ids + (newborn_profile.agent_id,)
        old_profiles = {profile.agent_id: profile for profile in prior.model.agents}
        profiles = tuple(old_profiles[agent_id] for agent_id in prior.agent_ids) + (
            newborn_profile,
        )
        model = _model_for(prior.config, agent_ids, topology, profiles)
        old_states = {state.agent_id: state for state in prior.population.agents}
        post_growth = PopulationState(
            model.model_id,
            model.content_hash,
            0,
            None,
            tuple(old_states[agent_id] for agent_id in prior.agent_ids) + (
                newborn_state,
            ),
        )
        normalized_newborn = BBRuntimeNewborn(
            newborn_profile.agent_id,
            newborn_profile.role,
            newborn_profile.receptivity,
            newborn_profile.broadcast_threshold,
            newborn_state.belief,
            newborn_profile.adoption_threshold,
        )
        tick = BBRuntimeTick(
            "birth",
            BBRuntimeBirth(checked.fitness, checked.targets, normalized_newborn),
        )
        tick_mass = checked.tick_mass
    else:
        tick = BBRuntimeTick()
        model = prior.model
        post_growth = prior.population
        agent_ids = prior.agent_ids
        topology = prior.topology
        tick_mass = Fraction(1)

    round_result = simulate_round(model, post_growth)
    next_frame = BBRuntimeFrame(
        config=prior.config,
        tick_count=prior.tick_count + 1,
        birth_count=prior.birth_count + int(tick.kind == "birth"),
        agent_ids=agent_ids,
        topology=topology,
        model=model,
        population=round_result.next_state,
        trace_mass=prior.trace_mass * tick_mass,
        parent_frame_hash=prior.content_hash,
        applied_tick=tick,
    )
    return BBRuntimeTransition(
        prior, tick, post_growth, round_result, tick_mass, next_frame,
    )

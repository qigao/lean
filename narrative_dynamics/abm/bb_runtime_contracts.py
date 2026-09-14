"""Immutable records for finite BB-driven V1 population histories.

Raw requests intentionally defer validation to the replay boundary. Result records
check structural bindings and lineage, not whether BB probabilities or behavioral
updates were computed correctly. They are not resumable execution certificates.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from narrative_dynamics.abm.contracts import (
    NetworkABMModel,
    NetworkAgentSpec,
    NetworkAgentState,
    PopulationState,
    _hash,
    _nonnegative_integer,
    _probability,
    _text,
)
from narrative_dynamics.abm.simulation import NetworkRoundResult, _validate_model_state
from narrative_dynamics.contracts import stable_content_hash


_SCHEMA = "bb-abm-runtime-v1"


def _instance(value: object, expected: type, *, label: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be {expected.__name__}")


def _positive_fraction(value: object, *, label: str) -> Fraction:
    if not isinstance(value, Fraction):
        raise TypeError(f"{label} must be Fraction")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _mass(value: object, *, label: str) -> Fraction:
    result = _positive_fraction(value, label=label)
    if result > 1:
        raise ValueError(f"{label} must not exceed one")
    return result


def _fraction_text(value: Fraction) -> str:
    if not isinstance(value, Fraction):
        raise TypeError("canonical rational must be Fraction")
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _same_content(left: object, right: object) -> bool:
    # Python equality alone equates 0.0 and -0.0; typed content hashes do not.
    return left.content_hash == right.content_hash


@dataclass(frozen=True)
class BBRuntimeRawSeed:
    node_count: object
    fitness: object
    edges: object


@dataclass(frozen=True)
class BBRuntimeAgent:
    agent_id: object
    role: object
    receptivity: object
    broadcast_threshold: object
    belief: object
    exposure_count: object = 0
    adoption_threshold: object = 0.5


@dataclass(frozen=True)
class BBRuntimeNewborn:
    agent_id: object
    role: object
    receptivity: object
    broadcast_threshold: object
    belief: object
    adoption_threshold: object = 0.5


@dataclass(frozen=True)
class BBRuntimeBirth:
    fitness: object
    targets: object
    agent: object


@dataclass(frozen=True)
class BBRuntimeTick:
    kind: object = "idle"
    birth: object = None

    def to_dict(self) -> dict[str, object]:
        """Serialize only a normalized applied tick, never an unchecked request."""
        _validate_tick(self)
        if self.kind == "idle":
            return {"kind": "idle", "birth": None}
        newborn = self.birth.agent
        return {
            "kind": "birth",
            "birth": {
                "fitness": _fraction_text(self.birth.fitness),
                "targets": list(self.birth.targets),
                "agent": {
                    "agent_id": newborn.agent_id,
                    "role": newborn.role,
                    "receptivity": newborn.receptivity,
                    "broadcast_threshold": newborn.broadcast_threshold,
                    "belief": newborn.belief,
                    "adoption_threshold": newborn.adoption_threshold,
                },
            },
        }


@dataclass(frozen=True)
class BBRuntimeSeed:
    network: object
    agents: object
    model_id: object = "bb-runtime"
    version: object = "1"


def _validate_tick(tick: object) -> None:
    """Check normalized result data without implementing raw-error precedence."""
    _instance(tick, BBRuntimeTick, label="applied tick")
    _text(tick.kind, label="applied tick kind")
    if tick.kind == "idle":
        if tick.birth is not None:
            raise ValueError("idle tick cannot contain birth data")
        return
    if tick.kind != "birth":
        raise ValueError("applied tick kind must be idle or birth")
    _instance(tick.birth, BBRuntimeBirth, label="applied birth")
    _positive_fraction(tick.birth.fitness, label="newborn fitness")
    targets = tick.birth.targets
    _instance(targets, tuple, label="ordered targets")
    if not targets:
        raise ValueError("birth targets must be nonempty")
    for target in targets:
        _nonnegative_integer(target, label="birth target")
    if len(set(targets)) != len(targets):
        raise ValueError("birth targets must be distinct")
    newborn = tick.birth.agent
    _instance(newborn, BBRuntimeNewborn, label="newborn")
    _text(newborn.agent_id, label="newborn id")
    _text(newborn.role, label="newborn role")
    for field in ("receptivity", "broadcast_threshold", "belief", "adoption_threshold"):
        value = getattr(newborn, field)
        if type(value) is not float:
            raise TypeError(f"applied newborn {field} must be normalized float")
        _probability(value, label=f"newborn {field}")


def _newborn_profile(newborn: BBRuntimeNewborn) -> NetworkAgentSpec:
    return NetworkAgentSpec(
        newborn.agent_id, newborn.role, newborn.receptivity,
        newborn.adoption_threshold, newborn.broadcast_threshold,
    )


@dataclass(frozen=True)
class BBRuntimeConfig:
    model_id: str
    version: str
    m: int
    seed_node_count: int
    seed_edge_count: int

    def __post_init__(self) -> None:
        _text(self.model_id, label="runtime model id")
        _text(self.version, label="runtime model version")
        for field in ("m", "seed_node_count", "seed_edge_count"):
            _nonnegative_integer(getattr(self, field), label=field)
        n = self.seed_node_count
        if n < 2:
            raise ValueError("BB seed requires at least two nodes")
        if not 1 <= self.m <= n:
            raise ValueError("initial attachment count must be in [1, seed nodes]")
        if not n - 1 <= self.seed_edge_count <= n * (n - 1) // 2:
            raise ValueError("seed edge count is outside connected simple-graph bounds")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "m": self.m,
            "seed_node_count": self.seed_node_count,
            "seed_edge_count": self.seed_edge_count,
        }


@dataclass(frozen=True)
class BBRuntimeTopology:
    """Canonical exact topology data; connectivity belongs to the seed parser."""

    fitness: tuple[Fraction, ...]
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        _instance(self.fitness, tuple, label="fitness")
        _instance(self.edges, tuple, label="edges")
        if self.node_count < 2:
            raise ValueError("BB topology requires at least two nodes")
        for fitness in self.fitness:
            _positive_fraction(fitness, label="fitness")
        for edge in self.edges:
            _instance(edge, tuple, label="edge")
            if len(edge) != 2:
                raise ValueError("an undirected edge requires two endpoints")
            u, v = edge
            _nonnegative_integer(u, label="edge source")
            _nonnegative_integer(v, label="edge target")
            if not u < v < self.node_count:
                raise ValueError("edges must satisfy 0 <= source < target < node count")
        if tuple(sorted(self.edges)) != self.edges or len(set(self.edges)) != len(self.edges):
            raise ValueError("edges must be sorted, canonical and distinct")

    @property
    def node_count(self) -> int:
        return len(self.fitness)

    def to_dict(self) -> dict[str, object]:
        return {
            "node_count": self.node_count,
            "fitness": [_fraction_text(f) for f in self.fitness],
            "edges": [list(edge) for edge in self.edges],
        }


@dataclass(frozen=True)
class BBRuntimeFrame:
    config: BBRuntimeConfig
    tick_count: int
    birth_count: int
    agent_ids: tuple[str, ...]
    topology: BBRuntimeTopology
    model: NetworkABMModel
    population: PopulationState
    trace_mass: Fraction
    parent_frame_hash: str | None
    applied_tick: BBRuntimeTick | None

    def __post_init__(self) -> None:
        _instance(self.config, BBRuntimeConfig, label="runtime config")
        _instance(self.topology, BBRuntimeTopology, label="BB topology")
        _instance(self.model, NetworkABMModel, label="frame model")
        _instance(self.agent_ids, tuple, label="numeric id registry")
        for agent_id in self.agent_ids:
            _text(agent_id, label="registry agent id")
        if len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("numeric id registry must contain unique external ids")
        if len(self.agent_ids) != self.topology.node_count:
            raise ValueError("registry must contain exactly the BB nodes")
        if set(self.agent_ids) != set(self.model.network.agent_ids):
            raise ValueError("registry must match the exact model roster")
        if (self.model.model_id, self.model.version) != (
            self.config.model_id, self.config.version,
        ):
            raise ValueError("frame model must bind the run configuration")
        _validate_model_state(self.model, self.population)
        expected_channels = {
            (self.agent_ids[source], self.agent_ids[target], "bb", 1.0, True)
            for u, v in self.topology.edges
            for source, target in ((u, v), (v, u))
        }
        actual_channels = {
            (e.source_agent_id, e.target_agent_id, e.relation_type, e.influence, e.active)
            for e in self.model.network.edges
        }
        if actual_channels != expected_channels:
            raise ValueError("model channels must be exactly the bidirectional unit BB edges")
        _nonnegative_integer(self.tick_count, label="global tick count")
        _nonnegative_integer(self.birth_count, label="birth count")
        if self.birth_count > self.tick_count:
            raise ValueError("birth count cannot exceed global tick count")
        if self.topology.node_count != self.config.seed_node_count + self.birth_count:
            raise ValueError("node count must equal seed nodes plus births")
        if len(self.topology.edges) != self.config.seed_edge_count + self.config.m * self.birth_count:
            raise ValueError("edge count must equal seed edges plus m times births")
        _mass(self.trace_mass, label="trace mass")
        local_round = self.population.round_index
        if self.tick_count == 0:
            if self.parent_frame_hash is not None or self.applied_tick is not None:
                raise ValueError("genesis cannot have a parent or applied tick")
            if local_round != 0 or self.population.parent_state_hash is not None:
                raise ValueError("genesis population must start at local round zero")
            if self.trace_mass != 1:
                raise ValueError("genesis trace mass must be one")
        else:
            _hash(self.parent_frame_hash, label="parent frame hash")
            _validate_tick(self.applied_tick)
            if not 1 <= local_round <= self.tick_count:
                raise ValueError("positive V1 local round cannot exceed global ticks")
            if self.birth_count == 0 and local_round != self.tick_count:
                raise ValueError("without births the local and global clocks coincide")
            if self.birth_count and local_round > self.tick_count - self.birth_count + 1:
                raise ValueError("local round cannot exceed the available epoch ticks")
            if self.applied_tick.kind == "birth":
                if self.birth_count == 0 or local_round != 1:
                    raise ValueError("birth frame must end at local round one in a new epoch")
                self._validate_birth_binding()
            elif self.birth_count and local_round < 2:
                raise ValueError("idle after a birth must continue its local round chain")

    def _validate_birth_binding(self) -> None:
        raw = self.applied_tick.birth
        n = self.topology.node_count
        if len(raw.targets) != self.config.m or any(t >= n - 1 for t in raw.targets):
            raise ValueError("applied targets must be m distinct pre-birth numeric ids")
        if raw.agent.agent_id != self.agent_ids[-1]:
            raise ValueError("applied newborn must be the last numeric id")
        if self.topology.fitness[-1] != raw.fitness:
            raise ValueError("applied newborn fitness must match the topology")
        incident = tuple(edge for edge in self.topology.edges if edge[1] == n - 1)
        expected = tuple(sorted((target, n - 1) for target in raw.targets))
        if incident != expected:
            raise ValueError("newborn edges must match the applied ordered targets")
        profiles = {p.agent_id: p for p in self.model.agents}
        if not _same_content(profiles[raw.agent.agent_id], _newborn_profile(raw.agent)):
            raise ValueError("newborn profile must match the applied metadata")

    @property
    def epoch(self) -> int:
        return self.birth_count

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "config": self.config.to_dict(),
            "tick_count": self.tick_count,
            "birth_count": self.birth_count,
            "agent_ids": list(self.agent_ids),
            "topology": self.topology.to_dict(),
            "model": self.model.to_dict(),
            "population": self.population.to_dict(),
            "trace_mass": _fraction_text(self.trace_mass),
            "parent_frame_hash": self.parent_frame_hash,
            "applied_tick": None if self.applied_tick is None else self.applied_tick.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class BBRuntimeTransition:
    prior: BBRuntimeFrame
    tick: BBRuntimeTick
    post_growth_population: PopulationState
    round_result: NetworkRoundResult
    tick_mass: Fraction
    next_frame: BBRuntimeFrame

    def __post_init__(self) -> None:
        _instance(self.prior, BBRuntimeFrame, label="prior frame")
        _instance(self.next_frame, BBRuntimeFrame, label="next frame")
        _instance(self.round_result, NetworkRoundResult, label="V1 round")
        _validate_tick(self.tick)
        _mass(self.tick_mass, label="tick mass")
        prior, next_frame = self.prior, self.next_frame
        if next_frame.config != prior.config:
            raise ValueError("transition cannot change run configuration")
        is_birth = self.tick.kind == "birth"
        if next_frame.tick_count != prior.tick_count + 1:
            raise ValueError("transition must advance exactly one global tick")
        if next_frame.birth_count != prior.birth_count + int(is_birth):
            raise ValueError("transition must count exactly its birth")
        if next_frame.parent_frame_hash != prior.content_hash:
            raise ValueError("next frame must bind the exact prior frame hash")
        if next_frame.applied_tick is None or stable_content_hash(self.tick.to_dict()) != \
                stable_content_hash(next_frame.applied_tick.to_dict()):
            raise ValueError("transition and next frame must bind the same applied tick")
        _validate_model_state(next_frame.model, self.post_growth_population)
        if not _same_content(self.round_result.prior_state, self.post_growth_population):
            raise ValueError("V1 round must bind the exact post-growth input")
        if not _same_content(self.round_result.next_state, next_frame.population):
            raise ValueError("V1 round output must bind the next frame population")
        if (self.round_result.model_id, self.round_result.model_hash) != (
            next_frame.model.model_id, next_frame.model.content_hash,
        ):
            raise ValueError("V1 round must bind the next frame model")
        if next_frame.trace_mass != prior.trace_mass * self.tick_mass:
            raise ValueError("cumulative trace mass must multiply by the exact tick mass")
        if is_birth:
            self._validate_birth()
        else:
            if not _same_content(next_frame.model, prior.model):
                raise ValueError("idle cannot change the model")
            if next_frame.topology != prior.topology or next_frame.agent_ids != prior.agent_ids:
                raise ValueError("idle cannot change topology, fitness or numeric ids")
            if not _same_content(self.post_growth_population, prior.population):
                raise ValueError("idle must propagate the exact prior population")
            if self.tick_mass != 1:
                raise ValueError("idle tick mass must be one")

    def _validate_birth(self) -> None:
        prior, next_frame = self.prior, self.next_frame
        raw = self.tick.birth
        newborn_id = raw.agent.agent_id
        if newborn_id in prior.agent_ids or next_frame.agent_ids != prior.agent_ids + (newborn_id,):
            raise ValueError("birth must append exactly one fresh external id")
        if next_frame.topology.fitness != prior.topology.fitness + (raw.fitness,):
            raise ValueError("birth must preserve all old fitness values")
        expected_edges = tuple(sorted(
            prior.topology.edges + tuple((target, prior.topology.node_count)
                                         for target in raw.targets)
        ))
        if next_frame.topology.edges != expected_edges:
            raise ValueError("birth must preserve old edges and append only the target edges")
        old_profiles = {p.agent_id: p for p in prior.model.agents}
        profiles = {p.agent_id: p for p in next_frame.model.agents}
        if any(not _same_content(profiles[key], profile) for key, profile in old_profiles.items()):
            raise ValueError("birth must preserve every old profile field")
        post = self.post_growth_population
        if post.round_index != 0 or post.parent_state_hash is not None:
            raise ValueError("birth propagation input must open a local-round-zero epoch")
        old_states = {a.agent_id: a for a in prior.population.agents}
        states = {a.agent_id: a for a in post.agents}
        if any(not _same_content(states[key], state) for key, state in old_states.items()):
            raise ValueError("birth must preserve every old agent state")
        newborn = NetworkAgentState(
            newborn_id, raw.agent.belief, 0,
            raw.agent.belief >= raw.agent.broadcast_threshold,
        )
        if not _same_content(states[newborn_id], newborn):
            raise ValueError("newborn input must bind its belief, zero exposure and broadcasting")

    @property
    def tick_index(self) -> int:
        return self.prior.tick_count

    @property
    def birth_index(self) -> int:
        return self.prior.birth_count

    @property
    def global_round(self) -> int:
        return self.prior.tick_count + 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "prior_frame_hash": self.prior.content_hash,
            "tick": self.tick.to_dict(),
            "post_growth_population": self.post_growth_population.to_dict(),
            "round_result": self.round_result.to_dict(),
            "tick_mass": _fraction_text(self.tick_mass),
            "next_frame_hash": self.next_frame.content_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class BBRuntimeReplay:
    initial: BBRuntimeFrame
    transitions: tuple[BBRuntimeTransition, ...]
    final: BBRuntimeFrame

    def __post_init__(self) -> None:
        _instance(self.initial, BBRuntimeFrame, label="initial frame")
        _instance(self.final, BBRuntimeFrame, label="final frame")
        _instance(self.transitions, tuple, label="transitions")
        if self.initial.tick_count != 0:
            raise ValueError("replay must begin at genesis; arbitrary resume is unsupported")
        current = self.initial
        for transition in self.transitions:
            _instance(transition, BBRuntimeTransition, label="transition")
            if not _same_content(transition.prior, current):
                raise ValueError("replay frame chain must be continuous in exact content")
            current = transition.next_frame
        if not _same_content(self.final, current):
            raise ValueError("replay final frame must equal the exact chain tail")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "initial": self.initial.to_dict(),
            "transitions": [
                {"transition": step.to_dict(), "next_frame": step.next_frame.to_dict()}
                for step in self.transitions
            ],
            "final_frame_hash": self.final.content_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class BBRuntimeError:
    """One failure location and cause, with no successful-prefix payload."""

    stage: str
    code: str
    field: str | None = None
    agent_index: int | None = None
    tick_index: int | None = None
    birth_index: int | None = None
    bb_cause: str | None = None
    expected: int | None = None
    actual: int | None = None

    def __post_init__(self) -> None:
        _text(self.stage, label="error stage")
        _text(self.code, label="error code")
        if self.field is not None:
            _text(self.field, label="error field")
        if self.bb_cause is not None:
            _text(self.bb_cause, label="underlying BB cause")
            if self.stage not in ("seed_network", "tick_network") or self.bb_cause != self.code:
                raise ValueError("underlying BB cause must match the network error code")
        for field in ("agent_index", "tick_index", "birth_index", "expected", "actual"):
            value = getattr(self, field)
            if value is not None:
                _nonnegative_integer(value, label=field)
        if (self.tick_index is None) != (self.birth_index is None):
            raise ValueError("tick and birth error indices must be supplied together")
        if (self.expected is None) != (self.actual is None):
            raise ValueError("expected and actual counts must be supplied together")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "code": self.code,
            "field": self.field,
            "agent_index": self.agent_index,
            "tick_index": self.tick_index,
            "birth_index": self.birth_index,
            "bb_cause": self.bb_cause,
            "expected": self.expected,
            "actual": self.actual,
        }


__all__ = (
    "BBRuntimeRawSeed",
    "BBRuntimeAgent",
    "BBRuntimeNewborn",
    "BBRuntimeBirth",
    "BBRuntimeTick",
    "BBRuntimeSeed",
    "BBRuntimeConfig",
    "BBRuntimeTopology",
    "BBRuntimeFrame",
    "BBRuntimeTransition",
    "BBRuntimeReplay",
    "BBRuntimeError",
)

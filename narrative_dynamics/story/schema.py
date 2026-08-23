from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType

from narrative_dynamics.contracts import stable_content_hash


NARRATIVE_CASE_SCHEMA_VERSION = 1


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot contain surrounding whitespace")
    return value


def _integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _freeze(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise TypeError(f"{label} does not admit floating-point values in V1 metadata")
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} keys must be non-empty strings")
            frozen[key] = _freeze(value[key], label=f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{label} values must be canonical JSON-like values")


def _freeze_mapping(value: Mapping[str, object], *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen = _freeze(value, label=label)
    assert isinstance(frozen, Mapping)
    return frozen


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _identifiers(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    identifiers = tuple(_text(value, label=label) for value in values)
    if not identifiers:
        raise ValueError(f"{label} values must be non-empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{label} values must be unique")
    return identifiers


def _exact_keys(value: object, expected: set[str], *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    keys = set(value)
    if keys != expected:
        raise ValueError(
            f"{label} keys must be exactly {sorted(expected)!r}; got {sorted(keys)!r}"
        )
    return value


@dataclass(frozen=True)
class StoryEntitiesV1:
    agents: tuple[str, ...]
    objects: tuple[str, ...]
    locations: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "agents", _identifiers(self.agents, label="agents"))
        object.__setattr__(self, "objects", _identifiers(self.objects, label="objects"))
        object.__setattr__(self, "locations", _identifiers(self.locations, label="locations"))

    def to_dict(self) -> dict[str, object]:
        return {
            "agents": list(self.agents),
            "objects": list(self.objects),
            "locations": list(self.locations),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> StoryEntitiesV1:
        data = _exact_keys(data, {"agents", "objects", "locations"}, label="entities")
        for key in ("agents", "objects", "locations"):
            if not isinstance(data[key], (list, tuple)):
                raise TypeError(f"entities {key} must be a sequence")
        return cls(
            agents=tuple(data["agents"]),
            objects=tuple(data["objects"]),
            locations=tuple(data["locations"]),
        )


@dataclass(frozen=True)
class RelocationEventV1:
    id: str
    logical_time: int
    actor: str
    object: str
    from_location: str | None
    to_location: str
    kind: str = "relocate_object"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, label="event id"))
        object.__setattr__(self, "logical_time", _integer(self.logical_time, label="event logical time"))
        object.__setattr__(self, "actor", _text(self.actor, label="event actor"))
        object.__setattr__(self, "object", _text(self.object, label="event object"))
        if self.from_location is not None:
            object.__setattr__(
                self,
                "from_location",
                _text(self.from_location, label="event from_location"),
            )
        object.__setattr__(self, "to_location", _text(self.to_location, label="event to_location"))
        if self.kind != "relocate_object":
            raise ValueError("event kind must be relocate_object")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "logical_time": self.logical_time,
            "actor": self.actor,
            "object": self.object,
            "from_location": self.from_location,
            "to_location": self.to_location,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RelocationEventV1:
        data = _exact_keys(
            data,
            {"id", "logical_time", "actor", "object", "from_location", "to_location", "kind"},
            label="event",
        )
        return cls(
            id=data["id"],
            logical_time=data["logical_time"],
            actor=data["actor"],
            object=data["object"],
            from_location=data["from_location"],
            to_location=data["to_location"],
            kind=data["kind"],
        )


@dataclass(frozen=True)
class DirectObservationV1:
    event: str
    agent: str
    channel: str = "direct_perception"

    def __post_init__(self) -> None:
        object.__setattr__(self, "event", _text(self.event, label="observation event"))
        object.__setattr__(self, "agent", _text(self.agent, label="observation agent"))
        if self.channel != "direct_perception":
            raise ValueError("observation channel must be direct_perception")

    def to_dict(self) -> dict[str, object]:
        return {"event": self.event, "agent": self.agent, "channel": self.channel}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DirectObservationV1:
        data = _exact_keys(data, {"event", "agent", "channel"}, label="observation")
        return cls(event=data["event"], agent=data["agent"], channel=data["channel"])


@dataclass(frozen=True)
class SearchActionV1:
    id: str
    location: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, label="action id"))
        object.__setattr__(self, "location", _text(self.location, label="action location"))

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "location": self.location}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SearchActionV1:
        data = _exact_keys(data, {"id", "location"}, label="action")
        return cls(id=data["id"], location=data["location"])


@dataclass(frozen=True)
class SearchDecisionV1:
    id: str
    time: int
    actor: str
    object: str
    actions: tuple[SearchActionV1, ...]
    kind: str = "search_object"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, label="decision id"))
        object.__setattr__(self, "time", _integer(self.time, label="decision time"))
        object.__setattr__(self, "actor", _text(self.actor, label="decision actor"))
        object.__setattr__(self, "object", _text(self.object, label="decision object"))
        actions = tuple(self.actions)
        if any(not isinstance(action, SearchActionV1) for action in actions):
            raise TypeError("decision actions must be SearchActionV1 values")
        object.__setattr__(self, "actions", actions)
        if self.kind != "search_object":
            raise ValueError("decision kind must be search_object")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "time": self.time,
            "actor": self.actor,
            "object": self.object,
            "actions": [action.to_dict() for action in self.actions],
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SearchDecisionV1:
        data = _exact_keys(data, {"id", "time", "actor", "object", "actions", "kind"}, label="decision")
        actions = data["actions"]
        if not isinstance(actions, (list, tuple)):
            raise TypeError("decision actions must be a sequence")
        return cls(
            id=data["id"],
            time=data["time"],
            actor=data["actor"],
            object=data["object"],
            actions=tuple(SearchActionV1.from_dict(action) for action in actions),
            kind=data["kind"],
        )


@dataclass(frozen=True)
class NarrativeOracleV1:
    objective_location: str
    actor_subjective_location: str
    agent_belief_ranking: tuple[str, ...]
    omniscient_ranking: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective_location",
            _text(self.objective_location, label="oracle objective location"),
        )
        object.__setattr__(
            self,
            "actor_subjective_location",
            _text(self.actor_subjective_location, label="oracle actor subjective location"),
        )
        object.__setattr__(
            self,
            "agent_belief_ranking",
            _identifiers(self.agent_belief_ranking, label="oracle agent belief ranking"),
        )
        object.__setattr__(
            self,
            "omniscient_ranking",
            _identifiers(self.omniscient_ranking, label="oracle omniscient ranking"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "objective_location": self.objective_location,
            "actor_subjective_location": self.actor_subjective_location,
            "agent_belief_ranking": list(self.agent_belief_ranking),
            "omniscient_ranking": list(self.omniscient_ranking),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> NarrativeOracleV1:
        data = _exact_keys(
            data,
            {"objective_location", "actor_subjective_location", "agent_belief_ranking", "omniscient_ranking"},
            label="oracle",
        )
        for key in ("agent_belief_ranking", "omniscient_ranking"):
            if not isinstance(data[key], (list, tuple)):
                raise TypeError(f"oracle {key} must be a sequence")
        return cls(
            objective_location=data["objective_location"],
            actor_subjective_location=data["actor_subjective_location"],
            agent_belief_ranking=tuple(data["agent_belief_ranking"]),
            omniscient_ranking=tuple(data["omniscient_ranking"]),
        )


def _validate_story_semantics(
    entities: StoryEntitiesV1,
    events: tuple[RelocationEventV1, ...],
    observations: tuple[DirectObservationV1, ...],
    decision: SearchDecisionV1,
    *,
    oracle: NarrativeOracleV1 | None = None,
) -> None:
    event_ids = tuple(event.id for event in events)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("event ids must be unique")

    prior_time: int | None = None
    objective: dict[str, str] = {}
    events_by_id: dict[str, RelocationEventV1] = {}
    for event in events:
        if event.actor not in entities.agents:
            raise ValueError("event actor must reference a declared agent")
        if event.object not in entities.objects:
            raise ValueError("event object must reference a declared object")
        if event.to_location not in entities.locations:
            raise ValueError("event location must reference a declared location")
        if event.from_location is not None and event.from_location not in entities.locations:
            raise ValueError("event from_location must reference a declared location")
        if prior_time is not None and event.logical_time <= prior_time:
            raise ValueError("event logical time must be strictly increasing")
        expected_from = objective.get(event.object)
        if event.from_location != expected_from:
            raise ValueError(
                "event from_location must match the current objective location"
            )
        objective[event.object] = event.to_location
        prior_time = event.logical_time
        events_by_id[event.id] = event

    observation_pairs: set[tuple[str, str]] = set()
    for observation in observations:
        if observation.event not in events_by_id:
            raise ValueError("observation event must reference a declared event")
        if observation.agent not in entities.agents:
            raise ValueError("observation agent must reference a declared agent")
        pair = (observation.event, observation.agent)
        if pair in observation_pairs:
            raise ValueError("observation duplicate event-agent pair is not allowed")
        observation_pairs.add(pair)

    if decision.actor not in entities.agents:
        raise ValueError("decision actor must reference a declared agent")
    if decision.object not in entities.objects:
        raise ValueError("decision object must reference a declared object")
    if events and decision.time <= max(event.logical_time for event in events):
        raise ValueError("decision time must be strictly after every story event")
    if decision.object not in objective:
        raise ValueError("decision object must have a known objective location")
    if len(decision.actions) < 2:
        raise ValueError("decision must contain at least two actions")
    action_ids = tuple(action.id for action in decision.actions)
    if len(set(action_ids)) != len(action_ids):
        raise ValueError("action ids must be unique")
    action_locations = tuple(action.location for action in decision.actions)
    if len(set(action_locations)) != len(action_locations):
        raise ValueError("action locations must be unique")
    for action in decision.actions:
        if action.location not in entities.locations:
            raise ValueError("action location must reference a declared location")

    observed_target = any(
        observation.agent == decision.actor
        and events_by_id[observation.event].object == decision.object
        and events_by_id[observation.event].logical_time < decision.time
        for observation in observations
    )
    if not observed_target:
        raise ValueError(
            "decision actor must have observed a relocation of the target object"
        )

    if oracle is not None:
        if oracle.objective_location not in entities.locations:
            raise ValueError("oracle objective location must be declared")
        if oracle.actor_subjective_location not in entities.locations:
            raise ValueError("oracle subjective location must be declared")
        action_set = set(action_ids)
        if set(oracle.agent_belief_ranking) != action_set or len(oracle.agent_belief_ranking) != len(action_ids):
            raise ValueError("oracle agent belief ranking must contain exactly the decision actions")
        if set(oracle.omniscient_ranking) != action_set or len(oracle.omniscient_ranking) != len(action_ids):
            raise ValueError("oracle omniscient ranking must contain exactly the decision actions")


@dataclass(frozen=True)
class NarrativeCaseV1:
    name: str
    version: str
    source: Mapping[str, object]
    provenance: Mapping[str, object]
    source_text: str
    entities: StoryEntitiesV1
    events: tuple[RelocationEventV1, ...]
    observations: tuple[DirectObservationV1, ...]
    decision: SearchDecisionV1
    oracle: NarrativeOracleV1
    schema_version: int = field(default=NARRATIVE_CASE_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="narrative case name"))
        object.__setattr__(self, "version", _text(self.version, label="narrative case version"))
        object.__setattr__(self, "source_text", _text(self.source_text, label="narrative source text"))
        if self.schema_version != NARRATIVE_CASE_SCHEMA_VERSION or isinstance(self.schema_version, bool):
            raise ValueError("unsupported narrative schema version")
        if not isinstance(self.entities, StoryEntitiesV1):
            raise TypeError("narrative entities must be StoryEntitiesV1")
        events = tuple(self.events)
        observations = tuple(self.observations)
        if any(not isinstance(event, RelocationEventV1) for event in events):
            raise TypeError("narrative events must be RelocationEventV1 values")
        if any(not isinstance(observation, DirectObservationV1) for observation in observations):
            raise TypeError("narrative observations must be DirectObservationV1 values")
        if not isinstance(self.decision, SearchDecisionV1):
            raise TypeError("narrative decision must be SearchDecisionV1")
        if not isinstance(self.oracle, NarrativeOracleV1):
            raise TypeError("narrative oracle must be NarrativeOracleV1")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "source", _freeze_mapping(self.source, label="narrative source"))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance, label="narrative provenance"))
        _validate_story_semantics(
            self.entities,
            events,
            observations,
            self.decision,
            oracle=self.oracle,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "source": _thaw(self.source),
            "provenance": _thaw(self.provenance),
            "source_text": self.source_text,
            "entities": self.entities.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "observations": [observation.to_dict() for observation in self.observations],
            "decision": self.decision.to_dict(),
            "oracle": self.oracle.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    def to_dict(self, include_content_hash: bool = True) -> dict[str, object]:
        payload = self.identity_payload()
        if include_content_hash:
            payload["content_hash"] = self.content_hash
        return payload

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
        *,
        verify_declared_hash: bool = True,
    ) -> NarrativeCaseV1:
        base_keys = {
            "schema_version", "name", "version", "source", "provenance",
            "source_text", "entities", "events", "observations", "decision", "oracle",
        }
        if not isinstance(data, Mapping):
            raise TypeError("narrative case must be a mapping")
        keys = set(data)
        allowed = base_keys | {"content_hash"}
        if not keys.issubset(allowed) or not base_keys.issubset(keys):
            raise ValueError("narrative case keys must match the V1 schema")
        if verify_declared_hash and "content_hash" not in data:
            raise ValueError("narrative case content hash is required")
        entities = StoryEntitiesV1.from_dict(data["entities"])
        raw_events = data["events"]
        raw_observations = data["observations"]
        if not isinstance(raw_events, (list, tuple)):
            raise TypeError("narrative events must be a sequence")
        if not isinstance(raw_observations, (list, tuple)):
            raise TypeError("narrative observations must be a sequence")
        case = cls(
            name=data["name"],
            version=data["version"],
            source=data["source"],
            provenance=data["provenance"],
            source_text=data["source_text"],
            entities=entities,
            events=tuple(RelocationEventV1.from_dict(event) for event in raw_events),
            observations=tuple(
                DirectObservationV1.from_dict(observation)
                for observation in raw_observations
            ),
            decision=SearchDecisionV1.from_dict(data["decision"]),
            oracle=NarrativeOracleV1.from_dict(data["oracle"]),
            schema_version=data["schema_version"],
        )
        if verify_declared_hash:
            declared = data["content_hash"]
            if not isinstance(declared, str) or declared != case.content_hash:
                raise ValueError("narrative case content hash does not match payload")
        return case


def load_narrative_case(path: str | Path) -> NarrativeCaseV1:
    location = Path(path)
    raw = json.loads(location.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("narrative fixture root must be a mapping")
    return NarrativeCaseV1.from_dict(raw)

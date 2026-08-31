"""Immutable, content-addressed contracts for deterministic narrative projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from types import MappingProxyType

from narrative_dynamics.contracts import stable_content_hash


_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    return None if value is None else _text(value, label=label)


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _nonnegative_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _unit_interval(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be finite and between zero and one")
    return number


def _nonnegative_finite(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _tuple_of_text(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be a tuple")
    return tuple(_text(item, label=label[:-1] if label.endswith("s") else label) for item in value)


class NarrativeAuthority(str, Enum):
    OBJECTIVE = "objective"
    AGENT_LIMITED = "agent_limited"
    MULTI_POV = "multi_pov"


class NarrativeTemporalOrder(str, Enum):
    CHRONOLOGICAL = "chronological"
    AUTHORED = "authored"


class NarrativeBeatKind(str, Enum):
    PHYSICAL = "physical"
    INFORMATION = "information"
    BELIEF_SHIFT = "belief_shift"
    ACTION_REVERSAL = "action_reversal"
    MEMORY_RECALL = "memory_recall"
    CLAIM_REVISION = "claim_revision"
    RELATIONSHIP_CHANGE = "relationship_change"
    CAUSAL_PAYOFF = "causal_payoff"


class NarrativeEntitlementScope(str, Enum):
    OBJECTIVE = "objective"
    PRIVATE = "private"


@dataclass(frozen=True)
class NarrativeSupportRef:
    artifact_kind: str
    artifact_id: str
    artifact_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_kind", _text(self.artifact_kind, label="narrative support artifact kind"))
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, label="narrative support artifact id"))
        object.__setattr__(self, "artifact_hash", _hash(self.artifact_hash, label="narrative support artifact hash"))

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_id": self.artifact_id,
            "artifact_hash": self.artifact_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NarrativeFact:
    key: str
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _text(self.key, label="narrative fact key"))
        object.__setattr__(self, "value", _text(self.value, label="narrative fact value"))

    def to_dict(self) -> dict[str, object]:
        return {"key": self.key, "value": self.value}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NarrativeEntitlement:
    entitlement_id: str
    scope: NarrativeEntitlementScope
    owner_agent_id: str | None
    round_index: int
    facts: tuple[NarrativeFact, ...]
    supporting_artifacts: tuple[NarrativeSupportRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entitlement_id", _text(self.entitlement_id, label="narrative entitlement id"))
        if not isinstance(self.scope, NarrativeEntitlementScope):
            raise TypeError("narrative entitlement scope must be NarrativeEntitlementScope")
        owner_agent_id = _optional_text(self.owner_agent_id, label="narrative entitlement owner")
        if self.scope is NarrativeEntitlementScope.PRIVATE and owner_agent_id is None:
            raise ValueError("private narrative entitlement requires an owner")
        if self.scope is NarrativeEntitlementScope.OBJECTIVE and owner_agent_id is not None:
            raise ValueError("objective narrative entitlement rejects an owner")
        object.__setattr__(self, "owner_agent_id", owner_agent_id)
        object.__setattr__(self, "round_index", _nonnegative_integer(self.round_index, label="narrative entitlement round index"))
        if not isinstance(self.facts, tuple) or any(not isinstance(item, NarrativeFact) for item in self.facts):
            raise TypeError("narrative entitlement facts must be a tuple of NarrativeFact values")
        fact_keys = tuple(item.key for item in self.facts)
        if len(set(fact_keys)) != len(fact_keys):
            raise ValueError("narrative entitlement fact keys must be unique")
        if not isinstance(self.supporting_artifacts, tuple) or any(not isinstance(item, NarrativeSupportRef) for item in self.supporting_artifacts):
            raise TypeError("narrative entitlement supporting artifacts must be a tuple of NarrativeSupportRef values")
        support_ids = tuple((item.artifact_kind, item.artifact_id) for item in self.supporting_artifacts)
        if len(set(support_ids)) != len(support_ids):
            raise ValueError("narrative entitlement supporting artifacts must be unique")
        object.__setattr__(self, "facts", tuple(sorted(self.facts, key=lambda item: (item.key, item.value))))
        object.__setattr__(self, "supporting_artifacts", tuple(sorted(self.supporting_artifacts, key=lambda item: (item.artifact_kind, item.artifact_id, item.artifact_hash))))

    def to_dict(self) -> dict[str, object]:
        return {
            "entitlement_id": self.entitlement_id,
            "scope": self.scope.value,
            "owner_agent_id": self.owner_agent_id,
            "round_index": self.round_index,
            "facts": [item.to_dict() for item in self.facts],
            "supporting_artifacts": [item.to_dict() for item in self.supporting_artifacts],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NarrativeProjectionPolicy:
    policy_id: str
    version: str
    authority: NarrativeAuthority
    pov_agent_ids: tuple[str, ...] = ()
    temporal_order: NarrativeTemporalOrder = NarrativeTemporalOrder.CHRONOLOGICAL
    authored_event_order: tuple[str, ...] = ()
    salience_weights: Mapping[NarrativeBeatKind, float] = field(default_factory=dict)
    minimum_salience: float = 0.0
    maximum_event_omission_gap: int = 0
    required_causal_coverage: float = 1.0
    scene_round_gap: int = 1
    maximum_scene_beats: int = 8
    include_beliefs: bool = True
    include_memories: bool = True
    include_claim_revisions: bool = True
    include_relationship_changes: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, label="narrative policy id"))
        object.__setattr__(self, "version", _text(self.version, label="narrative policy version"))
        if not isinstance(self.authority, NarrativeAuthority):
            raise TypeError("narrative policy authority must be NarrativeAuthority")
        if not isinstance(self.temporal_order, NarrativeTemporalOrder):
            raise TypeError("narrative policy temporal order must be NarrativeTemporalOrder")
        pov_agent_ids = _tuple_of_text(self.pov_agent_ids, label="narrative policy POV agent ids")
        if self.authority is NarrativeAuthority.MULTI_POV and len(set(pov_agent_ids)) < 2:
            raise ValueError("multi-POV narrative policy requires at least two unique POV agents")
        if len(set(pov_agent_ids)) != len(pov_agent_ids):
            raise ValueError("narrative policy POV agent ids must be unique")
        if self.authority is NarrativeAuthority.OBJECTIVE and pov_agent_ids:
            raise ValueError("objective narrative policy rejects POV agents")
        if self.authority is NarrativeAuthority.AGENT_LIMITED and len(pov_agent_ids) != 1:
            raise ValueError("limited narrative policy requires exactly one POV agent")
        if self.authority is NarrativeAuthority.MULTI_POV and len(pov_agent_ids) < 2:
            raise ValueError("multi-POV narrative policy requires at least two unique POV agents")
        object.__setattr__(self, "pov_agent_ids", tuple(sorted(pov_agent_ids)))
        authored_event_order = _tuple_of_text(self.authored_event_order, label="narrative policy authored event order")
        if len(set(authored_event_order)) != len(authored_event_order):
            raise ValueError("narrative policy authored event order ids must be unique")
        if self.temporal_order is NarrativeTemporalOrder.CHRONOLOGICAL and authored_event_order:
            raise ValueError("chronological narrative policy rejects authored event order")
        object.__setattr__(self, "authored_event_order", authored_event_order)
        if not isinstance(self.salience_weights, Mapping):
            raise TypeError("narrative policy salience weights must be a mapping")
        weights: dict[NarrativeBeatKind, float] = {}
        for kind, weight in self.salience_weights.items():
            if not isinstance(kind, NarrativeBeatKind):
                raise TypeError("narrative policy salience weight keys must be NarrativeBeatKind")
            weights[kind] = _nonnegative_finite(weight, label="narrative policy salience weight")
        object.__setattr__(self, "salience_weights", MappingProxyType(dict(sorted(weights.items(), key=lambda item: item[0].value))))
        object.__setattr__(self, "minimum_salience", _unit_interval(self.minimum_salience, label="narrative policy minimum salience"))
        object.__setattr__(self, "maximum_event_omission_gap", _nonnegative_integer(self.maximum_event_omission_gap, label="narrative policy maximum event omission gap"))
        object.__setattr__(self, "required_causal_coverage", _unit_interval(self.required_causal_coverage, label="narrative policy required causal coverage"))
        object.__setattr__(self, "scene_round_gap", _positive_integer(self.scene_round_gap, label="narrative policy scene round gap"))
        object.__setattr__(self, "maximum_scene_beats", _positive_integer(self.maximum_scene_beats, label="narrative policy maximum scene beats"))
        for name in ("include_beliefs", "include_memories", "include_claim_revisions", "include_relationship_changes"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"narrative policy {name.replace('_', ' ')} must be boolean")

    def weight_for(self, kind: NarrativeBeatKind) -> float:
        if not isinstance(kind, NarrativeBeatKind):
            raise TypeError("narrative beat kind must be NarrativeBeatKind")
        return self.salience_weights.get(kind, 1.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "authority": self.authority.value,
            "pov_agent_ids": list(self.pov_agent_ids),
            "temporal_order": self.temporal_order.value,
            "authored_event_order": list(self.authored_event_order),
            "salience_weights": {kind.value: self.salience_weights[kind] for kind in sorted(self.salience_weights, key=lambda item: item.value)},
            "minimum_salience": self.minimum_salience,
            "maximum_event_omission_gap": self.maximum_event_omission_gap,
            "required_causal_coverage": self.required_causal_coverage,
            "scene_round_gap": self.scene_round_gap,
            "maximum_scene_beats": self.maximum_scene_beats,
            "include_beliefs": self.include_beliefs,
            "include_memories": self.include_memories,
            "include_claim_revisions": self.include_claim_revisions,
            "include_relationship_changes": self.include_relationship_changes,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NarrativeBeat:
    beat_id: str
    kind: NarrativeBeatKind
    round_index: int
    sequence: int
    place_id: str
    active_pov_agent_id: str | None
    agent_ids: tuple[str, ...]
    salience: float
    supporting_artifacts: tuple[NarrativeSupportRef, ...]
    entitlement_ids: tuple[str, ...]
    cause_beat_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "beat_id", _text(self.beat_id, label="narrative beat id"))
        if not isinstance(self.kind, NarrativeBeatKind):
            raise TypeError("narrative beat kind must be NarrativeBeatKind")
        object.__setattr__(self, "round_index", _nonnegative_integer(self.round_index, label="narrative beat round index"))
        object.__setattr__(self, "sequence", _nonnegative_integer(self.sequence, label="narrative beat sequence"))
        object.__setattr__(self, "place_id", _text(self.place_id, label="narrative beat place id"))
        object.__setattr__(self, "active_pov_agent_id", _optional_text(self.active_pov_agent_id, label="narrative beat active POV agent id"))
        agent_ids = _tuple_of_text(self.agent_ids, label="narrative beat agent ids")
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("narrative beat agent ids must be unique")
        object.__setattr__(self, "agent_ids", tuple(sorted(agent_ids)))
        object.__setattr__(self, "salience", _nonnegative_finite(self.salience, label="narrative beat salience"))
        if not isinstance(self.supporting_artifacts, tuple) or any(not isinstance(item, NarrativeSupportRef) for item in self.supporting_artifacts):
            raise TypeError("narrative beat supporting artifacts must be a tuple of NarrativeSupportRef values")
        if not self.supporting_artifacts:
            raise ValueError("narrative beat requires supporting artifacts")
        support_ids = tuple((item.artifact_kind, item.artifact_id) for item in self.supporting_artifacts)
        if len(set(support_ids)) != len(support_ids):
            raise ValueError("narrative beat supporting artifacts must be unique")
        object.__setattr__(self, "supporting_artifacts", tuple(sorted(self.supporting_artifacts, key=lambda item: (item.artifact_kind, item.artifact_id, item.artifact_hash))))
        entitlement_ids = _tuple_of_text(self.entitlement_ids, label="narrative beat entitlement ids")
        if not entitlement_ids:
            raise ValueError("narrative beat requires entitlement ids")
        if len(set(entitlement_ids)) != len(entitlement_ids):
            raise ValueError("narrative beat entitlement ids must be unique")
        object.__setattr__(self, "entitlement_ids", tuple(sorted(entitlement_ids)))
        cause_beat_ids = _tuple_of_text(self.cause_beat_ids, label="narrative beat cause beat ids")
        if len(set(cause_beat_ids)) != len(cause_beat_ids):
            raise ValueError("narrative beat cause beat ids must be unique")
        if self.beat_id in cause_beat_ids:
            raise ValueError("narrative beat cannot cite itself as a cause")
        object.__setattr__(self, "cause_beat_ids", tuple(sorted(cause_beat_ids)))

    def to_dict(self) -> dict[str, object]:
        return {
            "beat_id": self.beat_id,
            "kind": self.kind.value,
            "round_index": self.round_index,
            "sequence": self.sequence,
            "place_id": self.place_id,
            "active_pov_agent_id": self.active_pov_agent_id,
            "agent_ids": list(self.agent_ids),
            "salience": self.salience,
            "supporting_artifacts": [item.to_dict() for item in self.supporting_artifacts],
            "entitlement_ids": list(self.entitlement_ids),
            "cause_beat_ids": list(self.cause_beat_ids),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NarrativeScene:
    scene_id: str
    beat_ids: tuple[str, ...]
    start_round_index: int
    end_round_index: int
    place_id: str
    active_pov_agent_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", _text(self.scene_id, label="narrative scene id"))
        beat_ids = _tuple_of_text(self.beat_ids, label="narrative scene beat ids")
        if not beat_ids:
            raise ValueError("narrative scene requires beat ids")
        if len(set(beat_ids)) != len(beat_ids):
            raise ValueError("narrative scene beat ids must be unique")
        object.__setattr__(self, "beat_ids", beat_ids)
        object.__setattr__(self, "start_round_index", _nonnegative_integer(self.start_round_index, label="narrative scene start round index"))
        object.__setattr__(self, "end_round_index", _nonnegative_integer(self.end_round_index, label="narrative scene end round index"))
        if self.end_round_index < self.start_round_index:
            raise ValueError("narrative scene end round index cannot precede start round index")
        object.__setattr__(self, "place_id", _text(self.place_id, label="narrative scene place id"))
        object.__setattr__(self, "active_pov_agent_id", _optional_text(self.active_pov_agent_id, label="narrative scene active POV agent id"))

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "beat_ids": list(self.beat_ids),
            "start_round_index": self.start_round_index,
            "end_round_index": self.end_round_index,
            "place_id": self.place_id,
            "active_pov_agent_id": self.active_pov_agent_id,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NarrativeCut:
    cut_id: str
    scene_ids: tuple[str, ...]
    entitlements: tuple[NarrativeEntitlement, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cut_id", _text(self.cut_id, label="narrative cut id"))
        scene_ids = _tuple_of_text(self.scene_ids, label="narrative cut scene ids")
        if len(set(scene_ids)) != len(scene_ids):
            raise ValueError("narrative cut scene ids must be unique")
        object.__setattr__(self, "scene_ids", scene_ids)
        if not isinstance(self.entitlements, tuple) or any(not isinstance(item, NarrativeEntitlement) for item in self.entitlements):
            raise TypeError("narrative cut entitlements must be a tuple of NarrativeEntitlement values")
        entitlement_ids = tuple(item.entitlement_id for item in self.entitlements)
        if len(set(entitlement_ids)) != len(entitlement_ids):
            raise ValueError("narrative cut entitlement ids must be unique")
        object.__setattr__(self, "entitlements", tuple(sorted(self.entitlements, key=lambda item: item.entitlement_id)))

    def to_dict(self) -> dict[str, object]:
        return {
            "cut_id": self.cut_id,
            "scene_ids": list(self.scene_ids),
            "entitlements": [item.to_dict() for item in self.entitlements],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class NarrativeProjection:
    story_hash: str
    trajectory_hash: str | None
    policy: NarrativeProjectionPolicy
    beats: tuple[NarrativeBeat, ...]
    scenes: tuple[NarrativeScene, ...]
    cut: NarrativeCut

    def __post_init__(self) -> None:
        object.__setattr__(self, "story_hash", _hash(self.story_hash, label="narrative projection story hash"))
        if self.trajectory_hash is not None:
            object.__setattr__(self, "trajectory_hash", _hash(self.trajectory_hash, label="narrative projection trajectory hash"))
        if not isinstance(self.policy, NarrativeProjectionPolicy):
            raise TypeError("narrative projection policy must be NarrativeProjectionPolicy")
        if not isinstance(self.beats, tuple) or any(not isinstance(item, NarrativeBeat) for item in self.beats):
            raise TypeError("narrative projection beats must be a tuple of NarrativeBeat values")
        beat_ids = tuple(item.beat_id for item in self.beats)
        if len(set(beat_ids)) != len(beat_ids):
            raise ValueError("narrative projection beat ids must be unique")
        if not isinstance(self.scenes, tuple) or any(not isinstance(item, NarrativeScene) for item in self.scenes):
            raise TypeError("narrative projection scenes must be a tuple of NarrativeScene values")
        scene_ids = tuple(item.scene_id for item in self.scenes)
        if len(set(scene_ids)) != len(scene_ids):
            raise ValueError("narrative projection scene ids must be unique")
        if not isinstance(self.cut, NarrativeCut):
            raise TypeError("narrative projection cut must be NarrativeCut")
        beat_map = {item.beat_id: item for item in self.beats}
        entitlement_map = {item.entitlement_id: item for item in self.cut.entitlements}
        for item in self.beats:
            missing_causes = set(item.cause_beat_ids).difference(beat_map)
            if missing_causes:
                raise ValueError("narrative beat cause ids must reference projected beats")
            missing_entitlements = set(item.entitlement_ids).difference(entitlement_map)
            if missing_entitlements:
                raise ValueError("narrative beat entitlement ids must be present in the cut")
            entitled_support = {
                (support.artifact_kind, support.artifact_id, support.artifact_hash)
                for entitlement_id in item.entitlement_ids
                for support in entitlement_map[entitlement_id].supporting_artifacts
            }
            for support in item.supporting_artifacts:
                if (support.artifact_kind, support.artifact_id, support.artifact_hash) not in entitled_support:
                    raise ValueError("narrative beat support must be present in its entitlement bundle")
        scene_membership: dict[str, int] = {beat_id: 0 for beat_id in beat_map}
        for scene in self.scenes:
            for beat_id in scene.beat_ids:
                if beat_id not in beat_map:
                    raise ValueError("narrative scene beat ids must reference projected beats")
                scene_membership[beat_id] += 1
        if any(count != 1 for count in scene_membership.values()):
            raise ValueError("every narrative beat must occur in exactly one scene")
        scene_map = {item.scene_id: item for item in self.scenes}
        missing_cut_scenes = set(self.cut.scene_ids).difference(scene_map)
        if missing_cut_scenes:
            raise ValueError("narrative cut scene ids must reference projected scenes")
        cut_membership: dict[str, int] = {scene_id: 0 for scene_id in scene_map}
        for scene_id in self.cut.scene_ids:
            cut_membership[scene_id] += 1
        if any(count != 1 for count in cut_membership.values()):
            raise ValueError("every narrative scene must occur in exactly one cut")
        self._validate_authority(entitlement_map)

    def _validate_authority(self, entitlement_map: Mapping[str, NarrativeEntitlement]) -> None:
        if self.policy.authority is NarrativeAuthority.OBJECTIVE:
            return
        authorized = set(self.policy.pov_agent_ids)
        for entitlement in entitlement_map.values():
            if entitlement.scope is not NarrativeEntitlementScope.PRIVATE or entitlement.owner_agent_id not in authorized:
                raise ValueError("limited and multi-POV cuts require private entitlements owned by authorized POV agents")
        for beat in self.beats:
            if beat.active_pov_agent_id is not None and beat.active_pov_agent_id not in authorized:
                raise ValueError("narrative beat active POV agent must be authorized by the policy")
        for scene in self.scenes:
            if scene.active_pov_agent_id is not None and scene.active_pov_agent_id not in authorized:
                raise ValueError("narrative scene active POV agent must be authorized by the policy")

    def to_dict(self) -> dict[str, object]:
        return {
            "story_hash": self.story_hash,
            "trajectory_hash": self.trajectory_hash,
            "policy": self.policy.to_dict(),
            "beats": [item.to_dict() for item in self.beats],
            "scenes": [item.to_dict() for item in self.scenes],
            "cut": self.cut.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


__all__ = (
    "NarrativeAuthority",
    "NarrativeTemporalOrder",
    "NarrativeBeatKind",
    "NarrativeEntitlementScope",
    "NarrativeSupportRef",
    "NarrativeFact",
    "NarrativeEntitlement",
    "NarrativeProjectionPolicy",
    "NarrativeBeat",
    "NarrativeScene",
    "NarrativeCut",
    "NarrativeProjection",
)

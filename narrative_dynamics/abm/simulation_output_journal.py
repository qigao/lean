"""Atomic public-only JSONL persistence for typed simulation output views."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Callable

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.simulation_output import filter_simulation_output
from narrative_dynamics.abm.simulation_output_contracts import (
    SIMULATION_OUTPUT_VIEW_SCHEMA,
    SimulationAgentDecisionPayload,
    SimulationAudienceCapability,
    SimulationBlenderDeltaPayload,
    SimulationCommandResultPayload,
    SimulationDiagnosticPayload,
    SimulationMemoryUpdatePayload,
    SimulationNarrativeScenePayload,
    SimulationNetworkMetricsPayload,
    SimulationObjectiveEventPayload,
    SimulationOutputAudience,
    SimulationOutputBatch,
    SimulationOutputKind,
    SimulationOutputPayload,
    SimulationOutputRecord,
    SimulationOutputView,
    SimulationPrivatePerceptPayload,
    SimulationSocialUpdatePayload,
    SimulationStateDeltaPayload,
    SimulationStoryProgressPayload,
)
from narrative_dynamics.abm.situated import SituatedActionIntent
from narrative_dynamics.abm.situated_cognition import SituatedCognitiveDecision
from narrative_dynamics.abm.situated_contracts import EvidenceFact
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkEmergenceMetrics,
)
from narrative_dynamics.abm.situated_perception_contracts import SituatedPercept
from narrative_dynamics.narrative.runtime_planning import PlanningBeliefState


SIMULATION_PUBLIC_JOURNAL_SCHEMA = (
    "narrative-dynamics.simulation-public-journal/v1"
)
SIMULATION_PUBLIC_JOURNAL_BATCH_SCHEMA = (
    "narrative-dynamics.simulation-public-journal-batch/v1"
)

_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_LINE_BYTES = 4 * 1024 * 1024
_MAX_TOTAL_BYTES = 64 * 1024 * 1024

_HEADER_KEYS = frozenset({
    "schema",
    "stream_id",
    "scenario_hash",
    "parent_journal_hash",
    "content_hash",
})
_BATCH_KEYS = frozenset({
    "schema",
    "stream_id",
    "scenario_hash",
    "prior_state_hash",
    "next_state_hash",
    "round_result_hash",
    "first_sequence",
    "last_sequence",
    "records",
    "source_batch_hash",
    "checkpoint",
    "view_hash",
})
_RECORD_KEYS = frozenset({
    "schema",
    "stream_id",
    "scenario_hash",
    "sequence",
    "round_index",
    "state_hash",
    "kind",
    "audience",
    "owner_agent_id",
    "source_artifact_hashes",
    "payload",
    "payload_hash",
    "record_hash",
})


def _content_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a content hash")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _mapping(
    value: object,
    keys: frozenset[str],
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} must match its exact schema")
    return value


def _list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def _payload_mapping(
    value: object,
    names: tuple[str, ...],
    *,
    label: str,
) -> dict[str, object]:
    return _mapping(value, frozenset(names), label=label)


def _trusted_metrics(
    value: object,
) -> SituatedNetworkEmergenceMetrics:
    if not isinstance(value, SituatedNetworkEmergenceMetrics):
        raise TypeError("network metrics payload requires typed metrics")
    return SituatedNetworkEmergenceMetrics(
        snapshot_hash=value.snapshot_hash,
        round_index=value.round_index,
        population_size=value.population_size,
        occupied_place_count=value.occupied_place_count,
        adopted_count=value.adopted_count,
        adoption_rate=value.adoption_rate,
        tracked_belief_mean=value.tracked_belief_mean,
        tracked_belief_variance=value.tracked_belief_variance,
        active_relationship_edge_count=value.active_relationship_edge_count,
        mean_relationship_trust=value.mean_relationship_trust,
        direct_interaction_pair_count=value.direct_interaction_pair_count,
        latest_tell_event_count=value.latest_tell_event_count,
        transmission_count=value.transmission_count,
        reached_observer_count=value.reached_observer_count,
        exact_transmission_count=value.exact_transmission_count,
        detected_transmission_count=value.detected_transmission_count,
        identified_transmission_count=value.identified_transmission_count,
        active_claim_count=value.active_claim_count,
        confirmed_claim_count=value.confirmed_claim_count,
        contradicted_claim_count=value.contradicted_claim_count,
        superseded_claim_count=value.superseded_claim_count,
        forgotten_claim_count=value.forgotten_claim_count,
    )


def _trusted_percept(value: object) -> SituatedPercept:
    if not isinstance(value, SituatedPercept):
        raise TypeError("private percept payload requires a typed percept")
    if not isinstance(value.details, tuple):
        raise TypeError("situated percept details must be a tuple")
    details = []
    for fact in value.details:
        if not isinstance(fact, EvidenceFact):
            raise TypeError("situated percept details require EvidenceFact values")
        details.append(EvidenceFact(fact.name, fact.value))
    return SituatedPercept(
        percept_id=value.percept_id,
        round_index=value.round_index,
        agent_id=value.agent_id,
        source_event_id=value.source_event_id,
        source_event_hash=value.source_event_hash,
        channels=value.channels,
        fidelity=value.fidelity,
        actor_agent_id=value.actor_agent_id,
        kind=value.kind,
        place_id=value.place_id,
        outcome=value.outcome,
        details=tuple(details),
    )


def _trusted_belief(value: object) -> PlanningBeliefState:
    if not isinstance(value, PlanningBeliefState):
        raise TypeError("situated decision requires typed belief states")
    return PlanningBeliefState(value.probabilities)


def _trusted_intent(value: object) -> SituatedActionIntent:
    if not isinstance(value, SituatedActionIntent):
        raise TypeError("situated decision requires a typed action intent")
    return SituatedActionIntent(
        action_id=value.action_id,
        agent_id=value.agent_id,
        kind=value.kind,
        target_id=value.target_id,
        message=value.message,
        source_event_ids=value.source_event_ids,
    )


def _trusted_decision(value: object) -> SituatedCognitiveDecision:
    if not isinstance(value, SituatedCognitiveDecision):
        raise TypeError("agent decision payload requires a typed decision")
    for name in ("admitted_observation_ids", "admitted_symbol_ids"):
        if not isinstance(getattr(value, name), tuple):
            raise TypeError(f"situated decision {name.replace('_', ' ')} must be a tuple")
    return SituatedCognitiveDecision(
        agent_id=value.agent_id,
        round_index=value.round_index,
        prior_belief=_trusted_belief(value.prior_belief),
        posterior_belief=_trusted_belief(value.posterior_belief),
        admitted_observation_ids=value.admitted_observation_ids,
        admitted_symbol_ids=value.admitted_symbol_ids,
        feasible_action_ids=value.feasible_action_ids,
        action_values=value.action_values,
        action_policy=value.action_policy,
        selected_action_id=value.selected_action_id,
        selected_goal_contributions=value.selected_goal_contributions,
        intent=_trusted_intent(value.intent),
        recalled_memory_ids=value.recalled_memory_ids,
        recalled_symbol_ids=value.recalled_symbol_ids,
    )


def _trusted_payload(value: object) -> SimulationOutputPayload:
    payload_type = type(value)
    if payload_type is SimulationStateDeltaPayload:
        return SimulationStateDeltaPayload(
            value.prior_snapshot_hash,
            value.next_snapshot_hash,
            value.changed_agent_ids,
            value.changed_passage_ids,
            value.changed_object_ids,
        )
    if payload_type is SimulationObjectiveEventPayload:
        return SimulationObjectiveEventPayload(
            value.event_id,
            value.event_hash,
            value.action_id,
            value.action_kind,
            value.actor_agent_id,
            value.place_id,
            value.target_id,
            value.success,
            value.cause_event_ids,
        )
    if payload_type is SimulationPrivatePerceptPayload:
        return SimulationPrivatePerceptPayload(_trusted_percept(value.percept))
    if payload_type is SimulationAgentDecisionPayload:
        return SimulationAgentDecisionPayload(_trusted_decision(value.decision))
    if payload_type is SimulationMemoryUpdatePayload:
        return SimulationMemoryUpdatePayload(
            value.agent_id,
            value.prior_mind_hash,
            value.next_mind_hash,
            value.recalled_memory_ids,
            value.admitted_memory_ids,
        )
    if payload_type is SimulationSocialUpdatePayload:
        return SimulationSocialUpdatePayload(
            value.observer_agent_id,
            value.prior_claim_hashes,
            value.next_claim_hashes,
            value.prior_relationship_hashes,
            value.next_relationship_hashes,
            value.admitted_evidence_ids,
        )
    if payload_type is SimulationNetworkMetricsPayload:
        return SimulationNetworkMetricsPayload(_trusted_metrics(value.metrics))
    if payload_type is SimulationStoryProgressPayload:
        return SimulationStoryProgressPayload(
            value.active_scene_id,
            value.completed_scene_ids,
            value.status,
        )
    if payload_type is SimulationNarrativeScenePayload:
        return SimulationNarrativeScenePayload(
            value.scene_id,
            value.projection_hash,
            value.realization_hash,
        )
    if payload_type is SimulationBlenderDeltaPayload:
        return SimulationBlenderDeltaPayload(
            value.agent_places,
            value.passage_states,
            value.object_placements,
        )
    if payload_type is SimulationCommandResultPayload:
        return SimulationCommandResultPayload(
            value.command_id,
            value.accepted,
            value.reason_code,
        )
    if payload_type is SimulationDiagnosticPayload:
        return SimulationDiagnosticPayload(value.code, value.message)
    raise TypeError("output payload must use an exact supported base type")


def _trusted_record(value: object) -> SimulationOutputRecord:
    if not isinstance(value, SimulationOutputRecord):
        raise TypeError("public journal requires typed output records")
    return SimulationOutputRecord(
        value.stream_id,
        value.scenario_hash,
        value.sequence,
        value.round_index,
        value.state_hash,
        value.kind,
        value.audience,
        value.owner_agent_id,
        value.source_artifact_hashes,
        _trusted_payload(value.payload),
        schema=value.schema,
    )


def _trusted_batch(value: object) -> SimulationOutputBatch:
    if not isinstance(value, SimulationOutputBatch):
        raise TypeError("public journal write requires a SimulationOutputBatch")
    if not isinstance(value.records, tuple):
        raise TypeError("output batch records must be a tuple")
    return SimulationOutputBatch(
        value.stream_id,
        value.scenario_hash,
        value.prior_state_hash,
        value.next_state_hash,
        value.round_result_hash,
        value.first_sequence,
        value.last_sequence,
        tuple(_trusted_record(record) for record in value.records),
        checkpoint=value.checkpoint,
        schema=value.schema,
    )


def _trusted_view(value: object) -> SimulationOutputView:
    if not isinstance(value, SimulationOutputView):
        raise TypeError("public journal requires typed output views")
    if not isinstance(value.records, tuple):
        raise TypeError("output view records must be a tuple")
    return SimulationOutputView(
        value.stream_id,
        value.scenario_hash,
        value.prior_state_hash,
        value.next_state_hash,
        value.round_result_hash,
        value.first_sequence,
        value.last_sequence,
        tuple(_trusted_record(record) for record in value.records),
        value.source_batch_hash,
        checkpoint=value.checkpoint,
        schema=value.schema,
    )


def _metrics_document(
    metrics: SituatedNetworkEmergenceMetrics,
) -> dict[str, object]:
    return {
        "snapshot_hash": metrics.snapshot_hash,
        "round_index": metrics.round_index,
        "population_size": metrics.population_size,
        "occupied_place_count": metrics.occupied_place_count,
        "adopted_count": metrics.adopted_count,
        "adoption_rate": metrics.adoption_rate,
        "tracked_belief_mean": metrics.tracked_belief_mean,
        "tracked_belief_variance": metrics.tracked_belief_variance,
        "active_relationship_edge_count": metrics.active_relationship_edge_count,
        "mean_relationship_trust": metrics.mean_relationship_trust,
        "direct_interaction_pair_count": metrics.direct_interaction_pair_count,
        "latest_tell_event_count": metrics.latest_tell_event_count,
        "transmission_count": metrics.transmission_count,
        "reached_observer_count": metrics.reached_observer_count,
        "exact_transmission_count": metrics.exact_transmission_count,
        "detected_transmission_count": metrics.detected_transmission_count,
        "identified_transmission_count": metrics.identified_transmission_count,
        "active_claim_count": metrics.active_claim_count,
        "confirmed_claim_count": metrics.confirmed_claim_count,
        "contradicted_claim_count": metrics.contradicted_claim_count,
        "superseded_claim_count": metrics.superseded_claim_count,
        "forgotten_claim_count": metrics.forgotten_claim_count,
    }


def _public_payload_document(
    payload: SimulationOutputPayload,
) -> dict[str, object]:
    payload_type = type(payload)
    if payload_type is SimulationStateDeltaPayload:
        return {
            "prior_snapshot_hash": payload.prior_snapshot_hash,
            "next_snapshot_hash": payload.next_snapshot_hash,
            "changed_agent_ids": list(payload.changed_agent_ids),
            "changed_passage_ids": list(payload.changed_passage_ids),
            "changed_object_ids": list(payload.changed_object_ids),
        }
    if payload_type is SimulationObjectiveEventPayload:
        return {
            "event_id": payload.event_id,
            "event_hash": payload.event_hash,
            "action_id": payload.action_id,
            "action_kind": payload.action_kind,
            "actor_agent_id": payload.actor_agent_id,
            "place_id": payload.place_id,
            "target_id": payload.target_id,
            "success": payload.success,
            "cause_event_ids": list(payload.cause_event_ids),
        }
    if payload_type is SimulationNetworkMetricsPayload:
        return {"metrics": _metrics_document(payload.metrics)}
    if payload_type is SimulationStoryProgressPayload:
        return {
            "active_scene_id": payload.active_scene_id,
            "completed_scene_ids": list(payload.completed_scene_ids),
            "status": payload.status,
        }
    if payload_type is SimulationNarrativeScenePayload:
        return {
            "scene_id": payload.scene_id,
            "projection_hash": payload.projection_hash,
            "realization_hash": payload.realization_hash,
        }
    if payload_type is SimulationBlenderDeltaPayload:
        return {
            "agent_places": [list(item) for item in payload.agent_places],
            "passage_states": [list(item) for item in payload.passage_states],
            "object_placements": [list(item) for item in payload.object_placements],
        }
    if payload_type is SimulationCommandResultPayload:
        return {
            "command_id": payload.command_id,
            "accepted": payload.accepted,
            "reason_code": payload.reason_code,
        }
    if payload_type is SimulationDiagnosticPayload:
        return {"code": payload.code, "message": payload.message}
    raise ValueError("public journal record kind is not public")


def _journal_hash_body(
    *,
    stream_id: str,
    scenario_hash: str,
    parent_journal_hash: str | None,
    batches: tuple[SimulationOutputView, ...],
) -> dict[str, object]:
    return {
        "schema": SIMULATION_PUBLIC_JOURNAL_SCHEMA,
        "stream_id": stream_id,
        "scenario_hash": scenario_hash,
        "parent_journal_hash": parent_journal_hash,
        "view_hashes": [batch.content_hash for batch in batches],
    }


@dataclass(frozen=True)
class SimulationPublicJournal:
    """One immutable, integrity-checked sequence of public output views."""

    stream_id: str
    scenario_hash: str
    parent_journal_hash: str | None
    batches: tuple[SimulationOutputView, ...]
    schema: str = SIMULATION_PUBLIC_JOURNAL_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stream_id",
            _text(self.stream_id, label="public journal stream id"),
        )
        object.__setattr__(
            self,
            "scenario_hash",
            _content_hash(
                self.scenario_hash,
                label="public journal scenario hash",
            ),
        )
        if self.parent_journal_hash is not None:
            object.__setattr__(
                self,
                "parent_journal_hash",
                _content_hash(
                    self.parent_journal_hash,
                    label="public journal parent hash",
                ),
            )
        if not isinstance(self.batches, tuple) or not self.batches or any(
            not isinstance(batch, SimulationOutputView) for batch in self.batches
        ):
            raise ValueError("public journal requires typed output views")
        object.__setattr__(
            self,
            "batches",
            tuple(_trusted_view(batch) for batch in self.batches),
        )
        if self.schema != SIMULATION_PUBLIC_JOURNAL_SCHEMA:
            raise ValueError("public journal schema must match the supported schema")
        previous: SimulationOutputView | None = None
        for batch in self.batches:
            if batch.stream_id != self.stream_id:
                raise ValueError("public journal output stream continuity")
            if batch.scenario_hash != self.scenario_hash:
                raise ValueError("public journal output scenario continuity")
            if any(
                record.audience is not SimulationOutputAudience.PUBLIC
                for record in batch.records
            ):
                raise ValueError("public journal accepts public output only")
            for record in batch.records:
                if record.kind not in _PUBLIC_PAYLOAD_DECODERS:
                    raise ValueError("public journal record kind is not public")
            if previous is not None:
                if batch.first_sequence != previous.last_sequence + 1:
                    raise ValueError("public journal source sequence continuity")
                if batch.prior_state_hash != previous.next_state_hash:
                    raise ValueError("public journal state continuity")
            previous = batch

    def to_dict(self) -> dict[str, object]:
        return _journal_hash_body(
            stream_id=self.stream_id,
            scenario_hash=self.scenario_hash,
            parent_journal_hash=self.parent_journal_hash,
            batches=self.batches,
        )

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _record_document(record: SimulationOutputRecord) -> dict[str, object]:
    payload_document = _public_payload_document(record.payload)
    document = {
        "schema": record.schema,
        "stream_id": record.stream_id,
        "scenario_hash": record.scenario_hash,
        "sequence": record.sequence,
        "round_index": record.round_index,
        "state_hash": record.state_hash,
        "kind": record.kind.value,
        "audience": record.audience.value,
        "owner_agent_id": record.owner_agent_id,
        "source_artifact_hashes": list(record.source_artifact_hashes),
        "payload": payload_document,
        "payload_hash": stable_content_hash(payload_document),
    }
    document["record_hash"] = stable_content_hash(document)
    return document


def _batch_document(view: SimulationOutputView) -> dict[str, object]:
    record_documents = [_record_document(record) for record in view.records]
    view_body = {
        "schema": view.schema,
        "stream_id": view.stream_id,
        "scenario_hash": view.scenario_hash,
        "prior_state_hash": view.prior_state_hash,
        "next_state_hash": view.next_state_hash,
        "round_result_hash": view.round_result_hash,
        "first_sequence": view.first_sequence,
        "last_sequence": view.last_sequence,
        "record_hashes": [
            document["record_hash"] for document in record_documents
        ],
        "source_batch_hash": view.source_batch_hash,
        "checkpoint": view.checkpoint,
    }
    return {
        "schema": SIMULATION_PUBLIC_JOURNAL_BATCH_SCHEMA,
        "stream_id": view.stream_id,
        "scenario_hash": view.scenario_hash,
        "prior_state_hash": view.prior_state_hash,
        "next_state_hash": view.next_state_hash,
        "round_result_hash": view.round_result_hash,
        "first_sequence": view.first_sequence,
        "last_sequence": view.last_sequence,
        "records": record_documents,
        "source_batch_hash": view.source_batch_hash,
        "checkpoint": view.checkpoint,
        "view_hash": stable_content_hash(view_body),
    }


def _encode_journal(journal: SimulationPublicJournal) -> bytes:
    batch_documents = tuple(
        _batch_document(batch) for batch in journal.batches
    )
    header_body = {
        "schema": journal.schema,
        "stream_id": journal.stream_id,
        "scenario_hash": journal.scenario_hash,
        "parent_journal_hash": journal.parent_journal_hash,
        "view_hashes": [
            document["view_hash"] for document in batch_documents
        ],
    }
    header = {
        "schema": journal.schema,
        "stream_id": journal.stream_id,
        "scenario_hash": journal.scenario_hash,
        "parent_journal_hash": journal.parent_journal_hash,
        "content_hash": stable_content_hash(header_body),
    }
    documents = (header,) + batch_documents
    return "".join(
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for document in documents
    ).encode("utf-8")


def _state_delta(value: object) -> SimulationOutputPayload:
    data = _payload_mapping(
        value,
        (
            "prior_snapshot_hash",
            "next_snapshot_hash",
            "changed_agent_ids",
            "changed_passage_ids",
            "changed_object_ids",
        ),
        label="state delta payload",
    )
    return SimulationStateDeltaPayload(
        data["prior_snapshot_hash"],
        data["next_snapshot_hash"],
        tuple(_list(data["changed_agent_ids"], label="changed agent ids")),
        tuple(_list(data["changed_passage_ids"], label="changed passage ids")),
        tuple(_list(data["changed_object_ids"], label="changed object ids")),
    )


def _objective_event(value: object) -> SimulationOutputPayload:
    data = _payload_mapping(
        value,
        (
            "event_id",
            "event_hash",
            "action_id",
            "action_kind",
            "actor_agent_id",
            "place_id",
            "target_id",
            "success",
            "cause_event_ids",
        ),
        label="objective event payload",
    )
    return SimulationObjectiveEventPayload(
        data["event_id"],
        data["event_hash"],
        data["action_id"],
        data["action_kind"],
        data["actor_agent_id"],
        data["place_id"],
        data["target_id"],
        data["success"],
        tuple(_list(data["cause_event_ids"], label="cause event ids")),
    )


def _network_metrics(value: object) -> SimulationOutputPayload:
    data = _payload_mapping(value, ("metrics",), label="network metrics payload")
    metric_names = tuple(SituatedNetworkEmergenceMetrics.__dataclass_fields__)
    metrics = _payload_mapping(
        data["metrics"],
        metric_names,
        label="network metrics",
    )
    return SimulationNetworkMetricsPayload(SituatedNetworkEmergenceMetrics(**metrics))


def _story_progress(value: object) -> SimulationOutputPayload:
    data = _payload_mapping(
        value,
        ("active_scene_id", "completed_scene_ids", "status"),
        label="story progress payload",
    )
    return SimulationStoryProgressPayload(
        data["active_scene_id"],
        tuple(_list(data["completed_scene_ids"], label="completed scene ids")),
        data["status"],
    )


def _narrative_scene(value: object) -> SimulationOutputPayload:
    data = _payload_mapping(
        value,
        ("scene_id", "projection_hash", "realization_hash"),
        label="narrative scene payload",
    )
    return SimulationNarrativeScenePayload(
        data["scene_id"],
        data["projection_hash"],
        data["realization_hash"],
    )


def _blender_delta(value: object) -> SimulationOutputPayload:
    data = _payload_mapping(
        value,
        ("agent_places", "passage_states", "object_placements"),
        label="Blender delta payload",
    )

    def tuples(name: str) -> tuple[tuple[object, ...], ...]:
        values = _list(data[name], label=f"Blender delta {name}")
        if any(not isinstance(item, list) for item in values):
            raise TypeError(f"Blender delta {name} entries must be lists")
        return tuple(tuple(item) for item in values)

    return SimulationBlenderDeltaPayload(
        tuples("agent_places"),
        tuples("passage_states"),
        tuples("object_placements"),
    )


def _command_result(value: object) -> SimulationOutputPayload:
    data = _payload_mapping(
        value,
        ("command_id", "accepted", "reason_code"),
        label="command result payload",
    )
    return SimulationCommandResultPayload(
        data["command_id"],
        data["accepted"],
        data["reason_code"],
    )


def _diagnostic(value: object) -> SimulationOutputPayload:
    data = _payload_mapping(
        value,
        ("code", "message"),
        label="diagnostic payload",
    )
    return SimulationDiagnosticPayload(data["code"], data["message"])


_PUBLIC_PAYLOAD_DECODERS: dict[
    SimulationOutputKind,
    Callable[[object], SimulationOutputPayload],
] = {
    SimulationOutputKind.STATE_DELTA: _state_delta,
    SimulationOutputKind.EVENT_OBJECTIVE: _objective_event,
    SimulationOutputKind.NETWORK_METRICS: _network_metrics,
    SimulationOutputKind.STORY_PROGRESS: _story_progress,
    SimulationOutputKind.NARRATIVE_SCENE: _narrative_scene,
    SimulationOutputKind.BLENDER_DELTA: _blender_delta,
    SimulationOutputKind.COMMAND_RESULT: _command_result,
    SimulationOutputKind.DIAGNOSTIC: _diagnostic,
}


def _decode_record(value: object) -> SimulationOutputRecord:
    data = _mapping(value, _RECORD_KEYS, label="public journal record")
    if data["audience"] != SimulationOutputAudience.PUBLIC.value:
        raise ValueError("public journal record audience must be public")
    kind = SimulationOutputKind(data["kind"])
    decoder = _PUBLIC_PAYLOAD_DECODERS.get(kind)
    if decoder is None:
        raise ValueError("public journal record kind is not public")
    payload = decoder(data["payload"])
    supplied_payload_hash = _content_hash(
        data["payload_hash"],
        label="public journal payload hash",
    )
    if supplied_payload_hash != payload.content_hash:
        raise ValueError("public journal payload hash mismatch")
    record = SimulationOutputRecord(
        data["stream_id"],
        data["scenario_hash"],
        data["sequence"],
        data["round_index"],
        data["state_hash"],
        kind,
        SimulationOutputAudience.PUBLIC,
        data["owner_agent_id"],
        tuple(
            _list(
                data["source_artifact_hashes"],
                label="source artifact hashes",
            )
        ),
        payload,
        schema=data["schema"],
    )
    supplied_record_hash = _content_hash(
        data["record_hash"],
        label="public journal record hash",
    )
    if supplied_record_hash != record.content_hash:
        raise ValueError("public journal record hash mismatch")
    return record


def _decode_batch(value: object) -> SimulationOutputView:
    data = _mapping(value, _BATCH_KEYS, label="public journal batch")
    if data["schema"] != SIMULATION_PUBLIC_JOURNAL_BATCH_SCHEMA:
        raise ValueError("public journal batch schema mismatch")
    records = tuple(
        _decode_record(record)
        for record in _list(data["records"], label="public journal records")
    )
    view = SimulationOutputView(
        data["stream_id"],
        data["scenario_hash"],
        data["prior_state_hash"],
        data["next_state_hash"],
        data["round_result_hash"],
        data["first_sequence"],
        data["last_sequence"],
        records,
        data["source_batch_hash"],
        data["checkpoint"],
        schema=SIMULATION_OUTPUT_VIEW_SCHEMA,
    )
    supplied_view_hash = _content_hash(
        data["view_hash"],
        label="public journal view hash",
    )
    if supplied_view_hash != view.content_hash:
        raise ValueError("public journal view hash mismatch")
    return view


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_: str) -> object:
    raise ValueError("non-finite JSON constant")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


def _load_documents(path: Path) -> list[dict[str, object]]:
    if path.stat().st_size > _MAX_TOTAL_BYTES:
        raise ValueError("public journal exceeds total size policy")
    raw = path.read_bytes()
    if not raw or len(raw) > _MAX_TOTAL_BYTES or not raw.endswith(b"\n"):
        raise ValueError("public journal is empty, oversized, or truncated")
    lines = raw.splitlines(keepends=True)
    if any(len(line) > _MAX_LINE_BYTES for line in lines):
        raise ValueError("public journal line exceeds size policy")
    text_lines = [line.decode("utf-8") for line in lines]
    documents: list[dict[str, object]] = []
    for line in text_lines:
        value = json.loads(
            line,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
        if not isinstance(value, dict):
            raise ValueError("public journal lines must be JSON objects")
        documents.append(value)
    return documents


def replay_public_simulation_journal(
    path: str | os.PathLike[str],
) -> SimulationPublicJournal:
    """Strictly replay one bounded public JSONL journal without external calls."""

    try:
        documents = _load_documents(Path(path))
        header = _mapping(
            documents[0],
            _HEADER_KEYS,
            label="public journal header",
        )
        if header["schema"] != SIMULATION_PUBLIC_JOURNAL_SCHEMA:
            raise ValueError("public journal header schema mismatch")
        batches = tuple(_decode_batch(document) for document in documents[1:])
        journal = SimulationPublicJournal(
            header["stream_id"],
            header["scenario_hash"],
            header["parent_journal_hash"],
            batches,
            schema=header["schema"],
        )
        supplied_hash = _content_hash(
            header["content_hash"],
            label="public journal header content hash",
        )
        if supplied_hash != journal.content_hash:
            raise ValueError("public journal header content hash mismatch")
        return journal
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        OverflowError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        raise ValueError("invalid public simulation journal") from None


def _validate_parent_hash(value: str | None) -> str | None:
    if value is None:
        return None
    return _content_hash(value, label="public journal parent hash")


def write_public_simulation_journal(
    path: str | os.PathLike[str],
    batch: SimulationOutputBatch,
    *,
    parent_journal_hash: str | None = None,
) -> SimulationPublicJournal:
    """Append a batch's public view by atomically replacing the complete JSONL file."""

    trusted_batch = _trusted_batch(batch)
    requested_parent = _validate_parent_hash(parent_journal_hash)
    destination = Path(path)
    if destination.exists():
        prior = replay_public_simulation_journal(destination)
        if requested_parent is not None and requested_parent != prior.parent_journal_hash:
            raise ValueError("public journal parent hash continuity")
        parent_hash = prior.parent_journal_hash
        prior_batches = prior.batches
    else:
        parent_hash = requested_parent
        prior_batches = ()

    public_view = filter_simulation_output(
        trusted_batch,
        SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
    )
    journal = SimulationPublicJournal(
        (
            trusted_batch.stream_id
            if not prior_batches
            else prior_batches[0].stream_id
        ),
        (
            trusted_batch.scenario_hash
            if not prior_batches
            else prior_batches[0].scenario_hash
        ),
        parent_hash,
        prior_batches + (public_view,),
    )
    encoded = _encode_journal(journal)
    if len(encoded) > _MAX_TOTAL_BYTES:
        raise ValueError("public journal exceeds total size policy")
    if any(
        len(line) > _MAX_LINE_BYTES
        for line in encoded.splitlines(keepends=True)
    ):
        raise ValueError("public journal line exceeds size policy")

    descriptor, stage_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    stage = Path(stage_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        stage.unlink(missing_ok=True)
    return journal


__all__ = (
    "SIMULATION_PUBLIC_JOURNAL_SCHEMA",
    "SIMULATION_PUBLIC_JOURNAL_BATCH_SCHEMA",
    "SimulationPublicJournal",
    "write_public_simulation_journal",
    "replay_public_simulation_journal",
)

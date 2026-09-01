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
    SimulationAudienceCapability,
    SimulationBlenderDeltaPayload,
    SimulationCommandResultPayload,
    SimulationDiagnosticPayload,
    SimulationNarrativeScenePayload,
    SimulationNetworkMetricsPayload,
    SimulationObjectiveEventPayload,
    SimulationOutputAudience,
    SimulationOutputBatch,
    SimulationOutputKind,
    SimulationOutputPayload,
    SimulationOutputRecord,
    SimulationOutputView,
    SimulationStateDeltaPayload,
    SimulationStoryProgressPayload,
)
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkEmergenceMetrics,
)


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
                SimulationOutputRecord(
                    record.stream_id,
                    record.scenario_hash,
                    record.sequence,
                    record.round_index,
                    record.state_hash,
                    record.kind,
                    record.audience,
                    record.owner_agent_id,
                    record.source_artifact_hashes,
                    record.payload,
                    schema=record.schema,
                )
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
    document = record.to_dict()
    document["record_hash"] = record.content_hash
    return document


def _batch_document(view: SimulationOutputView) -> dict[str, object]:
    return {
        "schema": SIMULATION_PUBLIC_JOURNAL_BATCH_SCHEMA,
        "stream_id": view.stream_id,
        "scenario_hash": view.scenario_hash,
        "prior_state_hash": view.prior_state_hash,
        "next_state_hash": view.next_state_hash,
        "round_result_hash": view.round_result_hash,
        "first_sequence": view.first_sequence,
        "last_sequence": view.last_sequence,
        "records": [_record_document(record) for record in view.records],
        "source_batch_hash": view.source_batch_hash,
        "checkpoint": view.checkpoint,
        "view_hash": view.content_hash,
    }


def _encode_journal(journal: SimulationPublicJournal) -> bytes:
    header = {
        "schema": journal.schema,
        "stream_id": journal.stream_id,
        "scenario_hash": journal.scenario_hash,
        "parent_journal_hash": journal.parent_journal_hash,
        "content_hash": journal.content_hash,
    }
    documents = (header,) + tuple(
        _batch_document(batch) for batch in journal.batches
    )
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

    if not isinstance(batch, SimulationOutputBatch):
        raise TypeError("public journal write requires a SimulationOutputBatch")
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
        batch,
        SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
    )
    journal = SimulationPublicJournal(
        batch.stream_id if not prior_batches else prior_batches[0].stream_id,
        batch.scenario_hash if not prior_batches else prior_batches[0].scenario_hash,
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

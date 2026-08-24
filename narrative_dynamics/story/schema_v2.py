from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.story.schema import (
    DirectObservationV1,
    RelocationEventV1,
    SearchDecisionV1,
    StoryEntitiesV1,
    _exact_keys,
    _freeze_mapping,
    _integer,
    _text,
    _thaw,
    _validate_story_semantics,
)


NARRATIVE_CASE_V2_SCHEMA_VERSION = 2


def _ranking(values: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{label} must be a sequence")
    items = tuple(_text(value, label=label) for value in values)
    if not items:
        raise ValueError(f"{label} must be non-empty")
    if len(set(items)) != len(items):
        raise ValueError(f"{label} must contain unique actions")
    return items


@dataclass(frozen=True)
class LocationReportV2:
    id: str
    logical_time: int
    speaker: str
    object: str
    location: str
    support_event: str
    kind: str = "report_object_location"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, label="report id"))
        object.__setattr__(
            self,
            "logical_time",
            _integer(self.logical_time, label="report logical time"),
        )
        object.__setattr__(self, "speaker", _text(self.speaker, label="report speaker"))
        object.__setattr__(self, "object", _text(self.object, label="report object"))
        object.__setattr__(self, "location", _text(self.location, label="report location"))
        object.__setattr__(
            self,
            "support_event",
            _text(self.support_event, label="report support_event"),
        )
        if self.kind != "report_object_location":
            raise ValueError("report kind must be report_object_location")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "logical_time": self.logical_time,
            "speaker": self.speaker,
            "object": self.object,
            "location": self.location,
            "support_event": self.support_event,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> LocationReportV2:
        data = _exact_keys(
            data,
            {"id", "logical_time", "speaker", "object", "location", "support_event", "kind"},
            label="report",
        )
        return cls(
            id=data["id"],
            logical_time=data["logical_time"],
            speaker=data["speaker"],
            object=data["object"],
            location=data["location"],
            support_event=data["support_event"],
            kind=data["kind"],
        )


@dataclass(frozen=True)
class ReportReceptionV2:
    report: str
    recipient: str
    channel: str = "direct_testimony"

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", _text(self.report, label="reception report"))
        object.__setattr__(
            self,
            "recipient",
            _text(self.recipient, label="reception recipient"),
        )
        if self.channel != "direct_testimony":
            raise ValueError("reception channel must be direct_testimony")

    def to_dict(self) -> dict[str, object]:
        return {
            "report": self.report,
            "recipient": self.recipient,
            "channel": self.channel,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ReportReceptionV2:
        data = _exact_keys(
            data,
            {"report", "recipient", "channel"},
            label="reception",
        )
        return cls(
            report=data["report"],
            recipient=data["recipient"],
            channel=data["channel"],
        )


@dataclass(frozen=True)
class NarrativeOracleV2:
    objective_location: str
    actor_direct_location: str
    actor_testimony_location: str
    testimony_ranking: tuple[str, ...]
    omniscient_ranking: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective_location",
            _text(self.objective_location, label="oracle objective location"),
        )
        object.__setattr__(
            self,
            "actor_direct_location",
            _text(self.actor_direct_location, label="oracle direct location"),
        )
        object.__setattr__(
            self,
            "actor_testimony_location",
            _text(self.actor_testimony_location, label="oracle testimony location"),
        )
        object.__setattr__(
            self,
            "testimony_ranking",
            _ranking(self.testimony_ranking, label="oracle testimony ranking"),
        )
        object.__setattr__(
            self,
            "omniscient_ranking",
            _ranking(self.omniscient_ranking, label="oracle omniscient ranking"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "objective_location": self.objective_location,
            "actor_direct_location": self.actor_direct_location,
            "actor_testimony_location": self.actor_testimony_location,
            "testimony_ranking": list(self.testimony_ranking),
            "omniscient_ranking": list(self.omniscient_ranking),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> NarrativeOracleV2:
        data = _exact_keys(
            data,
            {
                "objective_location",
                "actor_direct_location",
                "actor_testimony_location",
                "testimony_ranking",
                "omniscient_ranking",
            },
            label="oracle V2",
        )
        return cls(
            objective_location=data["objective_location"],
            actor_direct_location=data["actor_direct_location"],
            actor_testimony_location=data["actor_testimony_location"],
            testimony_ranking=data["testimony_ranking"],
            omniscient_ranking=data["omniscient_ranking"],
        )


def _validate_testimony_semantics(
    entities: StoryEntitiesV1,
    events: tuple[RelocationEventV1, ...],
    observations: tuple[DirectObservationV1, ...],
    reports: tuple[LocationReportV2, ...],
    receptions: tuple[ReportReceptionV2, ...],
    decision: SearchDecisionV1,
) -> None:
    _validate_story_semantics(
        entities,
        events,
        observations,
        decision,
        oracle=None,
    )

    if not reports:
        raise ValueError("V2 testimony story must contain at least one report")

    event_by_id = {event.id: event for event in events}
    observed_pairs = {(item.event, item.agent) for item in observations}
    occupied_times = {event.logical_time for event in events}
    report_by_id: dict[str, LocationReportV2] = {}
    previous_report_time: int | None = None

    for report in reports:
        if report.id in report_by_id:
            raise ValueError("report ids must be unique")
        if report.speaker not in entities.agents:
            raise ValueError("report speaker must reference a declared agent")
        if report.object not in entities.objects:
            raise ValueError("report object must reference a declared object")
        if report.location not in entities.locations:
            raise ValueError("report location must reference a declared location")
        support = event_by_id.get(report.support_event)
        if support is None:
            raise ValueError(
                "report support_event must reference a declared relocation"
            )
        if support.object != report.object:
            raise ValueError(
                "report support_event must relocate the reported object"
            )
        if support.logical_time >= report.logical_time:
            raise ValueError("report support_event must occur before the report")
        if (support.id, report.speaker) not in observed_pairs:
            raise ValueError(
                "report speaker must have directly observed the support_event"
            )
        if report.logical_time in occupied_times:
            raise ValueError("relocation and report logical times must be unique")
        if (
            previous_report_time is not None
            and report.logical_time <= previous_report_time
        ):
            raise ValueError("report logical time must be strictly increasing")
        if report.logical_time >= decision.time:
            raise ValueError("model-visible reports must occur before the decision")
        occupied_times.add(report.logical_time)
        previous_report_time = report.logical_time
        report_by_id[report.id] = report

    reception_pairs: set[tuple[str, str]] = set()
    for reception in receptions:
        if reception.report not in report_by_id:
            raise ValueError(
                "reception report must reference a declared report"
            )
        if reception.recipient not in entities.agents:
            raise ValueError(
                "reception recipient must reference a declared agent"
            )
        pair = (reception.report, reception.recipient)
        if pair in reception_pairs:
            raise ValueError(
                "duplicate report-recipient reception is not allowed"
            )
        reception_pairs.add(pair)


def _validate_oracle_v2(
    entities: StoryEntitiesV1,
    decision: SearchDecisionV1,
    oracle: NarrativeOracleV2,
) -> None:
    if oracle.objective_location not in entities.locations:
        raise ValueError("oracle objective location must be declared")
    if oracle.actor_direct_location not in entities.locations:
        raise ValueError("oracle direct location must be declared")
    if oracle.actor_testimony_location not in entities.locations:
        raise ValueError("oracle testimony location must be declared")

    action_ids = tuple(action.id for action in decision.actions)
    action_set = set(action_ids)
    if (
        set(oracle.testimony_ranking) != action_set
        or len(oracle.testimony_ranking) != len(action_ids)
    ):
        raise ValueError(
            "oracle testimony ranking must contain exactly the decision actions"
        )
    if (
        set(oracle.omniscient_ranking) != action_set
        or len(oracle.omniscient_ranking) != len(action_ids)
    ):
        raise ValueError(
            "oracle omniscient ranking must contain exactly the decision actions"
        )


@dataclass(frozen=True)
class NarrativeCaseV2:
    name: str
    version: str
    source: Mapping[str, object]
    provenance: Mapping[str, object]
    source_text: str
    entities: StoryEntitiesV1
    events: tuple[RelocationEventV1, ...]
    observations: tuple[DirectObservationV1, ...]
    reports: tuple[LocationReportV2, ...]
    receptions: tuple[ReportReceptionV2, ...]
    decision: SearchDecisionV1
    oracle: NarrativeOracleV2
    schema_version: int = field(default=NARRATIVE_CASE_V2_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _text(self.name, label="narrative V2 case name"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="narrative V2 case version"),
        )
        object.__setattr__(
            self,
            "source_text",
            _text(self.source_text, label="narrative V2 source text"),
        )
        if (
            self.schema_version != NARRATIVE_CASE_V2_SCHEMA_VERSION
            or isinstance(self.schema_version, bool)
        ):
            raise ValueError("unsupported narrative V2 schema version")
        if not isinstance(self.entities, StoryEntitiesV1):
            raise TypeError("narrative V2 entities must be StoryEntitiesV1")

        events = tuple(self.events)
        observations = tuple(self.observations)
        reports = tuple(self.reports)
        receptions = tuple(self.receptions)
        if any(not isinstance(event, RelocationEventV1) for event in events):
            raise TypeError("narrative V2 events must be RelocationEventV1 values")
        if any(
            not isinstance(observation, DirectObservationV1)
            for observation in observations
        ):
            raise TypeError(
                "narrative V2 observations must be DirectObservationV1 values"
            )
        if any(not isinstance(report, LocationReportV2) for report in reports):
            raise TypeError("narrative V2 reports must be LocationReportV2 values")
        if any(
            not isinstance(reception, ReportReceptionV2)
            for reception in receptions
        ):
            raise TypeError(
                "narrative V2 receptions must be ReportReceptionV2 values"
            )
        if not isinstance(self.decision, SearchDecisionV1):
            raise TypeError("narrative V2 decision must be SearchDecisionV1")
        if not isinstance(self.oracle, NarrativeOracleV2):
            raise TypeError("narrative V2 oracle must be NarrativeOracleV2")

        object.__setattr__(self, "events", events)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "reports", reports)
        object.__setattr__(self, "receptions", receptions)
        object.__setattr__(
            self,
            "source",
            _freeze_mapping(self.source, label="narrative V2 source"),
        )
        object.__setattr__(
            self,
            "provenance",
            _freeze_mapping(self.provenance, label="narrative V2 provenance"),
        )

        _validate_testimony_semantics(
            self.entities,
            events,
            observations,
            reports,
            receptions,
            self.decision,
        )
        _validate_oracle_v2(self.entities, self.decision, self.oracle)

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
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
            "reports": [report.to_dict() for report in self.reports],
            "receptions": [
                reception.to_dict() for reception in self.receptions
            ],
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
    ) -> NarrativeCaseV2:
        base_keys = {
            "schema_version",
            "name",
            "version",
            "source",
            "provenance",
            "source_text",
            "entities",
            "events",
            "observations",
            "reports",
            "receptions",
            "decision",
            "oracle",
        }
        if not isinstance(data, Mapping):
            raise TypeError("narrative V2 case must be a mapping")
        keys = set(data)
        allowed = base_keys | {"content_hash"}
        if not keys.issubset(allowed) or not base_keys.issubset(keys):
            raise ValueError("narrative case keys must match the V2 schema")
        if verify_declared_hash and "content_hash" not in data:
            raise ValueError("narrative V2 case content hash is required")

        raw_events = data["events"]
        raw_observations = data["observations"]
        raw_reports = data["reports"]
        raw_receptions = data["receptions"]
        if not isinstance(raw_events, (list, tuple)):
            raise TypeError("narrative V2 events must be a sequence")
        if not isinstance(raw_observations, (list, tuple)):
            raise TypeError("narrative V2 observations must be a sequence")
        if not isinstance(raw_reports, (list, tuple)):
            raise TypeError("narrative V2 reports must be a sequence")
        if not isinstance(raw_receptions, (list, tuple)):
            raise TypeError("narrative V2 receptions must be a sequence")

        case = cls(
            name=data["name"],
            version=data["version"],
            source=data["source"],
            provenance=data["provenance"],
            source_text=data["source_text"],
            entities=StoryEntitiesV1.from_dict(data["entities"]),
            events=tuple(
                RelocationEventV1.from_dict(event) for event in raw_events
            ),
            observations=tuple(
                DirectObservationV1.from_dict(observation)
                for observation in raw_observations
            ),
            reports=tuple(
                LocationReportV2.from_dict(report) for report in raw_reports
            ),
            receptions=tuple(
                ReportReceptionV2.from_dict(reception)
                for reception in raw_receptions
            ),
            decision=SearchDecisionV1.from_dict(data["decision"]),
            oracle=NarrativeOracleV2.from_dict(data["oracle"]),
            schema_version=data["schema_version"],
        )
        if verify_declared_hash:
            declared = data["content_hash"]
            if not isinstance(declared, str) or declared != case.content_hash:
                raise ValueError(
                    "narrative V2 case content hash does not match payload"
                )
        return case


def load_narrative_case_v2(path: str | Path) -> NarrativeCaseV2:
    location = Path(path)
    raw = json.loads(location.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("narrative V2 fixture root must be a mapping")
    return NarrativeCaseV2.from_dict(raw)

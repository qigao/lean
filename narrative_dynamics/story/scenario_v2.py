from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from narrative_dynamics.contracts import Scenario, stable_content_hash
from narrative_dynamics.story.schema import (
    DirectObservationV1,
    RelocationEventV1,
    SearchDecisionV1,
    StoryEntitiesV1,
)
from narrative_dynamics.story.schema_v2 import (
    LocationReportV2,
    NarrativeCaseV2,
    ReportReceptionV2,
    _validate_testimony_semantics,
)


_VISIBLE_KEYS_V2 = {
    "entities",
    "events",
    "observations",
    "reports",
    "receptions",
    "decision",
}


def _exact_visible_payload_v2(
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("testimony scenario payload must be a mapping")
    if set(payload) != _VISIBLE_KEYS_V2:
        raise ValueError(
            "testimony scenario payload keys must be exactly entities, events, "
            "observations, reports, receptions, and decision"
        )
    return payload


def _event_payload(event: RelocationEventV1) -> dict[str, object]:
    return {
        "id": event.id,
        "logical_time": event.logical_time,
        "actor": event.actor,
        "object": event.object,
        "from_location": event.from_location,
        "to_location": event.to_location,
        "kind": event.kind,
    }


def _observation_payload(observation: DirectObservationV1) -> dict[str, object]:
    return {
        "event": observation.event,
        "agent": observation.agent,
        "channel": observation.channel,
    }


def _report_payload(report: LocationReportV2) -> dict[str, object]:
    return {
        "id": report.id,
        "logical_time": report.logical_time,
        "speaker": report.speaker,
        "object": report.object,
        "location": report.location,
        "support_event": report.support_event,
        "kind": report.kind,
    }


def _reception_payload(reception: ReportReceptionV2) -> dict[str, object]:
    return {
        "report": reception.report,
        "recipient": reception.recipient,
        "channel": reception.channel,
    }


def _decision_payload(decision: SearchDecisionV1) -> dict[str, object]:
    return {
        "id": decision.id,
        "time": decision.time,
        "actor": decision.actor,
        "object": decision.object,
        "actions": tuple(
            {"id": action.id, "location": action.location}
            for action in decision.actions
        ),
        "kind": decision.kind,
    }


@dataclass(frozen=True)
class NarrativeScenarioV2:
    """Model-visible testimony semantics with authored metadata and oracle removed."""

    entities: StoryEntitiesV1
    events: tuple[RelocationEventV1, ...]
    observations: tuple[DirectObservationV1, ...]
    reports: tuple[LocationReportV2, ...]
    receptions: tuple[ReportReceptionV2, ...]
    decision: SearchDecisionV1

    def __post_init__(self) -> None:
        if not isinstance(self.entities, StoryEntitiesV1):
            raise TypeError("testimony scenario entities must be StoryEntitiesV1")
        events = tuple(self.events)
        observations = tuple(self.observations)
        reports = tuple(self.reports)
        receptions = tuple(self.receptions)
        if any(not isinstance(event, RelocationEventV1) for event in events):
            raise TypeError("testimony scenario events must be RelocationEventV1 values")
        if any(
            not isinstance(observation, DirectObservationV1)
            for observation in observations
        ):
            raise TypeError(
                "testimony scenario observations must be DirectObservationV1 values"
            )
        if any(not isinstance(report, LocationReportV2) for report in reports):
            raise TypeError("testimony scenario reports must be LocationReportV2 values")
        if any(
            not isinstance(reception, ReportReceptionV2)
            for reception in receptions
        ):
            raise TypeError(
                "testimony scenario receptions must be ReportReceptionV2 values"
            )
        if not isinstance(self.decision, SearchDecisionV1):
            raise TypeError("testimony scenario decision must be SearchDecisionV1")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "reports", reports)
        object.__setattr__(self, "receptions", receptions)
        _validate_testimony_semantics(
            self.entities,
            events,
            observations,
            reports,
            receptions,
            self.decision,
        )

    @classmethod
    def from_case(cls, case: NarrativeCaseV2) -> NarrativeScenarioV2:
        if not isinstance(case, NarrativeCaseV2):
            raise TypeError("testimony scenario source must be NarrativeCaseV2")
        return cls(
            entities=case.entities,
            events=case.events,
            observations=case.observations,
            reports=case.reports,
            receptions=case.receptions,
            decision=case.decision,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "entities": {
                "agents": tuple(self.entities.agents),
                "objects": tuple(self.entities.objects),
                "locations": tuple(self.entities.locations),
            },
            "events": tuple(_event_payload(event) for event in self.events),
            "observations": tuple(
                _observation_payload(observation)
                for observation in self.observations
            ),
            "reports": tuple(_report_payload(report) for report in self.reports),
            "receptions": tuple(
                _reception_payload(reception)
                for reception in self.receptions
            ),
            "decision": _decision_payload(self.decision),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> NarrativeScenarioV2:
        payload = _exact_visible_payload_v2(payload)
        raw_events = payload["events"]
        raw_observations = payload["observations"]
        raw_reports = payload["reports"]
        raw_receptions = payload["receptions"]
        if not isinstance(raw_events, (tuple, list)):
            raise TypeError("testimony scenario events must be a sequence")
        if not isinstance(raw_observations, (tuple, list)):
            raise TypeError("testimony scenario observations must be a sequence")
        if not isinstance(raw_reports, (tuple, list)):
            raise TypeError("testimony scenario reports must be a sequence")
        if not isinstance(raw_receptions, (tuple, list)):
            raise TypeError("testimony scenario receptions must be a sequence")
        return cls(
            entities=StoryEntitiesV1.from_dict(payload["entities"]),
            events=tuple(
                RelocationEventV1.from_dict(event)
                for event in raw_events
            ),
            observations=tuple(
                DirectObservationV1.from_dict(observation)
                for observation in raw_observations
            ),
            reports=tuple(
                LocationReportV2.from_dict(report)
                for report in raw_reports
            ),
            receptions=tuple(
                ReportReceptionV2.from_dict(reception)
                for reception in raw_receptions
            ),
            decision=SearchDecisionV1.from_dict(payload["decision"]),
        )


def _runtime_id_v2(payload: Mapping[str, object]) -> str:
    digest = stable_content_hash(payload).removeprefix("sha256:")
    return f"story-v2-{digest}"


def project_testimony_scenario(case: NarrativeCaseV2) -> Scenario:
    """Project an authored V2 fixture into canonical model-visible semantics."""

    story = NarrativeScenarioV2.from_case(case)
    payload = story.to_payload()
    return Scenario(id=_runtime_id_v2(payload), payload=payload)


def decode_testimony_scenario(scenario: Scenario) -> NarrativeScenarioV2:
    """Decode and revalidate a model-visible testimony scenario fail closed."""

    if not isinstance(scenario, Scenario):
        raise TypeError("testimony model scenario must be a Scenario")
    story = NarrativeScenarioV2.from_payload(scenario.payload)
    canonical_payload = story.to_payload()
    if stable_content_hash(scenario.payload) != stable_content_hash(canonical_payload):
        raise ValueError("testimony scenario payload is not canonical")
    if scenario.id != _runtime_id_v2(canonical_payload):
        raise ValueError(
            "testimony scenario id must match the visible payload digest"
        )
    return story

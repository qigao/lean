from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from narrative_dynamics.story.scenario_v2 import NarrativeScenarioV2


@dataclass(frozen=True)
class EpistemicLocationStateV2:
    object_id: str
    location: str
    evidence_kind: str
    supporting_id: str
    source_agent: str
    logical_time: int


def _at_time(value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("testimony replay at_time must be an integer or None")
    if value < 0:
        raise ValueError("testimony replay at_time must be non-negative")
    return value


def testimony_state(
    story: NarrativeScenarioV2,
    agent: str,
    *,
    at_time: int | None = None,
) -> Mapping[str, EpistemicLocationStateV2]:
    """Replay direct observations and received reports for one agent.

    The input scenario is already the validated provenance boundary. This
    function does not infer report truth: a received report contributes its
    asserted location exactly as communicated. For each object, the latest
    admissible evidence by logical time wins.
    """

    if not isinstance(story, NarrativeScenarioV2):
        raise TypeError("testimony replay requires a validated NarrativeScenarioV2")
    if not isinstance(agent, str) or not agent:
        raise ValueError("testimony replay agent must be a non-empty string")
    limit = _at_time(at_time)

    evidence: list[EpistemicLocationStateV2] = []
    event_by_id = {event.id: event for event in story.events}
    for observation in story.observations:
        if observation.agent != agent:
            continue
        event = event_by_id[observation.event]
        if limit is not None and event.logical_time > limit:
            continue
        evidence.append(
            EpistemicLocationStateV2(
                object_id=event.object,
                location=event.to_location,
                evidence_kind="direct_perception",
                supporting_id=event.id,
                source_agent=agent,
                logical_time=event.logical_time,
            )
        )

    report_by_id = {report.id: report for report in story.reports}
    for reception in story.receptions:
        if reception.recipient != agent:
            continue
        report = report_by_id[reception.report]
        if limit is not None and report.logical_time > limit:
            continue
        evidence.append(
            EpistemicLocationStateV2(
                object_id=report.object,
                location=report.location,
                evidence_kind="testimony",
                supporting_id=report.id,
                source_agent=report.speaker,
                logical_time=report.logical_time,
            )
        )

    state: dict[str, EpistemicLocationStateV2] = {}
    for item in sorted(evidence, key=lambda value: value.logical_time):
        state[item.object_id] = item
    return MappingProxyType(state)


def latest_epistemic_location(
    state: Mapping[str, EpistemicLocationStateV2],
    object_id: str,
) -> EpistemicLocationStateV2:
    """Return the latest supported private location or fail closed."""

    if not isinstance(object_id, str) or not object_id:
        raise ValueError("testimony replay object id must be a non-empty string")
    try:
        value = state[object_id]
    except KeyError as error:
        raise ValueError(
            f"no supported testimony-aware location for object {object_id!r}"
        ) from error
    if not isinstance(value, EpistemicLocationStateV2):
        raise TypeError(
            "testimony replay state must contain EpistemicLocationStateV2 values"
        )
    return value

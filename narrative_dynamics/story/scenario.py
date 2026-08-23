from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from narrative_dynamics.contracts import Scenario, stable_content_hash
from narrative_dynamics.story.schema import (
    DirectObservationV1,
    NarrativeCaseV1,
    RelocationEventV1,
    SearchDecisionV1,
    StoryEntitiesV1,
    _validate_story_semantics,
)


_VISIBLE_KEYS = {"entities", "events", "observations", "decision"}


def _exact_visible_payload(
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("narrative scenario payload must be a mapping")
    if set(payload) != _VISIBLE_KEYS:
        raise ValueError(
            "narrative scenario payload keys must be exactly "
            "entities, events, observations, and decision"
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
class NarrativeScenarioV1:
    """Model-visible V1 story semantics with all authored oracle data removed."""

    entities: StoryEntitiesV1
    events: tuple[RelocationEventV1, ...]
    observations: tuple[DirectObservationV1, ...]
    decision: SearchDecisionV1

    def __post_init__(self) -> None:
        if not isinstance(self.entities, StoryEntitiesV1):
            raise TypeError("narrative scenario entities must be StoryEntitiesV1")
        events = tuple(self.events)
        observations = tuple(self.observations)
        if any(not isinstance(event, RelocationEventV1) for event in events):
            raise TypeError(
                "narrative scenario events must be RelocationEventV1 values"
            )
        if any(
            not isinstance(observation, DirectObservationV1)
            for observation in observations
        ):
            raise TypeError(
                "narrative scenario observations must be DirectObservationV1 values"
            )
        if not isinstance(self.decision, SearchDecisionV1):
            raise TypeError(
                "narrative scenario decision must be SearchDecisionV1"
            )
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "observations", observations)
        _validate_story_semantics(
            self.entities,
            events,
            observations,
            self.decision,
            oracle=None,
        )

    @classmethod
    def from_case(cls, case: NarrativeCaseV1) -> NarrativeScenarioV1:
        if not isinstance(case, NarrativeCaseV1):
            raise TypeError("narrative scenario source must be NarrativeCaseV1")
        return cls(
            entities=case.entities,
            events=case.events,
            observations=case.observations,
            decision=case.decision,
        )

    def to_payload(self) -> dict[str, object]:
        """Return only canonical model-visible story semantics."""

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
            "decision": _decision_payload(self.decision),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> NarrativeScenarioV1:
        payload = _exact_visible_payload(payload)
        raw_events = payload["events"]
        raw_observations = payload["observations"]
        if not isinstance(raw_events, (tuple, list)):
            raise TypeError("narrative scenario events must be a sequence")
        if not isinstance(raw_observations, (tuple, list)):
            raise TypeError("narrative scenario observations must be a sequence")
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
            decision=SearchDecisionV1.from_dict(payload["decision"]),
        )


def _runtime_id(payload: Mapping[str, object]) -> str:
    digest = stable_content_hash(payload).removeprefix("sha256:")
    return f"story-v1-{digest}"


def project_narrative_scenario(case: NarrativeCaseV1) -> Scenario:
    """Project an authored fixture into the only supported model-visible input."""

    story = NarrativeScenarioV1.from_case(case)
    payload = story.to_payload()
    return Scenario(id=_runtime_id(payload), payload=payload)


def decode_narrative_scenario(scenario: Scenario) -> NarrativeScenarioV1:
    """Decode and revalidate a model-visible story scenario fail closed."""

    if not isinstance(scenario, Scenario):
        raise TypeError("story model scenario must be a Scenario")
    story = NarrativeScenarioV1.from_payload(scenario.payload)
    canonical_payload = story.to_payload()
    if stable_content_hash(scenario.payload) != stable_content_hash(canonical_payload):
        raise ValueError("narrative scenario payload is not canonical")
    expected_id = _runtime_id(canonical_payload)
    if scenario.id != expected_id:
        raise ValueError("narrative scenario id must match the visible payload digest")
    return story

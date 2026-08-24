from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from narrative_dynamics.story.replay import objective_state, subjective_state
from narrative_dynamics.story.replay_v2 import resolve_testimony_action, testimony_state
from narrative_dynamics.story.scenario_v2 import NarrativeScenarioV2


_ALLOWED_TRIGGERS = frozenset({"relocation", "report", "decision", "mixed"})
_ALLOWED_INTERVENTIONS = frozenset(
    {
        "change_report_content",
        "remove_direct_observation",
        "remove_reception",
        "remove_support_observation",
    }
)
_ALLOWED_REJECTION_STAGES = frozenset(
    {"scenario_validation", "action_resolution"}
)


@dataclass(frozen=True)
class _EvolutionAgentStateV2:
    agent: str
    direct_location: str | None
    direct_supporting_id: str | None
    testimony_location: str | None
    evidence_kind: str | None
    supporting_id: str | None
    source_agent: str | None
    evidence_logical_time: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "agent": self.agent,
            "direct_location": self.direct_location,
            "direct_supporting_id": self.direct_supporting_id,
            "testimony_location": self.testimony_location,
            "evidence_kind": self.evidence_kind,
            "supporting_id": self.supporting_id,
            "source_agent": self.source_agent,
            "evidence_logical_time": self.evidence_logical_time,
        }


@dataclass(frozen=True)
class EvolutionSnapshotV2:
    logical_time: int
    trigger_kind: str
    trigger_ids: tuple[str, ...]
    objective_location: str | None
    agents: Mapping[str, _EvolutionAgentStateV2]
    selected_action: str | None

    def __post_init__(self) -> None:
        if self.trigger_kind not in _ALLOWED_TRIGGERS:
            raise ValueError("evolution trigger kind is not supported")
        trigger_ids = tuple(self.trigger_ids)
        agents = dict(self.agents)
        if any(
            not isinstance(value, _EvolutionAgentStateV2)
            for value in agents.values()
        ):
            raise TypeError("evolution agents must contain canonical agent states")
        object.__setattr__(self, "trigger_ids", trigger_ids)
        object.__setattr__(self, "agents", MappingProxyType(agents))

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_time": self.logical_time,
            "trigger_kind": self.trigger_kind,
            "trigger_ids": list(self.trigger_ids),
            "objective_location": self.objective_location,
            "agents": {
                name: value.to_dict() for name, value in self.agents.items()
            },
            "selected_action": self.selected_action,
        }


@dataclass(frozen=True)
class EvolutionTrajectoryV2:
    target_object: str
    tracked_agents: tuple[str, ...]
    snapshots: tuple[EvolutionSnapshotV2, ...]
    selected_action: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tracked_agents", tuple(self.tracked_agents))
        object.__setattr__(self, "snapshots", tuple(self.snapshots))

    def to_dict(self) -> dict[str, object]:
        return {
            "target_object": self.target_object,
            "tracked_agents": list(self.tracked_agents),
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "selected_action": self.selected_action,
        }


@dataclass(frozen=True)
class EvolutionInterventionV2:
    kind: str
    subject_id: str
    agent: str | None
    from_value: str | None
    to_value: str | None
    logical_time: int | None

    def __post_init__(self) -> None:
        if self.kind not in _ALLOWED_INTERVENTIONS:
            raise ValueError("evolution intervention kind is not supported")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "subject_id": self.subject_id,
            "agent": self.agent,
            "from_value": self.from_value,
            "to_value": self.to_value,
            "logical_time": self.logical_time,
        }


@dataclass(frozen=True)
class _EvolutionDivergenceV2:
    logical_time: int
    changed_fields: tuple[str, ...]
    action_changed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed_fields", tuple(self.changed_fields))

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_time": self.logical_time,
            "changed_fields": list(self.changed_fields),
            "action_changed": self.action_changed,
        }


@dataclass(frozen=True)
class EvolutionCounterfactualV2:
    intervention: EvolutionInterventionV2
    status: str
    trajectory: EvolutionTrajectoryV2 | None
    first_divergence: _EvolutionDivergenceV2 | None
    rejection_stage: str | None
    rejection_reason: str | None
    rejection_logical_time: int | None

    def __post_init__(self) -> None:
        if self.status not in {"valid", "rejected"}:
            raise ValueError(
                "evolution counterfactual status must be valid or rejected"
            )
        if self.status == "valid":
            if self.trajectory is None:
                raise ValueError(
                    "valid evolution counterfactual requires a trajectory"
                )
            if any(
                value is not None
                for value in (
                    self.rejection_stage,
                    self.rejection_reason,
                    self.rejection_logical_time,
                )
            ):
                raise ValueError(
                    "valid evolution counterfactual cannot contain rejection data"
                )
        else:
            if self.trajectory is not None or self.first_divergence is not None:
                raise ValueError(
                    "rejected evolution counterfactual cannot contain a trajectory"
                )
            if self.rejection_stage not in _ALLOWED_REJECTION_STAGES:
                raise ValueError(
                    "rejected evolution counterfactual requires a rejection stage"
                )
            if (
                not isinstance(self.rejection_reason, str)
                or not self.rejection_reason
            ):
                raise ValueError(
                    "rejected evolution counterfactual requires a rejection reason"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "intervention": self.intervention.to_dict(),
            "status": self.status,
            "trajectory": (
                None if self.trajectory is None else self.trajectory.to_dict()
            ),
            "first_divergence": (
                None
                if self.first_divergence is None
                else self.first_divergence.to_dict()
            ),
            "rejection_stage": self.rejection_stage,
            "rejection_reason": self.rejection_reason,
            "rejection_logical_time": self.rejection_logical_time,
        }


@dataclass(frozen=True)
class EvolutionAnalysisV2:
    baseline: EvolutionTrajectoryV2
    counterfactuals: tuple[EvolutionCounterfactualV2, ...]
    mechanism_uniqueness_claimed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "counterfactuals", tuple(self.counterfactuals))

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_dict(),
            "counterfactuals": [
                item.to_dict() for item in self.counterfactuals
            ],
            "mechanism_uniqueness_claimed": self.mechanism_uniqueness_claimed,
        }


def _tracked_agents(story: NarrativeScenarioV2) -> tuple[str, ...]:
    target = story.decision.object
    ordered = [story.decision.actor]
    seen = {story.decision.actor}
    for report in story.reports:
        if report.object == target and report.speaker not in seen:
            ordered.append(report.speaker)
            seen.add(report.speaker)
    return tuple(ordered)


def _snapshot_times(story: NarrativeScenarioV2) -> tuple[int, ...]:
    values = {event.logical_time for event in story.events}
    values.update(report.logical_time for report in story.reports)
    values.add(story.decision.time)
    return tuple(sorted(values))


def _trigger_at(
    story: NarrativeScenarioV2,
    logical_time: int,
) -> tuple[str, tuple[str, ...]]:
    kinds: list[str] = []
    ids: list[str] = []
    event_ids = tuple(
        event.id
        for event in story.events
        if event.logical_time == logical_time
    )
    if event_ids:
        kinds.append("relocation")
        ids.extend(event_ids)
    report_ids = tuple(
        report.id
        for report in story.reports
        if report.logical_time == logical_time
    )
    if report_ids:
        kinds.append("report")
        ids.extend(report_ids)
    if story.decision.time == logical_time:
        kinds.append("decision")
        ids.append(story.decision.id)
    if not kinds:
        raise RuntimeError("evolution snapshot time has no canonical trigger")
    return (kinds[0] if len(kinds) == 1 else "mixed", tuple(ids))


def _agent_state(
    story: NarrativeScenarioV2,
    agent: str,
    target: str,
    logical_time: int,
) -> _EvolutionAgentStateV2:
    direct = subjective_state(
        story.events,
        story.observations,
        agent,
        at_time=logical_time,
    ).get(target)
    testimony = testimony_state(
        story,
        agent,
        at_time=logical_time,
    ).get(target)
    return _EvolutionAgentStateV2(
        agent=agent,
        direct_location=None if direct is None else direct.location,
        direct_supporting_id=(
            None if direct is None else direct.supporting_event_id
        ),
        testimony_location=(None if testimony is None else testimony.location),
        evidence_kind=(None if testimony is None else testimony.evidence_kind),
        supporting_id=(None if testimony is None else testimony.supporting_id),
        source_agent=(None if testimony is None else testimony.source_agent),
        evidence_logical_time=(
            None if testimony is None else testimony.logical_time
        ),
    )


def _build_trajectory(story: NarrativeScenarioV2) -> EvolutionTrajectoryV2:
    selected = resolve_testimony_action(story)
    target = story.decision.object
    tracked = _tracked_agents(story)
    snapshots: list[EvolutionSnapshotV2] = []
    for logical_time in _snapshot_times(story):
        objective = objective_state(
            story.events,
            at_time=logical_time,
        ).get(target)
        trigger_kind, trigger_ids = _trigger_at(story, logical_time)
        agents = MappingProxyType(
            {
                agent: _agent_state(story, agent, target, logical_time)
                for agent in tracked
            }
        )
        snapshots.append(
            EvolutionSnapshotV2(
                logical_time=logical_time,
                trigger_kind=trigger_kind,
                trigger_ids=trigger_ids,
                objective_location=(
                    None if objective is None else objective.location
                ),
                agents=agents,
                selected_action=(
                    selected
                    if logical_time == story.decision.time
                    else None
                ),
            )
        )
    return EvolutionTrajectoryV2(
        target_object=target,
        tracked_agents=tracked,
        snapshots=tuple(snapshots),
        selected_action=selected,
    )


def _snapshot_fields(snapshot: EvolutionSnapshotV2) -> dict[str, object]:
    fields: dict[str, object] = {
        "objective_location": snapshot.objective_location,
        "selected_action": snapshot.selected_action,
    }
    for agent, state in snapshot.agents.items():
        prefix = f"agents.{agent}."
        fields[prefix + "direct_location"] = state.direct_location
        fields[prefix + "direct_supporting_id"] = state.direct_supporting_id
        fields[prefix + "testimony_location"] = state.testimony_location
        fields[prefix + "evidence_kind"] = state.evidence_kind
        fields[prefix + "supporting_id"] = state.supporting_id
        fields[prefix + "source_agent"] = state.source_agent
        fields[prefix + "evidence_logical_time"] = state.evidence_logical_time
    return fields


def _first_divergence(
    baseline: EvolutionTrajectoryV2,
    candidate: EvolutionTrajectoryV2,
) -> _EvolutionDivergenceV2 | None:
    if baseline.target_object != candidate.target_object:
        raise RuntimeError("evolution trajectories must analyze the same target object")
    if baseline.tracked_agents != candidate.tracked_agents:
        raise RuntimeError("evolution trajectories must track the same agents")
    baseline_times = tuple(item.logical_time for item in baseline.snapshots)
    candidate_times = tuple(item.logical_time for item in candidate.snapshots)
    if baseline_times != candidate_times:
        raise RuntimeError("evolution trajectories must share canonical snapshot times")

    for baseline_snapshot, candidate_snapshot in zip(
        baseline.snapshots,
        candidate.snapshots,
        strict=True,
    ):
        baseline_fields = _snapshot_fields(baseline_snapshot)
        candidate_fields = _snapshot_fields(candidate_snapshot)
        if set(baseline_fields) != set(candidate_fields):
            raise RuntimeError("evolution trajectories must share canonical field paths")
        changed = tuple(
            sorted(
                key
                for key in baseline_fields
                if baseline_fields[key] != candidate_fields[key]
            )
        )
        if changed:
            return _EvolutionDivergenceV2(
                logical_time=baseline_snapshot.logical_time,
                changed_fields=changed,
                action_changed=(
                    baseline.selected_action != candidate.selected_action
                ),
            )
    return None


def _rejected_counterfactual(
    intervention: EvolutionInterventionV2,
    stage: str,
    error: ValueError,
) -> EvolutionCounterfactualV2:
    return EvolutionCounterfactualV2(
        intervention=intervention,
        status="rejected",
        trajectory=None,
        first_divergence=None,
        rejection_stage=stage,
        rejection_reason=str(error),
        rejection_logical_time=intervention.logical_time,
    )


def _evaluate_candidate(
    baseline: EvolutionTrajectoryV2,
    intervention: EvolutionInterventionV2,
    build_story: Callable[[], NarrativeScenarioV2],
) -> EvolutionCounterfactualV2:
    try:
        candidate = build_story()
    except ValueError as error:
        return _rejected_counterfactual(
            intervention,
            "scenario_validation",
            error,
        )

    try:
        trajectory = _build_trajectory(candidate)
    except ValueError as error:
        return _rejected_counterfactual(
            intervention,
            "action_resolution",
            error,
        )

    return EvolutionCounterfactualV2(
        intervention=intervention,
        status="valid",
        trajectory=trajectory,
        first_divergence=_first_divergence(baseline, trajectory),
        rejection_stage=None,
        rejection_reason=None,
        rejection_logical_time=None,
    )


def _remove_reception_counterfactuals(
    story: NarrativeScenarioV2,
    baseline: EvolutionTrajectoryV2,
) -> tuple[EvolutionCounterfactualV2, ...]:
    target = story.decision.object
    actor = story.decision.actor
    results: list[EvolutionCounterfactualV2] = []
    for report in story.reports:
        if report.object != target:
            continue
        if not any(
            reception.report == report.id and reception.recipient == actor
            for reception in story.receptions
        ):
            continue
        intervention = EvolutionInterventionV2(
            kind="remove_reception",
            subject_id=report.id,
            agent=actor,
            from_value="received",
            to_value=None,
            logical_time=report.logical_time,
        )
        receptions = tuple(
            reception
            for reception in story.receptions
            if not (
                reception.report == report.id
                and reception.recipient == actor
            )
        )
        results.append(
            _evaluate_candidate(
                baseline,
                intervention,
                lambda receptions=receptions: NarrativeScenarioV2(
                    entities=story.entities,
                    events=story.events,
                    observations=story.observations,
                    reports=story.reports,
                    receptions=receptions,
                    decision=story.decision,
                ),
            )
        )
    return tuple(results)


def _change_report_content_counterfactuals(
    story: NarrativeScenarioV2,
    baseline: EvolutionTrajectoryV2,
) -> tuple[EvolutionCounterfactualV2, ...]:
    target = story.decision.object
    results: list[EvolutionCounterfactualV2] = []
    for report in story.reports:
        if report.object != target:
            continue
        for action in story.decision.actions:
            alternate_location = action.location
            if alternate_location == report.location:
                continue
            intervention = EvolutionInterventionV2(
                kind="change_report_content",
                subject_id=report.id,
                agent=report.speaker,
                from_value=report.location,
                to_value=alternate_location,
                logical_time=report.logical_time,
            )
            reports = tuple(
                replace(item, location=alternate_location)
                if item.id == report.id
                else item
                for item in story.reports
            )
            results.append(
                _evaluate_candidate(
                    baseline,
                    intervention,
                    lambda reports=reports: NarrativeScenarioV2(
                        entities=story.entities,
                        events=story.events,
                        observations=story.observations,
                        reports=reports,
                        receptions=story.receptions,
                        decision=story.decision,
                    ),
                )
            )
    return tuple(results)


def _remove_direct_observation_counterfactuals(
    story: NarrativeScenarioV2,
    baseline: EvolutionTrajectoryV2,
) -> tuple[EvolutionCounterfactualV2, ...]:
    target = story.decision.object
    actor = story.decision.actor
    event_by_id = {event.id: event for event in story.events}
    results: list[EvolutionCounterfactualV2] = []
    for observation in story.observations:
        event = event_by_id[observation.event]
        if observation.agent != actor or event.object != target:
            continue
        intervention = EvolutionInterventionV2(
            kind="remove_direct_observation",
            subject_id=event.id,
            agent=actor,
            from_value="observed",
            to_value=None,
            logical_time=event.logical_time,
        )
        observations = tuple(
            item
            for item in story.observations
            if not (item.event == event.id and item.agent == actor)
        )
        results.append(
            _evaluate_candidate(
                baseline,
                intervention,
                lambda observations=observations: NarrativeScenarioV2(
                    entities=story.entities,
                    events=story.events,
                    observations=observations,
                    reports=story.reports,
                    receptions=story.receptions,
                    decision=story.decision,
                ),
            )
        )
    return tuple(results)


def _remove_support_observation_counterfactuals(
    story: NarrativeScenarioV2,
    baseline: EvolutionTrajectoryV2,
) -> tuple[EvolutionCounterfactualV2, ...]:
    target = story.decision.object
    event_by_id = {event.id: event for event in story.events}
    results: list[EvolutionCounterfactualV2] = []
    for report in story.reports:
        if report.object != target:
            continue
        support = event_by_id[report.support_event]
        intervention = EvolutionInterventionV2(
            kind="remove_support_observation",
            subject_id=report.id,
            agent=report.speaker,
            from_value=report.support_event,
            to_value=None,
            logical_time=support.logical_time,
        )
        observations = tuple(
            item
            for item in story.observations
            if not (
                item.event == report.support_event
                and item.agent == report.speaker
            )
        )
        results.append(
            _evaluate_candidate(
                baseline,
                intervention,
                lambda observations=observations: NarrativeScenarioV2(
                    entities=story.entities,
                    events=story.events,
                    observations=observations,
                    reports=story.reports,
                    receptions=story.receptions,
                    decision=story.decision,
                ),
            )
        )
    return tuple(results)


def _counterfactual_sort_key(
    item: EvolutionCounterfactualV2,
) -> tuple[str, int, str, str, str]:
    intervention = item.intervention
    return (
        intervention.kind,
        -1 if intervention.logical_time is None else intervention.logical_time,
        intervention.subject_id,
        "" if intervention.to_value is None else intervention.to_value,
        "" if intervention.agent is None else intervention.agent,
    )


def analyze_testimony_evolution(
    story: NarrativeScenarioV2,
) -> EvolutionAnalysisV2:
    if not isinstance(story, NarrativeScenarioV2):
        raise TypeError(
            "evolution analysis requires a validated NarrativeScenarioV2"
        )
    baseline = _build_trajectory(story)
    counterfactuals = (
        _change_report_content_counterfactuals(story, baseline)
        + _remove_direct_observation_counterfactuals(story, baseline)
        + _remove_reception_counterfactuals(story, baseline)
        + _remove_support_observation_counterfactuals(story, baseline)
    )
    return EvolutionAnalysisV2(
        baseline=baseline,
        counterfactuals=tuple(sorted(counterfactuals, key=_counterfactual_sort_key)),
    )
from __future__ import annotations

from dataclasses import dataclass, replace

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.analysis import (
    AnalysisScope,
    NarrativeTrajectory,
    build_trajectory,
)
from narrative_dynamics.narrative.decision import (
    DecisionModelSpec,
    DecisionResolutionError,
    EpistemicResolutionError,
)
from narrative_dynamics.narrative.domain import DomainSpec, validate_narrative
from narrative_dynamics.narrative.ir import (
    EntityRef,
    GenericNarrative,
    StateCellRef,
    TypedValue,
)


_ALLOWED_KINDS = frozenset(
    {
        "remove_event",
        "change_event_argument",
        "remove_observation",
        "change_claim_value",
        "remove_reception",
    }
)
_ALLOWED_REJECTION_STAGES = frozenset(
    {
        "domain_validation",
        "narrative_validation",
        "epistemic_resolution",
        "decision_resolution",
    }
)


@dataclass(frozen=True)
class Intervention:
    kind: str
    target_ref: str
    logical_time: int
    field_name: str | None = None
    from_value: TypedValue | None = None
    to_value: TypedValue | None = None

    def __post_init__(self) -> None:
        if self.kind not in _ALLOWED_KINDS:
            raise ValueError("generic intervention kind is not supported")
        if not isinstance(self.target_ref, str) or not self.target_ref:
            raise ValueError("generic intervention target ref must be non-empty")
        if (
            not isinstance(self.logical_time, int)
            or isinstance(self.logical_time, bool)
            or self.logical_time < 0
        ):
            raise ValueError("generic intervention logical time must be non-negative")
        if self.field_name is not None and (
            not isinstance(self.field_name, str) or not self.field_name
        ):
            raise ValueError("generic intervention field name must be non-empty")
        if self.from_value is not None and not isinstance(self.from_value, TypedValue):
            raise TypeError("generic intervention from value must be TypedValue or None")
        if self.to_value is not None and not isinstance(self.to_value, TypedValue):
            raise TypeError("generic intervention to value must be TypedValue or None")

        changing = self.kind in {"change_event_argument", "change_claim_value"}
        if changing:
            if self.field_name is None or self.from_value is None or self.to_value is None:
                raise ValueError("value-changing intervention requires field/from/to values")
            if self.from_value == self.to_value:
                raise ValueError("value-changing intervention must change the value")
        elif any(
            value is not None
            for value in (self.field_name, self.from_value, self.to_value)
        ):
            raise ValueError("removal intervention cannot contain field/from/to values")


@dataclass(frozen=True)
class Divergence:
    logical_time: int
    changed_fields: tuple[str, ...]
    action_changed: bool

    def __post_init__(self) -> None:
        fields = tuple(self.changed_fields)
        if tuple(sorted(set(fields))) != fields:
            raise ValueError("divergence changed fields must be unique and sorted")
        if not fields:
            raise ValueError("divergence requires at least one changed field")
        object.__setattr__(self, "changed_fields", fields)


@dataclass(frozen=True)
class CounterfactualResult:
    intervention: Intervention
    status: str
    trajectory: NarrativeTrajectory | None
    first_divergence: Divergence | None
    rejection_stage: str | None
    rejection_reason: str | None
    rejection_logical_time: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.intervention, Intervention):
            raise TypeError("counterfactual result requires an Intervention")
        if self.status not in {"valid", "rejected"}:
            raise ValueError("counterfactual status must be valid or rejected")
        if self.status == "valid":
            if not isinstance(self.trajectory, NarrativeTrajectory):
                raise TypeError("valid counterfactual requires a trajectory")
            if any(
                value is not None
                for value in (
                    self.rejection_stage,
                    self.rejection_reason,
                    self.rejection_logical_time,
                )
            ):
                raise ValueError("valid counterfactual cannot contain rejection data")
        else:
            if self.trajectory is not None or self.first_divergence is not None:
                raise ValueError("rejected counterfactual cannot contain a trajectory")
            if self.rejection_stage not in _ALLOWED_REJECTION_STAGES:
                raise ValueError("rejected counterfactual requires a typed rejection stage")
            if not isinstance(self.rejection_reason, str) or not self.rejection_reason:
                raise ValueError("rejected counterfactual requires a rejection reason")
            if self.rejection_logical_time != self.intervention.logical_time:
                raise ValueError("rejection logical time must match the intervention")


def _alternate_values(domain: DomainSpec, value_type_name: str, current: TypedValue) -> tuple[TypedValue, ...]:
    value_type = domain._value_type(value_type_name)
    if value_type.kind == "enum":
        return tuple(
            TypedValue(value_type.name, item)
            for item in value_type.allowed_values
            if item != current.value
        )
    if value_type.kind == "bool":
        if not isinstance(current.value, bool):
            raise TypeError("boolean domain value must contain a boolean")
        return (TypedValue(value_type.name, not current.value),)
    return ()


def _claim_alternate_values(
    story: GenericNarrative,
    domain: DomainSpec,
    value_type_name: str,
    current: TypedValue,
) -> tuple[TypedValue, ...]:
    simple = _alternate_values(domain, value_type_name, current)
    if simple:
        return simple

    value_type = domain._value_type(value_type_name)
    if value_type.kind != "entity_ref":
        return ()
    if not isinstance(current.value, EntityRef):
        raise TypeError("entity_ref domain value must contain an EntityRef")

    return tuple(
        TypedValue(
            value_type.name,
            EntityRef(entity.id, entity.type_name),
        )
        for entity in sorted(
            story.entities,
            key=lambda item: (item.type_name, item.id),
        )
        if entity.type_name == value_type.entity_type
        and EntityRef(entity.id, entity.type_name) != current.value
    )


def _decision_time(story: GenericNarrative, scope: AnalysisScope) -> int:
    matches = tuple(item for item in story.decisions if item.id == scope.decision_id)
    if len(matches) != 1:
        raise ValueError("analysis scope decision must identify exactly one decision")
    return matches[0].logical_time


def _scope_relevant_cell(scope: AnalysisScope, subject_id: str, variable: str) -> bool:
    return any(
        cell.subject.entity_id == subject_id and cell.state_variable == variable
        for cell in scope.tracked_state_cells
    )


def generate_minimal_interventions(
    story: GenericNarrative,
    domain: DomainSpec,
    scope: AnalysisScope,
) -> tuple[Intervention, ...]:
    validate_narrative(story, domain)
    if not isinstance(scope, AnalysisScope):
        raise TypeError("intervention generation requires AnalysisScope")
    cutoff = _decision_time(story, scope)
    event_by_id = {item.id: item for item in story.events}
    claim_by_id = {item.id: item for item in story.claims}
    results: list[Intervention] = []

    for event in story.events:
        if event.logical_time > cutoff:
            continue
        event_type = domain._event_type(event.type_name)
        relevant = False
        for effect in event_type.effects:
            subject = event.arguments[effect.subject_parameter].value
            subject_id = getattr(subject, "entity_id", None)
            if isinstance(subject_id, str) and _scope_relevant_cell(
                scope, subject_id, effect.state_variable
            ):
                relevant = True
                break
        if not relevant:
            continue
        results.append(Intervention("remove_event", event.id, event.logical_time))
        parameters = {item.name: item for item in event_type.parameters}
        for name, current in event.arguments.items():
            parameter = parameters[name]
            if not parameter.intervenable:
                continue
            for alternate in _alternate_values(
                domain, parameter.value_type, current
            ):
                results.append(
                    Intervention(
                        "change_event_argument",
                        event.id,
                        event.logical_time,
                        name,
                        current,
                        alternate,
                    )
                )

    for observation in story.observations:
        event = event_by_id.get(observation.event_id)
        if event is None or event.logical_time > cutoff:
            continue
        results.append(
            Intervention("remove_observation", observation.id, event.logical_time)
        )

    for claim in story.claims:
        if claim.logical_time > cutoff:
            continue
        if not _scope_relevant_cell(
            scope,
            claim.proposition.subject.entity_id,
            claim.proposition.state_variable,
        ):
            continue
        state_variable = domain._state_variable(claim.proposition.state_variable)
        for alternate in _claim_alternate_values(
            story,
            domain,
            state_variable.value_type,
            claim.proposition.value,
        ):
            results.append(
                Intervention(
                    "change_claim_value",
                    claim.id,
                    claim.logical_time,
                    "value",
                    claim.proposition.value,
                    alternate,
                )
            )

    for reception in story.receptions:
        claim = claim_by_id.get(reception.claim_id)
        if claim is None or claim.logical_time > cutoff:
            continue
        if not _scope_relevant_cell(
            scope,
            claim.proposition.subject.entity_id,
            claim.proposition.state_variable,
        ):
            continue
        results.append(
            Intervention("remove_reception", reception.id, claim.logical_time)
        )

    def sort_key(item: Intervention) -> tuple[object, ...]:
        value_hash = "" if item.to_value is None else stable_content_hash(item.to_value.to_dict())
        return (
            item.kind,
            item.logical_time,
            item.target_ref,
            "" if item.field_name is None else item.field_name,
            value_hash,
        )

    return tuple(sorted(results, key=sort_key))


def _rebuild_story(
    story: GenericNarrative,
    *,
    events=None,
    observations=None,
    claims=None,
    receptions=None,
) -> GenericNarrative:
    return GenericNarrative(
        domain_id=story.domain_id,
        domain_version=story.domain_version,
        domain_spec_hash=story.domain_spec_hash,
        entities=story.entities,
        events=story.events if events is None else tuple(events),
        observations=story.observations if observations is None else tuple(observations),
        claims=story.claims if claims is None else tuple(claims),
        receptions=story.receptions if receptions is None else tuple(receptions),
        decisions=story.decisions,
        schema_version=story.schema_version,
    )


def _single_match(values, target_ref: str, *, label: str):
    matches = tuple(item for item in values if item.id == target_ref)
    if len(matches) != 1:
        raise ValueError(f"{label} intervention target must identify exactly one record")
    return matches[0]


def _apply_intervention(
    story: GenericNarrative,
    domain: DomainSpec,
    intervention: Intervention,
) -> GenericNarrative:
    if intervention.kind == "remove_event":
        _single_match(story.events, intervention.target_ref, label="event")
        return _rebuild_story(
            story,
            events=tuple(item for item in story.events if item.id != intervention.target_ref),
        )

    if intervention.kind == "remove_observation":
        _single_match(story.observations, intervention.target_ref, label="observation")
        return _rebuild_story(
            story,
            observations=tuple(
                item for item in story.observations if item.id != intervention.target_ref
            ),
        )

    if intervention.kind == "remove_reception":
        _single_match(story.receptions, intervention.target_ref, label="reception")
        return _rebuild_story(
            story,
            receptions=tuple(
                item for item in story.receptions if item.id != intervention.target_ref
            ),
        )

    if intervention.kind == "change_claim_value":
        claim = _single_match(story.claims, intervention.target_ref, label="claim")
        if intervention.field_name != "value":
            raise ValueError("claim intervention may change only proposition value")
        assert intervention.from_value is not None and intervention.to_value is not None
        if claim.proposition.value != intervention.from_value:
            raise ValueError("claim intervention from value does not match the narrative")
        state_variable = domain._state_variable(claim.proposition.state_variable)
        domain._value_type(state_variable.value_type).validate(
            intervention.to_value,
            {item.id: item for item in story.entities},
        )
        claims = tuple(
            replace(
                item,
                proposition=replace(item.proposition, value=intervention.to_value),
            )
            if item.id == claim.id
            else item
            for item in story.claims
        )
        return _rebuild_story(story, claims=claims)

    if intervention.kind == "change_event_argument":
        event = _single_match(story.events, intervention.target_ref, label="event")
        assert intervention.field_name is not None
        assert intervention.from_value is not None and intervention.to_value is not None
        event_type = domain._event_type(event.type_name)
        parameters = {item.name: item for item in event_type.parameters}
        parameter = parameters.get(intervention.field_name)
        if parameter is None:
            raise ValueError("event intervention field must be a declared parameter")
        if not parameter.intervenable:
            raise ValueError("event intervention field must be declared intervenable")
        current = event.arguments.get(intervention.field_name)
        if current != intervention.from_value:
            raise ValueError("event intervention from value does not match the narrative")
        domain._value_type(parameter.value_type).validate(
            intervention.to_value,
            {item.id: item for item in story.entities},
        )
        arguments = dict(event.arguments)
        arguments[intervention.field_name] = intervention.to_value
        events = tuple(
            replace(item, arguments=arguments) if item.id == event.id else item
            for item in story.events
        )
        return _rebuild_story(story, events=events)

    raise ValueError("generic intervention kind is not supported")


def _typed_value(value: TypedValue | None) -> object:
    return None if value is None else value.to_dict()


def _cell_label(cell: StateCellRef) -> str:
    return f"{cell.subject.entity_type}:{cell.subject.entity_id}.{cell.state_variable}"


def _view_fields(prefix: str, view) -> dict[str, object]:
    return {
        prefix + ".status": view.status,
        prefix + ".resolved_value": _typed_value(view.resolved_value),
        prefix + ".constraints": tuple(item.to_dict() for item in view.constraints),
        prefix + ".evidence_kind": view.evidence_kind,
        prefix + ".supporting_id": view.supporting_id,
        prefix + ".source_agent": view.source_agent,
        prefix + ".evidence_logical_time": view.evidence_logical_time,
        prefix + ".evidence_refs": view.evidence_refs,
    }


def _snapshot_fields(snapshot) -> dict[str, object]:
    fields: dict[str, object] = {}
    for cell, value in snapshot.objective_cells.items():
        fields[f"objective.{_cell_label(cell)}"] = _typed_value(value)
    for agent_id, agent_view in snapshot.agent_views.items():
        for cell, view in agent_view.direct_cells.items():
            fields.update(
                _view_fields(
                    f"agents.{agent_id}.direct.{_cell_label(cell)}",
                    view,
                )
            )
        for cell, view in agent_view.epistemic_cells.items():
            fields.update(
                _view_fields(
                    f"agents.{agent_id}.epistemic.{_cell_label(cell)}",
                    view,
                )
            )
    fields["selected_action"] = (
        None
        if snapshot.decision_result is None
        else snapshot.decision_result.selected_action
    )
    return fields


def first_divergence(
    baseline: NarrativeTrajectory,
    candidate: NarrativeTrajectory,
) -> Divergence | None:
    if not isinstance(baseline, NarrativeTrajectory) or not isinstance(
        candidate, NarrativeTrajectory
    ):
        raise TypeError("trajectory comparison requires NarrativeTrajectory values")
    if baseline.scope != candidate.scope:
        raise ValueError("trajectory comparison requires the same frozen analysis scope")
    if tuple(item.logical_time for item in baseline.snapshots) != tuple(
        item.logical_time for item in candidate.snapshots
    ):
        raise ValueError("trajectory comparison requires identical snapshot times")

    action_changed = baseline.selected_action != candidate.selected_action
    for left, right in zip(baseline.snapshots, candidate.snapshots, strict=True):
        left_fields = _snapshot_fields(left)
        right_fields = _snapshot_fields(right)
        if set(left_fields) != set(right_fields):
            raise ValueError("trajectory comparison requires identical canonical fields")
        changed = tuple(
            sorted(
                key
                for key in left_fields
                if left_fields[key] != right_fields[key]
            )
        )
        if changed:
            return Divergence(left.logical_time, changed, action_changed)
    return None


def _rejected(
    intervention: Intervention,
    stage: str,
    error: Exception,
) -> CounterfactualResult:
    return CounterfactualResult(
        intervention=intervention,
        status="rejected",
        trajectory=None,
        first_divergence=None,
        rejection_stage=stage,
        rejection_reason=str(error),
        rejection_logical_time=intervention.logical_time,
    )


def evaluate_counterfactual(
    story: GenericNarrative,
    domain: DomainSpec,
    scope: AnalysisScope,
    model: DecisionModelSpec,
    baseline: NarrativeTrajectory,
    intervention: Intervention,
) -> CounterfactualResult:
    if not isinstance(scope, AnalysisScope):
        raise TypeError("counterfactual evaluation requires AnalysisScope")
    if not isinstance(model, DecisionModelSpec):
        raise TypeError("counterfactual evaluation requires DecisionModelSpec")
    if not isinstance(baseline, NarrativeTrajectory):
        raise TypeError("counterfactual evaluation requires NarrativeTrajectory baseline")
    if baseline.scope != scope:
        raise ValueError("counterfactual baseline must use the supplied frozen scope")
    if not isinstance(intervention, Intervention):
        raise TypeError("counterfactual evaluation requires Intervention")

    try:
        candidate = _apply_intervention(story, domain, intervention)
    except ValueError as error:
        return _rejected(intervention, "domain_validation", error)

    try:
        validate_narrative(candidate, domain)
    except ValueError as error:
        return _rejected(intervention, "narrative_validation", error)

    try:
        trajectory = build_trajectory(candidate, domain, scope, model)
    except EpistemicResolutionError as error:
        return _rejected(intervention, "epistemic_resolution", error)
    except DecisionResolutionError as error:
        return _rejected(intervention, "decision_resolution", error)

    return CounterfactualResult(
        intervention=intervention,
        status="valid",
        trajectory=trajectory,
        first_divergence=first_divergence(baseline, trajectory),
        rejection_stage=None,
        rejection_reason=None,
        rejection_logical_time=None,
    )

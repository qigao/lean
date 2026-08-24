from __future__ import annotations

from narrative_dynamics.narrative.decision import (
    DecisionChoice,
    DecisionModelSpec,
    DecisionResolutionError,
    EvidenceAccess,
    require_resolved_cell,
)
from narrative_dynamics.narrative.domain import (
    ActionTypeSpec,
    DecisionTypeSpec,
    DomainSpec,
    EntityTypeSpec,
    EventTypeSpec,
    ParameterSpec,
    SemanticHookBinding,
    StateDelta,
    StateDeltaOp,
    StateEffectSpec,
    StateVariableSpec,
    ValueTypeSpec,
)
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Claim,
    Decision,
    Entity,
    EntityRef,
    GenericNarrative,
    NarrativeEvent,
    Observation,
    Proposition,
    Reception,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.story.scenario_v2 import NarrativeScenarioV2


_DOMAIN_ID = "legacy-testimony-location"
_DOMAIN_VERSION = "1"
_OBJECT_LOCATION = "object.location"
_EVENT_TYPE = "relocate-object"
_ACTION_TYPE = "search-location"
_DECISION_TYPE = "search-object"
_ABSENT_LOCATION = "__absent__"


def _observation_id(event_id: str, agent_id: str) -> str:
    return f"obs:{event_id}:{agent_id}"


def _reception_id(report_id: str, recipient_id: str) -> str:
    return f"recv:{report_id}:{recipient_id}"


def _entity_ref(entity_id: str, entity_type: str) -> TypedValue:
    return TypedValue(
        f"{entity_type}Ref",
        EntityRef(entity_id, entity_type),
    )


def _location_value(location_id: str) -> TypedValue:
    return TypedValue("LocationRef", EntityRef(location_id, "Location"))


class _RelocationHook:
    def __call__(self, prior_state, event):
        object_ref = event.arguments["object"].value
        if not isinstance(object_ref, EntityRef):
            raise TypeError("relocation object must be an EntityRef")
        if object_ref.entity_type != "Object":
            raise ValueError("relocation object must reference Object")

        cell = StateCellRef(object_ref, _OBJECT_LOCATION)
        current = prior_state.get(cell)
        expected_from = _ABSENT_LOCATION
        if current is not None:
            current_value = current.value
            if not isinstance(current_value, EntityRef):
                raise TypeError("object location state must contain an EntityRef")
            expected_from = current_value.entity_id

        declared_from = event.arguments["from_location"].value
        if declared_from != expected_from:
            raise ValueError(
                "relocation from_location must match the current objective location"
            )

        to_location = event.arguments["to_location"]
        raw_to = to_location.value
        if not isinstance(raw_to, EntityRef) or raw_to.entity_type != "Location":
            raise ValueError("relocation destination must reference Location")

        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    object_ref.entity_id,
                    _OBJECT_LOCATION,
                    to_location,
                ),
            )
        )


class _SearchLocationHook:
    def __call__(self, context):
        if len(context.decision.context_cells) != 1:
            raise DecisionResolutionError(
                "legacy search decision requires exactly one context cell"
            )
        cell = context.decision.context_cells[0]
        view = require_resolved_cell(context, cell)
        assert view.resolved_value is not None
        location = view.resolved_value.value
        if not isinstance(location, EntityRef) or location.entity_type != "Location":
            raise DecisionResolutionError(
                "legacy search state must resolve to a LocationRef"
            )

        matches = tuple(
            action
            for action in context.decision.actions
            if action.arguments.get("location")
            == TypedValue("LocationRef", location)
        )
        if len(matches) != 1:
            raise DecisionResolutionError(
                "resolved location must map to exactly one declared search action"
            )
        evidence_refs = (
            () if view.supporting_id is None else (view.supporting_id,)
        )
        return DecisionChoice(
            matches[0].id,
            evidence_refs,
            (cell,),
        )


def location_testimony_domain() -> DomainSpec:
    return DomainSpec(
        domain_id=_DOMAIN_ID,
        version=_DOMAIN_VERSION,
        entity_types=(
            EntityTypeSpec("Agent"),
            EntityTypeSpec("Object"),
            EntityTypeSpec("Location"),
        ),
        value_types=(
            ValueTypeSpec("ObjectRef", "entity_ref", entity_type="Object"),
            ValueTypeSpec("LocationRef", "entity_ref", entity_type="Location"),
            ValueTypeSpec("LegacyLocationId", "text"),
        ),
        state_variables=(
            StateVariableSpec(_OBJECT_LOCATION, "Object", "LocationRef"),
        ),
        event_types=(
            EventTypeSpec(
                _EVENT_TYPE,
                "Agent",
                (
                    ParameterSpec("object", "ObjectRef"),
                    ParameterSpec("from_location", "LegacyLocationId"),
                    ParameterSpec("to_location", "LocationRef"),
                ),
                (StateEffectSpec(_OBJECT_LOCATION, "object"),),
                "legacy_relocation",
            ),
        ),
        action_types=(
            ActionTypeSpec(
                _ACTION_TYPE,
                (ParameterSpec("location", "LocationRef"),),
            ),
        ),
        decision_types=(
            DecisionTypeSpec(_DECISION_TYPE, "Agent", _ACTION_TYPE),
        ),
        semantic_hooks=(
            SemanticHookBinding(
                "legacy_relocation",
                "apply one validated legacy object relocation",
                _RelocationHook(),
            ),
        ),
    )


def adapt_testimony_story_v2(story: NarrativeScenarioV2) -> GenericNarrative:
    if not isinstance(story, NarrativeScenarioV2):
        raise TypeError("compatibility adapter requires NarrativeScenarioV2")

    all_ids = (
        tuple(story.entities.agents)
        + tuple(story.entities.objects)
        + tuple(story.entities.locations)
    )
    if len(set(all_ids)) != len(all_ids):
        raise ValueError(
            "legacy entity ids must be globally unique for generic adaptation"
        )

    domain = location_testimony_domain()
    entities = (
        tuple(Entity(item, "Agent") for item in story.entities.agents)
        + tuple(Entity(item, "Object") for item in story.entities.objects)
        + tuple(Entity(item, "Location") for item in story.entities.locations)
    )

    events = tuple(
        NarrativeEvent(
            id=event.id,
            logical_time=event.logical_time,
            type_name=_EVENT_TYPE,
            actor_id=event.actor,
            arguments={
                "object": _entity_ref(event.object, "Object"),
                "from_location": TypedValue(
                    "LegacyLocationId",
                    _ABSENT_LOCATION
                    if event.from_location is None
                    else event.from_location,
                ),
                "to_location": _location_value(event.to_location),
            },
        )
        for event in story.events
    )

    observations = tuple(
        Observation(
            _observation_id(item.event, item.agent),
            item.agent,
            item.event,
        )
        for item in story.observations
    )

    claims = tuple(
        Claim(
            id=report.id,
            logical_time=report.logical_time,
            speaker_id=report.speaker,
            proposition=Proposition(
                EntityRef(report.object, "Object"),
                _OBJECT_LOCATION,
                "equals",
                _location_value(report.location),
            ),
            support_refs=(report.support_event,),
        )
        for report in story.reports
    )

    receptions = tuple(
        Reception(
            _reception_id(item.report, item.recipient),
            item.report,
            item.recipient,
        )
        for item in story.receptions
    )

    decision = story.decision
    target_cell = StateCellRef(
        EntityRef(decision.object, "Object"),
        _OBJECT_LOCATION,
    )
    generic_decision = Decision(
        id=decision.id,
        logical_time=decision.time,
        actor_id=decision.actor,
        type_name=_DECISION_TYPE,
        context_cells=(target_cell,),
        actions=tuple(
            ActionOption(
                action.id,
                _ACTION_TYPE,
                {"location": _location_value(action.location)},
            )
            for action in decision.actions
        ),
    )

    return GenericNarrative(
        domain_id=domain.domain_id,
        domain_version=domain.version,
        domain_spec_hash=domain.content_hash,
        entities=entities,
        events=events,
        observations=observations,
        claims=claims,
        receptions=receptions,
        decisions=(generic_decision,),
    )


def _legacy_model(model_id: str, access: EvidenceAccess) -> DecisionModelSpec:
    return DecisionModelSpec(
        model_id=model_id,
        version="1",
        supported_decision_types=(_DECISION_TYPE,),
        evidence_access=access,
        parameter_schema=(),
        decision_hook=_SearchLocationHook(),
    )


def legacy_direct_search_model() -> DecisionModelSpec:
    return _legacy_model("legacy-v2-direct-search", EvidenceAccess.DIRECT_ONLY)


def legacy_epistemic_search_model() -> DecisionModelSpec:
    return _legacy_model("legacy-v2-epistemic-search", EvidenceAccess.EPISTEMIC)


def legacy_omniscient_search_model() -> DecisionModelSpec:
    return _legacy_model("legacy-v2-omniscient-search", EvidenceAccess.OMNISCIENT)

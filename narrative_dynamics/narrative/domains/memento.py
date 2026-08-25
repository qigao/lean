from __future__ import annotations

"""Small film-conformance slice for the Generic Narrative Engine.

The fixture abstracts a revealed sequence from *Memento* without reproducing
screenplay dialogue. Leonard is represented as two epistemic time slices around
a memory boundary. Teddy's explanation remains testimony rather than objective
truth, while Leonard's later persistent record is modeled as a self-authored
claim that can be received by the post-reset time slice.
"""

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


def _case_ref() -> EntityRef:
    return EntityRef("john-g-case", "Case")


def _case_cell() -> StateCellRef:
    return StateCellRef(_case_ref(), "case.directive")


def _record_ref() -> EntityRef:
    return EntityRef("teddy-target-record", "MemoryRecord")


def _agent_ref(agent_id: str) -> EntityRef:
    return EntityRef(agent_id, "Agent")


class ExplanationContextHook:
    def __call__(self, prior_state, event):
        del prior_state, event
        return StateDelta(())


class WriteMemoryRecordHook:
    def __call__(self, prior_state, event):
        del prior_state
        record = event.arguments["record"].value
        if not isinstance(record, EntityRef):
            raise TypeError("memory record writing requires a MemoryRecordRef")
        if event.actor_id is None:
            raise ValueError("memory record writing requires an actor")
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    record.entity_id,
                    "record.author",
                    TypedValue("AgentRef", _agent_ref(event.actor_id)),
                ),
            )
        )


def memento_domain() -> DomainSpec:
    return DomainSpec(
        domain_id="film-memento-conformance",
        version="1",
        entity_types=(
            EntityTypeSpec("Agent"),
            EntityTypeSpec("Case"),
            EntityTypeSpec("MemoryRecord"),
        ),
        value_types=(
            ValueTypeSpec("AgentRef", "entity_ref", entity_type="Agent"),
            ValueTypeSpec("MemoryRecordRef", "entity_ref", entity_type="MemoryRecord"),
            ValueTypeSpec(
                "Directive",
                "enum",
                ("target_teddy", "continue_search"),
            ),
        ),
        state_variables=(
            StateVariableSpec("case.directive", "Case", "Directive"),
            StateVariableSpec("record.author", "MemoryRecord", "AgentRef"),
        ),
        event_types=(
            EventTypeSpec(
                "ExplanationContext",
                "Agent",
                (),
                (),
                "explanation_context",
            ),
            EventTypeSpec(
                "WriteMemoryRecord",
                "Agent",
                (ParameterSpec("record", "MemoryRecordRef"),),
                (StateEffectSpec("record.author", "record"),),
                "write_memory_record",
            ),
        ),
        action_types=(ActionTypeSpec("investigation-action", ()),),
        decision_types=(
            DecisionTypeSpec(
                "post-reset-target-choice",
                "Agent",
                "investigation-action",
            ),
        ),
        semantic_hooks=(
            SemanticHookBinding(
                "explanation_context",
                "record an observed explanation context without asserting its claims as truth",
                ExplanationContextHook(),
            ),
            SemanticHookBinding(
                "write_memory_record",
                "record the objective authorship of a persistent external-memory artifact",
                WriteMemoryRecordHook(),
            ),
        ),
    )


def memento_self_record_story() -> GenericNarrative:
    domain = memento_domain()
    case = _case_ref()
    record = _record_ref()
    continue_search = TypedValue("Directive", "continue_search")
    target_teddy = TypedValue("Directive", "target_teddy")
    return GenericNarrative(
        domain_id=domain.domain_id,
        domain_version=domain.version,
        domain_spec_hash=domain.content_hash,
        entities=(
            Entity("teddy", "Agent"),
            Entity("leonard-before-reset", "Agent"),
            Entity("leonard-after-reset", "Agent"),
            Entity("john-g-case", "Case"),
            Entity("teddy-target-record", "MemoryRecord"),
        ),
        events=(
            NarrativeEvent(
                "teddy-explanation-context",
                1,
                "ExplanationContext",
                "teddy",
                {},
            ),
            NarrativeEvent(
                "write-target-record",
                3,
                "WriteMemoryRecord",
                "leonard-before-reset",
                {"record": TypedValue("MemoryRecordRef", record)},
            ),
        ),
        observations=(
            Observation(
                "obs-teddy-explanation-context",
                "teddy",
                "teddy-explanation-context",
            ),
            Observation(
                "obs-leonard-write-target-record",
                "leonard-before-reset",
                "write-target-record",
            ),
        ),
        claims=(
            Claim(
                "teddy-explanation",
                2,
                "teddy",
                Proposition(
                    case,
                    "case.directive",
                    "equals",
                    continue_search,
                ),
                ("teddy-explanation-context",),
            ),
            Claim(
                "leonard-target-record",
                4,
                "leonard-before-reset",
                Proposition(
                    case,
                    "case.directive",
                    "equals",
                    target_teddy,
                ),
                ("write-target-record",),
            ),
        ),
        receptions=(
            Reception(
                "recv-before-reset-teddy-explanation",
                "teddy-explanation",
                "leonard-before-reset",
            ),
            Reception(
                "recv-after-reset-self-record",
                "leonard-target-record",
                "leonard-after-reset",
            ),
        ),
        decisions=(
            Decision(
                "post-reset-target-choice",
                5,
                "leonard-after-reset",
                "post-reset-target-choice",
                (_case_cell(),),
                (
                    ActionOption("pursue_teddy", "investigation-action", {}),
                    ActionOption("continue_search", "investigation-action", {}),
                ),
            ),
        ),
    )


class PostResetTargetChoiceHook:
    def __call__(self, context):
        view = require_resolved_cell(context, _case_cell())
        assert view.resolved_value is not None
        directive = view.resolved_value.value
        if directive == "target_teddy":
            action = "pursue_teddy"
        elif directive == "continue_search":
            action = "continue_search"
        else:
            raise DecisionResolutionError(
                "resolved directive does not map to a declared post-reset action"
            )
        return DecisionChoice(
            selected_action=action,
            evidence_refs=(
                () if view.supporting_id is None else (view.supporting_id,)
            ),
            inspected_cells=(_case_cell(),),
        )


def _memento_model(model_id: str, access: EvidenceAccess) -> DecisionModelSpec:
    return DecisionModelSpec(
        model_id=model_id,
        version="1",
        supported_decision_types=("post-reset-target-choice",),
        evidence_access=access,
        parameter_schema=(),
        decision_hook=PostResetTargetChoiceHook(),
    )


def memento_direct_model() -> DecisionModelSpec:
    return _memento_model("memento-direct", EvidenceAccess.DIRECT_ONLY)


def memento_epistemic_model() -> DecisionModelSpec:
    return _memento_model("memento-epistemic", EvidenceAccess.EPISTEMIC)


def memento_omniscient_model() -> DecisionModelSpec:
    return _memento_model("memento-omniscient", EvidenceAccess.OMNISCIENT)

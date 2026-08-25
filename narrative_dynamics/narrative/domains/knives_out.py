from __future__ import annotations

"""Small film-conformance slice for the Generic Narrative Engine.

The fixture abstracts revealed plot facts from *Knives Out* into canonical
semantics. It intentionally stores no screenplay dialogue: Ransom's original
medication tampering establishes the objective culprit, Fran later observes
Ransom restoring/handling the medication evidence, and Ransom's confession is
received by Blanc and Marta.
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
    return EntityRef("harlan-case", "Case")


def _case_cell() -> StateCellRef:
    return StateCellRef(_case_ref(), "case.culprit")


def _agent_ref(agent_id: str) -> EntityRef:
    return EntityRef(agent_id, "Agent")


def _culprit_value(agent_id: str) -> TypedValue:
    return TypedValue("AgentRef", _agent_ref(agent_id))


class MedicationTamperingHook:
    def __call__(self, prior_state, event):
        del prior_state
        case = event.arguments["case"].value
        if not isinstance(case, EntityRef):
            raise TypeError("medication tampering requires a CaseRef")
        if event.actor_id is None:
            raise ValueError("medication tampering requires an actor")
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    case.entity_id,
                    "case.culprit",
                    _culprit_value(event.actor_id),
                ),
            )
        )


class MedicationEvidenceRestorationHook:
    def __call__(self, prior_state, event):
        case = event.arguments["case"].value
        if not isinstance(case, EntityRef):
            raise TypeError("medication evidence restoration requires a CaseRef")
        if event.actor_id is None:
            raise ValueError("medication evidence restoration requires an actor")
        expected = _culprit_value(event.actor_id)
        if prior_state.get(StateCellRef(case, "case.culprit")) != expected:
            raise ValueError(
                "medication evidence restoration must match the established culprit"
            )
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    case.entity_id,
                    "case.culprit",
                    expected,
                ),
            )
        )


def knives_out_domain() -> DomainSpec:
    return DomainSpec(
        domain_id="film-knives-out-conformance",
        version="1",
        entity_types=(EntityTypeSpec("Agent"), EntityTypeSpec("Case")),
        value_types=(
            ValueTypeSpec("AgentRef", "entity_ref", entity_type="Agent"),
            ValueTypeSpec("CaseRef", "entity_ref", entity_type="Case"),
        ),
        state_variables=(
            StateVariableSpec("case.culprit", "Case", "AgentRef"),
        ),
        event_types=(
            EventTypeSpec(
                "MedicationTampering",
                "Agent",
                (ParameterSpec("case", "CaseRef"),),
                (StateEffectSpec("case.culprit", "case"),),
                "medication_tampering",
            ),
            EventTypeSpec(
                "MedicationEvidenceRestoration",
                "Agent",
                (ParameterSpec("case", "CaseRef"),),
                (StateEffectSpec("case.culprit", "case"),),
                "medication_evidence_restoration",
            ),
        ),
        action_types=(ActionTypeSpec("investigation-action", ()),),
        decision_types=(
            DecisionTypeSpec(
                "investigative-focus",
                "Agent",
                "investigation-action",
            ),
        ),
        semantic_hooks=(
            SemanticHookBinding(
                "medication_tampering",
                "record the agent responsible for medication tampering",
                MedicationTamperingHook(),
            ),
            SemanticHookBinding(
                "medication_evidence_restoration",
                "confirm the established culprit when medication evidence is restored",
                MedicationEvidenceRestorationHook(),
            ),
        ),
    )


def knives_out_confession_story() -> GenericNarrative:
    domain = knives_out_domain()
    case = _case_ref()
    ransom = _agent_ref("ransom")
    return GenericNarrative(
        domain_id=domain.domain_id,
        domain_version=domain.version,
        domain_spec_hash=domain.content_hash,
        entities=(
            Entity("ransom", "Agent"),
            Entity("marta", "Agent"),
            Entity("fran", "Agent"),
            Entity("blanc", "Agent"),
            Entity("harlan-case", "Case"),
        ),
        events=(
            NarrativeEvent(
                "tamper-medication",
                1,
                "MedicationTampering",
                "ransom",
                {"case": TypedValue("CaseRef", case)},
            ),
            NarrativeEvent(
                "restore-medication-evidence",
                2,
                "MedicationEvidenceRestoration",
                "ransom",
                {"case": TypedValue("CaseRef", case)},
            ),
        ),
        observations=(
            Observation("obs-ransom-tamper", "ransom", "tamper-medication"),
            Observation(
                "obs-fran-restore",
                "fran",
                "restore-medication-evidence",
            ),
        ),
        claims=(
            Claim(
                "ransom-confession",
                3,
                "ransom",
                Proposition(
                    case,
                    "case.culprit",
                    "equals",
                    TypedValue("AgentRef", ransom),
                ),
                ("tamper-medication",),
            ),
        ),
        receptions=(
            Reception("recv-blanc-confession", "ransom-confession", "blanc"),
            Reception("recv-marta-confession", "ransom-confession", "marta"),
        ),
        decisions=(
            Decision(
                "blanc-focus",
                4,
                "blanc",
                "investigative-focus",
                (_case_cell(),),
                (
                    ActionOption("focus_ransom", "investigation-action", {}),
                    ActionOption("focus_marta", "investigation-action", {}),
                ),
            ),
        ),
    )


class InvestigationFocusHook:
    def __call__(self, context):
        view = require_resolved_cell(context, _case_cell())
        assert view.resolved_value is not None
        culprit = view.resolved_value.value
        if not isinstance(culprit, EntityRef) or culprit.entity_type != "Agent":
            raise DecisionResolutionError(
                "resolved culprit must be an AgentRef"
            )
        if culprit.entity_id == "ransom":
            action = "focus_ransom"
        elif culprit.entity_id == "marta":
            action = "focus_marta"
        else:
            raise DecisionResolutionError(
                "resolved culprit does not map to a declared investigative focus"
            )
        return DecisionChoice(
            selected_action=action,
            evidence_refs=(
                () if view.supporting_id is None else (view.supporting_id,)
            ),
            inspected_cells=(_case_cell(),),
        )


def _knives_out_model(model_id: str, access: EvidenceAccess) -> DecisionModelSpec:
    return DecisionModelSpec(
        model_id=model_id,
        version="1",
        supported_decision_types=("investigative-focus",),
        evidence_access=access,
        parameter_schema=(),
        decision_hook=InvestigationFocusHook(),
    )


def knives_out_direct_model() -> DecisionModelSpec:
    return _knives_out_model("knives-out-direct", EvidenceAccess.DIRECT_ONLY)


def knives_out_epistemic_model() -> DecisionModelSpec:
    return _knives_out_model("knives-out-epistemic", EvidenceAccess.EPISTEMIC)


def knives_out_omniscient_model() -> DecisionModelSpec:
    return _knives_out_model("knives-out-omniscient", EvidenceAccess.OMNISCIENT)

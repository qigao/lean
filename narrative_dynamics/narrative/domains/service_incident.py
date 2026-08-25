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


HEALTH_VALUES = ("healthy", "failed", "recovered")


def _service_ref() -> EntityRef:
    return EntityRef("svc", "Service")


def _health_cell() -> StateCellRef:
    return StateCellRef(_service_ref(), "service.health")


class ServiceFailureHook:
    def __call__(self, prior_state, event):
        service = event.arguments["service"].value
        if not isinstance(service, EntityRef):
            raise TypeError("service failure requires a ServiceRef")
        cell = StateCellRef(service, "service.health")
        current = prior_state.get(cell)
        if current is not None and current.value != "healthy":
            raise ValueError("service failure requires healthy prior state")
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    service.entity_id,
                    "service.health",
                    TypedValue("HealthState", "failed"),
                ),
            )
        )


class ServiceRecoveryHook:
    def __call__(self, prior_state, event):
        service = event.arguments["service"].value
        if not isinstance(service, EntityRef):
            raise TypeError("service recovery requires a ServiceRef")
        cell = StateCellRef(service, "service.health")
        if prior_state.get(cell) != TypedValue("HealthState", "failed"):
            raise ValueError("service recovery requires failed prior state")
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    service.entity_id,
                    "service.health",
                    TypedValue("HealthState", "recovered"),
                ),
            )
        )


def service_incident_domain() -> DomainSpec:
    return DomainSpec(
        domain_id="service-incident",
        version="1",
        entity_types=(EntityTypeSpec("Agent"), EntityTypeSpec("Service")),
        value_types=(
            ValueTypeSpec("ServiceRef", "entity_ref", entity_type="Service"),
            ValueTypeSpec("HealthState", "enum", allowed_values=HEALTH_VALUES),
        ),
        state_variables=(
            StateVariableSpec("service.health", "Service", "HealthState"),
        ),
        event_types=(
            EventTypeSpec(
                "ServiceFailure",
                None,
                (ParameterSpec("service", "ServiceRef"),),
                (StateEffectSpec("service.health", "service"),),
                "service_failure",
            ),
            EventTypeSpec(
                "ServiceRecovery",
                None,
                (ParameterSpec("service", "ServiceRef"),),
                (StateEffectSpec("service.health", "service"),),
                "service_recovery",
            ),
        ),
        action_types=(ActionTypeSpec("service-action", ()),),
        decision_types=(
            DecisionTypeSpec("service-response", "Agent", "service-action"),
        ),
        semantic_hooks=(
            SemanticHookBinding(
                "service_failure",
                "transition a service from healthy or unknown to failed",
                ServiceFailureHook(),
            ),
            SemanticHookBinding(
                "service_recovery",
                "transition a failed service to recovered",
                ServiceRecoveryHook(),
            ),
        ),
    )


def _service_incident_story(claim_value: str) -> GenericNarrative:
    if claim_value not in {"failed", "recovered"}:
        raise ValueError("service incident claim must be failed or recovered")
    domain = service_incident_domain()
    service = _service_ref()
    return GenericNarrative(
        domain_id=domain.domain_id,
        domain_version=domain.version,
        domain_spec_hash=domain.content_hash,
        entities=(
            Entity("bob", "Agent"),
            Entity("alice", "Agent"),
            Entity("svc", "Service"),
        ),
        events=(
            NarrativeEvent(
                "e1",
                1,
                "ServiceFailure",
                None,
                {"service": TypedValue("ServiceRef", service)},
            ),
            NarrativeEvent(
                "e2",
                2,
                "ServiceRecovery",
                None,
                {"service": TypedValue("ServiceRef", service)},
            ),
        ),
        observations=(
            Observation("o1", "bob", "e1"),
            Observation("o2", "alice", "e2"),
        ),
        claims=(
            Claim(
                "c1",
                3,
                "alice",
                Proposition(
                    service,
                    "service.health",
                    "equals",
                    TypedValue("HealthState", claim_value),
                ),
                ("e2",),
            ),
        ),
        receptions=(Reception("r1", "c1", "bob"),),
        decisions=(
            Decision(
                "d1",
                4,
                "bob",
                "service-response",
                (_health_cell(),),
                (
                    ActionOption("restart_service", "service-action", {}),
                    ActionOption("leave_running", "service-action", {}),
                ),
            ),
        ),
    )


def service_incident_recovered_claim_story() -> GenericNarrative:
    return _service_incident_story("recovered")


def service_incident_stale_claim_story() -> GenericNarrative:
    return _service_incident_story("failed")


class ServiceResponseHook:
    def __call__(self, context):
        view = require_resolved_cell(context, _health_cell())
        assert view.resolved_value is not None
        if view.resolved_value.value == "failed":
            action = "restart_service"
        elif view.resolved_value.value == "recovered":
            action = "leave_running"
        else:
            raise DecisionResolutionError(
                "resolved service health does not map to a declared response"
            )
        return DecisionChoice(
            selected_action=action,
            evidence_refs=(
                () if view.supporting_id is None else (view.supporting_id,)
            ),
            inspected_cells=(_health_cell(),),
        )


def _service_model(model_id: str, access: EvidenceAccess) -> DecisionModelSpec:
    return DecisionModelSpec(
        model_id=model_id,
        version="1",
        supported_decision_types=("service-response",),
        evidence_access=access,
        parameter_schema=(),
        decision_hook=ServiceResponseHook(),
    )


def service_direct_model() -> DecisionModelSpec:
    return _service_model("service-direct", EvidenceAccess.DIRECT_ONLY)


def service_epistemic_model() -> DecisionModelSpec:
    return _service_model("service-epistemic", EvidenceAccess.EPISTEMIC)


def service_omniscient_model() -> DecisionModelSpec:
    return _service_model("service-omniscient", EvidenceAccess.OMNISCIENT)

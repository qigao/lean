from __future__ import annotations

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


class TestSetHealthHook:
    def __call__(self, prior_state, event):
        service = event.arguments["service"].value
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    service.entity_id,
                    "service.health",
                    event.arguments["health"],
                ),
            )
        )


def make_test_domain() -> DomainSpec:
    return DomainSpec(
        domain_id="test-service",
        version="1",
        entity_types=(EntityTypeSpec("Agent"), EntityTypeSpec("Service")),
        value_types=(
            ValueTypeSpec("ServiceRef", "entity_ref", entity_type="Service"),
            ValueTypeSpec(
                "HealthState",
                "enum",
                allowed_values=("healthy", "failed", "recovered"),
            ),
        ),
        state_variables=(
            StateVariableSpec("service.health", "Service", "HealthState"),
        ),
        event_types=(
            EventTypeSpec(
                "SetHealth",
                None,
                (
                    ParameterSpec("service", "ServiceRef"),
                    ParameterSpec("health", "HealthState", intervenable=True),
                ),
                (StateEffectSpec("service.health", "service"),),
                "set_health",
            ),
        ),
        action_types=(ActionTypeSpec("service-action", ()),),
        decision_types=(
            DecisionTypeSpec("service-response", "Agent", "service-action"),
        ),
        semantic_hooks=(
            SemanticHookBinding(
                "set_health", "set one health cell", TestSetHealthHook()
            ),
        ),
    )


def target_cell() -> StateCellRef:
    return StateCellRef(EntityRef("svc", "Service"), "service.health")


def make_test_story(
    *,
    claim_value: str = "recovered",
    claim_relation: str = "equals",
    receive: bool = True,
    second_claim_value: str | None = None,
    bob_observes_failure: bool = True,
    alice_observes_recovery: bool = True,
) -> GenericNarrative:
    domain = make_test_domain()
    observations: list[Observation] = []
    if bob_observes_failure:
        observations.append(Observation("o1", "bob", "e1"))
    if alice_observes_recovery:
        observations.append(Observation("o2", "alice", "e2"))
    if second_claim_value is not None:
        observations.append(Observation("o3", "carol", "e2"))

    claims = [
        Claim(
            "c1",
            3,
            "alice",
            Proposition(
                EntityRef("svc", "Service"),
                "service.health",
                claim_relation,
                TypedValue("HealthState", claim_value),
            ),
            ("e2",),
        )
    ]
    receptions = [Reception("r1", "c1", "bob")] if receive else []
    decision_time = 4
    if second_claim_value is not None:
        claims.append(
            Claim(
                "c2",
                4,
                "carol",
                Proposition(
                    EntityRef("svc", "Service"),
                    "service.health",
                    "equals",
                    TypedValue("HealthState", second_claim_value),
                ),
                ("e2",),
            )
        )
        if receive:
            receptions.append(Reception("r2", "c2", "bob"))
        decision_time = 5

    return GenericNarrative(
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (
            Entity("bob", "Agent"),
            Entity("alice", "Agent"),
            Entity("carol", "Agent"),
            Entity("svc", "Service"),
        ),
        (
            NarrativeEvent(
                "e1",
                1,
                "SetHealth",
                None,
                {
                    "service": TypedValue(
                        "ServiceRef", EntityRef("svc", "Service")
                    ),
                    "health": TypedValue("HealthState", "failed"),
                },
            ),
            NarrativeEvent(
                "e2",
                2,
                "SetHealth",
                None,
                {
                    "service": TypedValue(
                        "ServiceRef", EntityRef("svc", "Service")
                    ),
                    "health": TypedValue("HealthState", "recovered"),
                },
            ),
        ),
        tuple(observations),
        tuple(claims),
        tuple(receptions),
        (
            Decision(
                "d1",
                decision_time,
                "bob",
                "service-response",
                (target_cell(),),
                (
                    ActionOption("restart", "service-action", {}),
                    ActionOption("leave", "service-action", {}),
                ),
            ),
        ),
    )


class RecordingFirstActionHook:
    def __init__(self) -> None:
        self.context = None

    def __call__(self, context):
        from narrative_dynamics.narrative.decision import DecisionChoice

        self.context = context
        return DecisionChoice(context.decision.actions[0].id, (), ())


class HealthActionHook:
    def __call__(self, context):
        from narrative_dynamics.narrative.decision import (
            DecisionChoice,
            DecisionResolutionError,
            require_resolved_cell,
        )

        view = require_resolved_cell(context, target_cell())
        value = view.resolved_value.value
        if value == "failed":
            action = "restart"
        elif value == "recovered":
            action = "leave"
        else:
            raise DecisionResolutionError(
                "resolved health does not map to a declared service response"
            )
        return DecisionChoice(
            action,
            () if view.supporting_id is None else (view.supporting_id,),
            (target_cell(),),
        )


class MissingActionHook:
    def __call__(self, context):
        from narrative_dynamics.narrative.decision import DecisionChoice

        return DecisionChoice("missing", (), ())


def make_model(model_id, access, hook):
    from narrative_dynamics.narrative.decision import DecisionModelSpec

    return DecisionModelSpec(
        model_id=model_id,
        version="1",
        supported_decision_types=("service-response",),
        evidence_access=access,
        parameter_schema=(),
        decision_hook=hook,
    )


def make_health_model(model_id, access):
    return make_model(model_id, access, HealthActionHook())

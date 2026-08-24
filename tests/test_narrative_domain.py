from __future__ import annotations

import unittest

_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from narrative_dynamics.narrative.domain import (
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
except ModuleNotFoundError as error:
    _IMPORT_ERROR = error

from narrative_dynamics.narrative.ir import Entity, EntityRef, NarrativeEvent, TypedValue


class SetHealthHook:
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


class IllegalHook:
    def __call__(self, prior_state, event):
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    "svc",
                    "service.secret",
                    TypedValue("HealthState", "failed"),
                ),
            )
        )


def make_domain(hook):
    return DomainSpec(
        domain_id="service-test",
        version="1",
        entity_types=(EntityTypeSpec("Service"),),
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
        action_types=(),
        decision_types=(),
        semantic_hooks=(
            SemanticHookBinding("set_health", "set service health", hook),
        ),
    )


class DomainSpecTests(unittest.TestCase):
    def test_hook_identity_and_delta_boundary(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"generic narrative domain semantics are missing: {_IMPORT_ERROR}")

        event = NarrativeEvent(
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
        )
        entities = {"svc": Entity("svc", "Service")}
        domain = make_domain(SetHealthHook())
        self.assertTrue(domain.content_hash.startswith("sha256:"))
        self.assertTrue(
            domain.semantic_hooks[0].implementation_hash.startswith("sha256:")
        )
        self.assertEqual(
            domain.apply_event({}, event, entities).operations[0].state_variable,
            "service.health",
        )
        with self.assertRaisesRegex(ValueError, "undeclared state variable"):
            make_domain(IllegalHook()).apply_event({}, event, entities)


if __name__ == "__main__":
    unittest.main()

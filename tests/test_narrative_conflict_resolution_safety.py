from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from narrative_dynamics.narrative.domain import (
    ActionTypeSpec,
    DecisionTypeSpec,
    ParameterSpec,
    StateDelta,
    StateDeltaOp,
)
from narrative_dynamics.narrative.ir import ActionOption, Decision, TypedValue
from narrative_dynamics.narrative.world import (
    ActionEffectSpec,
    ActionTransitionSpec,
    WorldTransitionModelSpec,
    advance_world_step,
    world_state_from_story,
)
import tests.test_narrative_conflict_resolution as base


_CONFLICT_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.conflict import ConflictResolverSpec
    from narrative_dynamics.narrative.world import WorldTransitionConflictResolutionError
except ImportError as error:
    _CONFLICT_IMPORT_ERROR = error


class OutsideCapabilityResolver:
    def __call__(self, snapshot, context):
        return StateDelta((StateDeltaOp(
            "set", "svc", "service.health",
            TypedValue("HealthState", "recovered"),
        ),))


class WrongOwnerValueResolver:
    def __call__(self, snapshot, context):
        return StateDelta((StateDeltaOp(
            "set", "svc", "service.owner",
            TypedValue("HealthState", "recovered"),
        ),))


class BadClearResolver:
    def __call__(self, snapshot, context):
        return StateDelta((StateDeltaOp(
            "clear", "svc", "service.owner", base._agent_ref("alice"),
        ),))


class MissingSetValueResolver:
    def __call__(self, snapshot, context):
        return StateDelta((StateDeltaOp(
            "set", "svc", "service.owner", None,
        ),))


class DuplicateOwnerWriteResolver:
    def __call__(self, snapshot, context):
        return StateDelta((
            StateDeltaOp("set", "svc", "service.owner", base._agent_ref("alice")),
            StateDeltaOp("set", "svc", "service.owner", base._agent_ref("bob")),
        ))


class ClaimAuditTransition:
    def __call__(self, snapshot, decision, action):
        service = action.arguments["service"].value.entity_id
        return StateDelta((StateDeltaOp(
            "set", service, "service.owner", base._agent_ref(decision.actor_id),
        ),))


class ClaimAuditResolver:
    def __init__(self):
        self.calls = []

    def __call__(self, snapshot, context):
        self.calls.append((dict(snapshot), context))
        winner = min(context.participants, key=lambda item: item.actor_id)
        service = winner.action.arguments["service"].value.entity_id
        audit = winner.action.arguments["audit"].value.entity_id
        return StateDelta((
            StateDeltaOp("set", service, "service.owner", base._agent_ref(winner.actor_id)),
            StateDeltaOp(
                "set", audit, "service.health",
                TypedValue("HealthState", "recovered"),
            ),
        ))


class FailSecondClaimAuditResolver:
    def __init__(self):
        self.calls = []

    def __call__(self, snapshot, context):
        self.calls.append((dict(snapshot), context))
        winner = min(context.participants, key=lambda item: item.actor_id)
        service = winner.action.arguments["service"].value.entity_id
        if service == "svc2":
            return StateDelta((StateDeltaOp(
                "set", service, "service.owner",
                TypedValue("HealthState", "recovered"),
            ),))
        return StateDelta((StateDeltaOp(
            "set", service, "service.owner", base._agent_ref(winner.actor_id),
        ),))


def make_safety_domain():
    domain = base.make_conflict_domain()
    return replace(
        domain,
        action_types=domain.action_types + (
            ActionTypeSpec(
                "claim-audit-action",
                (
                    ParameterSpec("service", "ServiceRef"),
                    ParameterSpec("audit", "ServiceRef"),
                ),
            ),
        ),
        decision_types=domain.decision_types + (
            DecisionTypeSpec(
                "claim-audit-choice", "Agent", "claim-audit-action",
            ),
        ),
    )


def make_safety_story(domain):
    story = base.make_conflict_story(domain)
    decisions = list(story.decisions)
    next_time = max(item.logical_time for item in decisions) + 1

    def add(decision):
        nonlocal next_time
        if not decision.context_cells:
            decision = replace(
                decision,
                context_cells=(base._cell("svc", "Service", "service.health"),),
            )
        decisions.append(replace(decision, logical_time=next_time))
        next_time += 1

    for decision_id, actor_id, action_id, service_id in (
        ("d-alice-claim-audit-svc", "alice", "alice-claim-audit-svc", "svc"),
        ("d-bob-claim-audit-svc", "bob", "bob-claim-audit-svc", "svc"),
        ("d-carol-claim-audit-svc2", "carol", "carol-claim-audit-svc2", "svc2"),
        ("d-dave-claim-audit-svc2", "dave", "dave-claim-audit-svc2", "svc2"),
    ):
        add(Decision(
            decision_id, 0, actor_id, "claim-audit-choice", (),
            (ActionOption(
                action_id,
                "claim-audit-action",
                {
                    "service": base._service_ref_for(service_id),
                    "audit": base._service_ref_for("svc3"),
                },
            ),),
        ))
    add(Decision(
        "d-carol-health-svc3", 0, "carol", "service-choice", (),
        (ActionOption(
            "carol-recover-svc3",
            "service-health-action",
            {
                "service": base._service_ref_for("svc3"),
                "health": TypedValue("HealthState", "recovered"),
            },
        ),),
    ))
    return replace(story, domain_spec_hash=domain.content_hash, decisions=tuple(decisions))


def make_safety_world_model(domain, resolver):
    base_model = base.make_conflict_world_model(domain, resolver)
    return replace(
        base_model,
        transitions=base_model.transitions + (
            ActionTransitionSpec(
                "claim-audit-action",
                (
                    ActionEffectSpec("service.owner", "argument", "service"),
                    ActionEffectSpec("service.health", "argument", "audit"),
                ),
                ClaimAuditTransition(),
            ),
        ),
    )


def make_resolver(domain, hook, action_types):
    return ConflictResolverSpec(
        "safety-resolver",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        tuple(action_types),
        hook,
    )


class NarrativeConflictResolutionSafetyTests(unittest.TestCase):
    def require_conflict(self) -> None:
        if _CONFLICT_IMPORT_ERROR is not None:
            self.fail(
                "narrative conflict resolution safety boundary is missing: "
                f"{_CONFLICT_IMPORT_ERROR}"
            )

    def test_invalid_resolver_outputs_are_typed_and_prior_remains_unchanged(self):
        self.require_conflict()
        domain = base.make_conflict_domain()
        story = base.make_conflict_story(domain)
        for hook in (
            OutsideCapabilityResolver(),
            WrongOwnerValueResolver(),
            BadClearResolver(),
            MissingSetValueResolver(),
            DuplicateOwnerWriteResolver(),
        ):
            with self.subTest(hook=type(hook).__name__):
                resolver = make_resolver(domain, hook, ("claim-service-action",))
                model = base.make_conflict_world_model(domain, resolver)
                prior = world_state_from_story(story, domain)
                before = prior.to_dict()
                with self.assertRaises(WorldTransitionConflictResolutionError):
                    advance_world_step(
                        story, domain, prior, model,
                        (
                            base._conflict_intent("d-alice-claim-svc", "alice-claim-svc", "1"),
                            base._conflict_intent("d-bob-claim-svc", "bob-claim-svc", "2"),
                        ),
                    )
                self.assertEqual(prior.to_dict(), before)

    def test_one_bad_component_aborts_entire_world_step(self):
        self.require_conflict()
        domain = make_safety_domain()
        story = make_safety_story(domain)
        hook = FailSecondClaimAuditResolver()
        resolver = make_resolver(domain, hook, ("claim-audit-action",))
        model = make_safety_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        before = prior.to_dict()
        with self.assertRaises(WorldTransitionConflictResolutionError):
            advance_world_step(
                story, domain, prior, model,
                (
                    base._conflict_intent("d-alice-claim-audit-svc", "alice-claim-audit-svc", "1"),
                    base._conflict_intent("d-bob-claim-audit-svc", "bob-claim-audit-svc", "2"),
                    base._conflict_intent("d-carol-claim-audit-svc2", "carol-claim-audit-svc2", "3"),
                    base._conflict_intent("d-dave-claim-audit-svc2", "dave-claim-audit-svc2", "4"),
                ),
            )
        self.assertEqual(len(hook.calls), 2)
        self.assertEqual(prior.to_dict(), before)

    def test_resolver_vs_nonconflicting_write_collision_rejects(self):
        self.require_conflict()
        domain = make_safety_domain()
        story = make_safety_story(domain)
        resolver = make_resolver(domain, ClaimAuditResolver(), ("claim-audit-action",))
        model = make_safety_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        before = prior.to_dict()
        with self.assertRaises(WorldTransitionConflictResolutionError):
            advance_world_step(
                story, domain, prior, model,
                (
                    base._conflict_intent("d-alice-claim-audit-svc", "alice-claim-audit-svc", "1"),
                    base._conflict_intent("d-bob-claim-audit-svc", "bob-claim-audit-svc", "2"),
                    base._conflict_intent("d-carol-health-svc3", "carol-recover-svc3", "3"),
                ),
            )
        self.assertEqual(prior.to_dict(), before)

    def test_resolution_vs_resolution_write_collision_rejects(self):
        self.require_conflict()
        domain = make_safety_domain()
        story = make_safety_story(domain)
        resolver = make_resolver(domain, ClaimAuditResolver(), ("claim-audit-action",))
        model = make_safety_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        before = prior.to_dict()
        with self.assertRaises(WorldTransitionConflictResolutionError):
            advance_world_step(
                story, domain, prior, model,
                (
                    base._conflict_intent("d-alice-claim-audit-svc", "alice-claim-audit-svc", "1"),
                    base._conflict_intent("d-bob-claim-audit-svc", "bob-claim-audit-svc", "2"),
                    base._conflict_intent("d-carol-claim-audit-svc2", "carol-claim-audit-svc2", "3"),
                    base._conflict_intent("d-dave-claim-audit-svc2", "dave-claim-audit-svc2", "4"),
                ),
            )
        self.assertEqual(prior.to_dict(), before)

    def test_forged_action_payload_rejects_before_resolver_hook(self):
        self.require_conflict()
        import narrative_dynamics.narrative.world as world_module

        self.assertTrue(hasattr(world_module, "_build_conflict_context"))
        original = world_module._build_conflict_context
        domain = base.make_conflict_domain()
        story = base.make_conflict_story(domain)
        hook = base.ClaimResolver()
        resolver = make_resolver(domain, hook, ("claim-service-action",))
        model = base.make_conflict_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)

        def forged_context(*args, **kwargs):
            context = original(*args, **kwargs)
            first = context.participants[0]
            forged_participant = replace(
                first,
                action=replace(
                    first.action,
                    arguments={"service": base._service_ref_for("svc2")},
                ),
            )
            return replace(
                context,
                participants=(forged_participant,) + context.participants[1:],
            )

        with patch.object(
            world_module,
            "_build_conflict_context",
            side_effect=forged_context,
        ):
            with self.assertRaises(WorldTransitionConflictResolutionError):
                advance_world_step(
                    story, domain, prior, model,
                    (
                        base._conflict_intent("d-alice-claim-svc", "alice-claim-svc", "1"),
                        base._conflict_intent("d-bob-claim-svc", "bob-claim-svc", "2"),
                    ),
                )
        self.assertEqual(hook.calls, [])


if __name__ == "__main__":
    unittest.main()

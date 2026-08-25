from __future__ import annotations

from dataclasses import replace
import unittest

from narrative_dynamics.contracts import stable_content_hash
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
    Decision,
    Entity,
    EntityRef,
    GenericNarrative,
    NarrativeEvent,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.replay import objective_state


_WORLD_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.world import (
        ActionEffectSpec,
        ActionIntent,
        ActionTransitionRecord,
        ActionTransitionSpec,
        WorldState,
        WorldStepResult,
        WorldTransitionConflictError,
        WorldTransitionError,
        WorldTransitionModelSpec,
        advance_world_step,
        world_state_from_story,
    )
except ImportError as error:
    _WORLD_IMPORT_ERROR = error


class SeedPhaseHook:
    def __call__(self, prior_state, event):
        actor = event.arguments["agent"].value
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    actor.entity_id,
                    "agent.phase",
                    event.arguments["phase"],
                ),
            )
        )


class SeedHealthHook:
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


class ActorPhaseTransitionHook:
    def __call__(self, prior_state, decision, action):
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    decision.actor_id,
                    "agent.phase",
                    action.arguments["phase"],
                ),
            )
        )


class ActorPhaseTransitionHookV2:
    def __call__(self, prior_state, decision, action):
        phase = action.arguments["phase"]
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    decision.actor_id,
                    "agent.phase",
                    phase,
                ),
            )
        )


class ServiceHealthTransitionHook:
    def __call__(self, prior_state, decision, action):
        service = action.arguments["service"].value
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    service.entity_id,
                    "service.health",
                    action.arguments["health"],
                ),
            )
        )


class NoopTransitionHook:
    def __call__(self, prior_state, decision, action):
        return StateDelta(())


class OutsideCapabilityHook:
    def __call__(self, prior_state, decision, action):
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    "svc",
                    "service.health",
                    TypedValue("HealthState", "recovered"),
                ),
            )
        )


class DuplicateWriteHook:
    def __call__(self, prior_state, decision, action):
        value = action.arguments["phase"]
        return StateDelta(
            (
                StateDeltaOp("set", decision.actor_id, "agent.phase", value),
                StateDeltaOp("set", decision.actor_id, "agent.phase", value),
            )
        )


class BadClearHook:
    def __call__(self, prior_state, decision, action):
        return StateDelta(
            (
                StateDeltaOp(
                    "clear",
                    decision.actor_id,
                    "agent.phase",
                    TypedValue("PhaseState", "ready"),
                ),
            )
        )


class UnknownCellHook:
    def __call__(self, prior_state, decision, action):
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    "missing",
                    "missing.state",
                    TypedValue("PhaseState", "ready"),
                ),
            )
        )


class WrongValueHook:
    def __call__(self, prior_state, decision, action):
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    decision.actor_id,
                    "agent.phase",
                    TypedValue("HealthState", "healthy"),
                ),
            )
        )


class RaisingHook:
    def __call__(self, prior_state, decision, action):
        raise RuntimeError("boom")


class RecordingPhaseHook:
    def __init__(self):
        self.snapshots = []

    def __call__(self, prior_state, decision, action):
        self.snapshots.append(dict(prior_state))
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    decision.actor_id,
                    "agent.phase",
                    action.arguments["phase"],
                ),
            )
        )


def make_world_domain() -> DomainSpec:
    return DomainSpec(
        domain_id="world-test",
        version="1",
        entity_types=(EntityTypeSpec("Agent"), EntityTypeSpec("Service")),
        value_types=(
            ValueTypeSpec("AgentRef", "entity_ref", entity_type="Agent"),
            ValueTypeSpec("ServiceRef", "entity_ref", entity_type="Service"),
            ValueTypeSpec(
                "PhaseState",
                "enum",
                allowed_values=("idle", "ready", "done"),
            ),
            ValueTypeSpec(
                "HealthState",
                "enum",
                allowed_values=("healthy", "failed", "recovered"),
            ),
        ),
        state_variables=(
            StateVariableSpec("agent.phase", "Agent", "PhaseState"),
            StateVariableSpec("service.health", "Service", "HealthState"),
        ),
        event_types=(
            EventTypeSpec(
                "SeedPhase",
                None,
                (
                    ParameterSpec("agent", "AgentRef"),
                    ParameterSpec("phase", "PhaseState"),
                ),
                (StateEffectSpec("agent.phase", "agent"),),
                "seed_phase",
            ),
            EventTypeSpec(
                "SeedHealth",
                None,
                (
                    ParameterSpec("service", "ServiceRef"),
                    ParameterSpec("health", "HealthState"),
                ),
                (StateEffectSpec("service.health", "service"),),
                "seed_health",
            ),
        ),
        action_types=(
            ActionTypeSpec(
                "actor-phase-action",
                (ParameterSpec("phase", "PhaseState"),),
            ),
            ActionTypeSpec(
                "service-health-action",
                (
                    ParameterSpec("service", "ServiceRef"),
                    ParameterSpec("health", "HealthState"),
                ),
            ),
            ActionTypeSpec("noop-action", ()),
        ),
        decision_types=(
            DecisionTypeSpec("phase-choice", "Agent", "actor-phase-action"),
            DecisionTypeSpec("service-choice", "Agent", "service-health-action"),
            DecisionTypeSpec("noop-choice", "Agent", "noop-action"),
        ),
        semantic_hooks=(
            SemanticHookBinding(
                "seed_phase",
                "seed agent phase",
                SeedPhaseHook(),
            ),
            SemanticHookBinding(
                "seed_health",
                "seed service health",
                SeedHealthHook(),
            ),
        ),
    )


def _agent_cell(agent_id: str) -> StateCellRef:
    return StateCellRef(EntityRef(agent_id, "Agent"), "agent.phase")


def _service_cell() -> StateCellRef:
    return StateCellRef(EntityRef("svc", "Service"), "service.health")


def make_world_story() -> GenericNarrative:
    domain = make_world_domain()
    return GenericNarrative(
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (
            Entity("bob", "Agent"),
            Entity("alice", "Agent"),
            Entity("svc", "Service"),
        ),
        (
            NarrativeEvent(
                "e1",
                1,
                "SeedHealth",
                None,
                {
                    "service": TypedValue(
                        "ServiceRef",
                        EntityRef("svc", "Service"),
                    ),
                    "health": TypedValue("HealthState", "failed"),
                },
            ),
            NarrativeEvent(
                "e2",
                2,
                "SeedPhase",
                None,
                {
                    "agent": TypedValue(
                        "AgentRef",
                        EntityRef("bob", "Agent"),
                    ),
                    "phase": TypedValue("PhaseState", "idle"),
                },
            ),
            NarrativeEvent(
                "e3",
                3,
                "SeedPhase",
                None,
                {
                    "agent": TypedValue(
                        "AgentRef",
                        EntityRef("alice", "Agent"),
                    ),
                    "phase": TypedValue("PhaseState", "idle"),
                },
            ),
        ),
        (),
        (),
        (),
        (
            Decision(
                "d-bob-phase",
                4,
                "bob",
                "phase-choice",
                (_agent_cell("bob"),),
                (
                    ActionOption(
                        "bob-ready",
                        "actor-phase-action",
                        {"phase": TypedValue("PhaseState", "ready")},
                    ),
                ),
            ),
            Decision(
                "d-alice-phase",
                5,
                "alice",
                "phase-choice",
                (_agent_cell("alice"),),
                (
                    ActionOption(
                        "alice-done",
                        "actor-phase-action",
                        {"phase": TypedValue("PhaseState", "done")},
                    ),
                ),
            ),
            Decision(
                "d-bob-service",
                6,
                "bob",
                "service-choice",
                (_service_cell(),),
                (
                    ActionOption(
                        "bob-recover",
                        "service-health-action",
                        {
                            "service": TypedValue(
                                "ServiceRef",
                                EntityRef("svc", "Service"),
                            ),
                            "health": TypedValue("HealthState", "recovered"),
                        },
                    ),
                ),
            ),
            Decision(
                "d-alice-service",
                7,
                "alice",
                "service-choice",
                (_service_cell(),),
                (
                    ActionOption(
                        "alice-fail",
                        "service-health-action",
                        {
                            "service": TypedValue(
                                "ServiceRef",
                                EntityRef("svc", "Service"),
                            ),
                            "health": TypedValue("HealthState", "failed"),
                        },
                    ),
                    ActionOption(
                        "alice-recover",
                        "service-health-action",
                        {
                            "service": TypedValue(
                                "ServiceRef",
                                EntityRef("svc", "Service"),
                            ),
                            "health": TypedValue("HealthState", "recovered"),
                        },
                    ),
                ),
            ),
            Decision(
                "d-alice-noop",
                8,
                "alice",
                "noop-choice",
                (_agent_cell("alice"),),
                (
                    ActionOption(
                        "alice-wait",
                        "noop-action",
                        {},
                    ),
                ),
            ),
        ),
    )


def make_transition_model(
    *,
    phase_hook=None,
    include_service: bool = True,
    include_noop: bool = True,
):
    domain = make_world_domain()
    transitions = [
        ActionTransitionSpec(
            "actor-phase-action",
            (ActionEffectSpec("agent.phase", "actor"),),
            ActorPhaseTransitionHook() if phase_hook is None else phase_hook,
        )
    ]
    if include_service:
        transitions.append(
            ActionTransitionSpec(
                "service-health-action",
                (
                    ActionEffectSpec(
                        "service.health",
                        "argument",
                        "service",
                    ),
                ),
                ServiceHealthTransitionHook(),
            )
        )
    if include_noop:
        transitions.append(
            ActionTransitionSpec(
                "noop-action",
                (),
                NoopTransitionHook(),
            )
        )
    return WorldTransitionModelSpec(
        "world-model",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        tuple(transitions),
    )


def intent(
    decision_id: str,
    action_id: str,
    *,
    marker: str = "1",
):
    return ActionIntent(
        decision_id,
        action_id,
        "selection-model",
        "sha256:" + marker * 64,
    )


class NarrativeWorldTransitionTests(unittest.TestCase):
    def require_world(self) -> None:
        if _WORLD_IMPORT_ERROR is not None:
            self.fail(
                f"narrative world transition API is missing: "
                f"{_WORLD_IMPORT_ERROR}"
            )

    def test_world_state_from_story_matches_objective_state_and_lineage(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        state = world_state_from_story(story, domain, at_time=3)
        self.assertEqual(
            dict(state.values),
            dict(objective_state(story, domain, at_time=3)),
        )
        self.assertEqual(state.source_story_hash, story.content_hash)
        self.assertEqual(state.source_at_time, 3)
        self.assertEqual(state.step_index, 0)
        self.assertIsNone(state.parent_state_hash)
        self.assertIsNone(state.transition_batch_hash)

    def test_single_canonical_action_changes_world_and_binds_lineage(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        model = make_transition_model()
        result = advance_world_step(
            story,
            domain,
            prior,
            model,
            (intent("d-bob-phase", "bob-ready"),),
        )
        self.assertIsInstance(result, WorldStepResult)
        self.assertEqual(result.prior_state.content_hash, prior.content_hash)
        self.assertEqual(
            result.next_state.values[_agent_cell("bob")],
            TypedValue("PhaseState", "ready"),
        )
        self.assertEqual(result.next_state.step_index, 1)
        self.assertEqual(
            result.next_state.parent_state_hash,
            prior.content_hash,
        )
        self.assertIsNotNone(result.next_state.transition_batch_hash)
        self.assertEqual(result.transitions[0].actor_id, "bob")
        self.assertEqual(result.transitions[0].action.id, "bob-ready")

    def test_forged_prior_state_rejects_before_hook_execution(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        recording = RecordingPhaseHook()
        model = make_transition_model(phase_hook=recording)
        good = world_state_from_story(story, domain)
        forged_values = dict(good.values)
        forged_values[
            StateCellRef(
                EntityRef("svc", "Service"),
                "agent.phase",
            )
        ] = TypedValue("PhaseState", "ready")
        forged = replace(good, values=forged_values)
        with self.assertRaises(WorldTransitionError):
            advance_world_step(
                story,
                domain,
                forged,
                model,
                (intent("d-bob-phase", "bob-ready"),),
            )
        self.assertEqual(recording.snapshots, [])

    def test_all_hooks_read_the_same_prior_snapshot(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        recording = RecordingPhaseHook()
        model = make_transition_model(phase_hook=recording)
        prior = world_state_from_story(story, domain)
        advance_world_step(
            story,
            domain,
            prior,
            model,
            (
                intent("d-bob-phase", "bob-ready"),
                intent(
                    "d-alice-phase",
                    "alice-done",
                    marker="2",
                ),
            ),
        )
        expected = dict(prior.values)
        self.assertEqual(recording.snapshots, [expected, expected])

    def test_disjoint_batch_is_order_invariant(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        model = make_transition_model()
        a = intent("d-bob-phase", "bob-ready")
        b = intent("d-alice-phase", "alice-done", marker="2")
        first = advance_world_step(story, domain, prior, model, (a, b))
        second = advance_world_step(story, domain, prior, model, (b, a))
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(
            first.next_state.content_hash,
            second.next_state.content_hash,
        )

    def test_different_value_overlap_is_typed_conflict(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        model = make_transition_model()
        with self.assertRaises(WorldTransitionConflictError):
            advance_world_step(
                story,
                domain,
                prior,
                model,
                (
                    intent("d-bob-service", "bob-recover"),
                    intent(
                        "d-alice-service",
                        "alice-fail",
                        marker="2",
                    ),
                ),
            )

    def test_same_value_overlap_is_also_typed_conflict(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        model = make_transition_model()
        with self.assertRaises(WorldTransitionConflictError):
            advance_world_step(
                story,
                domain,
                prior,
                model,
                (
                    intent("d-bob-service", "bob-recover"),
                    intent(
                        "d-alice-service",
                        "alice-recover",
                        marker="2",
                    ),
                ),
            )

    def test_invalid_delta_is_atomic_and_prior_state_is_unchanged(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        before = prior.to_dict()
        bad_model = WorldTransitionModelSpec(
            "bad-world",
            "1",
            domain.domain_id,
            domain.version,
            domain.content_hash,
            (
                ActionTransitionSpec(
                    "actor-phase-action",
                    (ActionEffectSpec("agent.phase", "actor"),),
                    OutsideCapabilityHook(),
                ),
            ),
        )
        with self.assertRaises(WorldTransitionError):
            advance_world_step(
                story,
                domain,
                prior,
                bad_model,
                (intent("d-bob-phase", "bob-ready"),),
            )
        self.assertEqual(prior.to_dict(), before)

    def test_duplicate_actor_is_rejected(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        model = make_transition_model()
        with self.assertRaises(WorldTransitionError):
            advance_world_step(
                story,
                domain,
                prior,
                model,
                (
                    intent("d-bob-phase", "bob-ready"),
                    intent(
                        "d-bob-service",
                        "bob-recover",
                        marker="2",
                    ),
                ),
            )

    def test_missing_decision_action_and_uncovered_action_type_reject(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        model = make_transition_model(include_noop=False)
        for bad in (
            intent("missing", "bob-ready"),
            intent("d-bob-phase", "missing"),
            intent("d-alice-noop", "alice-wait"),
        ):
            with self.subTest(bad=bad.to_dict()):
                with self.assertRaises(WorldTransitionError):
                    advance_world_step(
                        story,
                        domain,
                        prior,
                        model,
                        (bad,),
                    )

    def test_actor_and_argument_effect_capabilities_are_typed(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        wrong_actor_target = WorldTransitionModelSpec(
            "wrong-actor",
            "1",
            domain.domain_id,
            domain.version,
            domain.content_hash,
            (
                ActionTransitionSpec(
                    "actor-phase-action",
                    (
                        ActionEffectSpec(
                            "service.health",
                            "actor",
                        ),
                    ),
                    ActorPhaseTransitionHook(),
                ),
            ),
        )
        wrong_argument_target = WorldTransitionModelSpec(
            "wrong-argument",
            "1",
            domain.domain_id,
            domain.version,
            domain.content_hash,
            (
                ActionTransitionSpec(
                    "service-health-action",
                    (
                        ActionEffectSpec(
                            "service.health",
                            "argument",
                            "health",
                        ),
                    ),
                    ServiceHealthTransitionHook(),
                ),
            ),
        )
        with self.assertRaises(WorldTransitionError):
            advance_world_step(
                story,
                domain,
                prior,
                wrong_actor_target,
                (intent("d-bob-phase", "bob-ready"),),
            )
        with self.assertRaises(WorldTransitionError):
            advance_world_step(
                story,
                domain,
                prior,
                wrong_argument_target,
                (intent("d-bob-service", "bob-recover"),),
            )

    def test_outside_capability_duplicate_write_and_invalid_delta_shapes_reject(
        self,
    ):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        cases = (
            OutsideCapabilityHook(),
            DuplicateWriteHook(),
            BadClearHook(),
            UnknownCellHook(),
            WrongValueHook(),
            RaisingHook(),
        )
        for hook in cases:
            model = WorldTransitionModelSpec(
                "bad",
                "1",
                domain.domain_id,
                domain.version,
                domain.content_hash,
                (
                    ActionTransitionSpec(
                        "actor-phase-action",
                        (
                            ActionEffectSpec(
                                "agent.phase",
                                "actor",
                            ),
                        ),
                        hook,
                    ),
                ),
            )
            with self.subTest(hook=type(hook).__name__):
                with self.assertRaises(WorldTransitionError):
                    advance_world_step(
                        story,
                        domain,
                        prior,
                        model,
                        (intent("d-bob-phase", "bob-ready"),),
                    )

    def test_explicit_noop_succeeds_and_preserves_extensional_values(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        model = make_transition_model()
        result = advance_world_step(
            story,
            domain,
            prior,
            model,
            (intent("d-alice-noop", "alice-wait"),),
        )
        self.assertEqual(dict(result.next_state.values), dict(prior.values))
        self.assertEqual(
            result.next_state.step_index,
            prior.step_index + 1,
        )
        self.assertEqual(
            result.next_state.parent_state_hash,
            prior.content_hash,
        )
        self.assertNotEqual(
            result.next_state.content_hash,
            prior.content_hash,
        )

    def test_numeric_source_cutoff_rejects_later_decision(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain, at_time=5)
        with self.assertRaises(WorldTransitionError):
            advance_world_step(
                story,
                domain,
                prior,
                make_transition_model(),
                (intent("d-bob-service", "bob-recover"),),
            )

    def test_model_identity_binds_domain_effects_and_hook_implementation_not_order(
        self,
    ):
        self.require_world()
        domain = make_world_domain()
        first_phase = ActionTransitionSpec(
            "actor-phase-action",
            (ActionEffectSpec("agent.phase", "actor"),),
            ActorPhaseTransitionHook(),
        )
        second_phase = ActionTransitionSpec(
            "actor-phase-action",
            (ActionEffectSpec("agent.phase", "actor"),),
            ActorPhaseTransitionHookV2(),
        )
        service = ActionTransitionSpec(
            "service-health-action",
            (
                ActionEffectSpec(
                    "service.health",
                    "argument",
                    "service",
                ),
            ),
            ServiceHealthTransitionHook(),
        )
        base = WorldTransitionModelSpec(
            "m",
            "1",
            domain.domain_id,
            domain.version,
            domain.content_hash,
            (first_phase, service),
        )
        reversed_model = WorldTransitionModelSpec(
            "m",
            "1",
            domain.domain_id,
            domain.version,
            domain.content_hash,
            (service, first_phase),
        )
        changed_domain = replace(
            base,
            domain_spec_hash="sha256:" + "f" * 64,
        )
        changed_effect = WorldTransitionModelSpec(
            "m",
            "1",
            domain.domain_id,
            domain.version,
            domain.content_hash,
            (
                ActionTransitionSpec(
                    "actor-phase-action",
                    (),
                    ActorPhaseTransitionHook(),
                ),
                service,
            ),
        )
        changed_hook = WorldTransitionModelSpec(
            "m",
            "1",
            domain.domain_id,
            domain.version,
            domain.content_hash,
            (second_phase, service),
        )
        self.assertEqual(base.content_hash, reversed_model.content_hash)
        self.assertNotEqual(
            base.content_hash,
            changed_domain.content_hash,
        )
        self.assertNotEqual(
            base.content_hash,
            changed_effect.content_hash,
        )
        self.assertNotEqual(
            base.content_hash,
            changed_hook.content_hash,
        )

    def test_transition_and_result_hashes_bind_prior_intent_delta_and_model(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        model = make_transition_model()
        first = advance_world_step(
            story,
            domain,
            prior,
            model,
            (
                intent(
                    "d-bob-phase",
                    "bob-ready",
                    marker="1",
                ),
            ),
        )
        second = advance_world_step(
            story,
            domain,
            prior,
            model,
            (
                intent(
                    "d-bob-phase",
                    "bob-ready",
                    marker="2",
                ),
            ),
        )
        changed_model = replace(model, version="2")
        third = advance_world_step(
            story,
            domain,
            prior,
            changed_model,
            (
                intent(
                    "d-bob-phase",
                    "bob-ready",
                    marker="1",
                ),
            ),
        )
        self.assertNotEqual(
            first.transitions[0].content_hash,
            second.transitions[0].content_hash,
        )
        self.assertNotEqual(
            first.next_state.transition_batch_hash,
            second.next_state.transition_batch_hash,
        )
        self.assertNotEqual(first.content_hash, second.content_hash)
        self.assertNotEqual(first.model_hash, third.model_hash)
        self.assertNotEqual(first.content_hash, third.content_hash)

    def test_record_constructors_fail_closed(self):
        self.require_world()
        domain = make_world_domain()

        with self.assertRaises((TypeError, ValueError)):
            ActionEffectSpec("", "actor")
        with self.assertRaises(ValueError):
            ActionEffectSpec("agent.phase", "other")
        with self.assertRaises(ValueError):
            ActionEffectSpec("agent.phase", "actor", "target")
        with self.assertRaises((TypeError, ValueError)):
            ActionEffectSpec("service.health", "argument", None)

        with self.assertRaises(TypeError):
            ActionTransitionSpec(
                "actor-phase-action",
                (),
                object(),
            )
        phase = ActionTransitionSpec(
            "actor-phase-action",
            (ActionEffectSpec("agent.phase", "actor"),),
            ActorPhaseTransitionHook(),
        )
        with self.assertRaises(ValueError):
            WorldTransitionModelSpec(
                "m",
                "1",
                domain.domain_id,
                domain.version,
                domain.content_hash,
                (phase, phase),
            )

        for args in (
            ("", "bob-ready", "selection-model", "sha256:" + "1" * 64),
            ("d-bob-phase", "", "selection-model", "sha256:" + "1" * 64),
            ("d-bob-phase", "bob-ready", "", "sha256:" + "1" * 64),
            ("d-bob-phase", "bob-ready", "selection-model", "bad"),
        ):
            with self.subTest(intent_args=args):
                with self.assertRaises((TypeError, ValueError)):
                    ActionIntent(*args)

        valid_values = {
            _agent_cell("bob"): TypedValue("PhaseState", "idle"),
        }
        common = dict(
            domain_id=domain.domain_id,
            domain_version=domain.version,
            domain_spec_hash=domain.content_hash,
            source_story_hash="sha256:" + "a" * 64,
            source_at_time=None,
            step_index=0,
            parent_state_hash=None,
            transition_batch_hash=None,
            values=valid_values,
        )
        with self.assertRaises(ValueError):
            WorldState(**{**common, "step_index": -1})
        with self.assertRaises(TypeError):
            WorldState(
                **{
                    **common,
                    "values": {
                        "not-a-cell": TypedValue("PhaseState", "idle")
                    },
                }
            )
        with self.assertRaises(TypeError):
            WorldState(
                **{
                    **common,
                    "values": {
                        _agent_cell("bob"): "idle",
                    },
                }
            )
        with self.assertRaises(ValueError):
            WorldState(
                **{
                    **common,
                    "step_index": 1,
                    "parent_state_hash": None,
                    "transition_batch_hash": "sha256:" + "b" * 64,
                }
            )

        prior = WorldState(**common)
        action = ActionOption(
            "bob-ready",
            "actor-phase-action",
            {"phase": TypedValue("PhaseState", "ready")},
        )
        intent_value = ActionIntent(
            "d-bob-phase",
            "bob-ready",
            "selection-model",
            "sha256:" + "1" * 64,
        )
        with self.assertRaises(ValueError):
            ActionTransitionRecord(
                intent_value,
                "bob",
                action,
                "bad",
                prior.content_hash,
                StateDelta(()),
            )

        next_state = WorldState(
            domain_id=prior.domain_id,
            domain_version=prior.domain_version,
            domain_spec_hash=prior.domain_spec_hash,
            source_story_hash=prior.source_story_hash,
            source_at_time=prior.source_at_time,
            step_index=1,
            parent_state_hash="sha256:" + "c" * 64,
            transition_batch_hash="sha256:" + "d" * 64,
            values=prior.values,
        )
        record = ActionTransitionRecord(
            intent_value,
            "bob",
            action,
            phase.content_hash,
            prior.content_hash,
            StateDelta(()),
        )
        with self.assertRaises(ValueError):
            WorldStepResult(
                "world-model",
                "sha256:" + "e" * 64,
                prior,
                (record,),
                next_state,
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from dataclasses import fields, replace
import unittest
from unittest.mock import patch

from narrative_dynamics.attestation import ImplementationAttestationUnavailable
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import (
    ActionTypeSpec,
    DecisionTypeSpec,
    ParameterSpec,
    StateDelta,
    StateDeltaOp,
    StateVariableSpec,
    ValueTypeSpec,
)
from narrative_dynamics.narrative.intention import ChoiceModelSpec, GoalModelSpec, GoalSpec
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Decision,
    Entity,
    EntityRef,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.runtime_decision_dispatch import RuntimeDecisionModelSpec
from narrative_dynamics.narrative.runtime_intention import RuntimeIntentionalDecisionModelSpec
from narrative_dynamics.narrative.runtime_planning import (
    PlanningHiddenState,
    PlanningObservation,
    RuntimePlanningDecisionModelSpec,
)
from narrative_dynamics.narrative.runtime_reactive import RuntimeReactiveDecisionModelSpec
from narrative_dynamics.narrative.simulation import (
    RuntimeAgentSpec,
    SimulationModelSpec,
    simulate_step,
    simulate_trajectory,
    simulation_state_from_story,
)
from narrative_dynamics.narrative.world import (
    ActionEffectSpec,
    ActionIntent,
    ActionTransitionSpec,
    WorldStepResult,
    WorldTransitionConflictError,
    WorldTransitionError,
    WorldTransitionModelSpec,
    advance_world_step,
    world_state_from_story,
)
from tests.test_narrative_runtime_planning import (
    IdentityTransitionHook,
    NoInformationObservationHook,
    ProductCouplingHook,
)
from tests.test_narrative_simulation import (
    AlertControlTransition,
    make_a_choice_model,
    make_a_goal_model,
    make_intentional_models,
    make_projection_model,
    make_runtime_belief_model,
    make_scheduler_domain,
    make_scheduler_story,
    alert_cell,
    _service_ref,
)
from tests.test_narrative_world_transition import (
    ActorPhaseTransitionHook,
    NoopTransitionHook,
    ServiceHealthTransitionHook,
    intent,
    make_transition_model,
    make_world_domain as make_v1_world_domain,
    make_world_story as make_v1_world_story,
)


_CONFLICT_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.conflict import (
        ConflictParticipant,
        ConflictResolutionContext,
        ConflictResolutionRecord,
        ConflictResolverSpec,
    )
    from narrative_dynamics.narrative.world import (
        WorldTransitionConflictResolutionError,
    )
except ImportError as error:
    _CONFLICT_IMPORT_ERROR = error


def _hash(label: str) -> str:
    return stable_content_hash({"conflict-test": label})


def _forge(instance, **changes):
    forged = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            forged,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return forged


def _agent_ref(agent_id: str) -> TypedValue:
    return TypedValue("AgentRef", EntityRef(agent_id, "Agent"))


def _service_ref_for(service_id: str) -> TypedValue:
    return TypedValue("ServiceRef", EntityRef(service_id, "Service"))


def _cell(entity_id: str, entity_type: str, variable: str) -> StateCellRef:
    return StateCellRef(EntityRef(entity_id, entity_type), variable)


class PreferLexicalActorResolver:
    def __call__(self, snapshot, context):
        return min(context.participants, key=lambda item: item.actor_id).original_delta


class PreferLexicalActorResolverV2:
    def __call__(self, snapshot, context):
        ordered = tuple(sorted(context.participants, key=lambda item: item.actor_id))
        return ordered[0].original_delta


class RecordingNoopConflictResolver:
    def __init__(self):
        self.calls = []

    def __call__(self, snapshot, context):
        self.calls.append((dict(snapshot), context))
        return StateDelta(())


class ClaimTransition:
    def __call__(self, snapshot, decision, action):
        service = action.arguments["service"].value.entity_id
        return StateDelta((
            StateDeltaOp(
                "set",
                service,
                "service.owner",
                _agent_ref(decision.actor_id),
            ),
        ))


class BidTransition(ClaimTransition):
    pass


class AttackTransition:
    def __call__(self, snapshot, decision, action):
        target = action.arguments["target"].value.entity_id
        return StateDelta((
            StateDeltaOp(
                "set",
                target,
                "agent.health",
                TypedValue("CombatHealth", "injured"),
            ),
            StateDeltaOp(
                "set",
                decision.actor_id,
                "agent.combat",
                TypedValue("CombatStatus", "attack-success"),
            ),
        ))


class DefendTransition:
    def __call__(self, snapshot, decision, action):
        return StateDelta((
            StateDeltaOp(
                "set",
                decision.actor_id,
                "agent.health",
                TypedValue("CombatHealth", "healthy"),
            ),
            StateDeltaOp(
                "set",
                decision.actor_id,
                "agent.combat",
                TypedValue("CombatStatus", "defense-success"),
            ),
        ))


class TauntTransition:
    def __call__(self, snapshot, decision, action):
        target = action.arguments["target"].value.entity_id
        return StateDelta((
            StateDeltaOp(
                "set",
                target,
                "agent.combat",
                TypedValue("CombatStatus", "attack-failed"),
            ),
        ))


class ClaimResolver:
    def __init__(self):
        self.calls = []

    def __call__(self, snapshot, context):
        self.calls.append((dict(snapshot), context))
        winner = min(context.participants, key=lambda item: item.actor_id)
        service = winner.action.arguments["service"].value.entity_id
        return StateDelta((
            StateDeltaOp(
                "set",
                service,
                "service.owner",
                _agent_ref(winner.actor_id),
            ),
        ))


class AuctionResolver:
    def __call__(self, snapshot, context):
        winner = min(
            context.participants,
            key=lambda item: (-int(item.action.arguments["bid"].value), item.actor_id),
        )
        service = winner.action.arguments["service"].value.entity_id
        return StateDelta((
            StateDeltaOp(
                "set",
                service,
                "service.owner",
                _agent_ref(winner.actor_id),
            ),
        ))


class AttackDefenseResolver:
    def __call__(self, snapshot, context):
        attack = next(
            item for item in context.participants
            if item.action.type_name == "attack-action"
        )
        defense = next(
            item for item in context.participants
            if item.action.type_name == "defend-action"
        )
        target = attack.action.arguments["target"].value.entity_id
        if defense.actor_id != target:
            return attack.original_delta
        return StateDelta((
            StateDeltaOp(
                "set",
                target,
                "agent.health",
                TypedValue("CombatHealth", "healthy"),
            ),
            StateDeltaOp(
                "set",
                attack.actor_id,
                "agent.combat",
                TypedValue("CombatStatus", "attack-failed"),
            ),
            StateDeltaOp(
                "set",
                defense.actor_id,
                "agent.combat",
                TypedValue("CombatStatus", "defense-success"),
            ),
        ))


class RaisingResolver:
    def __init__(self):
        self.calls = 0

    def __call__(self, snapshot, context):
        self.calls += 1
        raise RuntimeError("resolver boom")


class ConflictAlertReactiveScoreHook:
    def __call__(self, context):
        return {"a2-raise-alert": 3.0, "a2-wait-alert": 0.0}


class ConflictAlertPlanningRewardHook:
    def __call__(self, context):
        if context.action.id == "a3-raise-alert":
            return 3.0
        if context.action.id == "a3-wait-alert":
            return 0.0
        raise AssertionError("unexpected conflict planning action")


class ConflictAlertResolver:
    def __init__(self):
        self.calls = []

    def __call__(self, snapshot, context):
        self.calls.append((dict(snapshot), context))
        return StateDelta((
            StateDeltaOp(
                "set",
                "svc",
                "service.alert",
                TypedValue("AlertState", True),
            ),
        ))


class ConflictAlertResolverV2:
    def __init__(self):
        self.calls = []

    def __call__(self, snapshot, context):
        self.calls.append((dict(snapshot), context))
        operations = (
            StateDeltaOp(
                "set",
                "svc",
                "service.alert",
                TypedValue("AlertState", True),
            ),
        )
        return StateDelta(tuple(operations))


def make_direct_participant_context_fixture():
    cell = _cell("svc", "Service", "service.health")
    action = ActionOption(
        "alice-recover",
        "service-health-action",
        {
            "service": _service_ref_for("svc"),
            "health": TypedValue("HealthState", "recovered"),
        },
    )
    delta = StateDelta((
        StateDeltaOp(
            "set",
            "svc",
            "service.health",
            TypedValue("HealthState", "recovered"),
        ),
    ))
    alice = ConflictParticipant(
        "alice",
        "d-alice-service",
        action,
        _hash("alice-transition"),
        _hash("service-transition-spec"),
        delta,
        (cell,),
    )
    bob = ConflictParticipant(
        "bob",
        "d-bob-service",
        replace(action, id="bob-recover"),
        _hash("bob-transition"),
        _hash("service-transition-spec"),
        delta,
        (cell,),
    )
    context = ConflictResolutionContext(_hash("prior"), (bob, alice), (cell,))
    return alice, bob, context


def make_conflict_domain():
    base = make_v1_world_domain()
    return replace(
        base,
        value_types=base.value_types + (
            ValueTypeSpec("BidLevel", "enum", allowed_values=("8", "11")),
            ValueTypeSpec(
                "CombatHealth",
                "enum",
                allowed_values=("healthy", "injured"),
            ),
            ValueTypeSpec(
                "CombatStatus",
                "enum",
                allowed_values=(
                    "idle",
                    "attack-success",
                    "attack-failed",
                    "defense-success",
                ),
            ),
        ),
        state_variables=base.state_variables + (
            StateVariableSpec("service.owner", "Service", "AgentRef"),
            StateVariableSpec("agent.health", "Agent", "CombatHealth"),
            StateVariableSpec("agent.combat", "Agent", "CombatStatus"),
        ),
        action_types=base.action_types + (
            ActionTypeSpec(
                "claim-service-action",
                (ParameterSpec("service", "ServiceRef"),),
            ),
            ActionTypeSpec(
                "bid-service-action",
                (
                    ParameterSpec("service", "ServiceRef"),
                    ParameterSpec("bid", "BidLevel"),
                ),
            ),
            ActionTypeSpec(
                "attack-action",
                (ParameterSpec("target", "AgentRef"),),
            ),
            ActionTypeSpec("defend-action", ()),
            ActionTypeSpec(
                "taunt-action",
                (ParameterSpec("target", "AgentRef"),),
            ),
        ),
        decision_types=base.decision_types + (
            DecisionTypeSpec("claim-choice", "Agent", "claim-service-action"),
            DecisionTypeSpec("bid-choice", "Agent", "bid-service-action"),
            DecisionTypeSpec("attack-choice", "Agent", "attack-action"),
            DecisionTypeSpec("defend-choice", "Agent", "defend-action"),
            DecisionTypeSpec("taunt-choice", "Agent", "taunt-action"),
        ),
    )


def make_conflict_story(domain):
    base = make_v1_world_story()
    decisions = list(base.decisions)
    next_time = max(item.logical_time for item in decisions) + 1

    def add(decision):
        nonlocal next_time
        if not decision.context_cells:
            decision = replace(
                decision,
                context_cells=(_cell("svc", "Service", "service.health"),),
            )
        decisions.append(replace(decision, logical_time=next_time))
        next_time += 1

    add(Decision(
        "d-alice-claim-svc", 0, "alice", "claim-choice", (),
        (ActionOption(
            "alice-claim-svc", "claim-service-action",
            {"service": _service_ref_for("svc")},
        ),),
    ))
    add(Decision(
        "d-bob-claim-svc", 0, "bob", "claim-choice", (),
        (ActionOption(
            "bob-claim-svc", "claim-service-action",
            {"service": _service_ref_for("svc")},
        ),),
    ))
    add(Decision(
        "d-carol-claim-svc2", 0, "carol", "claim-choice", (),
        (ActionOption(
            "carol-claim-svc2", "claim-service-action",
            {"service": _service_ref_for("svc2")},
        ),),
    ))
    add(Decision(
        "d-dave-claim-svc2", 0, "dave", "claim-choice", (),
        (ActionOption(
            "dave-claim-svc2", "claim-service-action",
            {"service": _service_ref_for("svc2")},
        ),),
    ))
    add(Decision(
        "d-alice-bid", 0, "alice", "bid-choice", (),
        (ActionOption(
            "alice-bid-11", "bid-service-action",
            {
                "service": _service_ref_for("svc"),
                "bid": TypedValue("BidLevel", "11"),
            },
        ),),
    ))
    add(Decision(
        "d-bob-bid", 0, "bob", "bid-choice", (),
        (ActionOption(
            "bob-bid-11", "bid-service-action",
            {
                "service": _service_ref_for("svc"),
                "bid": TypedValue("BidLevel", "11"),
            },
        ),),
    ))
    add(Decision(
        "d-carol-bid", 0, "carol", "bid-choice", (),
        (ActionOption(
            "carol-bid-8", "bid-service-action",
            {
                "service": _service_ref_for("svc"),
                "bid": TypedValue("BidLevel", "8"),
            },
        ),),
    ))
    add(Decision(
        "d-alice-attack-bob", 0, "alice", "attack-choice", (),
        (ActionOption(
            "alice-attack-bob", "attack-action",
            {"target": _agent_ref("bob")},
        ),),
    ))
    add(Decision(
        "d-bob-defend", 0, "bob", "defend-choice", (),
        (ActionOption("bob-defend", "defend-action", {}),),
    ))
    add(Decision(
        "d-carol-taunt-bob", 0, "carol", "taunt-choice", (),
        (ActionOption(
            "carol-taunt-bob", "taunt-action",
            {"target": _agent_ref("bob")},
        ),),
    ))
    add(Decision(
        "d-dave-noop", 0, "dave", "noop-choice", (),
        (ActionOption("dave-wait", "noop-action", {}),),
    ))
    return replace(
        base,
        domain_spec_hash=domain.content_hash,
        entities=base.entities + (
            Entity("carol", "Agent"),
            Entity("dave", "Agent"),
            Entity("svc2", "Service"),
            Entity("svc3", "Service"),
        ),
        decisions=tuple(decisions),
    )


def make_conflict_world_model(domain, resolver):
    return WorldTransitionModelSpec(
        "conflict-world",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (
            ActionTransitionSpec(
                "actor-phase-action",
                (ActionEffectSpec("agent.phase", "actor"),),
                ActorPhaseTransitionHook(),
            ),
            ActionTransitionSpec(
                "service-health-action",
                (ActionEffectSpec("service.health", "argument", "service"),),
                ServiceHealthTransitionHook(),
            ),
            ActionTransitionSpec("noop-action", (), NoopTransitionHook()),
            ActionTransitionSpec(
                "claim-service-action",
                (ActionEffectSpec("service.owner", "argument", "service"),),
                ClaimTransition(),
            ),
            ActionTransitionSpec(
                "bid-service-action",
                (ActionEffectSpec("service.owner", "argument", "service"),),
                BidTransition(),
            ),
            ActionTransitionSpec(
                "attack-action",
                (
                    ActionEffectSpec("agent.health", "argument", "target"),
                    ActionEffectSpec("agent.combat", "actor"),
                ),
                AttackTransition(),
            ),
            ActionTransitionSpec(
                "defend-action",
                (
                    ActionEffectSpec("agent.health", "actor"),
                    ActionEffectSpec("agent.combat", "actor"),
                ),
                DefendTransition(),
            ),
            ActionTransitionSpec(
                "taunt-action",
                (ActionEffectSpec("agent.combat", "argument", "target"),),
                TauntTransition(),
            ),
        ),
        resolver,
    )


def make_resolver(domain, hook, action_types):
    return ConflictResolverSpec(
        "conflict-resolver",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        tuple(action_types),
        hook,
    )


def _conflict_intent(decision_id: str, action_id: str, marker: str) -> ActionIntent:
    return ActionIntent(
        decision_id,
        action_id,
        "selection-model",
        "sha256:" + marker * 64,
    )


def make_heterogeneous_conflict_case(*, resolver_hook=None):
    domain = make_scheduler_domain()
    story = make_scheduler_story(domain)
    a2_decision = Decision(
        "d-a2-scheduler",
        9,
        "a2",
        "scheduler-alert-choice",
        (alert_cell(),),
        (
            ActionOption(
                "a2-raise-alert",
                "alert-control-action",
                {"service": _service_ref(), "raise": TypedValue("AlertState", True)},
            ),
            ActionOption(
                "a2-wait-alert",
                "alert-control-action",
                {"service": _service_ref(), "raise": TypedValue("AlertState", False)},
            ),
        ),
    )
    a3_decision = Decision(
        "d-a3-scheduler",
        10,
        "a3",
        "scheduler-alert-choice",
        (alert_cell(),),
        (
            ActionOption(
                "a3-raise-alert",
                "alert-control-action",
                {"service": _service_ref(), "raise": TypedValue("AlertState", True)},
            ),
            ActionOption(
                "a3-wait-alert",
                "alert-control-action",
                {"service": _service_ref(), "raise": TypedValue("AlertState", False)},
            ),
        ),
    )
    story = replace(
        story,
        decisions=(story.decisions[0], a2_decision, a3_decision),
    )
    a_intentional, _ = make_intentional_models()
    a2_reactive = RuntimeReactiveDecisionModelSpec(
        "a2-reactive-conflict",
        "1",
        ("scheduler-alert-choice",),
        (alert_cell(),),
        {},
        4.0,
        ConflictAlertReactiveScoreHook(),
    )
    a3_planning = RuntimePlanningDecisionModelSpec(
        "a3-planning-conflict",
        "1",
        ("scheduler-alert-choice",),
        (alert_cell(),),
        (),
        make_runtime_belief_model(),
        (
            PlanningHiddenState(
                "alert-off",
                {alert_cell(): TypedValue("AlertState", False)},
            ),
            PlanningHiddenState(
                "alert-on",
                {alert_cell(): TypedValue("AlertState", True)},
            ),
        ),
        (PlanningObservation("none", {}),),
        (("a3-raise-alert", "a3-wait-alert"),),
        1.0,
        4.0,
        {},
        ProductCouplingHook(),
        IdentityTransitionHook(),
        NoInformationObservationHook(),
        ConflictAlertPlanningRewardHook(),
    )
    if resolver_hook is None:
        resolver_hook = ConflictAlertResolver()
    resolver = ConflictResolverSpec(
        "alert-conflict-resolver",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        ("alert-control-action",),
        resolver_hook,
    )
    world_model = WorldTransitionModelSpec(
        "scheduler-world-conflict-v2",
        "2",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (
            ActionTransitionSpec(
                "alert-control-action",
                (ActionEffectSpec("service.alert", "argument", "service"),),
                AlertControlTransition(),
            ),
        ),
        resolver,
    )
    model = SimulationModelSpec(
        "heterogeneous-conflict-model",
        "3",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (
            RuntimeAgentSpec(
                "a1",
                "d-a1-scheduler",
                RuntimeDecisionModelSpec("intentional", a_intentional),
            ),
            RuntimeAgentSpec(
                "a2",
                "d-a2-scheduler",
                RuntimeDecisionModelSpec("reactive", a2_reactive),
            ),
            RuntimeAgentSpec(
                "a3",
                "d-a3-scheduler",
                RuntimeDecisionModelSpec("planning", a3_planning),
            ),
        ),
        world_model,
        make_projection_model(domain),
    )
    return domain, story, model


class NarrativeConflictResolutionTests(unittest.TestCase):
    def require_conflict(self) -> None:
        if _CONFLICT_IMPORT_ERROR is not None:
            self.fail(
                "narrative conflict resolution boundary is missing: "
                f"{_CONFLICT_IMPORT_ERROR}"
            )

    def test_resolver_free_world_model_keeps_exact_v1_payload(self):
        model = make_transition_model()
        expected = {
            "model_id": model.model_id,
            "version": model.version,
            "domain_id": model.domain_id,
            "domain_version": model.domain_version,
            "domain_spec_hash": model.domain_spec_hash,
            "transitions": [item.to_dict() for item in model.transitions],
        }
        self.assertEqual(model.to_dict(), expected)
        self.assertEqual(model.content_hash, stable_content_hash(expected))
        self.assertNotIn("conflict_resolver_hash", model.to_dict())

    def test_resolver_free_world_step_keeps_exact_v1_payload_and_batch_hash(self):
        story, domain = make_v1_world_story(), make_v1_world_domain()
        prior = world_state_from_story(story, domain)
        result = advance_world_step(
            story,
            domain,
            prior,
            make_transition_model(),
            (intent("d-bob-phase", "bob-ready"),),
        )
        expected_batch = stable_content_hash(
            [item.to_dict() for item in result.transitions]
        )
        expected_result = {
            "model_id": result.model_id,
            "model_hash": result.model_hash,
            "prior_state": result.prior_state.to_dict(),
            "transitions": [item.to_dict() for item in result.transitions],
            "next_state": result.next_state.to_dict(),
        }
        self.assertEqual(result.next_state.transition_batch_hash, expected_batch)
        self.assertEqual(result.to_dict(), expected_result)
        self.assertEqual(result.content_hash, stable_content_hash(expected_result))

    def test_existing_no_resolver_conflicts_remain_typed(self):
        story, domain = make_v1_world_story(), make_v1_world_domain()
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
                    intent("d-alice-service", "alice-fail", marker="2"),
                ),
            )
        with self.assertRaises(WorldTransitionConflictError):
            advance_world_step(
                story,
                domain,
                prior,
                model,
                (
                    intent("d-bob-service", "bob-recover"),
                    intent("d-alice-service", "alice-recover", marker="2"),
                ),
            )

    def test_records_bind_full_action_payload_and_exact_conflict_cells(self):
        self.require_conflict()
        alice, bob, context = make_direct_participant_context_fixture()
        self.assertEqual(
            tuple(item.actor_id for item in context.participants),
            ("alice", "bob"),
        )
        self.assertEqual(
            context.participants[0].action.arguments["service"],
            _service_ref_for("svc"),
        )
        self.assertEqual(
            context.conflict_cells,
            (_cell("svc", "Service", "service.health"),),
        )
        with self.assertRaises(ValueError):
            ConflictResolutionContext(context.prior_state_hash, (alice, bob), ())

    def test_action_argument_change_changes_participant_and_context_identity(self):
        self.require_conflict()
        alice, bob, context = make_direct_participant_context_fixture()
        changed = replace(
            alice,
            action=replace(
                alice.action,
                arguments={
                    "service": _service_ref_for("svc"),
                    "health": TypedValue("HealthState", "failed"),
                },
            ),
        )
        changed_context = ConflictResolutionContext(
            context.prior_state_hash,
            (changed, bob),
            context.conflict_cells,
        )
        self.assertNotEqual(alice.content_hash, changed.content_hash)
        self.assertNotEqual(context.content_hash, changed_context.content_hash)

    def test_resolution_record_rejects_prior_context_mismatch(self):
        self.require_conflict()
        alice, _, context = make_direct_participant_context_fixture()
        with self.assertRaises(ValueError):
            ConflictResolutionRecord(
                "resolver",
                _hash("resolver"),
                _hash("different-prior"),
                context,
                alice.original_delta,
            )

    def test_resolver_identity_binds_hook_and_supported_action_types(self):
        self.require_conflict()
        domain = make_v1_world_domain()
        first = ConflictResolverSpec(
            "resolver", "1", domain.domain_id, domain.version,
            domain.content_hash, ("service-health-action",),
            PreferLexicalActorResolver(),
        )
        same = ConflictResolverSpec(
            "resolver", "1", domain.domain_id, domain.version,
            domain.content_hash, ("service-health-action",),
            PreferLexicalActorResolver(),
        )
        changed_hook = ConflictResolverSpec(
            "resolver", "1", domain.domain_id, domain.version,
            domain.content_hash, ("service-health-action",),
            PreferLexicalActorResolverV2(),
        )
        changed_types = ConflictResolverSpec(
            "resolver", "1", domain.domain_id, domain.version,
            domain.content_hash,
            ("actor-phase-action", "service-health-action"),
            PreferLexicalActorResolver(),
        )
        self.assertEqual(first.content_hash, same.content_hash)
        self.assertNotEqual(first.content_hash, changed_hook.content_hash)
        self.assertNotEqual(first.content_hash, changed_types.content_hash)

    def test_configured_resolver_is_not_called_on_conflict_free_step(self):
        self.require_conflict()
        domain, story = make_v1_world_domain(), make_v1_world_story()
        hook = RecordingNoopConflictResolver()
        resolver = ConflictResolverSpec(
            "resolver", "1", domain.domain_id, domain.version,
            domain.content_hash,
            ("actor-phase-action", "service-health-action", "noop-action"),
            hook,
        )
        base = make_transition_model()
        model = WorldTransitionModelSpec(
            base.model_id,
            base.version,
            base.domain_id,
            base.domain_version,
            base.domain_spec_hash,
            base.transitions,
            resolver,
        )
        prior = world_state_from_story(story, domain)
        result = advance_world_step(
            story,
            domain,
            prior,
            model,
            (intent("d-bob-phase", "bob-ready"),),
        )
        self.assertNotEqual(model.content_hash, base.content_hash)
        self.assertEqual(hook.calls, [])
        self.assertEqual(result.conflict_resolutions, ())
        self.assertNotIn("conflict_resolutions", result.to_dict())
        self.assertEqual(
            result.next_state.transition_batch_hash,
            stable_content_hash([item.to_dict() for item in result.transitions]),
        )

    def test_two_way_claim_conflict_resolves_once_against_prior_snapshot(self):
        self.require_conflict()
        domain, story = make_conflict_domain(), None
        story = make_conflict_story(domain)
        hook = ClaimResolver()
        resolver = make_resolver(domain, hook, ("claim-service-action",))
        model = make_conflict_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        result = advance_world_step(
            story, domain, prior, model,
            (
                _conflict_intent("d-alice-claim-svc", "alice-claim-svc", "1"),
                _conflict_intent("d-bob-claim-svc", "bob-claim-svc", "2"),
            ),
        )
        self.assertEqual(len(result.conflict_resolutions), 1)
        context = result.conflict_resolutions[0].context
        self.assertEqual(tuple(item.actor_id for item in context.participants), ("alice", "bob"))
        self.assertEqual(len(hook.calls), 1)
        self.assertEqual(hook.calls[0][0], dict(prior.values))
        self.assertEqual(
            result.next_state.values[_cell("svc", "Service", "service.owner")],
            _agent_ref("alice"),
        )

    def test_three_way_transitive_attack_defend_taunt_is_one_component(self):
        self.require_conflict()
        domain = make_conflict_domain()
        story = make_conflict_story(domain)
        resolver = make_resolver(
            domain,
            AttackDefenseResolver(),
            ("attack-action", "defend-action", "taunt-action"),
        )
        model = make_conflict_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        result = advance_world_step(
            story, domain, prior, model,
            (
                _conflict_intent("d-alice-attack-bob", "alice-attack-bob", "1"),
                _conflict_intent("d-bob-defend", "bob-defend", "2"),
                _conflict_intent("d-carol-taunt-bob", "carol-taunt-bob", "3"),
            ),
        )
        self.assertEqual(len(result.conflict_resolutions), 1)
        self.assertEqual(
            tuple(item.actor_id for item in result.conflict_resolutions[0].context.participants),
            ("alice", "bob", "carol"),
        )

    def test_two_independent_claim_components_are_canonical_and_order_invariant(self):
        self.require_conflict()
        domain = make_conflict_domain()
        story = make_conflict_story(domain)
        resolver = make_resolver(domain, ClaimResolver(), ("claim-service-action",))
        model = make_conflict_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        intents = (
            _conflict_intent("d-alice-claim-svc", "alice-claim-svc", "1"),
            _conflict_intent("d-bob-claim-svc", "bob-claim-svc", "2"),
            _conflict_intent("d-carol-claim-svc2", "carol-claim-svc2", "3"),
            _conflict_intent("d-dave-claim-svc2", "dave-claim-svc2", "4"),
        )
        first = advance_world_step(story, domain, prior, model, intents)
        second = advance_world_step(story, domain, prior, model, tuple(reversed(intents)))
        self.assertEqual(len(first.conflict_resolutions), 2)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.content_hash, second.content_hash)

    def test_noop_delta_does_not_create_component(self):
        self.require_conflict()
        domain = make_conflict_domain()
        story = make_conflict_story(domain)
        resolver = make_resolver(domain, ClaimResolver(), ("claim-service-action",))
        model = make_conflict_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        result = advance_world_step(
            story, domain, prior, model,
            (
                _conflict_intent("d-alice-claim-svc", "alice-claim-svc", "1"),
                _conflict_intent("d-bob-claim-svc", "bob-claim-svc", "2"),
                _conflict_intent("d-dave-noop", "dave-wait", "3"),
            ),
        )
        participants = result.conflict_resolutions[0].context.participants
        self.assertNotIn("d-dave-noop", {item.decision_id for item in participants})
        self.assertEqual(len(result.transitions), 3)

    def test_undeclared_resolver_supported_type_rejects_before_action_hooks(self):
        self.require_conflict()
        domain = make_conflict_domain()
        story = make_conflict_story(domain)
        hook = ClaimResolver()
        resolver = ConflictResolverSpec(
            "resolver", "1", domain.domain_id, domain.version,
            domain.content_hash, ("missing-action-type",), hook,
        )
        model = make_conflict_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        with self.assertRaises(WorldTransitionError):
            advance_world_step(
                story, domain, prior, model,
                (
                    _conflict_intent("d-alice-claim-svc", "alice-claim-svc", "1"),
                    _conflict_intent("d-bob-claim-svc", "bob-claim-svc", "2"),
                ),
            )
        self.assertEqual(hook.calls, [])

    def test_conflict_participant_unsupported_type_rejects_before_resolver_hook(self):
        self.require_conflict()
        domain = make_conflict_domain()
        story = make_conflict_story(domain)
        hook = ClaimResolver()
        resolver = make_resolver(domain, hook, ("claim-service-action",))
        model = make_conflict_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        with self.assertRaises(WorldTransitionConflictResolutionError):
            advance_world_step(
                story, domain, prior, model,
                (
                    _conflict_intent("d-alice-attack-bob", "alice-attack-bob", "1"),
                    _conflict_intent("d-bob-defend", "bob-defend", "2"),
                ),
            )
        self.assertEqual(hook.calls, [])

    def test_resolver_attestation_unavailable_rejects_before_resolver_hook(self):
        self.require_conflict()
        domain = make_conflict_domain()
        story = make_conflict_story(domain)
        hook = ClaimResolver()
        resolver = make_resolver(domain, hook, ("claim-service-action",))
        model = make_conflict_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        with patch(
            "narrative_dynamics.narrative.conflict.measure_implementation",
            side_effect=ImplementationAttestationUnavailable("resolver unavailable"),
        ):
            with self.assertRaises(WorldTransitionConflictResolutionError) as caught:
                advance_world_step(
                    story, domain, prior, model,
                    (
                        _conflict_intent("d-alice-claim-svc", "alice-claim-svc", "1"),
                        _conflict_intent("d-bob-claim-svc", "bob-claim-svc", "2"),
                    ),
                )
        self.assertEqual(hook.calls, [])
        self.assertIsInstance(caught.exception.__cause__, ImplementationAttestationUnavailable)

    def test_resolver_exception_is_typed_and_preserves_runtime_error_cause(self):
        self.require_conflict()
        domain = make_conflict_domain()
        story = make_conflict_story(domain)
        hook = RaisingResolver()
        resolver = make_resolver(domain, hook, ("claim-service-action",))
        model = make_conflict_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        with self.assertRaises(WorldTransitionConflictResolutionError) as caught:
            advance_world_step(
                story, domain, prior, model,
                (
                    _conflict_intent("d-alice-claim-svc", "alice-claim-svc", "1"),
                    _conflict_intent("d-bob-claim-svc", "bob-claim-svc", "2"),
                ),
            )
        self.assertEqual(hook.calls, 1)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(str(caught.exception.__cause__), "resolver boom")

    def test_competing_claims_keep_losing_attempt_in_transition_lineage(self):
        self.require_conflict()
        domain = make_conflict_domain()
        story = make_conflict_story(domain)
        resolver = make_resolver(domain, ClaimResolver(), ("claim-service-action",))
        model = make_conflict_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        result = advance_world_step(
            story, domain, prior, model,
            (
                _conflict_intent("d-alice-claim-svc", "alice-claim-svc", "1"),
                _conflict_intent("d-bob-claim-svc", "bob-claim-svc", "2"),
            ),
        )
        self.assertEqual({item.actor_id for item in result.transitions}, {"alice", "bob"})
        self.assertEqual(
            {item.transition_record_hash for item in result.conflict_resolutions[0].context.participants},
            {item.content_hash for item in result.transitions},
        )

    def test_three_party_auction_uses_bid_argument_and_explicit_lexical_tie_break(self):
        self.require_conflict()
        domain = make_conflict_domain()
        story = make_conflict_story(domain)
        resolver = make_resolver(domain, AuctionResolver(), ("bid-service-action",))
        model = make_conflict_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        intents = (
            _conflict_intent("d-alice-bid", "alice-bid-11", "1"),
            _conflict_intent("d-bob-bid", "bob-bid-11", "2"),
            _conflict_intent("d-carol-bid", "carol-bid-8", "3"),
        )
        first = advance_world_step(story, domain, prior, model, intents)
        second = advance_world_step(story, domain, prior, model, tuple(reversed(intents)))
        self.assertEqual(
            first.next_state.values[_cell("svc", "Service", "service.owner")],
            _agent_ref("alice"),
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.content_hash, second.content_hash)

    def test_attack_defense_replaces_entire_conflicting_component_delta(self):
        self.require_conflict()
        domain = make_conflict_domain()
        story = make_conflict_story(domain)
        resolver = make_resolver(
            domain,
            AttackDefenseResolver(),
            ("attack-action", "defend-action"),
        )
        model = make_conflict_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        result = advance_world_step(
            story, domain, prior, model,
            (
                _conflict_intent("d-alice-attack-bob", "alice-attack-bob", "1"),
                _conflict_intent("d-bob-defend", "bob-defend", "2"),
            ),
        )
        self.assertEqual(
            result.next_state.values[_cell("bob", "Agent", "agent.health")],
            TypedValue("CombatHealth", "healthy"),
        )
        self.assertEqual(
            result.next_state.values[_cell("alice", "Agent", "agent.combat")],
            TypedValue("CombatStatus", "attack-failed"),
        )
        self.assertEqual(
            result.next_state.values[_cell("bob", "Agent", "agent.combat")],
            TypedValue("CombatStatus", "defense-success"),
        )

    def test_resolved_batch_hash_binds_original_transitions_and_resolution_records(self):
        self.require_conflict()
        domain = make_conflict_domain()
        story = make_conflict_story(domain)
        resolver = make_resolver(domain, ClaimResolver(), ("claim-service-action",))
        model = make_conflict_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        result = advance_world_step(
            story, domain, prior, model,
            (
                _conflict_intent("d-alice-claim-svc", "alice-claim-svc", "1"),
                _conflict_intent("d-bob-claim-svc", "bob-claim-svc", "2"),
            ),
        )
        self.assertEqual(
            result.next_state.transition_batch_hash,
            stable_content_hash({
                "transitions": [item.to_dict() for item in result.transitions],
                "conflict_resolutions": [
                    item.to_dict() for item in result.conflict_resolutions
                ],
            }),
        )

    def test_world_step_result_rejects_internal_resolution_forgery(self):
        self.require_conflict()
        domain = make_conflict_domain()
        story = make_conflict_story(domain)
        resolver = make_resolver(domain, ClaimResolver(), ("claim-service-action",))
        model = make_conflict_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        result = advance_world_step(
            story, domain, prior, model,
            (
                _conflict_intent("d-alice-claim-svc", "alice-claim-svc", "1"),
                _conflict_intent("d-bob-claim-svc", "bob-claim-svc", "2"),
            ),
        )
        resolution = result.conflict_resolutions[0]
        bad = _forge(resolution, prior_state_hash=_hash("bad-prior"))
        with self.assertRaises(ValueError):
            replace(result, conflict_resolutions=(bad,))
        participant = resolution.context.participants[0]
        bad_participant = replace(
            participant,
            transition_record_hash=_hash("missing-transition"),
        )
        bad_context = replace(
            resolution.context,
            participants=(bad_participant,) + resolution.context.participants[1:],
        )
        with self.assertRaises(ValueError):
            replace(
                result,
                conflict_resolutions=(replace(resolution, context=bad_context),),
            )

    def test_heterogeneous_conflict_resolution_keeps_shared_prior_and_projects_resolved_world(self):
        self.require_conflict()
        domain, story, model = make_heterogeneous_conflict_case()
        prior = simulation_state_from_story(story, domain, model, at_time=10)
        result = simulate_step(story, domain, prior, model)
        self.assertEqual(
            {item.decision_result.model_kind for item in result.agent_steps},
            {"reactive", "intentional", "planning"},
        )
        for step in result.agent_steps:
            self.assertEqual(
                step.decision_result.ledger_hash,
                prior.evidence_ledger.content_hash,
            )
        self.assertEqual(len(result.world_step.conflict_resolutions), 1)
        self.assertEqual(
            result.world_step.next_state.values[alert_cell()],
            TypedValue("AlertState", True),
        )
        self.assertEqual(
            result.admission_result.projection_result.source_world_step_hash,
            result.world_step.content_hash,
        )
        context = result.world_step.conflict_resolutions[0].context
        self.assertFalse(hasattr(context, "belief_state"))
        self.assertFalse(hasattr(context, "evidence_ledger"))
        self.assertFalse(hasattr(context, "planning_trace"))

    def test_two_round_heterogeneous_conflict_resolution_replays_exactly(self):
        self.require_conflict()
        domain1, story1, model1 = make_heterogeneous_conflict_case()
        domain2, story2, model2 = make_heterogeneous_conflict_case()
        initial1 = simulation_state_from_story(story1, domain1, model1, at_time=10)
        initial2 = simulation_state_from_story(story2, domain2, model2, at_time=10)
        first = simulate_trajectory(story1, domain1, initial1, model1, rounds=2)
        second = simulate_trajectory(story2, domain2, initial2, model2, rounds=2)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertTrue(
            all(step.world_step.conflict_resolutions for step in first.steps)
        )

    def test_simulation_model_identity_binds_conflict_resolver_implementation(self):
        self.require_conflict()
        _, _, first = make_heterogeneous_conflict_case(
            resolver_hook=ConflictAlertResolver(),
        )
        _, _, second = make_heterogeneous_conflict_case(
            resolver_hook=ConflictAlertResolverV2(),
        )
        self.assertNotEqual(
            first.world_model.conflict_resolver.content_hash,
            second.world_model.conflict_resolver.content_hash,
        )
        self.assertNotEqual(first.world_model.content_hash, second.world_model.content_hash)
        self.assertNotEqual(first.content_hash, second.content_hash)


if __name__ == "__main__":
    unittest.main()

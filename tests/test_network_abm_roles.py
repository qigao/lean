from dataclasses import replace
import unittest

from narrative_dynamics.abm.autonomy_contracts import (
    AutonomousNetworkModel,
    TruthObservation,
)
from narrative_dynamics.abm.interventions import EdgeSelector
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleEventKind,
    LifecycleStatus,
    PopulationLifecycleEvent,
)
from narrative_dynamics.abm.role_contracts import (
    DynamicRoleModel,
    RoleTransitionRule,
    initialize_dynamic_role_population,
)
from narrative_dynamics.abm.roles import (
    DynamicRoleRoundResult,
    simulate_dynamic_role_population,
    simulate_dynamic_role_round,
)
from tests.dynamic_role_fixtures import dynamic_role_model, role_state


def enter(agent_id: str, belief=None):
    return PopulationLifecycleEvent(agent_id, LifecycleEventKind.ENTER, belief)


def die(agent_id: str):
    return PopulationLifecycleEvent(agent_id, LifecycleEventKind.DEATH)


def edge(source: str, target: str):
    return EdgeSelector(source, target, "peer")


def truth(source: str, target: str, observed_truth: float = 1.0):
    return TruthObservation(edge(source, target), observed_truth)


def intent_for(result, agent_id: str):
    return next(
        item for item in result.autonomy_result.intents if item.agent_id == agent_id
    )


def transition_for(result, agent_id: str):
    return next(item for item in result.transitions if item.agent_id == agent_id)


def resource(state, agent_id: str):
    return next(
        item for item in state.autonomy_state.agents if item.agent_id == agent_id
    )


def member(state, agent_id: str):
    return next(
        item
        for item in state.autonomy_state.evolving_state.members
        if item.agent_id == agent_id
    )


def model_with(*, policy_changes=(), rules=None):
    base = dynamic_role_model()
    changes = dict(policy_changes)
    policies = tuple(
        replace(item, **changes[item.role]) if item.role in changes else item
        for item in base.autonomy_model.role_policies
    )
    autonomy = AutonomousNetworkModel(
        base.autonomy_model.model_id,
        base.autonomy_model.version,
        base.autonomy_model.evolving_model,
        policies,
    )
    return DynamicRoleModel(
        base.model_id,
        base.version,
        autonomy,
        base.transition_rules if rules is None else tuple(rules),
    )


class DynamicRoleSimulationTests(unittest.TestCase):
    def test_verified_relay_transitions_after_relay_policy_governs_round(self):
        model = dynamic_role_model()
        initial = initialize_dynamic_role_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )

        result = simulate_dynamic_role_round(
            model,
            initial,
            environment_events=(enter("b"),),
            truth_observations=(truth("a", "b"),),
        )

        self.assertIsInstance(result, DynamicRoleRoundResult)
        self.assertEqual(intent_for(result, "b").role, "relay")
        transition = transition_for(result, "b")
        self.assertEqual(transition.from_role, "relay")
        self.assertEqual(transition.to_role, "source")
        self.assertEqual(transition.observed_belief, 0.5)
        self.assertEqual(transition.completed_rounds_in_from_role, 1)
        self.assertEqual(transition.decision_count, 1)
        self.assertEqual(transition.verification_count, 1)
        self.assertEqual(role_state(result.next_state, "b").current_role, "source")
        self.assertEqual(role_state(result.next_state, "b").rounds_in_role, 0)

    def test_new_role_first_governs_the_following_round(self):
        model = dynamic_role_model()
        initial = initialize_dynamic_role_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )
        first = simulate_dynamic_role_round(
            model,
            initial,
            environment_events=(enter("b"),),
            truth_observations=(truth("a", "b"),),
        )

        second = simulate_dynamic_role_round(
            model,
            first.next_state,
            truth_observations=(truth("b", "c"),),
        )

        self.assertEqual(intent_for(first, "b").role, "relay")
        self.assertEqual(intent_for(second, "b").role, "source")
        self.assertEqual(role_state(second.next_state, "b").rounds_in_role, 1)

    def test_transition_never_replenishes_agent_verification_budget(self):
        model = model_with(
            policy_changes=(
                ("source", {"initial_verification_budget": 5}),
            ),
        )
        initial = initialize_dynamic_role_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )

        result = simulate_dynamic_role_round(
            model,
            initial,
            environment_events=(enter("b"),),
            truth_observations=(truth("a", "b"),),
        )

        self.assertEqual(role_state(result.next_state, "b").current_role, "source")
        self.assertEqual(resource(result.next_state, "b").remaining_verification_budget, 1)

    def test_agent_exiting_in_autonomy_round_does_not_transition(self):
        model = model_with(
            policy_changes=(("relay", {"exit_belief_threshold": 0.6}),),
        )
        initial = initialize_dynamic_role_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )

        result = simulate_dynamic_role_round(
            model,
            initial,
            environment_events=(enter("b"),),
            truth_observations=(truth("a", "b"),),
        )

        self.assertEqual(member(result.next_state, "b").status, LifecycleStatus.INACTIVE)
        self.assertFalse(any(item.agent_id == "b" for item in result.transitions))
        self.assertEqual(role_state(result.next_state, "b"), role_state(initial, "b"))

    def test_reentry_retains_changed_role_for_entrant_sharing_and_decision(self):
        model = model_with(
            policy_changes=(
                ("relay", {"sharing_enabled": False}),
                ("source", {"exit_belief_threshold": 0.9}),
            ),
        )
        initial = initialize_dynamic_role_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )
        transitioned = simulate_dynamic_role_round(
            model,
            initial,
            environment_events=(enter("b"),),
            truth_observations=(truth("a", "b"),),
        )
        exited = simulate_dynamic_role_round(model, transitioned.next_state)
        self.assertEqual(member(exited.next_state, "b").status, LifecycleStatus.INACTIVE)
        self.assertEqual(role_state(exited.next_state, "b").current_role, "source")

        reentered = simulate_dynamic_role_round(
            model,
            exited.next_state,
            environment_events=(enter("b", 1.0),),
            truth_observations=(truth("b", "c"),),
        )

        self.assertEqual(intent_for(reentered, "b").role, "source")
        self.assertTrue(
            any(
                item.source_agent_id == "b" and item.target_agent_id == "c"
                for item in reentered.autonomy_result.transmissions
            )
        )

    def test_smallest_matching_priority_wins(self):
        rules = (
            RoleTransitionRule(
                "relay", "recipient", priority=1, minimum_belief=0.5
            ),
            RoleTransitionRule(
                "relay", "source", priority=0, minimum_belief=0.5
            ),
        )
        model = model_with(rules=rules)
        initial = initialize_dynamic_role_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )

        result = simulate_dynamic_role_round(
            model,
            initial,
            environment_events=(enter("b"),),
            truth_observations=(truth("a", "b"),),
        )

        self.assertEqual(transition_for(result, "b").rule_priority, 0)
        self.assertEqual(role_state(result.next_state, "b").current_role, "source")

    def test_catalog_and_rule_input_order_do_not_change_replay(self):
        forward = dynamic_role_model()
        reverse = dynamic_role_model(reverse=True)
        forward_state = initialize_dynamic_role_population(
            forward,
            beliefs={"a": 1.0, "c": 0.5},
        )
        reverse_state = initialize_dynamic_role_population(
            reverse,
            beliefs={"c": 0.5, "a": 1.0},
        )

        first = simulate_dynamic_role_round(
            forward,
            forward_state,
            environment_events=(enter("b"),),
            truth_observations=(truth("a", "b"),),
        )
        second = simulate_dynamic_role_round(
            reverse,
            reverse_state,
            environment_events=(enter("b"),),
            truth_observations=(truth("a", "b"),),
        )

        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, second.content_hash)

    def test_no_match_increments_active_tenure_and_zero_active_preserves_it(self):
        model = dynamic_role_model()
        initial = initialize_dynamic_role_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )
        emptied = simulate_dynamic_role_round(
            model,
            initial,
            environment_events=(die("a"), die("c")),
        )
        self.assertEqual(emptied.transitions, ())
        self.assertEqual(emptied.autonomy_result.intents, ())
        self.assertEqual(emptied.next_state.agents, initial.agents)

        active = simulate_dynamic_role_round(model, initial)
        self.assertEqual(role_state(active.next_state, "a").rounds_in_role, 1)
        self.assertEqual(role_state(active.next_state, "c").rounds_in_role, 1)
        self.assertEqual(role_state(active.next_state, "b").rounds_in_role, 0)

    def test_multi_round_schedule_chains_role_and_autonomy_states(self):
        model = dynamic_role_model()
        initial = initialize_dynamic_role_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )

        trajectory = simulate_dynamic_role_population(
            model,
            initial,
            environment_event_schedule=((enter("b"),), ()),
            truth_observation_schedule=(
                (truth("a", "b"),),
                (truth("b", "c"),),
            ),
        )

        self.assertEqual(trajectory.final_state.round_index, 2)
        self.assertEqual(trajectory.rounds[0].next_state, trajectory.rounds[1].prior_state)
        self.assertEqual(
            trajectory.rounds[0].autonomy_result.next_state,
            trajectory.rounds[1].autonomy_result.prior_state,
        )

    def test_schedule_and_model_state_validation_fail_closed(self):
        model = dynamic_role_model()
        initial = initialize_dynamic_role_population(model)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            simulate_dynamic_role_population(
                model,
                initial,
                environment_event_schedule=(),
                truth_observation_schedule=(),
            )
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            simulate_dynamic_role_population(
                model,
                initial,
                environment_event_schedule=((), ()),
                truth_observation_schedule=((),),
            )
        different = DynamicRoleModel(
            "different",
            model.version,
            model.autonomy_model,
            model.transition_rules,
        )
        with self.assertRaisesRegex(ValueError, "exact model identity"):
            simulate_dynamic_role_round(different, initial)


if __name__ == "__main__":
    unittest.main()

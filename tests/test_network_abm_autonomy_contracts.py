import math
import unittest

from narrative_dynamics.abm import EdgeSelector
from narrative_dynamics.abm.autonomy_contracts import (
    AgentActionIntent,
    AutonomousAgentState,
    AutonomousNetworkModel,
    AutonomousPopulationState,
    RoleDecisionPolicy,
    SharingDecision,
    TruthObservation,
    initialize_autonomous_population,
)
from tests.autonomy_fixtures import autonomous_model, role_policies


def resource(state, agent_id: str):
    return next(item for item in state.agents if item.agent_id == agent_id)


class AgentAutonomyContractTests(unittest.TestCase):
    def test_initialization_assigns_role_policy_budget_and_initial_choice(self):
        model = autonomous_model()

        state = initialize_autonomous_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )

        self.assertEqual(state.model_id, model.model_id)
        self.assertEqual(state.model_hash, model.content_hash)
        self.assertEqual(state.round_index, 0)
        self.assertEqual(tuple(item.agent_id for item in state.agents), ("a", "b", "c"))
        self.assertTrue(resource(state, "a").sharing)
        self.assertEqual(resource(state, "a").remaining_verification_budget, 1)
        self.assertFalse(resource(state, "b").sharing)
        self.assertEqual(resource(state, "b").remaining_verification_budget, 2)
        self.assertFalse(resource(state, "c").sharing)
        self.assertEqual(resource(state, "c").remaining_verification_budget, 1)
        self.assertEqual(resource(state, "a").decision_count, 0)

    def test_model_and_initial_state_are_canonical(self):
        first = autonomous_model()
        second = autonomous_model(reverse=True)

        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(tuple(item.role for item in first.role_policies), (
            "recipient",
            "relay",
            "source",
        ))
        self.assertEqual(
            initialize_autonomous_population(
                first,
                beliefs={"a": 1.0, "c": 0.5},
            ).content_hash,
            initialize_autonomous_population(
                second,
                beliefs={"c": 0.5, "a": 1.0},
            ).content_hash,
        )

    def test_role_policy_validates_flags_thresholds_and_budget(self):
        for bad in (-0.1, 1.1, math.nan, math.inf, True):
            with self.subTest(value=bad):
                with self.assertRaises((TypeError, ValueError)):
                    RoleDecisionPolicy("role", True, bad, 0.5, None, 1)
                with self.assertRaises((TypeError, ValueError)):
                    RoleDecisionPolicy("role", True, 0.5, bad, None, 1)
                with self.assertRaises((TypeError, ValueError)):
                    RoleDecisionPolicy("role", True, 0.5, 0.5, bad, 1)
        with self.assertRaisesRegex(TypeError, "sharing enabled"):
            RoleDecisionPolicy("role", 1, 0.5, 0.5, None, 1)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            RoleDecisionPolicy("role", True, 0.5, 0.5, None, -1)

    def test_autonomous_model_requires_exact_role_policy_coverage(self):
        evolving = autonomous_model().evolving_model
        policies = role_policies()
        with self.assertRaisesRegex(ValueError, "exact role catalog"):
            AutonomousNetworkModel("a", "1", evolving, policies[:-1])
        with self.assertRaisesRegex(ValueError, "unique"):
            AutonomousNetworkModel("a", "1", evolving, (policies[0],) * 2 + policies[1:])
        with self.assertRaisesRegex(ValueError, "exact role catalog"):
            AutonomousNetworkModel(
                "a",
                "1",
                evolving,
                policies + (RoleDecisionPolicy("unknown", True, 0.5, 0.5, None, 0),),
            )

    def test_action_intent_binds_local_observation_and_budget_delta(self):
        edge = EdgeSelector("a", "b", "peer")
        intent = AgentActionIntent(
            1,
            "b",
            "relay",
            0.5,
            1,
            0.5,
            SharingDecision.SHARE,
            edge,
            False,
            2,
            1,
        )
        self.assertEqual(intent.verification_edge, edge)
        self.assertEqual(intent.verification_budget_after, 1)

        with self.assertRaisesRegex(ValueError, "budget delta"):
            AgentActionIntent(
                1,
                "b",
                "relay",
                0.5,
                1,
                0.5,
                SharingDecision.SHARE,
                edge,
                False,
                2,
                2,
            )
        with self.assertRaisesRegex(ValueError, "mean incoming trust must be None"):
            AgentActionIntent(
                1,
                "b",
                "relay",
                0.5,
                0,
                0.5,
                SharingDecision.SILENT,
                None,
                False,
                2,
                2,
            )
        with self.assertRaisesRegex(ValueError, "exiting agent must be silent"):
            AgentActionIntent(
                1,
                "b",
                "relay",
                0.1,
                0,
                None,
                SharingDecision.SHARE,
                None,
                True,
                2,
                2,
            )

    def test_truth_observation_and_resource_counters_validate(self):
        edge = EdgeSelector("a", "b", "peer")
        self.assertEqual(TruthObservation(edge, 1.0).to_dict()["observed_truth"], 1.0)
        with self.assertRaises((TypeError, ValueError)):
            TruthObservation(edge, math.nan)
        with self.assertRaisesRegex(ValueError, "cannot exceed decisions"):
            AutonomousAgentState("b", True, 1, 2, 1, 0)
        with self.assertRaisesRegex(ValueError, "cannot exceed decisions"):
            AutonomousAgentState("b", True, 1, 0, 1, 2)

    def test_autonomous_state_rejects_duplicate_resources_and_round_mismatch(self):
        model = autonomous_model()
        initial = initialize_autonomous_population(model)
        with self.assertRaisesRegex(ValueError, "agent ids must be unique"):
            AutonomousPopulationState(
                initial.model_id,
                initial.model_hash,
                0,
                None,
                initial.evolving_state,
                (initial.agents[0], initial.agents[0]),
            )
        later_embedded = type(initial.evolving_state)(
            initial.evolving_state.model_id,
            initial.evolving_state.model_hash,
            1,
            initial.evolving_state.content_hash,
            initial.evolving_state.members,
            initial.evolving_state.edge_trust,
            initial.evolving_state.edge_topology,
        )
        with self.assertRaisesRegex(ValueError, "round index"):
            AutonomousPopulationState(
                initial.model_id,
                initial.model_hash,
                0,
                None,
                later_embedded,
                initial.agents,
            )


if __name__ == "__main__":
    unittest.main()

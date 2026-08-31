from dataclasses import replace
import unittest

from narrative_dynamics.abm.role_contracts import (
    DynamicRoleAgentState,
    DynamicRoleModel,
    DynamicRolePopulationState,
    RoleTransitionRecord,
    RoleTransitionRule,
    initialize_dynamic_role_population,
)
from tests.dynamic_role_fixtures import (
    dynamic_role_model,
    role_state,
    role_transition_rules,
)


class DynamicRoleContractTests(unittest.TestCase):
    def test_initial_roles_come_from_profiles_with_zero_history(self):
        model = dynamic_role_model()

        state = initialize_dynamic_role_population(
            model,
            beliefs={"a": 1.0, "c": 0.5},
        )

        self.assertEqual(role_state(state, "a").current_role, "source")
        self.assertEqual(role_state(state, "b").current_role, "relay")
        self.assertEqual(role_state(state, "c").current_role, "recipient")
        self.assertEqual(role_state(state, "b").rounds_in_role, 0)
        self.assertEqual(role_state(state, "b").transition_count, 0)
        self.assertEqual(state.round_index, state.autonomy_state.round_index)

    def test_rule_contract_requires_directed_substantive_valid_interval(self):
        with self.assertRaisesRegex(ValueError, "self-transition"):
            RoleTransitionRule("relay", "relay", priority=0, minimum_belief=0.5)
        with self.assertRaisesRegex(ValueError, "substantive condition"):
            RoleTransitionRule("relay", "source", priority=0)
        with self.assertRaisesRegex(ValueError, "belief interval"):
            RoleTransitionRule(
                "relay",
                "source",
                priority=0,
                minimum_belief=0.7,
                maximum_belief=0.7,
            )

    def test_model_rejects_unknown_roles_and_ambiguous_priorities(self):
        autonomy = dynamic_role_model().autonomy_model
        unknown = RoleTransitionRule(
            "unknown",
            "source",
            priority=0,
            minimum_belief=0.5,
        )
        with self.assertRaisesRegex(ValueError, "configured role catalog"):
            DynamicRoleModel("bad", "1", autonomy, (unknown,))

        duplicate_priority = (
            RoleTransitionRule(
                "relay", "source", priority=0, minimum_belief=0.7
            ),
            RoleTransitionRule(
                "relay", "recipient", priority=0, maximum_belief=0.3
            ),
        )
        with self.assertRaisesRegex(ValueError, "unique priority"):
            DynamicRoleModel("bad", "1", autonomy, duplicate_priority)

    def test_model_and_initial_state_are_canonical_and_hash_stable(self):
        forward = dynamic_role_model()
        reverse = dynamic_role_model(reverse=True)

        self.assertEqual(forward.content_hash, reverse.content_hash)
        forward_state = initialize_dynamic_role_population(
            forward,
            beliefs={"a": 1.0, "c": 0.5},
        )
        reverse_state = initialize_dynamic_role_population(
            reverse,
            beliefs={"c": 0.5, "a": 1.0},
        )
        self.assertEqual(forward_state.content_hash, reverse_state.content_hash)

    def test_population_state_requires_exact_unique_agent_role_catalog(self):
        model = dynamic_role_model()
        state = initialize_dynamic_role_population(model)
        duplicate = (state.agents[0], state.agents[0], state.agents[2])
        with self.assertRaisesRegex(ValueError, "agent ids must be unique"):
            DynamicRolePopulationState(
                state.model_id,
                state.model_hash,
                state.round_index,
                state.parent_state_hash,
                state.autonomy_state,
                duplicate,
            )

    def test_role_state_and_transition_record_validate_audit_values(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            DynamicRoleAgentState("b", "relay", -1, 0)
        record = RoleTransitionRecord(
            1,
            "b",
            "relay",
            "source",
            0,
            0.75,
            1,
            1,
            1,
        )
        self.assertEqual(record.to_role, "source")
        self.assertEqual(record.completed_rounds_in_from_role, 1)
        with self.assertRaisesRegex(ValueError, "distinct roles"):
            replace(record, to_role="relay")

    def test_model_and_state_values_are_immutable(self):
        model = dynamic_role_model()
        state = initialize_dynamic_role_population(model)
        with self.assertRaises(Exception):
            model.version = "2"
        with self.assertRaises(Exception):
            state.agents = ()


if __name__ == "__main__":
    unittest.main()

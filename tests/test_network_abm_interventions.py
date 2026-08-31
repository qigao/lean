import unittest

from narrative_dynamics.abm import simulate_population
from narrative_dynamics.abm.interventions import (
    BeliefSeed,
    EdgeInfluenceChange,
    EdgeSelector,
    NetworkIntervention,
    apply_intervention,
    compare_intervention,
    no_propagation_intervention,
)
from tests.network_abm_fixtures import agent, line_case


class NetworkABMInterventionTests(unittest.TestCase):
    def test_disabling_bridge_blocks_second_hop_and_reports_metric_delta(self):
        model, initial = line_case()
        intervention = NetworkIntervention(
            "cut-bridge",
            disabled_edges=(EdgeSelector("b", "c", "peer"),),
        )

        comparison = compare_intervention(
            model,
            initial,
            intervention,
            rounds=2,
        )

        self.assertEqual(agent(comparison.baseline.final_state, "c").belief, 1.0)
        self.assertEqual(agent(comparison.treatment.final_state, "c").belief, 0.0)
        self.assertEqual(comparison.baseline_metrics.adoption_rate, 0.75)
        self.assertEqual(comparison.treatment_metrics.adoption_rate, 0.5)
        self.assertEqual(comparison.metric_deltas["adoption_rate"], -0.25)
        self.assertEqual(comparison.metric_deltas["informed_rate"], -0.25)
        self.assertEqual(agent(initial, "c").belief, 0.0)

    def test_influence_override_and_belief_seed_rebind_treatment_identity(self):
        model, initial = line_case()
        intervention = NetworkIntervention(
            "weak-link-and-seed",
            influence_changes=(
                EdgeInfluenceChange(EdgeSelector("a", "b", "peer"), 0.25),
            ),
            belief_seeds=(BeliefSeed("c", 0.6),),
        )

        applied = apply_intervention(model, initial, intervention)

        self.assertNotEqual(applied.model.content_hash, model.content_hash)
        self.assertNotEqual(applied.initial_state.model_hash, initial.model_hash)
        self.assertEqual(agent(applied.initial_state, "c").belief, 0.6)
        self.assertTrue(agent(applied.initial_state, "c").broadcasting)
        result = simulate_population(applied.model, applied.initial_state, rounds=1)
        self.assertEqual(agent(result.final_state, "b").belief, 0.25)

    def test_no_propagation_null_preserves_unseeded_population_state(self):
        model, initial = line_case()
        intervention = no_propagation_intervention(model)

        comparison = compare_intervention(model, initial, intervention, rounds=3)

        self.assertEqual(comparison.treatment.final_state.agents, initial.agents)
        self.assertEqual(comparison.treatment_metrics.adoption_rate, 0.25)
        self.assertEqual(comparison.treatment_metrics.informed_rate, 0.25)
        self.assertEqual(comparison.baseline_metrics.adoption_rate, 0.75)

    def test_intervention_order_is_canonical_and_hash_stable(self):
        first = NetworkIntervention(
            "ordered",
            disabled_edges=(
                EdgeSelector("b", "c", "peer"),
                EdgeSelector("a", "b", "peer"),
            ),
            belief_seeds=(BeliefSeed("c", 0.7), BeliefSeed("a", 0.9)),
        )
        second = NetworkIntervention(
            "ordered",
            disabled_edges=tuple(reversed(first.disabled_edges)),
            belief_seeds=tuple(reversed(first.belief_seeds)),
        )

        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, second.content_hash)

    def test_interventions_reject_unknown_duplicate_and_conflicting_targets(self):
        model, initial = line_case()
        edge = EdgeSelector("a", "b", "peer")
        with self.assertRaisesRegex(ValueError, "unique"):
            NetworkIntervention("duplicate", disabled_edges=(edge, edge))
        with self.assertRaisesRegex(ValueError, "both disabled and overridden"):
            NetworkIntervention(
                "conflict",
                disabled_edges=(edge,),
                influence_changes=(EdgeInfluenceChange(edge, 0.5),),
            )
        with self.assertRaisesRegex(ValueError, "unknown edge"):
            apply_intervention(
                model,
                initial,
                NetworkIntervention(
                    "missing",
                    disabled_edges=(EdgeSelector("a", "c", "peer"),),
                ),
            )
        with self.assertRaisesRegex(ValueError, "unknown agent"):
            apply_intervention(
                model,
                initial,
                NetworkIntervention(
                    "missing-agent",
                    belief_seeds=(BeliefSeed("missing", 1.0),),
                ),
            )


if __name__ == "__main__":
    unittest.main()

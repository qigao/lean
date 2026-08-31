import unittest

import narrative_dynamics.abm as abm


class NetworkABMPublicAPITests(unittest.TestCase):
    def test_public_api_exports_complete_v2_surface(self):
        self.assertEqual(
            set(abm.__all__),
            {
                "NetworkAgentSpec",
                "SocialEdge",
                "SocialNetwork",
                "NetworkABMModel",
                "NetworkAgentState",
                "PopulationState",
                "initialize_population",
                "InformationTransmission",
                "NetworkRoundResult",
                "PopulationTrajectory",
                "simulate_round",
                "simulate_population",
                "EmergenceMetrics",
                "measure_emergence",
                "EdgeSelector",
                "EdgeInfluenceChange",
                "BeliefSeed",
                "NetworkIntervention",
                "AppliedIntervention",
                "InterventionComparison",
                "apply_intervention",
                "compare_intervention",
                "no_propagation_intervention",
                "AdaptiveTrustModel",
                "EdgeTrustState",
                "AdaptivePopulationState",
                "TruthFeedback",
                "initialize_adaptive_population",
                "EdgeTrustUpdate",
                "AdaptiveRoundResult",
                "AdaptiveTrajectory",
                "population_view",
                "simulate_adaptive_round",
                "simulate_adaptive_population",
                "AdaptiveTrustMetrics",
                "measure_adaptive_trust",
            },
        )
        for name in abm.__all__:
            self.assertTrue(hasattr(abm, name), name)


if __name__ == "__main__":
    unittest.main()

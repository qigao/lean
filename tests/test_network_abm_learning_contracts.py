import math
import unittest

from narrative_dynamics.abm import (
    EdgeSelector,
    NetworkABMModel,
    NetworkAgentSpec,
    SocialNetwork,
)
from narrative_dynamics.abm.adaptive_contracts import (
    AdaptivePopulationState,
    AdaptiveTrustModel,
    EdgeTrustState,
    TruthFeedback,
    initialize_adaptive_population,
)
from tests.adaptive_trust_fixtures import trust_base_model


class AdaptiveTrustContractTests(unittest.TestCase):
    def test_adaptive_initialization_covers_every_edge_and_agent(self):
        model = AdaptiveTrustModel(
            "adaptive-sources",
            "1",
            trust_base_model(),
            learning_rate=0.5,
            initial_trust=0.5,
        )

        state = initialize_adaptive_population(
            model,
            beliefs={"accurate": 1.0, "inaccurate": 0.0},
        )

        self.assertIsInstance(state, AdaptivePopulationState)
        self.assertEqual(state.model_id, model.model_id)
        self.assertEqual(state.model_hash, model.content_hash)
        self.assertEqual(state.round_index, 0)
        self.assertIsNone(state.parent_state_hash)
        self.assertEqual(
            tuple(item.agent_id for item in state.agents),
            ("accurate", "inaccurate", "target"),
        )
        self.assertEqual(tuple(item.trust for item in state.edge_trust), (0.5, 0.5))
        self.assertEqual(
            tuple(item.feedback_count for item in state.edge_trust),
            (0, 0),
        )
        self.assertTrue(state.agents[0].broadcasting)
        self.assertTrue(state.agents[1].broadcasting)
        self.assertFalse(state.agents[2].broadcasting)

    def test_adaptive_identity_is_canonical_across_base_input_order(self):
        first = AdaptiveTrustModel("adaptive", "1", trust_base_model(), 0.5, 0.5)
        second = AdaptiveTrustModel(
            "adaptive",
            "1",
            trust_base_model(reverse=True),
            0.5,
            0.5,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(
            initialize_adaptive_population(
                first,
                beliefs={"accurate": 1.0, "inaccurate": 0.0},
            ).content_hash,
            initialize_adaptive_population(
                second,
                beliefs={"inaccurate": 0.0, "accurate": 1.0},
            ).content_hash,
        )

    def test_learning_contracts_reject_invalid_numeric_and_identity_values(self):
        base = trust_base_model()
        for bad in (-0.1, 1.1, math.nan, math.inf, True):
            with self.subTest(value=bad):
                with self.assertRaises((TypeError, ValueError)):
                    AdaptiveTrustModel("adaptive", "1", base, bad, 0.5)
                with self.assertRaises((TypeError, ValueError)):
                    EdgeTrustState(
                        EdgeSelector("accurate", "target", "report"),
                        bad,
                        0,
                    )
                with self.assertRaises((TypeError, ValueError)):
                    TruthFeedback(
                        EdgeSelector("accurate", "target", "report"),
                        bad,
                    )

        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            EdgeTrustState(
                EdgeSelector("accurate", "target", "report"),
                0.5,
                -1,
            )

    def test_adaptive_state_rejects_duplicate_agent_and_edge_identity(self):
        model = AdaptiveTrustModel("adaptive", "1", trust_base_model(), 0.5, 0.5)
        initial = initialize_adaptive_population(
            model,
            beliefs={"accurate": 1.0, "inaccurate": 0.0},
        )
        with self.assertRaisesRegex(ValueError, "agent ids must be unique"):
            AdaptivePopulationState(
                initial.model_id,
                initial.model_hash,
                0,
                None,
                (initial.agents[0], initial.agents[0]),
                initial.edge_trust,
            )
        with self.assertRaisesRegex(ValueError, "edge identities must be unique"):
            AdaptivePopulationState(
                initial.model_id,
                initial.model_hash,
                0,
                None,
                initial.agents,
                (initial.edge_trust[0], initial.edge_trust[0]),
            )

    def test_initialization_rejects_unknown_seed(self):
        model = AdaptiveTrustModel("adaptive", "1", trust_base_model(), 0.5, 0.5)
        with self.assertRaisesRegex(ValueError, "unknown agent"):
            initialize_adaptive_population(model, beliefs={"missing": 1.0})

    def test_adaptive_model_requires_an_edge_to_learn(self):
        base = NetworkABMModel(
            "isolated",
            "1",
            (NetworkAgentSpec("a", "isolated", 1.0, 0.5, 0.5),),
            SocialNetwork(("a",), ()),
        )
        with self.assertRaisesRegex(ValueError, "at least one social edge"):
            AdaptiveTrustModel("adaptive", "1", base, 0.5, 0.5)


if __name__ == "__main__":
    unittest.main()

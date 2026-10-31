"""R2.M5g.3: exact rational specification checked against ORIGINAL V19.

The V19/SQLite source and already pinned M5g.2 four-cell conformance are
authoritative. This checker is mathematical verification, not a second
cognitive engine or empirical estimate of influence.
"""
from fractions import Fraction
import json
import math
from pathlib import Path
import unittest

from narrative_dynamics.abm.mesh_neutral_audit_trust import _neutral_source_models
from narrative_dynamics.contracts import stable_content_hash

_PIN = Path(__file__).resolve().parents[1] / (
    "conformance/network_native_trust_channel_ablation_v1.json"
)
_APPROVED = Fraction(18, 25)
_DENIED = Fraction(2, 25)


def rational_posterior(t: Fraction, a: Fraction, b: Fraction) -> Fraction:
    """Independent finite algebraic contract, not an Agent inference runtime."""
    if not isinstance(t, Fraction) or not 0 <= t <= 1:
        raise ValueError("valid bounded source trust required")
    if not isinstance(a, Fraction) or not isinstance(b, Fraction):
        raise ValueError("finite rational source likelihood required")
    if not 0 <= b < a <= 1:
        raise ValueError("declared approved/denied likelihood ordering required")
    yes = 1 - t + t * a
    no = 1 - t + t * b
    if yes + no <= 0:
        raise ValueError("source Bayesian evidence must be positive")
    return yes / (yes + no)


class OriginalV19TrustBayesTheoremBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pinned = json.loads(_PIN.read_text(encoding="utf-8"))
        cls.record = pinned["report"]
        cls.report_hash = pinned["report_hash"]
        memory, _, _ = _neutral_source_models(
            allow_post_audit_target_tell=True,
            trust_weight_direct_testimony=True,
        )
        cls.source_cognitive = memory.cognitive_model
        cls.bob = next(a for a in cls.source_cognitive.agents if a.agent_id == "bob")

    def test_declared_likelihoods_come_from_exact_source_model(self):
        self.assertEqual(
            self.source_cognitive.content_hash,
            self.record["common_source_cognitive_hash"],
        )
        values = {}
        for hypothesis in ("approved", "denied"):
            matching = [
                p.probability for p in self.bob.likelihoods
                if p.action_id == "tell"
                and p.hypothesis_id == hypothesis
                and p.symbol_id == "approved"
            ]
            self.assertEqual(len(matching), 1)
            self.assertTrue(math.isfinite(matching[0]))
            values[hypothesis] = matching[0]
        self.assertTrue(math.isclose(values["approved"], float(_APPROVED),
                                      rel_tol=0, abs_tol=1e-14))
        self.assertTrue(math.isclose(values["denied"], float(_DENIED),
                                      rel_tol=0, abs_tol=1e-14))
        self.assertEqual(self.report_hash, stable_content_hash(self.record))
        self.assertEqual(self.record["source_private_history_match"], "NOT_MATCHED")
        self.assertEqual(self.record["trust_only_causal_effect"], "NOT_IDENTIFIED")

    def test_all_four_real_native_posteriors_match_lean_rational_values(self):
        self.assertEqual(set(self.record["cells"]), {"weighted", "legacy"})
        for history, learned in (
            ("confirmed_audit", Fraction(3, 5)),
            ("contradicted_audit", Fraction(3, 10)),
        ):
            weighted = self.record["cells"]["weighted"]["history_arms"][history]
            legacy = self.record["cells"]["legacy"]["history_arms"][history]
            self.assertEqual(
                weighted["source_target_tell_event_hash"],
                legacy["source_target_tell_event_hash"],
            )
            for protocol, arm, trust in (
                ("weighted", weighted, learned),
                ("legacy", legacy, Fraction(1)),
            ):
                with self.subTest(history=history, protocol=protocol):
                    self.assertEqual(arm["native_target_tell_received_fidelity"], "exact")
                    self.assertTrue(arm["native_admission_consumed"])
                    self.assertEqual(arm["native_feasible_actions"], ["tell", "wait"])
                    self.assertEqual(arm["native_map_action"], "tell")
                    self.assertEqual(arm["native_world_event_kind"], "tell")
                    self.assertTrue(arm["native_world_success"])
                    self.assertEqual(arm["native_bob_direct_prior_approved"], 0.5)
                    self.assertEqual(
                        arm["native_effective_tell_weight"],
                        float(trust),
                    )
                    self.assertTrue(math.isclose(
                        arm["native_bob_direct_posterior_approved"],
                        float(rational_posterior(trust, _APPROVED, _DENIED)),
                        rel_tol=0, abs_tol=1e-14,
                    ))
        self.assertEqual(self.record["within_history_map_diff"],
                         {"confirmed_audit": False, "contradicted_audit": False})

    def test_boundaries_monotonicity_and_uninformative_limit(self):
        expected = (
            (Fraction(0), Fraction(1, 2)),
            (Fraction(3, 10), Fraction(229, 410)),
            (Fraction(3, 5), Fraction(13, 20)),
            (Fraction(1), Fraction(9, 10)),
        )
        for trust, posterior in expected:
            self.assertEqual(rational_posterior(trust, _APPROVED, _DENIED), posterior)
        self.assertEqual(
            rational_posterior(Fraction(1, 2), Fraction(1, 4),
                               Fraction(1, 4) - Fraction(1, 100000)),
            rational_posterior(Fraction(1, 2), Fraction(1, 4),
                               Fraction(1, 4) - Fraction(1, 100000)),
        )
        neutral = Fraction(1, 4)
        actual = (1 - Fraction(3, 5) + Fraction(3, 5) * neutral)
        self.assertEqual(actual / (actual + actual), Fraction(1, 2))

    def test_fail_closed_math_inputs(self):
        for trust in (Fraction(-1, 10), Fraction(11, 10), 0.5,
                      float("nan"), float("inf"), None):
            with self.subTest(trust=trust):
                with self.assertRaises(ValueError):
                    rational_posterior(trust, _APPROVED, _DENIED)
        for a, b in ((Fraction(0), Fraction(0)),
                     (Fraction(1, 10), Fraction(9, 10)),
                     (Fraction(2), Fraction(1, 2))):
            with self.subTest(a=a, b=b):
                with self.assertRaises(ValueError):
                    rational_posterior(Fraction(1), a, b)


if __name__ == "__main__":
    unittest.main()

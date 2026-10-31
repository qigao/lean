"""R2.M5e.2: Bob must have >=2 *real* feasible original V19 actions.

Keep M5e.1 source evidence and full original reward/likelihood contracts.
Do NOT post-hoc tune the reward table, claim isolated trust causality,
or convert threshold classification into an invented world action.
"""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import unittest

from narrative_dynamics.abm.mesh_native_choice_qualification import (
    TwoChoiceSourceActionReport, observe_native_two_choice_source,
)
from narrative_dynamics.abm.mesh_source_history_trust import (
    _run_verified_trust_then_new_tell,
    observe_verified_trust_then_new_tell,
)
from tests.test_network_epistemic_influence import source_epistemic_case


def build_source_native_choice_case() -> TwoChoiceSourceActionReport:
    return observe_native_two_choice_source()


class NativeMultiActionSourceAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_source_native_choice_case()
        cls.data = cls.report.to_dict()

    def test_bob_tell_repeatability_is_the_only_registered_model_change(self):
        original, *_ = source_epistemic_case()
        bob = next(a for a in original.percept_memory_model.cognitive_model.agents
                   if a.agent_id == "bob")
        baseline_tell = next(a for a in bob.actions if a.action_id == "tell")
        self.assertFalse(baseline_tell.repeatable)
        self.assertEqual(baseline_tell.kind.value, "tell")
        self.assertEqual(
            self.data["pre_registered_model_change"],
            "BOB_EXISTING_TELL_REPEATABLE_FALSE_TO_TRUE_ONLY",
        )
        self.assertNotEqual(
            self.data["prior_cognitive_model_hash"],
            self.data["choice_cognitive_model_hash"],
        )
        self.assertNotEqual(
            self.data["prior_social_model_hash"],
            self.data["choice_social_model_hash"],
            "social model embeds the cognitive-model identity",
        )
        self.assertEqual(self.data["native_action_clock"], 10)
        with self.assertRaisesRegex(TypeError, "boolean"):
            _run_verified_trust_then_new_tell(bob_repeatable_tell="true")

    def test_no_regression_of_original_m5e1_history_bytes_and_hash(self):
        source = observe_verified_trust_then_new_tell()
        pin = Path(__file__).resolve().parents[1] / (
            "conformance/network_native_verified_trust_new_tell_v1.json"
        )
        from tools.render_native_verified_trust_new_tell_vectors import render_source_vector
        self.assertEqual(pin.read_bytes(), render_source_vector().encode("utf-8"))
        self.assertEqual(self.data["parent_m5e1_exact_report_hash"],
                         source.content_hash)
        self.assertNotEqual(
            self.data["parent_m5e1_exact_report_hash"],
            self.data["two_choice_source_report_hash"],
        )

    def test_two_genuine_native_choices_with_untuned_action_rewards(self):
        d = self.data
        self.assertEqual(d["source_owner"],
                         "V19_NATIVE_COGNITIVE_SOCIAL_WORLD_AND_PLANNER")
        self.assertTrue(d["native_direct_trust_enabled"])
        self.assertTrue(d["source_bob_rewards_hash"].startswith("sha256:"))
        self.assertEqual(d["source_bob_beta"], 4.0)
        for label, arm in d["arms"].items():
            with self.subTest(label=label):
                self.assertEqual(arm["parent_feasible_actions"], ["wait"])
                self.assertEqual(arm["native_feasible_actions"], ["tell", "wait"])
                self.assertEqual(set(arm["native_action_values"]), {"tell", "wait"})
                self.assertEqual(set(arm["native_action_policy"]), {"tell", "wait"})
                self.assertAlmostEqual(sum(arm["native_action_policy"].values()), 1)
                self.assertEqual(
                    arm["native_selected_action_probability"],
                    arm["native_action_policy"][arm["native_selected_action_id"]],
                )
                self.assertEqual(
                    arm["native_selected_action_value"],
                    arm["native_action_values"][arm["native_selected_action_id"]],
                )
                self.assertTrue(arm["native_world_event_hash"].startswith("sha256:"))
                self.assertTrue(arm["native_decision_hash"].startswith("sha256:"))
                self.assertIsInstance(arm["native_world_event_success"], bool)
                self.assertTrue(arm["native_world_event_outcome"])
                self.assertIn(arm["native_world_event_kind"], ("tell", "wait"))
                self.assertTrue(arm["native_new_tell_accepted"])
                self.assertEqual(
                    arm["new_exact_tell_source_trust"],
                    arm["source_trust_bob_alice"],
                )

    def test_historical_confounds_remain_explicit(self):
        d = self.data
        self.assertEqual(d["historical_context_match"],
                         "NOT_MATCHED_PRIVATE_EVIDENCE_OR_BELIEF")
        self.assertEqual(d["trust_only_causal_effect"], "NOT_IDENTIFIED")
        self.assertEqual(d["sample_size_per_arm"], 1)
        self.assertEqual(d["confidence_interval"], "NOT_ESTIMABLE")
        self.assertEqual(d["external_validation"], "NO_EXTERNAL_VALIDATION")
        self.assertEqual(d["cross_mechanism_ranking"], "FORBIDDEN")
        confirmed = d["arms"]["confirmed_history"]
        contradicted = d["arms"]["contradicted_history"]
        self.assertEqual(confirmed["source_trust_bob_alice"], 0.6)
        self.assertEqual(contradicted["source_trust_bob_alice"], 0.3)
        self.assertNotEqual(confirmed["source_history_story_hash"],
                            contradicted["source_history_story_hash"])
        self.assertEqual(
            d["native_selected_actions_differ"],
            confirmed["native_selected_action_id"] !=
            contradicted["native_selected_action_id"],
        )
        self.assertEqual(
            d["native_action_probabilities_differ"],
            confirmed["native_action_policy"] !=
            contradicted["native_action_policy"],
        )

    def test_report_forbids_forged_actions_rewards_private_content_and_causality(self):
        def rejects(mutation):
            data = self.report.to_dict()
            mutation(data)
            with self.assertRaises((ValueError, TypeError)):
                TwoChoiceSourceActionReport(json.dumps(data, sort_keys=True))
        rejects(lambda d: d.__setitem__("trust_only_causal_effect", "PROVEN"))
        rejects(lambda d: d.__setitem__("sample_size_per_arm", 100))
        rejects(lambda d: d.__setitem__("cross_mechanism_ranking", "ELITE_WIN"))
        rejects(lambda d: d.__setitem__("external_validation", "OBSERVED"))
        rejects(lambda d: d.__setitem__("universal_power_score", 0.95))
        rejects(lambda d: d["arms"]["confirmed_history"].__setitem__(
            "native_feasible_actions", ["wait"]))
        rejects(lambda d: d["arms"]["contradicted_history"].__setitem__(
            "native_action_values", {"tell": 1000.0, "wait": 0.0}))
        rejects(lambda d: d["arms"]["confirmed_history"][
            "native_action_policy"].__setitem__("tell", 0.0))
        rejects(lambda d: d["arms"]["confirmed_history"].__setitem__(
            "new_exact_tell_source_trust", 1.0))
        rejects(lambda d: d["arms"]["confirmed_history"].__setitem__(
            "native_selected_action_probability", 0.0001))
        rejects(lambda d: d["arms"]["contradicted_history"].__setitem__(
            "private_inspect_text", "secret"))
        serialized = json.dumps(self.data)
        self.assertNotIn('"message":', serialized)
        self.assertNotIn('"claim_symbol":', serialized)
        self.assertNotIn("The restructuring is denied.", serialized)
        with self.assertRaises(FrozenInstanceError):
            self.report._canonical_json = "{}"

    def test_exact_reviewed_native_source_bytes(self):
        from tools.render_native_two_choice_vectors import render_source_vector
        path = Path(__file__).resolve().parents[1] / (
            "conformance/network_native_two_choice_v1.json"
        )
        self.assertEqual(path.read_bytes(), render_source_vector().encode("utf-8"))


if __name__ == "__main__":
    unittest.main()

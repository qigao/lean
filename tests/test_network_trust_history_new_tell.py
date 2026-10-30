"""R2.M5e.1: original V19 learned trust before NEW TELL and native action.

Distinct source Story histories may have identical current physical
locations; cognition and old testimony are still confounders. Never
declare a trust-only causal effect or a universal power law.
"""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

from narrative_dynamics.abm.mesh_source_history_trust import (
    SourceHistoryNewTellReport, observe_verified_trust_then_new_tell,
)


def build_history_new_tell_case() -> SourceHistoryNewTellReport:
    return observe_verified_trust_then_new_tell()


class NativeHistoryBeforeNewTellAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_history_new_tell_case()
        cls.data = cls.report.to_dict()

    def test_one_world_model_but_distinct_historical_story_sources(self):
        d = self.data
        self.assertEqual(d["source_owner"],
                         "V19_ONE_WORLD_AND_ORIGINAL_SOCIAL_COGNITION")
        self.assertTrue(d["native_direct_trust_enabled"])
        self.assertTrue(d["same_world_positions_before_native_learning"])
        self.assertEqual(d["single_history_intervention"],
                         "PREVIOUS_SOURCE_TELL_CONTENT_ONLY")
        self.assertEqual(d["source_story_lineage"],
                         "DIVERGENT_HISTORICAL_STORIES")
        self.assertEqual(len(d["authored_common_action_ids"]), 7)
        arms = d["arms"]
        c, i = arms["confirmed_history"], arms["contradicted_history"]
        self.assertEqual(c["historical_tell_event_id"],
                         i["historical_tell_event_id"])
        self.assertNotEqual(c["historical_tell_event_hash"],
                            i["historical_tell_event_hash"])
        self.assertNotEqual(c["historical_story_hash"],
                            i["historical_story_hash"])
        for arm in arms.values():
            self.assertEqual(arm["historical_tell_fidelity_bob"], "exact")

    def test_legitimate_exact_evidence_changes_only_native_directional_trust(self):
        arms = self.data["arms"]
        c = arms["confirmed_history"]
        i = arms["contradicted_history"]
        for arm in (c, i):
            self.assertEqual(arm["source_social_trust_before_native"], 0.5)
            self.assertEqual(arm["history_admitted_testimony"], 1)
            self.assertEqual(arm["history_admitted_private_verification"], 1)
            self.assertTrue(0 <= arm["bob_belief_after_source_history"] <= 1)
        self.assertEqual(c["history_claim_status"], "confirmed")
        self.assertEqual(i["history_claim_status"], "contradicted")
        self.assertEqual(c["source_social_trust_after_learning"], 0.6)
        self.assertEqual(i["source_social_trust_after_learning"], 0.3)
        self.assertNotEqual(c["first_native_story_hash"], i["first_native_story_hash"])

    def test_fresh_tell_is_after_trust_and_is_really_consumed(self):
        for name, arm in self.data["arms"].items():
            with self.subTest(name=name):
                self.assertEqual(arm["native_new_tell_round"], 9)
                self.assertEqual(arm["native_new_tell_fidelity_bob"], "exact")
                self.assertEqual(arm["native_bob_direct_tell_source_trust"],
                                 arm["source_social_trust_after_learning"])
                self.assertTrue(arm["native_bob_source_admitted_new_tell"])
                self.assertEqual(len(arm["native_new_tell_causal_source_ids"]), 1)
                self.assertNotEqual(arm["native_new_tell_causal_source_ids"][0],
                                    arm["native_new_tell_event_id"])
                self.assertTrue(arm["native_new_tell_event_hash"].startswith("sha256:"))
                self.assertTrue(0 <= arm["native_bob_direct_tell_prior_approved"] <= 1)
                self.assertTrue(0 <= arm["native_bob_direct_tell_posterior_approved"] <= 1)
                self.assertTrue(
                    arm["native_bob_direct_tell_admission_id"].startswith("r0009:")
                )

    def test_original_native_round10_acts_but_confounding_not_hidden(self):
        d = self.data
        self.assertEqual(d["post_trust_action_provenance"], "NATIVE_V19_WORLD_EVENT")
        self.assertEqual(d["historical_context_match"],
                         "NOT_MATCHED_PRIVATE_EVIDENCE_OR_BELIEF")
        self.assertEqual(d["trust_only_causal_effect"], "NOT_IDENTIFIED")
        self.assertEqual(d["cross_mechanism_ranking"], "FORBIDDEN")
        self.assertEqual(d["sample_size_per_arm"], 1)
        self.assertEqual(d["confidence_interval"], "NOT_ESTIMABLE")
        self.assertEqual(d["external_validation"], "NO_EXTERNAL_VALIDATION")
        for arm in d["arms"].values():
            self.assertIn(arm["native_bob_selected_action"],
                          arm["native_bob_feasible_actions"])
            self.assertTrue(arm["native_bob_actual_event_hash"].startswith("sha256:"))
            self.assertTrue(arm["native_bob_actual_outcome"])
            self.assertIsInstance(arm["native_bob_actual_success"], bool)
            self.assertTrue(0 <= arm["native_bob_prior_action_belief"] <= 1)
            self.assertTrue(0 <= arm["native_bob_posterior_action_belief"] <= 1)

    def test_native_decision_lacks_choice_so_no_behavior_comparison(self):
        """Genuine source negative: round-10 decision cannot compare choices
        when only WAIT survives the original one-shot action schedule."""
        d = self.data
        self.assertEqual(d["historical_context_match"],
                         "NOT_MATCHED_PRIVATE_EVIDENCE_OR_BELIEF")
        self.assertEqual(d["trust_only_causal_effect"], "NOT_IDENTIFIED")
        c = d["arms"]["confirmed_history"]
        x = d["arms"]["contradicted_history"]
        self.assertAlmostEqual(c["native_bob_direct_tell_posterior_approved"],
                               0.9848288915887558)
        self.assertAlmostEqual(x["native_bob_direct_tell_posterior_approved"],
                               0.9019590565697919)
        for arm in (c, x):
            self.assertEqual(arm["native_new_tell_round"], 9)
            self.assertEqual(arm["native_bob_feasible_actions"], ["wait"])
            self.assertEqual(arm["native_bob_selected_action"], "wait")
            self.assertEqual(arm["native_bob_actual_outcome"], "waited")
            self.assertTrue(arm["native_bob_actual_success"])
        self.assertNotEqual(c["native_bob_direct_tell_source_trust"],
                            x["native_bob_direct_tell_source_trust"])
        # No nontrivial choice was present. Neither a measured action
        # divergence nor a claim of no trust effects is admissible.

    def test_no_synthetic_private_claim_or_trust_causation(self):
        def rejected(change):
            d = self.report.to_dict()
            change(d)
            with self.assertRaises((ValueError, TypeError)):
                SourceHistoryNewTellReport(json.dumps(d, sort_keys=True))
        rejected(lambda d: d.__setitem__("trust_only_causal_effect", "PROVEN"))
        rejected(lambda d: d.__setitem__("external_validation", "ESTABLISHED"))
        rejected(lambda d: d.__setitem__("sample_size_per_arm", 128))
        rejected(lambda d: d.__setitem__("private_INSPECT_text", "secret"))
        rejected(lambda d: d["arms"]["confirmed_history"].__setitem__(
            "history_claim_status", "contradicted"))
        rejected(lambda d: d["arms"]["contradicted_history"].__setitem__(
            "source_social_trust_after_learning", 0.6))
        rejected(lambda d: d["arms"]["confirmed_history"].__setitem__(
            "native_new_tell_round", 8))
        rejected(lambda d: d["arms"]["confirmed_history"].__setitem__(
            "native_bob_direct_tell_source_trust", 1.0))
        rejected(lambda d: d["arms"]["confirmed_history"].__setitem__(
            "private_tell_message", "The restructuring is denied."))
        serialized = json.dumps(self.data)
        self.assertNotIn('"message":', serialized)
        self.assertNotIn('"claim_symbol":', serialized)
        self.assertNotIn("The restructuring is denied.", serialized)
        with self.assertRaises(FrozenInstanceError):
            self.report._canonical_json = "{}"

    def test_source_replay_is_stable_and_requires_reviewed_bytes(self):
        again = build_history_new_tell_case()
        self.assertEqual(self.report.content_hash, again.content_hash)
        self.assertEqual(self.report.to_dict(), again.to_dict())
        from tools.render_native_verified_trust_new_tell_vectors import render_source_vector
        pinned = Path(__file__).resolve().parents[1] / (
            "conformance/network_native_verified_trust_new_tell_v1.json"
        )
        self.assertEqual(pinned.read_bytes(), render_source_vector().encode("utf-8"))


if __name__ == "__main__":
    unittest.main()

"""R2.M5d: observe enacted native actions AFTER source-owned social learning.

This is not a claim of a trust-only causal effect. Both source branches
have different world histories and cognitive evidence, and the original
decision owner may choose the same action in both.
"""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import unittest

from narrative_dynamics.abm.mesh_post_evidence_actions import (
    PostEvidenceActualActionReport, observe_post_evidence_actual_actions,
)
from tests.test_network_epistemic_influence import source_epistemic_case


def build_post_evidence_action_case() -> PostEvidenceActualActionReport:
    model, origin, control, intervention = source_epistemic_case()
    return observe_post_evidence_actual_actions(
        model, origin,
        control_schedule=control,
        intervention_schedule=intervention,
    )


class PostEvidenceActualActionAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_post_evidence_action_case()
        cls.data = cls.report.to_dict()

    def test_native_source_actually_executes_later_bob_action(self):
        d = self.data
        self.assertTrue(d["actual_native_actions_observed"])
        self.assertTrue(d["source_branches_divergent"])
        self.assertEqual(d["source_owner"],
                         "V19_NATIVE_SQLITE_COGNITION_SOCIAL_AND_WORLD")
        self.assertEqual(len(d["same_original_action_ids"]), 5)
        for name, arm in d["branches"].items():
            with self.subTest(branch=name):
                self.assertEqual(arm["post_evidence_action_round"], 8)
                self.assertIn(arm["native_bob_action"],
                              arm["native_bob_feasible_actions"])
                self.assertTrue(arm["native_bob_action_event_id"])
                self.assertTrue(arm["native_bob_action_event_hash"].startswith("sha256:"))
                self.assertIn(arm["native_bob_action_kind"],
                              ("wait", "move", "open_passage", "close_passage",
                               "look", "inspect", "take", "drop", "tell"))
                self.assertIsInstance(arm["native_bob_action_success"], bool)
                self.assertTrue(arm["native_bob_action_outcome"])
                self.assertGreater(arm["native_bob_policy_probability"], 0)
                self.assertLessEqual(arm["native_bob_policy_probability"], 1)
                self.assertNotEqual(
                    arm["source_after_first_cognition_hash"],
                    arm["source_after_post_evidence_action_hash"],
                )
                self.assertTrue(0 <= arm["native_bob_posterior_belief_approved"] <= 1)

    def test_trust_is_observed_from_native_social_owner_before_decision(self):
        d = self.data
        self.assertEqual(d["prior_trust_values"],
                         {"control_wait": 0.5, "authorized_open": 0.3})
        self.assertEqual(
            d["branches"]["control_wait"]
             ["source_trust_bob_alice_after_first_round"], 0.5,
        )
        self.assertEqual(
            d["branches"]["authorized_open"]
             ["source_trust_bob_alice_after_first_round"], 0.3,
        )
        for name, arm in d["branches"].items():
            self.assertEqual(
                arm["new_trust_weighted_tell_recall_count"],
                len(arm["native_bob_recalled_alice_tell"]),
            )
            for admission in arm["native_bob_recalled_alice_tell"]:
                self.assertEqual(admission["source_trust"],
                                 d["prior_trust_values"][name])
                self.assertGreaterEqual(admission["evidence_weight"], 0)
                self.assertLessEqual(admission["evidence_weight"], 1)

    def test_action_difference_is_descriptive_never_trust_only_causation(self):
        d = self.data
        c, i = (d["branches"][k]
                for k in ("control_wait", "authorized_open"))
        self.assertEqual(
            d["selected_action_ids_differ"],
            c["native_bob_action"] != i["native_bob_action"],
        )
        self.assertEqual(
            d["action_kind_or_outcome_differs"],
            (c["native_bob_action_kind"], c["native_bob_action_outcome"]) !=
            (i["native_bob_action_kind"], i["native_bob_action_outcome"]),
        )
        self.assertEqual(d["trust_isolation"],
                         "NOT_ISOLATED_BETWEEN_SOURCE_BRANCHES")
        self.assertEqual(d["post_trust_action_effect"],
                         "TRUST_ONLY_CAUSAL_EFFECT_NOT_ESTABLISHED")
        self.assertEqual(d["causal_attribution"], "UNATTRIBUTED")
        self.assertEqual(d["sample_size_per_arm"], 1)
        self.assertEqual(d["confidence_interval"], "NOT_ESTIMABLE")
        self.assertEqual(d["external_validation"], "NO_EXTERNAL_VALIDATION")
        self.assertEqual(d["cross_mechanism_ranking"], "FORBIDDEN")

    def test_mutations_cannot_fabricate_post_trust_action_or_private_details(self):
        def rejected(change):
            altered = self.report.to_dict()
            change(altered)
            with self.assertRaises((ValueError, TypeError)):
                PostEvidenceActualActionReport(json.dumps(altered, sort_keys=True))
        rejected(lambda d: d.__setitem__("actual_native_actions_observed", False))
        rejected(lambda d: d.__setitem__(
            "post_trust_action_effect", "PROVEN_TRUST_CAUSED_BEHAVIOR"))
        rejected(lambda d: d.__setitem__(
            "trust_isolation", "PERFECT_TRUST_ONLY_EXPERIMENT"))
        rejected(lambda d: d.__setitem__("sample_size_per_arm", 400))
        rejected(lambda d: d.__setitem__("cross_mechanism_ranking", "ELITES_WIN"))
        rejected(lambda d: d.__setitem__("message", "private TELL"))
        rejected(lambda d: d["branches"]["control_wait"]
                 .__setitem__("private_inspection_text", "secret"))
        rejected(lambda d: d["branches"]["authorized_open"]
                 .__setitem__("source_trust_bob_alice_after_first_round", 0.5))
        rejected(lambda d: d["branches"]["control_wait"]
                 .__setitem__("post_evidence_action_round", 7))
        rejected(lambda d: d["branches"]["authorized_open"]
                 .__setitem__("native_bob_action", "not_a_feasible_action"))
        rejected(lambda d: d["branches"]["authorized_open"]
                 .__setitem__("new_trust_weighted_tell_recall_count", 999))
        rejected(lambda d: d.__setitem__(
            "selected_action_ids_differ",
            not d["selected_action_ids_differ"]))
        with self.assertRaises(FrozenInstanceError):
            self.report._canonical_json = "{}"
        encoded = json.dumps(self.data)
        for forbidden in ('"message":', '"claim_symbol":',
                          "The restructuring is denied."):
            self.assertNotIn(forbidden, encoded)

    def test_source_replay_byte_vector_and_repeatability(self):
        again = build_post_evidence_action_case()
        self.assertEqual(self.report.content_hash, again.content_hash)
        self.assertEqual(self.report.to_dict(), again.to_dict())
        from tools.render_native_post_evidence_action_vectors import render_source_vector
        pinned = (Path(__file__).resolve().parents[1] /
                  "conformance/network_post_evidence_actual_actions_v1.json")
        self.assertEqual(pinned.read_bytes(), render_source_vector().encode("utf-8"))

    def test_modified_source_action_schedule_is_rejected(self):
        model, origin, control, intervention = source_epistemic_case()
        mutated = list(intervention)
        mutated[3] = (replace(mutated[3][0], message="Different message"),)
        with self.assertRaisesRegex(ValueError, "more than passage opening"):
            observe_post_evidence_actual_actions(
                model, origin, control_schedule=control,
                intervention_schedule=tuple(mutated),
            )


if __name__ == "__main__":
    unittest.main()

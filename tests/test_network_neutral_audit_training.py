"""R2.M5g0: neutral training source changes Bob→Alice trust, not target belief.

Full original V19 source world, SQLite memory, cognitive/social owner.
A new AUDIT topic is neutral under approved/denied hypotheses while
true target TELL evidence remains informative. This tests a necessary
source-matching prerequisite, NOT a trust-only causal action mechanism.
"""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

from narrative_dynamics.abm.mesh_neutral_audit_trust import (
    NeutralAuditSourceReport,
    _neutral_source_models,
    observe_neutral_training_trust,
)


def build_native_neutral_training_case() -> NeutralAuditSourceReport:
    return observe_neutral_training_trust()


class NativeNeutralAuditTrainingAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_native_neutral_training_case()
        cls.data = cls.report.to_dict()

    def test_same_source_model_and_separate_topics(self):
        d = self.data
        self.assertEqual(
            d["source_owner"],
            "ONE_NATIVE_V19_SQLITE_COGNITION_AND_SOCIAL_OWNER",
        )
        self.assertEqual(d["audit_topic"], "source-audit")
        self.assertEqual(d["target_topic"], "restructuring")
        self.assertTrue(d["source_action_ids_identical"])
        self.assertTrue(d["source_physical_world_snapshots_equal"])
        self.assertTrue(d["target_tell_likelihood_ratio_still_informative"])
        memory, social, _ = _neutral_source_models()
        self.assertEqual(d["source_cognitive_model_hash"],
                         memory.cognitive_model.content_hash)
        self.assertEqual(d["source_memory_model_hash"],
                         memory.content_hash)
        self.assertEqual(d["source_social_model_hash"],
                         social.content_hash)
        self.assertEqual(
            {topic.topic_id for topic in social.topics},
            {"source-audit", "restructuring"},
        )
        self.assertTrue(social.policy.trust_weight_direct_testimony)

    def test_native_likelihood_audit_neutral_target_informative(self):
        d = self.data
        row = d["audit_likelihood_per_target_hypothesis"]
        self.assertEqual(row["approved"], row["denied"])
        self.assertEqual(
            row["approved"],
            {"audit-denied": 0.1, "audit-verified": 0.1},
        )
        memory, _, _ = _neutral_source_models()
        bob = next(a for a in memory.cognitive_model.agents
                   if a.agent_id == "bob")
        likelihoods = {
            (x.action_id, x.hypothesis_id, x.symbol_id): x.probability
            for x in bob.likelihoods
        }
        for action in bob.actions:
            for symbol in ("audit-denied", "audit-verified"):
                self.assertEqual(
                    likelihoods[(action.action_id, "approved", symbol)],
                    likelihoods[(action.action_id, "denied", symbol)],
                )
        self.assertGreater(
            likelihoods[("tell", "approved", "approved")],
            likelihoods[("tell", "denied", "approved")],
        )
        for a in memory.cognitive_model.agents:
            for act in a.actions:
                for h in a.hypotheses:
                    self.assertAlmostEqual(
                        sum(x.probability for x in a.likelihoods
                            if x.action_id == act.action_id
                            and x.hypothesis_id == h.hypothesis_id), 1.0,
                    )

    def test_native_source_owns_confirmation_and_contradiction(self):
        d = self.data
        c = d["arms"]["confirmed_audit"]
        x = d["arms"]["contradicted_audit"]
        self.assertEqual(c["original_social_trust_before"], 0.5)
        self.assertEqual(x["original_social_trust_before"], 0.5)
        self.assertEqual(c["original_social_trust_after"], 0.6)
        self.assertEqual(x["original_social_trust_after"], 0.3)
        self.assertEqual(c["original_social_claim_status"], "confirmed")
        self.assertEqual(x["original_social_claim_status"], "contradicted")
        self.assertEqual(c["source_audit_symbol"], "audit-verified")
        self.assertEqual(x["source_audit_symbol"], "audit-denied")
        for arm in (c, x):
            self.assertEqual(arm["original_social_evidence_testimony_count"], 1)
            self.assertEqual(arm["original_social_evidence_private_verify_count"], 1)
            self.assertEqual(arm["source_bob_prior_approved_belief"], 0.5)
            self.assertEqual(arm["source_bob_posterior_approved_belief"], 0.5)
            self.assertTrue(arm["source_bob_approved_belief_unchanged"])
            self.assertEqual(arm["native_original_action_round"], 8)
            self.assertTrue(arm["bob_original_exact_tell_percept_hash"]
                            .startswith("sha256:"))
            self.assertTrue(arm["bob_own_inspect_event_hash"].startswith("sha256:"))
            self.assertEqual(arm["native_physical_positions"],
                             c["native_physical_positions"])
            self.assertEqual(arm["native_passage_states"],
                             c["native_passage_states"])
        self.assertEqual(c["prior_audit_tell_event_id"],
                         x["prior_audit_tell_event_id"])
        self.assertNotEqual(c["prior_audit_tell_event_hash"],
                            x["prior_audit_tell_event_hash"])
        self.assertNotEqual(c["source_historical_story_hash"],
                            x["source_historical_story_hash"])

    def test_scope_prevents_trust_only_behavior_claims(self):
        d = self.data
        self.assertEqual(d["native_social_learning_round"], 8)
        self.assertEqual(d["original_social_owner"],
                         "V19_ADVANCE_SITUATED_SOCIAL_MEMORY")
        self.assertEqual(d["target_belief_match"],
                         "EXACT_NATIVE_PRIOR_AND_POSTERIOR")
        self.assertEqual(
            d["other_cognitive_inputs_match"],
            "NOT_PROVEN_DIFFERENT_PRIVATE_MEMORIES_AND_CLAIMS",
        )
        self.assertEqual(d["trust_only_causal_effect"], "NOT_IDENTIFIED")
        self.assertEqual(d["post_training_target_action_effect"], "NOT_TESTED")
        self.assertEqual(d["sample_size_per_arm"], 1)
        self.assertEqual(d["confidence_interval"], "NOT_ESTIMABLE")
        self.assertEqual(d["external_validation"], "NO_EXTERNAL_VALIDATION")
        self.assertEqual(d["cross_mechanism_ranking"], "FORBIDDEN")

    def test_source_report_forbids_forged_trust_belief_or_private_content(self):
        def reject(mutation):
            d = self.report.to_dict()
            mutation(d)
            with self.assertRaises((ValueError, TypeError)):
                NeutralAuditSourceReport(json.dumps(d, sort_keys=True))
        reject(lambda d: d.__setitem__("trust_only_causal_effect", "PROVEN"))
        reject(lambda d: d.__setitem__("target_belief_match", "UNVERIFIED"))
        reject(lambda d: d.__setitem__("other_cognitive_inputs_match", "PERFECT"))
        reject(lambda d: d.__setitem__("sample_size_per_arm", 50))
        reject(lambda d: d.__setitem__("population_elite_influence", 0.8))
        reject(lambda d: d["arms"]["confirmed_audit"]
               .__setitem__("original_social_trust_after", 0.9))
        reject(lambda d: d["arms"]["contradicted_audit"]
               .__setitem__("original_social_claim_status", "confirmed"))
        reject(lambda d: d["arms"]["confirmed_audit"]
               .__setitem__("source_bob_posterior_approved_belief", 0.9))
        reject(lambda d: d["arms"]["confirmed_audit"]
               .__setitem__("private_tell_message", "secret"))
        reject(lambda d: d["arms"]["confirmed_audit"]
               .__setitem__("native_original_action_round", 9))
        with self.assertRaises(FrozenInstanceError):
            self.report._canonical_json = "{}"
        emitted = json.dumps(self.data)
        self.assertNotIn('"message":', emitted)
        self.assertNotIn('"claim_symbol":', emitted)
        self.assertNotIn("The audit marker confirms.", emitted)
        self.assertNotIn("The audit marker disputes.", emitted)

    def test_replay_matches_reviewed_utf8_bytes(self):
        from tools.render_native_neutral_audit_vectors import render_source_vector
        path = Path(__file__).resolve().parents[1] / (
            "conformance/network_native_neutral_audit_v1.json"
        )
        self.assertEqual(path.read_bytes(),
                         render_source_vector(self.report).encode("utf-8"))


if __name__ == "__main__":
    unittest.main()

"""R2.M5g.2: complete ORIGINAL V19 two-factor direct TELL ablation.

Two genuine audit source histories × two explicitly versioned ORIGINAL
direct-TELL protocols (source-weighted ON vs legacy full-likelihood OFF).
One original SQLite/cognition/social/world owner; not population
or isolated trust causal inference.
"""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

from narrative_dynamics.abm.mesh_native_trust_ablation import (
    NativeTrustChannelAblationReport,
    observe_native_trust_channel_ablation,
)
from narrative_dynamics.abm.mesh_neutral_audit_trust import _neutral_source_models
from narrative_dynamics.abm.mesh_neutral_target_decision import (
    observe_neutral_audit_then_new_target_choice,
)


def build_native_trust_channel_ablation() -> NativeTrustChannelAblationReport:
    return observe_native_trust_channel_ablation()


class NativeTrustChannelAblationAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=build_native_trust_channel_ablation()
        cls.data=cls.report.to_dict()

    def test_complete_grid_preserves_original_weighted_m5g1_report(self):
        d=self.data
        self.assertEqual(d["grid_design"],
                         "COMPLETE_TWO_FACTOR_NATIVE_PROTOCOL_ABLATION")
        self.assertEqual(d["source_owner"],
                         "ONE_ORIGINAL_V19_SQLITE_COGNITION_SOCIAL_WORLD")
        self.assertTrue(d["complete_factorial"])
        self.assertEqual(set(d["cells"]),{"weighted","legacy"})
        for protocol in ("weighted","legacy"):
            self.assertEqual(
                set(d["cells"][protocol]["history_arms"]),
                {"confirmed_audit","contradicted_audit"},
            )
        original=observe_neutral_audit_then_new_target_choice()
        self.assertEqual(d["parent_m5g1_report_hash"],original.content_hash)
        from tools.render_native_neutral_target_vectors import render_source_vector
        pinned=Path(__file__).resolve().parents[1]/(
            "conformance/network_native_neutral_target_v1.json"
        )
        self.assertEqual(pinned.read_bytes(),render_source_vector(original).encode("utf-8"))

    def test_two_protocol_versions_share_cognition_not_social_policy_hash(self):
        d=self.data
        _, weighted, _ = _neutral_source_models(
            allow_post_audit_target_tell=True,
            trust_weight_direct_testimony=True,
        )
        _, legacy, _ = _neutral_source_models(
            allow_post_audit_target_tell=True,
            trust_weight_direct_testimony=False,
        )
        self.assertTrue(weighted.policy.trust_weight_direct_testimony)
        self.assertFalse(legacy.policy.trust_weight_direct_testimony)
        self.assertEqual(
            weighted.memory_cognitive_model.cognitive_model,
            legacy.memory_cognitive_model.cognitive_model,
        )
        self.assertNotEqual(weighted.content_hash,legacy.content_hash)
        self.assertNotEqual(weighted.policy.content_hash,legacy.policy.content_hash)
        self.assertEqual(
            d["cells"]["weighted"]["source_social_model_hash"],
            weighted.content_hash,
        )
        self.assertEqual(
            d["cells"]["legacy"]["source_social_model_hash"],
            legacy.content_hash,
        )
        self.assertEqual(
            d["cells"]["weighted"]["source_model_hash"],
            d["cells"]["legacy"]["source_model_hash"],
        )
        self.assertTrue(d["same_v19_world_and_model_cognition"])
        with self.assertRaisesRegex(TypeError,"boolean"):
            _neutral_source_models(trust_weight_direct_testimony="false")

    def test_native_social_trust_learns_even_when_direct_tell_weighting_off(self):
        d=self.data
        for history,trust in (("confirmed_audit",0.6),
                              ("contradicted_audit",0.3)):
            for policy,effective in (("weighted",trust),("legacy",1.0)):
                with self.subTest(history=history,protocol=policy):
                    arm=d["cells"][policy]["history_arms"][history]
                    self.assertEqual(
                        arm["learned_observer_source_trust"],trust,
                    )
                    self.assertEqual(
                        arm["native_effective_tell_weight"],effective,
                    )
                    self.assertTrue(arm["native_admission_consumed"])
                    self.assertEqual(
                        arm["native_target_tell_received_fidelity"],"exact",
                    )
                    self.assertEqual(
                        arm["native_bob_direct_prior_approved"],0.5,
                    )
                    self.assertEqual(arm["native_bob_prior_approved"],0.5)
                    self.assertIn(
                        arm["native_map_action"],arm["native_feasible_actions"],
                    )
                    self.assertTrue(arm["native_world_event_hash"].startswith("sha256:"))
                    self.assertTrue(arm["native_world_outcome"])
                    self.assertIsInstance(arm["native_world_success"],bool)

    def test_original_same_history_qualified_native_event_across_protocols(self):
        d=self.data
        for history in ("confirmed_audit","contradicted_audit"):
            a=d["cells"]["weighted"]["history_arms"][history]
            b=d["cells"]["legacy"]["history_arms"][history]
            for key in (
                "training_story_hash",
                "training_audit_event_hash",
                "source_target_tell_event_id",
                "source_target_tell_event_hash",
                "source_target_tell_cause_event_ids",
                "post_learning_story_hash",
                "native_feasible_actions",
                "prior_selected_action_hash",
                "processed_observation_ids_hash",
            ):
                self.assertEqual(a[key],b[key],f"{key} changed across native protocol")
            self.assertEqual(a["native_feasible_actions"],["tell","wait"])
            self.assertAlmostEqual(
                sum(a["native_action_policy"].values()),1.0,
            )
            self.assertAlmostEqual(
                sum(b["native_action_policy"].values()),1.0,
            )
            self.assertEqual(a["native_world_event_kind"],a["native_map_action"])
            self.assertEqual(b["native_world_event_kind"],b["native_map_action"])
            self.assertGreater(
                b["native_bob_direct_posterior_approved"],
                a["native_bob_direct_posterior_approved"],
            )
            self.assertTrue(d["within_history_posterior_diff"][history])
            self.assertEqual(
                d["within_history_map_diff"][history],
                a["native_map_action"]!=b["native_map_action"],
            )

    def test_unchanged_private_histories_do_not_create_population_causality(self):
        d=self.data
        self.assertEqual(d["source_private_history_match"],"NOT_MATCHED")
        self.assertEqual(d["trust_only_causal_effect"],"NOT_IDENTIFIED")
        self.assertEqual(d["native_protocol_mechanism_scope"],
                         "FINITE_SOFTWARE_POLICY_ABLATION")
        self.assertEqual(d["sample_size_per_arm"],1)
        self.assertEqual(d["confidence_interval"],"NOT_ESTIMABLE")
        self.assertEqual(d["external_validation"],"NO_EXTERNAL_VALIDATION")
        self.assertEqual(d["cross_mechanism_ranking"],"FORBIDDEN")
        self.assertEqual(d["source_bob_beta"],4.0)
        self.assertTrue(d["common_original_rewards_hash"].startswith("sha256:"))

    def test_invalid_source_and_made_up_effects_fail_closed(self):
        def reject(mutation):
            d=self.report.to_dict()
            mutation(d)
            with self.assertRaises((ValueError,TypeError)):
                NativeTrustChannelAblationReport(json.dumps(d,sort_keys=True))
        reject(lambda d:d["cells"].pop("weighted"))
        reject(lambda d:d["cells"]["legacy"].__setitem__(
            "trust_weight_direct_testimony",True))
        reject(lambda d:d.__setitem__("grid_design","BLIND_CAUSAL_PROOF"))
        reject(lambda d:d.__setitem__("trust_only_causal_effect","PROVEN"))
        reject(lambda d:d.__setitem__("sample_size_per_arm",50))
        reject(lambda d:d.__setitem__("universal_elite_score",0.95))
        reject(lambda d:d["cells"]["legacy"]["history_arms"]["confirmed_audit"]
               .__setitem__("native_effective_tell_weight",0.6))
        reject(lambda d:d["cells"]["weighted"]["history_arms"]["confirmed_audit"]
               .__setitem__("native_bob_direct_posterior_approved",0.9))
        reject(lambda d:d["cells"]["legacy"]["history_arms"]["confirmed_audit"]
               .__setitem__("training_story_hash","sha256:"+"a"*64))
        reject(lambda d:d["cells"]["weighted"]["history_arms"]["confirmed_audit"]
               ["native_action_values"].__setitem__("tell",1000.0))
        reject(lambda d:d["cells"]["weighted"]["history_arms"]["confirmed_audit"]
               ["native_action_policy"].__setitem__("tell",0.0))
        reject(lambda d:d["cells"]["weighted"]["history_arms"]["confirmed_audit"]
               .__setitem__("private_inspection_body","secret"))
        reject(lambda d:d["within_history_posterior_diff"].__setitem__(
            "confirmed_audit",False))
        with self.assertRaises(FrozenInstanceError):
            self.report._canonical_json="{}"
        data=json.dumps(self.data)
        for forbidden in ('"message":','"claim_symbol":',
                          "The audit marker disputes.",
                          "The audit marker confirms.",
                          "The restructuring is approved."):
            self.assertNotIn(forbidden,data)

    def test_source_generated_exact_review_bytes(self):
        from tools.render_native_trust_channel_ablation_vectors import render_source_vector
        pinned=Path(__file__).resolve().parents[1]/(
            "conformance/network_native_trust_channel_ablation_v1.json"
        )
        self.assertEqual(pinned.read_bytes(),
                         render_source_vector(self.report).encode("utf-8"))


if __name__=="__main__":
    unittest.main()

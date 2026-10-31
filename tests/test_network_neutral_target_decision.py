"""R2.M5g1: neutral audit → later source-target TELL → native Bob action.

The initial approved/denied target belief is exactly matched after
source-audit training. Source histories, memories and subsequently
observed native events are NOT declared globally matched.
"""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import unittest

from narrative_dynamics.abm.mesh_neutral_audit_trust import (
    _neutral_source_models,
)
from narrative_dynamics.abm.mesh_neutral_target_decision import (
    NeutralAuditTargetDecisionReport,
    observe_neutral_audit_then_new_target_choice,
)


def build_native_neutral_target_choice() -> NeutralAuditTargetDecisionReport:
    return observe_neutral_audit_then_new_target_choice()


class NativeNeutralTargetDecisionAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_native_neutral_target_choice()
        cls.data = cls.report.to_dict()

    def test_one_new_native_model_shared_world_and_reward_integrity(self):
        d=self.data
        self.assertEqual(
            d["registered_model_change"],
            "ALICE_AND_BOB_EXISTING_TELL_REPEATABLE_TRUE_ONLY",
        )
        self.assertEqual(
            d["source_owner"],"V19_ONE_SQLITE_SOCIAL_COGNITIVE_WORLD_OWNER",
        )
        self.assertTrue(d["original_reward_policy_preserved"])
        self.assertTrue(d["training_source_symbol_neutral"])
        self.assertTrue(d["same_world_and_action_roster"])
        self.assertTrue(d["same_declared_target_message"])
        old,_,_=_neutral_source_models()
        new,_,_=_neutral_source_models(allow_post_audit_target_tell=True)
        self.assertNotEqual(
            old.cognitive_model.content_hash,new.cognitive_model.content_hash,
        )
        for a,b in zip(old.cognitive_model.agents,new.cognitive_model.agents,
                       strict=True):
            self.assertEqual(a.agent_id,b.agent_id)
            self.assertEqual(a.rewards,b.rewards)
            self.assertEqual(a.likelihoods,b.likelihoods)
            self.assertEqual(a.goals,b.goals)
            self.assertEqual(a.action_schedule,b.action_schedule)
            for old_action,new_action in zip(a.actions,b.actions,strict=True):
                self.assertEqual(
                    replace(new_action,repeatable=old_action.repeatable),
                    old_action,
                )
        self.assertEqual(d["source_cognitive_model_hash"],
                         new.cognitive_model.content_hash)
        with self.assertRaisesRegex(TypeError,"boolean"):
            _neutral_source_models(allow_post_audit_target_tell="true")

    def test_exact_native_trust_learning_before_a_new_target_tell(self):
        d=self.data
        self.assertEqual(d["native_social_learning_round"],8)
        self.assertEqual(d["new_target_tell_round"],9)
        self.assertEqual(d["native_bob_decision_round"],10)
        self.assertEqual(
            d["training_target_belief_match"],"EXACT_AT_CHECKPOINT_0_5",
        )
        a=d["arms"]["confirmed_audit"]
        b=d["arms"]["contradicted_audit"]
        self.assertEqual(a["training_source_claim_status"],"confirmed")
        self.assertEqual(b["training_source_claim_status"],"contradicted")
        self.assertEqual(a["training_bob_trust_alice_after"],0.6)
        self.assertEqual(b["training_bob_trust_alice_after"],0.3)
        self.assertNotEqual(a["training_story_hash"],b["training_story_hash"])
        for arm,trust in ((a,0.6),(b,0.3)):
            self.assertEqual(arm["training_bob_trust_alice_before"],0.5)
            self.assertEqual(arm["training_bob_belief_approved"],0.5)
            self.assertTrue(arm["new_target_tell_event_id"].startswith("r0009:"))
            self.assertEqual(arm["new_target_tell_fidelity_bob"],"exact")
            self.assertEqual(arm["native_bob_direct_tell_trust"],trust)
            self.assertTrue(arm["new_target_tell_source_observed_by_alice"])
            self.assertEqual(len(arm["new_target_tell_cause_event_ids"]),1)
            self.assertTrue(arm["native_bob_admission_consumed_by_decision"])
            self.assertTrue(arm["native_bob_direct_admission_id"].startswith("r0009:"))

    def test_original_v19_two_way_choice_actuation_and_bounded_policy(self):
        d=self.data
        for name,arm in d["arms"].items():
            with self.subTest(name=name):
                self.assertGreaterEqual(len(arm["native_bob_feasible_actions"]),2)
                self.assertIn("tell",arm["native_bob_feasible_actions"])
                self.assertIn("wait",arm["native_bob_feasible_actions"])
                self.assertEqual(
                    set(arm["native_bob_feasible_actions"]),
                    set(arm["native_bob_action_values"]),
                )
                self.assertEqual(
                    set(arm["native_bob_feasible_actions"]),
                    set(arm["native_bob_action_policy"]),
                )
                self.assertAlmostEqual(
                    sum(arm["native_bob_action_policy"].values()),1.0,
                )
                self.assertIn(arm["native_bob_selected_action"],
                              arm["native_bob_feasible_actions"])
                self.assertEqual(arm["native_bob_world_event_kind"],
                                 arm["native_bob_selected_action"])
                self.assertIsInstance(arm["native_bob_world_event_success"],bool)
                self.assertTrue(arm["native_bob_world_event_outcome"])
                self.assertTrue(arm["native_bob_decision_hash"].startswith("sha256:"))

    def test_other_source_decision_inputs_must_be_disclosed_not_assumed(self):
        d=self.data
        self.assertEqual(d["source_private_history_match"],"NOT_MATCHED")
        self.assertEqual(d["trust_only_causal_effect"],"NOT_IDENTIFIED")
        self.assertEqual(d["sample_size_per_arm"],1)
        self.assertEqual(d["confidence_interval"],"NOT_ESTIMABLE")
        self.assertEqual(d["external_validation"],"NO_EXTERNAL_VALIDATION")
        self.assertEqual(d["cross_mechanism_ranking"],"FORBIDDEN")
        arms=d["arms"]
        yes,no=arms["confirmed_audit"],arms["contradicted_audit"]
        current_equal=(yes["native_bob_pre_target_approved"]==
                       no["native_bob_pre_target_approved"])
        self.assertEqual(
            d["pre_new_target_belief_match"],
            "EXACT" if current_equal else
            "NOT_MATCHED_AFTER_NATIVE_INTERMEDIATE_EVENTS",
        )
        self.assertEqual(
            d["actual_action_comparison"],
            "DIFFERENT_MAP" if yes["native_bob_selected_action"]!=
                              no["native_bob_selected_action"] else "SAME_MAP",
        )

    def test_real_neutral_pre_target_equal_belief_then_weighted_different_posterior(self):
        """Original native result: comparable target belief *before*
        round-10 message, divergent source trust, and same enacted TELL.
        Private memory/claim histories remain NOT_MATCHED."""
        d=self.data
        a=d["arms"]["confirmed_audit"]
        b=d["arms"]["contradicted_audit"]
        self.assertEqual(d["pre_new_target_belief_match"],"EXACT")
        self.assertEqual(d["actual_action_comparison"],"SAME_MAP")
        self.assertEqual(a["native_bob_pre_target_approved"],0.5)
        self.assertEqual(b["native_bob_pre_target_approved"],0.5)
        self.assertEqual(a["native_bob_direct_prior_approved"],0.5)
        self.assertEqual(b["native_bob_direct_prior_approved"],0.5)
        self.assertAlmostEqual(a["native_bob_direct_posterior_approved"],0.65)
        self.assertAlmostEqual(b["native_bob_direct_posterior_approved"],
                               0.5585365853658536)
        self.assertEqual(a["native_bob_direct_tell_trust"],0.6)
        self.assertEqual(b["native_bob_direct_tell_trust"],0.3)
        self.assertEqual(a["new_target_tell_event_hash"],
                         b["new_target_tell_event_hash"])
        self.assertEqual(a["native_bob_feasible_actions"],["tell","wait"])
        self.assertEqual(b["native_bob_feasible_actions"],["tell","wait"])
        self.assertEqual(a["native_bob_selected_action"],"tell")
        self.assertEqual(b["native_bob_selected_action"],"tell")
        self.assertEqual(a["native_bob_world_event_outcome"],"told")
        self.assertEqual(b["native_bob_world_event_outcome"],"told")
        self.assertTrue(a["native_bob_world_event_success"])
        self.assertTrue(b["native_bob_world_event_success"])
        self.assertEqual(a["native_bob_recalled_memory_count"],2)
        self.assertEqual(b["native_bob_recalled_memory_count"],2)
        self.assertEqual(
            a["native_bob_previous_selected_actions_hash"],
            b["native_bob_previous_selected_actions_hash"],
        )
        self.assertEqual(
            a["native_bob_recent_observation_ids_hash"],
            b["native_bob_recent_observation_ids_hash"],
        )
        self.assertEqual(d["source_private_history_match"],"NOT_MATCHED")
        self.assertEqual(d["trust_only_causal_effect"],"NOT_IDENTIFIED")

    def test_private_source_and_forged_trust_behaviour_fail_closed(self):
        def reject(mutate):
            x=self.report.to_dict()
            mutate(x)
            with self.assertRaises((ValueError,TypeError)):
                NeutralAuditTargetDecisionReport(json.dumps(x,sort_keys=True))
        reject(lambda d:d.__setitem__("trust_only_causal_effect","PROVEN"))
        reject(lambda d:d.__setitem__("sample_size_per_arm",100))
        reject(lambda d:d.__setitem__("external_validation","ESTABLISHED"))
        reject(lambda d:d.__setitem__("source_private_history_match","MATCHED"))
        reject(lambda d:d.__setitem__("universal_elite_score",0.99))
        reject(lambda d:d["arms"]["confirmed_audit"].__setitem__(
            "native_bob_direct_tell_trust",1.0))
        reject(lambda d:d["arms"]["contradicted_audit"].__setitem__(
            "training_bob_belief_approved",0.7))
        reject(lambda d:d["arms"]["confirmed_audit"].__setitem__(
            "native_bob_feasible_actions",["wait"]))
        reject(lambda d:d["arms"]["confirmed_audit"].__setitem__(
            "native_bob_action_policy",{"wait":1.0}))
        reject(lambda d:d["arms"]["confirmed_audit"].__setitem__(
            "private_inspect_text","private"))
        with self.assertRaises(FrozenInstanceError):
            self.report._canonical_json="{}"
        raw=json.dumps(self.data)
        self.assertNotIn('"message":',raw)
        self.assertNotIn('"claim_symbol":',raw)
        self.assertNotIn("The audit marker confirms.",raw)
        self.assertNotIn("The audit marker disputes.",raw)

    def test_native_replay_equals_reviewed_utf8_source_vector(self):
        from tools.render_native_neutral_target_vectors import render_source_vector
        path=(Path(__file__).resolve().parents[1]/
              "conformance/network_native_neutral_target_v1.json")
        self.assertEqual(
            path.read_bytes(),render_source_vector(self.report).encode("utf-8"),
        )


if __name__=="__main__":
    unittest.main()

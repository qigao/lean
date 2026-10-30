"""R2.M5e.0: source trust applies to native *direct* external testimony.

The model's social policy opts in explicitly. No second Bayesian updater,
untrusted direct testimony as fact, historical trust mutation or empirical
inference is hidden in the experiment.
"""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import unittest

from narrative_dynamics.abm.mesh_direct_tell_trust import (
    DirectTellSourceTrustReport, observe_direct_tell_source_trust,
)
from narrative_dynamics.abm.situated_percept_cognition import (
    admit_situated_percepts,
)
from tests.test_network_epistemic_influence import source_epistemic_case


def build_direct_tell_trust_case() -> DirectTellSourceTrustReport:
    return observe_direct_tell_source_trust()


class DirectTellSourceTrustAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_direct_tell_trust_case()
        cls.data = cls.report.to_dict()

    def test_policy_optin_preserves_legacy_source_model_bytes(self):
        net, _, _, _ = source_epistemic_case()
        original_policy = net.social_memory_model.policy
        self.assertFalse(original_policy.trust_weight_direct_testimony)
        self.assertNotIn("trust_weight_direct_testimony", original_policy.to_dict())
        enabled = replace(original_policy, trust_weight_direct_testimony=True)
        self.assertNotEqual(original_policy.content_hash, enabled.content_hash)
        self.assertEqual(enabled.to_dict()["trust_weight_direct_testimony"], True)
        with self.assertRaisesRegex(TypeError, "must be boolean"):
            replace(original_policy, trust_weight_direct_testimony="true")
        self.assertEqual(self.data["source_policy_variants"],
                         ["legacy_unweighted", "weighted_zero",
                          "weighted_half", "weighted_one"])
        self.assertEqual(
            self.data["arms"]["legacy_unweighted"]["policy_hash"],
            original_policy.content_hash,
        )

    def test_one_real_exact_tell_untempered_at_legacy_and_unit_trust(self):
        d = self.data
        arms = d["arms"]
        self.assertEqual(d["source_owner"],
                         "V19_SINGLE_DIRECT_COGNITION_AND_SOCIAL_OWNER")
        self.assertEqual(d["source_direct_percept_fidelity"], "exact")
        self.assertEqual(d["source_trust_direction"], "BOB_OBSERVES_ALICE")
        self.assertEqual(d["source_world_branch_comparison"],
                         "SAME_WORLD_STORY_DIFFERENT_SOCIAL_POLICY")
        self.assertEqual(
            {arm["native_source_tell_event_hash"] for arm in arms.values()},
            {d["source_tell_event_hash"]},
        )
        self.assertFalse(arms["legacy_unweighted"]["weight_direct_tell"])
        self.assertTrue(arms["weighted_half"]["weight_direct_tell"])
        self.assertEqual(arms["legacy_unweighted"]
                         ["actual_effective_direct_tell_trust"], 1.0)
        self.assertEqual(arms["weighted_one"]
                         ["actual_effective_direct_tell_trust"], 1.0)
        self.assertEqual(arms["weighted_half"]
                         ["actual_effective_direct_tell_trust"], 0.5)
        self.assertEqual(arms["weighted_zero"]
                         ["actual_effective_direct_tell_trust"], 0.0)

    def test_source_trust_tempering_changes_real_percept_posterior(self):
        a = self.data["arms"]
        legacy = a["legacy_unweighted"]["direct_tell_posterior_approved"]
        zero = a["weighted_zero"]["direct_tell_posterior_approved"]
        half = a["weighted_half"]["direct_tell_posterior_approved"]
        one = a["weighted_one"]["direct_tell_posterior_approved"]
        self.assertEqual(legacy, one)
        self.assertEqual(zero, a["weighted_zero"]["direct_tell_prior_approved"])
        self.assertLess(zero, half)
        self.assertLess(half, legacy)
        for arm in a.values():
            self.assertEqual(arm["native_new_tell_round"], 4)
            self.assertTrue(arm["native_tell_admitted_to_bob_decision"])
            self.assertTrue(all(w == 1.0 for w in
                                arm["native_non_tell_direct_weights"]))
            self.assertEqual(arm["native_bob_admitted_testimony"], 1)

    def test_same_exact_world_native_sqlite_choice_uses_direct_policy(self):
        arms = self.data["arms"]
        self.assertEqual(self.data["native_decision_rule"],
                         "ORIGINAL_V19_SOFTMAX_MAP")
        for name, row in arms.items():
            with self.subTest(name=name):
                self.assertTrue(row["native_bob_decision_hash"].startswith("sha256:"))
                self.assertIn(row["native_bob_selected_action"],
                              row["native_bob_feasible_actions"])
                self.assertTrue(row["native_bob_event_hash"].startswith("sha256:"))
                self.assertIsInstance(row["native_bob_event_success"], bool)
                self.assertTrue(0 <= row["native_bob_posterior_approved"] <= 1)
                self.assertTrue(row["native_bob_event_outcome"])
        self.assertGreater(len(set(
            row["native_bob_posterior_approved"] for row in arms.values()
        )), 1)

    def test_direct_trust_map_must_be_typed_and_cannot_expose_private_content(self):
        net, _, _, _ = source_epistemic_case()
        from narrative_dynamics.abm.situated_percept_memory_cognition import (
            initialize_situated_percept_memory_cognition,
        )
        from narrative_dynamics.abm.situated_story import replay_situated_story
        from narrative_dynamics.abm.situated_percept_cognition import perceptual_timeline
        _, origin, _, intervention = source_epistemic_case()
        memory = net.percept_memory_model
        story = replay_situated_story(
            memory.cognitive_model.world_model,
            origin.initial_state, intervention,
            perception_model=memory.perception_model,
        )
        cognition = initialize_situated_percept_memory_cognition(memory, story)
        agent = next(x for x in memory.cognitive_model.agents
                     if x.agent_id == "bob")
        mind = next(x for x in cognition.minds if x.agent_id == "bob")
        percepts = perceptual_timeline(memory.perception_model, story, "bob")
        for bad in ({"alice": -0.1}, {"alice": 1.1}, {"alice": float("nan")},
                    {"alice": True}, {"": 0.5}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    admit_situated_percepts(
                        agent, mind, percepts,
                        _source_trust_by_source=bad,
                    )
        raw = json.dumps(self.data, sort_keys=True)
        self.assertNotIn('"message":', raw)
        self.assertNotIn('"claim_symbol":', raw)
        self.assertNotIn("The restructuring is denied.", raw)
        self.assertNotIn('"private_inspect_content"', raw)

    def test_immutable_source_anti_forgery_and_causal_firewall(self):
        def reject(change):
            d = self.report.to_dict()
            change(d)
            with self.assertRaises((ValueError, TypeError)):
                DirectTellSourceTrustReport(json.dumps(d, sort_keys=True))
        reject(lambda d: d.__setitem__("trust_only_causal_claim", "PROVEN"))
        reject(lambda d: d.__setitem__("sample_size_per_arm", 50))
        reject(lambda d: d.__setitem__("external_validation", "ESTABLISHED"))
        reject(lambda d: d.__setitem__("universal_leader_score", 0.97))
        reject(lambda d: d["arms"]["weighted_half"].__setitem__(
            "actual_effective_direct_tell_trust", 1.0))
        reject(lambda d: d["arms"]["weighted_zero"].__setitem__(
            "direct_tell_posterior_approved", 0.999))
        reject(lambda d: d["arms"]["legacy_unweighted"].__setitem__(
            "weight_direct_tell", True))
        reject(lambda d: d["arms"]["weighted_half"].__setitem__(
            "private_tell_message", "secret"))
        with self.assertRaises(FrozenInstanceError):
            self.report._canonical_json = "{}"

    def test_source_generated_review_vector_is_exact_bytes(self):
        from tools.render_native_direct_tell_trust_vectors import render_source_vector
        pinned = (Path(__file__).resolve().parents[1] /
                  "conformance/network_native_direct_tell_trust_v1.json")
        self.assertEqual(pinned.read_bytes(), render_source_vector().encode("utf-8"))


if __name__ == "__main__":
    unittest.main()

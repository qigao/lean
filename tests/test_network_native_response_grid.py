"""R2.M5f: disclose the FULL exploratory native V19 response cost grid.

This is a model-version sensitivity study chosen after the M5e2 baseline,
NOT preregistered blind experimental causal inference. Native actions,
action values, world events and exact source IDs come only from V19.
"""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

from narrative_dynamics.abm.mesh_response_cost_grid import (
    NativeResponseGridReport, observe_native_speaking_cost_grid,
    _COSTS,
)
from narrative_dynamics.abm.mesh_source_history_trust import (
    _run_verified_trust_then_new_tell,
)


class SourceResponseRegimeGridAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = observe_native_speaking_cost_grid()
        cls.data = cls.report.to_dict()

    def test_complete_registered_exploratory_grid_includes_null_results(self):
        d = self.data
        self.assertEqual(d["source_owner"],
                         "V19_NATIVE_PLANNER_AND_SQLITE_SOCIAL_COGNITION")
        self.assertEqual(d["grid_design"],
                         "EXPLORATORY_SELECTED_AFTER_M5E2_BASELINE")
        self.assertEqual(d["grid_costs"], [0.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(len(d["cells"]), len(_COSTS))
        self.assertEqual(
            [cell["speaking_cost"] for cell in d["cells"]],
            list(_COSTS),
        )
        self.assertEqual(
            len({cell["source_cognitive_model_hash"] for cell in d["cells"]}),
            len(_COSTS),
            "each incentive level has its OWN model revision",
        )
        discovered = [
            cell["speaking_cost"] for cell in d["cells"]
            if cell["native_selected_actions_differ"]
        ]
        self.assertEqual(d["discovered_switch_costs"], discovered)
        self.assertEqual(d["grid_completeness"], "ALL_REGISTERED_CELLS_INCLUDED")

    def test_every_cell_has_two_actually_feasible_original_native_actions(self):
        d = self.data
        self.assertEqual(d["source_bob_beta"], 4.0)
        self.assertEqual(d["registered_model_change"],
                         "SYMMETRIC_BOB_TELL_REWARD_MINUS_COST_EACH_HYPOTHESIS")
        for cell in d["cells"]:
            with self.subTest(cost=cell["speaking_cost"]):
                self.assertEqual(
                    cell["same_original_world_hash"],
                    d["source_world_hash"],
                )
                self.assertEqual(
                    cell["same_original_perception_hash"],
                    d["source_perception_hash"],
                )
                self.assertEqual(set(cell["history_arms"]),
                                 {"confirmed_history", "contradicted_history"})
                for name, trust in (("confirmed_history", 0.6),
                                    ("contradicted_history", 0.3)):
                    arm = cell["history_arms"][name]
                    self.assertEqual(arm["native_bob_learned_trust"], trust)
                    self.assertEqual(arm["native_action_feasible_ids"],
                                     ["tell", "wait"])
                    self.assertEqual(set(arm["native_action_values"]),
                                     {"tell", "wait"})
                    self.assertEqual(set(arm["native_action_policy"]),
                                     {"tell", "wait"})
                    self.assertAlmostEqual(
                        sum(arm["native_action_policy"].values()), 1,
                    )
                    self.assertEqual(
                        arm["native_selected_action_probability"],
                        arm["native_action_policy"][
                            arm["native_selected_action_id"]],
                    )
                    self.assertTrue(arm["native_decision_hash"].startswith("sha256:"))
                    self.assertTrue(arm["native_world_event_hash"].startswith("sha256:"))
                    self.assertTrue(arm["native_world_event_outcome"])
                    self.assertIsInstance(arm["native_world_event_success"], bool)
                    self.assertEqual(
                        arm["native_world_event_kind"],
                        arm["native_selected_action_id"],
                    )
                    self.assertEqual(arm["native_clock_round"], 10)
                    self.assertTrue(arm["new_source_tell_admitted"])

    def test_zero_cost_cell_matches_original_m5e2_reviewed_source(self):
        d = self.data
        self.assertEqual(
            d["cells"][0]["native_history_report_hash"],
            d["baseline_m5e2_source_report_hash"],
        )
        self.assertTrue(d["parent_m5e2_report_hash"].startswith("sha256:"))
        self.assertTrue(d["original_bob_reward_table_hash"].startswith("sha256:"))
        self.assertEqual(d["grid_costs"][0], 0.0)

    def test_no_cross_model_population_or_trust_only_inference(self):
        d = self.data
        self.assertEqual(d["source_history_comparison"],
                         "CONFIRMED_VS_CONTRADICTED")
        self.assertEqual(d["scientific_scope"],
                         "FINITE_MODEL_SENSITIVITY_NOT_CAUSAL_INFERENCE")
        self.assertEqual(d["trust_only_causal_effect"], "NOT_IDENTIFIED")
        self.assertEqual(d["sample_size_per_arm"], 1)
        self.assertEqual(d["confidence_interval"], "NOT_ESTIMABLE")
        self.assertEqual(d["external_validation"], "NO_EXTERNAL_VALIDATION")
        self.assertEqual(d["cross_mechanism_ranking"], "FORBIDDEN")

    def test_invalid_costs_fail_before_any_native_world_replay(self):
        for bad_cost in (-1, 5.01, float("nan"), True, "2"):
            with self.subTest(cost=bad_cost):
                with self.assertRaises(ValueError):
                    _run_verified_trust_then_new_tell(
                        bob_repeatable_tell=True, bob_tell_cost=bad_cost,
                    )
        with self.assertRaisesRegex(ValueError, "repeatable Bob TELL"):
            _run_verified_trust_then_new_tell(
                bob_repeatable_tell=False, bob_tell_cost=2.0,
            )

    def test_historical_story_source_identical_across_all_cost_models(self):
        cells = self.data["cells"]
        for arm_name in ("confirmed_history", "contradicted_history"):
            hashes = {
                c["history_arms"][arm_name]["prior_source_history_hash"]
                for c in cells
            }
            self.assertEqual(
                len(hashes), 1,
                "the historical world Story cannot depend on Bob's future TELL cost",
            )

    def test_wrong_grid_private_content_and_forged_metrics_fail_closed(self):
        def reject(mutate):
            data = self.report.to_dict()
            mutate(data)
            with self.assertRaises((ValueError, TypeError)):
                NativeResponseGridReport(json.dumps(data, sort_keys=True))
        reject(lambda d: d["cells"].pop())
        reject(lambda d: d["grid_costs"].append(6.0))
        reject(lambda d: d["cells"][0].__setitem__("speaking_cost", 4.0))
        reject(lambda d: d["cells"][0].__setitem__(
            "source_cognitive_model_hash",
            d["cells"][1]["source_cognitive_model_hash"]))
        reject(lambda d: d["cells"][1]["history_arms"]["confirmed_history"]
               .__setitem__("prior_source_history_hash",
                            d["cells"][1]["history_arms"]["contradicted_history"]
                            ["prior_source_history_hash"]))
        reject(lambda d: d.__setitem__("grid_design", "BLIND_CAUSAL_PROOF"))
        reject(lambda d: d.__setitem__("trust_only_causal_effect", "PROVEN"))
        reject(lambda d: d.__setitem__("confidence_interval", "ESTIMATED"))
        reject(lambda d: d.__setitem__("universal_elite_score", 0.9))
        reject(lambda d: d["cells"][0]["history_arms"]["confirmed_history"]
               .__setitem__("private_inspect_body", "secret"))
        reject(lambda d: d["cells"][0]["history_arms"]["confirmed_history"]
               ["native_action_values"].__setitem__("tell", 500.0))
        reject(lambda d: d["cells"][0]["history_arms"]["confirmed_history"]
               ["native_action_policy"].__setitem__("tell", 0.0))
        reject(lambda d: d["cells"][0]["history_arms"]["confirmed_history"]
               .__setitem__("native_action_feasible_ids", ["wait"]))
        reject(lambda d: d["cells"][0].__setitem__(
            "native_selected_actions_differ",
            not d["cells"][0]["native_selected_actions_differ"]))
        reject(lambda d: d.__setitem__("discovered_switch_costs", [0.0]))
        with self.assertRaises(FrozenInstanceError):
            self.report._canonical_json = "{}"
        report_json = json.dumps(self.data)
        self.assertNotIn('"message":', report_json)
        self.assertNotIn('"claim_symbol":', report_json)
        self.assertNotIn("The restructuring is denied.", report_json)

    def test_exact_replay_byte_vector_without_redoing_five_grid_runs(self):
        from tools.render_native_speaking_cost_grid_vectors import render_source_vector
        expected = render_source_vector(self.report).encode("utf-8")
        pinned = (Path(__file__).resolve().parents[1] /
                  "conformance/network_native_speaking_cost_grid_v1.json")
        self.assertEqual(pinned.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()

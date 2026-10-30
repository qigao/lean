"""M4c.1: paired native controls with precise per-family interventions."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import unittest

from narrative_dynamics.abm import NetworkABMModel
from narrative_dynamics.abm.bb_runtime_contracts import BBRuntimeTick
from narrative_dynamics.abm.native_paired_experiments import (
    NativePairedExperiment, run_bb_target_pair, run_rewiring_threshold_pair,
    run_evolving_entry_pair, compare_native_paired_experiments,
)
from tests.bb_runtime_fixtures import birth, seed
from tests.rewiring_fixtures import rewiring_base_model
from tests.test_network_abm_rewiring import model_for
from tests.test_network_abm_evolving import evolving_case, enter
from tests.test_network_coupled_evolution import build_paired_coupled_case


def build_native_paired_case() -> NativePairedExperiment:
    """Each model uses its existing executor and one legitimate changed input."""
    bb_origin = seed()
    bb = run_bb_target_pair(
        bb_origin,
        (birth("2", (1,)), BBRuntimeTick()),
        (birth("2", (0,)), BBRuntimeTick()),
    )
    base = rewiring_base_model()
    agents = tuple(
        replace(a, receptivity=0.8) if a.agent_id == "b" else a
        for a in base.agents
    )
    tuned_base = NetworkABMModel(
        base.model_id, base.version, agents, base.network
    )
    rew_model = model_for(tuned_base)
    rew = run_rewiring_threshold_pair(
        rew_model,
        replace(rew_model, formation_similarity=0.9),
        initial_beliefs={"a": 1.0, "b": 0.0, "c": 1.0},
    )
    evo_model, evo_initial = evolving_case(beliefs={"a": 1.0, "c": 0.5})
    evo = run_evolving_entry_pair(evo_model, evo_initial, entrant=enter("b"))
    v19 = build_paired_coupled_case()
    return compare_native_paired_experiments(bb, rew, evo, v19)


class NativePairedSourceAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_native_paired_case()
        cls.data = cls.report.to_dict()

    def test_bb_changes_birth_attachment_not_density_or_clock(self):
        d = self.data["families"]["bb_fitness_growth"]
        self.assertEqual(d["origin_identity"], "SAME_NATIVE_SEED_FRAME")
        self.assertEqual(d["native_metric_delta"], ["0", "0"])
        self.assertEqual(d["clock"], "GLOBAL_BB_TICK")
        c, i = d["control"], d["intervention"]
        self.assertEqual(c["last_native_topology"], [["0", "1"], ["1", "2"]])
        self.assertEqual(i["last_native_topology"], [["0", "1"], ["0", "2"]])
        self.assertEqual([x["native_time"] for x in c["metrics"]], [1, 2])
        self.assertEqual([x["exact_fraction"] for x in i["metrics"]], ["2/3", "2/3"])
        self.assertNotEqual(c["source_program_tick_hashes"][0],
                            i["source_program_tick_hashes"][0])
        self.assertEqual(c["source_program_tick_hashes"][1],
                         i["source_program_tick_hashes"][1])
        self.assertNotEqual(c["final_source_hash"], i["final_source_hash"])

    def test_rewiring_threshold_changes_next_round_only(self):
        d = self.data["families"]["directed_rewiring"]
        self.assertEqual(d["origin_identity"],
                         "MATCHED_INITIAL_COMPONENTS_DIFFERENT_MODEL_HASH")
        self.assertEqual(d["intervention_kind"], "FORMATION_THRESHOLD_ONLY")
        c, i = d["control"], d["intervention"]
        self.assertNotEqual(c["model_hash"], i["model_hash"])
        self.assertNotEqual(c["initial_source_hash"], i["initial_source_hash"])
        self.assertEqual(c["native_formation_threshold"], 0.8)
        self.assertEqual(i["native_formation_threshold"], 0.9)
        self.assertEqual(c["native_transmissions_per_round"][0],
                         [["a", "b", "peer"]])
        self.assertEqual(i["native_transmissions_per_round"][0],
                         [["a", "b", "peer"]])
        self.assertIn(["b", "c", "peer"],
                      c["native_transmissions_per_round"][1])
        self.assertNotIn(["b", "c", "peer"],
                         i["native_transmissions_per_round"][1])
        self.assertIn(["b", "c", "peer"], c["next_active_edges_per_round"][0])
        self.assertNotIn(["b", "c", "peer"], i["next_active_edges_per_round"][0])
        self.assertEqual(d["native_metric_delta"][0], "-1/2")

    def test_evolving_one_entry_event_and_no_external_truth(self):
        d = self.data["families"]["lifecycle_feedback"]
        self.assertEqual(d["origin_identity"], "SAME_NATIVE_MODEL_AND_STATE")
        self.assertEqual(d["shared_feedback_schedule"], "EMPTY_BOTH_ARMS")
        c, i = d["control"], d["intervention"]
        self.assertNotEqual(c["event_schedule_hash"], i["event_schedule_hash"])
        self.assertEqual(d["native_metric_delta"], ["1/3", "1/3"])
        self.assertEqual([x["exact_fraction"] for x in c["metrics"]],
                         ["2/3", "2/3"])
        self.assertEqual([x["exact_fraction"] for x in i["metrics"]],
                         ["1", "1"])
        self.assertEqual(c["native_transmissions_per_round"][0], [])
        self.assertIn(["a", "b", "peer"],
                      i["native_transmissions_per_round"][0])
        self.assertIn(["b", "c", "peer"],
                      i["native_transmissions_per_round"][1])
        for arm in (c, i):
            self.assertEqual(arm["external_feedback_targets"], [[], []])

    def test_v19_remains_separate_and_no_cross_family_stats(self):
        d = self.data
        self.assertEqual(d["normalization"], "NONE")
        self.assertEqual(d["cross_family_numeric_ranking"], "FORBIDDEN")
        self.assertEqual(d["sample_size_per_arm"], 1)
        self.assertEqual(d["confidence_interval"], "NOT_ESTIMABLE")
        self.assertEqual(d["causal_attribution"], "UNATTRIBUTED")
        self.assertEqual(d["v19_reference"]["story_lineage"],
                         "DIVERGENT_STORY_BRANCH")
        self.assertEqual(d["v19_reference"]["comparison_status"],
                         "SEPARATE_SOURCE_REFERENCE_ONLY")
        self.assertNotEqual(d["v19_reference"]["control_story_hash"],
                            d["v19_reference"]["intervention_story_hash"])

    def test_negative_source_and_semantics_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "only ordered birth targets"):
            run_bb_target_pair(seed(), (birth("2", (1,)), BBRuntimeTick()),
                               (birth("2", (1,)), BBRuntimeTick()))
        base = model_for()
        with self.assertRaisesRegex(ValueError, "only formation"):
            run_rewiring_threshold_pair(base, replace(base, dissolution_similarity=0.1),
                                        initial_beliefs={"a": 1.0})
        evo, init = evolving_case()
        with self.assertRaisesRegex(ValueError, "default-belief ENTER"):
            run_evolving_entry_pair(evo, init, entrant=enter("b", 0.4))
        with self.assertRaises(TypeError):
            run_evolving_entry_pair(object(), init, entrant=enter("b"))
        for key, fake in (
            ("normalization", "UNIVERSAL_GRAPH_NORMALIZATION"),
            ("causal_attribution", "GENERAL_POPULATION_CAUSAL_PROOF"),
            ("confidence_interval", "95%"),
            ("sample_size_per_arm", 99),
        ):
            d = self.report.to_dict()
            d[key] = fake
            with self.subTest(key=key), self.assertRaises(ValueError):
                NativePairedExperiment(json.dumps(d))
        altered = self.report.to_dict()
        altered["families"]["bb_fitness_growth"]["intervention_kind"] = "V19_TRUST"
        with self.assertRaises(ValueError):
            NativePairedExperiment(json.dumps(altered))
        with self.assertRaises(FrozenInstanceError):
            self.report._canonical_json = "{}"


    def test_nested_source_forgery_and_metric_deltas_fail_closed(self):
        """The verified source-run corpus must not accept shallow JSON relabeling."""
        def rejected(change):
            data = self.report.to_dict()
            change(data)
            with self.assertRaises(ValueError):
                NativePairedExperiment(json.dumps(data, sort_keys=True))

        rejected(lambda d: d["families"]["bb_fitness_growth"]
                 ["native_metric_delta"].__setitem__(0, "1/7"))
        rejected(lambda d: d["families"]["bb_fitness_growth"]["control"]
                 ["metrics"][1].__setitem__("denominator", 2))
        rejected(lambda d: d["families"]["bb_fitness_growth"]["intervention"]
                 ["last_native_topology"].__setitem__(1, ["1", "2"]))
        rejected(lambda d: d["families"]["bb_fitness_growth"]["control"]
                 .__setitem__("final_source_hash", "sha256:" + "a" * 64))
        rejected(lambda d: d["families"]["directed_rewiring"]
                 .__setitem__("first_round_transmission_equal", False))
        rejected(lambda d: d["families"]["directed_rewiring"]["control"]
                 ["native_transmissions_per_round"][0].append(["b", "c", "peer"]))
        rejected(lambda d: d["families"]["directed_rewiring"]["control"]
                 ["next_active_edges_per_round"][0].pop())
        rejected(lambda d: d["families"]["lifecycle_feedback"]["control"]
                 ["native_members_after"][0][0].__setitem__("status", "inactive"))
        rejected(lambda d: d["families"]["lifecycle_feedback"]["intervention"]
                 ["external_feedback_targets"][0].append(["a", "b", "peer"]))
        rejected(lambda d: d["families"]["lifecycle_feedback"]["intervention"]
                 .__setitem__("event_schedule_hash",
                    d["families"]["lifecycle_feedback"]["control"]["event_schedule_hash"]))
        rejected(lambda d: d["v19_reference"]
                 .__setitem__("source_owner", "UNIVERSAL_GRAPH_OWNER"))
        rejected(lambda d: d["v19_reference"]
                 .__setitem__("intervention_story_hash",
                    d["v19_reference"]["control_story_hash"]))

    def test_replays_are_deterministic_and_have_no_private_text(self):
        second = build_native_paired_case()
        self.assertEqual(self.report.content_hash, second.content_hash)
        self.assertEqual(self.report.to_dict(), second.to_dict())
        text = json.dumps(self.data, sort_keys=True)
        self.assertNotIn('"message":', text)
        self.assertNotIn('"claim_symbol":', text)

    def test_reviewed_source_vector_is_exact_bytes(self):
        from tools.render_network_native_paired_vectors import render_vector_corpus
        pinned = (Path(__file__).resolve().parents[1] /
                  "conformance/network_native_paired_v1.json")
        self.assertEqual(pinned.read_bytes(), render_vector_corpus().encode("utf-8"))


if __name__ == "__main__":
    unittest.main()

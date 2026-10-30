"""R2.M4b native-family qualification: BB, rewiring, evolving and V19.

The model families have fundamentally different topology/trust/time
semantics and their native scalar metrics must NEVER be pooled or
ranked as if they described identical social observations.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import unittest

from narrative_dynamics.abm.bb_runtime_contracts import BBRuntimeTick
from narrative_dynamics.abm.model_family_comparison import (
    NativeModelFamilyComparison, observe_bb_growth, observe_rewiring,
    observe_evolving, compare_native_model_families,
)
from narrative_dynamics.abm.rewiring_contracts import initialize_rewiring_population
from narrative_dynamics.abm.evolving_contracts import initialize_evolving_population
from tests.bb_runtime_fixtures import birth, seed
from tests.test_network_abm_rewiring import model_for
from tests.test_network_abm_evolving import evolving_case, enter, feedback
from tests.test_network_coupled_evolution import build_paired_coupled_case

SCHEMA="v19-abm-model-family-qualification-v1"


def build_family_fixture() -> NativeModelFamilyComparison:
    """Freeze source-native programs, never translate between graph owners."""
    bb = observe_bb_growth(
        seed(), 1, (birth("2", (1,)), BBRuntimeTick()),
    )
    rew_model = model_for()
    rew_start = initialize_rewiring_population(
        rew_model, beliefs={"a":1.0, "b":0.0, "c":1.0},
    )
    rew = observe_rewiring(rew_model, rew_start, 2)
    evo_model, evo_start = evolving_case(beliefs={"a":1.0, "c":0.5})
    evo = observe_evolving(
        evo_model, evo_start,
        ((enter("b"),), ()),
        ((feedback("a","b",1.0),), ()),
    )
    v19 = build_paired_coupled_case()
    return compare_native_model_families(bb, rew, evo, v19)


class NativeModelFamilyQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_family_fixture()
        cls.data = cls.report.to_dict()

    def test_bb_exact_birth_is_undirected_and_has_two_clocks(self):
        b = self.data["families"]["bb_fitness_growth"]
        self.assertEqual(b["native_owner"],"BBRuntimeReplay")
        self.assertEqual(b["edge_semantics"],"UNDIRECTED_FITNESS_WEIGHTED_BIRTH")
        self.assertEqual(b["m"],1)
        self.assertEqual(b["initial_undirected_topology"],[["0","1"]])
        self.assertEqual(b["final_node_count"],3)
        self.assertEqual(b["final_birth_count"],1)
        self.assertEqual([x["native_tick"] for x in b["steps"]],[1,2])
        self.assertEqual([x["native_birth_index"] for x in b["steps"]],[1,1])
        self.assertEqual([x["tick_kind"] for x in b["steps"]],["birth","idle"])
        self.assertEqual(b["steps"][0]["native_undirected_topology"],
                         [["0","1"],["1","2"]])
        self.assertEqual(b["steps"][1]["native_undirected_topology"],
                         [["0","1"],["1","2"]])
        self.assertEqual(b["steps"][1]["exact_cumulative_trace_mass"],
                         b["steps"][0]["exact_cumulative_trace_mass"])
        self.assertEqual(b["steps"][1]["parent_frame_hash"],
                         b["steps"][0]["next_frame_hash"])
        self.assertNotIn("trust_before",json.dumps(b))

    def test_rewiring_is_directed_post_propagation_and_no_membership(self):
        r = self.data["families"]["directed_rewiring"]
        self.assertEqual(r["native_owner"],"RewiringPopulationState")
        self.assertEqual(r["edge_semantics"],
                         "DIRECTED_FIXED_CANDIDATES_SIMILARITY_HYSTERESIS")
        self.assertEqual(r["candidate_count"],2)
        self.assertEqual(len(r["steps"]),2)
        self.assertEqual(r["steps"][0]["prior_active_candidate_edges"],
                         [["a","b","peer"]])
        self.assertIn(["b","c","peer"],r["steps"][0]["next_active_candidate_edges"])
        self.assertEqual(r["steps"][0]["native_transmissions"],
                         [["a","b","peer"]])
        self.assertIn(["b","c","peer"],r["steps"][1]["native_transmissions"])
        self.assertEqual(r["steps"][1]["source_state_hash"],
                         r["steps"][0]["next_state_hash"])
        self.assertNotIn("lifecycle_events",json.dumps(r))
        self.assertNotIn("v19_percepts",json.dumps(r))

    def test_evolving_feedback_is_external_and_not_private_v19_verification(self):
        e = self.data["families"]["lifecycle_feedback"]
        self.assertEqual(e["native_owner"],"EvolvingPopulationState")
        self.assertEqual(len(e["steps"]),2)
        first, second = e["steps"]
        self.assertEqual(first["lifecycle_events"],
                         [{"agent_id":"b","kind":"enter"}])
        self.assertIn(["a","b","peer"],first["native_transmissions"])
        self.assertIn(["b","c","peer"],second["native_transmissions"])
        self.assertIn(["b","c","peer"],first["native_active_candidate_edges"])
        self.assertEqual(second["source_state_hash"],first["next_state_hash"])
        learned = next(
            t for t in first["native_trust_values"]
            if t["source"]=="a" and t["target"]=="b"
        )
        self.assertEqual(learned["trust"],0.75)
        self.assertEqual(learned["feedback_count"],1)
        self.assertEqual(first["feedback_targets"],[["a","b","peer"]])
        self.assertIn("EXTERNAL_TRUTH_FEEDBACK",e["trust_semantics"])
        self.assertNotIn("private_inspection",json.dumps(e))

    def test_never_compare_incompatible_raw_edge_counts_as_a_ranking(self):
        d=self.data
        self.assertEqual(d["causal_attribution"],"UNATTRIBUTED")
        self.assertEqual(d["cross_family_numeric_ranking"],"FORBIDDEN")
        self.assertEqual(d["normalization"],"NONE")
        self.assertEqual(d["source_owner"],"INDEPENDENT_NATIVE_MODELS")
        self.assertEqual(set(d["families"]),
                         {"bb_fitness_growth","directed_rewiring","lifecycle_feedback"})
        self.assertEqual(d["v19_reference"]["story_lineage"],
                         "DIVERGENT_STORY_BRANCH")
        self.assertEqual(d["v19_reference"]["controlled_intervention"],
                         "OPEN_PASSAGE_VS_WAIT_THEN_IDENTICAL_MOVE_TELL")
        self.assertEqual(d["comparability"]["raw_edge_counts"],"NOT_RANKABLE")
        self.assertIn("INCOMPARABLE",d["comparability"]["evolving_vs_v19_trust"])
        self.assertIn("INCOMPARABLE",d["comparability"]["bb_vs_directed_others"])
        self.assertNotIn("The restructuring is approved.",json.dumps(d))
        self.assertNotIn('"message":',json.dumps(d))

    def test_repeated_native_replays_are_byte_identical(self):
        first=self.report
        second=build_family_fixture()
        self.assertEqual(first.to_dict(),second.to_dict())
        self.assertEqual(first.content_hash,second.content_hash)
        self.assertEqual(json.dumps(first.to_dict(),sort_keys=True),
                         json.dumps(second.to_dict(),sort_keys=True))
        self.assertEqual(first.to_dict()["families"]["bb_fitness_growth"]
                         ["final_frame_hash"],
                         second.to_dict()["families"]["bb_fitness_growth"]
                         ["final_frame_hash"])

    def test_malformed_sources_budget_and_false_comparability_fail_closed(self):
        with self.assertRaises(ValueError):
            observe_bb_growth(seed(),0,(BBRuntimeTick(),))
        with self.assertRaises(ValueError):
            observe_bb_growth(seed(),1,())
        with self.assertRaises(TypeError):
            observe_bb_growth(object(),1,(BBRuntimeTick(),))
        with self.assertRaises(ValueError):
            observe_rewiring(
                model_for(),
                initialize_rewiring_population(model_for()),
                0,
            )
        evo_model,evo_state=evolving_case()
        with self.assertRaises(ValueError):
            observe_evolving(evo_model,evo_state,(),())
        d=self.report.to_dict()
        d["cross_family_numeric_ranking"]="BB_WINS"
        with self.assertRaisesRegex(ValueError,"equivalent graphs or causal rankings"):
            NativeModelFamilyComparison(json.dumps(d,sort_keys=True))
        d=self.report.to_dict()
        d["families"]["bb_fitness_growth"]["native_owner"]="V19_GRAPH"
        with self.assertRaisesRegex(ValueError,"equivalent graphs or causal rankings"):
            compare_native_model_families(
                d["families"]["bb_fitness_growth"],
                d["families"]["directed_rewiring"],
                d["families"]["lifecycle_feedback"],
                {"schema":"v19-controlled-coupled-evolution-v1",
                 "cross_branch":{"story_lineage":"DIVERGENT_STORY_BRANCH",
                                 "causal_attribution":"UNATTRIBUTED"}},
            )
        with self.assertRaises(FrozenInstanceError):
            self.report._canonical_json="{}"
        self.assertEqual(self.report.to_dict()["cross_family_numeric_ranking"],
                         "FORBIDDEN")

    def test_source_qualified_reviewed_vector_is_byte_exact(self):
        from tools.render_model_family_comparison_vectors import (
            render_vector_corpus,build_vector_manifest,
        )
        x=build_vector_manifest()
        self.assertEqual(x["schema"],SCHEMA)
        self.assertEqual(x["report"]["normalization"],"NONE")
        text=render_vector_corpus()
        self.assertNotIn("The restructuring is approved.",text)
        path=Path(__file__).resolve().parents[1]/"conformance/network_model_family_comparison_v1.json"
        if not path.exists():
            # Initial, expected fail-closed first native CI review handoff.
            raise AssertionError("R2_FAMILIES_VECTOR_BEGIN\n"+text+"R2_FAMILIES_VECTOR_END")
        self.assertEqual(path.read_bytes(),text.encode("utf-8"))


if __name__=="__main__":
    unittest.main()

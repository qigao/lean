"""R2.M5b: source-backed directed structural influence is task-specific."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import unittest

from narrative_dynamics.abm.mesh_contracts import MeshRelation, MeshSnapshot, MeshSociety
from narrative_dynamics.abm.mesh_structural_influence import (
    DirectedSourceInfluenceReport, observe_source_structural_influence,
)
from narrative_dynamics.abm.situated_network import project_situated_network_snapshot
from narrative_dynamics.abm.situated_network_contracts import SituatedNetworkRuntimeModel
from narrative_dynamics.abm.situated_contracts import initialize_situated_world
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedPerceptionEdge, SituatedPerceptionLayer,
)
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_story import initialize_situated_story
from tests.situated_cognition_fixtures import cognitive_office_model
from tests.test_network_abm_situated_percept_cognition import perception_model
from tests.test_network_abm_situated_percept_memory_cognition import recall_model
from tests.test_network_abm_situated_percept_social_cognition import bound_social_model
from tests.test_network_power_authority import controlled_world


def native_structural_case():
    """Use the original V19 world/perception/cognitive/social source owners.

    Alice is an explicit passage controller in records but outside the
    DIRECT_INTERACTION network. The original perception model adds a directed
    Carol(lobby) -> Bob(open) -> Dana(manager) interaction opportunity.
    These edges are authored in the V19 perception world, not in V24.
    """
    from_world = controlled_world()
    placement = {
        "alice": "records",
        "bob": "open",
        "carol": "lobby",
        "dana": "manager",
    }
    world = replace(from_world, agents=tuple(
        replace(agent, initial_place_id=placement[agent.agent_id])
        for agent in from_world.agents
    ))
    cognition = replace(cognitive_office_model(), world_model=world)
    basic = perception_model()
    edges = (
        SituatedPerceptionEdge(
            "lobby-open-directed-interaction",
            SituatedPerceptionLayer.INTERACTION, "lobby", "open", 0.0,
        ),
        SituatedPerceptionEdge(
            "open-manager-directed-interaction",
            SituatedPerceptionLayer.INTERACTION, "open", "manager", 0.0,
        ),
    )
    perception = replace(basic, world_model=world, edges=basic.edges + edges)
    memory = recall_model({}, cognition=cognition, perception=perception)
    social = bound_social_model(memory)
    network = SituatedNetworkRuntimeModel(
        "r2-m5b-source-directed-structural", "1",
        memory, social, "approved", 0.7, 0.5,
    )
    story = initialize_situated_story(
        world, initialize_situated_world(world), perception_model=perception,
    )
    cognitive_state = initialize_situated_percept_memory_cognition(memory, story)
    social_state = initialize_situated_social_memory(social, cognitive_state)
    return network, story, cognitive_state, social_state


def build_native_structural_case() -> DirectedSourceInfluenceReport:
    model, story, cognition, social = native_structural_case()
    return observe_source_structural_influence(
        model, story, cognition, social,
        relation=MeshRelation.DIRECT_INTERACTION,
        controlled_passage_id="records-open",
    )


class StructuralSourceInfluenceAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_native_structural_case()
        cls.data = cls.report.to_dict()

    def test_native_selected_directed_opportunities_not_messages(self):
        d = self.data
        self.assertEqual(d["source_relation"], "direct_interaction")
        self.assertEqual(d["source_relation_semantics"],
                         "SELECTED_DIRECTED_TOPOLOGY_NOT_EVENT_DELIVERY")
        self.assertEqual(d["source_round"], 0)
        self.assertEqual(d["source_directed_edges"],
                         [["bob", "dana"], ["carol", "bob"]])
        self.assertEqual(d["source_latest_tell_events"], 0)
        self.assertEqual(d["source_latest_transmissions"], 0)
        self.assertEqual(d["source_agent_roster"],
                         ["alice", "bob", "carol", "dana"])
        self.assertEqual(d["source_roles"]["dana"], "manager")
        self.assertEqual(d["source_controller_agent_ids"], ["alice"])
        self.assertEqual(d["source_owner"],
                         "DERIVED_V24_FROM_V19_NOT_A_GRAPH_OWNER")

    def test_diffusion_and_fragmentation_have_distinct_key_players(self):
        d = self.data
        by = {x["agent_id"]: x for x in d["structural_measurements"]}
        self.assertEqual(d["highest_out_reach_candidates"], ["carol"])
        self.assertEqual(d["highest_removal_loss_candidates"], ["bob"])
        self.assertTrue(by["alice"]["declared_passage_controller"])
        self.assertFalse(by["bob"]["declared_passage_controller"])
        self.assertFalse(by["carol"]["declared_passage_controller"])
        self.assertFalse(by["dana"]["declared_passage_controller"])
        self.assertEqual(by["alice"]["out_reachable_agents"], [])
        self.assertEqual(by["alice"]["removal_lost_ordered_pairs"], [])
        self.assertEqual(by["carol"]["out_reachable_agents"], ["bob", "dana"])
        self.assertEqual(by["bob"]["out_reachable_agents"], ["dana"])
        self.assertEqual(by["bob"]["in_reachable_agents"], ["carol"])
        self.assertEqual(by["bob"]["removal_lost_ordered_pairs"],
                         [["carol", "dana"]])
        self.assertEqual(by["bob"]["removal_eligible_ordered_pairs"], 6)
        witness = next(w for w in d["bounded_witnesses"]
                       if w["source_agent_id"] == "carol"
                       and w["target_agent_id"] == "dana")
        self.assertEqual(witness["path_agent_ids"], ["carol", "bob", "dana"])
        self.assertEqual(witness["hop_count"], 2)

    def test_exact_source_relation_scope_prevents_equating_trust_and_opportunity(self):
        model, story, cognition, social = native_structural_case()
        source = project_situated_network_snapshot(model, story, cognition, social)
        original = source.to_dict()
        source_hash = source.content_hash
        opportunity = observe_source_structural_influence(
            model, story, cognition, social,
            relation=MeshRelation.DIRECT_INTERACTION,
            controlled_passage_id="records-open",
        )
        relationship = observe_source_structural_influence(
            model, story, cognition, social,
            relation=MeshRelation.RELATIONSHIP,
            controlled_passage_id="records-open",
        )
        self.assertEqual(opportunity.to_dict()["source_directed_edges"],
                         [["bob", "dana"], ["carol", "bob"]])
        self.assertNotEqual(
            relationship.to_dict()["source_directed_edges"],
            opportunity.to_dict()["source_directed_edges"],
        )
        self.assertEqual(source.to_dict(), original)
        self.assertEqual(source.content_hash, source_hash)
        self.assertEqual(opportunity.to_dict()["source_snapshot_hash"], source_hash)
        self.assertEqual(relationship.to_dict()["source_snapshot_hash"], source_hash)
        self.assertNotEqual(opportunity.content_hash, relationship.content_hash)

    def test_realized_transmission_cannot_be_treated_as_temporal_reach(self):
        model, story, cognition, social = native_structural_case()
        with self.assertRaisesRegex(ValueError, "not multi-hop temporal paths"):
            observe_source_structural_influence(
                model, story, cognition, social,
                relation=MeshRelation.TRANSMISSION,
                controlled_passage_id="records-open",
            )
        with self.assertRaisesRegex(TypeError, "typed MeshRelation"):
            observe_source_structural_influence(
                model, story, cognition, social,
                relation="direct_interaction",
                controlled_passage_id="records-open",
            )
        with self.assertRaisesRegex(ValueError, "source passage controllers"):
            observe_source_structural_influence(
                model, story, cognition, social,
                relation=MeshRelation.DIRECT_INTERACTION,
                controlled_passage_id="open-lobby",
            )
        with self.assertRaises(ValueError):
            observe_source_structural_influence(
                replace(model, version="unmatched"), story, cognition, social,
                relation=MeshRelation.DIRECT_INTERACTION,
                controlled_passage_id="records-open",
            )

    def test_private_source_and_forged_structural_scores_fail_closed(self):
        def reject(mutator):
            d = self.report.to_dict()
            mutator(d)
            with self.assertRaises((ValueError, TypeError, KeyError)):
                DirectedSourceInfluenceReport(json.dumps(d, sort_keys=True))
        reject(lambda d: d.__setitem__("normalization", "UNIVERSAL_POWER"))
        reject(lambda d: d.__setitem__("causal_attribution", "POPULATION_PROOF"))
        reject(lambda d: d.__setitem__("sample_size", 200))
        reject(lambda d: d.__setitem__("message", "private message"))
        reject(lambda d: d.__setitem__("universal_power_score", 0.95))
        reject(lambda d: d["source_directed_edges"].append(["alice", "dana"]))
        reject(lambda d: d["source_directed_edges"].append(["dana", "bob"]))
        reject(lambda d: d["source_directed_edges"].__setitem__(0, ["dana", "bob"]))
        reject(lambda d: d["structural_measurements"][0]
               .__setitem__("out_degree", 17))
        reject(lambda d: d["structural_measurements"][1]
               ["removal_lost_ordered_pairs"].clear())
        reject(lambda d: d["bounded_witnesses"][0]
               .__setitem__("path_agent_ids", ["alice", "bob"]))
        reject(lambda d: d.__setitem__("highest_out_reach_candidates", ["alice"]))
        reject(lambda d: d.__setitem__("highest_removal_loss_candidates", ["alice"]))
        reject(lambda d: d["structural_measurements"][0]
               .__setitem__("private_percept_text", "secret"))
        with self.assertRaises(FrozenInstanceError):
            self.report._canonical_json = "{}"
        dumped = json.dumps(self.data, sort_keys=True)
        self.assertNotIn("The restructuring is approved.", dumped)
        self.assertNotIn('"message":', dumped)
        self.assertNotIn('"claim_symbol":', dumped)

    def test_source_replay_remains_deterministic_and_pinned(self):
        again = build_native_structural_case()
        self.assertEqual(self.report.content_hash, again.content_hash)
        self.assertEqual(self.report.to_dict(), again.to_dict())
        from tools.render_source_structural_influence_vectors import render_source_vector
        pin = (Path(__file__).resolve().parents[1] /
               "conformance/network_source_structural_influence_v1.json")
        self.assertEqual(pin.read_bytes(), render_source_vector().encode("utf-8"))


if __name__ == "__main__":
    unittest.main()

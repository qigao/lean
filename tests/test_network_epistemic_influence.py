"""R2.M5c: source-owned authority → TELL → INSPECT → directed trust.

No alternative social memory, permission based on centrality, fictitious
temporal transmission path or action inference from adoption threshold.
"""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import unittest

from narrative_dynamics.abm.situated import SituatedActionIntent, SituatedActionKind
from narrative_dynamics.abm.situated_cognition_contracts import SituatedObservationRule
from narrative_dynamics.abm.situated_contracts import initialize_situated_world
from narrative_dynamics.abm.situated_memory_cognition_contracts import SituatedMemoryRecallCue
from narrative_dynamics.abm.situated_network_contracts import SituatedNetworkRuntimeModel
from narrative_dynamics.abm.mesh_epistemic_influence import (
    SourceEpistemicInfluenceReport, observe_epistemic_key_player_pair,
)
from narrative_dynamics.abm.situated_story import (
    advance_situated_story, initialize_situated_story,
)
from tests.test_network_structural_influence import native_structural_case
from tests.test_network_abm_situated_percept_memory_cognition import recall_model
from tests.test_network_abm_situated_percept_social_cognition import bound_social_model


def source_epistemic_case():
    """One original V19 world, no second topology or cognitive engine."""
    original, *_ = native_structural_case()
    base_cognition = original.percept_memory_model.cognitive_model
    world = base_cognition.world_model
    perception = original.percept_memory_model.perception_model
    bob = next(a for a in base_cognition.agents if a.agent_id == "bob")
    bob = replace(bob, observation_rules=bob.observation_rules + (
        SituatedObservationRule(
            "tell-denied", "denied", "tell", SituatedActionKind.TELL,
            "told", "message", "The restructuring is denied.",
        ),
    ))
    cognition = replace(
        base_cognition, agents=tuple(
            bob if a.agent_id == "bob" else a for a in base_cognition.agents
        ),
    )
    cues = (
        SituatedMemoryRecallCue(
            "tell", "restructuring", event_kinds=(SituatedActionKind.TELL,),
        ),
        SituatedMemoryRecallCue(
            "inspect", "restructuring", event_kinds=(SituatedActionKind.INSPECT,),
        ),
    )
    memory = recall_model({"bob": cues}, cognition=cognition, perception=perception)
    social = bound_social_model(memory)
    network = SituatedNetworkRuntimeModel(
        "r2-m5c-key-player-epistemic", "1",
        memory, social, "approved", 0.7, 0.5,
    )
    root = initialize_situated_story(
        world, initialize_situated_world(world), perception_model=perception,
    )
    inspect = SituatedActionIntent(
        "alice-inspect", "alice", SituatedActionKind.INSPECT, target_id="memo",
    )
    first = advance_situated_story(world, root, (inspect,))
    inspect_event = next(e for e in first.rounds[-1].events
                         if e.action_id == inspect.action_id)
    move = SituatedActionIntent(
        "alice-common-move", "alice", SituatedActionKind.MOVE,
        target_id="records-open",
    )
    tell = SituatedActionIntent(
        "alice-common-tell", "alice", SituatedActionKind.TELL,
        message="The restructuring is denied.",
        source_event_ids=(inspect_event.event_id,),
    )
    bob_move = SituatedActionIntent(
        "bob-common-move", "bob", SituatedActionKind.MOVE,
        target_id="open-records",
    )
    bob_inspect = SituatedActionIntent(
        "bob-common-inspect", "bob", SituatedActionKind.INSPECT,
        target_id="memo",
    )
    control = (
        (inspect,),
        (SituatedActionIntent("alice-control-wait", "alice", SituatedActionKind.WAIT),),
        (move,),
        (tell,),
        (bob_move,),
        (bob_inspect,),
    )
    intervention = (
        (inspect,),
        (SituatedActionIntent("alice-authorized-open", "alice",
                              SituatedActionKind.OPEN_PASSAGE, target_id="records-open"),),
        (move,),
        (tell,),
        (bob_move,),
        (bob_inspect,),
    )
    return network, root, control, intervention


def build_source_epistemic_case() -> SourceEpistemicInfluenceReport:
    network, root, control, intervention = source_epistemic_case()
    return observe_epistemic_key_player_pair(
        network, root, control_schedule=control,
        intervention_schedule=intervention,
    )


class EpistemicKeyPlayerSourceAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dossier = build_source_epistemic_case()
        cls.data = cls.dossier.to_dict()

    def test_same_world_and_single_resource_intervention(self):
        d = self.data
        self.assertEqual(d["source_owner"],
                         "V19_SOURCE_COGNITION_AND_SOCIAL_MEMORY_V24_READ_ONLY")
        self.assertEqual(d["intervention"], "ALICE_WAIT_VS_AUTHORIZED_OPEN_ONLY")
        self.assertEqual(d["controller_agent_ids"], ["alice"])
        self.assertEqual(d["resource_controller"], "alice")
        self.assertEqual(d["source_lineage"], "DIVERGENT_STORY_BRANCH")
        self.assertEqual(d["structural_directed_edges"],
                         [["bob", "dana"], ["carol", "bob"]])
        self.assertEqual(d["structural_out_reach_candidates"], ["carol"])
        self.assertEqual(d["structural_disruption_candidates"], ["bob"])
        self.assertEqual(d["structural_event_count"], 0)
        self.assertEqual(d["structural_transmission_count"], 0)
        self.assertEqual(len(d["shared_native_action_ids"]), 5)
        self.assertEqual(len(set(d["shared_native_action_ids"])), 5)

    def test_percept_and_claim_require_actual_source_actions(self):
        arms = self.data["branches"]
        c, i = arms["control_wait"], arms["authorized_open"]
        self.assertEqual(c["gate_outcome"], "waited")
        self.assertEqual(i["gate_outcome"], "passage_opened")
        self.assertFalse(c["move_success"])
        self.assertTrue(i["move_success"])
        self.assertEqual(c["move_outcome"], "passage_closed")
        self.assertEqual(i["move_outcome"], "moved")
        self.assertEqual(c["tell_fidelity_bob"], "detected")
        self.assertEqual(i["tell_fidelity_bob"], "exact")
        self.assertEqual(c["tell_event_id"], i["tell_event_id"])
        self.assertNotEqual(c["tell_event_hash"], i["tell_event_hash"])
        self.assertNotEqual(c["story_hash"], i["story_hash"])
        for a in (c, i):
            self.assertTrue(a["bob_inspect_success"])
            self.assertEqual(a["admitted_verification"], 1)
        self.assertEqual(c["admitted_testimony"], 0)
        self.assertEqual(i["admitted_testimony"], 1)
        self.assertEqual(c["bob_claim_statuses"], [])
        self.assertEqual(i["bob_claim_statuses"], ["contradicted"])

    def test_directed_source_trust_only_after_private_verification(self):
        c, i = (self.data["branches"][k] for k in
                ("control_wait", "authorized_open"))
        self.assertEqual(c["trust_revisions"], 0)
        self.assertEqual(i["trust_revisions"], 1)
        self.assertEqual(c["bob_trust_alice_before"], 0.5)
        self.assertEqual(c["bob_trust_alice_after"], 0.5)
        self.assertEqual(i["bob_trust_alice_before"], 0.5)
        self.assertEqual(i["bob_trust_alice_after"], 0.3)
        for a in (c, i):
            self.assertEqual(a["reverse_alice_trust_bob_before"], 0.5)
            self.assertEqual(a["reverse_alice_trust_bob_after"], 0.5)
            self.assertTrue(a["bob_to_alice_relationship_after"])
        self.assertTrue(c["alice_to_bob_relationship_after"])
        self.assertFalse(i["alice_to_bob_relationship_after"])
        self.assertNotEqual(i["source_social_before_hash"],
                            i["source_social_after_hash"])
        self.assertNotEqual(i["source_network_before_hash"],
                            i["source_network_after_hash"])

    def test_belief_threshold_only_classifies_not_acts(self):
        d = self.data
        self.assertEqual(d["tracked_hypothesis_id"], "approved")
        self.assertEqual(d["adoption_threshold"], 0.7)
        self.assertEqual(d["post_trust_action_effect"], "NOT_ESTABLISHED")
        self.assertEqual(d["cross_mechanism_ranking"], "FORBIDDEN")
        for arm in d["branches"].values():
            self.assertEqual(
                arm["bob_above_threshold_before"],
                arm["bob_tracked_belief_before"] >= 0.7,
            )
            self.assertEqual(
                arm["bob_above_threshold_after"],
                arm["bob_tracked_belief_after"] >= 0.7,
            )

    def test_anti_forgery_and_privacy_gate(self):
        def rejects(mutation):
            data = self.dossier.to_dict()
            mutation(data)
            with self.assertRaises((ValueError, TypeError)):
                SourceEpistemicInfluenceReport(json.dumps(data, sort_keys=True))
        rejects(lambda d: d.__setitem__("universal_influence_score", 0.99))
        rejects(lambda d: d.__setitem__("causal_attribution", "POPULATION_CAUSAL_PROOF"))
        rejects(lambda d: d.__setitem__("external_validation", "APPROVED"))
        rejects(lambda d: d.__setitem__("sample_size_per_arm", 12))
        rejects(lambda d: d.__setitem__("post_trust_action_effect", "ACTOR_PERSUADED"))
        rejects(lambda d: d["branches"]["control_wait"]
                .__setitem__("admitted_testimony", 1))
        rejects(lambda d: d["branches"]["authorized_open"]
                .__setitem__("bob_trust_alice_after", 0.5))
        rejects(lambda d: d["branches"]["authorized_open"]
                .__setitem__("alice_to_bob_relationship_after", True))
        rejects(lambda d: d["branches"]["control_wait"]
                .__setitem__("tell_fidelity_bob", "exact"))
        rejects(lambda d: d["branches"]["authorized_open"]
                .__setitem__("message", "private TELL payload"))
        rejects(lambda d: d["branches"]["authorized_open"]
                .__setitem__("private_inspection", "private observation"))
        rejects(lambda d: d["branches"]["authorized_open"]
                .__setitem__("bob_above_threshold_after",
                             not d["branches"]["authorized_open"]["bob_above_threshold_after"]))
        with self.assertRaises(FrozenInstanceError):
            self.dossier._canonical_json = "{}"
        serialized = json.dumps(self.data)
        self.assertNotIn('"message":', serialized)
        self.assertNotIn("The restructuring is denied.", serialized)
        self.assertNotIn('"claim_symbol":', serialized)

    def test_replay_determinism_and_reviewed_source_bytes(self):
        again = build_source_epistemic_case()
        self.assertEqual(self.dossier.content_hash, again.content_hash)
        self.assertEqual(self.dossier.to_dict(), again.to_dict())
        from tools.render_source_epistemic_influence_vectors import render_source_vector
        path = (Path(__file__).resolve().parents[1] /
                "conformance/network_source_epistemic_influence_v1.json")
        self.assertEqual(path.read_bytes(), render_source_vector().encode("utf-8"))

    def test_modified_or_unsourced_intervention_refused(self):
        network, root, control, intervention = source_epistemic_case()
        bad = list(intervention)
        bad[3] = (replace(bad[3][0], message="Changed independent variable"),)
        with self.assertRaisesRegex(ValueError, "more than passage opening"):
            observe_epistemic_key_player_pair(
                network, root, control_schedule=control,
                intervention_schedule=tuple(bad),
            )
        invalid_control, invalid_open = list(control), list(intervention)
        malformed_tell = (
            replace(control[3][0], source_event_ids=("r9999:e9999",)),
        )
        invalid_control[3] = malformed_tell
        invalid_open[3] = malformed_tell
        with self.assertRaisesRegex(ValueError, "causal reference"):
            observe_epistemic_key_player_pair(
                network, root, control_schedule=tuple(invalid_control),
                intervention_schedule=tuple(invalid_open),
            )


if __name__ == "__main__":
    unittest.main()

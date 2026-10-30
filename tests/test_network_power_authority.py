"""R2.M5a: power over a passage is not role, reach, trust or speech."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import unittest

from narrative_dynamics.abm.situated import SituatedActionIntent, SituatedActionKind
from narrative_dynamics.abm.situated_contracts import (
    initialize_situated_world,
)
from narrative_dynamics.abm.situated_story import (
    explain_situated_event, replay_situated_story,
)
from narrative_dynamics.abm.mesh_authority_observation import (
    PassageAuthorityEvidence, observe_passage_authority,
)
from tests.situated_fixtures import office_model
from tests.test_network_abm_situated_percept_cognition import perception_model


def controlled_world():
    original = office_model()
    return replace(
        original,
        passages=tuple(
            replace(p, initially_open=False, controller_agent_ids=("alice",))
            if p.passage_id == "records-open" else p
            for p in original.passages
        ),
        agents=tuple(
            replace(agent, initial_place_id="records")
            if agent.agent_id == "dana" else agent
            for agent in original.agents
        ),
    )


def build_authority_case() -> PassageAuthorityEvidence:
    world = controlled_world()
    perception = replace(perception_model(), world_model=world)
    return observe_passage_authority(world, perception)


class NativeResourceAuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_authority_case()
        cls.data = cls.report.to_dict()

    def test_unrestricted_legacy_model_has_original_canonical_shape(self):
        original = office_model()
        for passage in original.passages:
            self.assertEqual(passage.controller_agent_ids, ())
            self.assertNotIn("controller_agent_ids", passage.to_dict())
        self.assertEqual(set(original.to_dict()),
                         {"model_id", "version", "places", "passages", "agents", "objects"})
        self.assertNotEqual(original.content_hash, controlled_world().content_hash)

    def test_controller_identity_is_validated_and_not_inferred_from_role(self):
        world = controlled_world()
        controlled = next(p for p in world.passages if p.passage_id == "records-open")
        self.assertEqual(controlled.controller_agent_ids, ("alice",))
        self.assertEqual(controlled.to_dict()["controller_agent_ids"], ["alice"])
        with self.assertRaisesRegex(TypeError, "immutable tuple"):
            replace(controlled, controller_agent_ids=["alice"])
        with self.assertRaisesRegex(ValueError, "unique"):
            replace(controlled, controller_agent_ids=("alice", "alice"))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            replace(controlled, controller_agent_ids=("",))
        with self.assertRaisesRegex(ValueError, "exact world Agent"):
            replace(world, passages=tuple(
                replace(p, controller_agent_ids=("unknown",))
                if p.passage_id == "records-open" else p
                for p in world.passages
            ))
        roles = {x.agent_id: x.role for x in world.agents}
        self.assertEqual(roles["alice"], "analyst")
        self.assertEqual(roles["dana"], "manager")

    def test_source_denies_manager_but_analyst_changes_world(self):
        d = self.data
        self.assertEqual(d["source_owner"], "V19_SITUATED_WORLD_MODEL_AND_STORY")
        self.assertEqual(d["mechanism"], "EXPLICIT_PASSAGE_RESOURCE_CONTROL")
        self.assertEqual(d["controller_agent_ids"], ["alice"])
        self.assertEqual(d["cross_branch_relation"], "DIVERGENT_STORY_BRANCH")
        arms = d["arms"]
        for name in ("control_wait", "unauthorized_manager_open"):
            with self.subTest(name=name):
                self.assertFalse(arms[name]["gate_open_at_tell"])
                self.assertFalse(arms[name]["move_success"])
                self.assertEqual(arms[name]["move_event_outcome"], "passage_closed")
                self.assertEqual(arms[name]["bob_tell_fidelity"], "detected")
                self.assertEqual(arms[name]["alice_place_at_tell"], "records")
        self.assertEqual(arms["unauthorized_manager_open"]["gate_event_outcome"],
                         "not_authorized")
        self.assertFalse(arms["unauthorized_manager_open"]["gate_event_success"])
        self.assertEqual(arms["authorized_open"]["gate_event_outcome"], "passage_opened")
        self.assertTrue(arms["authorized_open"]["gate_event_success"])
        self.assertTrue(arms["authorized_open"]["move_success"])
        self.assertTrue(arms["authorized_open"]["gate_open_at_tell"])
        self.assertEqual(arms["authorized_open"]["bob_tell_fidelity"], "exact")
        self.assertEqual(arms["authorized_open"]["alice_place_at_tell"], "open")
        self.assertEqual(len({a["story_hash"] for a in arms.values()}), 3)
        self.assertEqual(len({a["tell_event_hash"] for a in arms.values()}), 2)

    def test_denied_contender_cannot_veto_a_permitted_open(self):
        world = controlled_world()
        initial = initialize_situated_world(world)
        actions = (
            SituatedActionIntent("a-authorized", "alice",
                                 SituatedActionKind.OPEN_PASSAGE, target_id="records-open"),
            SituatedActionIntent("b-manager-unauthorized", "dana",
                                 SituatedActionKind.OPEN_PASSAGE, target_id="records-open"),
        )
        story = replay_situated_story(world, initial, (actions,))
        events = {e.action_id: e for e in story.rounds[0].events}
        self.assertTrue(events["a-authorized"].success)
        self.assertEqual(events["a-authorized"].outcome, "passage_opened")
        self.assertFalse(events["b-manager-unauthorized"].success)
        self.assertEqual(events["b-manager-unauthorized"].outcome, "not_authorized")
        self.assertTrue(next(x.open for x in story.current_state.passages
                             if x.passage_id == "records-open"))
        explanation = explain_situated_event(
            story, events["b-manager-unauthorized"].event_id)
        self.assertIn("not a declared controller", explanation.rule)
        self.assertFalse(explanation.success)

    def test_denied_close_cannot_mutate_an_open_passage(self):
        model = controlled_world()
        initial = initialize_situated_world(model)
        opened = replay_situated_story(
            model, initial,
            ((SituatedActionIntent("alice-open", "alice",
                                  SituatedActionKind.OPEN_PASSAGE, target_id="records-open"),),),
        )
        after_attempt = replay_situated_story(
            model, initial,
            ((SituatedActionIntent("alice-open", "alice",
                                  SituatedActionKind.OPEN_PASSAGE, target_id="records-open"),),
             (SituatedActionIntent("dana-close", "dana",
                                   SituatedActionKind.CLOSE_PASSAGE, target_id="records-open"),)),
        )
        denied = next(e for e in after_attempt.rounds[-1].events
                      if e.action_id == "dana-close")
        self.assertFalse(denied.success)
        self.assertEqual(denied.outcome, "not_authorized")
        self.assertEqual(after_attempt.current_state.passages,
                         opened.current_state.passages)

    def test_report_is_deterministic_private_and_fail_closed(self):
        again = build_authority_case()
        self.assertEqual(self.report.content_hash, again.content_hash)
        self.assertEqual(self.report.to_dict(), again.to_dict())
        self.assertEqual(self.data["causal_attribution"], "UNATTRIBUTED")
        self.assertEqual(self.data["confidence_interval"], "NOT_ESTIMABLE")
        self.assertEqual(self.data["network_centrality"], "NOT_MEASURED")
        self.assertEqual(self.data["trust_change"], "NOT_MEASURED")
        body = json.dumps(self.data, sort_keys=True)
        self.assertNotIn("The restructuring is approved.", body)
        self.assertNotIn('"message":', body)

        def reject(change):
            d = self.report.to_dict()
            change(d)
            with self.assertRaises(ValueError):
                PassageAuthorityEvidence(json.dumps(d, sort_keys=True))

        reject(lambda d: d.__setitem__("causal_attribution", "REAL_WORLD_PROVEN"))
        reject(lambda d: d.__setitem__("trust_change", "AUTO_TRUST"))
        reject(lambda d: d.__setitem__("universal_power_score", 0.99))
        reject(lambda d: d["arms"]["authorized_open"]
               .__setitem__("message", "private TELL payload"))
        reject(lambda d: d["arms"]["authorized_open"]
               .__setitem__("private_percept_text", "private INSPECT data"))
        reject(lambda d: d["arms"]["unauthorized_manager_open"]
               .__setitem__("gate_event_outcome", "passage_opened"))
        reject(lambda d: d["arms"]["authorized_open"]
               .__setitem__("move_success", False))
        reject(lambda d: d["arms"]["control_wait"]
               .__setitem__("bob_tell_fidelity", "exact"))
        reject(lambda d: d["arms"]["control_wait"]
               .__setitem__("move_event_outcome", "moved"))
        reject(lambda d: d["arms"]["unauthorized_manager_open"]
               .__setitem__("tell_event_hash",
                            d["arms"]["authorized_open"]["tell_event_hash"]))
        with self.assertRaises(FrozenInstanceError):
            self.report._canonical_json = "{}"

    def test_source_review_vector_is_byte_identical(self):
        from tools.render_passage_authority_vectors import render_source_vector
        pinned = (Path(__file__).resolve().parents[1] /
                  "conformance/network_passage_authority_v1.json")
        self.assertEqual(pinned.read_bytes(), render_source_vector().encode("utf-8"))


if __name__ == "__main__":
    unittest.main()

from dataclasses import replace
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.situated import (
    ObservationChannel,
    SituatedActionIntent,
    SituatedActionKind,
)
from narrative_dynamics.abm.situated_cognition_contracts import SituatedObservationRule
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedMemoryRecallCue,
)
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    initialize_situated_percept_memory_cognition,
)
from narrative_dynamics.abm.situated_percept_social_cognition import (
    recall_situated_percept_memories_with_social_trust,
    simulate_situated_percept_social_cognitive_round,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedClaimStatus,
    SituatedSocialEvidenceKind,
    initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_social_memory import advance_situated_social_memory
from narrative_dynamics.abm.situated_story import (
    advance_situated_story,
    initialize_situated_story,
)
from tests.situated_cognition_fixtures import cognitive_office_model
from tests.test_network_abm_situated_percept_cognition import (
    SECRET,
    initial_state,
    perception_model,
)
from tests.test_network_abm_situated_percept_memory_cognition import (
    agent,
    inspection_checkpoint,
    mind,
    recall_model,
)
from tests.test_network_abm_situated_social_memory_contracts import social_model


def bound_social_model(memory_model):
    legacy = social_model()
    return replace(
        legacy,
        memory_cognitive_model=replace(
            legacy.memory_cognitive_model,
            cognitive_model=memory_model.cognitive_model,
        ),
    )


def tell_checkpoint(*, door_open, visual=False, kind_only=False):
    perception = perception_model(visual=visual)
    cognition = cognitive_office_model()
    if kind_only:
        bob = next(item for item in cognition.agents if item.agent_id == "bob")
        bob = replace(
            bob,
            observation_rules=(SituatedObservationRule(
                "saw-tell", "approved", "tell", SituatedActionKind.TELL
            ),),
        )
        cognition = replace(
            cognition,
            agents=tuple(bob if item.agent_id == "bob" else item for item in cognition.agents),
        )
    story = initialize_situated_story(
        cognition.world_model,
        initial_state(perception, door_open=door_open),
        perception_model=perception,
    )
    story = advance_situated_story(
        cognition.world_model,
        story,
        (SituatedActionIntent("alice-tell", "alice", SituatedActionKind.TELL, message=SECRET),),
    )
    return perception, cognition, story


class SituatedPerceptSocialEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.database = f"{self.temporary.name}/social.sqlite3"

    def tearDown(self):
        self.temporary.cleanup()

    def _case(self, *, door_open, visual=False, kind_only=False, cue_text=SECRET):
        perception, cognition, story = tell_checkpoint(
            door_open=door_open, visual=visual, kind_only=kind_only
        )
        cue = SituatedMemoryRecallCue(
            "tell", cue_text,
            event_kinds=(SituatedActionKind.TELL,) if door_open or visual else (),
            channels=(ObservationChannel.AUDITORY,),
        )
        memory_model = recall_model(
            {"bob": (cue,)}, cognition=cognition, perception=perception
        )
        social = bound_social_model(memory_model)
        cognitive_state = initialize_situated_percept_memory_cognition(memory_model, story)
        social_state = initialize_situated_social_memory(social, cognitive_state)
        return memory_model, social, story, cognitive_state, social_state

    def test_only_exact_tell_creates_testimony(self):
        exact = self._case(door_open=True)
        exact_result = simulate_situated_percept_social_cognitive_round(
            self.database, exact[0], exact[1], exact[2], exact[3], exact[4]
        )
        self.assertEqual(len(exact_result.social_update.admitted_evidence), 1)
        evidence = exact_result.social_update.admitted_evidence[0]
        self.assertIs(evidence.kind, SituatedSocialEvidenceKind.TESTIMONY)
        self.assertEqual(evidence.observer_agent_id, "bob")
        self.assertEqual(evidence.source_agent_id, "alice")
        replay = advance_situated_social_memory(
            exact[1],
            exact_result.next_cognitive_state,
            exact_result.next_social_state,
            exact_result.next_cognitive_state,
            exact_result.social_update.admitted_evidence,
        )
        self.assertEqual(replay.admitted_evidence, ())
        self.assertEqual(replay.next_state, exact_result.next_social_state)

        detected = self._case(door_open=False, cue_text="detected")
        detected_result = simulate_situated_percept_social_cognitive_round(
            f"{self.temporary.name}/detected.sqlite3",
            detected[0], detected[1], detected[2], detected[3], detected[4],
        )
        bob_recall = next(
            item for item in detected_result.recalls if item.prior_mind.agent_id == "bob"
        )
        self.assertEqual(len(bob_recall.admissions), 1)
        self.assertEqual(detected_result.social_update.admitted_evidence, ())
        self.assertEqual(detected_result.next_social_state.claims, ())

        identified = self._case(
            door_open=False, visual=True, kind_only=True, cue_text="tell"
        )
        identified_result = simulate_situated_percept_social_cognitive_round(
            f"{self.temporary.name}/identified.sqlite3",
            identified[0], identified[1], identified[2], identified[3], identified[4],
        )
        bob_recall = next(
            item for item in identified_result.recalls if item.prior_mind.agent_id == "bob"
        )
        self.assertEqual(bob_recall.admissions[0].rule_id, "saw-tell")
        self.assertEqual(identified_result.social_update.admitted_evidence, ())

    def test_new_direct_exact_percept_also_creates_testimony(self):
        perception, cognition, story = tell_checkpoint(door_open=True)
        memory_model = recall_model({}, cognition=cognition, perception=perception)
        social = bound_social_model(memory_model)
        checkpoint = initialize_situated_percept_memory_cognition(memory_model, story)
        bob = replace(mind(checkpoint, "bob"), observation_floor_round=0)
        cognitive_state = replace(
            checkpoint,
            parent_state_hash="sha256:" + "f" * 64,
            minds=tuple(
                bob if item.agent_id == "bob" else item for item in checkpoint.minds
            ),
            checkpoint=False,
        )
        social_state = initialize_situated_social_memory(social, cognitive_state)

        result = simulate_situated_percept_social_cognitive_round(
            self.database, memory_model, social, story, cognitive_state, social_state
        )

        evidence = result.social_update.admitted_evidence
        self.assertEqual(len(evidence), 1)
        self.assertTrue(evidence[0].evidence_id.startswith("direct-percept:"))
        self.assertIs(evidence[0].kind, SituatedSocialEvidenceKind.TESTIMONY)

    def test_exact_private_inspection_creates_source_less_verification(self):
        perception = perception_model()
        cognition, story, _ = inspection_checkpoint(perception)
        cue = SituatedMemoryRecallCue(
            "inspect", "restructuring",
            event_kinds=(SituatedActionKind.INSPECT,),
            channels=(ObservationChannel.INSPECTION,),
        )
        memory_model = recall_model(
            {"alice": (cue,)}, cognition=cognition, perception=perception
        )
        social = bound_social_model(memory_model)
        cognitive_state = initialize_situated_percept_memory_cognition(memory_model, story)
        social_state = initialize_situated_social_memory(social, cognitive_state)

        result = simulate_situated_percept_social_cognitive_round(
            self.database, memory_model, social, story, cognitive_state, social_state
        )

        self.assertEqual(len(result.social_update.admitted_evidence), 1)
        evidence = result.social_update.admitted_evidence[0]
        self.assertIs(evidence.kind, SituatedSocialEvidenceKind.VERIFICATION)
        self.assertEqual(evidence.observer_agent_id, "alice")
        self.assertIsNone(evidence.source_agent_id)

    def test_recalled_testimony_then_private_contradiction_updates_only_directed_edge(self):
        perception = perception_model()
        cognition = cognitive_office_model()
        bob = next(item for item in cognition.agents if item.agent_id == "bob")
        bob = replace(
            bob,
            observation_rules=bob.observation_rules + (SituatedObservationRule(
                "tell-denied", "denied", "tell", SituatedActionKind.TELL,
                "told", "message", "The restructuring is denied.",
            ),),
        )
        cognition = replace(
            cognition,
            agents=tuple(bob if item.agent_id == "bob" else item for item in cognition.agents),
        )
        story = initialize_situated_story(
            cognition.world_model,
            initial_state(perception, door_open=True),
            perception_model=perception,
        )
        story = advance_situated_story(
            cognition.world_model, story,
            (SituatedActionIntent("alice-inspect", "alice", SituatedActionKind.INSPECT, "memo"),),
        )
        alice_inspection = story.rounds[-1].events[0]
        story = advance_situated_story(
            cognition.world_model, story,
            (SituatedActionIntent("alice-move", "alice", SituatedActionKind.MOVE, "records-open"),),
        )
        story = advance_situated_story(
            cognition.world_model, story,
            (SituatedActionIntent(
                "alice-denied", "alice", SituatedActionKind.TELL,
                message="The restructuring is denied.",
                source_event_ids=(alice_inspection.event_id,),
            ),),
        )
        story = advance_situated_story(
            cognition.world_model, story,
            (SituatedActionIntent("bob-move", "bob", SituatedActionKind.MOVE, "open-records"),),
        )
        story = advance_situated_story(
            cognition.world_model, story,
            (SituatedActionIntent("bob-inspect", "bob", SituatedActionKind.INSPECT, "memo"),),
        )
        cues = (
            SituatedMemoryRecallCue("tell", "restructuring", event_kinds=(SituatedActionKind.TELL,)),
            SituatedMemoryRecallCue("inspect", "restructuring", event_kinds=(SituatedActionKind.INSPECT,)),
        )
        memory_model = recall_model(
            {"bob": cues}, cognition=cognition, perception=perception
        )
        social = bound_social_model(memory_model)
        cognitive_state = initialize_situated_percept_memory_cognition(memory_model, story)
        social_state = initialize_situated_social_memory(social, cognitive_state)
        before = {
            (item.observer_agent_id, item.source_agent_id): item
            for item in social_state.relationships
        }

        result = simulate_situated_percept_social_cognitive_round(
            self.database, memory_model, social, story, cognitive_state, social_state
        )
        after = {
            (item.observer_agent_id, item.source_agent_id): item
            for item in result.next_social_state.relationships
        }

        self.assertEqual(
            len(result.social_update.admitted_evidence),
            2,
            (
                result.social_update.admitted_evidence,
                tuple(
                    (item.prior_mind.agent_id, item.admissions)
                    for item in result.recalls
                ),
            ),
        )
        claim = result.next_social_state.claims[0]
        self.assertIs(claim.status, SituatedClaimStatus.CONTRADICTED)
        self.assertEqual(after[("bob", "alice")].trust, 0.3)
        self.assertEqual(after[("bob", "alice")].affinity, -0.2)
        self.assertEqual(
            {key: value for key, value in after.items() if key != ("bob", "alice")},
            {key: value for key, value in before.items() if key != ("bob", "alice")},
        )

    def test_social_recall_uses_current_directed_trust(self):
        memory_model, social, story, cognitive_state, social_state = self._case(door_open=True)
        result = recall_situated_percept_memories_with_social_trust(
            self.database,
            memory_model,
            social,
            agent(memory_model, "bob"),
            mind(cognitive_state, "bob"),
            story=story,
            cognitive_state=cognitive_state,
            social_state=social_state,
        )
        self.assertEqual(result.admissions[0].source_trust, 0.5)
        self.assertEqual(result.admissions[0].evidence_weight, 0.5)


if __name__ == "__main__":
    unittest.main()

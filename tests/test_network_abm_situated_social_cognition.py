from dataclasses import replace
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.situated import ObservationChannel, SituatedActionKind
from narrative_dynamics.abm.situated_memory import ingest_situated_story
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedMemoryRecallCue,
    initialize_situated_memory_cognition,
)
from narrative_dynamics.abm.situated_social_cognition import (
    recall_situated_memories_with_social_trust,
    simulate_situated_social_cognition,
    simulate_situated_social_cognitive_round,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedClaimStatus,
    initialize_situated_social_memory,
)
from tests.test_network_abm_situated_memory_cognition import (
    agent_model,
    inspection_and_move_story,
    mind,
    recall_model,
    with_telling,
)
from tests.test_network_abm_situated_social_memory_contracts import social_model


def testimony_social_case():
    cognition, story, inspection = inspection_and_move_story()
    story = with_telling(cognition, story, inspection)
    cue = SituatedMemoryRecallCue(
        "heard-restructuring",
        "The restructuring is approved.",
        event_kinds=(SituatedActionKind.TELL,),
        channels=(ObservationChannel.AUDITORY,),
    )
    memory_model = recall_model({"bob": (cue,)})
    model = replace(social_model(), memory_cognitive_model=memory_model)
    cognitive_state = initialize_situated_memory_cognition(memory_model, story)
    social_state = initialize_situated_social_memory(model, cognitive_state)
    return model, story, cognitive_state, social_state


def social_relationship(state, observer, source):
    return next(
        item for item in state.relationships
        if item.observer_agent_id == observer and item.source_agent_id == source
    )


class SituatedSocialCognitionTests(unittest.TestCase):
    def test_half_trusted_auditory_memory_has_literal_tempered_weight(self):
        model, story, cognitive_state, social_state = testimony_social_case()
        with TemporaryDirectory() as directory:
            database = f"{directory}/memory.sqlite3"
            ingest_situated_story(database, story, "bob", model.memory_cognitive_model.memory_policy)
            result = recall_situated_memories_with_social_trust(
                database,
                model,
                agent_model(model.memory_cognitive_model, "bob"),
                mind(cognitive_state, "bob"),
                story=story,
                cognitive_state=cognitive_state,
                social_state=social_state,
            )

        self.assertEqual(len(result.admissions), 1)
        admission = result.admissions[0]
        self.assertAlmostEqual(admission.source_trust, 0.5)
        self.assertAlmostEqual(admission.evidence_weight, 0.2625)
        self.assertAlmostEqual(
            admission.posterior_belief.probabilities["approved"],
            0.560431654676259,
        )

    def test_social_round_projects_recalled_testimony_into_one_directed_claim(self):
        model, story, cognitive_state, social_state = testimony_social_case()
        with TemporaryDirectory() as directory:
            result = simulate_situated_social_cognitive_round(
                f"{directory}/memory.sqlite3",
                model,
                story,
                cognitive_state,
                social_state,
            )

        self.assertEqual(len(result.social_update.admitted_evidence), 1)
        evidence = result.social_update.admitted_evidence[0]
        self.assertEqual(evidence.observer_agent_id, "bob")
        self.assertEqual(evidence.source_agent_id, "alice")
        self.assertEqual(evidence.topic_id, "restructuring")
        self.assertEqual(evidence.symbol_id, "approved")
        self.assertIsNotNone(evidence.memory_id)
        self.assertEqual(len(result.next_social_state.claims), 1)
        claim = result.next_social_state.claims[0]
        self.assertEqual(claim.status, SituatedClaimStatus.ACTIVE)
        self.assertEqual(claim.observer_agent_id, "bob")
        self.assertEqual(claim.source_agent_id, "alice")
        self.assertEqual(result.next_social_state.cognitive_state_hash, result.next_cognitive_state.content_hash)
        self.assertEqual(social_relationship(result.next_social_state, "bob", "alice").trust, 0.5)

    def test_social_trajectory_is_database_path_independent(self):
        model, story, cognitive_state, social_state = testimony_social_case()
        with TemporaryDirectory() as left_directory, TemporaryDirectory() as right_directory:
            left = simulate_situated_social_cognition(
                f"{left_directory}/memory.sqlite3",
                model,
                story,
                cognitive_state,
                social_state,
                round_count=2,
            )
            right = simulate_situated_social_cognition(
                f"{right_directory}/memory.sqlite3",
                model,
                story,
                cognitive_state,
                social_state,
                round_count=2,
            )

        self.assertEqual(left, right)
        self.assertEqual(left.content_hash, right.content_hash)
        self.assertEqual(left.final_social_state.round_index, social_state.round_index + 2)


if __name__ == "__main__":
    unittest.main()

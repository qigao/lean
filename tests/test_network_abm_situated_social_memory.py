from dataclasses import replace
import unittest

from narrative_dynamics.abm.situated import SituatedActionIntent, SituatedActionKind
from narrative_dynamics.abm.situated_contracts import initialize_situated_world
from narrative_dynamics.abm.situated_cognition_contracts import SituatedCognitiveState
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    initialize_situated_memory_cognition,
)
from narrative_dynamics.abm.situated_social_memory import advance_situated_social_memory
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedClaimStatus,
    SituatedSocialEvidence,
    SituatedSocialEvidenceKind,
    initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_story import advance_situated_story, initialize_situated_story
from tests.test_network_abm_situated_social_memory_contracts import social_model


def cognitive_checkpoint(model, round_index):
    cognition = model.memory_cognitive_model.cognitive_model
    story = initialize_situated_story(
        cognition.world_model,
        initialize_situated_world(cognition.world_model),
    )
    for index in range(1, round_index + 1):
        story = advance_situated_story(cognition.world_model, story, (
            SituatedActionIntent(f"wait-{index}", "alice", SituatedActionKind.WAIT),
        ))
    return initialize_situated_memory_cognition(model.memory_cognitive_model, story)


def cognitive_child(state):
    round_index = state.round_index + 1
    minds = tuple(
        replace(
            item,
            selected_action_ids=item.selected_action_ids + ("wait",),
            decision_count=item.decision_count + 1,
        )
        for item in state.minds
    )
    return SituatedCognitiveState(
        state.model_id,
        state.model_hash,
        round_index,
        state.content_hash,
        "sha256:" + f"{round_index:064x}",
        minds,
    )


def testimony(evidence_id, symbol, round_index, *, source="alice", observer="bob"):
    return SituatedSocialEvidence(
        evidence_id,
        SituatedSocialEvidenceKind.TESTIMONY,
        observer,
        "restructuring",
        symbol,
        round_index,
        f"event-{evidence_id}",
        source_agent_id=source,
        memory_id=f"memory-{evidence_id}",
    )


def verification(evidence_id, symbol, round_index, *, observer="bob"):
    return SituatedSocialEvidence(
        evidence_id,
        SituatedSocialEvidenceKind.VERIFICATION,
        observer,
        "restructuring",
        symbol,
        round_index,
        f"event-{evidence_id}",
    )


def relationship(state, observer, source):
    return next(
        item for item in state.relationships
        if item.observer_agent_id == observer and item.source_agent_id == source
    )


class SituatedSocialMemoryRuntimeTests(unittest.TestCase):
    def test_transition_requires_same_cognition_or_exact_one_round_child(self):
        model = social_model()
        cognition = cognitive_checkpoint(model, 1)
        initial = initialize_situated_social_memory(model, cognition)
        unrelated = replace(
            cognition,
            story_hash="sha256:" + "f" * 64,
        )
        jumped = cognitive_child(cognitive_child(cognition))

        with self.assertRaisesRegex(ValueError, "exact cognitive child"):
            advance_situated_social_memory(
                model, cognition, initial, unrelated, ()
            )
        with self.assertRaisesRegex(ValueError, "exact cognitive child"):
            advance_situated_social_memory(
                model, cognition, initial, jumped, ()
            )

    def test_same_batch_duplicate_evidence_is_idempotent_but_conflicts_fail(self):
        model = social_model()
        cognition = cognitive_checkpoint(model, 1)
        initial = initialize_situated_social_memory(model, cognition)
        item = testimony("duplicate", "approved", 1)

        result = advance_situated_social_memory(
            model, cognition, initial, cognition, (item, item)
        )

        self.assertEqual(result.admitted_evidence, (item,))
        self.assertEqual(result.next_state.claims[0].support_count, 1)
        with self.assertRaisesRegex(ValueError, "conflicting payloads"):
            advance_situated_social_memory(
                model,
                cognition,
                initial,
                cognition,
                (item, replace(item, symbol_id="denied")),
            )

    def test_earlier_verification_cannot_validate_later_testimony(self):
        model = social_model()
        cognition = cognitive_checkpoint(model, 2)
        initial = initialize_situated_social_memory(model, cognition)

        result = advance_situated_social_memory(
            model,
            cognition,
            initial,
            cognition,
            (
                verification("early-inspection", "approved", 1),
                testimony("later-claim", "approved", 2),
            ),
        )

        self.assertEqual(result.next_state.claims[0].status, SituatedClaimStatus.ACTIVE)
        self.assertEqual(relationship(result.next_state, "bob", "alice").trust, 0.5)

    def test_backdated_verification_cannot_validate_claim_from_a_later_round(self):
        model = social_model()
        cognition = cognitive_checkpoint(model, 3)
        initial = initialize_situated_social_memory(model, cognition)
        claimed = advance_situated_social_memory(
            model,
            cognition,
            initial,
            cognition,
            (testimony("later-claim", "approved", 3),),
        ).next_state

        result = advance_situated_social_memory(
            model,
            cognition,
            claimed,
            cognition,
            (verification("backdated-inspection", "approved", 2),),
        )

        self.assertEqual(result.next_state.claims[0].status, SituatedClaimStatus.ACTIVE)
        self.assertEqual(relationship(result.next_state, "bob", "alice").trust, 0.5)

    def test_backdated_opposite_testimony_cannot_supersede_a_newer_claim(self):
        model = social_model()
        cognition = cognitive_checkpoint(model, 3)
        initial = initialize_situated_social_memory(model, cognition)
        claimed = advance_situated_social_memory(
            model,
            cognition,
            initial,
            cognition,
            (testimony("newer-approved", "approved", 3),),
        ).next_state

        result = advance_situated_social_memory(
            model,
            cognition,
            claimed,
            cognition,
            (testimony("older-denied", "denied", 2),),
        )

        by_symbol = {item.symbol_id: item for item in result.next_state.claims}
        self.assertEqual(by_symbol["approved"].status, SituatedClaimStatus.ACTIVE)
        self.assertEqual(by_symbol["denied"].status, SituatedClaimStatus.SUPERSEDED)

    def test_backdated_same_symbol_cannot_merge_across_an_intermediate_revision(self):
        model = social_model()
        cognition = cognitive_checkpoint(model, 3)
        initial = initialize_situated_social_memory(model, cognition)
        revised = advance_situated_social_memory(
            model,
            cognition,
            initial,
            cognition,
            (
                testimony("denied-middle", "denied", 2),
                testimony("approved-new", "approved", 3),
            ),
        ).next_state
        inserted = advance_situated_social_memory(
            model,
            cognition,
            revised,
            cognition,
            (testimony("approved-old", "approved", 1),),
        ).next_state
        verified = advance_situated_social_memory(
            model,
            cognition,
            inserted,
            cognition,
            (verification("verify-middle", "approved", 2),),
        ).next_state

        approved = sorted(
            (item for item in verified.claims if item.symbol_id == "approved"),
            key=lambda item: item.first_round,
        )
        self.assertEqual(len(approved), 2)
        self.assertEqual(approved[0].status, SituatedClaimStatus.SUPERSEDED)
        self.assertEqual(approved[0].first_round, 1)
        self.assertEqual(approved[1].status, SituatedClaimStatus.ACTIVE)
        self.assertEqual(approved[1].first_round, 3)
        self.assertEqual(approved[1].support_count, 1)
        self.assertEqual(relationship(verified, "bob", "alice").trust, 0.5)

    def test_repeated_testimony_consolidates_and_opposite_testimony_supersedes(self):
        model = social_model()
        cognition = cognitive_checkpoint(model, 3)
        initial = initialize_situated_social_memory(model, cognition)

        repeated = advance_situated_social_memory(
            model,
            cognition,
            initial,
            cognition,
            (
                testimony("alice-approved-2", "approved", 2),
                testimony("alice-approved-1", "approved", 1),
            ),
        )
        revised = advance_situated_social_memory(
            model,
            cognition,
            repeated.next_state,
            cognition,
            (testimony("alice-denied-3", "denied", 3),),
        )

        self.assertEqual(len(repeated.next_state.claims), 1)
        consolidated = repeated.next_state.claims[0]
        self.assertEqual(consolidated.symbol_id, "approved")
        self.assertEqual(consolidated.support_count, 2)
        self.assertEqual(
            consolidated.event_ids,
            ("event-alice-approved-1", "event-alice-approved-2"),
        )
        self.assertEqual(len(revised.next_state.claims), 2)
        by_symbol = {item.symbol_id: item for item in revised.next_state.claims}
        self.assertEqual(by_symbol["approved"].status, SituatedClaimStatus.SUPERSEDED)
        self.assertEqual(by_symbol["denied"].status, SituatedClaimStatus.ACTIVE)

    def test_verification_learns_only_the_directed_observer_source_relationship(self):
        model = social_model()
        cognition = cognitive_checkpoint(model, 2)
        initial = initialize_situated_social_memory(model, cognition)
        claimed = advance_situated_social_memory(
            model,
            cognition,
            initial,
            cognition,
            (testimony("claim", "approved", 1),),
        ).next_state

        confirmed = advance_situated_social_memory(
            model,
            cognition,
            claimed,
            cognition,
            (verification("inspect", "approved", 2),),
        )
        replayed = advance_situated_social_memory(
            model,
            cognition,
            confirmed.next_state,
            cognition,
            (verification("inspect", "approved", 2),),
        )

        bob_alice = relationship(confirmed.next_state, "bob", "alice")
        self.assertAlmostEqual(bob_alice.trust, 0.6)
        self.assertAlmostEqual(bob_alice.affinity, 0.1)
        self.assertEqual(bob_alice.confirmation_count, 1)
        self.assertEqual(bob_alice.contradiction_count, 0)
        self.assertEqual(confirmed.next_state.claims[0].status, SituatedClaimStatus.CONFIRMED)
        self.assertEqual(relationship(confirmed.next_state, "alice", "bob").trust, 0.5)
        self.assertEqual(replayed.admitted_evidence, ())
        self.assertEqual(replayed.next_state, confirmed.next_state)

    def test_same_symbol_testimony_after_terminal_claim_opens_a_new_revision(self):
        model = social_model()
        cognition = cognitive_checkpoint(model, 3)
        initial = initialize_situated_social_memory(model, cognition)
        confirmed = advance_situated_social_memory(
            model,
            cognition,
            initial,
            cognition,
            (
                testimony("approved-first", "approved", 1),
                verification("verified-first", "approved", 2),
            ),
        ).next_state

        reopened = advance_situated_social_memory(
            model,
            cognition,
            confirmed,
            cognition,
            (testimony("approved-again", "approved", 3),),
        ).next_state

        approved = sorted(
            (item for item in reopened.claims if item.symbol_id == "approved"),
            key=lambda item: item.first_round,
        )
        self.assertEqual(len(approved), 2)
        self.assertEqual(approved[0].status, SituatedClaimStatus.CONFIRMED)
        self.assertEqual(approved[0].support_count, 1)
        self.assertEqual(approved[1].status, SituatedClaimStatus.ACTIVE)
        self.assertEqual(approved[1].first_round, 3)

    def test_contradictory_verification_decreases_trust_and_affinity_once(self):
        model = social_model()
        cognition = cognitive_checkpoint(model, 2)
        initial = initialize_situated_social_memory(model, cognition)
        claimed = advance_situated_social_memory(
            model,
            cognition,
            initial,
            cognition,
            (testimony("claim", "approved", 1),),
        ).next_state

        contradicted = advance_situated_social_memory(
            model,
            cognition,
            claimed,
            cognition,
            (verification("inspect", "denied", 2),),
        )

        bob_alice = relationship(contradicted.next_state, "bob", "alice")
        self.assertAlmostEqual(bob_alice.trust, 0.3)
        self.assertAlmostEqual(bob_alice.affinity, -0.2)
        self.assertEqual(bob_alice.confirmation_count, 0)
        self.assertEqual(bob_alice.contradiction_count, 1)
        self.assertEqual(contradicted.next_state.claims[0].status, SituatedClaimStatus.CONTRADICTED)

    def test_unresolved_claim_age_boundary_and_capacity_forget_deterministically(self):
        model = replace(
            social_model(),
            policy=replace(
                social_model().policy,
                max_unresolved_age_rounds=2,
                max_active_claims=2,
            ),
        )
        round_three = cognitive_checkpoint(model, 3)
        initial = initialize_situated_social_memory(model, round_three)
        populated = advance_situated_social_memory(
            model,
            round_three,
            initial,
            round_three,
            (
                testimony("oldest", "approved", 1, source="alice"),
                testimony("middle", "approved", 2, source="carol"),
                testimony("newest", "approved", 3, source="dana"),
            ),
        ).next_state

        by_source = {item.source_agent_id: item for item in populated.claims}
        self.assertEqual(by_source["alice"].status, SituatedClaimStatus.FORGOTTEN)
        self.assertEqual(by_source["carol"].status, SituatedClaimStatus.ACTIVE)
        self.assertEqual(by_source["dana"].status, SituatedClaimStatus.ACTIVE)

        round_four = cognitive_child(round_three)
        aged = advance_situated_social_memory(
            model,
            round_three,
            populated,
            round_four,
            (),
        ).next_state
        aged_by_source = {item.source_agent_id: item for item in aged.claims}
        self.assertEqual(aged_by_source["carol"].status, SituatedClaimStatus.ACTIVE)
        self.assertEqual(aged_by_source["dana"].status, SituatedClaimStatus.ACTIVE)

        round_five = cognitive_child(round_four)
        expired = advance_situated_social_memory(
            model,
            round_four,
            aged,
            round_five,
            (),
        ).next_state
        expired_by_source = {item.source_agent_id: item for item in expired.claims}
        self.assertEqual(expired_by_source["carol"].status, SituatedClaimStatus.FORGOTTEN)
        self.assertEqual(expired_by_source["dana"].status, SituatedClaimStatus.ACTIVE)


if __name__ == "__main__":
    unittest.main()

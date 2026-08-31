from dataclasses import replace
import unittest

from narrative_dynamics.abm.situated import SituatedActionIntent, SituatedActionKind
from narrative_dynamics.abm.situated_contracts import initialize_situated_world
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

        round_four = cognitive_checkpoint(model, 4)
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

        round_five = cognitive_checkpoint(model, 5)
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

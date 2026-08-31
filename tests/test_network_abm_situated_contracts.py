import unittest
from dataclasses import replace

from narrative_dynamics.abm.situated_contracts import (
    AgentBodyState,
    EmbodiedAgentSpec,
    EvidenceFact,
    PassageSpec,
    PlaceSpec,
    SituatedWorldModel,
    WorldObjectSpec,
    initialize_situated_world,
    validate_situated_state,
)
from tests.situated_fixtures import office_model, office_state


class SituatedWorldContractTests(unittest.TestCase):
    def test_initialization_is_canonical_and_content_hashed(self):
        model, state = office_state()
        self.assertEqual(tuple(item.place_id for item in model.places), ("lobby", "manager", "open", "records"))
        self.assertEqual(tuple(item.agent_id for item in state.agents), ("alice", "bob", "carol", "dana"))
        self.assertEqual(tuple(item.object_id for item in state.objects), ("badge", "memo"))
        self.assertEqual(tuple(item.passage_id for item in state.passages), tuple(sorted(item.passage_id for item in model.passages)))
        self.assertEqual(state.round_index, 0)
        self.assertIsNone(state.parent_state_hash)
        self.assertEqual(state.model_hash, model.content_hash)
        self.assertTrue(model.content_hash.startswith("sha256:"))
        self.assertTrue(state.content_hash.startswith("sha256:"))
        self.assertTrue(all(item.content_hash.startswith("sha256:") for item in model.places))
        self.assertTrue(all(item.content_hash.startswith("sha256:") for item in state.objects))
        validate_situated_state(model, state)

    def test_model_rejects_unknown_endpoints_and_initial_locations(self):
        base = office_model()
        with self.assertRaisesRegex(ValueError, "passage endpoint"):
            replace(base, passages=base.passages + (PassageSpec("bad", "open", "void"),))
        with self.assertRaisesRegex(ValueError, "agent initial place"):
            replace(base, agents=(EmbodiedAgentSpec("x", "visitor", "void", 0),))
        with self.assertRaisesRegex(ValueError, "object initial place"):
            replace(base, objects=(WorldObjectSpec("x", "note", "void"),))

    def test_model_rejects_duplicate_identity_and_bad_capacity(self):
        base = office_model()
        with self.assertRaisesRegex(ValueError, "place ids must be unique"):
            replace(base, places=base.places + (PlaceSpec("open", "Duplicate"),))
        with self.assertRaisesRegex(ValueError, "inventory capacity"):
            EmbodiedAgentSpec("x", "visitor", "open", -1)

    def test_state_requires_exclusive_object_location(self):
        model, state = office_state()
        memo = next(item for item in state.objects if item.object_id == "memo")
        with self.assertRaisesRegex(ValueError, "exactly one place or holder"):
            replace(state, objects=tuple(
                replace(item, holder_agent_id="alice") if item == memo else item
                for item in state.objects
            ))

    def test_state_requires_exact_rosters_and_inventory_capacity(self):
        model, state = office_state(alice_capacity=0)
        with self.assertRaisesRegex(ValueError, "agent roster"):
            validate_situated_state(model, replace(state, agents=state.agents[:-1]))
        forged_objects = tuple(
            replace(item, place_id=None, holder_agent_id="alice") if item.object_id == "memo" else item
            for item in state.objects
        )
        with self.assertRaisesRegex(ValueError, "inventory capacity"):
            validate_situated_state(model, replace(state, objects=forged_objects))

    def test_non_initial_state_requires_hash_parent(self):
        _, state = office_state()
        with self.assertRaisesRegex(ValueError, "parent state hash"):
            replace(state, round_index=1, parent_state_hash=None)
        with self.assertRaisesRegex(ValueError, "round zero"):
            replace(state, parent_state_hash="sha256:" + "a" * 64)

    def test_value_contracts_validate_text_and_evidence(self):
        with self.assertRaises(ValueError):
            PlaceSpec("", "Room")
        with self.assertRaises(ValueError):
            EvidenceFact("fact", "")


if __name__ == "__main__":
    unittest.main()

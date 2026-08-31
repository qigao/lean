from __future__ import annotations

import math
import unittest
from types import MappingProxyType

from narrative_dynamics.abm.situated import SituatedActionKind, ObservationChannel
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedAgentPerceptionProfile,
    SituatedEdgeActivation,
    SituatedEventSignalProfile,
    SituatedPerceptionEdge,
    SituatedPerceptionLayer,
    SituatedPerceptFidelity,
    SituatedPerceptionModel,
    SituatedPerceptionReach,
    SituatedPercept,
    SituatedPerceptualProjection,
)
from narrative_dynamics.abm.situated_contracts import (
    EvidenceFact,
    EmbodiedAgentSpec,
    PassageSpec,
    PlaceSpec,
    SituatedWorldModel,
)


def office_world() -> SituatedWorldModel:
    return SituatedWorldModel(
        "office",
        "1",
        (PlaceSpec("lobby", "Lobby"), PlaceSpec("corridor", "Corridor"),
         PlaceSpec("meeting", "Meeting"), PlaceSpec("records", "Records")),
        (PassageSpec("meeting-door", "corridor", "meeting", initially_open=False),),
        (EmbodiedAgentSpec("alice", "staff", "lobby"), EmbodiedAgentSpec("bob", "staff", "meeting")),
    )


def office_edges() -> tuple[SituatedPerceptionEdge, ...]:
    return (
        SituatedPerceptionEdge("z-interaction", SituatedPerceptionLayer.INTERACTION, "meeting", "meeting", 0.0),
        SituatedPerceptionEdge("meeting-door-closed-audio", SituatedPerceptionLayer.AUDITORY, "corridor", "meeting", 25.0, SituatedEdgeActivation.PASSAGE_CLOSED, "meeting-door"),
        SituatedPerceptionEdge("lobby-corridor-visual", SituatedPerceptionLayer.VISIBILITY, "lobby", "corridor", 1.0),
    )


def office_profiles() -> tuple[SituatedAgentPerceptionProfile, ...]:
    return (
        SituatedAgentPerceptionProfile("bob", 2.0, 20.0, 45.0),
        SituatedAgentPerceptionProfile("alice", 3.0, 10.0, 30.0),
    )


def office_signals() -> tuple[SituatedEventSignalProfile, ...]:
    return (SituatedEventSignalProfile(SituatedActionKind.TELL, True, 60.0),)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64


class SituatedPerceptionLeafContractTests(unittest.TestCase):
    def test_public_enum_values_are_stable(self) -> None:
        self.assertEqual(
            [item.value for item in SituatedPerceptionLayer],
            ["visibility", "auditory", "interaction"],
        )
        self.assertEqual(
            [item.value for item in SituatedEdgeActivation],
            ["always", "passage_open", "passage_closed"],
        )
        self.assertEqual(
            [item.value for item in SituatedPerceptFidelity],
            ["detected", "identified", "exact"],
        )

    def test_edge_and_profiles_construct_and_canonicalize_numbers(self) -> None:
        edge = SituatedPerceptionEdge(
            "meeting-door-closed-audio",
            SituatedPerceptionLayer.AUDITORY,
            "corridor",
            "meeting",
            25,
            SituatedEdgeActivation.PASSAGE_CLOSED,
            "meeting-door",
        )
        profile = SituatedAgentPerceptionProfile(
            "bob",
            max_visual_cost=2,
            minimum_detectable_sound=20,
            minimum_clear_sound=45,
        )
        signal = SituatedEventSignalProfile(
            SituatedActionKind.TELL,
            visually_observable=True,
            auditory_intensity=60,
        )
        self.assertEqual(edge.cost, 25.0)
        self.assertEqual(profile.max_visual_cost, 2.0)
        self.assertEqual(profile.minimum_detectable_sound, 20.0)
        self.assertEqual(profile.minimum_clear_sound, 45.0)
        self.assertEqual(signal.auditory_intensity, 60.0)
        self.assertTrue(edge.content_hash.startswith("sha256:"))
        self.assertTrue(profile.content_hash.startswith("sha256:"))
        self.assertTrue(signal.content_hash.startswith("sha256:"))

    def test_edge_cost_and_activation_contract(self) -> None:
        for cost in (-1.0, math.inf, math.nan):
            with self.subTest(cost=cost):
                with self.assertRaises((ValueError, TypeError)):
                    SituatedPerceptionEdge("e", SituatedPerceptionLayer.VISIBILITY, "a", "b", cost)
        with self.assertRaises((ValueError, TypeError)):
            SituatedPerceptionEdge(
                "e", SituatedPerceptionLayer.INTERACTION, "a", "b", 1.0
            )
        with self.assertRaises((ValueError, TypeError)):
            SituatedPerceptionEdge(
                "e", SituatedPerceptionLayer.VISIBILITY, "a", "b", 1.0,
                SituatedEdgeActivation.ALWAYS, "passage",
            )
        with self.assertRaises((ValueError, TypeError)):
            SituatedPerceptionEdge(
                "e", SituatedPerceptionLayer.VISIBILITY, "a", "b", 1.0,
                SituatedEdgeActivation.PASSAGE_OPEN,
            )

    def test_profile_threshold_and_signal_intensity_contracts(self) -> None:
        with self.assertRaises((ValueError, TypeError)):
            SituatedAgentPerceptionProfile("bob", 1.0, 40.0, 20.0)
        for intensity in (math.inf, math.nan):
            with self.subTest(intensity=intensity):
                with self.assertRaises((ValueError, TypeError)):
                    SituatedEventSignalProfile(SituatedActionKind.TELL, True, intensity)


class SituatedPerceptionModelContractTests(unittest.TestCase):
    def test_model_requires_closed_world_references_and_exact_agent_profiles(self) -> None:
        world = office_world()
        with self.assertRaises((ValueError, TypeError)):
            SituatedPerceptionModel("m", "1", world, (
                SituatedPerceptionEdge("bad", SituatedPerceptionLayer.VISIBILITY, "unknown", "lobby", 1.0),
            ), office_profiles(), office_signals())
        with self.assertRaises((ValueError, TypeError)):
            SituatedPerceptionModel("m", "1", world, (
                SituatedPerceptionEdge("bad", SituatedPerceptionLayer.VISIBILITY, "lobby", "corridor", 1.0, SituatedEdgeActivation.PASSAGE_CLOSED, "unknown"),
            ), office_profiles(), office_signals())
        with self.assertRaises((ValueError, TypeError)):
            SituatedPerceptionModel("m", "1", world, office_edges(), (office_profiles()[0],), office_signals())
        with self.assertRaises((ValueError, TypeError)):
            SituatedPerceptionModel("m", "1", world, office_edges(), office_profiles() + (office_profiles()[0],), office_signals())

    def test_model_rejects_duplicate_edge_and_signal_identities(self) -> None:
        world = office_world()
        duplicate_edge = office_edges() + (office_edges()[0],)
        with self.assertRaises((ValueError, TypeError)):
            SituatedPerceptionModel("m", "1", world, duplicate_edge, office_profiles(), office_signals())
        duplicate_signal = office_signals() + (SituatedEventSignalProfile(SituatedActionKind.TELL, False, None),)
        with self.assertRaises((ValueError, TypeError)):
            SituatedPerceptionModel("m", "1", world, office_edges(), office_profiles(), duplicate_signal)

    def test_model_canonicalizes_order_and_hashes_only_world_identity(self) -> None:
        world = office_world()
        left = SituatedPerceptionModel("m", "1", world, office_edges(), office_profiles(), office_signals())
        right = SituatedPerceptionModel("m", "1", world, tuple(reversed(office_edges())), tuple(reversed(office_profiles())), office_signals())
        self.assertEqual(left, right)
        self.assertEqual(left.content_hash, right.content_hash)
        self.assertEqual(tuple(item.edge_id for item in left.edges), tuple(sorted(item.edge_id for item in left.edges)))
        self.assertEqual(tuple(item.agent_id for item in left.agent_profiles), ("alice", "bob"))
        self.assertEqual(tuple(item.kind.value for item in left.signal_profiles), ("tell",))
        self.assertEqual(left.to_dict()["world_model_hash"], world.content_hash)
        self.assertNotIn("world_model", left.to_dict())


class SituatedPerceptionReachAndPerceptContractTests(unittest.TestCase):
    def test_reach_freezes_sorted_finite_cost_mappings_and_interactions(self) -> None:
        reach = SituatedPerceptionReach(
            HASH_A, HASH_B, "lobby",
            {"records": 3, "corridor": 1},
            {"records": 7, "corridor": 2},
            ("meeting", "corridor"),
        )
        self.assertIsInstance(reach.visual_costs, MappingProxyType)
        self.assertEqual(tuple(reach.visual_costs), ("corridor", "records"))
        self.assertEqual(tuple(reach.auditory_losses), ("corridor", "records"))
        self.assertEqual(reach.interaction_place_ids, ("corridor", "meeting"))
        self.assertEqual(reach.visual_costs["records"], 3.0)
        with self.assertRaises((ValueError, TypeError)):
            SituatedPerceptionReach(HASH_A, HASH_B, "lobby", {"x": math.inf}, {}, ())

    def test_percept_fidelity_controls_sanitized_fields(self) -> None:
        detected = SituatedPercept("p1", 1, "bob", "event-1", HASH_C, (ObservationChannel.AUDITORY,), SituatedPerceptFidelity.DETECTED)
        self.assertIsNone(detected.actor_agent_id)
        identified = SituatedPercept("p2", 1, "bob", "event-1", HASH_C, (ObservationChannel.VISUAL,), SituatedPerceptFidelity.IDENTIFIED, "alice", SituatedActionKind.TELL, "meeting")
        self.assertEqual(identified.actor_agent_id, "alice")
        exact = SituatedPercept(
            "p3", 1, "bob", "event-1", HASH_C,
            (ObservationChannel.VISUAL, ObservationChannel.AUDITORY),
            SituatedPerceptFidelity.EXACT, "alice", SituatedActionKind.TELL,
            "meeting", "said hello", (EvidenceFact("z", "2"), EvidenceFact("a", "1")),
        )
        self.assertEqual(exact.channels, (ObservationChannel.AUDITORY, ObservationChannel.VISUAL))
        self.assertEqual(tuple(item.name for item in exact.details), ("a", "z"))
        with self.assertRaises((ValueError, TypeError)):
            SituatedPercept("p4", 1, "bob", "event-1", HASH_C, (), SituatedPerceptFidelity.DETECTED)
        with self.assertRaises((ValueError, TypeError)):
            SituatedPercept("p5", 1, "bob", "event-1", HASH_C, (ObservationChannel.VISUAL, ObservationChannel.VISUAL), SituatedPerceptFidelity.DETECTED)
        with self.assertRaises((ValueError, TypeError)):
            SituatedPercept("p6", 1, "bob", "event-1", "not-a-hash", (ObservationChannel.VISUAL,), SituatedPerceptFidelity.DETECTED)

    def test_percept_fidelity_rejects_leaked_or_incomplete_fields(self) -> None:
        with self.assertRaises((ValueError, TypeError)):
            SituatedPercept("p", 1, "bob", "event", HASH_C, (ObservationChannel.VISUAL,), SituatedPerceptFidelity.DETECTED, actor_agent_id="alice")
        with self.assertRaises((ValueError, TypeError)):
            SituatedPercept("p", 1, "bob", "event", HASH_C, (ObservationChannel.VISUAL,), SituatedPerceptFidelity.IDENTIFIED, "alice", None, "meeting")
        with self.assertRaises((ValueError, TypeError)):
            SituatedPercept("p", 1, "bob", "event", HASH_C, (ObservationChannel.VISUAL,), SituatedPerceptFidelity.IDENTIFIED, "alice", SituatedActionKind.TELL, "meeting", "leak")
        with self.assertRaises((ValueError, TypeError)):
            SituatedPercept("p", 1, "bob", "event", HASH_C, (ObservationChannel.VISUAL,), SituatedPerceptFidelity.EXACT, "alice", SituatedActionKind.TELL, "meeting")

    def test_projection_canonicalizes_percepts_and_rejects_duplicate_identity(self) -> None:
        first = SituatedPercept("p1", 1, "bob", "z-event", HASH_C, (ObservationChannel.VISUAL,), SituatedPerceptFidelity.DETECTED)
        second = SituatedPercept("p2", 1, "alice", "a-event", HASH_C, (ObservationChannel.AUDITORY,), SituatedPerceptFidelity.DETECTED)
        projection = SituatedPerceptualProjection("m", HASH_A, HASH_B, HASH_C, (first, second))
        self.assertEqual(tuple(item.source_event_id for item in projection.percepts), ("a-event", "z-event"))
        duplicate_id = SituatedPercept("p1", 1, "alice", "other-event", HASH_C, (ObservationChannel.VISUAL,), SituatedPerceptFidelity.DETECTED)
        with self.assertRaises((ValueError, TypeError)):
            SituatedPerceptualProjection("m", HASH_A, HASH_B, HASH_C, (first, duplicate_id))
        duplicate_pair = SituatedPercept("p3", 1, "bob", "z-event", HASH_C, (ObservationChannel.VISUAL,), SituatedPerceptFidelity.DETECTED)
        with self.assertRaises((ValueError, TypeError)):
            SituatedPerceptualProjection("m", HASH_A, HASH_B, HASH_C, (first, duplicate_pair))


if __name__ == "__main__":
    unittest.main()

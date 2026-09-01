from __future__ import annotations

import math
import unittest

import narrative_dynamics.abm.situated_projection_contracts as projection_contracts
from narrative_dynamics.abm.situated_projection_contracts import (
    NarrativeAuthority,
    NarrativeBeat,
    NarrativeBeatPhase,
    NarrativeBeatKind,
    NarrativeCut,
    NarrativeEntitlement,
    NarrativeEntitlementScope,
    NarrativeFact,
    NarrativeProjection,
    NarrativeProjectionPolicy,
    NarrativeScene,
    NarrativeSupportRef,
    NarrativeTemporalOrder,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
POLICY = NarrativeProjectionPolicy("objective", "1.0", NarrativeAuthority.OBJECTIVE)


def entitlement(entitlement_id: str = "entitlement-1") -> NarrativeEntitlement:
    return NarrativeEntitlement(
        entitlement_id,
        NarrativeEntitlementScope.OBJECTIVE,
        None,
        1,
        (NarrativeFact("event", "arrived"),),
        (NarrativeSupportRef("world_event", "event-1", HASH_A),),
    )


def beat(
    beat_id: str = "beat-1",
    *,
    entitlement_id: str = "entitlement-1",
    round_index: int = 1,
    sequence: int = 1,
    place_id: str = "office",
    active_pov_agent_id: str | None = None,
    source_event_id: str | None = None,
) -> NarrativeBeat:
    args = (
        beat_id,
        NarrativeBeatKind.PHYSICAL,
        round_index,
        sequence,
        place_id,
        active_pov_agent_id,
        ("alice",),
        1.0,
        (NarrativeSupportRef("world_event", "event-1", HASH_A),),
        (entitlement_id,),
        (),
    )
    if source_event_id is None:
        return NarrativeBeat(*args)
    return NarrativeBeat(*args, source_event_id=source_event_id)


def scene(scene_id: str = "scene-1", beat_ids: tuple[str, ...] = ("beat-1",)) -> NarrativeScene:
    return NarrativeScene(scene_id, beat_ids, 1, 1, "office", None)


class NarrativeProjectionContractTests(unittest.TestCase):
    def test_narrative_beat_phase_is_public_and_world_is_the_compatible_default(self):
        self.assertTrue(hasattr(projection_contracts, "NarrativeBeatPhase"))
        phase = projection_contracts.NarrativeBeatPhase
        self.assertEqual(
            [item.value for item in phase],
            ["memory_recall", "belief", "decision", "world", "social"],
        )
        self.assertIs(beat().phase, phase.WORLD)
        self.assertEqual(beat().to_dict()["phase"], "world")

    def test_limited_policy_requires_exactly_one_named_agent(self):
        with self.assertRaisesRegex(ValueError, "exactly one POV agent"):
            NarrativeProjectionPolicy(
                "bob-cut", "1.0", NarrativeAuthority.AGENT_LIMITED,
                pov_agent_ids=(),
            )

    def test_authored_order_requires_unique_event_ids(self):
        with self.assertRaisesRegex(ValueError, "authored event order"):
            NarrativeProjectionPolicy(
                "flashback", "1.0", NarrativeAuthority.OBJECTIVE,
                temporal_order=NarrativeTemporalOrder.AUTHORED,
                authored_event_order=("event-2", "event-2"),
            )

    def test_projection_rejects_beat_support_outside_entitlement_bundle(self):
        support = NarrativeSupportRef("world_event", "event-1", HASH_A)
        selected_beat = NarrativeBeat(
            "beat-1", NarrativeBeatKind.PHYSICAL,
            1, 1, "office", None, ("alice",),
            1.0, (support,), ("missing-entitlement",), (),
        )
        selected_scene = NarrativeScene("scene-1", ("beat-1",), 1, 1, "office", None)
        cut = NarrativeCut("cut-1", ("scene-1",), ())
        with self.assertRaisesRegex(ValueError, "entitlement"):
            NarrativeProjection(HASH_A, None, POLICY, (selected_beat,), (selected_scene,), cut)

    def test_projection_rejects_known_entitlement_with_mismatching_support_triple(self):
        selected_beat = beat()
        mismatching_entitlement = NarrativeEntitlement(
            "entitlement-1",
            NarrativeEntitlementScope.OBJECTIVE,
            None,
            1,
            (NarrativeFact("event", "arrived"),),
            (NarrativeSupportRef("world_event", "event-1", HASH_B),),
        )
        with self.assertRaisesRegex(ValueError, "support.*entitlement"):
            NarrativeProjection(
                HASH_A,
                None,
                POLICY,
                (selected_beat,),
                (scene(),),
                NarrativeCut("cut-1", ("scene-1",), (mismatching_entitlement,)),
            )

    def test_content_hash_is_stable_under_input_mapping_order(self):
        left = NarrativeProjectionPolicy(
            "stable", "1.0", NarrativeAuthority.OBJECTIVE,
            salience_weights={
                NarrativeBeatKind.PHYSICAL: 0.5,
                NarrativeBeatKind.INFORMATION: 0.8,
            },
        )
        right = NarrativeProjectionPolicy(
            "stable", "1.0", NarrativeAuthority.OBJECTIVE,
            salience_weights={
                NarrativeBeatKind.INFORMATION: 0.8,
                NarrativeBeatKind.PHYSICAL: 0.5,
            },
        )
        self.assertEqual(left.content_hash, right.content_hash)
        self.assertEqual(1.0, left.weight_for(NarrativeBeatKind.CAUSAL_PAYOFF))

    def test_objective_policy_rejects_pov_agents(self):
        with self.assertRaisesRegex(ValueError, "objective"):
            NarrativeProjectionPolicy(
                "objective", "1.0", NarrativeAuthority.OBJECTIVE,
                pov_agent_ids=("alice",),
            )

    def test_multi_pov_policy_requires_two_unique_agents(self):
        with self.assertRaisesRegex(ValueError, "at least two unique POV agents"):
            NarrativeProjectionPolicy(
                "shared", "1.0", NarrativeAuthority.MULTI_POV,
                pov_agent_ids=("alice", "alice"),
            )

    def test_chronological_policy_rejects_authored_ids(self):
        with self.assertRaisesRegex(ValueError, "chronological"):
            NarrativeProjectionPolicy(
                "linear", "1.0", NarrativeAuthority.OBJECTIVE,
                authored_event_order=("event-1",),
            )

    def test_policy_numeric_fields_are_finite_and_bounded(self):
        invalid_values = (
            {"minimum_salience": math.inf},
            {"maximum_event_omission_gap": -1},
            {"required_causal_coverage": 1.1},
            {"scene_round_gap": 0},
            {"maximum_scene_beats": 0},
            {"salience_weights": {NarrativeBeatKind.PHYSICAL: math.nan}},
        )
        for kwargs in invalid_values:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NarrativeProjectionPolicy("bounded", "1.0", NarrativeAuthority.OBJECTIVE, **kwargs)

    def test_entitlement_sorts_facts_and_supports_and_rejects_duplicate_fact_keys(self):
        ordered = NarrativeEntitlement(
            "entitlement-1",
            NarrativeEntitlementScope.OBJECTIVE,
            None,
            1,
            (NarrativeFact("z", "last"), NarrativeFact("a", "first")),
            (
                NarrativeSupportRef("world_event", "event-2", HASH_B),
                NarrativeSupportRef("world_event", "event-1", HASH_A),
            ),
        )
        self.assertEqual(("a", "z"), tuple(fact.key for fact in ordered.facts))
        self.assertEqual(("event-1", "event-2"), tuple(ref.artifact_id for ref in ordered.supporting_artifacts))
        with self.assertRaisesRegex(ValueError, "fact keys"):
            NarrativeEntitlement(
                "duplicate", NarrativeEntitlementScope.OBJECTIVE, None, 1,
                (NarrativeFact("same", "one"), NarrativeFact("same", "two")), (),
            )

    def test_entitlement_scope_enforces_owner(self):
        with self.assertRaisesRegex(ValueError, "private.*owner"):
            NarrativeEntitlement("private", NarrativeEntitlementScope.PRIVATE, None, 1, (), ())
        with self.assertRaisesRegex(ValueError, "objective.*owner"):
            NarrativeEntitlement("objective", NarrativeEntitlementScope.OBJECTIVE, "alice", 1, (), ())

    def test_content_addressed_values_require_sha256_hashes(self):
        with self.assertRaisesRegex(ValueError, "sha256"):
            NarrativeSupportRef("world_event", "event-1", "not-a-hash")
        value = NarrativeSupportRef("world_event", "event-1", HASH_A)
        self.assertRegex(value.content_hash, r"^sha256:[0-9a-f]{64}$")

    def test_beat_rejects_duplicate_support_and_cause_ids(self):
        support = NarrativeSupportRef("world_event", "event-1", HASH_A)
        with self.assertRaisesRegex(ValueError, "support"):
            NarrativeBeat(
                "beat-1", NarrativeBeatKind.PHYSICAL, 1, 1, "office", None,
                (), 1.0, (support, support), ("entitlement-1",), (),
            )
        with self.assertRaisesRegex(ValueError, "cause"):
            NarrativeBeat(
                "beat-1", NarrativeBeatKind.PHYSICAL, 1, 1, "office", None,
                (), 1.0, (support,), ("entitlement-1",), ("beat-0", "beat-0"),
            )

    def test_scene_and_cut_reject_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "beat ids"):
            NarrativeScene("scene-1", ("beat-1", "beat-1"), 1, 1, "office", None)
        with self.assertRaisesRegex(ValueError, "scene ids"):
            NarrativeCut("cut-1", ("scene-1", "scene-1"), ())

    def test_projection_rejects_dangling_beat_scene_and_cause_ids(self):
        valid_entitlement = entitlement()
        valid_cut = NarrativeCut("cut-1", ("scene-1",), (valid_entitlement,))
        with self.assertRaisesRegex(ValueError, "scene.*beat"):
            NarrativeProjection(HASH_A, None, POLICY, (beat(),), (scene("scene-1", ("missing",)),), valid_cut)
        with self.assertRaisesRegex(ValueError, "cause"):
            NarrativeProjection(
                HASH_A, None, POLICY,
                (NarrativeBeat("beat-1", NarrativeBeatKind.PHYSICAL, 1, 1, "office", None, (), 1.0,
                               (NarrativeSupportRef("world_event", "event-1", HASH_A),), ("entitlement-1",), ("missing",)),),
                (scene(),), valid_cut,
            )
        with self.assertRaisesRegex(ValueError, "cut.*scene"):
            NarrativeProjection(HASH_A, None, POLICY, (beat(),), (scene(),), NarrativeCut("cut", ("missing",), (valid_entitlement,)))

    def test_projection_requires_complete_single_scene_and_cut_membership(self):
        valid_entitlement = entitlement()
        two_beats = (beat("beat-1"), beat("beat-2"))
        with self.assertRaisesRegex(ValueError, "exactly one scene"):
            NarrativeProjection(
                HASH_A, None, POLICY, two_beats, (scene("scene-1", ("beat-1",)),),
                NarrativeCut("cut-1", ("scene-1",), (valid_entitlement,)),
            )
        two_scenes = (scene("scene-1", ("beat-1",)), scene("scene-2", ("beat-2",)))
        with self.assertRaisesRegex(ValueError, "exactly one cut"):
            NarrativeProjection(
                HASH_A, None, POLICY, two_beats, two_scenes,
                NarrativeCut("cut-1", ("scene-1",), (valid_entitlement,)),
            )

    def test_multi_pov_active_beat_cannot_cite_another_agents_private_entitlement(self):
        multi_policy = NarrativeProjectionPolicy(
            "shared", "1.0", NarrativeAuthority.MULTI_POV, pov_agent_ids=("alice", "bob"),
        )
        alice_entitlement = NarrativeEntitlement(
            "alice-private", NarrativeEntitlementScope.PRIVATE, "alice", 1,
            (NarrativeFact("secret", "memo"),),
            (NarrativeSupportRef("percept", "alice-1", HASH_A),),
        )
        bob_beat = NarrativeBeat(
            "beat-1", NarrativeBeatKind.INFORMATION, 1, 1, "office", "bob", ("bob",),
            1.0, (NarrativeSupportRef("percept", "alice-1", HASH_A),), ("alice-private",), (),
        )
        bob_scene = NarrativeScene("scene-1", ("beat-1",), 1, 1, "office", "bob")
        with self.assertRaisesRegex(ValueError, "active POV"):
            NarrativeProjection(
                HASH_A, None, multi_policy, (bob_beat,), (bob_scene,),
                NarrativeCut("cut-1", ("scene-1",), (alice_entitlement,)),
            )

    def test_objective_active_beat_cannot_cite_another_agents_private_entitlement(self):
        bob_entitlement = NarrativeEntitlement(
            "bob-private", NarrativeEntitlementScope.PRIVATE, "bob", 1,
            (NarrativeFact("belief", "approved"),),
            (NarrativeSupportRef("cognitive_decision", "bob:1", HASH_A),),
        )
        alice_beat = NarrativeBeat(
            "beat-1", NarrativeBeatKind.BELIEF_SHIFT, 1, 1, "office", "alice",
            ("alice",), 1.0,
            (NarrativeSupportRef("cognitive_decision", "bob:1", HASH_A),),
            ("bob-private",), (), phase=NarrativeBeatPhase.BELIEF,
        )
        alice_scene = NarrativeScene("scene-1", ("beat-1",), 1, 1, "office", "alice")
        with self.assertRaisesRegex(ValueError, "active POV"):
            NarrativeProjection(
                HASH_A, None, POLICY, (alice_beat,), (alice_scene,),
                NarrativeCut("cut-1", ("scene-1",), (bob_entitlement,)),
            )

    def test_objective_internal_beat_may_cite_its_owners_private_entitlement(self):
        alice_entitlement = NarrativeEntitlement(
            "alice-private", NarrativeEntitlementScope.PRIVATE, "alice", 1,
            (NarrativeFact("belief", "approved"),),
            (NarrativeSupportRef("cognitive_decision", "alice:1", HASH_A),),
        )
        alice_beat = NarrativeBeat(
            "beat-1", NarrativeBeatKind.BELIEF_SHIFT, 1, 1, "office", "alice",
            ("alice",), 1.0,
            (NarrativeSupportRef("cognitive_decision", "alice:1", HASH_A),),
            ("alice-private",), (), phase=NarrativeBeatPhase.BELIEF,
        )
        alice_scene = NarrativeScene("scene-1", ("beat-1",), 1, 1, "office", "alice")
        projection = NarrativeProjection(
            HASH_A, None, POLICY, (alice_beat,), (alice_scene,),
            NarrativeCut("cut-1", ("scene-1",), (alice_entitlement,)),
        )
        self.assertEqual("alice", projection.beats[0].active_pov_agent_id)

    def test_authored_policy_requires_every_beat_to_bind_a_source_event(self):
        authored_policy = NarrativeProjectionPolicy(
            "flashback", "1.0", NarrativeAuthority.OBJECTIVE,
            temporal_order=NarrativeTemporalOrder.AUTHORED,
            authored_event_order=("event-1",),
        )
        with self.assertRaisesRegex(ValueError, "source event"):
            NarrativeProjection(
                HASH_A, None, authored_policy, (beat(),), (scene(),),
                NarrativeCut("cut-1", ("scene-1",), (entitlement(),)),
            )
        with self.assertRaisesRegex(ValueError, "source event"):
            NarrativeProjection(
                HASH_A, None, authored_policy,
                (beat(source_event_id="unlisted-event"),), (scene(),),
                NarrativeCut("cut-1", ("scene-1",), (entitlement(),)),
            )

    def test_authored_order_binds_explicit_source_events_without_rewriting_beat_time(self):
        authored_policy = NarrativeProjectionPolicy(
            "flashback", "1.0", NarrativeAuthority.OBJECTIVE,
            temporal_order=NarrativeTemporalOrder.AUTHORED,
            authored_event_order=("event-2", "event-1"),
        )
        first = beat("beat-1", source_event_id="event-1", round_index=1, sequence=3)
        second = beat("beat-2", source_event_id="event-2", round_index=2, sequence=4)
        projection = NarrativeProjection(
            HASH_A, None, authored_policy, (second, first),
            (
                NarrativeScene("scene-2", ("beat-2",), 2, 2, "office", None),
                scene("scene-1", ("beat-1",)),
            ),
            NarrativeCut("cut-1", ("scene-2", "scene-1"), (entitlement(),)),
        )
        self.assertEqual((2, 4), (projection.beats[0].round_index, projection.beats[0].sequence))
        self.assertEqual((1, 3), (projection.beats[1].round_index, projection.beats[1].sequence))
        self.assertEqual(("event-2", "event-1"), tuple(beat.source_event_id for beat in projection.beats))

    def test_projection_rejects_top_level_beat_catalog_outside_cut_presentation_order(self):
        first = beat("beat-1", source_event_id="event-1", round_index=1)
        second = beat("beat-2", source_event_id="event-2", round_index=2)
        scenes = (
            NarrativeScene("scene-1", ("beat-1",), 1, 1, "office", None),
            NarrativeScene("scene-2", ("beat-2",), 2, 2, "office", None),
        )
        with self.assertRaisesRegex(ValueError, "beat.*presentation order"):
            NarrativeProjection(
                HASH_A, None, POLICY, (second, first), scenes,
                NarrativeCut("cut-1", ("scene-1", "scene-2"), (entitlement(),)),
            )

    def test_projection_rejects_top_level_scene_catalog_outside_cut_presentation_order(self):
        first = beat("beat-1", source_event_id="event-1", round_index=1)
        second = beat("beat-2", source_event_id="event-2", round_index=2)
        first_scene = NarrativeScene("scene-1", ("beat-1",), 1, 1, "office", None)
        second_scene = NarrativeScene("scene-2", ("beat-2",), 2, 2, "office", None)
        with self.assertRaisesRegex(ValueError, "scene.*presentation order"):
            NarrativeProjection(
                HASH_A, None, POLICY, (first, second), (second_scene, first_scene),
                NarrativeCut("cut-1", ("scene-1", "scene-2"), (entitlement(),)),
            )

    def test_chronological_projection_rejects_reversed_cut_order(self):
        first = beat("beat-1", source_event_id="event-1", round_index=1)
        second = beat("beat-2", source_event_id="event-2", round_index=2)
        first_scene = NarrativeScene("scene-1", ("beat-1",), 1, 1, "office", None)
        second_scene = NarrativeScene("scene-2", ("beat-2",), 2, 2, "office", None)
        with self.assertRaisesRegex(ValueError, "chronological"):
            NarrativeProjection(
                HASH_A, None, POLICY, (second, first), (second_scene, first_scene),
                NarrativeCut("cut-1", ("scene-2", "scene-1"), (entitlement(),)),
            )

    def test_scene_metadata_must_match_all_member_beats(self):
        selected_beat = beat(active_pov_agent_id="bob")
        valid_cut = NarrativeCut("cut-1", ("scene-1",), (entitlement(),))
        invalid_scenes = (
            NarrativeScene("scene-1", ("beat-1",), 0, 1, "office", "bob"),
            NarrativeScene("scene-1", ("beat-1",), 1, 1, "corridor", "bob"),
            NarrativeScene("scene-1", ("beat-1",), 1, 1, "office", "alice"),
        )
        for inconsistent_scene in invalid_scenes:
            with self.subTest(scene=inconsistent_scene), self.assertRaisesRegex(ValueError, "scene.*match"):
                NarrativeProjection(HASH_A, None, POLICY, (selected_beat,), (inconsistent_scene,), valid_cut)

    def test_unknown_place_is_optional_content_addressed_metadata(self):
        unknown_beat = NarrativeBeat(
            "beat-1", NarrativeBeatKind.INFORMATION, 1, 1, None, "bob",
            ("bob",), 1.0,
            (NarrativeSupportRef("world_event", "event-1", HASH_A),),
            ("entitlement-1",), (),
        )
        unknown_scene = NarrativeScene(
            "scene-1", ("beat-1",), 1, 1, None, "bob"
        )
        first = NarrativeProjection(
            HASH_A, None, POLICY, (unknown_beat,), (unknown_scene,),
            NarrativeCut("cut-1", ("scene-1",), (entitlement(),)),
        )
        second = NarrativeProjection(
            HASH_A, None, POLICY, (unknown_beat,), (unknown_scene,),
            NarrativeCut("cut-1", ("scene-1",), (entitlement(),)),
        )
        self.assertIsNone(first.to_dict()["beats"][0]["place_id"])
        self.assertIsNone(first.to_dict()["scenes"][0]["place_id"])
        self.assertEqual(first.content_hash, second.content_hash)

        known_scene = NarrativeScene(
            "scene-1", ("beat-1",), 1, 1, "undisclosed", "bob"
        )
        with self.assertRaisesRegex(ValueError, "scene.*match"):
            NarrativeProjection(
                HASH_A, None, POLICY, (unknown_beat,), (known_scene,),
                NarrativeCut("cut-1", ("scene-1",), (entitlement(),)),
            )

    def test_authored_policy_rejects_cut_scene_order_that_differs_from_event_order(self):
        authored_policy = NarrativeProjectionPolicy(
            "flashback", "1.0", NarrativeAuthority.OBJECTIVE,
            temporal_order=NarrativeTemporalOrder.AUTHORED,
            authored_event_order=("event-2", "event-1"),
        )
        event_one = beat("beat-1", source_event_id="event-1", round_index=1)
        event_two = beat("beat-2", source_event_id="event-2", round_index=2)
        with self.assertRaisesRegex(ValueError, "authored.*order"):
            NarrativeProjection(
                HASH_A, None, authored_policy, (event_one, event_two),
                (
                    NarrativeScene("scene-1", ("beat-1",), 1, 1, "office", None),
                    NarrativeScene("scene-2", ("beat-2",), 2, 2, "office", None),
                ),
                NarrativeCut("cut-1", ("scene-1", "scene-2"), (entitlement(),)),
            )

    def test_authored_policy_rejects_noncontiguous_repeated_source_event_groups(self):
        authored_policy = NarrativeProjectionPolicy(
            "flashback", "1.0", NarrativeAuthority.OBJECTIVE,
            temporal_order=NarrativeTemporalOrder.AUTHORED,
            authored_event_order=("event-1", "event-2"),
        )
        first_event_one = beat("beat-1", source_event_id="event-1", round_index=1)
        event_two = beat("beat-2", source_event_id="event-2", round_index=2)
        second_event_one = beat("beat-3", source_event_id="event-1", round_index=3)
        with self.assertRaisesRegex(ValueError, "authored.*contiguous"):
            NarrativeProjection(
                HASH_A, None, authored_policy,
                (first_event_one, event_two, second_event_one),
                (
                    NarrativeScene("scene-1", ("beat-1",), 1, 1, "office", None),
                    NarrativeScene("scene-2", ("beat-2",), 2, 2, "office", None),
                    NarrativeScene("scene-3", ("beat-3",), 3, 3, "office", None),
                ),
                NarrativeCut("cut-1", ("scene-1", "scene-2", "scene-3"), (entitlement(),)),
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import unittest

from narrative_dynamics.abm.situated_projection_contracts import (
    NarrativeAuthority,
    NarrativeBeat,
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


def beat(beat_id: str = "beat-1", *, entitlement_id: str = "entitlement-1") -> NarrativeBeat:
    return NarrativeBeat(
        beat_id,
        NarrativeBeatKind.PHYSICAL,
        1,
        1,
        "office",
        None,
        ("alice",),
        1.0,
        (NarrativeSupportRef("world_event", "event-1", HASH_A),),
        (entitlement_id,),
        (),
    )


def scene(scene_id: str = "scene-1", beat_ids: tuple[str, ...] = ("beat-1",)) -> NarrativeScene:
    return NarrativeScene(scene_id, beat_ids, 1, 1, "office", None)


class NarrativeProjectionContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

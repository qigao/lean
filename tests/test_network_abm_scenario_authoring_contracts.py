from __future__ import annotations

import math
import unittest

from narrative_dynamics.abm.scenario_authoring_contracts import (
    ScenarioAssetCatalog,
    ScenarioAssetResource,
    ScenarioExecutionMode,
    ScenarioInstitution,
    ScenarioKnowledgeCatalog,
    ScenarioKnowledgeResource,
    ScenarioMembership,
    ScenarioNorm,
    ScenarioNormEffect,
    ScenarioPredicate,
    ScenarioPredicateKind,
    ScenarioRelationship,
    ScenarioResourceGrant,
    ScenarioResourceKind,
    ScenarioRunPolicy,
    ScenarioSceneContract,
    ScenarioSceneDependency,
    ScenarioSocialWorld,
    ScenarioStoryAct,
    ScenarioStoryPlan,
)


HASH = "sha256:" + "a" * 64


def scene(scene_id: str) -> ScenarioSceneContract:
    return ScenarioSceneContract(
        scene_id,
        ("office",),
        ("alice",),
        (ScenarioPredicate(ScenarioPredicateKind.AGENT_AT, "alice", "office", None),),
        (),
        ("narrator_prompt",),
        ("outcome",),
        3,
    )


def knowledge(resource_id: str = "statute") -> ScenarioKnowledgeResource:
    return ScenarioKnowledgeResource(
        resource_id,
        ScenarioResourceKind.DOCUMENT,
        HASH,
        "https://example.test/statute.pdf",
        "application/pdf",
        "en",
        "2026-09-01",
        "official",
        "CC-BY-4.0",
        ("law", "evidence"),
        "index-statute",
    )


class ScenarioAuthoringContractTests(unittest.TestCase):
    def test_institution_containment_and_story_dependencies_are_acyclic(self):
        with self.assertRaisesRegex(ValueError, "institution.*cycle"):
            ScenarioSocialWorld(
                institutions=(
                    ScenarioInstitution("firm", "company", "team"),
                    ScenarioInstitution("team", "department", "firm"),
                ),
                memberships=(), relationships=(), norms=(),
            )
        with self.assertRaisesRegex(ValueError, "story.*cycle"):
            ScenarioStoryPlan(
                "plan", "1", ScenarioExecutionMode.HYBRID,
                acts=(ScenarioStoryAct("act", ("a", "b")),),
                scenes=(scene("a"), scene("b")),
                dependencies=(
                    ScenarioSceneDependency("a", "b"),
                    ScenarioSceneDependency("b", "a"),
                ),
            )

    def test_relationship_direction_and_resource_grants_are_canonical(self):
        relationship = ScenarioRelationship("alice", "bob", "reports_to", 1.0)
        self.assertNotEqual(
            relationship,
            ScenarioRelationship("bob", "alice", "reports_to", 1.0),
        )
        grant = ScenarioResourceGrant("role", "lawyer", ("statute", "case-file"))
        self.assertEqual(grant.resource_ids, ("case-file", "statute"))

    def test_social_world_canonicalizes_unordered_values_and_rejects_duplicate_edges(self):
        world = ScenarioSocialWorld(
            institutions=(
                ScenarioInstitution("firm", "company"),
                ScenarioInstitution("archive", "department", "firm"),
            ),
            memberships=(ScenarioMembership("alice", "firm", "lawyer"),),
            relationships=(ScenarioRelationship("alice", "bob", "trusts", 0.5),),
            norms=(),
        )
        self.assertEqual(tuple(item.institution_id for item in world.institutions), ("archive", "firm"))
        with self.assertRaisesRegex(ValueError, "relationship.*unique"):
            ScenarioSocialWorld(
                (), (),
                (
                    ScenarioRelationship("alice", "bob", "trusts", 0.5),
                    ScenarioRelationship("alice", "bob", "trusts", 0.2),
                ), (),
            )

    def test_social_world_rejects_duplicate_memberships_and_nonfinite_or_out_of_range_strengths(self):
        membership = ScenarioMembership("alice", "firm", "lawyer")
        with self.assertRaisesRegex(ValueError, "membership.*unique"):
            ScenarioSocialWorld((), (membership, membership), (), ())
        for strength in (math.inf, -math.inf, math.nan, -1.01, 1.01):
            with self.subTest(strength=strength):
                with self.assertRaisesRegex(ValueError, "strength"):
                    ScenarioRelationship("alice", "bob", "trusts", strength)

    def test_non_descriptive_norms_require_registered_targets_and_descriptive_norm_has_no_hook(self):
        rule = ScenarioNorm(
            "obey-policy", "lawyer", ScenarioNormEffect.DENY_ACTION, 10,
            action_id="disclose", descriptive_text="Do not disclose evidence.",
        )
        with self.assertRaisesRegex(ValueError, "action"):
            ScenarioSocialWorld((), (), (), (rule,))
        world = ScenarioSocialWorld(
            (), (), (), (rule,), registered_action_ids=("disclose",),
        )
        self.assertEqual(world.norms, (rule,))
        descriptive = ScenarioNorm(
            "custom", "lawyer", ScenarioNormEffect.DESCRIPTIVE, 1,
            descriptive_text="A free-form author note.",
        )
        self.assertIsNone(descriptive.enforcement_hook)

    def test_predicates_are_typed_and_scene_limits_are_positive(self):
        with self.assertRaisesRegex(ValueError, "predicate"):
            ScenarioPredicate("arbitrary", "alice", "office", None)
        with self.assertRaisesRegex(ValueError, "passage_open.*value"):
            ScenarioPredicate(ScenarioPredicateKind.PASSAGE_OPEN, "door", None, "yes")
        with self.assertRaisesRegex(ValueError, "maximum rounds"):
            ScenarioSceneContract("scene", (), (), (), (), (), (), 0)

    def test_claim_status_predicates_use_the_runtime_status_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "claim_status.*status"):
            ScenarioPredicate(
                ScenarioPredicateKind.CLAIM_STATUS, "alice", "case-file", "invented",
            )

    def test_story_values_canonicalize_unordered_scene_scopes_and_plan_values(self):
        first_scene = ScenarioSceneContract(
            "a", ("office", "archive"), ("bob", "alice"), (), (),
            ("prompt", "camera"), ("win", "learn"), 3,
        )
        canonical_scene = ScenarioSceneContract(
            "a", ("archive", "office"), ("alice", "bob"), (), (),
            ("camera", "prompt"), ("learn", "win"), 3,
        )
        self.assertEqual(first_scene, canonical_scene)
        agent_at = ScenarioPredicate(ScenarioPredicateKind.AGENT_AT, "alice", "office", None)
        passage_open = ScenarioPredicate(ScenarioPredicateKind.PASSAGE_OPEN, "door", None, True)
        first = ScenarioStoryPlan(
            "plan", "1", ScenarioExecutionMode.HYBRID,
            (ScenarioStoryAct("act", ("a", "b")),),
            (scene("b"), scene("a")), (), (passage_open, agent_at), (agent_at, passage_open),
        )
        second = ScenarioStoryPlan(
            "plan", "1", ScenarioExecutionMode.HYBRID,
            (ScenarioStoryAct("act", ("a", "b")),),
            (scene("a"), scene("b")), (), (agent_at, passage_open), (passage_open, agent_at),
        )
        self.assertEqual(first, second)

    def test_scene_predicate_conjunctions_are_canonical(self):
        agent_at = ScenarioPredicate(ScenarioPredicateKind.AGENT_AT, "alice", "office", None)
        passage_open = ScenarioPredicate(ScenarioPredicateKind.PASSAGE_OPEN, "door", None, True)
        first = ScenarioSceneContract(
            "a", (), (), (passage_open, agent_at), (agent_at, passage_open), (), (), 3,
        )
        second = ScenarioSceneContract(
            "a", (), (), (agent_at, passage_open), (passage_open, agent_at), (), (), 3,
        )
        self.assertEqual(first, second)

    def test_story_plan_rejects_duplicate_scene_membership_and_missing_dependency_endpoint(self):
        with self.assertRaisesRegex(ValueError, "scene.*act"):
            ScenarioStoryPlan(
                "plan", "1", ScenarioExecutionMode.HYBRID,
                acts=(ScenarioStoryAct("one", ("a",)), ScenarioStoryAct("two", ("a",))),
                scenes=(scene("a"),), dependencies=(),
            )
        with self.assertRaisesRegex(ValueError, "dependency.*scene"):
            ScenarioStoryPlan(
                "plan", "1", ScenarioExecutionMode.HYBRID,
                acts=(ScenarioStoryAct("one", ("a",)),), scenes=(scene("a"),),
                dependencies=(ScenarioSceneDependency("a", "missing"),),
            )

    def test_authored_story_plan_requires_a_scene(self):
        with self.assertRaisesRegex(ValueError, "authored.*scene"):
            ScenarioStoryPlan(
                "plan", "1", ScenarioExecutionMode.AUTHORED,
                acts=(), scenes=(), dependencies=(),
            )

    def test_resource_catalogs_canonicalize_resources_and_validate_known_grants(self):
        grant = ScenarioResourceGrant("agent", "alice", ("statute",))
        catalog = ScenarioKnowledgeCatalog((knowledge("statute"), knowledge("case-file")), (grant,))
        self.assertEqual(tuple(item.resource_id for item in catalog.resources), ("case-file", "statute"))
        with self.assertRaisesRegex(ValueError, "known resource"):
            ScenarioKnowledgeCatalog((knowledge(),), (ScenarioResourceGrant("public", None, ("missing",)),))
        assets = ScenarioAssetCatalog((
            ScenarioAssetResource("office", ScenarioResourceKind.MODEL_3D, HASH,
                                  "https://example.test/office.glb", "model/gltf-binary",
                                  "official", "CC-BY-4.0", (4.0, 3.0, 2.5), "m", "glb"),
        ), ())
        self.assertEqual(assets.resources[0].resource_id, "office")

    def test_asset_resources_retain_required_authority_metadata(self):
        asset = ScenarioAssetResource(
            "office", ScenarioResourceKind.MODEL_3D, HASH,
            "https://example.test/office.glb", "model/gltf-binary",
            authority="official", license_tag="CC-BY-4.0",
        )
        self.assertEqual(asset.authority, "official")

    def test_catalog_rejects_multiple_grants_for_the_same_subject(self):
        with self.assertRaisesRegex(ValueError, "grant.*unique"):
            ScenarioKnowledgeCatalog(
                (knowledge("case-file"), knowledge("statute")),
                (
                    ScenarioResourceGrant("agent", "alice", ("case-file",)),
                    ScenarioResourceGrant("agent", "alice", ("statute",)),
                ),
            )

    def test_resource_grants_reject_a_bare_string_instead_of_a_resource_tuple(self):
        with self.assertRaisesRegex(TypeError, "tuple"):
            ScenarioResourceGrant("public", None, "statute")

    def test_resources_reject_invalid_hash_and_unbounded_metadata(self):
        with self.assertRaisesRegex(ValueError, "content hash"):
            ScenarioKnowledgeResource(
                "statute", ScenarioResourceKind.DOCUMENT, "bad", "https://example.test/x",
                "application/pdf", "en", "1", "official", "CC-BY-4.0",
            )
        with self.assertRaisesRegex(ValueError, "URI"):
            ScenarioAssetResource(
                "office", ScenarioResourceKind.MODEL_3D, HASH, "x" * 2049,
                "model/gltf-binary", "official", "CC-BY-4.0", (), "m", "glb",
            )

    def test_run_policy_validates_its_own_mode_and_limits(self):
        policy = ScenarioRunPolicy(
            ScenarioExecutionMode.HYBRID, 12, 3, 100,
            ("network.metrics",), True, "final_blend", 7, 4096,
        )
        self.assertEqual(policy.allowed_output_kinds, ("network.metrics",))
        with self.assertRaisesRegex(ValueError, "execution mode"):
            ScenarioRunPolicy("freeform", 12, 3, 100, ("network.metrics",), False, "none")
        with self.assertRaisesRegex(ValueError, "maximum output"):
            ScenarioRunPolicy(ScenarioExecutionMode.SANDBOX, 12, 3, 0, ("network.metrics",), False, "none")
        with self.assertRaisesRegex(ValueError, "Blender"):
            ScenarioRunPolicy(ScenarioExecutionMode.SANDBOX, 12, 3, 1, ("network.metrics",), False, "render")
        with self.assertRaisesRegex(ValueError, "output kind"):
            ScenarioRunPolicy(ScenarioExecutionMode.SANDBOX, 12, 3, 1, ("metrics",), False, "none")


if __name__ == "__main__":
    unittest.main()

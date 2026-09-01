from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.scenario_authoring_contracts import (
    ScenarioAssetCatalog,
    ScenarioExecutionMode,
    ScenarioKnowledgeCatalog,
    ScenarioRunPolicy,
    ScenarioSocialWorld,
    ScenarioStoryPlan,
)
from narrative_dynamics.abm.scenario_compiler import (
    ScenarioCompilationError,
    _compile_situated_scenario_components,
)
from narrative_dynamics.abm.scenario_package import load_situated_scenario_package
from narrative_dynamics.abm.situated_network_contracts import SituatedNetworkRuntimeModel
from narrative_dynamics.abm.situated_spatial_map_contracts import SituatedSpatialMap
from tests.scenario_package_fixtures import (
    mutate_json,
    refresh_manifest_hash,
    write_law_firm_package,
)


def _value(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))["value"]


class SituatedScenarioCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def compile_fixture(self, name: str = "law-firm"):
        root = write_law_firm_package(self.root / name)
        return _compile_situated_scenario_components(
            load_situated_scenario_package(root)
        )

    def test_package_compiles_to_exact_v20_models_and_task_2_values(self) -> None:
        compiled = self.compile_fixture()

        self.assertEqual(compiled.scenario_id, "law-firm-case")
        self.assertIsInstance(compiled.runtime_model, SituatedNetworkRuntimeModel)
        self.assertIsInstance(compiled.spatial_map, SituatedSpatialMap)
        self.assertEqual(
            compiled.runtime_model.percept_memory_model.cognitive_model.world_model,
            compiled.spatial_map.world_model,
        )
        self.assertIsInstance(compiled.social_world, ScenarioSocialWorld)
        self.assertIsInstance(compiled.story_plan, ScenarioStoryPlan)
        self.assertIsInstance(compiled.knowledge_catalog, ScenarioKnowledgeCatalog)
        self.assertIsInstance(compiled.asset_catalog, ScenarioAssetCatalog)
        self.assertIsInstance(compiled.run_policy, ScenarioRunPolicy)
        self.assertIs(compiled.story_plan.mode, ScenarioExecutionMode.HYBRID)
        self.assertIs(compiled.run_policy.mode, compiled.story_plan.mode)
        self.assertEqual(len(compiled.source_document_hashes), 16)
        self.assertEqual(
            compiled.source_document_hashes,
            tuple(sorted(compiled.source_document_hashes)),
        )
        self.assertRegex(compiled.content_hash, r"^sha256:[0-9a-f]{64}$")

    def test_compilation_is_path_independent_and_preserves_authored_scene_order(self) -> None:
        first = self.compile_fixture("first")
        second = self.compile_fixture("unrelated-layout")

        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(
            first.story_plan.acts[0].scene_ids,
            ("discover", "confront"),
        )

    def test_unknown_cross_document_id_reports_role_pointer_and_stable_code(self) -> None:
        root = write_law_firm_package(self.root / "unknown-place")
        mutate_json(root / "agents/alice.json", "/body/initial_place", "missing")
        refresh_manifest_hash(root, "agent", "alice")

        with self.assertRaises(ScenarioCompilationError) as raised:
            _compile_situated_scenario_components(
                load_situated_scenario_package(root)
            )

        self.assertEqual(raised.exception.document_role, "agent:alice")
        self.assertEqual(raised.exception.json_pointer, "/body/initial_place")
        self.assertEqual(raised.exception.code, "unknown_reference")
        self.assertRegex(str(raised.exception), r"agent:alice.*\/body\/initial_place")
        self.assertNotIn(str(root), str(raised.exception))
        self.assertNotIn("missing", str(raised.exception))

    def test_boolean_numeric_field_is_rejected_at_nearest_pointer(self) -> None:
        root = write_law_firm_package(self.root / "boolean-number")
        mutate_json(root / "physical/perception.json", "/edges/0/cost", True)
        refresh_manifest_hash(root, "physical.perception", "perception")

        with self.assertRaises(ScenarioCompilationError) as raised:
            _compile_situated_scenario_components(
                load_situated_scenario_package(root)
            )

        self.assertEqual(raised.exception.document_role, "physical.perception")
        self.assertEqual(raised.exception.json_pointer, "/edges/0/cost")
        self.assertEqual(raised.exception.code, "invalid_type")

    def test_exact_document_and_nested_object_keys_are_required(self) -> None:
        root = write_law_firm_package(self.root / "extra-key")
        action = _value(root / "agents/bob.json")["cognition"]["actions"][0]  # type: ignore[index]
        mutate_json(
            root / "agents/bob.json",
            "/cognition/actions/0",
            {**action, "callable": "tests.fake:factory"},  # type: ignore[arg-type]
        )
        refresh_manifest_hash(root, "agent", "bob")

        with self.assertRaises(ScenarioCompilationError) as raised:
            _compile_situated_scenario_components(
                load_situated_scenario_package(root)
            )

        self.assertEqual(raised.exception.document_role, "agent:bob")
        self.assertEqual(raised.exception.json_pointer, "/cognition/actions/0")
        self.assertEqual(raised.exception.code, "unsupported_shape")
        self.assertNotIn("callable", str(raised.exception))

    def test_fixed_rosters_cover_relationships_knowledge_and_initial_state(self) -> None:
        corruptions = (
            (
                "relationships",
                "social/relationships.json",
                "social.relationships",
                "relationships",
                "/relationships",
            ),
            (
                "knowledge",
                "knowledge/access.json",
                "knowledge.access",
                "access",
                "/grants",
            ),
            (
                "initial-state",
                "physical/initial-state.json",
                "physical.initial_state",
                "initial",
                "/agents",
            ),
        )
        for name, relative_path, role, logical_id, pointer in corruptions:
            with self.subTest(name=name):
                root = write_law_firm_package(self.root / name)
                value = _value(root / relative_path)
                items = value[pointer[1:]]
                mutate_json(root / relative_path, pointer, items[:-1])  # type: ignore[index]
                refresh_manifest_hash(root, role, logical_id)

                with self.assertRaises(ScenarioCompilationError) as raised:
                    _compile_situated_scenario_components(
                        load_situated_scenario_package(root)
                    )

                self.assertEqual(raised.exception.document_role, role)
                self.assertEqual(raised.exception.json_pointer, pointer)
                self.assertEqual(raised.exception.code, "roster_mismatch")

    def test_story_references_and_runtime_norm_targets_use_global_catalogs(self) -> None:
        corruptions = (
            (
                "story-place",
                "story/outline.json",
                "/scenes/0/place_ids/0",
                "missing-place",
                "story.outline",
                "outline",
                "unknown_reference",
            ),
            (
                "norm-action",
                "social/norms.json",
                "/norms/0/action_id",
                "missing-action",
                "social.norms",
                "norms",
                "unsupported_norm_effect",
            ),
        )
        for name, relative_path, pointer, replacement, role, logical_id, code in corruptions:
            with self.subTest(name=name):
                root = write_law_firm_package(self.root / name)
                mutate_json(root / relative_path, pointer, replacement)
                refresh_manifest_hash(root, role, logical_id)

                with self.assertRaises(ScenarioCompilationError) as raised:
                    _compile_situated_scenario_components(
                        load_situated_scenario_package(root)
                    )

                self.assertEqual(raised.exception.document_role, role)
                self.assertEqual(raised.exception.json_pointer, pointer)
                self.assertEqual(raised.exception.code, code)

    def test_all_cross_document_catalogs_report_the_authored_reference_pointer(self) -> None:
        corruptions = (
            (
                "tracked-hypothesis",
                "run.json",
                "/runtime_model/tracked_hypothesis_id",
                "missing-hypothesis",
                "run",
                "run",
            ),
            (
                "topic-symbol",
                "agents/alice.json",
                "/social/topics/0/symbol_ids/0",
                "missing-symbol",
                "agent",
                "alice",
            ),
            (
                "membership-institution",
                "social/institutions.json",
                "/memberships/0/institution_id",
                "missing-institution",
                "social.institutions",
                "institutions",
            ),
            (
                "grant-resource",
                "knowledge/access.json",
                "/grants/1/resource_ids/0",
                "missing-resource",
                "knowledge.access",
                "access",
            ),
            (
                "asset-grant-resource",
                "assets/catalog.json",
                "/grants/0/resource_ids/0",
                "missing-asset",
                "asset.catalog",
                "catalog",
            ),
            (
                "story-dependency",
                "story/outline.json",
                "/dependencies/0/successor_scene_id",
                "missing-scene",
                "story.outline",
                "outline",
            ),
            (
                "story-intervention",
                "story/outline.json",
                "/scenes/0/allowed_intervention_kinds/0",
                "missing-intervention",
                "story.outline",
                "outline",
            ),
            (
                "grant-subject",
                "knowledge/access.json",
                "/grants/1/subject_id",
                "missing-agent",
                "knowledge.access",
                "access",
            ),
            (
                "map-place",
                "physical/map.json",
                "/places/0/place_id",
                "missing-place",
                "physical.map",
                "map",
            ),
            (
                "norm-target-scope",
                "social/norms.json",
                "/norms/0/target_scope_id",
                "missing-target",
                "social.norms",
                "norms",
            ),
            (
                "body-role-membership",
                "agents/alice.json",
                "/body/role",
                "missing-role",
                "agent",
                "alice",
            ),
        )
        for name, relative_path, pointer, replacement, role, logical_id in corruptions:
            with self.subTest(name=name):
                root = write_law_firm_package(self.root / name)
                mutate_json(root / relative_path, pointer, replacement)
                refresh_manifest_hash(root, role, logical_id)

                with self.assertRaises(ScenarioCompilationError) as raised:
                    _compile_situated_scenario_components(
                        load_situated_scenario_package(root)
                    )

                expected_role = f"agent:{logical_id}" if role == "agent" else role
                self.assertEqual(raised.exception.document_role, expected_role)
                self.assertEqual(raised.exception.json_pointer, pointer)
                self.assertEqual(raised.exception.code, "unknown_reference")

    def test_initial_object_holder_must_reference_the_fixed_agent_roster(self) -> None:
        root = write_law_firm_package(self.root / "unknown-holder")
        mutate_json(root / "physical/initial-state.json", "/objects/0/place_id", None)
        mutate_json(
            root / "physical/initial-state.json",
            "/objects/0/holder_agent_id",
            "missing-agent",
        )
        refresh_manifest_hash(root, "physical.initial_state", "initial")

        with self.assertRaises(ScenarioCompilationError) as raised:
            _compile_situated_scenario_components(
                load_situated_scenario_package(root)
            )

        self.assertEqual(raised.exception.document_role, "physical.initial_state")
        self.assertEqual(raised.exception.json_pointer, "/objects/0/holder_agent_id")
        self.assertEqual(raised.exception.code, "unknown_reference")

    def test_relationship_types_may_be_reused_across_distinct_directed_pairs(self) -> None:
        root = write_law_firm_package(self.root / "shared-relationship-type")
        mutate_json(
            root / "social/relationships.json",
            "/relationships/1/relationship_type",
            "supervises",
        )
        refresh_manifest_hash(root, "social.relationships", "relationships")

        compiled = _compile_situated_scenario_components(
            load_situated_scenario_package(root)
        )

        self.assertEqual(compiled.social_world.registered_relationship_types.count("supervises"), 1)

    def test_signed_relationship_strength_seeds_affinity_not_bounded_trust(self) -> None:
        root = write_law_firm_package(self.root / "negative-affinity")
        mutate_json(root / "social/relationships.json", "/relationships/1/strength", -0.4)
        refresh_manifest_hash(root, "social.relationships", "relationships")

        compiled = _compile_situated_scenario_components(
            load_situated_scenario_package(root)
        )
        seed = next(
            item
            for item in compiled.relationship_seeds
            if item.observer_agent_id == "alice" and item.source_agent_id == "client"
        )

        self.assertEqual(seed.trust, 0.5)
        self.assertEqual(seed.affinity, -0.4)

    def test_run_policy_and_story_plan_execution_modes_must_match(self) -> None:
        root = write_law_firm_package(self.root / "mode-mismatch")
        mutate_json(root / "run.json", "/mode", "sandbox")
        refresh_manifest_hash(root, "run", "run")

        with self.assertRaises(ScenarioCompilationError) as raised:
            _compile_situated_scenario_components(
                load_situated_scenario_package(root)
            )

        self.assertEqual(raised.exception.document_role, "run")
        self.assertEqual(raised.exception.json_pointer, "/mode")
        self.assertEqual(raised.exception.code, "execution_mode_mismatch")


if __name__ == "__main__":
    unittest.main()

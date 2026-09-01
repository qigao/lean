from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

import narrative_dynamics.abm.scenario_compiler as scenario_compiler
import narrative_dynamics.abm.scenario_package_contracts as scenario_package_contracts
from narrative_dynamics.abm.scenario_authoring_contracts import (
    ScenarioAssetCatalog,
    ScenarioExecutionMode,
    ScenarioKnowledgeCatalog,
    ScenarioRelationship,
    ScenarioRunPolicy,
    ScenarioSocialWorld,
    ScenarioStoryPlan,
)
from narrative_dynamics.abm.scenario_compiler import (
    ScenarioCompilationError,
    _compile_situated_scenario_components,
)
from narrative_dynamics.abm.scenario_package import load_situated_scenario_package
from narrative_dynamics.abm.situated import (
    ObservationChannel,
    SituatedActionIntent,
    SituatedActionKind,
)
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedActionSpec,
    SituatedAgentCognitiveModel,
    SituatedCognitiveModel,
    SituatedGoalReward,
    SituatedGoalSpec,
    SituatedHypothesis,
    SituatedObservationLikelihood,
    SituatedObservationRule,
    SituatedObservationSymbol,
)
from narrative_dynamics.abm.situated_contracts import (
    AgentBodyState,
    EmbodiedAgentSpec,
    EvidenceFact,
    PassageState,
    PassageSpec,
    PlaceSpec,
    SituatedWorldState,
    SituatedWorldModel,
    WorldObjectState,
    WorldObjectSpec,
)
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedAgentRecallPolicy,
    SituatedMemoryCognitiveModel,
    SituatedMemoryRecallCue,
)
from narrative_dynamics.abm.situated_memory_contracts import (
    MemoryChannelPolicy,
    SituatedMemoryPolicy,
)
from narrative_dynamics.abm.situated_network_contracts import SituatedNetworkRuntimeModel
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    SituatedPerceptMemoryCognitiveModel,
)
from narrative_dynamics.abm.situated_percept_memory import (
    ingest_situated_percept_story,
    initialize_situated_percept_memory,
    list_situated_percept_memories,
)
from narrative_dynamics.abm.situated_percept_memory_contracts import (
    SituatedPerceptMemoryFidelityPolicy,
    SituatedPerceptMemoryPolicy,
)
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedAgentPerceptionProfile,
    SituatedEdgeActivation,
    SituatedEventSignalProfile,
    SituatedPerceptFidelity,
    SituatedPerceptionEdge,
    SituatedPerceptionLayer,
    SituatedPerceptionModel,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedClaimTopic,
    SituatedSocialMemoryModel,
    SituatedSocialMemoryPolicy,
    SituatedSourceRelationship,
    initialize_situated_social_memory,
)
from narrative_dynamics.abm.situated_spatial_map_contracts import (
    SituatedSpatialMap,
    SpatialPassage,
    SpatialPlace,
)
from narrative_dynamics.abm.situated_story import advance_situated_story
from narrative_dynamics.narrative.runtime_planning import PlanningBeliefState
from tests.scenario_package_fixtures import (
    mutate_json,
    refresh_manifest_hash,
    write_law_firm_package,
)


def _value(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))["value"]


def _expected_world_model() -> SituatedWorldModel:
    return SituatedWorldModel(
        "law-firm-world",
        "1",
        (
            PlaceSpec("lobby", "Public lobby"),
            PlaceSpec("meeting", "Meeting room"),
            PlaceSpec("archive", "Restricted archive"),
        ),
        (
            PassageSpec("lobby-meeting", "lobby", "meeting", True),
            PassageSpec("meeting-archive", "meeting", "archive", False),
        ),
        (
            EmbodiedAgentSpec("alice", "partner", "meeting", 1),
            EmbodiedAgentSpec("bob", "lawyer", "lobby", 1),
            EmbodiedAgentSpec("client", "client", "lobby", 1),
        ),
        (
            WorldObjectSpec(
                "case-file",
                "document",
                "archive",
                True,
                (EvidenceFact("status", "found"),),
            ),
        ),
    )


def _expected_cognitive_agent(agent_id: str) -> SituatedAgentCognitiveModel:
    action_ids = (
        f"{agent_id}-inspect-file",
        f"{agent_id}-move-archive",
        f"{agent_id}-take-file",
        f"{agent_id}-tell-status",
        f"{agent_id}-wait",
    )
    actions = (
        SituatedActionSpec(
            action_ids[0],
            SituatedActionKind.INSPECT,
            "case-file",
            None,
            ("archive",),
            False,
            (),
        ),
        SituatedActionSpec(
            action_ids[1],
            SituatedActionKind.MOVE,
            "meeting-archive",
            None,
            ("meeting",),
            False,
            (),
        ),
        SituatedActionSpec(
            action_ids[2],
            SituatedActionKind.TAKE,
            "case-file",
            None,
            ("archive",),
            False,
            (),
        ),
        SituatedActionSpec(
            action_ids[3],
            SituatedActionKind.TELL,
            None,
            "The case file has been found.",
            ("meeting",),
            False,
            (SituatedActionKind.INSPECT, SituatedActionKind.TELL),
        ),
        SituatedActionSpec(action_ids[4], SituatedActionKind.WAIT, repeatable=True),
    )
    hypotheses = (
        SituatedHypothesis("file-found", "The missing file has been found."),
        SituatedHypothesis("file-missing", "The file remains missing."),
    )
    symbols = (
        SituatedObservationSymbol(
            "evidence-found",
            "Evidence supports finding the file.",
        ),
        SituatedObservationSymbol(
            "evidence-missing",
            "Evidence supports the file remaining missing.",
        ),
    )
    likelihoods = tuple(
        SituatedObservationLikelihood(
            action_id,
            hypothesis_id,
            symbol_id,
            0.8 if hypothesis_id == expected_hypothesis else 0.2,
        )
        for action_id in action_ids
        for hypothesis_id in ("file-found", "file-missing")
        for symbol_id, expected_hypothesis in (
            ("evidence-found", "file-found"),
            ("evidence-missing", "file-missing"),
        )
    )
    reward_values = (3.0, 1.0, 2.0, 2.0, 0.0)
    rewards = tuple(
        SituatedGoalReward(
            "resolve-case",
            hypothesis_id,
            action_id,
            -value if hypothesis_id == "file-missing" and action_id == action_ids[3] else value,
        )
        for hypothesis_id in ("file-found", "file-missing")
        for action_id, value in zip(action_ids, reward_values)
    )
    return SituatedAgentCognitiveModel(
        agent_id,
        hypotheses,
        PlanningBeliefState({"file-found": 0.4, "file-missing": 0.6}),
        symbols,
        (
            SituatedObservationRule(
                f"{agent_id}-inspect-found",
                "evidence-found",
                action_ids[0],
                SituatedActionKind.INSPECT,
                "inspected",
                "status",
                "found",
            ),
        ),
        likelihoods,
        actions,
        (action_ids, (action_ids[3], action_ids[4])),
        (),
        (
            SituatedGoalSpec(
                "resolve-case",
                "Resolve the missing-document dispute.",
                1.0,
            ),
        ),
        rewards,
        0.8,
        4.0,
    )


def _expected_runtime_and_spatial_models() -> tuple[
    SituatedNetworkRuntimeModel,
    SituatedSpatialMap,
]:
    world = _expected_world_model()
    cognition = SituatedCognitiveModel(
        "law-firm-case-cognition",
        "1",
        world,
        tuple(_expected_cognitive_agent(agent_id) for agent_id in ("alice", "bob", "client")),
    )
    perception = SituatedPerceptionModel(
        "law-firm-perception",
        "1",
        world,
        (
            SituatedPerceptionEdge(
                "visual-lobby-meeting",
                SituatedPerceptionLayer.VISIBILITY,
                "lobby",
                "meeting",
                1.0,
                SituatedEdgeActivation.PASSAGE_OPEN,
                "lobby-meeting",
            ),
            SituatedPerceptionEdge(
                "auditory-meeting-archive",
                SituatedPerceptionLayer.AUDITORY,
                "meeting",
                "archive",
                12.0,
                SituatedEdgeActivation.PASSAGE_CLOSED,
                "meeting-archive",
            ),
            SituatedPerceptionEdge(
                "interaction-meeting",
                SituatedPerceptionLayer.INTERACTION,
                "meeting",
                "meeting",
                0.0,
            ),
        ),
        tuple(
            SituatedAgentPerceptionProfile(agent_id, 5.0, 10.0, 30.0)
            for agent_id in ("alice", "bob", "client")
        ),
        (
            SituatedEventSignalProfile(SituatedActionKind.WAIT, False, None),
            SituatedEventSignalProfile(SituatedActionKind.MOVE, True, 15.0),
            SituatedEventSignalProfile(SituatedActionKind.INSPECT, True, None),
            SituatedEventSignalProfile(SituatedActionKind.TAKE, True, 10.0),
            SituatedEventSignalProfile(SituatedActionKind.TELL, True, 60.0),
        ),
    )
    recall_cue = SituatedMemoryRecallCue(
        "case-file-recall",
        "case file",
        ("meeting",),
        (SituatedActionKind.INSPECT, SituatedActionKind.TELL),
        (ObservationChannel.AUDITORY, ObservationChannel.INSPECTION),
        0.5,
        3,
    )
    recall_policies = tuple(
        SituatedAgentRecallPolicy(agent_id, (recall_cue,), 4)
        for agent_id in ("alice", "bob", "client")
    )
    percept_memory_policy = SituatedPerceptMemoryPolicy(
        "law-firm-percept-memory",
        "1",
        (
            SituatedPerceptMemoryFidelityPolicy(SituatedPerceptFidelity.DETECTED, 0.25, 0.35),
            SituatedPerceptMemoryFidelityPolicy(SituatedPerceptFidelity.IDENTIFIED, 0.6, 0.55),
            SituatedPerceptMemoryFidelityPolicy(SituatedPerceptFidelity.EXACT, 1.0, 1.0),
        ),
    )
    percept_memory = SituatedPerceptMemoryCognitiveModel(
        "law-firm-percept-memory-cognition",
        "1",
        perception,
        cognition,
        percept_memory_policy,
        recall_policies,
    )
    channel_memory_policy = SituatedMemoryPolicy(
        "law-firm-channel-memory",
        "1",
        (
            MemoryChannelPolicy(ObservationChannel.SELF, 1.0, 0.5),
            MemoryChannelPolicy(ObservationChannel.VISUAL, 0.9, 0.65),
            MemoryChannelPolicy(ObservationChannel.AUDITORY, 0.7, 0.75),
            MemoryChannelPolicy(ObservationChannel.INSPECTION, 1.0, 1.0),
        ),
    )
    memory_cognition = SituatedMemoryCognitiveModel(
        "law-firm-memory-cognition",
        "1",
        cognition,
        channel_memory_policy,
        recall_policies,
    )
    social_memory = SituatedSocialMemoryModel(
        "law-firm-social-memory",
        "1",
        memory_cognition,
        (SituatedClaimTopic("file-location", ("evidence-found", "evidence-missing")),),
        SituatedSocialMemoryPolicy(0.5, 0.2, 0.4, 0.1, 0.2, 20, 100),
    )
    runtime = SituatedNetworkRuntimeModel(
        "law-firm-network",
        "1",
        percept_memory,
        social_memory,
        "file-found",
        0.7,
        0.5,
    )
    spatial = SituatedSpatialMap(
        "law-firm-map",
        "1",
        world,
        (
            SpatialPlace("lobby", 0.0, 0.0, 4.0, 4.0, 2.8),
            SpatialPlace("meeting", 5.0, 0.0, 5.0, 4.0, 2.8),
            SpatialPlace("archive", 11.0, 0.0, 4.0, 3.0, 2.8),
        ),
        (
            SpatialPassage("lobby-meeting", "lobby", "meeting", 2.25, 0.0, 0.5, 1.2, 2.1),
            SpatialPassage("meeting-archive", "meeting", "archive", 8.0, 0.0, 0.5, 1.2, 2.1),
        ),
    )
    return runtime, spatial


def _reverse_mapping_order(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _reverse_mapping_order(item)
            for key, item in reversed(tuple(value.items()))
        }
    if isinstance(value, list):
        return [_reverse_mapping_order(item) for item in value]
    return value


def _write_authored_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _seeded_social_checkpoint(compiled, cognitive_state):
    base = initialize_situated_social_memory(
        compiled.runtime_model.social_memory_model,
        cognitive_state,
    )
    affinities = {
        (item.source_agent_id, item.target_agent_id): item.strength
        for item in compiled.social_world.relationships
    }
    return replace(
        base,
        relationships=tuple(
            replace(
                item,
                affinity=affinities[(item.observer_agent_id, item.source_agent_id)],
            )
            for item in base.relationships
        ),
    )


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

    def compile_public_fixture(self, name: str = "law-firm"):
        root = write_law_firm_package(self.root / name)
        return scenario_compiler.compile_situated_scenario_package(
            load_situated_scenario_package(root)
        )

    def test_compiled_initial_state_binds_every_subsystem_and_database(self) -> None:
        root = write_law_firm_package(self.root / "explicit-initial-state")
        path = root / "physical/initial-state.json"
        mutate_json(path, "/agents/2/place_id", "meeting")
        mutate_json(path, "/objects/0/place_id", None)
        mutate_json(path, "/objects/0/holder_agent_id", "bob")
        mutate_json(path, "/passages/1/open", True)
        refresh_manifest_hash(root, "physical.initial_state", "initial")

        compiled = scenario_compiler.compile_situated_scenario_package(
            load_situated_scenario_package(root)
        )
        world = compiled.runtime_model.percept_memory_model.cognitive_model.world_model
        expected_world_state = SituatedWorldState(
            world.model_id,
            world.content_hash,
            0,
            None,
            (
                AgentBodyState("alice", "meeting"),
                AgentBodyState("bob", "lobby"),
                AgentBodyState("client", "meeting"),
            ),
            (WorldObjectState("case-file", None, "bob"),),
            (
                PassageState("lobby-meeting", True),
                PassageState("meeting-archive", True),
            ),
        )
        expected_relationships = (
            SituatedSourceRelationship("alice", "bob", 0.5, 0.8, 0, 0),
            SituatedSourceRelationship("alice", "client", 0.5, 0.9, 0, 0),
            SituatedSourceRelationship("bob", "alice", 0.5, 0.9, 0, 0),
            SituatedSourceRelationship("bob", "client", 0.5, 0.6, 0, 0),
            SituatedSourceRelationship("client", "alice", 0.5, 0.7, 0, 0),
            SituatedSourceRelationship("client", "bob", 0.5, 0.5, 0, 0),
        )

        self.assertIsInstance(
            compiled,
            scenario_package_contracts.CompiledSituatedScenario,
        )
        self.assertEqual(compiled.initial_story.initial_state, expected_world_state)
        self.assertEqual(
            compiled.initial_story.perception_model,
            compiled.runtime_model.percept_memory_model.perception_model,
        )
        self.assertEqual(
            {mind.agent_id: mind.own_place_id for mind in compiled.initial_cognitive_state.minds},
            {"alice": "meeting", "bob": "lobby", "client": "meeting"},
        )
        self.assertTrue(compiled.initial_cognitive_state.checkpoint)
        self.assertEqual(
            compiled.initial_cognitive_state.story_hash,
            compiled.initial_story.content_hash,
        )
        self.assertEqual(compiled.initial_social_state.relationships, expected_relationships)
        self.assertTrue(compiled.initial_social_state.checkpoint)

        with TemporaryDirectory() as temporary:
            memory_path = Path(temporary) / "memory.sqlite3"
            self.assertFalse(memory_path.exists())
            state = scenario_compiler.initialize_compiled_scenario(memory_path, compiled)
            self.assertTrue(memory_path.is_file())
            moved_path = memory_path.with_name("moved.sqlite3")
            memory_path.replace(moved_path)
            self.assertTrue(moved_path.is_file())

        self.assertEqual(state.story, compiled.initial_story)
        self.assertEqual(state.cognitive_state, compiled.initial_cognitive_state)
        self.assertEqual(state.social_state, compiled.initial_social_state)
        self.assertEqual(state.snapshot.story_hash, compiled.initial_story.content_hash)
        self.assertTrue(state.checkpoint)

    def test_compiled_initial_state_door_change_cascades_through_every_identity(self) -> None:
        first_root = write_law_firm_package(self.root / "closed-door")
        second_root = write_law_firm_package(self.root / "open-door")
        second_initial_path = second_root / "physical/initial-state.json"
        mutate_json(second_initial_path, "/passages/1/open", True)
        refresh_manifest_hash(second_root, "physical.initial_state", "initial")

        first = scenario_compiler.compile_situated_scenario_package(
            load_situated_scenario_package(first_root)
        )
        second = scenario_compiler.compile_situated_scenario_package(
            load_situated_scenario_package(second_root)
        )
        self.assertNotEqual(first.package_hash, second.package_hash)
        self.assertNotEqual(first.initial_story.content_hash, second.initial_story.content_hash)
        self.assertNotEqual(first.content_hash, second.content_hash)

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first_state = scenario_compiler.initialize_compiled_scenario(
                directory / "first.sqlite3",
                first,
            )
            second_state = scenario_compiler.initialize_compiled_scenario(
                directory / "second.sqlite3",
                second,
            )

        self.assertNotEqual(first_state.content_hash, second_state.content_hash)

    def test_compiled_initial_state_rejects_forged_cognitive_location(self) -> None:
        compiled = self.compile_public_fixture("forged-cognitive-location")
        forged_minds = tuple(
            replace(mind, own_place_id="archive")
            if mind.agent_id == "alice"
            else mind
            for mind in compiled.initial_cognitive_state.minds
        )
        forged_cognition = replace(
            compiled.initial_cognitive_state,
            minds=forged_minds,
        )
        forged_social = _seeded_social_checkpoint(compiled, forged_cognition)

        with self.assertRaisesRegex(ValueError, "exact public initializer"):
            replace(
                compiled,
                initial_cognitive_state=forged_cognition,
                initial_social_state=forged_social,
            )

    def test_compiled_initial_state_rejects_forged_cognitive_history(self) -> None:
        compiled = self.compile_public_fixture("forged-cognitive-history")
        forged_minds = tuple(
            replace(mind, processed_observation_ids=("forged-observation",))
            if mind.agent_id == "alice"
            else mind
            for mind in compiled.initial_cognitive_state.minds
        )
        forged_cognition = replace(
            compiled.initial_cognitive_state,
            minds=forged_minds,
        )
        forged_social = _seeded_social_checkpoint(compiled, forged_cognition)

        with self.assertRaisesRegex(ValueError, "exact public initializer"):
            replace(
                compiled,
                initial_cognitive_state=forged_cognition,
                initial_social_state=forged_social,
            )

    def test_compiled_initial_state_rejects_forged_social_counts(self) -> None:
        compiled = self.compile_public_fixture("forged-social-counts")
        first_relationship, *remaining = compiled.initial_social_state.relationships
        forged_social = replace(
            compiled.initial_social_state,
            relationships=(
                replace(first_relationship, confirmation_count=1),
                *remaining,
            ),
        )

        with self.assertRaisesRegex(ValueError, "exact public initializer"):
            replace(compiled, initial_social_state=forged_social)

    def test_compiled_initial_state_rejects_nonempty_memory_store(self) -> None:
        compiled = self.compile_public_fixture("nonempty-memory-store")
        world = compiled.runtime_model.percept_memory_model.cognitive_model.world_model
        foreign_story = advance_situated_story(
            world,
            compiled.initial_story,
            (
                SituatedActionIntent(
                    "alice-wait",
                    "alice",
                    SituatedActionKind.WAIT,
                ),
            ),
        )

        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "memory.sqlite3"
            report = ingest_situated_percept_story(
                database_path,
                compiled.runtime_model.percept_memory_model.perception_model,
                foreign_story,
                "alice",
                compiled.runtime_model.percept_memory_model.memory_policy,
            )
            before = list_situated_percept_memories(database_path, "alice")
            self.assertGreater(report.inserted_count, 0)

            with self.assertRaisesRegex(ValueError, "empty"):
                scenario_compiler.initialize_compiled_scenario(
                    database_path,
                    compiled,
                )

            self.assertEqual(
                list_situated_percept_memories(database_path, "alice"),
                before,
            )

    def test_compiled_initial_state_rejects_noncanonical_empty_memory_store(self) -> None:
        compiled = self.compile_public_fixture("noncanonical-empty-memory-store")

        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "memory.sqlite3"
            initialize_situated_percept_memory(database_path)
            connection = sqlite3.connect(database_path)
            try:
                connection.execute(
                    "INSERT INTO percept_memory_metadata(key, value) VALUES (?, ?)",
                    ("foreign-owner", "other-scenario"),
                )
                connection.commit()
            finally:
                connection.close()

            with self.assertRaisesRegex(ValueError, "canonical empty"):
                scenario_compiler.initialize_compiled_scenario(
                    database_path,
                    compiled,
                )

            connection = sqlite3.connect(database_path)
            try:
                foreign_metadata = connection.execute(
                    "SELECT value FROM percept_memory_metadata WHERE key = ?",
                    ("foreign-owner",),
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(foreign_metadata, ("other-scenario",))

    def test_compiled_initial_state_rejects_foreign_schema_without_mutation(self) -> None:
        compiled = self.compile_public_fixture("foreign-database-schema")

        def snapshot(database_path: Path):
            connection = sqlite3.connect(database_path)
            try:
                schema = tuple(
                    connection.execute(
                        "SELECT type, name, tbl_name, sql FROM sqlite_master "
                        "ORDER BY type, name"
                    ).fetchall()
                )
                rows = tuple(
                    connection.execute(
                        "SELECT state_id, typeof(payload), payload "
                        "FROM foreign_state ORDER BY state_id"
                    ).fetchall()
                )
            finally:
                connection.close()
            return schema, rows

        with TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "foreign.sqlite3"
            connection = sqlite3.connect(database_path)
            try:
                connection.execute(
                    "CREATE TABLE foreign_state ("
                    "state_id TEXT PRIMARY KEY, payload BLOB NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO foreign_state(state_id, payload) VALUES (?, ?)",
                    ("foreign-1", sqlite3.Binary(b"\x00foreign-state\xff")),
                )
                connection.commit()
            finally:
                connection.close()
            before = snapshot(database_path)

            with self.assertRaisesRegex(ValueError, "noncanonical") as raised:
                scenario_compiler.initialize_compiled_scenario(
                    database_path,
                    compiled,
                )

            after = snapshot(database_path)
            self.assertEqual(after, before)
            self.assertEqual(
                after[1],
                (("foreign-1", "blob", b"\x00foreign-state\xff"),),
            )
            self.assertNotIn(str(database_path), str(raised.exception))
            self.assertFalse(
                any("percept_memory" in name for _, name, _, _ in after[0])
            )

    def test_compiled_initial_state_rejects_spoofed_canonical_ddl_without_mutation(
        self,
    ) -> None:
        compiled = self.compile_public_fixture("spoofed-canonical-schema")

        def snapshot(database_path: Path):
            connection = sqlite3.connect(database_path)
            try:
                schema = tuple(
                    connection.execute(
                        "SELECT type, name, tbl_name, sql FROM sqlite_master "
                        "ORDER BY type, name"
                    ).fetchall()
                )
                metadata = tuple(
                    connection.execute(
                        "SELECT * FROM percept_memory_metadata ORDER BY key"
                    ).fetchall()
                )
                record_count = connection.execute(
                    "SELECT COUNT(*) FROM percept_memory_records"
                ).fetchone()[0]
            finally:
                connection.close()
            return schema, metadata, record_count

        spoof_scripts = {
            "index": """
                DROP INDEX percept_memory_records_agent_round;
                CREATE INDEX percept_memory_records_agent_round
                ON percept_memory_records(agent_id);
            """,
            "trigger": """
                DROP TRIGGER percept_memory_records_ai;
                CREATE TRIGGER percept_memory_records_ai
                AFTER INSERT ON percept_memory_records BEGIN
                    SELECT 1;
                END;
            """,
            "table": """
                DROP TABLE percept_memory_metadata;
                CREATE TABLE percept_memory_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    foreign_marker TEXT
                );
                INSERT INTO percept_memory_metadata(key, value)
                VALUES ('schema_version', '1');
            """,
        }

        with TemporaryDirectory() as temporary:
            canonical_path = Path(temporary) / "canonical.sqlite3"
            initialize_situated_percept_memory(canonical_path)
            initialized = scenario_compiler.initialize_compiled_scenario(
                canonical_path,
                compiled,
            )
            self.assertEqual(initialized.story, compiled.initial_story)

            for object_type, spoof_script in spoof_scripts.items():
                with self.subTest(object_type=object_type):
                    database_path = Path(temporary) / f"spoofed-{object_type}.sqlite3"
                    initialize_situated_percept_memory(database_path)
                    connection = sqlite3.connect(database_path)
                    try:
                        connection.executescript(spoof_script)
                        connection.commit()
                    finally:
                        connection.close()
                    before = snapshot(database_path)

                    with self.assertRaisesRegex(ValueError, "noncanonical") as raised:
                        scenario_compiler.initialize_compiled_scenario(
                            database_path,
                            compiled,
                        )

                    self.assertEqual(snapshot(database_path), before)
                    self.assertNotIn(str(database_path), str(raised.exception))

    def test_compiled_initial_state_requires_one_authored_relationship_per_pair(self) -> None:
        compiled = self.compile_public_fixture("duplicate-authored-pair")
        social_world = replace(
            compiled.social_world,
            relationships=compiled.social_world.relationships
            + (ScenarioRelationship("alice", "bob", "mentors", 0.1),),
            registered_relationship_types=compiled.social_world.registered_relationship_types
            + ("mentors",),
        )

        with self.assertRaisesRegex(ValueError, "one authored relationship"):
            replace(compiled, social_world=social_world)

    def test_compiled_initial_state_rejects_empty_source_document_identities(self) -> None:
        compiled = self.compile_public_fixture("empty-source-identities")

        with self.assertRaisesRegex(ValueError, "required document roles"):
            replace(compiled, source_document_hashes=())

    def test_compiled_initial_state_rejects_arbitrary_source_document_identity(self) -> None:
        compiled = self.compile_public_fixture("arbitrary-source-identity")

        with self.assertRaisesRegex(ValueError, "document role"):
            replace(
                compiled,
                source_document_hashes=(
                    ("arbitrary.role", "arbitrary", "sha256:" + "0" * 64),
                ),
            )

    def test_compiled_initial_state_requires_semantic_package_hash(self) -> None:
        compiled = self.compile_public_fixture("semantic-package-hash")
        first_role, first_id, first_hash = compiled.source_document_hashes[0]
        mismatches = (
            (
                "document-hash",
                {
                    "source_document_hashes": (
                        (first_role, first_id, "sha256:" + "0" * 64),
                        *compiled.source_document_hashes[1:],
                    )
                },
            ),
            (
                "package-hash",
                {"package_hash": "sha256:" + "0" * 64},
            ),
        )
        self.assertNotEqual(first_hash, "sha256:" + "0" * 64)

        for name, changes in mismatches:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "semantic package hash"):
                    replace(compiled, **changes)

    def test_complete_compilation_is_path_and_unordered_input_independent(self) -> None:
        first_root = write_law_firm_package(self.root / "canonical")
        second_root = write_law_firm_package(self.root / "reversed")
        manifest_path = second_root / "scenario-package.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for locator in manifest["documents"]:
            path = second_root / locator["relative_path"]
            authored = _reverse_mapping_order(
                json.loads(path.read_text(encoding="utf-8"))
            )
            value = authored["value"]
            if locator["role"] == "social.relationships":
                value["relationships"] = list(reversed(value["relationships"]))
            elif locator["role"] == "knowledge.catalog":
                value["resources"] = list(reversed(value["resources"]))
            elif locator["role"] == "knowledge.access":
                value["grants"] = list(reversed(value["grants"]))
            elif locator["role"] == "asset.catalog":
                value["resources"] = list(reversed(value["resources"]))
                value["grants"] = list(reversed(value["grants"]))
            _write_authored_json(path, authored)
            refresh_manifest_hash(second_root, locator["role"], locator["logical_id"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["documents"] = list(reversed(manifest["documents"]))
        _write_authored_json(manifest_path, manifest)

        first = scenario_compiler.compile_situated_scenario_package(
            load_situated_scenario_package(first_root)
        )
        second = scenario_compiler.compile_situated_scenario_package(
            load_situated_scenario_package(second_root)
        )

        self.assertEqual(first, second)
        self.assertEqual(first.package_hash, second.package_hash)
        self.assertEqual(first.initial_story, second.initial_story)
        self.assertEqual(first.content_hash, second.content_hash)

    def test_compiled_contract_rejects_mismatched_runtime_bindings(self) -> None:
        compiled = self.compile_public_fixture("binding-contract")
        story_without_perception = replace(
            compiled.initial_story,
            perception_model=None,
        )
        with self.assertRaisesRegex(ValueError, "perception"):
            replace(compiled, initial_story=story_without_perception)

    def test_compiled_contract_rejects_mismatched_execution_modes(self) -> None:
        compiled = self.compile_public_fixture("mode-contract")
        mismatched_policy = replace(
            compiled.run_policy,
            mode=ScenarioExecutionMode.SANDBOX,
        )
        with self.assertRaisesRegex(ValueError, "mode"):
            replace(compiled, run_policy=mismatched_policy)

    def test_compiled_contract_canonicalizes_source_document_identities(self) -> None:
        compiled = self.compile_public_fixture("source-order-contract")

        reordered = replace(
            compiled,
            source_document_hashes=tuple(reversed(compiled.source_document_hashes)),
        )

        self.assertEqual(reordered, compiled)

    def test_compiled_contract_rejects_missing_subsystems(self) -> None:
        compiled = self.compile_public_fixture("complete-contract")

        for field in (
            "runtime_model",
            "spatial_map",
            "initial_story",
            "initial_cognitive_state",
            "initial_social_state",
            "social_world",
            "story_plan",
            "knowledge_catalog",
            "asset_catalog",
            "run_policy",
        ):
            with self.subTest(field=field):
                with self.assertRaises(TypeError):
                    replace(compiled, **{field: None})

    def test_package_compiles_to_exact_v20_models_and_task_2_values(self) -> None:
        compiled = self.compile_fixture()
        expected_runtime, expected_spatial = _expected_runtime_and_spatial_models()

        self.assertEqual(compiled.scenario_id, "law-firm-case")
        self.assertEqual(compiled.runtime_model, expected_runtime)
        self.assertEqual(compiled.spatial_map, expected_spatial)
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

    def test_compiled_identity_ignores_all_semantically_unordered_source_order(self) -> None:
        baseline = self.compile_fixture("canonical-order")

        for ordering in (
            "json-object",
            "documents",
            "agents",
            "relationships",
            "catalog",
        ):
            with self.subTest(ordering=ordering):
                root = write_law_firm_package(self.root / ordering)
                manifest_path = root / "scenario-package.json"
                if ordering == "json-object":
                    path = root / "social/relationships.json"
                    authored = json.loads(path.read_text(encoding="utf-8"))
                    _write_authored_json(path, _reverse_mapping_order(authored))
                    refresh_manifest_hash(root, "social.relationships", "relationships")
                elif ordering in {"documents", "agents"}:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    documents = manifest["documents"]
                    if ordering == "documents":
                        manifest["documents"] = list(reversed(documents))
                    else:
                        positions = [
                            index
                            for index, item in enumerate(documents)
                            if item["role"] == "agent"
                        ]
                        agents = [documents[index] for index in reversed(positions)]
                        for index, agent in zip(positions, agents):
                            documents[index] = agent
                    _write_authored_json(manifest_path, manifest)
                elif ordering == "relationships":
                    path = root / "social/relationships.json"
                    relationships = _value(path)["relationships"]
                    mutate_json(path, "/relationships", list(reversed(relationships)))
                    refresh_manifest_hash(root, "social.relationships", "relationships")
                else:
                    for relative_path, role, logical_id, pointer in (
                        ("knowledge/catalog.json", "knowledge.catalog", "catalog", "/resources"),
                        ("knowledge/access.json", "knowledge.access", "access", "/grants"),
                    ):
                        path = root / relative_path
                        items = _value(path)[pointer[1:]]
                        mutate_json(path, pointer, list(reversed(items)))
                        refresh_manifest_hash(root, role, logical_id)

                candidate = _compile_situated_scenario_components(
                    load_situated_scenario_package(root)
                )
                self.assertEqual(candidate.runtime_model, baseline.runtime_model)
                self.assertEqual(candidate.social_world, baseline.social_world)
                self.assertEqual(candidate.relationship_seeds, baseline.relationship_seeds)
                self.assertEqual(candidate.source_document_hashes, baseline.source_document_hashes)
                self.assertEqual(candidate.package_hash, baseline.package_hash)
                self.assertEqual(candidate.content_hash, baseline.content_hash)
                self.assertEqual(candidate, baseline)

    def test_run_fallbacks_are_a_closed_validated_mapping_even_when_unused(self) -> None:
        corruptions = (
            ("unknown-key", "/fallbacks/unknown", "empty", "unsupported_shape"),
            ("invalid-map", "/fallbacks/physical.map", "empty", "unsupported_fallback"),
        )
        for name, pointer, replacement, code in corruptions:
            with self.subTest(name=name):
                root = write_law_firm_package(self.root / f"fallback-{name}")
                run_path = root / "run.json"
                if name == "unknown-key":
                    fallbacks = dict(_value(run_path)["fallbacks"])
                    fallbacks["unknown"] = replacement
                    mutate_json(run_path, "/fallbacks", fallbacks)
                else:
                    fallbacks = dict(_value(run_path)["fallbacks"])
                    fallbacks["physical.map"] = replacement
                    mutate_json(run_path, "/fallbacks", fallbacks)
                refresh_manifest_hash(root, "run", "run")

                with self.assertRaises(ScenarioCompilationError) as raised:
                    _compile_situated_scenario_components(
                        load_situated_scenario_package(root)
                    )

                self.assertEqual(raised.exception.document_role, "run")
                self.assertEqual(raised.exception.json_pointer, pointer)
                self.assertEqual(raised.exception.code, code)

    def test_norm_effect_is_a_strict_tagged_union(self) -> None:
        for field in ("resource_id", "relationship_type"):
            with self.subTest(field=field):
                root = write_law_firm_package(self.root / f"norm-{field}")
                path = root / "social/norms.json"
                mutate_json(path, f"/norms/0/{field}", f"unexpected-{field}")
                refresh_manifest_hash(root, "social.norms", "norms")

                with self.assertRaises(ScenarioCompilationError) as raised:
                    _compile_situated_scenario_components(
                        load_situated_scenario_package(root)
                    )

                self.assertEqual(raised.exception.document_role, "social.norms")
                self.assertEqual(raised.exception.json_pointer, f"/norms/0/{field}")
                self.assertEqual(raised.exception.code, "invalid_value")

    def test_social_references_report_raw_authored_indices(self) -> None:
        corruptions = (
            (
                "institution-parent",
                "/institutions/1/parent_institution_id",
                "missing-parent",
            ),
            (
                "membership-after-reorder",
                "/memberships/0/institution_id",
                "missing-institution",
            ),
        )
        for name, pointer, replacement in corruptions:
            with self.subTest(name=name):
                root = write_law_firm_package(self.root / name)
                path = root / "social/institutions.json"
                if name == "membership-after-reorder":
                    memberships = _value(path)["memberships"]
                    mutate_json(path, "/memberships", list(reversed(memberships)))
                mutate_json(path, pointer, replacement)
                refresh_manifest_hash(root, "social.institutions", "institutions")

                with self.assertRaises(ScenarioCompilationError) as raised:
                    _compile_situated_scenario_components(
                        load_situated_scenario_package(root)
                    )

                self.assertEqual(raised.exception.document_role, "social.institutions")
                self.assertEqual(raised.exception.json_pointer, pointer)
                self.assertEqual(raised.exception.code, "unknown_reference")

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

    def test_initial_state_rejects_unknown_holder_and_duplicate_object_location(self) -> None:
        corruptions = (
            ("unknown-holder", None, "missing-agent", "/objects/0/holder_agent_id", "unknown_reference"),
            ("duplicate-location", "archive", "bob", "/objects/0", "contract_violation"),
        )
        for name, place_id, holder_id, pointer, code in corruptions:
            with self.subTest(name=name):
                root = write_law_firm_package(self.root / name)
                mutate_json(root / "physical/initial-state.json", "/objects/0/place_id", place_id)
                mutate_json(
                    root / "physical/initial-state.json",
                    "/objects/0/holder_agent_id",
                    holder_id,
                )
                refresh_manifest_hash(root, "physical.initial_state", "initial")

                with self.assertRaises(ScenarioCompilationError) as raised:
                    scenario_compiler.compile_situated_scenario_package(
                        load_situated_scenario_package(root)
                    )

                self.assertEqual(raised.exception.document_role, "physical.initial_state")
                self.assertEqual(raised.exception.json_pointer, pointer)
                self.assertEqual(raised.exception.code, code)

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

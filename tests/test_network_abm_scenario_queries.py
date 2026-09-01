from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.scenario_compiler import (
    compile_situated_scenario_package,
    initialize_compiled_scenario,
)
from narrative_dynamics.abm.scenario_coordinator_contracts import ScenarioRunStatus
from narrative_dynamics.abm.scenario_package import load_situated_scenario_package
from narrative_dynamics.abm.scenario_queries import (
    project_scenario_agent_state,
    project_scenario_network_state,
    project_scenario_output_view,
    project_scenario_public_state,
    project_scenario_run_view,
)
from narrative_dynamics.abm.simulation_output import project_simulation_output
from narrative_dynamics.abm.simulation_output_contracts import (
    SimulationAudienceCapability,
    SimulationOutputAudience,
    SimulationOutputKind,
)
from narrative_dynamics.abm.situated_network import simulate_situated_network_round
from tests.scenario_package_fixtures import (
    mutate_json,
    refresh_manifest_hash,
    write_law_firm_package,
)


class ScenarioQueryProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = TemporaryDirectory()
        root = Path(cls.temporary.name)
        package_root = write_law_firm_package(root / "law-firm")
        mutate_json(
            package_root / "run.json",
            "/allowed_output_kinds",
            [kind.value for kind in SimulationOutputKind],
        )
        refresh_manifest_hash(package_root, "run", "run")
        cls.scenario = compile_situated_scenario_package(
            load_situated_scenario_package(package_root)
        )
        cls.initial_state = initialize_compiled_scenario(
            root / "law-firm.sqlite3",
            cls.scenario,
        )
        cls.round_result = simulate_situated_network_round(
            root / "law-firm.sqlite3",
            cls.scenario.runtime_model,
            cls.initial_state,
        )
        cls.state = cls.round_result.next_state
        cls.batch = project_simulation_output(
            cls.scenario,
            cls.round_result,
            stream_id="law-firm-stream",
            first_sequence=17,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def capability(
        self,
        audience: SimulationOutputAudience,
        owner_agent_id: str | None = None,
    ) -> SimulationAudienceCapability:
        return SimulationAudienceCapability(audience, owner_agent_id)

    def test_public_state_projects_exact_sorted_physical_state_and_metrics(self) -> None:
        view = project_scenario_public_state(
            "run-1",
            self.scenario.content_hash,
            self.state,
        )
        world = self.state.story.current_state

        # Projecting model defaults would ignore physical changes accepted by later rounds.
        self.assertEqual(
            view.agent_places,
            tuple((item.agent_id, item.place_id) for item in world.agents),
        )
        self.assertEqual(dict(view.agent_places)["alice"], "meeting")
        self.assertEqual(
            view.passage_states,
            tuple((item.passage_id, item.open) for item in world.passages),
        )
        self.assertEqual(
            view.object_placements,
            tuple(
                (item.object_id, item.place_id, item.holder_agent_id)
                for item in world.objects
            ),
        )
        self.assertEqual(view.agent_places, tuple(sorted(view.agent_places)))
        self.assertEqual(view.passage_states, tuple(sorted(view.passage_states)))
        self.assertEqual(
            view.object_placements,
            tuple(sorted(view.object_placements, key=lambda item: item[0])),
        )
        self.assertIs(view.metrics, self.state.metrics)
        self.assertEqual(view.state_hash, self.state.content_hash)

    def test_matching_agent_receives_exact_mind_and_only_own_social_state(self) -> None:
        mind = next(
            item for item in self.state.cognitive_state.minds if item.agent_id == "alice"
        )
        view = project_scenario_agent_state(
            "run-1",
            self.scenario.content_hash,
            self.state,
            "alice",
            self.capability(SimulationOutputAudience.AGENT, "alice"),
        )

        # Returning the whole social state would expose what other Agents privately know.
        self.assertIs(view.mind, mind)
        self.assertEqual(view.place_id, "meeting")
        self.assertTrue(view.relationships)
        self.assertTrue(
            all(item.observer_agent_id == "alice" for item in view.relationships)
        )
        self.assertTrue(
            all(item.observer_agent_id == "alice" for item in view.claims)
        )
        self.assertEqual(
            view.relationships,
            tuple(
                item
                for item in self.state.social_state.relationships
                if item.observer_agent_id == "alice"
            ),
        )
        self.assertEqual(
            view.claims,
            tuple(
                item
                for item in self.state.social_state.claims
                if item.observer_agent_id == "alice"
            ),
        )

    def test_agent_private_query_rejects_wrong_owner_public_and_analyst(self) -> None:
        unauthorized = (
            self.capability(SimulationOutputAudience.AGENT, "bob"),
            self.capability(SimulationOutputAudience.PUBLIC),
            self.capability(SimulationOutputAudience.ANALYST),
        )

        for capability in unauthorized:
            with self.subTest(capability=capability):
                # Trusting the requested id instead of the capability would disclose Alice's mind.
                with self.assertRaisesRegex(
                    PermissionError,
                    "^scenario agent state is not authorized$",
                ):
                    project_scenario_agent_state(
                        "run-1",
                        self.scenario.content_hash,
                        self.state,
                        "alice",
                        capability,
                    )

    def test_internal_capability_can_query_exact_agent_private_state(self) -> None:
        view = project_scenario_agent_state(
            "run-1",
            self.scenario.content_hash,
            self.state,
            "alice",
            self.capability(SimulationOutputAudience.INTERNAL),
        )
        expected = next(
            item for item in self.state.cognitive_state.minds if item.agent_id == "alice"
        )

        self.assertIs(view.mind, expected)

    def test_network_snapshot_is_capability_first_and_preserves_exact_identity(self) -> None:
        for audience in (
            SimulationOutputAudience.PUBLIC,
            SimulationOutputAudience.AGENT,
        ):
            owner = "alice" if audience is SimulationOutputAudience.AGENT else None
            with self.subTest(audience=audience):
                # Filtering a full snapshot after return would already cross the boundary.
                with self.assertRaisesRegex(
                    PermissionError,
                    "^scenario network state is not authorized$",
                ):
                    project_scenario_network_state(
                        self.state,
                        self.capability(audience, owner),
                    )

        for audience in (
            SimulationOutputAudience.OBJECTIVE,
            SimulationOutputAudience.ANALYST,
            SimulationOutputAudience.INTERNAL,
        ):
            with self.subTest(audience=audience):
                self.assertIs(
                    project_scenario_network_state(
                        self.state,
                        self.capability(audience),
                    ),
                    self.state.snapshot,
                )

    def test_output_view_delegates_to_agent_capability_without_cross_agent_records(
        self,
    ) -> None:
        self.assertTrue(
            any(
                record.audience is SimulationOutputAudience.AGENT
                and record.owner_agent_id != "alice"
                for record in self.batch.records
            )
        )

        view = project_scenario_output_view(
            self.batch,
            self.capability(SimulationOutputAudience.AGENT, "alice"),
        )

        # Copying the source batch records would leak Bob or Carol's private output.
        self.assertEqual(view.source_batch_hash, self.batch.content_hash)
        self.assertEqual(view.first_sequence, self.batch.first_sequence)
        self.assertEqual(view.last_sequence, self.batch.last_sequence)
        self.assertTrue(view.records)
        self.assertTrue(
            all(
                record.audience is SimulationOutputAudience.PUBLIC
                or (
                    record.audience is SimulationOutputAudience.AGENT
                    and record.owner_agent_id == "alice"
                )
                for record in view.records
            )
        )

    def test_run_view_binds_current_state_and_retained_batch_hashes(self) -> None:
        view = project_scenario_run_view(
            run_id="run-1",
            stream_id="law-firm-stream",
            scenario_hash=self.scenario.content_hash,
            coordinator_epoch=2,
            status=ScenarioRunStatus.PAUSED,
            state=self.state,
            next_sequence=self.batch.last_sequence + 1,
            output_batches=(self.batch,),
            checkpoints=(),
        )

        # Reading batch payloads into the public run view would expose retained output.
        self.assertEqual(view.round_index, self.state.round_index)
        self.assertEqual(view.state_hash, self.state.content_hash)
        self.assertEqual(view.output_batch_hashes, (self.batch.content_hash,))
        self.assertEqual(view.checkpoint_hashes, ())


if __name__ == "__main__":
    unittest.main()

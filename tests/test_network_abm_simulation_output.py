from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.scenario_compiler import (
    compile_situated_scenario_package,
    initialize_compiled_scenario,
)
from narrative_dynamics.abm.scenario_package import load_situated_scenario_package
from narrative_dynamics.abm.simulation_output import (
    filter_simulation_output,
    project_simulation_output,
)
from narrative_dynamics.abm.simulation_output_contracts import (
    SimulationAgentDecisionPayload,
    SimulationAudienceCapability,
    SimulationBlenderDeltaPayload,
    SimulationMemoryUpdatePayload,
    SimulationNetworkMetricsPayload,
    SimulationObjectiveEventPayload,
    SimulationOutputAudience,
    SimulationOutputKind,
    SimulationOutputView,
    SimulationPrivatePerceptPayload,
    SimulationSocialUpdatePayload,
    SimulationStateDeltaPayload,
)
from narrative_dynamics.abm.situated import SituatedActionKind
from narrative_dynamics.abm.situated_network import simulate_situated_network_round
from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkRoundResult,
)
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedPerceptFidelity,
)
from tests.scenario_package_fixtures import (
    mutate_json,
    refresh_manifest_hash,
    write_law_firm_package,
)


TELL_MESSAGE = "The case file has been found."


class SimulationOutputProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.scenario, cls.round_result = cls._compile_and_advance("ordinary")
        cls.tell_scenario, cls.tell_round_result = cls._compile_and_advance(
            "detected-tell",
            detected_tell=True,
        )
        cls.movement_scenario, cls.movement_round_result = cls._compile_and_advance(
            "successful-movement",
            open_archive_passage=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    @classmethod
    def _compile_and_advance(
        cls,
        name: str,
        *,
        detected_tell: bool = False,
        open_archive_passage: bool = False,
    ):
        package_root = write_law_firm_package(cls.root / name)
        mutate_json(
            package_root / "run.json",
            "/allowed_output_kinds",
            [kind.value for kind in SimulationOutputKind],
        )
        refresh_manifest_hash(package_root, "run", "run")
        if detected_tell:
            mutate_json(
                package_root / "agents/alice.json",
                "/cognition/actions/3/source_event_kinds",
                [],
            )
            mutate_json(
                package_root / "agents/alice.json",
                "/cognition/rewards/3/value",
                100.0,
            )
            mutate_json(
                package_root / "agents/alice.json",
                "/cognition/rewards/8/value",
                100.0,
            )
            refresh_manifest_hash(package_root, "agent", "alice")
            mutate_json(
                package_root / "physical/perception.json",
                "/signal_profiles/4/visually_observable",
                False,
            )
            mutate_json(
                package_root / "physical/perception.json",
                "/signal_profiles/4/auditory_intensity",
                20.0,
            )
            refresh_manifest_hash(
                package_root,
                "physical.perception",
                "perception",
            )
        if open_archive_passage:
            mutate_json(
                package_root / "physical/initial-state.json",
                "/passages/1/open",
                True,
            )
            refresh_manifest_hash(
                package_root,
                "physical.initial_state",
                "initial",
            )
        scenario = compile_situated_scenario_package(
            load_situated_scenario_package(package_root)
        )
        database = cls.root / f"{name}.sqlite3"
        initial_state = initialize_compiled_scenario(database, scenario)
        round_result = simulate_situated_network_round(
            database,
            scenario.runtime_model,
            initial_state,
        )
        return scenario, round_result

    def _project(self, *, first_sequence: int = 41):
        return project_simulation_output(
            self.scenario,
            self.round_result,
            stream_id="law-firm-run-1",
            first_sequence=first_sequence,
        )

    def test_real_round_projects_one_atomic_canonical_batch(self) -> None:
        batch = self._project()

        # Removing any accepted-round binding makes the batch point at the wrong branch.
        self.assertEqual(batch.scenario_hash, self.scenario.content_hash)
        self.assertEqual(
            batch.prior_state_hash,
            self.round_result.prior_state.content_hash,
        )
        self.assertEqual(
            batch.next_state_hash,
            self.round_result.next_state.content_hash,
        )
        self.assertEqual(batch.round_result_hash, self.round_result.content_hash)
        self.assertEqual(
            tuple(record.sequence for record in batch.records),
            tuple(range(41, 41 + len(batch.records))),
        )
        self.assertTrue(
            any(
                record.kind is SimulationOutputKind.NETWORK_METRICS
                for record in batch.records
            )
        )
        self.assertTrue(
            any(
                record.kind is SimulationOutputKind.PERCEPT_PRIVATE
                for record in batch.records
            )
        )
        self.assertTrue(
            any(
                record.kind is SimulationOutputKind.AGENT_DECISION
                for record in batch.records
            )
        )
        self.assertEqual(
            {record.kind for record in batch.records},
            {
                SimulationOutputKind.STATE_DELTA,
                SimulationOutputKind.EVENT_OBJECTIVE,
                SimulationOutputKind.PERCEPT_PRIVATE,
                SimulationOutputKind.AGENT_DECISION,
                SimulationOutputKind.MEMORY_UPDATE,
                SimulationOutputKind.SOCIAL_UPDATE,
                SimulationOutputKind.NETWORK_METRICS,
                SimulationOutputKind.BLENDER_DELTA,
            },
        )
        self.assertTrue(
            all(
                record.state_hash == self.round_result.next_state.content_hash
                for record in batch.records
            )
        )

    def test_unchanged_entities_do_not_appear_in_state_or_blender_deltas(self) -> None:
        batch = self._project()
        state_record = next(
            record
            for record in batch.records
            if record.kind is SimulationOutputKind.STATE_DELTA
        )
        blender_record = next(
            record
            for record in batch.records
            if record.kind is SimulationOutputKind.BLENDER_DELTA
        )
        self.assertIsInstance(state_record.payload, SimulationStateDeltaPayload)
        self.assertIsInstance(blender_record.payload, SimulationBlenderDeltaPayload)
        prior_world = self.round_result.prior_state.story.current_state
        next_world = self.round_result.next_state.story.current_state

        # Comparing whole round states would falsely report every stable entity as changed.
        self.assertEqual(state_record.payload.prior_snapshot_hash, prior_world.content_hash)
        self.assertEqual(state_record.payload.next_snapshot_hash, next_world.content_hash)
        self.assertEqual(state_record.payload.changed_agent_ids, ())
        self.assertEqual(state_record.payload.changed_passage_ids, ())
        self.assertEqual(state_record.payload.changed_object_ids, ())
        self.assertEqual(blender_record.payload.agent_places, ())
        self.assertEqual(blender_record.payload.passage_states, ())
        self.assertEqual(blender_record.payload.object_placements, ())

    def test_changed_agents_are_projected_by_stable_id_with_their_next_places(
        self,
    ) -> None:
        batch = project_simulation_output(
            self.movement_scenario,
            self.movement_round_result,
            stream_id="law-firm-movement",
        )
        state_payload = next(
            record.payload
            for record in batch.records
            if record.kind is SimulationOutputKind.STATE_DELTA
        )
        blender_payload = next(
            record.payload
            for record in batch.records
            if record.kind is SimulationOutputKind.BLENDER_DELTA
        )

        # Returning an empty or positional diff would omit the two accepted moves.
        self.assertEqual(state_payload.changed_agent_ids, ("alice", "carol"))
        self.assertEqual(state_payload.changed_passage_ids, ())
        self.assertEqual(state_payload.changed_object_ids, ())
        self.assertEqual(
            blender_payload.agent_places,
            (("alice", "archive"), ("carol", "archive")),
        )
        self.assertEqual(blender_payload.passage_states, ())
        self.assertEqual(blender_payload.object_placements, ())

    def test_detected_tell_percept_stays_sanitized_and_objective_event_stays_safe(
        self,
    ) -> None:
        batch = project_simulation_output(
            self.tell_scenario,
            self.tell_round_result,
            stream_id="law-firm-private-round",
        )
        transition = self.tell_round_result.transition
        self.assertIsNotNone(transition)
        projection = transition.cognitive_round.percept_cognitive_round.next_projection
        detected = next(
            percept
            for percept in projection.percepts
            if percept.fidelity is SituatedPerceptFidelity.DETECTED
        )
        percept_record = next(
            record
            for record in batch.records
            if isinstance(record.payload, SimulationPrivatePerceptPayload)
            and record.payload.percept.percept_id == detected.percept_id
        )

        # Reconstructing the source event would make a detected percept regain secrets.
        self.assertIs(percept_record.payload.percept, detected)
        self.assertIsNone(percept_record.payload.percept.actor_agent_id)
        self.assertIsNone(percept_record.payload.percept.kind)
        self.assertIsNone(percept_record.payload.percept.outcome)
        self.assertEqual(percept_record.payload.percept.details, ())

        tell_record = next(
            record
            for record in batch.records
            if isinstance(record.payload, SimulationObjectiveEventPayload)
            and record.payload.action_kind == SituatedActionKind.TELL.value
        )
        safe_payload = tell_record.payload.to_dict()
        # Copying an intent or event dictionary would expose the TELL message and details.
        self.assertEqual(
            set(safe_payload),
            {
                "event_id",
                "event_hash",
                "action_id",
                "action_kind",
                "actor_agent_id",
                "place_id",
                "target_id",
                "success",
                "cause_event_ids",
            },
        )
        self.assertNotIn(TELL_MESSAGE, repr(safe_payload))

    def test_every_private_record_owner_matches_its_payload_agent(self) -> None:
        batch = self._project()
        private_records = tuple(
            record
            for record in batch.records
            if record.audience is SimulationOutputAudience.AGENT
        )
        self.assertTrue(private_records)

        # Assigning a shared or source Agent owner would cross the privacy boundary.
        for record in private_records:
            payload = record.payload
            if isinstance(payload, SimulationPrivatePerceptPayload):
                expected_owner = payload.percept.agent_id
            elif isinstance(payload, SimulationAgentDecisionPayload):
                expected_owner = payload.decision.agent_id
            elif isinstance(payload, SimulationMemoryUpdatePayload):
                expected_owner = payload.agent_id
            elif isinstance(payload, SimulationSocialUpdatePayload):
                expected_owner = payload.observer_agent_id
            else:
                self.fail(f"unexpected private payload: {type(payload).__name__}")
            self.assertEqual(record.owner_agent_id, expected_owner)

    def test_metrics_use_the_exact_next_state_artifact_and_separate_hash_domains(
        self,
    ) -> None:
        batch = self._project()
        record = next(
            record
            for record in batch.records
            if record.kind is SimulationOutputKind.NETWORK_METRICS
        )
        self.assertIsInstance(record.payload, SimulationNetworkMetricsPayload)

        # Substituting the enclosing state hash for the nested snapshot hash breaks provenance.
        self.assertIs(record.payload.metrics, self.round_result.next_state.metrics)
        self.assertEqual(record.state_hash, self.round_result.next_state.content_hash)
        self.assertIn(
            self.round_result.next_state.metrics.snapshot_hash,
            record.source_artifact_hashes,
        )
        self.assertNotEqual(
            record.state_hash,
            self.round_result.next_state.metrics.snapshot_hash,
        )

    def test_projection_is_deterministic_and_sequences_do_not_change_payloads(self) -> None:
        first = self._project()
        repeated = self._project()
        shifted = self._project(first_sequence=91)

        # Reading mutable or ambient state would make repeated projection diverge.
        self.assertEqual(first, repeated)
        self.assertEqual(first.content_hash, repeated.content_hash)
        self.assertNotEqual(first, shifted)
        self.assertNotEqual(first.content_hash, shifted.content_hash)
        self.assertEqual(
            tuple(record.payload.content_hash for record in first.records),
            tuple(record.payload.content_hash for record in shifted.records),
        )
        self.assertTrue(
            all(
                left.content_hash != right.content_hash
                for left, right in zip(first.records, shifted.records, strict=True)
            )
        )

    def test_run_policy_filters_candidates_before_sequences_are_assigned(self) -> None:
        metrics_only = replace(
            self.scenario,
            run_policy=replace(
                self.scenario.run_policy,
                allowed_output_kinds=(SimulationOutputKind.NETWORK_METRICS.value,),
            ),
        )

        batch = project_simulation_output(
            metrics_only,
            self.round_result,
            stream_id="law-firm-metrics",
            first_sequence=17,
        )

        # Filtering after sequence assignment would leave a gapped or wrong sequence.
        self.assertEqual(len(batch.records), 1)
        self.assertEqual(batch.records[0].kind, SimulationOutputKind.NETWORK_METRICS)
        self.assertEqual(batch.records[0].sequence, 17)

    def test_unsupported_only_policy_has_one_stable_projector_failure(self) -> None:
        unsupported_only = replace(
            self.scenario,
            run_policy=replace(
                self.scenario.run_policy,
                allowed_output_kinds=(
                    SimulationOutputKind.STORY_PROGRESS.value,
                    SimulationOutputKind.NARRATIVE_SCENE.value,
                    SimulationOutputKind.COMMAND_RESULT.value,
                    SimulationOutputKind.DIAGNOSTIC.value,
                ),
            ),
        )
        messages = []

        for first_sequence in (1, 41):
            with self.subTest(first_sequence=first_sequence):
                # Empty candidates must fail at the projector, not batch arithmetic.
                with self.assertRaises(ValueError) as raised:
                    project_simulation_output(
                        unsupported_only,
                        self.round_result,
                        stream_id="law-firm-unsupported-only",
                        first_sequence=first_sequence,
                    )
                messages.append(str(raised.exception))

        self.assertEqual(
            messages,
            [
                "simulation output projection produced no supported records",
                "simulation output projection produced no supported records",
            ],
        )

    def test_filter_delegates_to_the_typed_view_without_serialization(self) -> None:
        batch = self._project()
        capability = SimulationAudienceCapability(
            SimulationOutputAudience.AGENT,
            "alice",
        )

        view = filter_simulation_output(batch, capability)

        # Rebuilding records during filtering would change typed identity or source hashes.
        self.assertEqual(view, SimulationOutputView.from_batch(batch, capability))
        self.assertTrue(
            all(
                record.audience is SimulationOutputAudience.PUBLIC
                or record.owner_agent_id == "alice"
                for record in view.records
            )
        )

    def test_legacy_round_without_transition_evidence_is_rejected(self) -> None:
        legacy = SituatedNetworkRoundResult(
            self.round_result.model_id,
            self.round_result.model_hash,
            self.round_result.prior_state,
            self.round_result.next_state,
            transition=None,
        )

        # Projecting a legacy result would fabricate evidence no accepted round retained.
        with self.assertRaises(ValueError) as raised:
            project_simulation_output(
                self.scenario,
                legacy,
                stream_id="legacy-law-firm",
            )
        self.assertEqual(str(raised.exception), "transition evidence")

    def test_scenario_must_bind_the_exact_round_runtime_model(self) -> None:
        # Accepting a foreign round would bind output to the wrong compiled scenario.
        with self.assertRaisesRegex(ValueError, "runtime model"):
            project_simulation_output(
                self.scenario,
                self.tell_round_result,
                stream_id="foreign-law-firm-round",
            )


if __name__ == "__main__":
    unittest.main()

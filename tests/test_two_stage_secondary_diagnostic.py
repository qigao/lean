from __future__ import annotations

import unittest

from narrative_dynamics.contracts import ExperimentManifest, ExperimentStage, Scenario, stable_content_hash
from narrative_dynamics.observations import (
    ObservationDataset,
    ObservationPartition,
    ObservationPartitionRole,
    ObservationRecord,
)

_DIAGNOSTIC_IMPORT_ERROR: Exception | None = None
try:
    from narrative_dynamics.external_prediction import (
        ExternalFinalPredictionArtifact,
        ExternalModelPrediction,
        ExternalSeedPrediction,
    )
    from narrative_dynamics.studies.feher_hare_two_stage_v1 import (
        build_two_stage_stay_switch_diagnostic,
    )
except Exception as error:
    _DIAGNOSTIC_IMPORT_ERROR = error


def _record(trial: int, action: str, *, previous=None):
    return ObservationRecord(
        id=f"magic_carpet/p1/{trial}",
        scenario=Scenario(
            id=f"magic-p1-{trial}",
            payload={
                "task_variant": "magic_carpet",
                "history": () if previous is None else (previous,),
            },
        ),
        counts={
            "action_0": 1 if action == "action_0" else 0,
            "action_1": 1 if action == "action_1" else 0,
        },
        metadata={
            "task_variant": "magic_carpet",
            "source_participant_id": "p1",
            "source_trial_id": trial,
            "causal_history_hash": stable_content_hash(("history", trial)),
        },
    )


def _dataset():
    previous = {
        "trial_id": 0,
        "first_stage_action": "action_0",
        "transition_common": True,
        "final_state": "state_0",
        "second_stage_action": "action_0",
        "reward": 1,
    }
    return ObservationDataset(
        name="two-stage-diagnostic",
        version="1",
        source={"kind": "external_observational"},
        provenance={"external_observational": True, "test_only": True},
        partitions=(
            ObservationPartition("train", ObservationPartitionRole.TRAIN, records=(_record(10, "action_0"),)),
            ObservationPartition("selection", ObservationPartitionRole.SELECTION_VALIDATION, records=(_record(20, "action_0"),)),
            ObservationPartition(
                "final",
                ObservationPartitionRole.FINAL_TEST,
                records=(
                    _record(0, "action_0"),
                    _record(2, "action_0", previous=previous),
                ),
            ),
        ),
    )


def _artifact(dataset):
    model = ExternalModelPrediction(
        model_name="reactive",
        frozen_model_hash=stable_content_hash({"model": "reactive"}),
        parameters=(("beta", 1.0),),
        predictions=(
            ExternalSeedPrediction(
                case_name="magic_carpet/p1/0",
                scenario_id="magic-p1-0",
                seed=301,
                metrics=(("first_stage.action_0", 0.5), ("first_stage.action_1", 0.5)),
                run_manifest_hash=stable_content_hash({"run": 0}),
            ),
            ExternalSeedPrediction(
                case_name="magic_carpet/p1/2",
                scenario_id="magic-p1-2",
                seed=301,
                metrics=(("first_stage.action_0", 0.8), ("first_stage.action_1", 0.2)),
                run_manifest_hash=stable_content_hash({"run": 2}),
            ),
        ),
    )
    manifest = ExperimentManifest(
        stage=ExperimentStage.EXTERNAL_PREDICTION,
        inputs={
            "preregistration_hash": stable_content_hash({"prereg": 1}),
            "preflight_hash": stable_content_hash({"preflight": 1}),
            "brier_protocol_hash": stable_content_hash({"brier": 1}),
            "log_protocol_hash": stable_content_hash({"log": 1}),
            "repository_revision": "test",
            "dataset_hash": dataset.content_hash,
            "final_partition_hash": dataset.partition(ObservationPartitionRole.FINAL_TEST).content_hash,
            "final_target_hash": stable_content_hash({"target": 1}),
            "metric_identity": {"name": "two-stage", "version": "1"},
            "simulation_seeds": (301,),
            "model_prediction_hashes": (stable_content_hash({"model-pred": 1}),),
        },
        parent_hashes=(stable_content_hash({"run": 0}), stable_content_hash({"run": 2})),
    )
    return ExternalFinalPredictionArtifact(
        preregistration_hash=stable_content_hash({"prereg": 1}),
        preflight_hash=stable_content_hash({"preflight": 1}),
        brier_protocol_hash=stable_content_hash({"brier": 1}),
        log_protocol_hash=stable_content_hash({"log": 1}),
        repository_revision="test",
        dataset_hash=dataset.content_hash,
        final_partition_hash=dataset.partition(ObservationPartitionRole.FINAL_TEST).content_hash,
        final_target_hash=stable_content_hash({"target": 1}),
        metric_identity={"name": "two-stage", "version": "1"},
        simulation_seeds=(301,),
        model_predictions=(model,),
        manifest=manifest,
    )


class TwoStageSecondaryDiagnosticTests(unittest.TestCase):
    def require_diagnostic(self):
        self.assertIsNone(
            _DIAGNOSTIC_IMPORT_ERROR,
            f"two-stage secondary diagnostic is missing: {_DIAGNOSTIC_IMPORT_ERROR}",
        )

    def test_adjacent_retained_pair_uses_previous_reward_transition_and_sealed_policy(self):
        self.require_diagnostic()
        dataset = _dataset()
        artifact = _artifact(dataset)
        result = build_two_stage_stay_switch_diagnostic(dataset=dataset, artifact=artifact)
        self.assertEqual(len(result.cells), 1)
        cell = result.cells[0]
        self.assertEqual((cell.previous_reward, cell.previous_transition_common), (1, True))
        self.assertEqual(cell.observation_count, 1)
        self.assertEqual(cell.observed_stay_rate, 1.0)
        self.assertEqual(dict(cell.predicted_stay_probability_by_model)["reactive"], 0.8)
        self.assertEqual(result.prediction_artifact_hash, artifact.content_hash)

    def test_diagnostic_is_task_stratified_and_has_no_runner_interface(self):
        self.require_diagnostic()
        dataset = _dataset()
        result = build_two_stage_stay_switch_diagnostic(dataset=dataset, artifact=_artifact(dataset))
        self.assertEqual({cell.task_variant for cell in result.cells}, {"magic_carpet"})


if __name__ == "__main__":
    unittest.main()

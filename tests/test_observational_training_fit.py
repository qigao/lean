from __future__ import annotations

import unittest

from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics
from narrative_dynamics.contracts import ExperimentStage, ModelRun, Scenario
from narrative_dynamics.losses import CategoricalBrierLoss, CategoricalMetricGroup
from narrative_dynamics.observations import (
    CategoricalTargetSpec,
    ObservationDataset,
    ObservationPartition,
    ObservationPartitionRole,
    ObservationRecord,
    construct_categorical_targets,
)
from narrative_dynamics.report_artifact import attest_report
from narrative_dynamics.simulation import SimulationRunner


class TrainingProbabilityModel:
    name = "training-probability-model"
    version = "1.0.0"
    implementation_revision = "training-probability-v1"

    def __init__(self) -> None:
        self.calls = 0

    def simulate(self, scenario, parameters, rng):
        self.calls += 1
        escape = float(parameters["p"])
        return ModelRun(
            events=(),
            outcome={
                "initial_policy": {
                    "scout": 0.0,
                    "escape": escape,
                    "submit": 1.0 - escape,
                }
            },
        )


def _record(record_id: str, role: str, escape: int, submit: int) -> ObservationRecord:
    return ObservationRecord(
        id=record_id,
        scenario=Scenario(id=f"{role}-{record_id}", payload={"condition": record_id}),
        counts={"scout": 0, "escape": escape, "submit": submit},
    )


def dataset() -> ObservationDataset:
    return ObservationDataset(
        name="training-fit-observations",
        version="1.0.0",
        source={"kind": "synthetic_test"},
        provenance={"purpose": "training-fit-boundary"},
        partitions=(
            ObservationPartition(
                name="train",
                role=ObservationPartitionRole.TRAIN,
                records=(
                    _record("train-1", "train", 8, 2),
                    _record("train-2", "train", 8, 2),
                ),
            ),
            ObservationPartition(
                name="selection",
                role=ObservationPartitionRole.SELECTION_VALIDATION,
                records=(_record("selection-1", "selection", 7, 3),),
            ),
            ObservationPartition(
                name="final",
                role=ObservationPartitionRole.FINAL_TEST,
                records=(_record("final-1", "final", 6, 4),),
            ),
        ),
    )


def target_spec() -> CategoricalTargetSpec:
    return CategoricalTargetSpec(
        name="initial-choice",
        version="1",
        categories=("scout", "escape", "submit"),
        metric_prefix="initial",
    )


def brier_loss() -> CategoricalBrierLoss:
    return CategoricalBrierLoss(
        (
            CategoricalMetricGroup(
                "initial-action",
                ("initial.scout", "initial.escape", "initial.submit"),
            ),
        )
    )


def targets_for(role: ObservationPartitionRole):
    return construct_categorical_targets(dataset(), role=role, spec=target_spec())


class TrainingTargetFitTests(unittest.TestCase):
    def test_train_fit_ranks_one_shared_generator_grid_across_all_train_cases(self):
        from narrative_dynamics.observations.training import fit_training_target_grid

        model = TrainingProbabilityModel()
        report = fit_training_target_grid(
            runner=SimulationRunner(),
            model=model,
            target_report=targets_for(ObservationPartitionRole.TRAIN),
            parameter_grid={"p": (value for value in (0.2, 0.5, 0.8))},
            simulation_seeds=(101, 102),
            extractor=prison_initial_action_metrics,
            loss=brier_loss(),
        )

        expected_candidates = (
            (("p", 0.2),),
            (("p", 0.5),),
            (("p", 0.8),),
        )
        self.assertIs(report.manifest.stage, ExperimentStage.TRAINING_TARGET_FIT)
        self.assertEqual(len(report.ranking), 3)
        self.assertEqual(report.best.parameters, (("p", 0.8),))
        self.assertEqual(set(report.candidate_parameters), set(expected_candidates))
        self.assertEqual(
            report.manifest.inputs.get("parameter_candidates"),
            expected_candidates,
        )
        self.assertTrue(all(len(candidate.cases) == 2 for candidate in report.ranking))
        self.assertEqual(len(report.manifest.parent_hashes), 3)
        self.assertIs(attest_report(report).require_integrity(), report)

    def test_train_fit_rejects_selection_or_final_targets_before_execution(self):
        from narrative_dynamics.observations.training import fit_training_target_grid

        model = TrainingProbabilityModel()
        for role in (
            ObservationPartitionRole.SELECTION_VALIDATION,
            ObservationPartitionRole.FINAL_TEST,
        ):
            with self.subTest(role=role):
                before = model.calls
                with self.assertRaises(ValueError):
                    fit_training_target_grid(
                        runner=SimulationRunner(),
                        model=model,
                        target_report=targets_for(role),
                        parameter_grid={"p": (0.2, 0.8)},
                        simulation_seeds=(101,),
                        extractor=prison_initial_action_metrics,
                        loss=brier_loss(),
                    )
                self.assertEqual(model.calls, before)


if __name__ == "__main__":
    unittest.main()

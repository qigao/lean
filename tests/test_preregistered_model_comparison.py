from __future__ import annotations

import unittest

from narrative_dynamics.contracts import ExperimentStage, ModelRun, Scenario, stable_content_hash
from narrative_dynamics.losses import (
    CategoricalBrierLoss,
    CategoricalLogLoss,
    CategoricalMetricGroup,
)
from narrative_dynamics.model_comparison import (
    ComparisonModel,
    compare_models_on_final_partition,
)
from narrative_dynamics.observations import (
    AdequacyThresholds,
    CategoricalTargetSpec,
    FrozenModelCandidate,
    ObservationDataset,
    ObservationPartition,
    ObservationPartitionRole,
    ObservationRecord,
    PreregisteredEvaluationProtocol,
    construct_categorical_targets,
)
from narrative_dynamics.report_artifact import attest_report
from narrative_dynamics.simulation import SimulationRunner


class ProbabilityModel:
    name = "comparison-probability-model"
    version = "1.0.0"

    def simulate(self, scenario, parameters, rng):
        probability = float(parameters["p"])
        return ModelRun(
            events=(),
            outcome={"policy": {"a": probability, "b": 1.0 - probability}},
        )


class DriftedProbabilityModel:
    name = "comparison-probability-model"
    version = "1.0.0"

    def simulate(self, scenario, parameters, rng):
        probability = float(parameters["p"])
        return ModelRun(
            events=(),
            outcome={"policy": {"a": probability, "b": 1.0 - probability}},
        )


def policy_metrics(trace):
    return {
        "choice.a": float(trace.outcome["policy"]["a"]),
        "choice.b": float(trace.outcome["policy"]["b"]),
    }


def observation_dataset() -> ObservationDataset:
    def record(record_id: str, scenario_id: str, a: int, b: int) -> ObservationRecord:
        return ObservationRecord(
            id=record_id,
            scenario=Scenario(id=scenario_id, payload={"condition": scenario_id}),
            counts={"a": a, "b": b},
        )

    return ObservationDataset(
        name="comparison-observations",
        version="1.0.0",
        source="synthetic:test-fixture",
        provenance={"purpose": "comparison"},
        partitions=(
            ObservationPartition(
                name="train",
                role=ObservationPartitionRole.TRAIN,
                records=(record("train-1", "train-scenario", 6, 4),),
            ),
            ObservationPartition(
                name="selection",
                role=ObservationPartitionRole.SELECTION_VALIDATION,
                records=(record("selection-1", "selection-scenario", 7, 3),),
            ),
            ObservationPartition(
                name="final",
                role=ObservationPartitionRole.FINAL_TEST,
                records=(
                    record("final-1", "final-scenario-1", 8, 2),
                    record("final-2", "final-scenario-2", 8, 2),
                ),
            ),
        ),
    )


def protocol_fixture():
    dataset = observation_dataset()
    spec = CategoricalTargetSpec(
        name="choice-target",
        version="1",
        categories=("a", "b"),
        metric_prefix="choice",
    )
    loss = CategoricalBrierLoss(
        (CategoricalMetricGroup("choice", ("choice.a", "choice.b")),)
    )
    model = ProbabilityModel()
    selection_hash = stable_content_hash({"selection": "complete"})
    baseline = FrozenModelCandidate.freeze(
        name="baseline",
        model=model,
        parameters={"p": 0.8},
        selection_manifest_hash=selection_hash,
    )
    alternative = FrozenModelCandidate.freeze(
        name="alternative",
        model=model,
        parameters={"p": 0.2},
        selection_manifest_hash=selection_hash,
    )
    protocol = PreregisteredEvaluationProtocol.create(
        name="comparison-protocol",
        version="1",
        dataset=dataset,
        target_spec=spec,
        loss=loss,
        simulation_seeds=(11, 12),
        baseline_name="baseline",
        candidates=(baseline, alternative),
        thresholds=AdequacyThresholds(max_mean_loss=0.0, max_worst_loss=0.0),
    )
    final_targets = construct_categorical_targets(
        dataset,
        role=ObservationPartitionRole.FINAL_TEST,
        spec=spec,
    )
    return dataset, spec, loss, model, baseline, alternative, protocol, final_targets


class FairModelComparisonTests(unittest.TestCase):
    def test_frozen_models_share_final_cases_loss_seeds_and_report_lineage(self):
        (
            _dataset,
            _spec,
            loss,
            model,
            baseline,
            alternative,
            protocol,
            final_targets,
        ) = protocol_fixture()

        report = compare_models_on_final_partition(
            runner=SimulationRunner(),
            models=(
                ComparisonModel(frozen=alternative, model=model),
                ComparisonModel(frozen=baseline, model=model),
            ),
            target_set=final_targets,
            extractor=policy_metrics,
            loss=loss,
            simulation_seeds=(11, 12),
            protocol=protocol,
        )

        self.assertEqual(report.manifest.stage, ExperimentStage.MODEL_COMPARISON)
        self.assertEqual(report.baseline_name, "baseline")
        self.assertEqual(report.best.name, "baseline")
        self.assertTrue(report.baseline_adequate)
        self.assertEqual(report.entry_map["baseline"].mean_loss, 0.0)
        self.assertEqual(report.entry_map["baseline"].delta_mean_from_baseline, 0.0)
        self.assertGreater(report.entry_map["alternative"].mean_loss, 0.0)
        self.assertGreater(report.entry_map["alternative"].delta_mean_from_baseline, 0.0)
        self.assertEqual(len(report.manifest.parent_hashes), 2)
        self.assertIs(attest_report(report).require_integrity(), report)

    def test_comparison_rejects_partition_loss_seed_candidate_and_runtime_drift(self):
        (
            dataset,
            spec,
            loss,
            model,
            baseline,
            alternative,
            protocol,
            final_targets,
        ) = protocol_fixture()
        models = (
            ComparisonModel(frozen=baseline, model=model),
            ComparisonModel(frozen=alternative, model=model),
        )

        with self.assertRaises(ValueError):
            compare_models_on_final_partition(
                runner=SimulationRunner(),
                models=models,
                target_set=construct_categorical_targets(
                    dataset,
                    role=ObservationPartitionRole.SELECTION_VALIDATION,
                    spec=spec,
                ),
                extractor=policy_metrics,
                loss=loss,
                simulation_seeds=(11, 12),
                protocol=protocol,
            )

        with self.assertRaises(ValueError):
            compare_models_on_final_partition(
                runner=SimulationRunner(),
                models=models,
                target_set=final_targets,
                extractor=policy_metrics,
                loss=loss,
                simulation_seeds=(11, 99),
                protocol=protocol,
            )

        different_loss = CategoricalLogLoss(
            (CategoricalMetricGroup("choice", ("choice.a", "choice.b")),)
        )
        with self.assertRaises(ValueError):
            compare_models_on_final_partition(
                runner=SimulationRunner(),
                models=models,
                target_set=final_targets,
                extractor=policy_metrics,
                loss=different_loss,
                simulation_seeds=(11, 12),
                protocol=protocol,
            )

        substituted = FrozenModelCandidate.freeze(
            name="alternative",
            model=model,
            parameters={"p": 0.3},
            selection_manifest_hash=alternative.selection_manifest_hash,
        )
        with self.assertRaises(ValueError):
            compare_models_on_final_partition(
                runner=SimulationRunner(),
                models=(
                    ComparisonModel(frozen=baseline, model=model),
                    ComparisonModel(frozen=substituted, model=model),
                ),
                target_set=final_targets,
                extractor=policy_metrics,
                loss=loss,
                simulation_seeds=(11, 12),
                protocol=protocol,
            )

        with self.assertRaises(ValueError):
            compare_models_on_final_partition(
                runner=SimulationRunner(),
                models=(
                    ComparisonModel(frozen=baseline, model=DriftedProbabilityModel()),
                    ComparisonModel(frozen=alternative, model=model),
                ),
                target_set=final_targets,
                extractor=policy_metrics,
                loss=loss,
                simulation_seeds=(11, 12),
                protocol=protocol,
            )


if __name__ == "__main__":
    unittest.main()

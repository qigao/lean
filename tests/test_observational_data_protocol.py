from __future__ import annotations

from dataclasses import replace
import unittest

from narrative_dynamics.contracts import ExperimentStage, Scenario, stable_content_hash
from narrative_dynamics.losses import CategoricalBrierLoss, CategoricalMetricGroup
from narrative_dynamics.manifest import component_identity
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


class ProtocolModel:
    name = "protocol-model"
    version = "1.0.0"

    def simulate(self, scenario, parameters, rng):
        raise AssertionError("dataset protocol tests must not execute a model")


def record(
    record_id: str,
    scenario_id: str,
    a: int,
    b: int,
    *,
    group: str,
) -> ObservationRecord:
    return ObservationRecord(
        id=record_id,
        scenario=Scenario(
            id=scenario_id,
            payload={"group": group, "difficulty": float(a + b)},
        ),
        counts={"a": a, "b": b},
        metadata={"source_group": group},
    )


def dataset(*, reverse: bool = False) -> ObservationDataset:
    train_records = (
        record("train-1", "train-scenario-1", 6, 4, group="train"),
        record("train-2", "train-scenario-2", 7, 3, group="train"),
    )
    partitions = (
        ObservationPartition(
            name="train",
            role=ObservationPartitionRole.TRAIN,
            records=tuple(reversed(train_records)) if reverse else train_records,
        ),
        ObservationPartition(
            name="selection",
            role=ObservationPartitionRole.SELECTION_VALIDATION,
            records=(
                record(
                    "selection-1",
                    "selection-scenario-1",
                    7,
                    3,
                    group="selection",
                ),
            ),
        ),
        ObservationPartition(
            name="final",
            role=ObservationPartitionRole.FINAL_TEST,
            records=(
                record("final-1", "final-scenario-1", 8, 2, group="final"),
                record("final-2", "final-scenario-2", 8, 2, group="final"),
            ),
        ),
    )
    return ObservationDataset(
        name="prison-observations",
        version="1.0.0",
        source="synthetic:test-fixture",
        provenance={"collector": "unit-test", "release": 1},
        partitions=tuple(reversed(partitions)) if reverse else partitions,
    )


class ObservationDatasetTests(unittest.TestCase):
    def test_dataset_identity_is_order_stable_detached_and_role_complete(self):
        provenance = {"collector": "unit-test", "release": 1}
        original = dataset()
        reordered = dataset(reverse=True)

        self.assertEqual(original.content_hash, reordered.content_hash)
        self.assertEqual(
            tuple(partition.role for partition in original.partitions),
            (
                ObservationPartitionRole.TRAIN,
                ObservationPartitionRole.SELECTION_VALIDATION,
                ObservationPartitionRole.FINAL_TEST,
            ),
        )
        self.assertEqual(
            original.partition(ObservationPartitionRole.FINAL_TEST).name,
            "final",
        )

        provenance["release"] = 999
        self.assertEqual(original.provenance["release"], 1)

        missing_final = tuple(
            partition
            for partition in original.partitions
            if partition.role is not ObservationPartitionRole.FINAL_TEST
        )
        with self.assertRaises(ValueError):
            ObservationDataset(
                name="missing-final",
                version="1",
                source="test",
                provenance={},
                partitions=missing_final,
            )

    def test_cross_partition_ids_and_renamed_observations_are_rejected(self):
        base = dataset()
        train = base.partition(ObservationPartitionRole.TRAIN)
        selection = base.partition(ObservationPartitionRole.SELECTION_VALIDATION)
        final = base.partition(ObservationPartitionRole.FINAL_TEST)

        duplicate_id = ObservationPartition(
            name="selection",
            role=ObservationPartitionRole.SELECTION_VALIDATION,
            records=(
                record(
                    "train-1",
                    "different-scenario",
                    9,
                    1,
                    group="selection",
                ),
            ),
        )
        with self.assertRaises(ValueError):
            ObservationDataset(
                name="id-leak",
                version="1",
                source="test",
                provenance={},
                partitions=(train, duplicate_id, final),
            )

        source_record = train.records[0]
        renamed_copy = ObservationRecord(
            id="renamed-copy",
            scenario=source_record.scenario,
            counts=source_record.counts,
            metadata=source_record.metadata,
        )
        duplicate_content = ObservationPartition(
            name="selection",
            role=ObservationPartitionRole.SELECTION_VALIDATION,
            records=(renamed_copy,),
        )
        with self.assertRaises(ValueError):
            ObservationDataset(
                name="content-leak",
                version="1",
                source="test",
                provenance={},
                partitions=(train, duplicate_content, final),
            )


class TargetConstructionTests(unittest.TestCase):
    def test_categorical_targets_bind_dataset_partition_spec_and_records(self):
        observations = dataset()
        spec = CategoricalTargetSpec(
            name="initial-action-target",
            version="1",
            categories=("a", "b"),
            metric_prefix="choice",
            pseudocount=1.0,
        )
        target_set = construct_categorical_targets(
            observations,
            role=ObservationPartitionRole.FINAL_TEST,
            spec=spec,
        )

        self.assertEqual(target_set.role, ObservationPartitionRole.FINAL_TEST)
        self.assertEqual(target_set.dataset_hash, observations.content_hash)
        self.assertEqual(target_set.spec_hash, spec.content_hash)
        self.assertEqual(target_set.manifest.stage, ExperimentStage.OBSERVATION_TARGET_CONSTRUCTION)
        self.assertEqual(len(target_set.cases), 2)
        self.assertEqual(
            target_set.cases[0].target_map,
            {"choice.a": 0.75, "choice.b": 0.25},
        )
        self.assertEqual(target_set.cases[0].observation_count, 10)
        self.assertIs(attest_report(target_set).require_integrity(), target_set)

        with self.assertRaises(ValueError):
            construct_categorical_targets(
                observations,
                role=ObservationPartitionRole.FINAL_TEST,
                spec=CategoricalTargetSpec(
                    name="wrong-schema",
                    version="1",
                    categories=("a", "b", "c"),
                    metric_prefix="choice",
                ),
            )


class PreregisteredProtocolTests(unittest.TestCase):
    def test_protocol_freezes_candidates_loss_seeds_thresholds_and_precommitment(self):
        observations = dataset()
        spec = CategoricalTargetSpec(
            name="initial-action-target",
            version="1",
            categories=("a", "b"),
            metric_prefix="choice",
        )
        loss = CategoricalBrierLoss(
            (CategoricalMetricGroup("choice", ("choice.a", "choice.b")),)
        )
        selection_hash = stable_content_hash({"selection": "frozen"})
        model = ProtocolModel()
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
        thresholds = AdequacyThresholds(max_mean_loss=0.01, max_worst_loss=0.02)

        protocol = PreregisteredEvaluationProtocol.create(
            name="prison-baseline-v1",
            version="1",
            dataset=observations,
            target_spec=spec,
            loss=loss,
            simulation_seeds=(101, 102),
            baseline_name="baseline",
            candidates=(alternative, baseline),
            thresholds=thresholds,
        )

        self.assertEqual(protocol.baseline_name, "baseline")
        self.assertEqual(protocol.dataset_hash, observations.content_hash)
        self.assertEqual(
            protocol.final_partition_hash,
            observations.partition(ObservationPartitionRole.FINAL_TEST).content_hash,
        )
        self.assertEqual(protocol.candidate_names, ("alternative", "baseline"))
        self.assertEqual(protocol.content_hash, protocol.declared_precommitment_hash)
        self.assertEqual(protocol.thresholds, thresholds)
        self.assertEqual(
            baseline.model_identity,
            component_identity(model),
        )

        with self.assertRaises(ValueError):
            replace(
                protocol,
                declared_precommitment_hash="sha256:" + "0" * 64,
            )


if __name__ == "__main__":
    unittest.main()

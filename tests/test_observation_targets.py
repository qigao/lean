from __future__ import annotations

import unittest

from narrative_dynamics.contracts import ExperimentStage
from narrative_dynamics.losses import CategoricalMetricGroup
from narrative_dynamics.observations import (
    CategoricalTargetPlan,
    ObservationPartitionRole,
    construct_categorical_targets,
)
from narrative_dynamics.report_artifact import attest_report
from narrative_dynamics.validation import EvaluationRole
from tests.test_observation_dataset import make_dataset


class ObservationTargetTests(unittest.TestCase):
    def test_counts_become_provenance_bound_selection_targets_and_suite(self):
        dataset = make_dataset()
        plan = CategoricalTargetPlan(
            name="choice-frequency-v1",
            version="1",
            role=ObservationPartitionRole.SELECTION_VALIDATION,
            groups=(
                CategoricalMetricGroup("choice", ("choice.a", "choice.b")),
            ),
            seeds_by_case={"selection-case": (11, 12)},
        )
        report = construct_categorical_targets(dataset, plan)

        self.assertEqual(report.dataset_hash, dataset.content_hash)
        self.assertEqual(
            report.partition_hash,
            dataset.partition(ObservationPartitionRole.SELECTION_VALIDATION).content_hash,
        )
        self.assertEqual(report.plan_hash, plan.content_hash)
        self.assertEqual(report.role, ObservationPartitionRole.SELECTION_VALIDATION)
        self.assertEqual(report.manifest.stage, ExperimentStage.TARGET_CONSTRUCTION)
        case = report.cases[0]
        self.assertEqual(case.count_map, {"choice.a": 8, "choice.b": 2})
        self.assertEqual(case.target_map, {"choice.a": 0.8, "choice.b": 0.2})
        self.assertEqual(case.observation_ids, tuple(f"selection-{index}" for index in range(10)))
        self.assertEqual(case.seeds, (11, 12))

        suite = report.as_held_out_suite()
        self.assertEqual(suite.role, EvaluationRole.SELECTION_VALIDATION)
        self.assertEqual(suite.cases[0].target, case.target_map)
        self.assertEqual(suite.cases[0].seeds, (11, 12))
        self.assertIs(attest_report(report).require_integrity(), report)

    def test_target_groups_count_totals_seed_plan_and_train_role_fail_closed(self):
        dataset = make_dataset()

        def plan(groups, seeds, role=ObservationPartitionRole.SELECTION_VALIDATION):
            return CategoricalTargetPlan(
                name="bad-plan",
                version="1",
                role=role,
                groups=groups,
                seeds_by_case=seeds,
            )

        with self.assertRaises(ValueEror):
            construct_categorical_targets(
                dataset,
                plan(
                    (CategoricalMetricGroup("incomplete", ("choice.a",)),),
                    {"selection-case": (1,)},
                ),
            )
        with self.assertRaises(ValueEror):
            construct_categorical_targets(
                dataset,
                plan(
                    (
                        CategoricalMetricGroup("first", ("choice.a",)),
                        CategoricalMetricGroup("second", ("choice.a", "choice.b")),
                    ),
                    {"selection-case": (1,)},
                ),
            )
        for seeds in ({}, {"selection-case": ()}, {"selection-case": (1,), "extra": (2,)}):
            with self.subTest(seeds=seeds):
                with self.assertRaises(ValueError):
                    construct_categorical_targets(
                        dataset,
                        plan(
                            (CategoricalMetricGroup("choice", ("choice.a", "choice.b")),),
                            seeds,
                        ),
                    )

        train_report = construct_categorical_targets(
            dataset,
            plan(
                (CategoricalMetricGroup("choice", ("choice.a", "choice.b")),),
                {"train-case": (1,)},
                role=ObservationPartitionRole.TRAIN,
            ),
        )
        with self.assertRaises(ValueEror):
            train_report.as_held_out_suite()


if __name__ == "__main__":
    unittest.main()

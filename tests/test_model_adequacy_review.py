from __future__ import annotations

import unittest

from narrative_dynamics.adequacy import (
    CoverageAxis,
    CoverageBin,
    diagnose_final_test_coverage,
)
from narrative_dynamics.contracts import (
    ExperimentManifest,
    ExperimentStage,
    Scenario,
)
from narrative_dynamics.losses import (
    CategoricalBrierLoss,
    CategoricalMetricGroup,
    metric_loss_identity,
)
from narrative_dynamics.manifest import scenario_identity, stable_content_hash
from narrative_dynamics.validation import (
    EvaluationRole,
    FinalTestReport,
    HeldOutCase,
    HeldOutCaseEvaluation,
    HeldOutSuite,
    HeldOutValidationReport,
)


class ModelAdequacyReviewTests(unittest.TestCase):
    def test_categorical_loss_rejects_ungrouped_metrics(self):
        loss = CategoricalBrierLoss(
            groups=(CategoricalMetricGroup("choice", ("choice.a", "choice.b")),)
        )
        with self.assertRaises(ValueError):
            loss(
                {"choice.a": 0.8, "choice.b": 0.2, "utility": 1.0},
                {"choice.a": 0.8, "choice.b": 0.2, "utility": 1.0},
            )

    def test_forged_loss_identity_is_rejected(self):
        class ForgedLoss:
            def __call__(self, observed, target, *, weights=None):
                return 0.0

            def manifest_identity(self):
                return {
                    "name": "forged",
                    "version": "1",
                    "content_hash": "sha256:" + "0" * 64,
                }

        with self.assertRaises(ValueError):
            metric_loss_identity(ForgedLoss())

    def test_seed_reuse_counts_duplicates_within_one_case(self):
        scenario = Scenario(id="duplicate-seeds", payload={"prior_weak": 0.5})
        case = HeldOutCase(
            scenario=scenario,
            seeds=(7, 7),
            target={"x": 0.0},
            name="duplicate-seeds",
        )
        suite = HeldOutSuite(
            name="duplicate-seed-suite",
            role=EvaluationRole.FINAL_TEST,
            cases=(case,),
        )
        validation_manifest = ExperimentManifest(
            stage=ExperimentStage.HELD_OUT_VALIDATION,
            inputs={
                "cases": (
                    {
                        "name": case.effective_name,
                        "scenario": scenario_identity(case.scenario),
                        "seeds": tuple(case.seeds),
                        "target_hash": stable_content_hash(case.target),
                        "weights_hash": None,
                    },
                ),
            },
        )
        validation = HeldOutValidationReport(
            parameters=(("beta", 2.0),),
            cases=(
                HeldOutCaseEvaluation(
                    name="duplicate-seeds",
                    scenario_id="duplicate-seeds",
                    metrics=(("x", 0.0),),
                    target=(("x", 0.0),),
                    loss=0.0,
                ),
            ),
            mean_loss=0.0,
            worst_loss=0.0,
            manifest=validation_manifest,
        )
        final = FinalTestReport(
            suite_name=suite.name,
            role=EvaluationRole.FINAL_TEST,
            validation=validation,
            manifest=ExperimentManifest(
                stage=ExperimentStage.FINAL_TEST,
                inputs={"test": "seed-reuse"},
                parent_hashes=(validation_manifest.content_hash,),
            ),
        )
        report = diagnose_final_test_coverage(
            final_report=final,
            suite=suite,
            axes=(
                CoverageAxis(
                    name="prior",
                    field="prior_weak",
                    bins=(CoverageBin("all", 0.0, 1.0, include_upper=True),),
                ),
            ),
        )
        self.assertEqual(report.total_seed_count, 2)
        self.assertEqual(report.unique_seed_count, 1)
        self.assertEqual(report.reused_seeds, (7,))

    def test_coverage_rejects_same_ids_with_different_payload_or_seed_plan(self):
        original = Scenario(id="same-id", payload={"prior_weak": 0.5})
        case = HeldOutCase(
            scenario=original,
            seeds=(11,),
            target={"x": 0.0},
            name="same-name",
        )
        suite = HeldOutSuite(
            name="bound-suite",
            role=EvaluationRole.FINAL_TEST,
            cases=(case,),
        )
        validation_manifest = ExperimentManifest(
            stage=ExperimentStage.HELD_OUT_VALIDATION,
            inputs={
                "cases": (
                    {
                        "name": case.effective_name,
                        "scenario": scenario_identity(case.scenario),
                        "seeds": tuple(case.seeds),
                        "target_hash": stable_content_hash(case.target),
                        "weights_hash": None,
                    },
                ),
            },
        )
        validation = HeldOutValidationReport(
            parameters=(("beta", 2.0),),
            cases=(
                HeldOutCaseEvaluation(
                    name=case.effective_name,
                    scenario_id=case.scenario.id,
                    metrics=(("x", 0.0),),
                    target=(("x", 0.0),),
                    loss=0.0,
                ),
            ),
            mean_loss=0.0,
            worst_loss=0.0,
            manifest=validation_manifest,
        )
        final = FinalTestReport(
            suite_name=suite.name,
            role=EvaluationRole.FINAL_TEST,
            validation=validation,
            manifest=ExperimentManifest(
                stage=ExperimentStage.FINAL_TEST,
                inputs={"test": "suite-binding"},
                parent_hashes=(validation_manifest.content_hash,),
            ),
        )
        axis = CoverageAxis(
            name="prior",
            field="prior_weak",
            bins=(CoverageBin("all", 0.0, 1.0, include_upper=True),),
        )

        changed_payload = HeldOutSuite(
            name=suite.name,
            role=EvaluationRole.FINAL_TEST,
            cases=(
                HeldOutCase(
                    scenario=Scenario(
                        id=case.scenario.id,
                        payload={"prior_weak": 0.8},
                    ),
                    seeds=case.seeds,
                    target=case.target,
                    name=case.name,
                ),
            ),
        )
        with self.assertRaises(ValueError):
            diagnose_final_test_coverage(
                final_report=final,
                suite=changed_payload,
                axes=(axis,),
            )

        changed_seeds = HeldOutSuite(
            name=suite.name,
            role=EvaluationRole.FINAL_TEST,
            cases=(
                HeldOutCase(
                    scenario=case.scenario,
                    seeds=(12,),
                    target=case.target,
                    name=case.name,
                ),
            ),
        )
        with self.assertRaises(ValueError):
            diagnose_final_test_coverage(
                final_report=final,
                suite=changed_seeds,
                axes=(axis,),
            )


if __name__ == "__main__":
    unittest.main()

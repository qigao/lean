from __future__ import annotations

import unittest

from narrative_dynamics.adequacy import (
    CoverageAxis,
    CoverageBin,
    diagnose_final_test_coverage,
)
from narrative_dynamics.contracts import ExperimentStage
from narrative_dynamics.losses import (
    CategoricalBrierLoss,
    CategoricalMetricGroup,
)
from narrative_dynamics.report_artifact import attest_report
from narrative_dynamics.validation import (
    EvaluationRole,
    HeldOutCase,
    HeldOutSuite,
    evaluate_on_final_test_suite,
)
from tests.model_adequacy_fixtures import (
    adequacy_runner,
    bound_prison_source,
    initial_policy_metrics,
    initial_policy_target,
    prison_scenario,
)


class FinalTestCoverageTests(unittest.TestCase):
    def setUp(self):
        self.runner = adequacy_runner()
        self.model = bound_prison_source()
        self.loss = CategoricalBrierLoss(
            groups=(
                CategoricalMetricGroup(
                    "initial-action",
                    ("initial.scout", "initial.escape", "initial.submit"),
                ),
            )
        )

    def build_final_report(self):
        cases = []
        definitions = (
            ("coverage-low", 0.2, 1, 701, 601),
            ("coverage-middle", 0.5, 2, 701, 602),
            ("coverage-high", 0.8, 1, 703, 603),
        )
        for name, prior, horizon, evaluation_seed, target_seed in definitions:
            scenario = prison_scenario(
                name,
                prior_weak=prior,
                horizon=horizon,
            )
            cases.append(
                HeldOutCase(
                    name=name,
                    scenario=scenario,
                    seeds=(evaluation_seed,),
                    target=initial_policy_target(
                        self.runner,
                        self.model,
                        scenario,
                        2.0,
                        seed=target_seed,
                    ),
                )
            )
        suite = HeldOutSuite(
            name="prison-final-coverage",
            role=EvaluationRole.FINAL_TEST,
            cases=tuple(cases),
        )
        final = evaluate_on_final_test_suite(
            runner=self.runner,
            model=self.model,
            parameters={"beta": 2.0},
            suite=suite,
            extractor=initial_policy_metrics,
            loss=self.loss,
        )
        return suite, final

    @staticmethod
    def complete_axes():
        return (
            CoverageAxis(
                name="prior",
                field="prior_weak",
                bins=(
                    CoverageBin("low", 0.0, 0.34),
                    CoverageBin("middle", 0.34, 0.67),
                    CoverageBin("high", 0.67, 1.0, include_upper=True),
                ),
            ),
            CoverageAxis(
                name="horizon",
                field="horizon",
                bins=(
                    CoverageBin("one", 1.0, 1.0, include_upper=True),
                    CoverageBin("two", 2.0, 2.0, include_upper=True),
                ),
            ),
        )

    def test_final_suite_reports_strata_and_seed_reuse(self):
        suite, final = self.build_final_report()
        report = diagnose_final_test_coverage(
            final_report=final,
            suite=suite,
            axes=self.complete_axes(),
        )

        self.assertTrue(report.complete)
        self.assertEqual(report.case_count, 3)
        self.assertEqual(report.total_seed_count, 3)
        self.assertEqual(report.unique_seed_count, 2)
        self.assertEqual(report.reused_seeds, (701,))
        self.assertEqual(
            report.axis_map["prior"].count_map,
            {"low": 1, "middle": 1, "high": 1},
        )
        self.assertEqual(report.axis_map["horizon"].missing_bins, ())
        self.assertIs(report.manifest.stage, ExperimentStage.FINAL_TEST_COVERAGE)
        self.assertIs(attest_report(report).require_integrity(), report)

    def test_missing_bin_and_mismatched_suite_are_explicit(self):
        suite, final = self.build_final_report()
        incomplete_prior = CoverageAxis(
            name="prior",
            field="prior_weak",
            bins=(
                CoverageBin("low", 0.0, 0.34),
                CoverageBin("middle", 0.34, 0.67),
                CoverageBin("high", 0.67, 0.9),
                CoverageBin("extreme", 0.9, 1.0, include_upper=True),
            ),
        )
        report = diagnose_final_test_coverage(
            final_report=final,
            suite=suite,
            axes=(incomplete_prior,),
        )
        self.assertFalse(report.complete)
        self.assertEqual(
            report.axis_map["prior"].missing_bins,
            ("extreme",),
        )

        mismatched = HeldOutSuite(
            name="wrong-suite",
            role=EvaluationRole.FINAL_TEST,
            cases=suite.cases,
        )
        with self.assertRaises(ValueError):
            diagnose_final_test_coverage(
                final_report=final,
                suite=mismatched,
                axes=self.complete_axes(),
            )

        selection = HeldOutSuite(
            name=suite.name,
            role=EvaluationRole.SELECTION_VALIDATION,
            cases=suite.cases,
        )
        with self.assertRaises(ValueError):
            diagnose_final_test_coverage(
                final_report=final,
                suite=selection,
                axes=self.complete_axes(),
            )

    def test_overlapping_bins_are_rejected(self):
        with self.assertRaises(ValueError):
            CoverageAxis(
                name="overlap",
                field="prior_weak",
                bins=(
                    CoverageBin("first", 0.0, 0.6, include_upper=True),
                    CoverageBin("second", 0.5, 1.0, include_upper=True),
                ),
            )


if __name__ == "__main__":
    unittest.main()

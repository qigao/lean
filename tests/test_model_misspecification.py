from __future__ import annotations

import unittest

from narrative_dynamics.adequacy import assess_misspecification
from narrative_dynamics.contracts import ExperimentStage
from narrative_dynamics.losses import (
    CategoricalBrierLoss,
    CategoricalMetricGroup,
)
from narrative_dynamics.report_artifact import attest_report
from narrative_dynamics.validation import HeldOutCase
from tests.model_adequacy_fixtures import (
    adequacy_runner,
    bound_prison_source,
    initial_policy_metrics,
    initial_policy_target,
    prison_scenario,
)


class PrisonMisspecificationTests(unittest.TestCase):
    def setUp(self):
        self.runner = adequacy_runner()
        self.model = bound_prison_source()
        self.loss = CategoricalBrierLoss(
            groups=(
                CategoricalMetricGroup(
                    name="initial-action",
                    keys=(
                        "initial.scout",
                        "initial.escape",
                        "initial.submit",
                    ),
                ),
            )
        )
        self.grid = {"beta": (0.5, 2.0, 5.0)}

    def case(self, name, beta, seed):
        scenario = prison_scenario(name)
        return HeldOutCase(
            name=name,
            scenario=scenario,
            seeds=(seed + 100,),
            target=initial_policy_target(
                self.runner,
                self.model,
                scenario,
                beta,
                seed=seed,
            ),
        )

    def test_common_generating_beta_is_adequate_on_shared_grid(self):
        report = assess_misspecification(
            runner=self.runner,
            model=self.model,
            suite_name="common-beta",
            parameter_grid=self.grid,
            cases=(
                self.case("common-a", 2.0, 11),
                self.case("common-b", 2.0, 12),
            ),
            extractor=initial_policy_metrics,
            loss=self.loss,
            max_mean_loss=0.0,
            max_worst_loss=0.0,
        )

        self.assertTrue(report.adequate)
        self.assertFalse(report.misspecified)
        self.assertEqual(report.best.parameters, (("beta", 2.0),))
        self.assertEqual(report.best.mean_loss, 0.0)
        self.assertEqual(report.best.worst_loss, 0.0)
        self.assertIs(report.manifest.stage, ExperimentStage.MODEL_MISSPECIFICATION)
        self.assertIs(attest_report(report).require_integrity(), report)

    def test_incompatible_generating_betas_leave_positive_residual_floor(self):
        report = assess_misspecification(
            runner=self.runner,
            model=self.model,
            suite_name="heterogeneous-beta",
            parameter_grid=self.grid,
            cases=(
                self.case("heterogeneous-low", 0.5, 21),
                self.case("heterogeneous-high", 5.0, 22),
            ),
            extractor=initial_policy_metrics,
            loss=self.loss,
            max_mean_loss=0.0,
            max_worst_loss=0.0,
        )

        self.assertFalse(report.adequate)
        self.assertTrue(report.misspecified)
        self.assertGreater(report.best.mean_loss, 0.0)
        self.assertGreater(report.best.worst_loss, 0.0)
        self.assertEqual(len(report.ranking), 3)
        self.assertEqual(
            tuple(candidate.mean_loss for candidate in report.ranking),
            tuple(sorted(candidate.mean_loss for candidate in report.ranking)),
        )

    def test_misspecification_thresholds_must_be_non_negative(self):
        with self.assertRaises(ValueError):
            assess_misspecification(
                runner=self.runner,
                model=self.model,
                suite_name="invalid-threshold",
                parameter_grid=self.grid,
                cases=(self.case("threshold-case", 2.0, 31),),
                extractor=initial_policy_metrics,
                loss=self.loss,
                max_mean_loss=-1.0,
                max_worst_loss=0.0,
            )


if __name__ == "__main__":
    unittest.main()

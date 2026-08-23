from __future__ import annotations

import unittest

from narrative_dynamics.adapters.prison_pomdp import prison_policy_metrics
from narrative_dynamics.adequacy import (
    FactorSource,
    LocalFactor,
    local_factor_interaction_report,
)
from narrative_dynamics.contracts import ExperimentStage
from narrative_dynamics.report_artifact import attest_report
from tests.model_adequacy_fixtures import (
    adequacy_runner,
    bound_prison_source,
    prison_scenario,
)


class PrisonFactorInteractionTests(unittest.TestCase):
    def setUp(self):
        self.runner = adequacy_runner()
        self.model = bound_prison_source()
        self.scenario = prison_scenario(
            "interaction-prison",
            signal_accuracy=0.8,
        )

    def test_beta_and_signal_accuracy_have_reported_mixed_sensitivity(self):
        report = local_factor_interaction_report(
            runner=self.runner,
            model=self.model,
            scenario=self.scenario,
            parameters={"beta": 2.0},
            seeds=(31, 32),
            extractor=prison_policy_metrics,
            first=LocalFactor(
                source=FactorSource.PARAMETER,
                name="beta",
                step=0.25,
            ),
            second=LocalFactor(
                source=FactorSource.SCENARIO,
                name="signal_accuracy",
                step=0.05,
            ),
        )

        self.assertEqual(len(report.corners), 4)
        self.assertIn("initial.scout", report.interaction_map)
        self.assertGreater(abs(report.interaction_map["initial.scout"]), 1e-8)
        self.assertIs(report.manifest.stage, ExperimentStage.INTERACTION_SENSITIVITY)
        self.assertIs(attest_report(report).require_integrity(), report)
        self.assertTrue(
            all(len(corner.run_manifest_hashes) == 2 for corner in report.corners)
        )

    def test_duplicate_or_invalid_factors_fail_before_simulation(self):
        duplicate = LocalFactor(
            source=FactorSource.PARAMETER,
            name="beta",
            step=0.25,
        )
        with self.assertRaises(ValueError):
            local_factor_interaction_report(
                runner=self.runner,
                model=self.model,
                scenario=self.scenario,
                parameters={"beta": 2.0},
                seeds=(1,),
                extractor=prison_policy_metrics,
                first=duplicate,
                second=duplicate,
            )

        with self.assertRaises(ValueError):
            LocalFactor(
                source=FactorSource.SCENARIO,
                name="signal_accuracy",
                step=0.0,
            )

        with self.assertRaises(ValueError):
            local_factor_interaction_report(
                runner=self.runner,
                model=self.model,
                scenario=self.scenario,
                parameters={"beta": 2.0},
                seeds=(1,),
                extractor=prison_policy_metrics,
                first=LocalFactor(FactorSource.PARAMETER, "beta", 0.25),
                second=LocalFactor(
                    FactorSource.SCENARIO,
                    "missing_field",
                    0.05,
                ),
            )


if __name__ == "__main__":
    unittest.main()

from dataclasses import replace
import unittest

from narrative_dynamics.abm.calibration import calibrate_dynamic_role_model
from narrative_dynamics.abm.calibration_contracts import ABMCalibrationWeights
from narrative_dynamics.abm.experiments import (
    ABMExperimentArm,
    ABMExperimentProtocol,
    CalibratedABMExperimentReport,
    ExperimentObjective,
    run_calibrated_abm_experiment,
)
from tests.test_network_abm_calibration import TRUE, WRONG, dataset


def calibrated_inputs():
    model, observations = dataset()
    calibration = calibrate_dynamic_role_model(
        model,
        observations,
        candidates=(WRONG, TRUE),
        weights=ABMCalibrationWeights.uniform(),
    )
    return model, observations, calibration


def protocol(objective=ExperimentObjective.MAXIMIZE, *, reverse=False):
    arms = (
        ABMExperimentArm("baseline", TRUE),
        ABMExperimentArm("treatment", WRONG),
    )
    if reverse:
        arms = tuple(reversed(arms))
    return ABMExperimentProtocol(
        "holdout-comparison",
        "1",
        "baseline",
        "mean_trust",
        objective,
        arms,
    )


class CalibratedABMExperimentTests(unittest.TestCase):
    def test_runs_complete_paired_holdout_and_computes_baseline_delta(self):
        model, observations, calibration = calibrated_inputs()

        report = run_calibrated_abm_experiment(
            model,
            observations,
            calibration,
            protocol(),
        )

        self.assertIsInstance(report, CalibratedABMExperimentReport)
        results = {item.arm.arm_id: item for item in report.arm_results}
        self.assertEqual(tuple(item.case_id for item in results["baseline"].case_results), ("holdout",))
        self.assertEqual(results["baseline"].delta_from_baseline, 0.0)
        self.assertEqual(
            results["treatment"].delta_from_baseline,
            results["treatment"].mean_outcome - results["baseline"].mean_outcome,
        )
        self.assertTrue(results["baseline"].case_results[0].final_state_hash.startswith("sha256:"))

    def test_objective_controls_ranking_without_changing_outcomes(self):
        model, observations, calibration = calibrated_inputs()

        maximize = run_calibrated_abm_experiment(
            model, observations, calibration, protocol(ExperimentObjective.MAXIMIZE)
        )
        minimize = run_calibrated_abm_experiment(
            model, observations, calibration, protocol(ExperimentObjective.MINIMIZE)
        )

        self.assertEqual(maximize.ranking, ("baseline", "treatment"))
        self.assertEqual(minimize.ranking, ("treatment", "baseline"))
        self.assertEqual(maximize.arm_results, minimize.arm_results)

    def test_arm_input_order_is_canonical_and_hash_stable(self):
        model, observations, calibration = calibrated_inputs()
        first = run_calibrated_abm_experiment(
            model, observations, calibration, protocol()
        )
        second = run_calibrated_abm_experiment(
            model, observations, calibration, protocol(reverse=True)
        )
        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, second.content_hash)

    def test_protocol_rejects_invalid_metric_missing_baseline_and_duplicates(self):
        with self.assertRaisesRegex(ValueError, "primary metric"):
            replace(protocol(), primary_metric="unknown")
        with self.assertRaisesRegex(ValueError, "baseline arm"):
            replace(protocol(), baseline_arm_id="missing")
        with self.assertRaisesRegex(ValueError, "candidate values must be unique"):
            replace(
                protocol(),
                arms=(
                    ABMExperimentArm("baseline", TRUE),
                    ABMExperimentArm("duplicate", TRUE),
                ),
            )

    def test_baseline_must_equal_selected_calibration_candidate(self):
        model, observations, calibration = calibrated_inputs()
        wrong_baseline = ABMExperimentProtocol(
            "bad",
            "1",
            "baseline",
            "mean_trust",
            ExperimentObjective.MAXIMIZE,
            (
                ABMExperimentArm("baseline", WRONG),
                ABMExperimentArm("treatment", TRUE),
            ),
        )
        with self.assertRaisesRegex(ValueError, "selected calibration candidate"):
            run_calibrated_abm_experiment(
                model, observations, calibration, wrong_baseline
            )

    def test_report_rejects_tampered_delta_and_external_identity(self):
        model, observations, calibration = calibrated_inputs()
        report = run_calibrated_abm_experiment(
            model, observations, calibration, protocol()
        )
        changed = tuple(
            replace(item, delta_from_baseline=0.0)
            if item.arm.arm_id == "treatment"
            else item
            for item in report.arm_results
        )
        with self.assertRaisesRegex(ValueError, "baseline deltas"):
            replace(report, arm_results=changed)
        with self.assertRaisesRegex(ValueError, "base model"):
            run_calibrated_abm_experiment(
                replace(model, model_id="different"),
                observations,
                calibration,
                protocol(),
            )


if __name__ == "__main__":
    unittest.main()

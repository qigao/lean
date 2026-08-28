from __future__ import annotations

import unittest
from unittest.mock import patch

from narrative_dynamics.simulation import SimulationRunner

from tests.external_validation_fixtures import (
    ProbabilityModel,
    brier_loss,
    build_constraint_plan,
    external_dataset,
    final_targets,
    log_loss,
    policy_metrics,
    selection_targets,
)

try:
    from narrative_dynamics.external_validation import (
        ExternalConstraintStatus,
        ExternalValidationError,
        evaluate_external_constraint,
    )
except ImportError as error:
    ExternalConstraintStatus = None
    ExternalValidationError = None
    evaluate_external_constraint = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


class ExternalValidationConstraintTests(unittest.TestCase):
    def _evaluate(self, *, plan=None, targets=None, model=None):
        if evaluate_external_constraint is None:
            self.fail(f"external constraint API is missing: {IMPORT_ERROR}")
        data = external_dataset()
        return evaluate_external_constraint(
            plan=build_constraint_plan() if plan is None else plan,
            runner=SimulationRunner(),
            model=ProbabilityModel() if model is None else model,
            selection_targets=selection_targets(data) if targets is None else targets,
            extractor=policy_metrics,
            brier_loss=brier_loss(),
            log_loss=log_loss(),
        )

    def test_constraint_evaluation_requires_selection_validation_targets(self):
        if evaluate_external_constraint is None:
            self.fail(f"external constraint API is missing: {IMPORT_ERROR}")
        with self.assertRaises(ExternalValidationError):
            self._evaluate(targets=final_targets(external_dataset()))

    def test_brier_and_log_compatible_sets_are_intersected(self):
        finding = self._evaluate(
            plan=build_constraint_plan(parameter_grid={"p": (0.2, 0.8)})
        )
        self.assertEqual(finding.compatible_parameters, ((("p", 0.8),),))

    def test_full_candidate_loss_table_and_parent_calibration_hashes_are_retained(self):
        finding = self._evaluate(
            plan=build_constraint_plan(parameter_grid={"p": (0.2, 0.8)})
        )
        self.assertEqual(len(finding.candidate_losses), 2)
        self.assertTrue(
            all(item.brier_loss >= 0.0 and item.log_loss >= 0.0 for item in finding.candidate_losses)
        )
        self.assertTrue(finding.parent_manifest_hashes)
        self.assertTrue(
            all(value.startswith("sha256:") for value in finding.parent_manifest_hashes)
        )

    def test_singleton_compatible_set_is_constrained_not_identified(self):
        finding = self._evaluate(
            plan=build_constraint_plan(parameter_grid={"p": (0.2, 0.8)})
        )
        self.assertEqual(
            finding.status,
            ExternalConstraintStatus.CONSTRAINED,
        )
        self.assertFalse(hasattr(finding, "identification_status"))

    def test_multi_value_coordinate_is_not_constrained(self):
        finding = self._evaluate(
            plan=build_constraint_plan(
                parameter_grid={"p": (0.2, 0.8)},
                brier_delta=100.0,
                log_delta=100.0,
            )
        )
        self.assertEqual(
            finding.status,
            ExternalConstraintStatus.NOT_CONSTRAINED,
        )
        self.assertEqual(len(finding.compatible_parameters), 2)

    def test_empty_compatible_intersection_is_typed_failure(self):
        plan = build_constraint_plan(
            parameter_grid={"p": (0.6, 0.95)},
            brier_delta=0.0,
            log_delta=0.0,
        )
        with self.assertRaises(ExternalValidationError):
            self._evaluate(plan=plan)

    def test_final_test_targets_fail_before_any_model_execution(self):
        if evaluate_external_constraint is None:
            self.fail(f"external constraint API is missing: {IMPORT_ERROR}")
        model = ProbabilityModel()
        with self.assertRaises(ExternalValidationError):
            self._evaluate(targets=final_targets(external_dataset()), model=model)
        self.assertEqual(model.calls, 0)

    def test_constraint_path_has_no_generator_truth_or_p2_identifiability_dependency(self):
        with patch(
            "narrative_dynamics.uncertainty.diagnose_identifiability",
            side_effect=AssertionError("P2 identifiability must not run"),
        ):
            finding = self._evaluate()
        self.assertFalse(hasattr(finding, "true_parameters"))
        self.assertFalse(hasattr(finding, "generator_truth"))


if __name__ == "__main__":
    unittest.main()

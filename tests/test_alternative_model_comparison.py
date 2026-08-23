from __future__ import annotations

import unittest

import narrative_dynamics
from narrative_dynamics.contracts import ExperimentStage
from narrative_dynamics.losses import CategoricalBrierLoss, CategoricalMetricGroup
from narrative_dynamics.observations import (
    AdequacyThresholds,
    ComparisonPreregistration,
    FrozenModelSpec,
    ObservationPartitionRole,
    compare_registered_models,
)
from narrative_dynamics.report_artifact import attest_report
from narrative_dynamics.simulation import SimulationRunner
from tests.observational_data_fixtures import (
    ConservativePolicyModel,
    FlexiblePolicyModel,
    ReplacementPolicyModel,
    choice_metrics,
)
from tests.test_comparison_preregistration import selection_report, target_report


class AlternativeModelComparisonTests(unittest.TestCase):
    def _protocol(self):
        flexible = FlexiblePolicyModel()
        conservative = ConservativePolicyModel()
        flexible_spec = FrozenModelSpec.from_selection(
            "flexible",
            flexible,
            selection_report(flexible, ((("p", 0.5),), (("p", 0.8),))),
        )
        conservative_spec = FrozenModelSpec.from_selection(
            "conservative",
            conservative,
            selection_report(conservative, ((("q", 0.0),), (("q", 1.0),))),
        )
        targets = target_report(ObservationPartitionRole.FINAL_TEST)
        loss = CategoricalBrierLoss(targets.plan.groups)
        registration = ComparisonPreregistration.create(
            name="fair-final-comparison",
            version="1",
            target_report=targets,
            extractor=choice_metrics,
            loss=loss,
            thresholds=AdequacyThresholds(0.25, 0.25),
            models=(conservative_spec, flexible_spec),
        )
        return flexible, conservative, targets, loss, registration

    def test_registered_models_share_one_final_protocol_and_are_ranked_fairly(self):
        flexible, conservative, targets, loss, registration = self._protocol()
        report = compare_registered_models(
            runner=SimulationRunner(),
            registration=registration,
            target_report=targets,
            models={"flexible": flexible, "conservative": conservative},
            extractor=choice_metrics,
            loss=loss,
        )

        self.assertEqual(report.manifest.stage, ExperimentStage.ALTERNATIVE_MODEL_COMPARISON)
        self.assertEqual(report.registration_hash, registration.content_hash)
        self.assertEqual(report.best.name, "flexible")
        self.assertLess(report.best.mean_loss, report.ranking[1].mean_loss)
        self.assertEqual(
            {evaluation.final_test.suite_name for evaluation in report.ranking},
            {targets.as_held_out_suite().name},
        )
        self.assertEqual(
            tuple(evaluation.name for evaluation in report.ranking),
            ("flexible", "conservative"),
        )
        self.assertIs(attest_report(report).require_integrity(), report)

    def test_every_registration_mismatch_fails_before_any_model_execution(self):
        flexible, conservative, targets, loss, registration = self._protocol()
        replacement = ReplacementPolicyModel()

        bad_calls = (
            lambda: compare_registered_models(
                runner=SimulationRunner(),
                registration=registration,
                target_report=target_report(ObservationPartitionRole.SELECTION_VALIDATION),
                models={"flexible": flexible, "conservative": conservative},
                extractor=choice_metrics,
                loss=loss,
            ),
            lambda: compare_registered_models(
                runner=SimulationRunner(),
                registration=registration,
                target_report=targets,
                models={"flexible": flexible},
                extractor=choice_metrics,
                loss=loss,
            ),
            lambda: compare_registered_models(
                runner=SimulationRunner(),
                registration=registration,
                target_report=targets,
                models={"flexible": replacement, "conservative": conservative},
                extractor=choice_metrics,
                loss=loss,
            ),
            lambda: compare_registered_models(
                runner=SimulationRunner(),
                registration=registration,
                target_report=targets,
                models={"flexible": flexible, "conservative": conservative},
                extractor=lambda trace: choice_metrics(trace),
                loss=loss,
            ),
            lambda: compare_registered_models(
                runner=SimulationRunner(),
                registration=registration,
                target_report=targets,
                models={"flexible": flexible, "conservative": conservative},
                extractor=choice_metrics,
                loss=CategoricalBrierLoss(
                    (CategoricalMetricGroup("renamed", ("choice.a", "choice.b")),)
                ),
            ),
        )
        for call in bad_calls:
            with self.subTest(call=call):
                before = flexible.calls + conservative.calls + replacement.calls
                with self.assertRaises((TypeError, ValueError)):
                    call()
                self.assertEqual(
                    flexible.calls + conservative.calls + replacement.calls,
                    before,
                )

    def test_public_observational_protocol_is_exported_from_package_root(self):
        expected = (
            "ObservationDataset",
            "ObservationPartitionRole",
            "CategoricalTargetPlan",
            "FrozenModelSpec",
            "AdequacyThresholds",
            "ComparisonPreregistration",
            "compare_registered_models",
        )
        self.assertEqual(
            tuple(name for name in expected if not hasattr(narrative_dynamics, name)),
            (),
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from narrative_dynamics.losses import CategoricalBrierLoss, CategoricalMetricGroup
from narrative_dynamics.observations import (
    AdequacyThresholds,
    CategoricalTargetPlan,
    ComparisonPreregistration,
    FrozenModelSpec,
    ObservationPartitionRole,
    construct_categorical_targets,
)
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.uncertainty import ParameterAcceptanceSet
from narrative_dynamics.validation import select_on_validation_suite
from tests.observational_data_fixtures import (
    ConservativePolicyModel,
    FlexiblePolicyModel,
    choice_metrics,
)
from tests.test_observation_dataset import make_dataset


def target_report(role: ObservationPartitionRole):
    case_name = {
        ObservationPartitionRole.TRAIN: "train-case",
        ObservationPartitionRole.SELECTION_VALIDATION: "selection-case",
        ObservationPartitionRole.FINAL_TEST: "final-case",
    }[role]
    return construct_categorical_targets(
        make_dataset(),
        CategoricalTargetPlan(
            name=f"{role.value}-targets",
            version="1",
            role=role,
            groups=(CategoricalMetricGroup("choice", ("choice.a", "choice.b")),),
            seeds_by_case={case_name: (31, 32)},
        ),
    )


def selection_report(model, candidates):
    report = target_report(ObservationPartitionRole.SELECTION_VALIDATION)
    loss = CategoricalBrierLoss(report.plan.groups)
    return select_on_validation_suite(
        runner=SimulationRunner(),
        model=model,
        accepted_parameters=ParameterAcceptanceSet.from_parameters(candidates),
        suite=report.as_held_out_suite(),
        extractor=choice_metrics,
        loss=loss,
    )


class ComparisonPreregistrationTests(unittest.TestCase):
    def test_selected_models_thresholds_and_final_protocol_are_frozen(self):
        flexible = FlexiblePolicyModel()
        conservative = ConservativePolicyModel()
        flexible_selection = selection_report(
            flexible,
            ((("p", 0.5),), (("p", 0.8),)),
        )
        conservative_selection = selection_report(
            conservative,
            ((("q", 0.0),), (("q", 1.0),)),
        )
        flexible_spec = FrozenModelSpec.from_selection(
            "flexible",
            flexible,
            flexible_selection,
        )
        conservative_spec = FrozenModelSpec.from_selection(
            "conservative",
            conservative,
            conservative_selection,
        )
        thresholds = AdequacyThresholds(max_mean_loss=0.25, max_worst_loss=0.25)
        final_targets = target_report(ObservationPartitionRole.FINAL_TEST)
        loss = CategoricalBrierLoss(final_targets.plan.groups)
        registration = ComparisonPreregistration.create(
            name="prison-baseline-comparison",
            version="1",
            target_report=final_targets,
            extractor=choice_metrics,
            loss=loss,
            thresholds=thresholds,
            models=(flexible_spec, conservative_spec),
        )

        self.assertEqual(flexible_spec.parameters, (("p", 0.8),))
        self.assertEqual(
            flexible_spec.selection_manifest_hash,
            flexible_selection.manifest.content_hash,
        )
        self.assertEqual(
            tuple(model.name for model in registration.models),
            ("conservative", "flexible"),
        )
        self.assertEqual(registration.dataset_hash, final_targets.dataset_hash)
        self.assertEqual(registration.final_partition_hash, final_targets.partition_hash)
        self.assertEqual(
            registration.target_construction_manifest_hash,
            final_targets.manifest.content_hash,
        )
        self.assertEqual(
            registration.ranking_rule,
            ("mean_loss", "worst_loss", "model_name"),
        )
        self.assertTrue(registration.content_hash.startswith("sha256:"))

    def test_invalid_selection_thresholds_duplicates_and_non_final_targets_fail_closed(self):
        flexible = FlexiblePolicyModel()
        report = selection_report(flexible, ((("p", 0.8),),))
        spec = FrozenModelSpec.from_selection("flexible", flexible, report)
        with self.assertRaises(ValueError):
            AdequacyThresholds(max_mean_loss=-1.0, max_worst_loss=0.0)
        with self.assertRaises((TypeError, ValueError)):
            FrozenModelSpec.from_selection(
                "not-selection",
                flexible,
                target_report(ObservationPartitionRole.FINAL_TEST),
            )
        with self.assertRaises(ValueError):
            ComparisonPreregistration.create(
                name="not-final",
                version="1",
                target_report=target_report(ObservationPartitionRole.SELECTION_VALIDATION),
                extractor=choice_metrics,
                loss=CategoricalBrierLoss(
                    (CategoricalMetricGroup("choice", ("choice.a", "choice.b")),)
                ),
                thresholds=AdequacyThresholds(1.0, 1.0),
                models=(spec,),
            )
        with self.assertRaises(ValueError):
            ComparisonPreregistration.create(
                name="duplicate",
                version="1",
                target_report=target_report(ObservationPartitionRole.FINAL_TEST),
                extractor=choice_metrics,
                loss=CategoricalBrierLoss(
                    (CategoricalMetricGroup("choice", ("choice.a", "choice.b")),)
                ),
                thresholds=AdequacyThresholds(1.0, 1.0),
                models=(spec, spec),
            )


if __name__ == "__main__":
    unittest.main()

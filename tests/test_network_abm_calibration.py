from dataclasses import replace
import unittest

from narrative_dynamics.abm.autonomy_contracts import TruthObservation
from narrative_dynamics.abm.calibration import (
    EmpiricalABMCalibrationReport,
    apply_calibration_candidate,
    calibrate_dynamic_role_model,
    observe_dynamic_role_state,
)
from narrative_dynamics.abm.calibration_contracts import (
    ABMCalibrationCandidate,
    ABMCalibrationWeights,
    CalibrationSplit,
    EmpiricalABMCase,
    EmpiricalABMDataset,
)
from narrative_dynamics.abm.interventions import EdgeSelector
from narrative_dynamics.abm.lifecycle_contracts import (
    LifecycleEventKind,
    PopulationLifecycleEvent,
)
from narrative_dynamics.abm.role_contracts import initialize_dynamic_role_population
from narrative_dynamics.abm.roles import simulate_dynamic_role_population
from tests.dynamic_role_fixtures import dynamic_role_model


TRUE = ABMCalibrationCandidate(0.5, 0.2, 0.8)
WRONG = ABMCalibrationCandidate(0.2, 0.6, 0.7)


def schedules():
    return (
        ((PopulationLifecycleEvent("b", LifecycleEventKind.ENTER),),),
        ((TruthObservation(EdgeSelector("a", "b", "peer"), 1.0),),),
    )


def generated_case(
    base_model,
    candidate,
    case_id,
    split,
    *,
    beliefs=(("a", 1.0), ("c", 0.0)),
):
    environment_schedule, truth_schedule = schedules()
    model = apply_calibration_candidate(base_model, candidate)
    initial = initialize_dynamic_role_population(model, beliefs=dict(beliefs))
    trajectory = simulate_dynamic_role_population(
        model,
        initial,
        environment_event_schedule=environment_schedule,
        truth_observation_schedule=truth_schedule,
    )
    observations = tuple(
        observe_dynamic_role_state(model, item.next_state)
        for item in trajectory.rounds
    )
    return EmpiricalABMCase(
        case_id,
        split,
        tuple(beliefs),
        environment_schedule,
        truth_schedule,
        observations,
    )


def dataset(*, train_candidate=TRUE, holdout_candidate=TRUE, reverse=False):
    base = dynamic_role_model()
    cases = (
        generated_case(
            base,
            train_candidate,
            "train",
            CalibrationSplit.TRAIN,
        ),
        generated_case(
            base,
            holdout_candidate,
            "holdout",
            CalibrationSplit.HOLDOUT,
            beliefs=(("a", 0.8), ("c", 0.2)),
        ),
    )
    if reverse:
        cases = tuple(reversed(cases))
    return base, EmpiricalABMDataset("synthetic", "1", cases)


class EmpiricalABMCalibrationTests(unittest.TestCase):
    def test_synthetic_grid_recovers_true_candidate_with_complete_case_audit(self):
        model, observations = dataset()

        report = calibrate_dynamic_role_model(
            model,
            observations,
            candidates=(WRONG, TRUE),
            weights=ABMCalibrationWeights.uniform(),
        )

        self.assertIsInstance(report, EmpiricalABMCalibrationReport)
        self.assertEqual(report.selected_candidate, TRUE)
        self.assertEqual(report.ranking[0], TRUE)
        true_fit = next(item for item in report.candidate_fits if item.candidate == TRUE)
        self.assertEqual(true_fit.training_loss, 0.0)
        self.assertEqual(true_fit.holdout_loss, 0.0)
        self.assertEqual(len(true_fit.case_fits), 2)
        self.assertTrue(
            all(
                item.observed_snapshot_hashes == item.predicted_snapshot_hashes
                for item in true_fit.case_fits
            )
        )

    def test_holdout_loss_never_changes_training_selection(self):
        model, observations = dataset(holdout_candidate=WRONG)

        report = calibrate_dynamic_role_model(
            model,
            observations,
            candidates=(TRUE, WRONG),
            weights=ABMCalibrationWeights.uniform(),
        )

        fits = {item.candidate: item for item in report.candidate_fits}
        self.assertEqual(report.selected_candidate, TRUE)
        self.assertEqual(fits[TRUE].training_loss, 0.0)
        self.assertEqual(fits[WRONG].holdout_loss, 0.0)

    def test_training_ties_break_by_canonical_candidate_tuple(self):
        model, observations = dataset()
        weights = ABMCalibrationWeights(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )

        report = calibrate_dynamic_role_model(
            model,
            observations,
            candidates=(TRUE, WRONG),
            weights=weights,
        )

        self.assertEqual(report.selected_candidate, WRONG)
        self.assertEqual(report.ranking, (WRONG, TRUE))

    def test_candidate_and_case_input_order_do_not_change_report(self):
        model, forward_data = dataset()
        _, reverse_data = dataset(reverse=True)

        first = calibrate_dynamic_role_model(
            model,
            forward_data,
            candidates=(TRUE, WRONG),
            weights=ABMCalibrationWeights.uniform(),
        )
        second = calibrate_dynamic_role_model(
            model,
            reverse_data,
            candidates=(WRONG, TRUE),
            weights=ABMCalibrationWeights.uniform(),
        )

        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, second.content_hash)

    def test_candidate_reconstruction_changes_only_declared_parameters(self):
        base = dynamic_role_model()
        original_hash = base.content_hash

        calibrated = apply_calibration_candidate(base, WRONG)

        evolving = calibrated.autonomy_model.evolving_model
        self.assertEqual(evolving.learning_rate, WRONG.learning_rate)
        self.assertEqual(
            evolving.dissolution_similarity,
            WRONG.dissolution_similarity,
        )
        self.assertEqual(evolving.formation_similarity, WRONG.formation_similarity)
        self.assertEqual(evolving.base_model, base.autonomy_model.evolving_model.base_model)
        self.assertEqual(
            calibrated.autonomy_model.role_policies,
            base.autonomy_model.role_policies,
        )
        self.assertEqual(calibrated.transition_rules, base.transition_rules)
        self.assertEqual(base.content_hash, original_hash)

    def test_incompatible_frozen_truth_schedule_names_candidate_and_case(self):
        base, observations = dataset()
        train = next(
            item for item in observations.cases if item.split is CalibrationSplit.TRAIN
        )
        bad_train = replace(train, truth_observation_schedule=((),))
        holdout = next(
            item for item in observations.cases if item.split is CalibrationSplit.HOLDOUT
        )
        bad_data = EmpiricalABMDataset("bad", "1", (bad_train, holdout))

        with self.assertRaisesRegex(
            ValueError,
            "candidate .* cannot replay case 'train'",
        ):
            calibrate_dynamic_role_model(
                base,
                bad_data,
                candidates=(TRUE,),
                weights=ABMCalibrationWeights.uniform(),
            )

    def test_report_rejects_selected_candidate_not_first_in_training_ranking(self):
        model, observations = dataset()
        report = calibrate_dynamic_role_model(
            model,
            observations,
            candidates=(TRUE, WRONG),
            weights=ABMCalibrationWeights.uniform(),
        )
        with self.assertRaisesRegex(ValueError, "selected candidate"):
            replace(report, selected_candidate=WRONG)

    def test_zero_active_projection_is_finite(self):
        model = apply_calibration_candidate(dynamic_role_model(), TRUE)
        initial = initialize_dynamic_role_population(model)
        environment_schedule = ((
            PopulationLifecycleEvent("a", LifecycleEventKind.DEATH),
            PopulationLifecycleEvent("c", LifecycleEventKind.DEATH),
        ),)
        trajectory = simulate_dynamic_role_population(
            model,
            initial,
            environment_event_schedule=environment_schedule,
            truth_observation_schedule=((),),
        )

        observed = observe_dynamic_role_state(model, trajectory.final_state)

        self.assertEqual(observed.active_share, 0.0)
        self.assertEqual(observed.mean_active_belief, 0.0)
        self.assertEqual(observed.active_sharing_rate, 0.0)
        self.assertEqual(observed.role_entropy, 0.0)


if __name__ == "__main__":
    unittest.main()

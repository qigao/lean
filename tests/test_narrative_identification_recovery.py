from __future__ import annotations

import math
import unittest

from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics
from narrative_dynamics.contracts import ModelRun
from narrative_dynamics.simulation import SimulationRunner

from tests.test_narrative_identification_protocol import (
    BETA_GRID,
    GENERATION_SEEDS,
    SCALE_GRID,
    TRUTH,
    brier_loss,
    case_by_name,
    make_recovery_experiment,
    require_identification,
)

_IDENTIFICATION_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.identification import (
        IdentificationRecoveryError,
        IdentificationStatus,
        ParameterCandidateLoss,
        SyntheticIdentificationError,
        certify_observational_equivalence,
        evaluate_parameter_recovery_experiment,
        interpret_parameter_identification,
    )
except ImportError as error:
    _IDENTIFICATION_IMPORT_ERROR = error
    IdentificationRecoveryError = None
    IdentificationStatus = None
    ParameterCandidateLoss = None
    SyntheticIdentificationError = None
    certify_observational_equivalence = None
    evaluate_parameter_recovery_experiment = None
    interpret_parameter_identification = None

_ADAPTER_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.adapters.narrative_prison_identification import (
        create_narrative_identification_intentional_source,
    )
except ImportError as error:
    _ADAPTER_IMPORT_ERROR = error
    create_narrative_identification_intentional_source = None


METRIC_KEYS = ("initial.scout", "initial.escape", "initial.submit")
TRUTH_TUPLE = tuple(sorted((key, float(value)) for key, value in TRUTH.items()))


def require_adapter(test: unittest.TestCase) -> None:
    if _ADAPTER_IMPORT_ERROR is not None:
        test.fail(f"synthetic identification adapter is missing: {_ADAPTER_IMPORT_ERROR}")


def intentional_source():
    if create_narrative_identification_intentional_source is None:
        raise RuntimeError("identification adapter unavailable")
    return create_narrative_identification_intentional_source()


def evaluate(experiment):
    return evaluate_parameter_recovery_experiment(
        runner=SimulationRunner(),
        model=intentional_source(),
        experiment=experiment,
        extractor=prison_initial_action_metrics,
        loss=brier_loss(),
    )


def joint_experiment(*, expected_status=None, cases=None):
    source = intentional_source()
    if expected_status is None:
        expected_status = IdentificationStatus.IDENTIFIED_UNDER_PROTOCOL
    return make_recovery_experiment(
        name="joint-beta",
        cases=cases,
        parameter_grid={
            "beta_goal": BETA_GRID,
            "beta_action": BETA_GRID,
            "goal_pressure_scale": (1.0,),
            "instrumentality_scale": (1.0,),
        },
        true_parameters=TRUTH,
        target_coordinates=("beta_goal", "beta_action"),
        expected_status=expected_status,
        model_source=source,
    )


def scale_experiment(*, name, pressure_values, instrumentality_values, target_coordinates, expected_status):
    return make_recovery_experiment(
        name=name,
        cases=(
            case_by_name("train-goal-low").scenario,
            case_by_name("selection-goal-high").scenario,
        ),
        parameter_grid={
            "beta_goal": (TRUTH["beta_goal"],),
            "beta_action": (TRUTH["beta_action"],),
            "goal_pressure_scale": tuple(pressure_values),
            "instrumentality_scale": tuple(instrumentality_values),
        },
        true_parameters=TRUTH,
        target_coordinates=tuple(target_coordinates),
        expected_status=expected_status,
        model_source=intentional_source(),
    )


class BadPolicyModel:
    name = "bad-policy-model"
    version = "1"

    def simulate(self, scenario, parameters, rng):
        return ModelRun(
            events=(),
            outcome={
                "initial_policy": {
                    "scout": math.nan,
                    "escape": 0.5,
                    "submit": 0.5,
                }
            },
        )


class NarrativeIdentificationRecoveryTests(unittest.TestCase):
    def setUp(self):
        require_identification(self)
        require_adapter(self)

    def test_joint_beta_goal_beta_action_recovery_is_exact(self):
        report = evaluate(joint_experiment())
        self.assertEqual(report.status, IdentificationStatus.IDENTIFIED_UNDER_PROTOCOL)
        self.assertEqual(report.accepted_parameters.parameters, (TRUTH_TUPLE,))
        by_name = {item.name: item for item in report.identifiability.parameters}
        self.assertTrue(by_name["beta_goal"].identified)
        self.assertTrue(by_name["beta_action"].identified)

    def test_action_temperature_anchor_is_invariant_to_beta_goal(self):
        runner = SimulationRunner()
        scenario = case_by_name("train-temp-anchor").scenario
        source = intentional_source()
        low = dict(TRUTH, beta_goal=0.5)
        high = dict(TRUTH, beta_goal=4.0)
        left = runner.run_once(source, scenario, low, seed=11)
        right = runner.run_once(source, scenario, high, seed=11)
        self.assertEqual(left.outcome["initial_policy"], right.outcome["initial_policy"])

    def test_low_and_high_goal_gaps_change_with_beta_goal(self):
        runner = SimulationRunner()
        source = intentional_source()
        for name in ("train-goal-low", "selection-goal-high"):
            with self.subTest(case=name):
                scenario = case_by_name(name).scenario
                low = runner.run_once(source, scenario, dict(TRUTH, beta_goal=0.5), seed=11)
                high = runner.run_once(source, scenario, dict(TRUTH, beta_goal=4.0), seed=11)
                self.assertNotEqual(low.outcome["initial_policy"], high.outcome["initial_policy"])

    def test_equivalence_fixture_retains_multiple_latent_candidates_and_reports_nonidentification(self):
        experiment = joint_experiment(
            expected_status=IdentificationStatus.NOT_IDENTIFIED_UNDER_PROTOCOL,
            cases=(case_by_name("final-equivalence").scenario,),
        )
        report = evaluate(experiment)
        self.assertEqual(report.status, IdentificationStatus.NOT_IDENTIFIED_UNDER_PROTOCOL)
        self.assertTrue(report.truth_retained)
        self.assertEqual(report.accepted_parameters.count, 16)

    def test_pressure_recovery_with_fixed_instrumentality(self):
        experiment = scale_experiment(
            name="pressure",
            pressure_values=SCALE_GRID,
            instrumentality_values=(1.0,),
            target_coordinates=("goal_pressure_scale",),
            expected_status=IdentificationStatus.IDENTIFIED_UNDER_PROTOCOL,
        )
        report = evaluate(experiment)
        self.assertEqual(report.status, IdentificationStatus.IDENTIFIED_UNDER_PROTOCOL)
        self.assertEqual(
            {dict(item)["goal_pressure_scale"] for item in report.accepted_parameters.parameters},
            {1.0},
        )

    def test_instrumentality_recovery_with_fixed_pressure(self):
        experiment = scale_experiment(
            name="instrumentality",
            pressure_values=(1.0,),
            instrumentality_values=SCALE_GRID,
            target_coordinates=("instrumentality_scale",),
            expected_status=IdentificationStatus.IDENTIFIED_UNDER_PROTOCOL,
        )
        report = evaluate(experiment)
        self.assertEqual(report.status, IdentificationStatus.IDENTIFIED_UNDER_PROTOCOL)
        self.assertEqual(
            {dict(item)["instrumentality_scale"] for item in report.accepted_parameters.parameters},
            {1.0},
        )

    def test_scale_confounded_full_grid_retains_exact_product_one_equivalence_class(self):
        experiment = scale_experiment(
            name="scale-confounded",
            pressure_values=SCALE_GRID,
            instrumentality_values=SCALE_GRID,
            target_coordinates=("goal_pressure_scale", "instrumentality_scale"),
            expected_status=IdentificationStatus.NOT_IDENTIFIED_UNDER_PROTOCOL,
        )
        report = evaluate(experiment)
        self.assertEqual(report.status, IdentificationStatus.NOT_IDENTIFIED_UNDER_PROTOCOL)
        accepted = {
            (
                dict(parameters)["goal_pressure_scale"],
                dict(parameters)["instrumentality_scale"],
            )
            for parameters in report.accepted_parameters.parameters
        }
        self.assertEqual(accepted, {(0.5, 2.0), (1.0, 1.0), (2.0, 0.5)})
        self.assertTrue(report.truth_retained)

    def test_truth_excluded_is_recovery_error_not_nonidentification(self):
        experiment = make_recovery_experiment(
            name="truth-excluded",
            cases=(case_by_name("final-equivalence").scenario,),
            parameter_grid={
                "beta_goal": (1.0, 2.0),
                "beta_action": (1.0,),
                "goal_pressure_scale": (1.0,),
                "instrumentality_scale": (1.0,),
            },
            true_parameters=TRUTH,
            target_coordinates=("beta_goal",),
            expected_status=None,
            model_source=intentional_source(),
        )
        rows = []
        for parameters in experiment.candidate_parameters:
            accepted = dict(parameters)["beta_goal"] == 1.0
            rows.append(
                ParameterCandidateLoss(
                    parameters=parameters,
                    block_losses=(0.0,),
                    mean_loss=0.0,
                    accepted_blocks=1 if accepted else 0,
                    acceptance_fraction=1.0 if accepted else 0.0,
                )
            )
        with self.assertRaises(IdentificationRecoveryError):
            interpret_parameter_identification(experiment, tuple(rows))

    def test_empty_accepted_set_is_recovery_error(self):
        experiment = make_recovery_experiment(
            name="empty",
            cases=(case_by_name("final-equivalence").scenario,),
            parameter_grid={
                "beta_goal": (2.0,),
                "beta_action": (1.0,),
                "goal_pressure_scale": (1.0,),
                "instrumentality_scale": (1.0,),
            },
            true_parameters=TRUTH,
            target_coordinates=("beta_goal",),
            expected_status=None,
            model_source=intentional_source(),
        )
        parameters = experiment.candidate_parameters[0]
        row = ParameterCandidateLoss(
            parameters=parameters,
            block_losses=(1.0,),
            mean_loss=1.0,
            accepted_blocks=0,
            acceptance_fraction=0.0,
        )
        with self.assertRaises(IdentificationRecoveryError):
            interpret_parameter_identification(experiment, (row,))

    def test_candidate_loss_table_retains_every_grid_candidate_and_ties(self):
        experiment = joint_experiment(
            expected_status=IdentificationStatus.NOT_IDENTIFIED_UNDER_PROTOCOL,
            cases=(case_by_name("final-equivalence").scenario,),
        )
        report = evaluate(experiment)
        self.assertEqual(len(report.candidate_losses), 16)
        self.assertEqual(
            {row.parameters for row in report.candidate_losses},
            set(experiment.candidate_parameters),
        )
        self.assertEqual(len({row.mean_loss for row in report.candidate_losses}), 1)

    def test_observational_equivalence_certifies_complete_policy_not_equal_loss(self):
        source = intentional_source()
        equivalent = certify_observational_equivalence(
            name="latent-equivalence",
            runner=SimulationRunner(),
            left_model=source,
            right_model=source,
            left_parameters=dict(TRUTH, beta_goal=0.5, beta_action=0.5),
            right_parameters=dict(TRUTH, beta_goal=4.0, beta_action=4.0),
            cases=(case_by_name("final-equivalence").scenario,),
            seeds=GENERATION_SEEDS,
            extractor=prison_initial_action_metrics,
            action_keys=METRIC_KEYS,
            tolerance=1e-12,
        )
        self.assertTrue(equivalent.equivalent)
        self.assertLessEqual(equivalent.max_abs_policy_delta, 1e-12)
        separated = certify_observational_equivalence(
            name="latent-separated",
            runner=SimulationRunner(),
            left_model=source,
            right_model=source,
            left_parameters=dict(TRUTH, beta_goal=0.5),
            right_parameters=dict(TRUTH, beta_goal=4.0),
            cases=(case_by_name("train-goal-low").scenario,),
            seeds=GENERATION_SEEDS,
            extractor=prison_initial_action_metrics,
            action_keys=METRIC_KEYS,
            tolerance=1e-12,
        )
        self.assertFalse(separated.equivalent)
        self.assertGreater(separated.max_abs_policy_delta, 1e-12)

    def test_nonfinite_candidate_loss_or_policy_coordinate_is_rejected(self):
        with self.assertRaises(SyntheticIdentificationError):
            ParameterCandidateLoss(
                parameters=(("x", 1.0),),
                block_losses=(math.nan,),
                mean_loss=math.nan,
                accepted_blocks=1,
                acceptance_fraction=1.0,
            )
        bad = BadPolicyModel()
        with self.assertRaises(SyntheticIdentificationError):
            certify_observational_equivalence(
                name="bad-policy",
                runner=SimulationRunner(),
                left_model=bad,
                right_model=bad,
                left_parameters={"x": 1.0},
                right_parameters={"x": 1.0},
                cases=(case_by_name("final-equivalence").scenario,),
                seeds=(1,),
                extractor=prison_initial_action_metrics,
                action_keys=METRIC_KEYS,
                tolerance=1e-12,
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib
import unittest

from narrative_dynamics.attestation import RepositoryIdentity, measure_implementation
from narrative_dynamics.calibration import calibrate_grid
from narrative_dynamics.contracts import Scenario
from narrative_dynamics.execution_policy import (
    TrustedExecutionPolicy,
    TrustedModelPin,
)
from narrative_dynamics.process_execution import ProcessLimits
from narrative_dynamics.registry import ModelKind, ModelRegistry
from narrative_dynamics.report_artifact import attest_report
from narrative_dynamics.schema_validation import ModelSchemaViolation
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.uncertainty import ParameterAcceptanceSet
from narrative_dynamics.validation import (
    EvaluationRole,
    HeldOutCase,
    HeldOutSuite,
    evaluate_on_final_test_suite,
    select_on_validation_suite,
)


def prison_api(test_case):
    try:
        module = importlib.import_module(
            "narrative_dynamics.adapters.prison_pomdp"
        )
    except ModuleNotFoundError as error:
        test_case.fail(f"finite prison POMDP adapter is missing: {error}")

    names = (
        "FinitePrisonPOMDPModel",
        "create_prison_pomdp_model",
        "prison_pomdp_contract",
        "prison_pomdp_source",
        "prison_policy_metrics",
    )
    missing = tuple(name for name in names if getattr(module, name, None) is None)
    test_case.assertEqual(missing, (), f"prison POMDP API is missing: {missing}")
    return module


def prison_scenario(
    scenario_id,
    *,
    prior_weak=0.5,
    signal_accuracy=0.85,
    guard_persistence=0.9,
    horizon=2,
):
    return Scenario(
        id=scenario_id,
        payload={
            "prior_weak": prior_weak,
            "signal_accuracy": signal_accuracy,
            "guard_persistence": guard_persistence,
            "escape_reward": 8.0,
            "capture_cost": 10.0,
            "submit_reward": 1.0,
            "scout_cost": 0.25,
            "discount": 0.95,
            "horizon": horizon,
        },
    )


def bound_source(api):
    source = api.prison_pomdp_source(
        limits=ProcessLimits(
            timeout_seconds=3.0,
            max_output_bytes=64 * 1024,
            max_trace_bytes=128 * 1024,
        )
    )
    contract = api.prison_pomdp_contract()
    policy = TrustedExecutionPolicy(
        name="prison-adapter-test-policy",
        version="1",
        pins=(
            TrustedModelPin(
                model_name=source.name,
                declared_contract_hash=contract.content_hash,
                expected_implementation_hash=(
                    measure_implementation(source).content_hash
                ),
            ),
        ),
    )
    return source, contract, policy, policy.bind(source, contract=contract)


class PrisonPOMDPBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner(
            repository_identity=RepositoryIdentity(provider="test")
        )

    def test_finite_horizon_policy_values_information_and_replays_seed(self):
        api = prison_api(self)
        model = api.create_prison_pomdp_model()
        scenario = prison_scenario("uncertain-prison")

        first = self.runner.run_once(model, scenario, {"beta": 2.0}, seed=17)
        replay = self.runner.run_once(model, scenario, {"beta": 2.0}, seed=17)

        self.assertEqual(first, replay)
        policy = first.outcome["initial_policy"]
        self.assertAlmostEqual(sum(policy.values()), 1.0)
        self.assertGreater(policy["scout"], policy["escape"])
        self.assertGreater(policy["scout"], policy["submit"])
        self.assertIn(first.outcome["terminal_action"], ("escape", "submit"))
        self.assertLessEqual(first.outcome["steps"], scenario.payload["horizon"])
        self.assertEqual(first.events[0].kind, "belief_state")
        self.assertEqual(first.events[-1].kind, "episode_ended")

        final_step = self.runner.run_once(
            model,
            prison_scenario("final-step", prior_weak=0.8, horizon=1),
            {"beta": 2.0},
            seed=17,
        )
        self.assertEqual(final_step.outcome["initial_policy"]["scout"], 0.0)
        self.assertGreater(
            final_step.outcome["initial_policy"]["escape"],
            final_step.outcome["initial_policy"]["submit"],
        )

    def test_production_source_is_policy_pinned_isolated_and_schema_attested(self):
        api = prison_api(self)
        source, contract, policy, bound = bound_source(api)

        descriptor = ModelRegistry().register(
            source,
            kind=ModelKind.POMDP,
            contract=contract,
        )
        self.assertIs(descriptor.kind, ModelKind.POMDP)
        self.assertFalse(descriptor.production_ready)
        self.assertIsNone(contract.expected_implementation_hash)
        self.assertFalse(contract.complete)
        self.assertTrue(bound.contract.complete)

        trace = self.runner.run_once(
            bound,
            prison_scenario("production-prison"),
            {"beta": 2.0},
            seed=23,
        )

        self.assertIsNotNone(trace.execution)
        self.assertTrue(trace.execution.isolated)
        model_identity = trace.manifest.inputs["model"]
        self.assertEqual(
            model_identity["trusted_execution_policy"]["content_hash"],
            policy.content_hash,
        )
        implementation = model_identity["implementation_attestation"]
        self.assertEqual(implementation["verification"], "matched")
        self.assertEqual(
            implementation["expected_content_hash"],
            measure_implementation(source).content_hash,
        )
        schema = trace.manifest.inputs["schema_validation"]
        for boundary in ("parameters", "scenario", "events", "outcome"):
            self.assertEqual(schema[boundary]["status"], "validated")
        self.assertIn("result_artifact", trace.manifest.inputs)

    def test_invalid_scenario_is_rejected_before_worker_launch(self):
        api = prison_api(self)
        _, _, _, bound = bound_source(api)

        with self.assertRaises(ModelSchemaViolation) as raised:
            self.runner.run_once(
                bound,
                prison_scenario("invalid-prior", prior_weak=1.5),
                {"beta": 2.0},
                seed=1,
            )

        self.assertEqual(raised.exception.boundary, "scenario")
        self.assertEqual(raised.exception.path, "$.prior_weak")
        self.assertIsNone(raised.exception.execution)


class PrisonPOMDPValidationTests(unittest.TestCase):
    def setUp(self):
        self.api = prison_api(self)
        _, _, _, self.model = bound_source(self.api)
        self.runner = SimulationRunner(
            repository_identity=RepositoryIdentity(provider="test")
        )

    def target(self, scenario, beta, seed):
        trace = self.runner.run_once(
            self.model,
            scenario,
            {"beta": beta},
            seed=seed,
        )
        return self.api.prison_policy_metrics(trace)

    def test_calibration_selection_final_test_and_report_attestation(self):
        calibration_scenario = prison_scenario("prison-calibration")
        true_target = self.target(calibration_scenario, 2.0, 100)
        calibration = calibrate_grid(
            runner=self.runner,
            model=self.model,
            scenario=calibration_scenario,
            parameter_grid={"beta": (0.5, 2.0, 5.0)},
            seeds=(101,),
            extractor=self.api.prison_policy_metrics,
            target=true_target,
        )
        self.assertEqual(calibration.best.parameters, (("beta", 2.0),))

        accepted = ParameterAcceptanceSet.from_parameters(
            (
                (("beta", 0.5),),
                (("beta", 2.0),),
                (("beta", 5.0),),
            ),
            source_manifest_hashes=(calibration.manifest.content_hash,),
        )
        selection_scenario = prison_scenario(
            "prison-selection",
            prior_weak=0.35,
        )
        selection_suite = HeldOutSuite(
            name="prison-selection-suite",
            role=EvaluationRole.SELECTION_VALIDATION,
            cases=(
                HeldOutCase(
                    scenario=selection_scenario,
                    seeds=(201,),
                    target=self.target(selection_scenario, 2.0, 200),
                ),
            ),
        )
        selection = select_on_validation_suite(
            runner=self.runner,
            model=self.model,
            accepted_parameters=accepted,
            suite=selection_suite,
            extractor=self.api.prison_policy_metrics,
        )
        self.assertEqual(selection.selected_parameters, (("beta", 2.0),))

        final_scenario = prison_scenario(
            "prison-final",
            prior_weak=0.8,
            signal_accuracy=0.75,
            horizon=1,
        )
        final_suite = HeldOutSuite(
            name="prison-final-suite",
            role=EvaluationRole.FINAL_TEST,
            cases=(
                HeldOutCase(
                    scenario=final_scenario,
                    seeds=(301,),
                    target=self.target(final_scenario, 2.0, 300),
                ),
            ),
        )
        final = evaluate_on_final_test_suite(
            runner=self.runner,
            model=self.model,
            parameters=dict(selection.selected_parameters),
            suite=final_suite,
            extractor=self.api.prison_policy_metrics,
        )

        self.assertEqual(final.validation.mean_loss, 0.0)
        attested = attest_report(final)
        self.assertIs(attested.require_integrity(), final)
        self.assertEqual(
            attested.artifact.manifest_hash,
            final.manifest.content_hash,
        )


if __name__ == "__main__":
    unittest.main()

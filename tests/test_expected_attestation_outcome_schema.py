from __future__ import annotations

import types
import unittest

import narrative_dynamics.attestation as attestation
from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import ModelRun, Scenario, TraceEvent
from narrative_dynamics.model_contract import ModelContract, ModelSchema
from narrative_dynamics.process_execution import ProcessLimits, SubprocessModel
from narrative_dynamics.registry import ModelKind, ModelRegistry
from narrative_dynamics.schema_validation import ModelSchemaViolation
from narrative_dynamics.simulation import ModelFactory, SimulationRunner


class ValidOutcomeModel:
    name = "valid-outcome"
    version = "1.0.0"
    implementation_revision = "test:valid-outcome-v1"

    def __init__(self, calls=None):
        self.calls = calls

    def simulate(self, scenario, parameters, rng):
        if self.calls is not None:
            self.calls["simulate"] += 1
        value = float(parameters["value"])
        return ModelRun(
            events=(TraceEvent(tick=0, kind="emitted", data={"value": value}),),
            outcome={"value": value},
        )


class InvalidOutcomeModel(ValidOutcomeModel):
    name = "invalid-outcome"
    implementation_revision = "test:invalid-outcome-v1"

    def simulate(self, scenario, parameters, rng):
        if self.calls is not None:
            self.calls["simulate"] += 1
        value = float(parameters["value"])
        return ModelRun(
            events=(TraceEvent(tick=0, kind="emitted", data={"value": value}),),
            outcome={"value": "not-a-number"},
        )


def _object_schema(name, properties, required=()):
    return ModelSchema(
        name=name,
        version="1.0.0",
        definition={
            "type": "object",
            "required": tuple(required),
            "properties": properties,
            "additional_properties": False,
        },
    )


def _event_schema():
    return ModelSchema(
        name="emitted-event",
        version="1.0.0",
        definition={
            "event_kinds": ("emitted",),
            "event_data": {
                "emitted": {
                    "type": "object",
                    "required": ("value",),
                    "properties": {"value": {"type": "number"}},
                    "additional_properties": False,
                }
            },
            "allow_unlisted_events": False,
        },
    )


def _parameter_schema():
    return _object_schema(
        "parameters",
        {"value": {"type": "number"}},
        required=("value",),
    )


def _scenario_schema():
    return _object_schema("scenario", {})


def _outcome_schema():
    return _object_schema(
        "outcome",
        {"value": {"type": "number"}},
        required=("value",),
    )


def _contract_for(source, *, expected_hash, outcome_schema=None):
    return ModelContract(
        version=source.version,
        implementation_revision=source.implementation_revision,
        expected_implementation_hash=expected_hash,
        parameter_schema=_parameter_schema(),
        scenario_schema=_scenario_schema(),
        event_schema=_event_schema(),
        outcome_schema=_outcome_schema() if outcome_schema is None else outcome_schema,
    )


def _registered_source(source, contract):
    registry = ModelRegistry()
    registry.register(source, kind=next(iter(ModelKind)), contract=contract)
    return registry, registry.execution_source(source.name)


def _mismatch_type(test_case):
    mismatch = getattr(attestation, "ImplementationAttestationMismatch", None)
    if not isinstance(mismatch, type) or not issubclass(mismatch, Exception):
        test_case.fail("ImplementationAttestationMismatch is missing")
    return mismatch


class ExpectedImplementationAttestationTests(unittest.TestCase):
    def test_contract_validates_expected_hash_and_requires_complete_result_boundary(self):
        with self.assertRaises(ValueError):
            ModelContract(expected_implementation_hash="not-a-hash")

        three_boundary_contract = ModelContract(
            version="1.0.0",
            implementation_revision="test:v1",
            parameter_schema=_parameter_schema(),
            scenario_schema=_scenario_schema(),
            event_schema=_event_schema(),
        )
        self.assertFalse(three_boundary_contract.complete)

        source = ValidOutcomeModel()
        expected = measure_implementation(source).content_hash
        complete = _contract_for(source, expected_hash=expected)
        self.assertTrue(complete.complete)
        self.assertEqual(complete.expected_implementation_hash, expected)
        self.assertTrue(complete.outcome_schema.enforceable)
        self.assertIn("outcome", complete.schema_identities)

    def test_matching_measurement_is_attested_before_execution(self):
        calls = {"simulate": 0}
        source = ValidOutcomeModel(calls)
        expected = measure_implementation(source).content_hash
        contract = _contract_for(source, expected_hash=expected)
        _, execution_source = _registered_source(source, contract)

        trace = SimulationRunner().run_once(
            execution_source,
            Scenario(id="valid", payload={}),
            {"value": 3.0},
            seed=7,
        )

        self.assertEqual(calls["simulate"], 1)
        measured = trace.manifest.inputs["model"]["implementation_attestation"]
        self.assertEqual(measured["content_hash"], expected)
        self.assertEqual(measured["expected_content_hash"], expected)
        self.assertEqual(measured["verification"], "matched")
        outcome_validation = trace.manifest.inputs["schema_validation"]["outcome"]
        self.assertEqual(
            outcome_validation["schema_hash"],
            contract.outcome_schema.content_hash,
        )

    def test_mismatch_stops_before_factory_construction(self):
        calls = {"create": 0}

        def create_model():
            calls["create"] += 1
            return ValidOutcomeModel()

        factory = ModelFactory(
            name="valid-outcome",
            create=create_model,
            version="1.0.0",
            implementation_revision="test:valid-outcome-v1",
        )
        contract = _contract_for(
            factory,
            expected_hash="sha256:" + "0" * 64,
        )
        _, execution_source = _registered_source(factory, contract)

        with self.assertRaises(_mismatch_type(self)) as raised:
            SimulationRunner().run_once(
                execution_source,
                Scenario(id="mismatch", payload={}),
                {"value": 1.0},
                seed=1,
            )

        self.assertEqual(calls["create"], 0)
        self.assertEqual(raised.exception.expected_hash, "sha256:" + "0" * 64)
        self.assertIsNotNone(raised.exception.actual_hash)
        self.assertEqual(raised.exception.status, "mismatch")

    def test_unavailable_measurement_is_rejected_before_model_call(self):
        calls = {"simulate": 0}
        module_name = "unavailable_expected_model"
        dynamic_module = types.ModuleType(module_name)

        def simulate(self, scenario, parameters, rng):
            calls["simulate"] += 1
            return ModelRun(events=(), outcome={"value": 1.0})

        dynamic_type = type(
            "DynamicModel",
            (),
            {
                "__module__": module_name,
                "name": "dynamic-unavailable",
                "version": "1.0.0",
                "implementation_revision": "test:dynamic-v1",
                "simulate": simulate,
            },
        )
        dynamic_module.DynamicModel = dynamic_type

        import sys

        sys.modules[module_name] = dynamic_module
        try:
            source = dynamic_type()
            contract = _contract_for(
                source,
                expected_hash="sha256:" + "1" * 64,
            )
            _, execution_source = _registered_source(source, contract)
            with self.assertRaises(_mismatch_type(self)) as raised:
                SimulationRunner().run_once(
                    execution_source,
                    Scenario(id="unavailable", payload={}),
                    {"value": 1.0},
                    seed=1,
                )
        finally:
            sys.modules.pop(module_name, None)

        self.assertEqual(calls["simulate"], 0)
        self.assertIsNone(raised.exception.actual_hash)
        self.assertEqual(raised.exception.status, "unavailable")


class OutcomeSchemaTests(unittest.TestCase):
    def test_invalid_local_outcome_is_rejected_after_execution(self):
        calls = {"simulate": 0}
        source = InvalidOutcomeModel(calls)
        expected = measure_implementation(source).content_hash
        contract = _contract_for(source, expected_hash=expected)
        _, execution_source = _registered_source(source, contract)

        with self.assertRaises(ModelSchemaViolation) as raised:
            SimulationRunner().run_once(
                execution_source,
                Scenario(id="invalid-local", payload={}),
                {"value": 2.0},
                seed=2,
            )

        self.assertEqual(calls["simulate"], 1)
        self.assertEqual(raised.exception.boundary, "outcome")
        self.assertEqual(raised.exception.path, "$.value")
        self.assertIsNone(raised.exception.execution)

    def test_invalid_remote_outcome_retains_execution_capture(self):
        source = SubprocessModel(
            name="invalid-outcome-process",
            factory="tests.outcome_subprocess_fixture:create_invalid_outcome_model",
            version="1.0.0",
            implementation_revision="test:invalid-outcome-process-v1",
            limits=ProcessLimits(timeout_seconds=3.0),
        )
        expected = measure_implementation(source).content_hash
        contract = _contract_for(source, expected_hash=expected)
        _, execution_source = _registered_source(source, contract)

        with self.assertRaises(ModelSchemaViolation) as raised:
            SimulationRunner().run_once(
                execution_source,
                Scenario(id="invalid-remote", payload={}),
                {"value": 2.0},
                seed=2,
            )

        self.assertEqual(raised.exception.boundary, "outcome")
        self.assertEqual(raised.exception.path, "$.value")
        self.assertIsNotNone(raised.exception.execution)
        self.assertTrue(raised.exception.execution.isolated)
        self.assertIn("invalid-outcome-stdout", raised.exception.execution.stdout)
        self.assertIn("invalid-outcome-stderr", raised.exception.execution.stderr)


if __name__ == "__main__":
    unittest.main()

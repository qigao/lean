import importlib
import unittest

from narrative_dynamics.contracts import ModelRun, Scenario, TraceEvent
from narrative_dynamics.model_contract import ModelContract, ModelSchema
from narrative_dynamics.process_execution import ProcessLimits, SubprocessModel
from narrative_dynamics.registry import ModelRegistry
from narrative_dynamics.simulation import SimulationRunner


def schema_api(test_case):
    try:
        module = importlib.import_module("narrative_dynamics.schema_validation")
    except ModuleNotFoundError as error:
        test_case.fail(f"runtime schema validation module is missing: {error}")

    names = (
        "ModelSchemaDefinitionError",
        "ModelSchemaViolation",
        "RUNTIME_SCHEMA_DIALECT",
        "validate_schema_value",
    )
    missing = tuple(name for name in names if getattr(module, name, None) is None)
    test_case.assertEqual(missing, (), f"runtime schema API is missing: {missing}")
    return module


class CountingContractModel:
    name = "counting-contract"
    version = "1.0.0"
    implementation_revision = "git:schema-v1"

    def __init__(self):
        self.calls = 0

    def simulate(self, scenario, parameters, rng):
        self.calls += 1
        level = float(parameters["level"])
        return ModelRun(
            events=(
                TraceEvent(
                    tick=0,
                    kind="accepted",
                    data={"score": level / 10.0},
                ),
            ),
            outcome={"value": level},
        )


class ForbiddenEventModel:
    name = "forbidden-event"
    version = "1.0.0"
    implementation_revision = "git:schema-v1"

    def __init__(self):
        self.calls = 0

    def simulate(self, scenario, parameters, rng):
        self.calls += 1
        return ModelRun(
            events=(TraceEvent(tick=0, kind="forbidden", data={}),),
            outcome={"value": float(parameters["level"])},
        )


class LegacyUncontractedModel:
    name = "legacy-uncontracted"

    def simulate(self, scenario, parameters, rng):
        return ModelRun(events=(), outcome={"value": float(parameters["value"])})


def in_process_contract(*, allowed_events=("accepted",)):
    return ModelContract(
        version="1.0.0",
        implementation_revision="git:schema-v1",
        parameter_schema=ModelSchema(
            name="strict-level-parameters",
            version="1.0.0",
            definition={
                "type": "object",
                "required": ("level",),
                "properties": {
                    "level": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 10.0,
                    },
                },
                "additional_properties": False,
            },
        ),
        scenario_schema=ModelSchema(
            name="scaled-runtime-scenario",
            version="1.0.0",
            definition={
                "type": "object",
                "required": ("scale",),
                "properties": {
                    "scale": {"type": "number", "minimum": 0.0},
                },
                "additional_properties": False,
            },
        ),
        event_schema=ModelSchema(
            name="accepted-runtime-events",
            version="1.0.0",
            definition={
                "event_kinds": allowed_events,
                "event_data": {
                    "accepted": {
                        "type": "object",
                        "required": ("score",),
                        "properties": {
                            "score": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                        "additional_properties": False,
                    },
                }
                if "accepted" in allowed_events
                else {},
            },
        ),
    )


def subprocess_contract(*, allowed_events=("echo",)):
    return ModelContract(
        version="1.0.0",
        implementation_revision="git:test-process-v1",
        parameter_schema=ModelSchema(
            name="echo-parameters",
            version="1.0.0",
            definition={
                "type": "object",
                "required": ("value",),
                "properties": {"value": {"type": "number"}},
                "additional_properties": False,
            },
        ),
        scenario_schema=ModelSchema(
            name="echo-scenario",
            version="1.0.0",
            definition={
                "type": "object",
                "required": ("scale",),
                "properties": {"scale": {"type": "number"}},
                "additional_properties": False,
            },
        ),
        event_schema=ModelSchema(
            name="echo-events",
            version="1.0.0",
            definition={
                "event_kinds": allowed_events,
                "event_data": {
                    "echo": {
                        "type": "object",
                        "required": ("draw",),
                        "properties": {"draw": {"type": "number"}},
                        "additional_properties": False,
                    },
                }
                if "echo" in allowed_events
                else {},
            },
        ),
    )


class RuntimeSchemaDefinitionTests(unittest.TestCase):
    def test_schema_dialect_is_explicit_and_invalid_definitions_are_rejected(self):
        api = schema_api(self)
        schema = ModelSchema(
            name="nested-profile",
            version="1.0.0",
            definition={
                "type": "object",
                "required": ("profile",),
                "properties": {
                    "profile": {
                        "type": "object",
                        "required": ("roles",),
                        "properties": {
                            "roles": {
                                "type": "array",
                                "min_items": 1,
                                "items": {
                                    "type": "string",
                                    "enum": ("guard", "prisoner"),
                                },
                            },
                        },
                    },
                },
            },
        )

        self.assertEqual(schema.dialect, api.RUNTIME_SCHEMA_DIALECT)
        self.assertTrue(schema.enforceable)
        self.assertEqual(
            schema.manifest_identity()["dialect"],
            api.RUNTIME_SCHEMA_DIALECT,
        )

        invalid_definitions = (
            {"type": "number", "unknown_keyword": True},
            {
                "type": "object",
                "required": ("missing",),
                "properties": {"present": {"type": "number"}},
            },
            {
                "event_kinds": ("known",),
                "event_data": {"unknown": {"type": "object"}},
            },
        )
        for definition in invalid_definitions:
            with self.subTest(definition=definition):
                with self.assertRaises(api.ModelSchemaDefinitionError):
                    ModelSchema(
                        name="invalid",
                        version="1.0.0",
                        definition=definition,
                    )

        with self.assertRaises(api.ModelSchemaDefinitionError):
            ModelSchema(
                name="unsupported-dialect",
                version="1.0.0",
                definition={"type": "object"},
                dialect="other/schema-v9",
            )

    def test_nested_validation_reports_the_exact_value_path(self):
        api = schema_api(self)
        schema = ModelSchema(
            name="nested-profile",
            version="1.0.0",
            definition={
                "type": "object",
                "required": ("profile",),
                "properties": {
                    "profile": {
                        "type": "object",
                        "required": ("roles",),
                        "properties": {
                            "roles": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ("guard", "prisoner"),
                                },
                            },
                        },
                    },
                },
            },
        )

        with self.assertRaises(api.ModelSchemaViolation) as raised:
            api.validate_schema_value(
                schema,
                {"profile": {"roles": ("guard", "spy")}},
                boundary="scenario",
            )

        self.assertEqual(raised.exception.boundary, "scenario")
        self.assertEqual(raised.exception.path, "$.profile.roles[1]")
        self.assertEqual(raised.exception.schema_name, "nested-profile")


class TrustedExecutionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner()

    def test_parameter_schema_violation_stops_before_in_process_model_call(self):
        api = schema_api(self)
        model = CountingContractModel()
        descriptor = ModelRegistry().register(
            model,
            contract=in_process_contract(),
        )

        with self.assertRaises(api.ModelSchemaViolation) as raised:
            self.runner.run_once(
                descriptor,
                Scenario(id="strict", payload={"scale": 1.0}),
                {"level": 0.0},
                seed=1,
            )

        self.assertEqual(raised.exception.boundary, "parameters")
        self.assertEqual(raised.exception.path, "$.level")
        self.assertEqual(model.calls, 0)

        with self.assertRaises(api.ModelSchemaViolation) as extra:
            self.runner.run_once(
                descriptor,
                Scenario(id="strict", payload={"scale": 1.0}),
                {"level": 5.0, "undeclared": 1.0},
                seed=1,
            )
        self.assertEqual(extra.exception.path, "$.undeclared")
        self.assertEqual(model.calls, 0)

    def test_scenario_schema_violation_stops_before_subprocess_launch(self):
        api = schema_api(self)
        source = SubprocessModel(
            name="process-echo",
            factory="tests.subprocess_fixtures:create_echo_model",
            version="1.0.0",
            implementation_revision="git:test-process-v1",
            limits=ProcessLimits(timeout_seconds=2.0),
        )
        descriptor = ModelRegistry().register(
            source,
            kind="pomdp",
            contract=subprocess_contract(),
        )

        with self.assertRaises(api.ModelSchemaViolation) as raised:
            self.runner.run_once(
                descriptor,
                Scenario(id="missing-scale", payload={}),
                {"value": 2.0},
                seed=2,
            )

        self.assertEqual(raised.exception.boundary, "scenario")
        self.assertEqual(raised.exception.path, "$.scale")
        self.assertIsNone(raised.exception.execution)

    def test_event_schema_violation_rejects_in_process_and_remote_results(self):
        api = schema_api(self)
        local_model = ForbiddenEventModel()
        local = ModelRegistry().register(
            local_model,
            contract=in_process_contract(),
        )
        with self.assertRaises(api.ModelSchemaViolation) as local_error:
            self.runner.run_once(
                local,
                Scenario(id="local-event", payload={"scale": 1.0}),
                {"level": 5.0},
                seed=3,
            )
        self.assertEqual(local_error.exception.boundary, "events")
        self.assertEqual(local_error.exception.path, "$[0].kind")
        self.assertEqual(local_model.calls, 1)
        self.assertIsNone(local_error.exception.execution)

        remote_source = SubprocessModel(
            name="process-echo",
            factory="tests.subprocess_fixtures:create_echo_model",
            version="1.0.0",
            implementation_revision="git:test-process-v1",
            limits=ProcessLimits(timeout_seconds=2.0),
        )
        remote = ModelRegistry().register(
            remote_source,
            kind="pomdp",
            contract=subprocess_contract(allowed_events=("allowed",)),
        )
        with self.assertRaises(api.ModelSchemaViolation) as remote_error:
            self.runner.run_once(
                remote,
                Scenario(id="remote-event", payload={"scale": 1.0}),
                {"value": 2.0},
                seed=4,
            )
        self.assertEqual(remote_error.exception.boundary, "events")
        self.assertEqual(remote_error.exception.path, "$[0].kind")
        self.assertIsNotNone(remote_error.exception.execution)
        self.assertIn("echo stdout remote-event", remote_error.exception.execution.stdout)

    def test_valid_contract_is_enforced_and_attested_in_manifest(self):
        api = schema_api(self)
        model = CountingContractModel()
        contract = in_process_contract()
        descriptor = ModelRegistry().register(model, contract=contract)

        trace = self.runner.run_once(
            descriptor,
            Scenario(id="valid", payload={"scale": 1.0}),
            {"level": 5.0},
            seed=5,
        )

        self.assertEqual(trace.outcome["value"], 5.0)
        self.assertEqual(model.calls, 1)
        attestation = trace.manifest.inputs["schema_validation"]
        self.assertEqual(attestation["dialect"], api.RUNTIME_SCHEMA_DIALECT)
        self.assertEqual(attestation["contract_hash"], contract.content_hash)
        for boundary, schema in (
            ("parameters", contract.parameter_schema),
            ("scenario", contract.scenario_schema),
            ("events", contract.event_schema),
        ):
            self.assertEqual(attestation[boundary]["status"], "validated")
            self.assertEqual(
                attestation[boundary]["schema_hash"],
                schema.content_hash,
            )

    def test_uncontracted_legacy_model_keeps_existing_execution_semantics(self):
        trace = self.runner.run_once(
            LegacyUncontractedModel(),
            Scenario(id="legacy", payload={"anything": "is still opaque"}),
            {"value": 9.0},
            seed=6,
        )

        self.assertEqual(trace.outcome["value"], 9.0)
        self.assertNotIn("schema_validation", trace.manifest.inputs)


if __name__ == "__main__":
    unittest.main()

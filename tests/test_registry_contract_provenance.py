import importlib
import unittest

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.calibration import calibrate_grid
from narrative_dynamics.contracts import ModelRun, Scenario
from narrative_dynamics.manifest import stable_content_hash
from narrative_dynamics.registry import ModelRegistry
from narrative_dynamics.simulation import ModelFactory, SimulationRunner
from narrative_dynamics.uncertainty import (
    ParameterAcceptanceSet,
    calibrate_seed_block_variants,
    repeated_grid_calibration,
)
from narrative_dynamics.validation import HeldOutCase, validate_acceptance_set_held_out


class VersionedStatefulModel:
    name = "versioned-stateful"
    version = "3.2.0"
    implementation_revision = "git:abc123"

    def __init__(self):
        self.call_count = 0

    def simulate(self, scenario, parameters, rng):
        value = float(parameters["level"]) + float(self.call_count)
        self.call_count += 1
        return ModelRun(events=(), outcome={"value": value})


class LegacyLevelModel:
    name = "legacy-level"

    def simulate(self, scenario, parameters, rng):
        return ModelRun(
            events=(),
            outcome={"value": float(parameters["level"])},
        )


class ScaledLevelModel:
    name = "scaled-level"
    version = "1.1.0"
    implementation_revision = "git:def456"

    def simulate(self, scenario, parameters, rng):
        scale = float(scenario.payload.get("scale", 1.0))
        return ModelRun(
            events=(),
            outcome={"value": float(parameters["level"]) * scale},
        )


def value_metrics(trace):
    return {"value": float(trace.outcome["value"])}


value_metrics.version = "1.0.0"


def registry_contract_types(test_case):
    registry = importlib.import_module("narrative_dynamics.registry")
    names = ("ModelLifecycle", "ModelSchema", "ModelContract")
    missing = tuple(name for name in names if getattr(registry, name, None) is None)
    test_case.assertEqual(
        missing,
        (),
        f"registry contract types are missing: {missing}",
    )
    return tuple(getattr(registry, name) for name in names)


def complete_contract(test_case, source):
    _, schema_type, contract_type = registry_contract_types(test_case)
    parameter_source = {
        "type": "object",
        "required": ["level"],
        "properties": {"level": {"type": "number"}},
    }
    parameter_schema = schema_type(
        name="level-parameters",
        version="1.0.0",
        definition=parameter_source,
    )
    scenario_schema = schema_type(
        name="scaled-scenario",
        version="1.0.0",
        definition={
            "type": "object",
            "properties": {"scale": {"type": "number"}},
        },
    )
    event_schema = schema_type(
        name="no-events",
        version="1.0.0",
        definition={"event_kinds": ()},
    )
    outcome_schema = schema_type(
        name="value-outcome",
        version="1.0.0",
        definition={
            "type": "object",
            "required": ("value",),
            "properties": {"value": {"type": "number"}},
        },
    )
    contract = contract_type(
        version="3.2.0",
        implementation_revision="git:abc123",
        expected_implementation_hash=measure_implementation(source).content_hash,
        parameter_schema=parameter_schema,
        scenario_schema=scenario_schema,
        event_schema=event_schema,
        outcome_schema=outcome_schema,
    )
    return contract, parameter_source


class ModelSchemaTests(unittest.TestCase):
    def test_schema_is_detached_immutable_and_content_hashed(self):
        _, schema_type, _ = registry_contract_types(self)
        source = {
            "type": "object",
            "required": ["level"],
            "properties": {"level": {"type": "number"}},
        }
        schema = schema_type(
            name="level-parameters",
            version="1.0.0",
            definition=source,
        )
        same = schema_type(
            name="level-parameters",
            version="1.0.0",
            definition={
                "properties": {"level": {"type": "number"}},
                "required": ("level",),
                "type": "object",
            },
        )
        source["required"].append("mutated")
        source["properties"]["level"]["type"] = "string"

        self.assertEqual(schema.definition["required"], ("level",))
        self.assertEqual(
            schema.definition["properties"]["level"]["type"],
            "number",
        )
        self.assertEqual(schema.content_hash, same.content_hash)
        self.assertRegex(schema.content_hash, r"^sha256:[0-9a-f]{64}$")
        self.assertTrue(schema.specified)
        with self.assertRaises(TypeError):
            schema.definition["new"] = "forbidden"

        unspecified = schema_type.unspecified("parameters")
        self.assertFalse(unspecified.specified)
        self.assertEqual(unspecified.version, "unversioned")


class RegistryContractTests(unittest.TestCase):
    def test_factory_lifecycle_and_contract_are_bound_into_run_manifests(self):
        lifecycle_type, _, _ = registry_contract_types(self)
        instances = []

        def create_model():
            model = VersionedStatefulModel()
            instances.append(model)
            return model

        factory = ModelFactory(
            name="versioned-stateful",
            create=create_model,
            version="3.2.0",
            implementation_revision="git:abc123",
        )
        contract, _ = complete_contract(self, factory)
        registry = ModelRegistry()
        descriptor = registry.register(
            factory,
            kind="pomdp",
            contract=contract,
        )

        self.assertEqual(descriptor.lifecycle, lifecycle_type.FRESH_PER_BATCH)
        self.assertIs(descriptor.source, factory)
        self.assertEqual(descriptor.contract, contract)
        self.assertTrue(descriptor.production_ready)
        self.assertIs(registry.execution_source("versioned-stateful"), descriptor)
        self.assertIs(registry.require_production_ready("versioned-stateful"), descriptor)

        result = calibrate_grid(
            runner=SimulationRunner(),
            model=descriptor,
            scenario=Scenario(id="factory", payload={}),
            parameter_grid={"level": (1.0, 2.0, 3.0)},
            seeds=(10, 11),
            extractor=value_metrics,
            target={"value": 2.5},
        )

        self.assertEqual(result.best.parameters, (("level", 2.0),))
        self.assertEqual(len(instances), 3)
        self.assertTrue(all(instance.call_count == 2 for instance in instances))
        model_identity = result.manifest.inputs["model"]
        self.assertEqual(model_identity["version"], "3.2.0")
        self.assertEqual(
            model_identity["implementation_revision"],
            "git:abc123",
        )
        self.assertEqual(model_identity["lifecycle"], "fresh_per_batch")
        self.assertEqual(model_identity["contract_hash"], contract.content_hash)
        self.assertEqual(
            model_identity["schemas"]["parameters"]["content_hash"],
            contract.parameter_schema.content_hash,
        )
        self.assertEqual(
            model_identity["schemas"]["outcome"]["content_hash"],
            contract.outcome_schema.content_hash,
        )

    def test_legacy_models_remain_resolvable_but_are_explicitly_incomplete(self):
        lifecycle_type, _, _ = registry_contract_types(self)
        model = LegacyLevelModel()
        registry = ModelRegistry()
        descriptor = registry.register(model)

        self.assertIs(registry.resolve("legacy-level"), model)
        self.assertEqual(descriptor.lifecycle, lifecycle_type.SHARED_INSTANCE)
        self.assertEqual(descriptor.contract.version, "unversioned")
        self.assertEqual(
            descriptor.contract.implementation_revision,
            "unversioned",
        )
        self.assertIsNone(descriptor.contract.expected_implementation_hash)
        self.assertFalse(descriptor.contract.parameter_schema.specified)
        self.assertFalse(descriptor.contract.outcome_schema.specified)
        self.assertFalse(descriptor.production_ready)
        with self.assertRaises(ValueError):
            registry.require_production_ready("legacy-level")

    def test_explicit_contract_cannot_disagree_with_declared_source_identity(self):
        _, _, contract_type = registry_contract_types(self)
        factory = ModelFactory(
            name="versioned-stateful",
            create=VersionedStatefulModel,
            version="3.2.0",
            implementation_revision="git:abc123",
        )
        contract, _ = complete_contract(self, factory)
        mismatched = contract_type(
            version="9.9.9",
            implementation_revision=contract.implementation_revision,
            expected_implementation_hash=contract.expected_implementation_hash,
            parameter_schema=contract.parameter_schema,
            scenario_schema=contract.scenario_schema,
            event_schema=contract.event_schema,
            outcome_schema=contract.outcome_schema,
        )

        with self.assertRaises(ValueError):
            ModelRegistry().register(factory, kind="pomdp", contract=mismatched)


class AcceptanceSetProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner()
        self.model = ScaledLevelModel()
        self.calibration_scenario = Scenario(
            id="calibration",
            payload={"scale": 1.0},
        )

    def test_repeated_acceptance_lineage_survives_external_filtering(self):
        repeated = repeated_grid_calibration(
            runner=self.runner,
            model=self.model,
            scenario=self.calibration_scenario,
            parameter_grid={"level": (1.0, 2.0, 3.0)},
            seed_blocks=((1, 2), (3, 4)),
            extractor=value_metrics,
            target={"value": 2.0},
        )
        accepted = repeated.acceptance_set
        source_hash = repeated.manifest.content_hash

        self.assertEqual(accepted.source_manifest_hashes, (source_hash,))

        validation = validate_acceptance_set_held_out(
            runner=self.runner,
            model=self.model,
            accepted_parameters=accepted,
            cases=(
                HeldOutCase(
                    scenario=Scenario(id="scale-2", payload={"scale": 2.0}),
                    seeds=(101, 102),
                    target={"value": 4.0},
                ),
            ),
            extractor=value_metrics,
            max_mean_loss=0.0,
            max_worst_loss=0.0,
        )

        self.assertIn(source_hash, validation.manifest.parent_hashes)
        self.assertEqual(
            validation.retained_parameters.source_manifest_hashes,
            (validation.manifest.content_hash,),
        )

        parent_a = stable_content_hash({"source": "a"})
        parent_b = stable_content_hash({"source": "b"})
        manual = ParameterAcceptanceSet.from_parameters(
            ((("level", 2.0),),),
            source_manifest_hashes=(parent_b, parent_a, parent_a),
        )
        self.assertEqual(
            manual.source_manifest_hashes,
            tuple(sorted((parent_a, parent_b))),
        )

    def test_seed_variation_union_and_intersection_point_to_variation_report(self):
        report = calibrate_seed_block_variants(
            runner=self.runner,
            model=self.model,
            scenario=self.calibration_scenario,
            parameter_grid={"level": (1.0, 2.0, 3.0)},
            seed_blocks=((10, 11), (20, 21)),
            seed_offsets=(0, 100),
            extractor=value_metrics,
            target={"value": 2.0},
        )
        source = (report.manifest.content_hash,)

        self.assertEqual(report.accepted_union.source_manifest_hashes, source)
        self.assertEqual(report.accepted_intersection.source_manifest_hashes, source)


if __name__ == "__main__":
    unittest.main()

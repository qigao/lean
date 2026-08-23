import unittest

from narrative_dynamics.contracts import ModelRun, Scenario
from narrative_dynamics.registry import ModelKind, ModelRegistry
from narrative_dynamics.simulation import SimulationRunner


class ConstantModel:
    def __init__(self, name, value):
        self.name = name
        self.value = float(value)

    def simulate(self, scenario, parameters, rng):
        return ModelRun(events=(), outcome={"value": self.value})


class MissingSimulate:
    name = "missing-simulate"


class NonCallableSimulate:
    name = "non-callable"
    simulate = 1


class ModelRegistryTests(unittest.TestCase):
    def test_models_are_registered_resolved_and_filtered_by_kind(self):
        registry = ModelRegistry()
        planner = ConstantModel("planner-a", 1.0)
        pomdp = ConstantModel("pomdp-a", 2.0)
        population = ConstantModel("population-a", 3.0)

        planner_entry = registry.register(planner, kind=ModelKind.PLANNER)
        registry.register(pomdp, kind="pomdp")
        registry.register(population, kind=ModelKind.POPULATION)

        self.assertIs(registry.resolve("planner-a"), planner)
        self.assertEqual(planner_entry.name, "planner-a")
        self.assertEqual(planner_entry.kind, ModelKind.PLANNER)
        self.assertEqual(
            registry.names(),
            ("planner-a", "pomdp-a", "population-a"),
        )
        self.assertEqual(registry.names(kind=ModelKind.POMDP), ("pomdp-a",))
        self.assertEqual(
            tuple(entry.kind for entry in registry.entries()),
            (ModelKind.PLANNER, ModelKind.POMDP, ModelKind.POPULATION),
        )

    def test_resolved_model_still_runs_through_the_canonical_runner(self):
        registry = ModelRegistry()
        registry.register(ConstantModel("rule-a", 7.0), kind=ModelKind.RULE_ENGINE)

        trace = SimulationRunner().run_once(
            registry.resolve("rule-a"),
            Scenario(id="registry", payload={}),
            {},
            seed=42,
        )

        self.assertEqual(trace.model_name, "rule-a")
        self.assertEqual(trace.outcome, {"value": 7.0})
        self.assertEqual(trace.seed, 42)

    def test_duplicate_names_and_unknown_models_are_rejected(self):
        registry = ModelRegistry()
        registry.register(ConstantModel("same", 1.0))

        with self.assertRaises(ValueError):
            registry.register(ConstantModel("same", 2.0), kind=ModelKind.POMDP)
        with self.assertRaises(KeyError):
            registry.resolve("missing")

    def test_registry_rejects_invalid_protocol_shapes_and_kinds(self):
        registry = ModelRegistry()

        with self.assertRaises(ValueError):
            registry.register(ConstantModel("", 1.0))
        with self.assertRaises(TypeError):
            registry.register(MissingSimulate())
        with self.assertRaises(TypeError):
            registry.register(NonCallableSimulate())
        with self.assertRaises(ValueError):
            registry.register(ConstantModel("bad-kind", 1.0), kind="unknown")
        self.assertEqual(registry.names(kind="missing"), ())

    def test_supported_model_kinds_are_explicit_and_stable(self):
        self.assertEqual(
            tuple(kind.value for kind in ModelKind),
            ("generic", "planner", "pomdp", "rule_engine", "population"),
        )


if __name__ == "__main__":
    unittest.main()

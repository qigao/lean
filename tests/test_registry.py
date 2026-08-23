import unittest

from narrative_dynamics.contracts import ModelRun, Scenario
from narrative_dynamics.registry import ModelRegistry
from narrative_dynamics.simulation import SimulationRunner


class EchoModel:
    def __init__(self, name):
        self.name = name

    def simulate(self, scenario, parameters, rng):
        return ModelRun(
            events=(),
            outcome={
                "scenario": scenario.id,
                "value": float(parameters["value"]),
                "draw": rng.random(),
            },
        )


class MissingSimulateModel:
    name = "missing-simulate"


class ModelRegistryTests(unittest.TestCase):
    def test_register_resolve_and_list_are_deterministic(self):
        registry = ModelRegistry()
        planner = EchoModel("planner-alpha")
        population = EchoModel("population-zeta")
        belief = EchoModel("belief-beta")

        registry.register(population, kind="population")
        registry.register(planner, kind="planner")
        registry.register(belief, kind="pomdp")

        self.assertIs(registry.resolve("planner-alpha"), planner)
        self.assertEqual(
            registry.names(),
            ("belief-beta", "planner-alpha", "population-zeta"),
        )
        self.assertEqual(registry.names(kind="planner"), ("planner-alpha",))
        self.assertEqual(registry.names(kind="missing"), ())
        self.assertEqual(
            tuple(descriptor.kind for descriptor in registry.descriptors()),
            ("pomdp", "planner", "population"),
        )

    def test_duplicate_or_invalid_adapters_are_rejected(self):
        registry = ModelRegistry()
        registry.register(EchoModel("shared"), kind="planner")

        with self.assertRaises(ValueError):
            registry.register(EchoModel("shared"), kind="pomdp")
        with self.assertRaises(ValueError):
            registry.register(EchoModel(""), kind="planner")
        with self.assertRaises(ValueError):
            registry.register(EchoModel("spaces"), kind="   ")
        with self.assertRaises(TypeError):
            registry.register(MissingSimulateModel(), kind="planner")

    def test_unknown_resolution_is_explicit(self):
        registry = ModelRegistry()
        with self.assertRaises(KeyError):
            registry.resolve("unknown")
        with self.assertRaises(KeyError):
            registry.descriptor("unknown")

    def test_registered_adapter_still_runs_only_through_simulation_runner(self):
        registry = ModelRegistry()
        model = EchoModel("echo")
        registry.register(model, kind="rule-engine")

        resolved = registry.resolve("echo")
        trace = SimulationRunner().run_once(
            resolved,
            Scenario(id="registered", payload={}),
            {"value": 2.5},
            seed=17,
        )

        self.assertEqual(trace.model_name, "echo")
        self.assertEqual(trace.scenario_id, "registered")
        self.assertEqual(trace.parameters, (("value", 2.5),))
        self.assertEqual(trace.seed, 17)
        self.assertEqual(trace.outcome["value"], 2.5)


if __name__ == "__main__":
    unittest.main()

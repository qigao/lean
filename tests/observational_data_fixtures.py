from __future__ import annotations

from narrative_dynamics.contracts import ModelRun, Scenario


class FlexiblePolicyModel:
    name = "flexible-policy"
    version = "1.0.0"

    def __init__(self) -> None:
        self.calls = 0

    def simulate(self, scenario, parameters, rng):
        self.calls += 1
        probability = float(parameters["p"])
        return ModelRun(
            events=(),
            outcome={
                "policy": {
                    "a": probability,
                    "b": 1.0 - probability,
                }
            },
        )


class ConservativePolicyModel:
    name = "conservative-policy"
    version = "1.0.0"

    def __init__(self) -> None:
        self.calls = 0

    def simulate(self, scenario, parameters, rng):
        self.calls += 1
        probability = 0.5 + 0.1 * float(parameters["q"])
        return ModelRun(
            events=(),
            outcome={
                "policy": {
                    "a": probability,
                    "b": 1.0 - probability,
                }
            },
        )


class ReplacementPolicyModel:
    name = "replacement-policy"
    version = "1.0.0"

    def __init__(self) -> None:
        self.calls = 0

    def simulate(self, scenario, parameters, rng):
        self.calls += 1
        return ModelRun(
            events=(),
            outcome={"policy": {"a": 0.5, "b": 0.5}},
        )


def choice_metrics(trace):
    policy = trace.outcome["policy"]
    return {
        "choice.a": float(policy["a"]),
        "choice.b": float(policy["b"]),
    }


def scenario(name: str) -> Scenario:
    return Scenario(id=name, payload={"context": name})

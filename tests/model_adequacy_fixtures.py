from __future__ import annotations

from narrative_dynamics.adapters.prison_pomdp import (
    prison_pomdp_contract,
    prison_pomdp_source,
)
from narrative_dynamics.attestation import RepositoryIdentity, measure_implementation
from narrative_dynamics.contracts import Scenario
from narrative_dynamics.execution_policy import (
    TrustedExecutionPolicy,
    TrustedModelPin,
)
from narrative_dynamics.process_execution import ProcessLimits
from narrative_dynamics.simulation import SimulationRunner


def prison_scenario(
    scenario_id: str,
    *,
    prior_weak: float = 0.5,
    signal_accuracy: float = 0.85,
    guard_persistence: float = 0.9,
    horizon: int = 2,
) -> Scenario:
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


def adequacy_runner() -> SimulationRunner:
    return SimulationRunner(
        repository_identity=RepositoryIdentity(provider="test")
    )


def bound_prison_source():
    source = prison_pomdp_source(
        limits=ProcessLimits(
            timeout_seconds=3.0,
            max_output_bytes=64 * 1024,
            max_trace_bytes=128 * 1024,
        )
    )
    contract = prison_pomdp_contract()
    policy = TrustedExecutionPolicy(
        name="prison-adequacy-test-policy",
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
    return policy.bind(source, contract=contract)


def initial_policy_metrics(trace) -> dict[str, float]:
    policy = trace.outcome["initial_policy"]
    return {
        "initial.scout": float(policy["scout"]),
        "initial.escape": float(policy["escape"]),
        "initial.submit": float(policy["submit"]),
    }


initial_policy_metrics.version = "1.0.0"


def initial_policy_target(
    runner: SimulationRunner,
    model,
    scenario: Scenario,
    beta: float,
    *,
    seed: int,
) -> dict[str, float]:
    trace = runner.run_once(
        model,
        scenario,
        {"beta": beta},
        seed=seed,
    )
    return initial_policy_metrics(trace)

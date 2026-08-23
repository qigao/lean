from __future__ import annotations

import unittest

from narrative_dynamics.attestation import RepositoryIdentity
from narrative_dynamics.contracts import Scenario
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.adapters.prison_pomdp import create_prison_pomdp_model


class PrisonPOMDPNumericalBoundaryTests(unittest.TestCase):
    def test_softmax_never_assigns_negative_mass_to_underflowed_action(self):
        trace = SimulationRunner(
            repository_identity=RepositoryIdentity(provider="test")
        ).run_once(
            create_prison_pomdp_model(),
            Scenario(
                id="underflow-policy",
                payload={
                    "prior_weak": 0.1,
                    "signal_accuracy": 0.5,
                    "guard_persistence": 0.2,
                    "escape_reward": 1.0,
                    "capture_cost": 1.0,
                    "submit_reward": -100.0,
                    "scout_cost": 5.0,
                    "discount": 0.2,
                    "horizon": 2,
                },
            ),
            {"beta": 0.5},
            seed=1,
        )

        policy = trace.outcome["initial_policy"]
        self.assertTrue(all(probability >= 0.0 for probability in policy.values()))
        self.assertAlmostEqual(sum(policy.values()), 1.0)


if __name__ == "__main__":
    unittest.main()

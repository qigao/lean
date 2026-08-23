from __future__ import annotations

import math
import unittest

from narrative_dynamics.contracts import SimulationTrace


class PrisonInitialActionMetricTests(unittest.TestCase):
    def _trace(self, policy):
        return SimulationTrace(
            model_name="fixture-model",
            scenario_id="fixture-scenario",
            parameters=(("beta", 1.0),),
            seed=1,
            events=(),
            outcome={"initial_policy": policy},
        )

    def test_shared_extractor_returns_exact_three_initial_action_coordinates(self):
        from narrative_dynamics.adapters.prison_metrics import (
            prison_initial_action_metrics,
        )

        metrics = prison_initial_action_metrics(
            self._trace({"scout": 0.25, "escape": 0.5, "submit": 0.25})
        )
        self.assertEqual(
            metrics,
            {
                "initial.scout": 0.25,
                "initial.escape": 0.5,
                "initial.submit": 0.25,
            },
        )
        self.assertEqual(prison_initial_action_metrics.version, "1.0.0")

    def test_shared_extractor_rejects_missing_or_non_finite_coordinates(self):
        from narrative_dynamics.adapters.prison_metrics import (
            prison_initial_action_metrics,
        )

        with self.assertRaises(ValueError):
            prison_initial_action_metrics(
                self._trace({"scout": 0.5, "escape": 0.5})
            )
        with self.assertRaises(ValueError):
            prison_initial_action_metrics(
                self._trace({"scout": math.nan, "escape": 0.5, "submit": 0.5})
            )


if __name__ == "__main__":
    unittest.main()

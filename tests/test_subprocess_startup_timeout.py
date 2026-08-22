from __future__ import annotations

import time
import unittest

from narrative_dynamics.contracts import Scenario
from narrative_dynamics.process_execution import (
    ModelTimeout,
    ProcessLimits,
    SubprocessModel,
)
from narrative_dynamics.simulation import SimulationRunner


class SubprocessStartupTimeoutTests(unittest.TestCase):
    def test_timeout_covers_worker_startup_before_stdin_is_read(self):
        source = SubprocessModel(
            name="process-echo",
            factory="tests.subprocess_fixtures:create_echo_model",
            version="1.0.0",
            implementation_revision="git:test-process-v1",
            worker_module="tests.delayed_subprocess_worker",
            limits=ProcessLimits(
                timeout_seconds=0.20,
                max_output_bytes=64 * 1024,
                max_trace_bytes=64 * 1024,
            ),
        )
        large_scenario = Scenario(
            id="delayed-startup",
            payload={"blob": "x" * (512 * 1024)},
        )

        started = time.monotonic()
        with self.assertRaises(ModelTimeout):
            SimulationRunner().run_once(
                source,
                large_scenario,
                {"value": 1.0},
                seed=11,
            )
        self.assertLess(time.monotonic() - started, 1.0)


if __name__ == "__main__":
    unittest.main()

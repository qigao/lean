from __future__ import annotations

import importlib
import math
import os
import random
import signal
import threading
import time
import unittest

from narrative_dynamics.contracts import Scenario
from narrative_dynamics.model_contract import ModelContract, ModelLifecycle, ModelSchema
from narrative_dynamics.registry import ModelRegistry
from narrative_dynamics.simulation import SimulationRunner


def execution_api(test_case):
    try:
        module = importlib.import_module("narrative_dynamics.process_execution")
    except ModuleNotFoundError as error:
        test_case.fail(f"subprocess execution module is missing: {error}")

    names = (
        "CancellationToken",
        "ModelCancelled",
        "ModelOutputLimitExceeded",
        "ModelProtocolError",
        "ModelRemoteError",
        "ModelResourceLimitExceeded",
        "ModelTimeout",
        "ModelTraceLimitExceeded",
        "ProcessLimits",
        "SubprocessModel",
    )
    missing = tuple(name for name in names if getattr(module, name, None) is None)
    test_case.assertEqual(missing, (), f"subprocess execution API is missing: {missing}")
    return module


def full_contract():
    return ModelContract(
        version="1.0.0",
        implementation_revision="git:test-process-v1",
        parameter_schema=ModelSchema(
            name="process-parameters",
            version="1.0.0",
            definition={"type": "object"},
        ),
        scenario_schema=ModelSchema(
            name="process-scenario",
            version="1.0.0",
            definition={"type": "object"},
        ),
        event_schema=ModelSchema(
            name="process-events",
            version="1.0.0",
            definition={"event_kinds": ("echo",)},
        ),
    )


class SubprocessExecutionTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner()
        self.scenario = Scenario(id="isolated", payload={"label": "fixture"})

    def source(
        self,
        name,
        factory,
        *,
        timeout_seconds=2.0,
        max_output_bytes=64 * 1024,
        max_trace_bytes=64 * 1024,
        max_memory_bytes=None,
        max_cpu_seconds=None,
    ):
        api = execution_api(self)
        return api.SubprocessModel(
            name=name,
            factory=factory,
            version="1.0.0",
            implementation_revision="git:test-process-v1",
            limits=api.ProcessLimits(
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
                max_trace_bytes=max_trace_bytes,
                max_memory_bytes=max_memory_bytes,
                max_cpu_seconds=max_cpu_seconds,
            ),
        )

    def test_success_is_seeded_isolated_and_captures_stdout_stderr(self):
        source = self.source(
            "process-echo",
            "tests.subprocess_fixtures:create_echo_model",
        )

        first = self.runner.run_once(
            source,
            self.scenario,
            {"value": 2.5},
            seed=17,
        )
        second = self.runner.run_once(
            source,
            self.scenario,
            {"value": 2.5},
            seed=17,
        )

        expected_draw = random.Random(17).random()
        self.assertAlmostEqual(first.outcome["draw"], expected_draw)
        self.assertEqual(first.outcome["value"], 2.5)
        self.assertEqual(first.outcome["instance_count"], 1)
        self.assertEqual(second.outcome["instance_count"], 1)
        self.assertEqual(first.events[0].kind, "echo")
        self.assertIsNotNone(first.execution)
        self.assertTrue(first.execution.isolated)
        self.assertEqual(first.execution.return_code, 0)
        self.assertIn("echo stdout isolated", first.execution.stdout)
        self.assertIn("echo stderr isolated", first.execution.stderr)
        self.assertEqual(
            first.manifest.inputs["model"]["lifecycle"],
            "fresh_process_per_run",
        )
        self.assertEqual(
            first.manifest.inputs["model"]["factory"],
            "tests.subprocess_fixtures:create_echo_model",
        )

    def test_timeout_terminates_worker_and_preserves_partial_output(self):
        api = execution_api(self)
        source = self.source(
            "process-sleep",
            "tests.subprocess_fixtures:create_sleep_model",
            timeout_seconds=1.0,
        )

        started = time.monotonic()
        with self.assertRaises(api.ModelTimeout) as raised:
            self.runner.run_once(
                source,
                self.scenario,
                {"seconds": 5.0},
                seed=1,
            )

        self.assertLess(time.monotonic() - started, 2.0)
        self.assertIn("sleep started", raised.exception.stdout)
        self.assertEqual(raised.exception.timeout_seconds, 1.0)

    def test_cancellation_terminates_worker(self):
        api = execution_api(self)
        source = self.source(
            "process-sleep",
            "tests.subprocess_fixtures:create_sleep_model",
            timeout_seconds=5.0,
        )
        cancellation = api.CancellationToken()
        timer = threading.Timer(1.0, cancellation.cancel)
        timer.start()
        try:
            with self.assertRaises(api.ModelCancelled) as raised:
                self.runner.run_once(
                    source,
                    self.scenario,
                    {"seconds": 5.0},
                    seed=2,
                    cancellation=cancellation,
                )
        finally:
            timer.cancel()

        self.assertIn("sleep started", raised.exception.stdout)

    def test_remote_exception_is_typed_with_traceback_and_output(self):
        api = execution_api(self)
        source = self.source(
            "process-failure",
            "tests.subprocess_fixtures:create_failing_model",
        )

        with self.assertRaises(api.ModelRemoteError) as raised:
            self.runner.run_once(
                source,
                self.scenario,
                {},
                seed=3,
            )

        error = raised.exception
        self.assertEqual(error.remote_type, "RuntimeError")
        self.assertIn("fixture boom", str(error))
        self.assertIn("Traceback", error.remote_traceback)
        self.assertIn("before remote failure", error.stdout)
        self.assertIn("remote failure stderr", error.stderr)

    def test_invalid_factory_is_a_protocol_error(self):
        api = execution_api(self)
        source = self.source(
            "process-missing",
            "tests.subprocess_fixtures:missing_factory",
        )

        with self.assertRaises(api.ModelProtocolError) as raised:
            self.runner.run_once(
                source,
                self.scenario,
                {},
                seed=4,
            )

        self.assertIn("missing_factory", str(raised.exception))

    def test_output_and_trace_size_limits_are_distinct(self):
        api = execution_api(self)
        noisy = self.source(
            "process-noisy",
            "tests.subprocess_fixtures:create_noisy_model",
            max_output_bytes=4096,
        )
        with self.assertRaises(api.ModelOutputLimitExceeded) as output_error:
            self.runner.run_once(
                noisy,
                self.scenario,
                {"bytes": 128 * 1024},
                seed=5,
            )
        self.assertGreater(output_error.exception.captured_bytes, 4096)

        huge_trace = self.source(
            "process-huge-trace",
            "tests.subprocess_fixtures:create_huge_trace_model",
            max_trace_bytes=1024,
        )
        with self.assertRaises(api.ModelTraceLimitExceeded) as trace_error:
            self.runner.run_once(
                huge_trace,
                self.scenario,
                {"bytes": 16 * 1024},
                seed=6,
            )
        self.assertGreater(trace_error.exception.trace_bytes, 1024)

    @unittest.skipUnless(
        os.name == "posix" and hasattr(signal, "SIGXCPU"),
        "POSIX CPU resource limits are required",
    )
    def test_cpu_resource_limit_is_typed(self):
        api = execution_api(self)
        source = self.source(
            "process-busy",
            "tests.subprocess_fixtures:create_busy_model",
            timeout_seconds=5.0,
            max_cpu_seconds=1,
        )

        with self.assertRaises(api.ModelResourceLimitExceeded) as raised:
            self.runner.run_once(
                source,
                self.scenario,
                {},
                seed=7,
            )

        self.assertEqual(raised.exception.resource, "cpu")
        self.assertIn("busy started", raised.exception.stdout)

    def test_limits_reject_non_positive_or_non_finite_values(self):
        api = execution_api(self)

        invalid = (
            {"timeout_seconds": 0.0},
            {"timeout_seconds": math.inf},
            {"max_output_bytes": 0},
            {"max_trace_bytes": 0},
            {"max_memory_bytes": 0},
            {"max_cpu_seconds": 0},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    api.ProcessLimits(**kwargs)


class SubprocessRegistryTests(unittest.TestCase):
    def test_registry_binds_fresh_process_lifecycle_contract_and_manifest(self):
        api = execution_api(self)
        source = api.SubprocessModel(
            name="process-echo",
            factory="tests.subprocess_fixtures:create_echo_model",
            version="1.0.0",
            implementation_revision="git:test-process-v1",
            limits=api.ProcessLimits(timeout_seconds=2.0),
        )
        registry = ModelRegistry()
        descriptor = registry.register(
            source,
            kind="pomdp",
            contract=full_contract(),
        )

        self.assertEqual(
            descriptor.lifecycle,
            ModelLifecycle.FRESH_PROCESS_PER_RUN,
        )
        self.assertTrue(descriptor.production_ready)
        self.assertIs(registry.execution_source("process-echo"), descriptor)

        trace = SimulationRunner().run_once(
            descriptor,
            Scenario(id="registry-process", payload={}),
            {"value": 4.0},
            seed=8,
        )
        identity = trace.manifest.inputs["model"]
        self.assertEqual(identity["kind"], "pomdp")
        self.assertEqual(identity["lifecycle"], "fresh_process_per_run")
        self.assertEqual(identity["contract_hash"], full_contract().content_hash)
        self.assertEqual(
            identity["factory"],
            "tests.subprocess_fixtures:create_echo_model",
        )


if __name__ == "__main__":
    unittest.main()

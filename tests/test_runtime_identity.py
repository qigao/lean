from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from narrative_dynamics.attestation import (
    RepositoryIdentity,
    ResultArtifact,
    detect_repository_identity,
    measure_implementation,
)
from narrative_dynamics.contracts import ModelRun, Scenario, TraceEvent
from narrative_dynamics.process_execution import ProcessLimits, SubprocessModel
from narrative_dynamics.simulation import SimulationRunner


class AttestedModel:
    name = "attested-model"
    version = "1.0.0"

    def simulate(self, scenario, parameters, rng):
        value = float(parameters["value"])
        return ModelRun(
            events=(TraceEvent(tick=0, kind="value", data={"value": value}),),
            outcome={"value": value},
        )


class ImplementationMeasurementTests(unittest.TestCase):
    def test_module_byte_change_changes_measured_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            module_path = Path(directory) / "measured_fixture.py"
            module_path.write_text("class Model: pass\n", encoding="utf-8")
            sys.path.insert(0, directory)
            try:
                module = importlib.import_module("measured_fixture")
                first = measure_implementation(module.Model())
                module_path.write_text(
                    "class Model:\n    changed = True\n",
                    encoding="utf-8",
                )
                second = measure_implementation(module.Model())
            finally:
                sys.path.remove(directory)
                sys.modules.pop("measured_fixture", None)

        self.assertNotEqual(first.content_hash, second.content_hash)
        self.assertEqual(
            tuple(item.locator for item in first.artifacts),
            ("python-module:measured_fixture",),
        )
        self.assertTrue(
            all(
                "/" not in item.locator and "\\" not in item.locator
                for item in first.artifacts
            )
        )

    def test_subprocess_measurement_does_not_import_external_factory(self):
        source = SubprocessModel(
            name="process-echo",
            factory="tests.subprocess_fixtures:create_echo_model",
            version="1.0.0",
            implementation_revision="git:test-process-v1",
            limits=ProcessLimits(timeout_seconds=2.0),
        )
        sys.modules.pop("tests.subprocess_fixtures", None)
        measured = measure_implementation(source)

        self.assertNotIn("tests.subprocess_fixtures", sys.modules)
        self.assertEqual(
            tuple(item.locator for item in measured.artifacts),
            (
                "python-module:narrative_dynamics.subprocess_worker",
                "python-module:tests.subprocess_fixtures",
            ),
        )

    def test_runner_snapshots_implementation_before_model_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            module_path = Path(directory) / "mutating_fixture.py"
            module_path.write_text(
                "from pathlib import Path\n"
                "from narrative_dynamics.contracts import ModelRun\n"
                "class Model:\n"
                "    name = 'mutating-fixture'\n"
                "    def simulate(self, scenario, parameters, rng):\n"
                "        Path(__file__).write_text('changed = True\\n', encoding='utf-8')\n"
                "        return ModelRun(events=(), outcome={'value': 1.0})\n",
                encoding="utf-8",
            )
            sys.path.insert(0, directory)
            try:
                module = importlib.import_module("mutating_fixture")
                model = module.Model()
                before = measure_implementation(model)
                trace = SimulationRunner(
                    repository_identity=RepositoryIdentity(provider="test")
                ).run_once(
                    model,
                    Scenario(id="mutating", payload={}),
                    {},
                    seed=1,
                )
                after = measure_implementation(model)
            finally:
                sys.path.remove(directory)
                sys.modules.pop("mutating_fixture", None)

        self.assertNotEqual(before.content_hash, after.content_hash)
        self.assertEqual(
            trace.manifest.inputs["model"]["implementation_attestation"][
                "content_hash"
            ],
            before.content_hash,
        )


class RepositoryAndResultArtifactTests(unittest.TestCase):
    def test_github_pr_identity_separates_checkout_head_and_base(self):
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "event.json"
            event_path.write_text(
                json.dumps(
                    {
                        "pull_request": {
                            "head": {"sha": "b" * 40},
                            "base": {"sha": "c" * 40},
                        }
                    }
                ),
                encoding="utf-8",
            )
            identity = detect_repository_identity(
                environ={
                    "GITHUB_ACTIONS": "true",
                    "GITHUB_SHA": "a" * 40,
                    "GITHUB_REPOSITORY": "qigao/lean",
                    "GITHUB_REF": "refs/pull/2/merge",
                    "GITHUB_EVENT_PATH": str(event_path),
                }
            )

        self.assertEqual(identity.checkout_commit, "a" * 40)
        self.assertEqual(identity.source_commit, "b" * 40)
        self.assertEqual(identity.base_commit, "c" * 40)
        self.assertEqual(identity.manifest_identity()["status"], "measured")

    def test_run_manifest_hashes_code_repository_events_and_outcome(self):
        repository = RepositoryIdentity(
            provider="test",
            repository="qigao/lean",
            checkout_commit="a" * 40,
            source_commit="b" * 40,
            base_commit="c" * 40,
            ref="refs/pull/2/merge",
        )
        trace = SimulationRunner(repository_identity=repository).run_once(
            AttestedModel(),
            Scenario(id="artifact", payload={}),
            {"value": 3.0},
            seed=5,
        )

        measured = trace.manifest.inputs["model"]["implementation_attestation"]
        artifact = trace.manifest.inputs["result_artifact"]
        self.assertEqual(measured["status"], "measured")
        self.assertEqual(
            trace.manifest.inputs["repository"]["content_hash"],
            repository.content_hash,
        )
        expected = ResultArtifact.from_result(trace.events, trace.outcome)
        self.assertEqual(artifact["content_hash"], expected.content_hash)
        self.assertTrue(expected.matches(trace.events, trace.outcome))

        event_changed = ResultArtifact.from_result(
            (TraceEvent(tick=0, kind="value", data={"value": 4.0}),),
            trace.outcome,
        )
        outcome_changed = ResultArtifact.from_result(
            trace.events,
            {"value": 4.0},
        )
        self.assertNotEqual(expected.content_hash, event_changed.content_hash)
        self.assertNotEqual(expected.content_hash, outcome_changed.content_hash)


if __name__ == "__main__":
    unittest.main()

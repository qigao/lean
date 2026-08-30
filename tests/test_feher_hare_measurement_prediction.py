from __future__ import annotations

from contextlib import ExitStack
from dataclasses import fields, replace
import unittest
from unittest.mock import patch

from narrative_dynamics.adapters.narrative_two_stage import (
    create_narrative_two_stage_intentional_source,
    create_narrative_two_stage_planning_source,
    create_narrative_two_stage_reactive_source,
)
from narrative_dynamics.attestation import RepositoryIdentity
from narrative_dynamics.candidate_execution import (
    CandidateExecutionError,
    ProcessCandidateExecutor,
    SequentialCandidateExecutor,
)
from narrative_dynamics.contracts import Scenario
from narrative_dynamics.measurement_validity import (
    MeasurementAuditCase,
    MeasurementPredictionArtifact,
)
from narrative_dynamics.observations import (
    ObservationDataset,
    ObservationPartition,
    ObservationPartitionRole,
)
from narrative_dynamics.observations.preregistration import FrozenModelSpec
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.studies.feher_hare_measurement_validity_v1 import (
    FeherHareMeasurementPredictionTask,
    execute_feher_hare_measurement_predictions,
)
from narrative_dynamics.studies.feher_hare_two_stage_v1 import (
    PreparedFeherHareTwoStageV1,
)

from tests.measurement_validity_fixtures import (
    measurement_cases,
    measurement_input,
    measurement_protocol,
)


_SELECTION_HASHES = {
    "reactive": (
        "sha256:e8414e301d19fc6acfd5accf055402ac995790f785402186c67ff95a48ee0c2e"
    ),
    "intentional": (
        "sha256:498a512cd931afe276f78bf135bd5c8269e051920d4b876177d0aa9ea250806d"
    ),
    "planning": (
        "sha256:e347a3d7d523e2a35d0bb6b0f662343f5a01ddd1ff7d2efe95a8a5efa1986e74"
    ),
}
_PARAMETERS = {
    "reactive": {"beta": 0.5},
    "intentional": {"beta": 2.0, "memory_decay": 0.5},
    "planning": {"beta": 4.0, "memory_decay": 0.75},
}


def _candidates() -> tuple[FrozenModelSpec, ...]:
    sources = {
        "reactive": create_narrative_two_stage_reactive_source(),
        "intentional": create_narrative_two_stage_intentional_source(),
        "planning": create_narrative_two_stage_planning_source(),
    }
    return tuple(
        FrozenModelSpec.freeze(
            name=family,
            model=sources[family],
            parameters=_PARAMETERS[family],
            selection_manifest_hash=_SELECTION_HASHES[family],
        )
        for family in ("reactive", "intentional", "planning")
    )


def _valid_cases() -> tuple[MeasurementAuditCase, ...]:
    rows = []
    for case in measurement_cases():
        payload = dict(case.scenario.payload)
        if case.task_variant == "spaceship":
            payload["first_stage_configuration"] = (
                ("symbol0", 0),
                ("symbol1", 1),
            )
        rows.append(
            replace(
                case,
                scenario=Scenario(id=case.scenario.id, payload=payload),
            )
        )
    return tuple(rows)


def _fixture(*, reversed_cases: bool = False):
    cases = _valid_cases()
    if reversed_cases:
        cases = tuple(reversed(cases))
    audit_input = measurement_input(cases=cases, candidates=_candidates())
    protocol = measurement_protocol(audit_input)
    identity = RepositoryIdentity(
        provider="test",
        repository="qigao/lean",
        checkout_commit="a" * 40,
        source_commit="a" * 40,
        ref="measurement-test",
        dirty=False,
    )
    return identity, audit_input, protocol


class AuditingExecutor:
    def __init__(self, *, reverse_results: bool = False) -> None:
        self.reverse_results = reverse_results
        self.requests = []
        self.initargs = []

    def execute(self, function, tasks, *, initializer=None, initargs=()):
        task_rows = tuple(tasks)
        arguments = tuple(initargs)
        self.requests.append(task_rows)
        self.initargs.append(arguments)
        if initializer is not None:
            initializer(*arguments)
        results = tuple(function(task) for task in task_rows)
        return tuple(reversed(results)) if self.reverse_results else results


class MissingRowExecutor(AuditingExecutor):
    def execute(self, function, tasks, *, initializer=None, initargs=()):
        rows = super().execute(
            function,
            tasks,
            initializer=initializer,
            initargs=initargs,
        )
        return rows[:-1]


class DuplicateRowExecutor(AuditingExecutor):
    def execute(self, function, tasks, *, initializer=None, initargs=()):
        rows = super().execute(
            function,
            tasks,
            initializer=initializer,
            initargs=initargs,
        )
        return rows + (rows[0],)


class ForgedFamilyExecutor(AuditingExecutor):
    def execute(self, function, tasks, *, initializer=None, initargs=()):
        rows = super().execute(
            function,
            tasks,
            initializer=initializer,
            initargs=initargs,
        )
        forged = dict(rows[0])
        forged["family"] = (
            "planning" if forged["family"] != "planning" else "reactive"
        )
        return (forged,) + rows[1:]


class MismatchedTaskExecutor(AuditingExecutor):
    def execute(self, function, tasks, *, initializer=None, initargs=()):
        task_rows = tuple(tasks)
        arguments = tuple(initargs)
        if initializer is not None:
            initializer(*arguments)
        wrong_family = (
            "planning" if task_rows[0].family != "planning" else "reactive"
        )
        return (function(replace(task_rows[0], family=wrong_family)),) + tuple(
            function(task) for task in task_rows[1:]
        )


class FailingExecutor:
    def execute(self, function, tasks, *, initializer=None, initargs=()):
        raise CandidateExecutionError("injected measurement executor failure")


def _contains_forbidden_worker_object(value: object) -> bool:
    if isinstance(
        value,
        (
            PreparedFeherHareTwoStageV1,
            ObservationDataset,
            ObservationPartition,
        ),
    ):
        return True
    if isinstance(value, dict):
        return any(_contains_forbidden_worker_object(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_forbidden_worker_object(item) for item in value)
    return False


class FeherHareMeasurementPredictionTests(unittest.TestCase):
    def test_task_contract_has_closed_family_case_seed_identity(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(FeherHareMeasurementPredictionTask)),
            ("family", "case_hash", "seed"),
        )
        _identity, audit_input, _protocol = _fixture()
        task = FeherHareMeasurementPredictionTask(
            family="reactive",
            case_hash=audit_input.cases[0].case_hash,
            seed=101,
        )
        self.assertEqual(task.family, "reactive")
        for family, seed in (("alien", 101), ("reactive", 301)):
            with self.subTest(family=family, seed=seed):
                with self.assertRaises((TypeError, ValueError)):
                    FeherHareMeasurementPredictionTask(
                        family=family,
                        case_hash=task.case_hash,
                        seed=seed,
                    )

    def test_selected_only_execution_uses_exact_role_seeds_and_run_count(self) -> None:
        identity, audit_input, protocol = _fixture()
        executor = AuditingExecutor()
        original_run_once = SimulationRunner.run_once
        run_calls = []

        def counted_run_once(runner, model, scenario, parameters, *, seed, cancellation=None):
            run_calls.append((getattr(model, "family", None), scenario.id, seed))
            return original_run_once(
                runner,
                model,
                scenario,
                parameters,
                seed=seed,
                cancellation=cancellation,
            )

        forbidden = (
            "narrative_dynamics.calibration._grid_candidates",
            "narrative_dynamics.observations.training.fit_training_target_grid",
            "narrative_dynamics.validation.select_on_validation_suite",
            "narrative_dynamics.uncertainty.ParameterAcceptanceSet.from_parameters",
            "narrative_dynamics.observations.preregistration.FrozenModelSpec.from_selection",
            "narrative_dynamics.external_prediction.predict_external_final_once",
        )
        with ExitStack() as stack:
            mocks = tuple(
                stack.enter_context(
                    patch(path, side_effect=AssertionError(f"forbidden call: {path}"))
                )
                for path in forbidden
            )
            stack.enter_context(
                patch.object(SimulationRunner, "run_once", new=counted_run_once)
            )
            artifact = execute_feher_hare_measurement_predictions(
                identity,
                audit_input,
                protocol,
                executor,
            )
        for mocked in mocks:
            mocked.assert_not_called()

        self.assertIsInstance(artifact, MeasurementPredictionArtifact)
        self.assertEqual(
            tuple(model.model_name for model in artifact.models),
            ("reactive", "intentional", "planning"),
        )
        self.assertEqual(len(executor.requests), 3)
        self.assertEqual(
            len(run_calls),
            3 * len(audit_input.cases) * 2,
        )
        self.assertTrue({301, 302}.isdisjoint({seed for _, _, seed in run_calls}))
        case_by_hash = {case.case_hash: case for case in audit_input.cases}
        seeds_by_role = dict(protocol.seeds_by_role)
        for family_tasks in executor.requests:
            self.assertEqual(len({task.family for task in family_tasks}), 1)
            self.assertEqual(
                {(task.case_hash, task.seed) for task in family_tasks},
                {
                    (case.case_hash, seed)
                    for case in audit_input.cases
                    for seed in seeds_by_role[case.role]
                },
            )
            self.assertTrue(
                all(
                    task.seed in seeds_by_role[case_by_hash[task.case_hash].role]
                    for task in family_tasks
                )
            )
        self.assertFalse(
            any(
                _contains_forbidden_worker_object(arguments)
                for arguments in executor.initargs
            )
        )

    def test_case_and_batch_order_do_not_change_artifact_hash(self) -> None:
        identity, audit_input, protocol = _fixture()
        canonical = execute_feher_hare_measurement_predictions(
            identity,
            audit_input,
            protocol,
            SequentialCandidateExecutor(),
        )
        reversed_identity, reversed_input, reversed_protocol = _fixture(
            reversed_cases=True
        )
        regrouped = execute_feher_hare_measurement_predictions(
            reversed_identity,
            reversed_input,
            reversed_protocol,
            AuditingExecutor(reverse_results=True),
        )
        self.assertEqual(audit_input.content_hash, reversed_input.content_hash)
        self.assertEqual(protocol.content_hash, reversed_protocol.content_hash)
        self.assertEqual(canonical.content_hash, regrouped.content_hash)
        self.assertEqual(canonical, regrouped)

    def test_spawn_process_and_sequential_execution_match_exactly(self) -> None:
        identity, audit_input, protocol = _fixture()
        sequential = execute_feher_hare_measurement_predictions(
            identity,
            audit_input,
            protocol,
            SequentialCandidateExecutor(),
        )
        process = execute_feher_hare_measurement_predictions(
            identity,
            audit_input,
            protocol,
            ProcessCandidateExecutor(max_workers=2),
        )
        self.assertEqual(sequential.content_hash, process.content_hash)
        self.assertEqual(sequential, process)

    def test_worker_and_executor_failures_are_atomic(self) -> None:
        identity, audit_input, protocol = _fixture()
        invalid_executors = (
            (ForgedFamilyExecutor(), "family"),
            (MissingRowExecutor(), "coverage"),
            (DuplicateRowExecutor(), "duplicate"),
            (MismatchedTaskExecutor(), "family"),
            (FailingExecutor(), "injected"),
        )
        for executor, message in invalid_executors:
            with self.subTest(executor=type(executor).__name__):
                with self.assertRaisesRegex((ValueError, CandidateExecutionError), message):
                    execute_feher_hare_measurement_predictions(
                        identity,
                        audit_input,
                        protocol,
                        executor,
                    )


if __name__ == "__main__":
    unittest.main()

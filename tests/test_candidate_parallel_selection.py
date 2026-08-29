from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import unittest

from narrative_dynamics.candidate_execution import (
    ProcessCandidateExecutor,
    SequentialCandidateExecutor,
)
from narrative_dynamics.contracts import ModelRun, Scenario, stable_content_hash
from narrative_dynamics.losses import DEFAULT_METRIC_LOSS
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.uncertainty import ParameterAcceptanceSet
from narrative_dynamics.validation import (
    EvaluationRole,
    HeldOutCase,
    HeldOutSuite,
    select_on_validation_suite,
)


_IMPORT_ERROR: Exception | None = None
try:
    from narrative_dynamics.observations.selection_shards import (
        SelectionCandidateShard,
        assemble_selection_validation_report,
        evaluate_selection_candidate,
    )
except Exception as error:
    _IMPORT_ERROR = error


class ScaledLevelModel:
    name = "candidate-parallel-selection-scaled-level"

    def simulate(self, scenario, parameters, rng):
        return ModelRun(
            events=(),
            outcome={
                "value": float(parameters["level"])
                * float(scenario.payload["scale"])
            },
        )


@dataclass(frozen=True)
class SelectionShardTask:
    parameters: tuple[tuple[str, float], ...]


class ReverseResultCandidateExecutor:
    def execute(self, function, tasks, *, initializer=None, initargs=()):
        if initializer is not None:
            initializer(*tuple(initargs))
        results = tuple(function(task) for task in tuple(tasks))
        return tuple(reversed(results))


def value_metrics(trace):
    return {"value": float(trace.outcome["value"])}


def _fixture():
    suite = HeldOutSuite(
        name="candidate-parallel-selection",
        role=EvaluationRole.SELECTION_VALIDATION,
        cases=(
            HeldOutCase(
                scenario=Scenario(id="selection-scale-2", payload={"scale": 2.0}),
                seeds=(201, 202),
                target={"value": 4.0},
            ),
            HeldOutCase(
                scenario=Scenario(id="selection-scale-1", payload={"scale": 1.0}),
                seeds=(201, 202),
                target={"value": 2.0},
            ),
        ),
    )
    accepted = ParameterAcceptanceSet.from_parameters(
        (
            (("level", 1.0),),
            (("level", 2.0),),
            (("level", 3.0),),
        ),
        source_manifest_hashes=(stable_content_hash({"source": "selection-red"}),),
    )
    return suite, accepted


def evaluate_selection_shard_task(task: SelectionShardTask):
    suite, accepted = _fixture()
    return evaluate_selection_candidate(
        runner=SimulationRunner(),
        model=ScaledLevelModel(),
        accepted_parameter_set=accepted,
        parameters=task.parameters,
        suite=suite,
        extractor=value_metrics,
        loss=DEFAULT_METRIC_LOSS,
    ).to_payload()


def _reference():
    suite, accepted = _fixture()
    return select_on_validation_suite(
        runner=SimulationRunner(),
        model=ScaledLevelModel(),
        accepted_parameters=accepted,
        suite=suite,
        extractor=value_metrics,
        loss=DEFAULT_METRIC_LOSS,
    )


def _shards():
    suite, accepted = _fixture()
    model = ScaledLevelModel()
    shards = tuple(
        evaluate_selection_candidate(
            runner=SimulationRunner(),
            model=model,
            accepted_parameter_set=accepted,
            parameters=parameters,
            suite=suite,
            extractor=value_metrics,
            loss=DEFAULT_METRIC_LOSS,
        )
        for parameters in accepted.parameters
    )
    return suite, accepted, model, shards


def _assemble_with_executor(executor):
    suite, accepted = _fixture()
    model = ScaledLevelModel()
    tasks = tuple(
        SelectionShardTask(parameters=parameters)
        for parameters in accepted.parameters
    )
    payloads = executor.execute(evaluate_selection_shard_task, tasks)
    shards = tuple(
        SelectionCandidateShard.from_payload(payload)
        for payload in payloads
    )
    return assemble_selection_validation_report(
        model=model,
        accepted_parameters=accepted,
        suite=suite,
        extractor=value_metrics,
        loss=DEFAULT_METRIC_LOSS,
        shards=shards,
    )


@unittest.skipIf(_IMPORT_ERROR is not None, "selection shard API is not implemented yet")
class CandidateParallelSelectionContractTests(unittest.TestCase):
    def test_candidate_shards_reassemble_exact_reference_report(self):
        suite, accepted, model, shards = _shards()
        assembled = assemble_selection_validation_report(
            model=model,
            accepted_parameters=accepted,
            suite=suite,
            extractor=value_metrics,
            loss=DEFAULT_METRIC_LOSS,
            shards=tuple(reversed(shards)),
        )
        reference = _reference()

        self.assertEqual(assembled.candidate_report, reference.candidate_report)
        self.assertEqual(assembled.selected_parameters, reference.selected_parameters)
        self.assertEqual(assembled.manifest, reference.manifest)
        self.assertEqual(assembled.manifest.content_hash, reference.manifest.content_hash)

    def test_shard_payload_round_trip_recomputes_declared_hash(self):
        _, _, _, shards = _shards()
        shard = shards[0]
        payload = shard.to_payload()

        self.assertEqual(SelectionCandidateShard.from_payload(payload), shard)
        self.assertEqual(payload["content_hash"], shard.content_hash)

        tampered = deepcopy(payload)
        tampered["parameters"] = [["level", 2.0]]
        with self.assertRaises(ValueError):
            SelectionCandidateShard.from_payload(tampered)

    def test_assembly_rejects_missing_and_duplicate_candidates(self):
        suite, accepted, model, shards = _shards()

        def assemble(values):
            return assemble_selection_validation_report(
                model=model,
                accepted_parameters=accepted,
                suite=suite,
                extractor=value_metrics,
                loss=DEFAULT_METRIC_LOSS,
                shards=values,
            )

        with self.assertRaises(ValueError):
            assemble(shards[:-1])
        with self.assertRaises(ValueError):
            assemble(shards + (shards[0],))

    def test_assembly_rejects_forged_selection_shard_fields_and_validation_lineage(self):
        suite, accepted, model, shards = _shards()
        first = shards[0]
        first_case = first.cases[0]
        forged_hash = stable_content_hash({"forged": "selection-shard"})
        forged_run_hash = stable_content_hash({"forged": "selection-run"})

        def assemble(values):
            return assemble_selection_validation_report(
                model=model,
                accepted_parameters=accepted,
                suite=suite,
                extractor=value_metrics,
                loss=DEFAULT_METRIC_LOSS,
                shards=values,
            )

        mutations = (
            (
                "accepted-set",
                lambda: replace(first, accepted_parameter_set_hash=forged_hash),
            ),
            (
                "suite-hash",
                lambda: replace(first, suite_hash=forged_hash),
            ),
            (
                "suite-name",
                lambda: replace(first, suite_name="forged-selection-suite"),
            ),
            (
                "repository-identity",
                lambda: replace(
                    first,
                    repository_identity={"forged": "repository"},
                ),
            ),
            (
                "model-identity",
                lambda: replace(first, model_identity={"forged": "model"}),
            ),
            (
                "parameters",
                lambda: replace(first, parameters=(("level", 9.0),)),
            ),
            (
                "case-coverage",
                lambda: replace(
                    first,
                    cases=(replace(first_case, name="forged-case"),) + first.cases[1:],
                ),
            ),
            (
                "validation-manifest-lineage",
                lambda: replace(
                    first,
                    cases=(
                        replace(
                            first_case,
                            run_manifest_hashes=(
                                forged_run_hash,
                                first_case.run_manifest_hashes[1],
                            ),
                        ),
                    )
                    + first.cases[1:],
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(forgery=label):
                with self.assertRaises(ValueError):
                    forged = mutate()
                    assemble((forged,) + shards[1:])

    def test_executor_backends_preserve_exact_selection_report(self):
        reference = _reference()
        executors = (
            ("sequential", SequentialCandidateExecutor()),
            ("process-1", ProcessCandidateExecutor(max_workers=1)),
            ("process-2", ProcessCandidateExecutor(max_workers=2)),
        )
        for name, executor in executors:
            with self.subTest(executor=name):
                self.assertEqual(_assemble_with_executor(executor), reference)

    def test_reverse_completion_order_preserves_exact_selection_report(self):
        self.assertEqual(
            _assemble_with_executor(ReverseResultCandidateExecutor()),
            _reference(),
        )


class CandidateParallelSelectionRedTests(unittest.TestCase):
    def test_selection_candidate_shard_api_exists(self):
        self.assertIsNone(
            _IMPORT_ERROR,
            f"candidate-parallel selection shard API is missing: {_IMPORT_ERROR}",
        )


if __name__ == "__main__":
    unittest.main()

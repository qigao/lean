from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import tempfile
import unittest

from narrative_dynamics.attestation import RepositoryIdentity
from narrative_dynamics.candidate_execution import (
    ProcessCandidateExecutor,
    SequentialCandidateExecutor,
)
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.studies import feher_hare_two_stage_v1 as study
from narrative_dynamics.studies.two_stage_source import TwoStageSourceManifest
from tests.test_two_stage_source import _files
from tests.two_stage_test_support import build_synthetic_two_stage_checkout


_PARALLEL_IMPORT_ERROR: Exception | None = None
try:
    from narrative_dynamics.studies.feher_hare_two_stage_parallel import (
        FeherHareCandidateTask,
        fit_and_freeze_feher_hare_models_parallel,
    )
    from narrative_dynamics.studies import (
        fit_and_freeze_feher_hare_models_parallel as exported_parallel_freeze,
    )
except Exception as error:
    _PARALLEL_IMPORT_ERROR = error


class AuditingCandidateExecutor:
    def __init__(self) -> None:
        self.requests: list[tuple[str, tuple[int, ...]]] = []

    def execute(self, function, tasks, *, initializer=None, initargs=()):
        if initializer is not None:
            initializer(*tuple(initargs))
        results = []
        for task in tuple(tasks):
            phase = getattr(task.phase, "value", task.phase)
            seeds = tuple(task.simulation_seeds)
            self.requests.append((phase, seeds))
            results.append(function(task))
        return tuple(results)


def _fixture(tmp: str):
    root, revision = build_synthetic_two_stage_checkout(
        Path(tmp),
        magic_n=6,
        spaceship_n=6,
    )
    manifest = TwoStageSourceManifest(
        name="synthetic-feher-hare-parallel",
        version="1",
        repository="test/synthetic-two-stage",
        revision=revision,
        license_reference="test-only",
        files=_files(root),
    )
    prepared = study.prepare_feher_hare_two_stage_v1(
        root=root,
        manifest=manifest,
    )
    identity = RepositoryIdentity(
        provider="test",
        repository="qigao/lean",
        checkout_commit="a" * 40,
        source_commit="a" * 40,
        ref="test",
        dirty=False,
    )
    return root, manifest, prepared, identity


def _assert_exact_freeze(testcase, actual, expected):
    testcase.assertEqual(actual.frozen_candidates, expected.frozen_candidates)
    testcase.assertEqual(
        actual.training_manifest_hashes,
        expected.training_manifest_hashes,
    )
    testcase.assertEqual(
        actual.selection_manifest_hashes,
        expected.selection_manifest_hashes,
    )


class FeherHareParallelTrainingRedTests(unittest.TestCase):
    def test_parallel_study_api_exists(self):
        self.assertIsNone(
            _PARALLEL_IMPORT_ERROR,
            f"Feher/Hare candidate-parallel study API is missing: {_PARALLEL_IMPORT_ERROR}",
        )


@unittest.skipIf(
    _PARALLEL_IMPORT_ERROR is not None,
    "Feher/Hare candidate-parallel study API is not implemented yet",
)
class FeherHareParallelTrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.root, cls.manifest, cls.prepared, cls.identity = _fixture(
            cls._temporary_directory.name
        )
        cls.reference = study.fit_and_freeze_feher_hare_models(
            runner=SimulationRunner(repository_identity=cls.identity),
            prepared=cls.prepared,
        )

    @classmethod
    def tearDownClass(cls):
        cls._temporary_directory.cleanup()

    def test_candidate_task_has_closed_phase_and_no_seed_payload(self):
        self.assertIs(exported_parallel_freeze, fit_and_freeze_feher_hare_models_parallel)
        field_names = {field.name for field in fields(FeherHareCandidateTask)}
        self.assertNotIn("seeds", field_names)
        self.assertNotIn("simulation_seeds", field_names)

        training = FeherHareCandidateTask(
            family="reactive",
            phase="train",
            parameters=(("beta", 0.5),),
        )
        selection = FeherHareCandidateTask(
            family="reactive",
            phase="selection_validation",
            parameters=(("beta", 0.5),),
        )
        self.assertEqual(tuple(training.simulation_seeds), study.TRAIN_SEEDS)
        self.assertEqual(tuple(selection.simulation_seeds), study.SELECTION_SEEDS)

        with self.assertRaises((TypeError, ValueError)):
            FeherHareCandidateTask(
                family="reactive",
                phase="final_test",
                parameters=(("beta", 0.5),),
            )

    def test_sequential_wrapper_matches_exact_reference(self):
        parallel = fit_and_freeze_feher_hare_models_parallel(
            root=self.root,
            manifest=self.manifest,
            repository_identity=self.identity,
            executor=SequentialCandidateExecutor(),
        )
        _assert_exact_freeze(self, parallel, self.reference)

    def test_candidate_executor_seed_firewall_excludes_final(self):
        executor = AuditingCandidateExecutor()
        parallel = fit_and_freeze_feher_hare_models_parallel(
            root=self.root,
            manifest=self.manifest,
            repository_identity=self.identity,
            executor=executor,
        )
        _assert_exact_freeze(self, parallel, self.reference)

        phases = {phase for phase, _ in executor.requests}
        self.assertEqual(phases, {"train", "selection_validation"})
        training_seed_set = {
            seed
            for phase, seeds in executor.requests
            if phase == "train"
            for seed in seeds
        }
        selection_seed_set = {
            seed
            for phase, seeds in executor.requests
            if phase == "selection_validation"
            for seed in seeds
        }
        all_requested_seeds = {
            seed for _, seeds in executor.requests for seed in seeds
        }
        self.assertEqual(training_seed_set, {101, 102})
        self.assertEqual(selection_seed_set, {201, 202})
        self.assertTrue({301, 302}.isdisjoint(all_requested_seeds))

    def test_spawn_process_wrapper_matches_exact_reference(self):
        parallel = fit_and_freeze_feher_hare_models_parallel(
            root=self.root,
            manifest=self.manifest,
            repository_identity=self.identity,
            executor=ProcessCandidateExecutor(max_workers=2),
        )
        _assert_exact_freeze(self, parallel, self.reference)


if __name__ == "__main__":
    unittest.main()

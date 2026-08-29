from __future__ import annotations

import os
import unittest


_IMPORT_ERROR: Exception | None = None
try:
    from narrative_dynamics.candidate_execution import (
        CandidateExecutionError,
        ProcessCandidateExecutor,
        SequentialCandidateExecutor,
    )
except Exception as error:
    _IMPORT_ERROR = error


_INITIALIZED_VALUE = None


def process_probe(value: int) -> tuple[int, int]:
    return value * value, os.getpid()


def failing_probe(value: int) -> int:
    if value == 2:
        raise ValueError("candidate boom")
    return value


def set_initializer(value: str) -> None:
    global _INITIALIZED_VALUE
    _INITIALIZED_VALUE = value


def initialized_probe(value: int) -> tuple[str | None, int]:
    return _INITIALIZED_VALUE, value


@unittest.skipIf(_IMPORT_ERROR is not None, "candidate executor API is not implemented yet")
class CandidateProcessExecutorContractTests(unittest.TestCase):
    def test_sequential_and_process_results_preserve_task_order(self):
        tasks = (3, 1, 2)
        sequential = SequentialCandidateExecutor().execute(process_probe, tasks)
        process = ProcessCandidateExecutor(max_workers=2).execute(process_probe, tasks)

        self.assertEqual(tuple(value for value, _pid in sequential), (9, 1, 4))
        self.assertEqual(tuple(value for value, _pid in process), (9, 1, 4))
        self.assertTrue(all(pid == os.getpid() for _value, pid in sequential))
        self.assertTrue(all(pid != os.getpid() for _value, pid in process))

    def test_one_worker_is_valid_and_nonpositive_worker_counts_are_rejected(self):
        result = ProcessCandidateExecutor(max_workers=1).execute(process_probe, (2,))
        self.assertEqual(result[0][0], 4)
        self.assertNotEqual(result[0][1], os.getpid())

        for invalid in (0, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    ProcessCandidateExecutor(max_workers=invalid)

    def test_worker_exception_is_typed_and_returns_no_partial_result(self):
        for executor in (
            SequentialCandidateExecutor(),
            ProcessCandidateExecutor(max_workers=2),
        ):
            with self.subTest(executor=type(executor).__name__):
                with self.assertRaises(CandidateExecutionError) as captured:
                    executor.execute(failing_probe, (1, 2, 3))
                self.assertIsInstance(captured.exception.__cause__, ValueError)

    def test_initializer_and_initargs_are_applied_by_both_backends(self):
        for executor in (
            SequentialCandidateExecutor(),
            ProcessCandidateExecutor(max_workers=2),
        ):
            with self.subTest(executor=type(executor).__name__):
                result = executor.execute(
                    initialized_probe,
                    (1, 2, 3),
                    initializer=set_initializer,
                    initargs=("ready",),
                )
                self.assertEqual(result, (("ready", 1), ("ready", 2), ("ready", 3)))


class CandidateProcessExecutorRedTests(unittest.TestCase):
    def test_candidate_executor_api_exists(self):
        self.assertIsNone(
            _IMPORT_ERROR,
            f"candidate process executor API is missing: {_IMPORT_ERROR}",
        )


if __name__ == "__main__":
    unittest.main()

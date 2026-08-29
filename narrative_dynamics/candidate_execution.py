from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
import os
from typing import Protocol, TypeVar


_Task = TypeVar("_Task")
_Result = TypeVar("_Result")


class CandidateExecutionError(RuntimeError):
    """One candidate executor invocation failed before producing a complete result."""


class CandidateExecutor(Protocol):
    """Operational scheduler for independent candidate tasks."""

    def execute(
        self,
        function: Callable[[_Task], _Result],
        tasks: Iterable[_Task],
        *,
        initializer: Callable[..., object] | None = None,
        initargs: Iterable[object] = (),
    ) -> tuple[_Result, ...]: ...


def _validate_initializer(
    initializer: Callable[..., object] | None,
    initargs: Iterable[object],
) -> tuple[Callable[..., object] | None, tuple[object, ...]]:
    if initializer is not None and not callable(initializer):
        raise TypeError("candidate executor initializer must be callable or None")
    arguments = tuple(initargs)
    if initializer is None and arguments:
        raise ValueError("candidate executor initargs require an initializer")
    return initializer, arguments


def _execute_sequential(
    function: Callable[[_Task], _Result],
    tasks: tuple[_Task, ...],
    *,
    initializer: Callable[..., object] | None,
    initargs: tuple[object, ...],
) -> tuple[_Result, ...]:
    if initializer is not None:
        initializer(*initargs)
    return tuple(function(task) for task in tasks)


@dataclass(frozen=True)
class SequentialCandidateExecutor:
    """Deterministic in-process executor with the candidate-executor contract."""

    def execute(
        self,
        function: Callable[[_Task], _Result],
        tasks: Iterable[_Task],
        *,
        initializer: Callable[..., object] | None = None,
        initargs: Iterable[object] = (),
    ) -> tuple[_Result, ...]:
        if not callable(function):
            raise TypeError("candidate executor function must be callable")
        task_values = tuple(tasks)
        selected_initializer, initializer_args = _validate_initializer(
            initializer,
            initargs,
        )
        try:
            return _execute_sequential(
                function,
                task_values,
                initializer=selected_initializer,
                initargs=initializer_args,
            )
        except Exception as error:
            raise CandidateExecutionError(
                "sequential candidate execution failed"
            ) from error


@dataclass(frozen=True)
class ProcessCandidateExecutor:
    """Bounded spawn-process executor preserving declared candidate order."""

    max_workers: int | None = None

    def __post_init__(self) -> None:
        if self.max_workers is None:
            return
        if (
            not isinstance(self.max_workers, int)
            or isinstance(self.max_workers, bool)
            or self.max_workers <= 0
        ):
            raise ValueError("candidate process executor max_workers must be positive")

    def execute(
        self,
        function: Callable[[_Task], _Result],
        tasks: Iterable[_Task],
        *,
        initializer: Callable[..., object] | None = None,
        initargs: Iterable[object] = (),
    ) -> tuple[_Result, ...]:
        if not callable(function):
            raise TypeError("candidate executor function must be callable")
        task_values = tuple(tasks)
        if not task_values:
            return ()
        selected_initializer, initializer_args = _validate_initializer(
            initializer,
            initargs,
        )
        available_workers = os.cpu_count() or 1
        requested_workers = (
            available_workers if self.max_workers is None else self.max_workers
        )
        effective_workers = min(
            requested_workers,
            available_workers,
            len(task_values),
        )
        context = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(
                max_workers=effective_workers,
                mp_context=context,
                initializer=selected_initializer,
                initargs=initializer_args,
            ) as executor:
                futures = tuple(
                    executor.submit(function, task)
                    for task in task_values
                )
                try:
                    return tuple(future.result() for future in futures)
                except Exception:
                    for future in futures:
                        future.cancel()
                    raise
        except Exception as error:
            raise CandidateExecutionError(
                "process candidate execution failed"
            ) from error


__all__ = [
    "CandidateExecutionError",
    "CandidateExecutor",
    "ProcessCandidateExecutor",
    "SequentialCandidateExecutor",
]

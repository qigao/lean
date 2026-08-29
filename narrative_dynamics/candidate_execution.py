from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TypeVar


_Task = TypeVar("_Task")
_Result = TypeVar("_Result")


class CandidateExecutionError(RuntimeError):
    """One candidate executor invocation failed before producing a complete result."""


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
    """Bounded process executor that preserves the declared candidate order."""

    max_workers: int

    def __post_init__(self) -> None:
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
        selected_initializer, initializer_args = _validate_initializer(
            initializer,
            initargs,
        )
        try:
            with ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=selected_initializer,
                initargs=initializer_args,
            ) as executor:
                return tuple(executor.map(function, task_values))
        except Exception as error:
            raise CandidateExecutionError(
                "process candidate execution failed"
            ) from error


__all__ = [
    "CandidateExecutionError",
    "ProcessCandidateExecutor",
    "SequentialCandidateExecutor",
]

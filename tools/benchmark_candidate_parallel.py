from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from time import perf_counter


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics
from narrative_dynamics.attestation import RepositoryIdentity
from narrative_dynamics.candidate_execution import (
    CandidateExecutor,
    ProcessCandidateExecutor,
    SequentialCandidateExecutor,
)
from narrative_dynamics.observations.dataset import ObservationPartitionRole
from narrative_dynamics.observations.training import fit_training_target_grid
from narrative_dynamics.observations.training_shards import (
    TrainingCandidateShard,
    assemble_training_fit_report,
    evaluate_training_candidate,
)
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.uncertainty import ParameterTuple
from tests.test_observational_training_fit import (
    TrainingProbabilityModel,
    brier_loss,
    targets_for,
)


_P_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_CANDIDATES: tuple[ParameterTuple, ...] = tuple(
    (("p", value),) for value in _P_VALUES
)
_SEEDS = (101, 102)
_CPU_ROUNDS = 600_000
_REPOSITORY_IDENTITY = RepositoryIdentity(
    provider="benchmark",
    repository="qigao/lean",
    checkout_commit="d" * 40,
    source_commit="d" * 40,
    ref="candidate-parallel-v2-benchmark",
    dirty=False,
)


@dataclass(frozen=True)
class BenchmarkCandidateTask:
    parameters: ParameterTuple


def _cpu_burn(parameters: ParameterTuple) -> int:
    """Deterministic synthetic CPU work that never enters scientific identity."""

    accumulator = int(parameters[0][1] * 1_000_000.0) & 0xFFFFFFFF
    for index in range(_CPU_ROUNDS):
        accumulator = (
            ((accumulator * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF)
            ^ ((index * 2_654_435_761) & 0xFFFFFFFF)
        )
    return accumulator


def evaluate_benchmark_candidate(task: BenchmarkCandidateTask) -> dict[str, object]:
    checksum = _cpu_burn(task.parameters)
    if checksum < 0:  # pragma: no cover - keeps the synthetic work observable.
        raise AssertionError("unreachable benchmark checksum")
    return evaluate_training_candidate(
        runner=SimulationRunner(repository_identity=_REPOSITORY_IDENTITY),
        model=TrainingProbabilityModel(),
        target_report=targets_for(ObservationPartitionRole.TRAIN),
        parameters=task.parameters,
        simulation_seeds=_SEEDS,
        extractor=prison_initial_action_metrics,
        loss=brier_loss(),
    ).to_payload()


def _reference_report():
    return fit_training_target_grid(
        runner=SimulationRunner(repository_identity=_REPOSITORY_IDENTITY),
        model=TrainingProbabilityModel(),
        target_report=targets_for(ObservationPartitionRole.TRAIN),
        parameter_grid={"p": _P_VALUES},
        simulation_seeds=_SEEDS,
        extractor=prison_initial_action_metrics,
        loss=brier_loss(),
    )


def _timed_report(executor: CandidateExecutor):
    tasks = tuple(BenchmarkCandidateTask(parameters) for parameters in _CANDIDATES)
    started = perf_counter()
    payloads = executor.execute(evaluate_benchmark_candidate, tasks)
    shards = tuple(
        TrainingCandidateShard.from_payload(payload)
        for payload in payloads
    )
    report = assemble_training_fit_report(
        model=TrainingProbabilityModel(),
        target_report=targets_for(ObservationPartitionRole.TRAIN),
        parameter_candidates=_CANDIDATES,
        simulation_seeds=_SEEDS,
        extractor=prison_initial_action_metrics,
        loss=brier_loss(),
        shards=shards,
    )
    return report, perf_counter() - started


def main() -> int:
    reference = _reference_report()
    max_four_workers = min(4, os.cpu_count() or 1)
    backends = (
        ("sequential", SequentialCandidateExecutor()),
        ("process-2", ProcessCandidateExecutor(max_workers=2)),
        (
            f"process-{max_four_workers}",
            ProcessCandidateExecutor(max_workers=max_four_workers),
        ),
    )

    results = tuple(
        (label, *_timed_report(executor))
        for label, executor in backends
    )

    reference_hash = reference.manifest.content_hash
    for label, report, _elapsed in results:
        if report != reference:
            raise AssertionError(
                f"{label} benchmark report differs from sequential scientific reference"
            )
        if report.manifest.content_hash != reference_hash:
            raise AssertionError(
                f"{label} benchmark manifest hash differs from sequential reference"
            )

    sequential_elapsed = results[0][2]
    print("Candidate-parallel V2 synthetic CPU benchmark (non-authoritative)")
    print(f"scientific_manifest_hash={reference_hash}")
    print(f"cpu_count={os.cpu_count() or 1} cpu_rounds_per_candidate={_CPU_ROUNDS}")
    for label, _report, elapsed in results:
        speedup = sequential_elapsed / elapsed if elapsed > 0.0 else float("inf")
        print(f"{label}: elapsed_seconds={elapsed:.6f} speedup_vs_sequential={speedup:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

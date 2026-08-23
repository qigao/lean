from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import fmean

from narrative_dynamics.calibration import calibrate_grid
from narrative_dynamics.contracts import ExperimentManifest, ExperimentStage
from narrative_dynamics.losses import MetricLoss, metric_loss_identity
from narrative_dynamics.manifest import (
    callable_identity,
    component_identity,
    required_manifest_hash,
    scenario_identity,
    stable_content_hash,
)
from narrative_dynamics.simulation import ModelSource, SimulationRunner

from .dataset import ObservationPartitionRole
from .targets import TargetConstructionReport


ParameterTuple = tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class TrainingCaseFit:
    name: str
    loss: float
    calibration_manifest_hash: str
    run_manifest_hashes: tuple[str, ...]


@dataclass(frozen=True)
class TrainingCandidateFit:
    parameters: ParameterTuple
    cases: tuple[TrainingCaseFit, ...]
    mean_loss: float
    worst_loss: float

    @property
    def parameter_map(self) -> dict[str, float]:
        return dict(self.parameters)


@dataclass(frozen=True)
class TrainingFitReport:
    target_report_hash: str
    ranking: tuple[TrainingCandidateFit, ...]
    manifest: ExperimentManifest

    @property
    def best(self) -> TrainingCandidateFit:
        return self.ranking[0]

    @property
    def candidate_parameters(self) -> tuple[ParameterTuple, ...]:
        return tuple(candidate.parameters for candidate in self.ranking)


def _materialize_grid(
    parameter_grid: Mapping[str, Iterable[float]],
) -> dict[str, tuple[float, ...]]:
    if not isinstance(parameter_grid, Mapping):
        raise TypeError("training parameter grid must be a mapping")
    return {name: tuple(values) for name, values in parameter_grid.items()}


def fit_training_target_grid(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    target_report: TargetConstructionReport,
    parameter_grid: Mapping[str, Iterable[float]],
    simulation_seeds: Iterable[int],
    extractor: object,
    loss: MetricLoss,
) -> TrainingFitReport:
    if not isinstance(target_report, TargetConstructionReport):
        raise TypeError("training target fit requires TargetConstructionReport")
    if target_report.role is not ObservationPartitionRole.TRAIN:
        raise ValueError("training target fit requires train targets")

    seeds = tuple(simulation_seeds)
    if not seeds:
        raise ValueError("training target fit requires simulation seeds")
    replayable_grid = _materialize_grid(parameter_grid)

    calibrations = tuple(
        calibrate_grid(
            runner=runner,
            model=model,
            scenario=case.scenario,
            parameter_grid=replayable_grid,
            seeds=seeds,
            extractor=extractor,
            target=case.target_map,
            loss=loss,
        )
        for case in target_report.cases
    )
    if not calibrations:
        raise ValueError("training target fit requires at least one train case")

    candidate_sets = tuple(
        {candidate.parameters for candidate in calibration.ranking}
        for calibration in calibrations
    )
    first_candidates = candidate_sets[0]
    if any(candidate_set != first_candidates for candidate_set in candidate_sets[1:]):
        raise RuntimeError("training target fit changed its finite candidate set")
    canonical_candidates = tuple(sorted(first_candidates))

    calibration_hashes = tuple(
        required_manifest_hash(calibration, label="training calibration")
        for calibration in calibrations
    )
    ranking_entries: list[TrainingCandidateFit] = []
    for parameters in canonical_candidates:
        case_fits: list[TrainingCaseFit] = []
        for case, calibration, calibration_hash in zip(
            target_report.cases,
            calibrations,
            calibration_hashes,
            strict=True,
        ):
            candidate = next(
                item for item in calibration.ranking if item.parameters == parameters
            )
            case_fits.append(
                TrainingCaseFit(
                    name=case.name,
                    loss=candidate.loss,
                    calibration_manifest_hash=calibration_hash,
                    run_manifest_hashes=candidate.run_manifest_hashes,
                )
            )
        losses = tuple(case_fit.loss for case_fit in case_fits)
        ranking_entries.append(
            TrainingCandidateFit(
                parameters=parameters,
                cases=tuple(case_fits),
                mean_loss=fmean(losses),
                worst_loss=max(losses),
            )
        )

    ranking = tuple(
        sorted(
            ranking_entries,
            key=lambda candidate: (
                candidate.mean_loss,
                candidate.worst_loss,
                candidate.parameters,
            ),
        )
    )
    manifest = ExperimentManifest(
        stage=ExperimentStage.TRAINING_TARGET_FIT,
        inputs={
            "model": component_identity(model),
            "target_report_hash": target_report.content_hash,
            "simulation_seeds": seeds,
            "metric": callable_identity(extractor),
            "loss": metric_loss_identity(loss),
            "parameter_candidates": canonical_candidates,
            "cases": tuple({
                "name": case.name,
                "scenario": scenario_identity(case.scenario),
                "target_hash": stable_content_hash(case.target_map),
            } for case in target_report.cases),
        },
        parent_hashes=(target_report.manifest.content_hash,) + calibration_hashes,
    )
    return TrainingFitReport(
        target_report_hash=target_report.content_hash,
        ranking=ranking,
        manifest=manifest,
    )


__all__ = [
    "TrainingCaseFit",
    "TrainingCandidateFit",
    "TrainingFitReport",
    "fit_training_target_grid",
]

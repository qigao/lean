from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from narrative_dynamics.adapters.narrative_two_stage import NarrativeTwoStageModelSource
from narrative_dynamics.adapters.two_stage_metrics import (
    two_stage_brier_loss,
    two_stage_first_stage_policy_metrics,
)
from narrative_dynamics.attestation import RepositoryIdentity
from narrative_dynamics.calibration import _grid_candidates
from narrative_dynamics.candidate_execution import CandidateExecutor
from narrative_dynamics.observations.preregistration import FrozenModelSpec
from narrative_dynamics.observations.selection_shards import (
    SelectionCandidateShard,
    assemble_selection_validation_report,
    evaluate_selection_candidate,
)
from narrative_dynamics.observations.training_shards import (
    TrainingCandidateShard,
    assemble_training_fit_report,
    evaluate_training_candidate,
)
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.uncertainty import (
    ParameterAcceptanceSet,
    ParameterTuple,
    _canonical_parameter_tuple,
)
from narrative_dynamics.validation import HeldOutSuite

from . import feher_hare_two_stage_v1 as study
from .two_stage_source import TwoStageSourceManifest


class _CandidatePhase(str, Enum):
    TRAIN = "train"
    SELECTION_VALIDATION = "selection_validation"


@dataclass(frozen=True)
class FeherHareCandidateTask:
    """One closed TRAIN or SELECTION candidate evaluation request."""

    family: str
    phase: _CandidatePhase | str
    parameters: ParameterTuple

    def __post_init__(self) -> None:
        if self.family not in study._FAMILIES:
            raise ValueError("Feher/Hare candidate task family is unsupported")
        try:
            phase = (
                self.phase
                if isinstance(self.phase, _CandidatePhase)
                else _CandidatePhase(self.phase)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Feher/Hare candidate task phase must be train or selection_validation"
            ) from error
        object.__setattr__(self, "phase", phase)
        object.__setattr__(
            self,
            "parameters",
            _canonical_parameter_tuple(self.parameters),
        )

    @property
    def simulation_seeds(self) -> tuple[int, ...]:
        if self.phase is _CandidatePhase.TRAIN:
            return study.TRAIN_SEEDS
        return study.SELECTION_SEEDS


@dataclass(frozen=True)
class _WorkerContext:
    prepared: study.PreparedFeherHareTwoStageV1
    source: NarrativeTwoStageModelSource
    runner: SimulationRunner
    selection_suite: HeldOutSuite
    accepted_parameters: ParameterAcceptanceSet | None


_WORKER_CONTEXT: _WorkerContext | None = None


def _family_source(family: str) -> NarrativeTwoStageModelSource:
    for declared_family, source, _grid in study._family_sources():
        if declared_family == family:
            return source
    raise ValueError("Feher/Hare candidate worker family is unsupported")


def _initialize_feher_hare_worker(
    root: str,
    manifest: TwoStageSourceManifest,
    repository_identity: RepositoryIdentity,
    family: str,
    accepted_parameters: tuple[ParameterTuple, ...] | None,
    accepted_source_hashes: tuple[str, ...],
) -> None:
    """Reconstruct all scientific worker state explicitly under spawn."""

    if not isinstance(manifest, TwoStageSourceManifest):
        raise TypeError("Feher/Hare worker requires TwoStageSourceManifest")
    if not isinstance(repository_identity, RepositoryIdentity):
        raise TypeError("Feher/Hare worker requires RepositoryIdentity")
    prepared = study.prepare_feher_hare_two_stage_v1(
        root=Path(root),
        manifest=manifest,
    )
    accepted = (
        None
        if accepted_parameters is None
        else ParameterAcceptanceSet.from_parameters(
            accepted_parameters,
            source_manifest_hashes=accepted_source_hashes,
        )
    )
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = _WorkerContext(
        prepared=prepared,
        source=_family_source(family),
        runner=SimulationRunner(repository_identity=repository_identity),
        selection_suite=study._selection_suite(prepared.selection_targets),
        accepted_parameters=accepted,
    )


def _evaluate_feher_hare_candidate(task: FeherHareCandidateTask) -> dict[str, object]:
    context = _WORKER_CONTEXT
    if context is None:
        raise RuntimeError("Feher/Hare candidate worker is not initialized")
    if task.family != context.source.family:
        raise ValueError("Feher/Hare candidate task family changed after initialization")

    loss = two_stage_brier_loss()
    if task.phase is _CandidatePhase.TRAIN:
        return evaluate_training_candidate(
            runner=context.runner,
            model=context.source,
            target_report=context.prepared.train_targets,
            parameters=task.parameters,
            simulation_seeds=task.simulation_seeds,
            extractor=two_stage_first_stage_policy_metrics,
            loss=loss,
        ).to_payload()

    if context.accepted_parameters is None:
        raise RuntimeError("Feher/Hare selection worker is missing accepted parameters")
    return evaluate_selection_candidate(
        runner=context.runner,
        model=context.source,
        accepted_parameter_set=context.accepted_parameters,
        parameters=task.parameters,
        suite=context.selection_suite,
        extractor=two_stage_first_stage_policy_metrics,
        loss=loss,
    ).to_payload()


def _require_worker_repository_identity(
    shard_repository_identity: Mapping[str, object],
    repository_identity: RepositoryIdentity,
) -> None:
    if shard_repository_identity != repository_identity.manifest_identity():
        raise ValueError("Feher/Hare worker repository identity changed")


def _training_shards(
    values: tuple[object, ...],
    *,
    repository_identity: RepositoryIdentity,
) -> tuple[TrainingCandidateShard, ...]:
    shards: list[TrainingCandidateShard] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise TypeError("Feher/Hare TRAIN worker must return a shard payload")
        shard = TrainingCandidateShard.from_payload(value)
        _require_worker_repository_identity(
            shard.repository_identity,
            repository_identity,
        )
        shards.append(shard)
    return tuple(shards)


def _selection_shards(
    values: tuple[object, ...],
    *,
    repository_identity: RepositoryIdentity,
) -> tuple[SelectionCandidateShard, ...]:
    shards: list[SelectionCandidateShard] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise TypeError("Feher/Hare SELECTION worker must return a shard payload")
        shard = SelectionCandidateShard.from_payload(value)
        _require_worker_repository_identity(
            shard.repository_identity,
            repository_identity,
        )
        shards.append(shard)
    return tuple(shards)


def fit_and_freeze_feher_hare_models_parallel(
    *,
    root: Path,
    manifest: TwoStageSourceManifest,
    repository_identity: RepositoryIdentity,
    executor: CandidateExecutor,
) -> study.FrozenFeherHareModels:
    """Run Feher/Hare TRAIN and SELECTION through deterministic candidate shards."""

    if not isinstance(manifest, TwoStageSourceManifest):
        raise TypeError("parallel Feher/Hare freezing requires TwoStageSourceManifest")
    if not isinstance(repository_identity, RepositoryIdentity):
        raise TypeError("parallel Feher/Hare freezing requires RepositoryIdentity")
    execute = getattr(executor, "execute", None)
    if not callable(execute):
        raise TypeError("parallel Feher/Hare freezing requires a CandidateExecutor")

    source_root = Path(root)
    prepared = study.prepare_feher_hare_two_stage_v1(
        root=source_root,
        manifest=manifest,
    )
    selection_suite = study._selection_suite(prepared.selection_targets)
    loss = two_stage_brier_loss()
    rows: list[study.FeherHareFamilyFreeze] = []

    for family, source, grid in study._family_sources():
        candidates = _grid_candidates(grid)
        training_tasks = tuple(
            FeherHareCandidateTask(
                family=family,
                phase=_CandidatePhase.TRAIN,
                parameters=parameters,
            )
            for parameters in candidates
        )
        training_values = execute(
            _evaluate_feher_hare_candidate,
            training_tasks,
            initializer=_initialize_feher_hare_worker,
            initargs=(
                str(source_root),
                manifest,
                repository_identity,
                family,
                None,
                (),
            ),
        )
        training = assemble_training_fit_report(
            model=source,
            target_report=prepared.train_targets,
            parameter_candidates=candidates,
            simulation_seeds=study.TRAIN_SEEDS,
            extractor=two_stage_first_stage_policy_metrics,
            loss=loss,
            shards=_training_shards(
                tuple(training_values),
                repository_identity=repository_identity,
            ),
        )
        accepted = ParameterAcceptanceSet.from_parameters(
            training.candidate_parameters,
            source_manifest_hashes=(training.manifest.content_hash,),
        )

        selection_tasks = tuple(
            FeherHareCandidateTask(
                family=family,
                phase=_CandidatePhase.SELECTION_VALIDATION,
                parameters=parameters,
            )
            for parameters in accepted.parameters
        )
        selection_values = execute(
            _evaluate_feher_hare_candidate,
            selection_tasks,
            initializer=_initialize_feher_hare_worker,
            initargs=(
                str(source_root),
                manifest,
                repository_identity,
                family,
                accepted.parameters,
                accepted.source_manifest_hashes,
            ),
        )
        selection = assemble_selection_validation_report(
            model=source,
            accepted_parameters=accepted,
            suite=selection_suite,
            extractor=two_stage_first_stage_policy_metrics,
            loss=loss,
            shards=_selection_shards(
                tuple(selection_values),
                repository_identity=repository_identity,
            ),
        )
        frozen = FrozenModelSpec.from_selection(family, source, selection)
        rows.append(
            study.FeherHareFamilyFreeze(
                family=family,
                source=source,
                training=training,
                selection=selection,
                frozen=frozen,
            )
        )

    return study.FrozenFeherHareModels(tuple(rows))


__all__ = [
    "FeherHareCandidateTask",
    "fit_and_freeze_feher_hare_models_parallel",
]

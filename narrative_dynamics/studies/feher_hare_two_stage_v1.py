from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from narrative_dynamics.adapters.narrative_two_stage import (
    NarrativeTwoStageModelSource,
    create_narrative_two_stage_intentional_source,
    create_narrative_two_stage_planning_source,
    create_narrative_two_stage_reactive_source,
)
from narrative_dynamics.adapters.two_stage_metrics import (
    two_stage_brier_loss,
    two_stage_brier_thresholds,
    two_stage_first_stage_policy_metrics,
    two_stage_log_loss,
    two_stage_log_thresholds,
    two_stage_target_spec,
)
from narrative_dynamics.external_validation import (
    ExternalScoreRole,
    ExternalStratum,
    ExternalValidationPreregistration,
    PairwiseSeparationRule,
)
from narrative_dynamics.observations.dataset import (
    ObservationDataset,
    ObservationPartitionRole,
)
from narrative_dynamics.observations.external import ExternalEvidenceDeclaration
from narrative_dynamics.observations.preregistration import (
    FrozenModelSpec,
    PreregisteredEvaluationProtocol,
)
from narrative_dynamics.observations.targets import (
    CategoricalTargetSpec,
    TargetConstructionReport,
    construct_categorical_targets,
)
from narrative_dynamics.observations.training import (
    TrainingFitReport,
    fit_training_target_grid,
)
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.uncertainty import ParameterAcceptanceSet
from narrative_dynamics.validation import (
    EvaluationRole,
    HeldOutCase,
    HeldOutSuite,
    SelectionValidationReport,
    select_on_validation_suite,
)

from .two_stage_source import (
    TwoStageSourceManifest,
    verify_two_stage_snapshot,
)
from .two_stage_transform import (
    TwoStageParticipantAssignment,
    TwoStageParticipantSplitPlan,
    TwoStageTransformReport,
    assign_two_stage_participants,
    build_two_stage_observation_dataset,
    transform_two_stage_snapshot,
)


UPSTREAM_REPOSITORY = "carolfs/muddled_models"
UPSTREAM_REVISION = "4567763780a2c596fd6510af720ec468a8214a8f"
TRAIN_SEEDS = (101, 102)
SELECTION_SEEDS = (201, 202)
FINAL_SEEDS = (301, 302)
REACTIVE_GRID = {"beta": (0.5, 1.0, 2.0, 4.0)}
HISTORY_GRID = {
    "beta": (0.5, 1.0, 2.0, 4.0),
    "memory_decay": (0.5, 0.75, 0.9, 1.0),
}
_STUDY_NAME = "feher-hare-two-stage-v1"
_STUDY_VERSION = "1"
_BRIER_SEPARATION_DELTA = 0.005
_LOG_SEPARATION_DELTA = 0.006931471805599453
_FAMILIES = ("reactive", "intentional", "planning")


@dataclass(frozen=True)
class PreparedFeherHareTwoStageV1:
    source_manifest: TwoStageSourceManifest
    transform_report: TwoStageTransformReport
    assignment: TwoStageParticipantAssignment
    dataset: ObservationDataset
    evidence: ExternalEvidenceDeclaration
    target_spec: CategoricalTargetSpec
    train_targets: TargetConstructionReport
    selection_targets: TargetConstructionReport
    final_targets: TargetConstructionReport

    def __post_init__(self) -> None:
        if not isinstance(self.source_manifest, TwoStageSourceManifest):
            raise TypeError("prepared two-stage study requires TwoStageSourceManifest")
        if not isinstance(self.transform_report, TwoStageTransformReport):
            raise TypeError("prepared two-stage study requires TwoStageTransformReport")
        if not isinstance(self.assignment, TwoStageParticipantAssignment):
            raise TypeError("prepared two-stage study requires participant assignment")
        if not isinstance(self.dataset, ObservationDataset):
            raise TypeError("prepared two-stage study requires ObservationDataset")
        if not isinstance(self.evidence, ExternalEvidenceDeclaration):
            raise TypeError("prepared two-stage study requires external evidence declaration")
        if not isinstance(self.target_spec, CategoricalTargetSpec):
            raise TypeError("prepared two-stage study requires categorical target spec")
        roles = (
            (self.train_targets, ObservationPartitionRole.TRAIN),
            (self.selection_targets, ObservationPartitionRole.SELECTION_VALIDATION),
            (self.final_targets, ObservationPartitionRole.FINAL_TEST),
        )
        for report, role in roles:
            if not isinstance(report, TargetConstructionReport) or report.role is not role:
                raise TypeError("prepared two-stage target reports have invalid roles")
            if report.dataset_hash != self.dataset.content_hash:
                raise ValueError("prepared two-stage target dataset identity changed")
            if report.spec_hash != self.target_spec.content_hash:
                raise ValueError("prepared two-stage target spec identity changed")
        if self.evidence.dataset_hash != self.dataset.content_hash:
            raise ValueError("prepared external evidence dataset identity changed")


@dataclass(frozen=True)
class FeherHareFamilyFreeze:
    family: str
    source: NarrativeTwoStageModelSource
    training: TrainingFitReport
    selection: SelectionValidationReport
    frozen: FrozenModelSpec

    def __post_init__(self) -> None:
        if self.family not in _FAMILIES:
            raise ValueError("two-stage frozen family is unsupported")
        if not isinstance(self.source, NarrativeTwoStageModelSource):
            raise TypeError("two-stage frozen family requires narrative model source")
        if self.source.family != self.family:
            raise ValueError("two-stage frozen family/source identity changed")
        if not isinstance(self.training, TrainingFitReport):
            raise TypeError("two-stage frozen family requires training report")
        if not isinstance(self.selection, SelectionValidationReport):
            raise TypeError("two-stage frozen family requires selection report")
        if not isinstance(self.frozen, FrozenModelSpec):
            raise TypeError("two-stage frozen family requires FrozenModelSpec")
        if self.frozen.name != self.family:
            raise ValueError("two-stage frozen candidate family name changed")


@dataclass(frozen=True)
class FrozenFeherHareModels:
    families: tuple[FeherHareFamilyFreeze, ...]

    def __post_init__(self) -> None:
        rows = tuple(sorted(self.families, key=lambda item: _FAMILIES.index(item.family)))
        if tuple(item.family for item in rows) != _FAMILIES:
            raise ValueError("two-stage frozen models require reactive, intentional, planning")
        object.__setattr__(self, "families", rows)

    @property
    def frozen_candidates(self) -> tuple[FrozenModelSpec, ...]:
        return tuple(item.frozen for item in self.families)

    @property
    def training_manifest_hashes(self) -> tuple[str, ...]:
        return tuple(item.training.manifest.content_hash for item in self.families)

    @property
    def selection_manifest_hashes(self) -> tuple[str, ...]:
        return tuple(item.selection.manifest.content_hash for item in self.families)


@dataclass(frozen=True)
class FeherHareProtocolBundle:
    brier_protocol: PreregisteredEvaluationProtocol
    log_protocol: PreregisteredEvaluationProtocol
    preregistration: ExternalValidationPreregistration

    def __post_init__(self) -> None:
        if not isinstance(self.brier_protocol, PreregisteredEvaluationProtocol):
            raise TypeError("two-stage protocol bundle requires Brier protocol")
        if not isinstance(self.log_protocol, PreregisteredEvaluationProtocol):
            raise TypeError("two-stage protocol bundle requires Log protocol")
        if not isinstance(self.preregistration, ExternalValidationPreregistration):
            raise TypeError("two-stage protocol bundle requires external preregistration")
        if self.preregistration.brier_protocol_hash != self.brier_protocol.content_hash:
            raise ValueError("two-stage Brier protocol/preregistration identity changed")
        if self.preregistration.log_protocol_hash != self.log_protocol.content_hash:
            raise ValueError("two-stage Log protocol/preregistration identity changed")


def _external_transform_identity(dataset: ObservationDataset) -> Mapping[str, object]:
    value = dataset.provenance.get("transform_identity")
    if not isinstance(value, Mapping):
        raise ValueError("two-stage dataset is missing transform identity")
    return value


def prepare_feher_hare_two_stage_v1(
    *,
    root: Path,
    manifest: TwoStageSourceManifest,
) -> PreparedFeherHareTwoStageV1:
    if not isinstance(manifest, TwoStageSourceManifest):
        raise TypeError("two-stage study preparation requires source manifest")
    snapshot = verify_two_stage_snapshot(Path(root), manifest)
    transform_report = transform_two_stage_snapshot(snapshot, manifest)
    assignment = assign_two_stage_participants(
        transform_report,
        TwoStageParticipantSplitPlan(),
    )
    dataset = build_two_stage_observation_dataset(transform_report, assignment)
    target_spec = two_stage_target_spec()
    train_targets = construct_categorical_targets(
        dataset,
        role=ObservationPartitionRole.TRAIN,
        spec=target_spec,
    )
    selection_targets = construct_categorical_targets(
        dataset,
        role=ObservationPartitionRole.SELECTION_VALIDATION,
        spec=target_spec,
    )
    final_targets = construct_categorical_targets(
        dataset,
        role=ObservationPartitionRole.FINAL_TEST,
        spec=target_spec,
    )
    evidence = ExternalEvidenceDeclaration.from_dataset(
        dataset,
        name=_STUDY_NAME,
        version=_STUDY_VERSION,
        source_snapshot_hash=transform_report.source_snapshot_hash,
        source_reference=manifest.repository,
        source_revision={
            "repository": manifest.repository,
            "revision": manifest.revision,
            "source_manifest_hash": manifest.content_hash,
        },
        transform_identity=_external_transform_identity(dataset),
        record_namespace=_STUDY_NAME,
    )
    return PreparedFeherHareTwoStageV1(
        source_manifest=manifest,
        transform_report=transform_report,
        assignment=assignment,
        dataset=dataset,
        evidence=evidence,
        target_spec=target_spec,
        train_targets=train_targets,
        selection_targets=selection_targets,
        final_targets=final_targets,
    )


def _selection_suite(targets: TargetConstructionReport) -> HeldOutSuite:
    if targets.role is not ObservationPartitionRole.SELECTION_VALIDATION:
        raise ValueError("two-stage selection suite requires selection targets")
    return HeldOutSuite(
        name=f"{_STUDY_NAME}:selection_validation",
        role=EvaluationRole.SELECTION_VALIDATION,
        cases=tuple(
            HeldOutCase(
                scenario=case.scenario,
                seeds=SELECTION_SEEDS,
                target=case.target_map,
                name=case.name,
            )
            for case in targets.cases
        ),
    )


def _family_sources() -> tuple[tuple[str, NarrativeTwoStageModelSource, Mapping[str, tuple[float, ...]]], ...]:
    return (
        ("reactive", create_narrative_two_stage_reactive_source(), REACTIVE_GRID),
        ("intentional", create_narrative_two_stage_intentional_source(), HISTORY_GRID),
        ("planning", create_narrative_two_stage_planning_source(), HISTORY_GRID),
    )


def fit_and_freeze_feher_hare_models(
    *,
    runner: SimulationRunner,
    prepared: PreparedFeherHareTwoStageV1,
) -> FrozenFeherHareModels:
    if not isinstance(runner, SimulationRunner):
        raise TypeError("two-stage model freezing requires SimulationRunner")
    if not isinstance(prepared, PreparedFeherHareTwoStageV1):
        raise TypeError("two-stage model freezing requires prepared study")

    brier = two_stage_brier_loss()
    selection_suite = _selection_suite(prepared.selection_targets)
    rows: list[FeherHareFamilyFreeze] = []
    for family, source, grid in _family_sources():
        training = fit_training_target_grid(
            runner=runner,
            model=source,
            target_report=prepared.train_targets,
            parameter_grid=grid,
            simulation_seeds=TRAIN_SEEDS,
            extractor=two_stage_first_stage_policy_metrics,
            loss=brier,
        )
        accepted = ParameterAcceptanceSet.from_parameters(
            training.candidate_parameters,
            source_manifest_hashes=(training.manifest.content_hash,),
        )
        selection = select_on_validation_suite(
            runner=runner,
            model=source,
            accepted_parameters=accepted,
            suite=selection_suite,
            extractor=two_stage_first_stage_policy_metrics,
            loss=brier,
        )
        frozen = FrozenModelSpec.from_selection(family, source, selection)
        rows.append(
            FeherHareFamilyFreeze(
                family=family,
                source=source,
                training=training,
                selection=selection,
                frozen=frozen,
            )
        )
    return FrozenFeherHareModels(tuple(rows))


def _task_strata(final_targets: TargetConstructionReport) -> tuple[ExternalStratum, ...]:
    by_task: dict[str, list[str]] = {"magic_carpet": [], "spaceship": []}
    for case in final_targets.cases:
        task = case.provenance.get("task_variant")
        if task not in by_task:
            raise ValueError("two-stage final target is missing task stratum")
        by_task[task].append(case.name)
    if any(not names for names in by_task.values()):
        raise ValueError("two-stage final targets require both task strata")
    return tuple(
        ExternalStratum(task, tuple(sorted(by_task[task])))
        for task in ("magic_carpet", "spaceship")
    )


def build_feher_hare_protocol_bundle(
    *,
    prepared: PreparedFeherHareTwoStageV1,
    frozen_models: FrozenFeherHareModels,
) -> FeherHareProtocolBundle:
    if not isinstance(prepared, PreparedFeherHareTwoStageV1):
        raise TypeError("two-stage protocol construction requires prepared study")
    if not isinstance(frozen_models, FrozenFeherHareModels):
        raise TypeError("two-stage protocol construction requires frozen models")

    candidates = frozen_models.frozen_candidates
    brier_thresholds = two_stage_brier_thresholds()
    log_thresholds = two_stage_log_thresholds()
    brier_protocol = PreregisteredEvaluationProtocol.create(
        name=f"{_STUDY_NAME}:brier",
        version=_STUDY_VERSION,
        dataset=prepared.dataset,
        target_spec=prepared.target_spec,
        extractor=two_stage_first_stage_policy_metrics,
        loss=two_stage_brier_loss(),
        simulation_seeds=FINAL_SEEDS,
        baseline_name="planning",
        candidates=candidates,
        thresholds=brier_thresholds,
    )
    log_protocol = PreregisteredEvaluationProtocol.create(
        name=f"{_STUDY_NAME}:log",
        version=_STUDY_VERSION,
        dataset=prepared.dataset,
        target_spec=prepared.target_spec,
        extractor=two_stage_first_stage_policy_metrics,
        loss=two_stage_log_loss(),
        simulation_seeds=FINAL_SEEDS,
        baseline_name="planning",
        candidates=candidates,
        thresholds=log_thresholds,
    )
    preregistration = ExternalValidationPreregistration.create(
        name=_STUDY_NAME,
        version=_STUDY_VERSION,
        evidence_declaration=prepared.evidence,
        brier_protocol=brier_protocol,
        log_protocol=log_protocol,
        final_target_set=prepared.final_targets,
        adequacy_thresholds_by_score=(
            (ExternalScoreRole.BRIER, brier_thresholds),
            (ExternalScoreRole.LOG, log_thresholds),
        ),
        strata=_task_strata(prepared.final_targets),
        separation_rule=PairwiseSeparationRule(
            min_mean_loss_delta_brier=_BRIER_SEPARATION_DELTA,
            min_mean_loss_delta_log=_LOG_SEPARATION_DELTA,
        ),
        constraint_plans=(),
        method_validation_hashes=(
            frozen_models.training_manifest_hashes
            + frozen_models.selection_manifest_hashes
        ),
    )
    return FeherHareProtocolBundle(
        brier_protocol=brier_protocol,
        log_protocol=log_protocol,
        preregistration=preregistration,
    )


__all__ = [
    "FINAL_SEEDS",
    "HISTORY_GRID",
    "REACTIVE_GRID",
    "SELECTION_SEEDS",
    "TRAIN_SEEDS",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_REVISION",
    "FeherHareFamilyFreeze",
    "FeherHareProtocolBundle",
    "FrozenFeherHareModels",
    "PreparedFeherHareTwoStageV1",
    "build_feher_hare_protocol_bundle",
    "fit_and_freeze_feher_hare_models",
    "prepare_feher_hare_two_stage_v1",
]

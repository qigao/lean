from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from narrative_dynamics.contracts import ExperimentManifest, ExperimentStage
from narrative_dynamics.losses import MetricLoss, metric_loss_identity
from narrative_dynamics.manifest import component_identity, required_manifest_hash
from narrative_dynamics.simulation import ModelSource, SimulationRunner
from narrative_dynamics.validation import (
    EvaluationRole,
    FinalTestReport,
    HeldOutCase,
    HeldOutSuite,
    evaluate_on_final_test_suite,
)
from narrative_dynamics.observations.dataset import ObservationPartitionRole
from narrative_dynamics.observations.preregistration import (
    FrozenModelCandidate,
    FrozenModelSpec,
    PreregisteredEvaluationProtocol,
)
from narrative_dynamics.observations.targets import TargetConstructionReport


@dataclass(frozen=True)
class ComparisonModel:
    frozen: FrozenModelCandidate
    model: ModelSource

    def __post_init__(self) -> None:
        if not isinstance(self.frozen, FrozenModelSpec):
            raise TypeError("comparison model requires a frozen model candidate")


@dataclass(frozen=True)
class ModelComparisonEntry:
    name: str
    parameters: tuple[tuple[str, float], ...]
    final_test: FinalTestReport
    mean_loss: float
    worst_loss: float
    delta_mean_from_baseline: float
    delta_worst_from_baseline: float
    adequate: bool


@dataclass(frozen=True)
class ModelComparisonReport:
    baseline_name: str
    ranking: tuple[ModelComparisonEntry, ...]
    manifest: ExperimentManifest

    @property
    def best(self) -> ModelComparisonEntry:
        return self.ranking[0]

    @property
    def entry_map(self):
        return MappingProxyType({entry.name: entry for entry in self.ranking})

    @property
    def baseline_adequate(self) -> bool:
        return self.entry_map[self.baseline_name].adequate


def _preflight(
    *,
    models: tuple[ComparisonModel, ...],
    target_set: TargetConstructionReport,
    extractor: object,
    loss: MetricLoss,
    simulation_seeds: tuple[int, ...],
    protocol: PreregisteredEvaluationProtocol,
) -> tuple[ComparisonModel, ...]:
    if not isinstance(protocol, PreregisteredEvaluationProtocol):
        raise TypeError("final model comparison requires PreregisteredEvaluationProtocol")
    if not isinstance(target_set, TargetConstructionReport):
        raise TypeError("final model comparison requires constructed targets")
    if target_set.role is not ObservationPartitionRole.FINAL_TEST:
        raise ValueError("final model comparison requires final-test targets")
    if target_set.dataset_hash != protocol.dataset_hash:
        raise ValueError("final model comparison dataset changed")
    if target_set.partition_hash != protocol.final_partition_hash:
        raise ValueError("final model comparison partition changed")
    if target_set.spec_hash != protocol.target_spec_hash:
        raise ValueError("final model comparison target spec changed")
    if target_set.content_hash != protocol.final_target_hash:
        raise ValueError("final model comparison target payload changed")
    if target_set.manifest.content_hash != protocol.final_target_manifest_hash:
        raise ValueError("final model comparison target lineage changed")
    if metric_loss_identity(loss) != dict(protocol.loss_identity):
        raise ValueError("final model comparison loss changed")
    if tuple(simulation_seeds) != protocol.simulation_seeds:
        raise ValueError("final model comparison simulation seeds changed")

    resolved = tuple(sorted(models, key=lambda item: item.frozen.name))
    if not resolved:
        raise ValueError("final model comparison requires models")
    names = tuple(item.frozen.name for item in resolved)
    if names != protocol.candidate_names:
        raise ValueError("final model comparison requires exactly the preregistered candidates")
    frozen_by_name = {candidate.name: candidate for candidate in protocol.candidates}
    for item in resolved:
        expected = frozen_by_name[item.frozen.name]
        if item.frozen.content_hash != expected.content_hash:
            raise ValueError(f"frozen candidate changed for {item.frozen.name!r}")
        if component_identity(item.model) != dict(expected.model_identity):
            raise ValueError(f"runtime model identity changed for {item.frozen.name!r}")
    return resolved


def compare_models_on_final_partition(
    *,
    runner: SimulationRunner,
    models: tuple[ComparisonModel, ...],
    target_set: TargetConstructionReport,
    extractor: object,
    loss: MetricLoss,
    simulation_seeds: tuple[int, ...],
    protocol: PreregisteredEvaluationProtocol,
) -> ModelComparisonReport:
    resolved = _preflight(
        models=tuple(models),
        target_set=target_set,
        extractor=extractor,
        loss=loss,
        simulation_seeds=tuple(simulation_seeds),
        protocol=protocol,
    )
    suite = HeldOutSuite(
        name=f"{protocol.name}:final_test",
        role=EvaluationRole.FINAL_TEST,
        cases=tuple(
            HeldOutCase(
                scenario=case.scenario,
                seeds=protocol.simulation_seeds,
                target=case.target_map,
                name=case.name,
            )
            for case in target_set.cases
        ),
    )

    raw_entries: list[tuple[ComparisonModel, FinalTestReport]] = []
    for item in resolved:
        final_test = evaluate_on_final_test_suite(
            runner=runner,
            model=item.model,
            parameters=dict(item.frozen.parameters),
            suite=suite,
            extractor=extractor,
            loss=loss,
        )
        raw_entries.append((item, final_test))

    baseline_report = next(
        report for item, report in raw_entries
        if item.frozen.name == protocol.baseline_name
    )
    baseline_mean = baseline_report.validation.mean_loss
    baseline_worst = baseline_report.validation.worst_loss
    entries = tuple(
        ModelComparisonEntry(
            name=item.frozen.name,
            parameters=item.frozen.parameters,
            final_test=report,
            mean_loss=report.validation.mean_loss,
            worst_loss=report.validation.worst_loss,
            delta_mean_from_baseline=report.validation.mean_loss - baseline_mean,
            delta_worst_from_baseline=report.validation.worst_loss - baseline_worst,
            adequate=(
                report.validation.mean_loss <= protocol.thresholds.max_mean_loss
                and report.validation.worst_loss <= protocol.thresholds.max_worst_loss
            ),
        )
        for item, report in raw_entries
    )
    ranking = tuple(sorted(
        entries,
        key=lambda entry: (entry.mean_loss, entry.worst_loss, entry.name),
    ))
    manifest = ExperimentManifest(
        stage=ExperimentStage.MODEL_COMPARISON,
        inputs={
            "protocol_hash": protocol.content_hash,
            "declared_precommitment_hash": protocol.declared_precommitment_hash,
            "dataset_hash": protocol.dataset_hash,
            "final_partition_hash": protocol.final_partition_hash,
            "final_target_hash": protocol.final_target_hash,
            "final_target_manifest_hash": protocol.final_target_manifest_hash,
            "loss_identity": protocol.loss_identity,
            "simulation_seeds": protocol.simulation_seeds,
            "baseline_name": protocol.baseline_name,
            "ranking_rule": ("mean_loss", "worst_loss", "model_name"),
            "entries": tuple({
                "name": entry.name,
                "parameters": entry.parameters,
                "mean_loss": entry.mean_loss,
                "worst_loss": entry.worst_loss,
                "delta_mean_from_baseline": entry.delta_mean_from_baseline,
                "delta_worst_from_baseline": entry.delta_worst_from_baseline,
                "adequate": entry.adequate,
            } for entry in ranking),
        },
        parent_hashes=tuple(
            required_manifest_hash(
                entry.final_test,
                label=f"model comparison {entry.name!r} final test",
            )
            for entry in ranking
        ),
    )
    return ModelComparisonReport(
        baseline_name=protocol.baseline_name,
        ranking=ranking,
        manifest=manifest,
    )


__all__ = [
    "ComparisonModel",
    "ModelComparisonEntry",
    "ModelComparisonReport",
    "compare_models_on_final_partition",
]

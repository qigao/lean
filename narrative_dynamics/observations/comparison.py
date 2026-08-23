from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from narrative_dynamics.contracts import ExperimentManifest, ExperimentStage
from narrative_dynamics.losses import MetricLoss, metric_loss_identity
from narrative_dynamics.manifest import callable_identity, component_identity, required_manifest_hash
from narrative_dynamics.simulation import ModelSource, SimulationRunner
from narrative_dynamics.validation import FinalTestReport, evaluate_on_final_test_suite

from .dataset import ObservationPartitionRole
from .preregistration import ComparisonPreregistration, FrozenModelSpec
from .targets import TargetConstructionReport


@dataclass(frozen=True)
class AlternativeModelEvaluation:
    name: str
    parameters: tuple[tuple[str, float], ...]
    final_test: FinalTestReport
    mean_loss: float
    worst_loss: float
    adequate: bool


@dataclass(frozen=True)
class AlternativeModelComparisonReport:
    registration_hash: str
    ranking: tuple[AlternativeModelEvaluation, ...]
    manifest: ExperimentManifest

    @property
    def best(self) -> AlternativeModelEvaluation:
        return self.ranking[0]


def _preflight(
    *,
    registration: ComparisonPreregistration,
    target_report: TargetConstructionReport,
    models: Mapping[str, ModelSource],
    extractor: object,
    loss: MetricLoss,
) -> tuple[tuple[FrozenModelSpec, ModelSource], ...]:
    if not isinstance(registration, ComparisonPreregistration):
        raise TypeError("registered comparison requires ComparisonPreregistration")
    if not isinstance(target_report, TargetConstructionReport):
        raise TypeError("registered comparison target must be TargetConstructionReport")
    if target_report.role is not ObservationPartitionRole.FINAL_TEST:
        raise ValueError("registered comparison requires final-test targets")
    if target_report.dataset_hash != registration.dataset_hash:
        raise ValueError("registered comparison dataset identity changed")
    if target_report.partition_hash != registration.final_partition_hash:
        raise ValueError("registered comparison final partition changed")
    if target_report.manifest.content_hash != registration.target_construction_manifest_hash:
        raise ValueError("registered comparison target construction changed")
    if callable_identity(extractor) != dict(registration.metric_identity):
        raise ValueError("registered comparison metric extractor changed")
    if metric_loss_identity(loss) != dict(registration.loss_identity):
        raise ValueError("registered comparison loss changed")
    if not isinstance(models, Mapping):
        raise TypeError("registered comparison models must be a mapping")
    expected_names = tuple(model.name for model in registration.models)
    if set(models) != set(expected_names):
        raise ValueError("registered comparison requires exactly the preregistered model set")

    resolved: list[tuple[FrozenModelSpec, ModelSource]] = []
    for frozen in registration.models:
        runtime = models[frozen.name]
        if component_identity(runtime) != dict(frozen.model_identity):
            raise ValueError(f"registered comparison runtime identity changed for {frozen.name!r}")
        resolved.append((frozen, runtime))
    return tuple(resolved)


def compare_registered_models(
    *,
    runner: SimulationRunner,
    registration: ComparisonPreregistration,
    target_report: TargetConstructionReport,
    models: Mapping[str, ModelSource],
    extractor: object,
    loss: MetricLoss,
) -> AlternativeModelComparisonReport:
    resolved = _preflight(
        registration=registration,
        target_report=target_report,
        models=models,
        extractor=extractor,
        loss=loss,
    )
    suite = target_report.as_held_out_suite()
    evaluations: list[AlternativeModelEvaluation] = []
    parent_hashes: list[str] = [target_report.manifest.content_hash]
    for frozen, runtime in resolved:
        final_test = evaluate_on_final_test_suite(
            runner=runner,
            model=runtime,
            parameters=dict(frozen.parameters),
            suite=suite,
            extractor=extractor,
            loss=loss,
        )
        parent_hashes.append(required_manifest_hash(
            final_test,
            label=f"alternative model {frozen.name!r} final test",
        ))
        mean_loss = final_test.validation.mean_loss
        worst_loss = final_test.validation.worst_loss
        evaluations.append(AlternativeModelEvaluation(
            name=frozen.name,
            parameters=frozen.parameters,
            final_test=final_test,
            mean_loss=mean_loss,
            worst_loss=worst_loss,
            adequate=(
                mean_loss <= registration.thresholds.max_mean_loss
                and worst_loss <= registration.thresholds.max_worst_loss
            ),
        ))

    ranking = tuple(sorted(
        evaluations,
        key=lambda item: (item.mean_loss, item.worst_loss, item.name),
    ))
    manifest = ExperimentManifest(
        stage=ExperimentStage.ALTERNATIVE_MODEL_COMPARISON,
        inputs={
            "registration_hash": registration.content_hash,
            "dataset_hash": registration.dataset_hash,
            "final_partition_hash": registration.final_partition_hash,
            "target_construction_manifest_hash": registration.target_construction_manifest_hash,
            "metric_identity": registration.metric_identity,
            "loss_identity": registration.loss_identity,
            "thresholds": registration.thresholds.identity_payload(),
            "ranking_rule": registration.ranking_rule,
            "models": tuple({
                "name": evaluation.name,
                "parameters": evaluation.parameters,
                "mean_loss": evaluation.mean_loss,
                "worst_loss": evaluation.worst_loss,
                "adequate": evaluation.adequate,
            } for evaluation in ranking),
        },
        parent_hashes=tuple(parent_hashes),
    )
    return AlternativeModelComparisonReport(
        registration_hash=registration.content_hash,
        ranking=ranking,
        manifest=manifest,
    )


__all__ = [
    "AlternativeModelComparisonReport",
    "AlternativeModelEvaluation",
    "compare_registered_models",
]

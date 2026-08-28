from __future__ import annotations

from narrative_dynamics.contracts import ModelRun, Scenario, stable_content_hash
from narrative_dynamics.losses import (
    CategoricalBrierLoss,
    CategoricalLogLoss,
    CategoricalMetricGroup,
)
from narrative_dynamics.model_comparison import ComparisonModel
from narrative_dynamics.observations import (
    AdequacyThresholds,
    CategoricalTargetSpec,
    FrozenModelCandidate,
    ObservationDataset,
    ObservationPartition,
    ObservationPartitionRole,
    ObservationRecord,
    PreregisteredEvaluationProtocol,
    ProtocolRelease,
    WitnessReceipt,
    construct_categorical_targets,
    verify_protocol_release,
)


FINAL_SEEDS = (41, 42)
SELECTION_SEEDS = (31, 32)
GROUP = CategoricalMetricGroup("choice", ("choice.a", "choice.b"))
SELECTION_HASH = stable_content_hash({"selection": "external-validation-v1"})
ALTERNATIVE_SELECTION_HASH = stable_content_hash(
    {"selection": "external-validation-v1-drift"}
)
METHOD_HASH = stable_content_hash({"method": "synthetic-identification-v1"})


def policy_metrics(trace):
    policy = trace.outcome["policy"]
    return {
        "choice.a": float(policy["a"]),
        "choice.b": float(policy["b"]),
    }


policy_metrics.version = "external-validation-v1"


def policy_metrics_v2(trace):
    return policy_metrics(trace)


policy_metrics_v2.version = "external-validation-v2"


class ProbabilityModel:
    name = "external-probability-model"
    version = "1"
    implementation_revision = "external-probability-v1"

    def __init__(self) -> None:
        self.calls = 0

    def simulate(self, scenario, parameters, rng):
        self.calls += 1
        p = float(parameters["p"])
        return ModelRun(
            events=(),
            outcome={"policy": {"a": p, "b": 1.0 - p}},
        )


class AlternativeProbabilityModel(ProbabilityModel):
    name = "external-alternative-model"
    implementation_revision = "external-alternative-v1"


class FixtureWitnessVerifier:
    name = "external-validation-test-verifier"
    version = "1"

    def verify(self, release, receipt):
        return (
            receipt.provider == "test-fixture"
            and dict(receipt.proof) == {"nonce": "external-validation-v1"}
        )


def _record(record_id: str, scenario_id: str, a: int, b: int) -> ObservationRecord:
    return ObservationRecord(
        id=record_id,
        scenario=Scenario(id=scenario_id, payload={"condition": scenario_id}),
        counts={"a": a, "b": b},
        metadata={"test_only": True},
    )


def external_dataset(
    *,
    source_kind: str = "external_observational",
    external_observational: bool = True,
    empirical_human_data: bool = False,
    reverse_partition_input: bool = False,
    swap_train_selection: bool = False,
) -> ObservationDataset:
    train = ObservationPartition(
        "train",
        ObservationPartitionRole.TRAIN,
        records=(_record("ext:train:1", "train-1", 7, 3),),
    )
    selection_records = (
        _record("ext:selection:1", "selection-1", 8, 2),
        _record("ext:selection:2", "selection-2", 8, 2),
    )
    selection = ObservationPartition(
        "selection",
        ObservationPartitionRole.SELECTION_VALIDATION,
        records=selection_records,
    )
    final = ObservationPartition(
        "final",
        ObservationPartitionRole.FINAL_TEST,
        records=(
            _record("ext:final:1", "final-1", 9, 1),
            _record("ext:final:2", "final-2", 6, 4),
        ),
    )
    if swap_train_selection:
        train = ObservationPartition(
            "train",
            ObservationPartitionRole.TRAIN,
            records=selection_records,
        )
        selection = ObservationPartition(
            "selection",
            ObservationPartitionRole.SELECTION_VALIDATION,
            records=(_record("ext:train:1", "train-1", 7, 3),),
        )
    partitions = (train, selection, final)
    if reverse_partition_input:
        partitions = tuple(reversed(partitions))
    return ObservationDataset(
        name="external-observations-test-only",
        version="1",
        source={
            "kind": source_kind,
            "release": "test-snapshot-v1",
        },
        provenance={
            "external_observational": external_observational,
            "empirical_human_data": empirical_human_data,
            "test_only": True,
        },
        partitions=partitions,
    )


def target_spec() -> CategoricalTargetSpec:
    return CategoricalTargetSpec(
        name="external-choice",
        version="1",
        categories=("a", "b"),
        metric_prefix="choice",
    )


def brier_loss() -> CategoricalBrierLoss:
    return CategoricalBrierLoss((GROUP,))


def log_loss() -> CategoricalLogLoss:
    return CategoricalLogLoss((GROUP,))


def frozen_candidates(
    *,
    baseline_p: float = 0.8,
    alternative_p: float = 0.2,
    baseline_selection_hash: str = SELECTION_HASH,
    alternative_selection_hash: str = SELECTION_HASH,
):
    baseline = FrozenModelCandidate.freeze(
        name="baseline",
        model=ProbabilityModel(),
        parameters={"p": baseline_p},
        selection_manifest_hash=baseline_selection_hash,
    )
    alternative = FrozenModelCandidate.freeze(
        name="alternative",
        model=AlternativeProbabilityModel(),
        parameters={"p": alternative_p},
        selection_manifest_hash=alternative_selection_hash,
    )
    return (baseline, alternative)


def sibling_protocols(
    *,
    dataset: ObservationDataset | None = None,
    brier_thresholds: AdequacyThresholds | None = None,
    log_thresholds: AdequacyThresholds | None = None,
    brier_candidates=None,
    log_candidates=None,
    brier_extractor=policy_metrics,
    log_extractor=policy_metrics,
    brier_seeds=FINAL_SEEDS,
    log_seeds=FINAL_SEEDS,
    brier_baseline: str = "baseline",
    log_baseline: str = "baseline",
    version: str = "1",
):
    data = external_dataset() if dataset is None else dataset
    brier_candidates = (
        frozen_candidates() if brier_candidates is None else tuple(brier_candidates)
    )
    log_candidates = (
        brier_candidates if log_candidates is None else tuple(log_candidates)
    )
    brier_thresholds = (
        AdequacyThresholds(1.0, 1.0)
        if brier_thresholds is None
        else brier_thresholds
    )
    log_thresholds = (
        AdequacyThresholds(3.0, 3.0)
        if log_thresholds is None
        else log_thresholds
    )
    brier_protocol = PreregisteredEvaluationProtocol.create(
        name="external-validation:brier",
        version=version,
        dataset=data,
        target_spec=target_spec(),
        extractor=brier_extractor,
        loss=brier_loss(),
        simulation_seeds=tuple(brier_seeds),
        baseline_name=brier_baseline,
        candidates=tuple(brier_candidates),
        thresholds=brier_thresholds,
    )
    log_protocol = PreregisteredEvaluationProtocol.create(
        name="external-validation:log",
        version=version,
        dataset=data,
        target_spec=target_spec(),
        extractor=log_extractor,
        loss=log_loss(),
        simulation_seeds=tuple(log_seeds),
        baseline_name=log_baseline,
        candidates=tuple(log_candidates),
        thresholds=log_thresholds,
    )
    return data, brier_protocol, log_protocol


def final_targets(dataset: ObservationDataset | None = None):
    data = external_dataset() if dataset is None else dataset
    return construct_categorical_targets(
        data,
        role=ObservationPartitionRole.FINAL_TEST,
        spec=target_spec(),
    )


def selection_targets(dataset: ObservationDataset | None = None):
    data = external_dataset() if dataset is None else dataset
    return construct_categorical_targets(
        data,
        role=ObservationPartitionRole.SELECTION_VALIDATION,
        spec=target_spec(),
    )


def runtime_models(candidates):
    by_name = {candidate.name: candidate for candidate in candidates}
    baseline_model = ProbabilityModel()
    alternative_model = AlternativeProbabilityModel()
    return (
        (
            ComparisonModel(frozen=by_name["baseline"], model=baseline_model),
            ComparisonModel(frozen=by_name["alternative"], model=alternative_model),
        ),
        (baseline_model, alternative_model),
    )


def declaration_kwargs():
    return {
        "name": "external-evidence-test-only",
        "version": "1",
        "source_snapshot_hash": stable_content_hash(
            {"snapshot": "external-test-snapshot-v1"}
        ),
        "source_reference": "test://external-validation/source-v1",
        "source_revision": {"release": "test-snapshot-v1"},
        "transform_identity": {
            "name": "test-observation-transform",
            "version": "1",
        },
        "record_namespace": "external-test",
    }


def build_external_declaration(dataset: ObservationDataset | None = None):
    from narrative_dynamics.observations.external import ExternalEvidenceDeclaration

    data = external_dataset() if dataset is None else dataset
    return ExternalEvidenceDeclaration.from_dataset(
        data,
        **declaration_kwargs(),
    )


def default_strata():
    from narrative_dynamics.external_validation import ExternalStratum

    return (
        ExternalStratum("early", ("ext:final:1",)),
        ExternalStratum("late", ("ext:final:2",)),
    )


def default_separation_rule():
    from narrative_dynamics.external_validation import PairwiseSeparationRule

    return PairwiseSeparationRule(
        min_mean_loss_delta_brier=1e-6,
        min_mean_loss_delta_log=1e-6,
    )


def build_constraint_plan(
    *,
    parameter_grid=None,
    target_coordinates=("p",),
    selection_case_names=("ext:selection:1", "ext:selection:2"),
    brier_delta: float = 0.0,
    log_delta: float = 0.0,
):
    from narrative_dynamics.external_validation import ExternalConstraintPlan

    if parameter_grid is None:
        parameter_grid = {"p": (0.2, 0.8)}
    return ExternalConstraintPlan.create(
        name="probability-constraint",
        model=ProbabilityModel(),
        parameter_grid=parameter_grid,
        target_coordinates=target_coordinates,
        selection_target_set=selection_targets(external_dataset()),
        selection_case_names=selection_case_names,
        brier_acceptance_loss_delta=brier_delta,
        log_acceptance_loss_delta=log_delta,
        simulation_seeds=SELECTION_SEEDS,
        extractor=policy_metrics,
        brier_loss=brier_loss(),
        log_loss=log_loss(),
    )


def build_preregistration(
    *,
    dataset: ObservationDataset | None = None,
    brier_protocol=None,
    log_protocol=None,
    brier_thresholds: AdequacyThresholds | None = None,
    log_thresholds: AdequacyThresholds | None = None,
    strata=None,
    separation_rule=None,
    constraint_plans=(),
    method_validation_hashes=(METHOD_HASH,),
):
    from narrative_dynamics.external_validation import (
        ExternalScoreRole,
        ExternalValidationPreregistration,
    )

    data = external_dataset() if dataset is None else dataset
    if brier_protocol is None or log_protocol is None:
        _, default_brier, default_log = sibling_protocols(dataset=data)
        brier_protocol = default_brier if brier_protocol is None else brier_protocol
        log_protocol = default_log if log_protocol is None else log_protocol
    declaration = build_external_declaration(data)
    brier_thresholds = (
        brier_protocol.thresholds if brier_thresholds is None else brier_thresholds
    )
    log_thresholds = (
        log_protocol.thresholds if log_thresholds is None else log_thresholds
    )
    return ExternalValidationPreregistration.create(
        name="external-validation-v1",
        version="1",
        evidence_declaration=declaration,
        brier_protocol=brier_protocol,
        log_protocol=log_protocol,
        final_target_set=final_targets(data),
        adequacy_thresholds_by_score=(
            (ExternalScoreRole.BRIER, brier_thresholds),
            (ExternalScoreRole.LOG, log_thresholds),
        ),
        strata=default_strata() if strata is None else tuple(strata),
        separation_rule=(
            default_separation_rule()
            if separation_rule is None
            else separation_rule
        ),
        constraint_plans=tuple(constraint_plans),
        method_validation_hashes=tuple(method_validation_hashes),
    )


def make_release(
    protocol,
    *,
    evidence_hash: str,
    preregistration_hash: str,
    score_role: str,
    repository_revision: str = "test-repository-revision",
    source_revision_extra=None,
):
    source_revision = {
        "repository_revision": repository_revision,
        "external_evidence_declaration_hash": evidence_hash,
        "external_validation_preregistration_hash": preregistration_hash,
        "score_role": score_role,
    }
    if source_revision_extra:
        source_revision.update(source_revision_extra)
    return ProtocolRelease.create(
        name=f"external-validation:{score_role}:release",
        version="1",
        protocol=protocol,
        source_revision=source_revision,
    )


def verify_release(release, protocol):
    receipt = WitnessReceipt.create(
        provider="test-fixture",
        authority="unit-test",
        subject_hash=release.content_hash,
        reference=f"receipt:{release.name}",
        claimed_at="2026-08-28T00:00:00Z",
        proof={"nonce": "external-validation-v1"},
    )
    return verify_protocol_release(
        release,
        protocol=protocol,
        receipts=(receipt,),
        verifier=FixtureWitnessVerifier(),
    )

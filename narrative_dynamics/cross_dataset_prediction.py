"""Worker-only one-pass FINAL prediction matrix for transfer V1."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Protocol

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.cross_dataset_capabilities import FinalWorkerProjection
from narrative_dynamics.cross_dataset_candidates import (
    FrozenTransferCandidate,
    RefitCandidateFreeze,
    TransferFamily,
    ZeroShotCandidateFreeze,
)


FINAL_SEEDS = (301, 302)
PREDICTION_ARTIFACT_IDENTITY = (
    "sha256:0db59a143a75846cbe8c700568f52c205ebec78f4fd341ac5270c75032092e9c"
)


@dataclass(frozen=True)
class TransferModelInput:
    """Only model-visible pre-choice coordinates for one opaque FINAL case."""

    case_token: str
    trial_id: int
    source_stratum: str


@dataclass(frozen=True)
class _ExecutionCandidate:
    path: str
    family: TransferFamily
    parameters: tuple[tuple[str, float], ...]
    candidate_hash: str


class TransferEvaluator(Protocol):
    def __call__(
        self,
        model_input: TransferModelInput,
        candidate: _ExecutionCandidate,
        seed: int,
    ) -> tuple[float, float]:
        ...


@dataclass(frozen=True)
class TransferRunProgress:
    completed_model_runs: int
    execution_receipt_hash: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.completed_model_runs, bool)
            or not isinstance(self.completed_model_runs, int)
            or self.completed_model_runs < 1
        ):
            raise ValueError("completed_model_runs must be a positive integer")
        if (
            not isinstance(self.execution_receipt_hash, str)
            or not self.execution_receipt_hash.startswith("sha256:")
            or len(self.execution_receipt_hash) != 71
        ):
            raise ValueError("execution_receipt_hash must be canonical sha256")

    def to_payload(self) -> dict[str, object]:
        return {
            "completed_model_runs": self.completed_model_runs,
            "execution_receipt_hash": self.execution_receipt_hash,
        }


@dataclass(frozen=True, repr=False)
class _PrivateCasePrediction:
    case_token: str = field(repr=False)
    source_stratum: str
    target_action: str = field(repr=False)
    path: str
    family: TransferFamily
    candidate_hash: str
    seed: int
    probability_action_0: float = field(repr=False)
    probability_action_1: float = field(repr=False)
    execution_receipt_hash: str

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("private FINAL predictions are not serializable")


class SealedTransferPredictionArtifact:
    """Ephemeral private prediction rows, consumable once by the scorer."""

    __slots__ = (
        "_consumed",
        "_rows",
        "candidate_hashes",
        "case_count",
        "completed_model_runs",
        "final_commitment_hash",
        "prediction_artifact_identity",
        "sealed_artifact_hash",
        "seeds",
    )

    def __init__(
        self,
        *,
        rows: tuple[_PrivateCasePrediction, ...],
        candidates: tuple[_ExecutionCandidate, ...],
        seeds: tuple[int, ...],
        case_count: int,
        final_commitment_hash: str,
        prediction_artifact_identity: str,
    ) -> None:
        self._rows = rows
        self._consumed = False
        self.candidate_hashes = tuple(row.candidate_hash for row in candidates)
        self.seeds = seeds
        self.case_count = case_count
        self.completed_model_runs = len(rows)
        self.final_commitment_hash = final_commitment_hash
        self.prediction_artifact_identity = prediction_artifact_identity
        self.sealed_artifact_hash = stable_content_hash(
            {
                "domain": "sealed-transfer-prediction-v1",
                "final_commitment_hash": final_commitment_hash,
                "prediction_artifact_identity": prediction_artifact_identity,
                "candidate_hashes": list(self.candidate_hashes),
                "seeds": list(seeds),
                "case_count": case_count,
                "execution_receipts": [
                    row.execution_receipt_hash for row in rows
                ],
            }
        )

    def consume_for_scoring_once(self) -> tuple[_PrivateCasePrediction, ...]:
        if self._consumed:
            raise RuntimeError("sealed prediction artifact already consumed")
        self._consumed = True
        rows = self._rows
        self._rows = ()
        return rows

    def __repr__(self) -> str:
        state = "consumed" if self._consumed else "sealed"
        return (
            "SealedTransferPredictionArtifact(rows=<sealed>, "
            f"case_count={self.case_count}, state={state!r})"
        )

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("sealed prediction artifacts are not serializable")


def _selected_refit_evaluations(
    refit: RefitCandidateFreeze,
) -> tuple[FrozenTransferCandidate, ...]:
    expected: list[FrozenTransferCandidate] = []
    for family in TransferFamily:
        rows = tuple(row for row in refit.evaluations if row.family is family)
        if not rows:
            raise ValueError("refit freeze does not contain complete evaluations")
        selected = min(
            rows,
            key=lambda row: (row.selection_brier_loss, row.parameters),
        )
        expected.append(
            FrozenTransferCandidate(
                family=family,
                parameters=selected.parameters,
                source_candidate_hash=selected.evaluation_receipt_hash,
            )
        )
    return tuple(expected)


def _exact_six_candidates(
    zero_shot: ZeroShotCandidateFreeze,
    refit: RefitCandidateFreeze,
) -> tuple[_ExecutionCandidate, ...]:
    if not isinstance(zero_shot, ZeroShotCandidateFreeze):
        raise TypeError("zero_shot must be ZeroShotCandidateFreeze")
    if not isinstance(refit, RefitCandidateFreeze):
        raise TypeError("refit must be RefitCandidateFreeze")
    zero = tuple(zero_shot.candidates)
    selected = tuple(refit.selected_candidates)
    if (
        len(zero) != 3
        or len(selected) != 3
        or tuple(row.family for row in zero) != tuple(TransferFamily)
        or tuple(row.family for row in selected) != tuple(TransferFamily)
        or selected != _selected_refit_evaluations(refit)
    ):
        raise ValueError("prediction requires the exact six candidates and paths")
    candidates = tuple(
        _ExecutionCandidate(
            path=path,
            family=row.family,
            parameters=row.parameters,
            candidate_hash=row.content_hash,
        )
        for path, rows in (("ZERO_SHOT", zero), ("REFIT", selected))
        for row in rows
    )
    keys = tuple((row.path, row.family, row.parameters) for row in candidates)
    if len(keys) != 6 or len(set(keys)) != 6:
        raise ValueError("prediction requires the exact six candidates and paths")
    return candidates


def _probabilities(value: object) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("evaluator probabilities must contain exactly two values")
    probabilities: list[float] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("evaluator probabilities must be finite numeric values")
        number = float(raw)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError("evaluator probabilities must be finite and in [0, 1]")
        probabilities.append(number)
    if abs(math.fsum(probabilities) - 1.0) > 1e-12:
        raise ValueError("evaluator probabilities must form a complete simplex")
    return probabilities[0], probabilities[1]


def execute_transfer_final_predictions(
    *,
    projection: FinalWorkerProjection,
    zero_shot: ZeroShotCandidateFreeze,
    refit: RefitCandidateFreeze,
    seeds: tuple[int, ...],
    evaluator: TransferEvaluator,
    progress: Callable[[TransferRunProgress], None],
    expected_final_commitment_hash: str,
    prediction_artifact_identity: str,
) -> SealedTransferPredictionArtifact:
    if not isinstance(projection, FinalWorkerProjection):
        raise TypeError("projection must be FinalWorkerProjection")
    if tuple(seeds) != FINAL_SEEDS:
        raise ValueError("FINAL seeds must be exactly (301, 302)")
    if expected_final_commitment_hash != projection.commitment_hash:
        raise ValueError("FINAL projection commitment mismatch")
    if prediction_artifact_identity != PREDICTION_ARTIFACT_IDENTITY:
        raise ValueError("prediction artifact identity changed")
    if not callable(evaluator) or not callable(progress):
        raise TypeError("evaluator and progress must be callable")
    candidates = _exact_six_candidates(zero_shot, refit)
    cases = projection.consume_once()
    case_tokens = tuple(
        stable_content_hash(
            {
                "domain": "ephemeral-transfer-case-v1",
                "row_commitment": case.row_commitment,
            }
        )
        for case in cases
    )
    if len(case_tokens) != len(set(case_tokens)):
        raise ValueError("FINAL cases are not unique")

    private_rows: list[_PrivateCasePrediction] = []
    for case, case_token in zip(cases, case_tokens, strict=True):
        model_input = TransferModelInput(
            case_token=case_token,
            trial_id=case.trial_id,
            source_stratum=case.source_stratum,
        )
        for candidate in candidates:
            for seed in FINAL_SEEDS:
                p0, p1 = _probabilities(evaluator(model_input, candidate, seed))
                receipt = stable_content_hash(
                    {
                        "domain": "transfer-execution-receipt-v1",
                        "case_token": case_token,
                        "path": candidate.path,
                        "candidate_hash": candidate.candidate_hash,
                        "seed": seed,
                        "probabilities": [p0, p1],
                    }
                )
                private_row = _PrivateCasePrediction(
                    case_token=case_token,
                    source_stratum=case.source_stratum,
                    target_action=case.first_stage_action,
                    path=candidate.path,
                    family=candidate.family,
                    candidate_hash=candidate.candidate_hash,
                    seed=seed,
                    probability_action_0=p0,
                    probability_action_1=p1,
                    execution_receipt_hash=receipt,
                )
                private_rows.append(private_row)
                progress(
                    TransferRunProgress(
                        completed_model_runs=len(private_rows),
                        execution_receipt_hash=receipt,
                    )
                )
    return SealedTransferPredictionArtifact(
        rows=tuple(private_rows),
        candidates=candidates,
        seeds=FINAL_SEEDS,
        case_count=len(cases),
        final_commitment_hash=projection.commitment_hash,
        prediction_artifact_identity=prediction_artifact_identity,
    )

"""Frozen sibling score releases and authoritative dual preflight."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Mapping

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.cross_dataset_ledger import GitTransferAttemptStore


_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


class TransferScore(str, Enum):
    BRIER = "BRIER"
    LOG = "LOG"


CARRY_FORWARD_REQUIREMENT_IDS = (
    "MV1_RI_SELECTION_SPACESHIP_TASK_LOG_BOTH_AGGREGATIONS",
    "MV1_RI_SELECTION_SPACESHIP_SCORE_BOTH_AGGREGATIONS",
    "MV1_RI_SELECTION_SPACESHIP_LOPO_BRIER_LOG_BOTH_AGGREGATIONS",
    "MV1_IP_TRAIN_SPACESHIP_LOPO_BRIER_LOG_BOTH_AGGREGATIONS",
)
TOP_LEVEL_TERMINALS = (
    "GREEN",
    "SCIENTIFIC_RED",
    "INFRASTRUCTURE_INCOMPLETE",
)
LIMITATIONS = (
    "LIMITED_REPRESENTATIVENESS_AND_EXTERNAL_VALIDITY",
    "PREDICTIVE_ONLY_NON_CAUSAL",
    "NO_UNIVERSAL_PARAMETERS_OR_TRUE_FAMILY_CLAIM",
    "NO_REPEATED_SPLIT_OR_POPULATION_WIDE_INTERVAL",
    "NO_EXTERNAL_PREREGISTRATION",
)
_FAMILIES = ("reactive", "intentional", "planning")
_AGGREGATIONS = ("PARTICIPANT_EQUAL", "TRIAL_EQUAL")
_TASK_STRATA = ("SOURCE_STRATUM", "CONDITION_STRATUM")
_PARTICIPANT_INFLUENCE = "DETERMINISTIC_LEAVE_ONE_PARTICIPANT_OUT"
_PERCENTILE_RULE = "linear_order_statistic_v1"


def _strict_fields(
    payload: object,
    *,
    expected: tuple[str, ...],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must be a mapping")
    actual = frozenset(payload)
    required = frozenset(expected)
    if actual != required:
        missing = tuple(sorted(required - actual))
        unknown = tuple(sorted(actual - required))
        raise ValueError(
            f"{label} fields mismatch: missing={missing!r}, unknown={unknown!r}"
        )
    return payload


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical sha256:<64 lowercase hex>")
    return value


def _revision(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase 40-hex revision")
    return value


def _timestamp(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be a canonical UTC timestamp") from exc
    if parsed.isoformat().replace("+00:00", "Z") != value:
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    return value


def _score(value: object) -> TransferScore:
    if isinstance(value, TransferScore):
        return value
    try:
        return TransferScore(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown transfer score") from exc


@dataclass(frozen=True)
class TransferProtocol:
    scientific_revision: str
    source_identity_hash: str
    transform_identity_hash: str
    split_manifest_hash: str
    candidate_hashes: tuple[str, ...]
    baseline_hash: str
    final_commitment_hash: str
    prediction_artifact_identity: str
    score: TransferScore
    score_identity: str
    train_seeds: tuple[int, ...] = (101, 102)
    selection_seeds: tuple[int, ...] = (201, 202)
    final_seeds: tuple[int, ...] = (301, 302)
    aggregations: tuple[str, ...] = _AGGREGATIONS
    task_strata: tuple[str, ...] = _TASK_STRATA
    participant_influence: str = _PARTICIPANT_INFLUENCE
    bootstrap_seed: int = 43001
    bootstrap_replicates: int = 10000
    percentile_rule: str = _PERCENTILE_RULE
    relative_improvement_threshold: float = 0.01
    family_vocabulary: tuple[str, ...] = _FAMILIES
    top_level_terminals: tuple[str, ...] = TOP_LEVEL_TERMINALS
    limitations: tuple[str, ...] = LIMITATIONS
    external_registration: bool = False
    carry_forward_requirement_ids: tuple[str, ...] = CARRY_FORWARD_REQUIREMENT_IDS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scientific_revision",
            _revision(self.scientific_revision, label="scientific_revision"),
        )
        for field_name in (
            "source_identity_hash",
            "transform_identity_hash",
            "split_manifest_hash",
            "baseline_hash",
            "final_commitment_hash",
            "prediction_artifact_identity",
            "score_identity",
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=field_name),
            )
        if not isinstance(self.candidate_hashes, (tuple, list)):
            raise TypeError("candidate_hashes must be a sequence")
        candidates = tuple(
            _hash(value, label=f"candidate_hashes[{index}]")
            for index, value in enumerate(self.candidate_hashes)
        )
        if len(candidates) != 6 or len(set(candidates)) != 6:
            raise ValueError("transfer protocol requires six unique candidate hashes")
        object.__setattr__(self, "candidate_hashes", candidates)
        object.__setattr__(self, "score", _score(self.score))
        exact = (
            ("train_seeds", tuple(self.train_seeds), (101, 102)),
            ("selection_seeds", tuple(self.selection_seeds), (201, 202)),
            ("final_seeds", tuple(self.final_seeds), (301, 302)),
            ("aggregations", tuple(self.aggregations), _AGGREGATIONS),
            ("task_strata", tuple(self.task_strata), _TASK_STRATA),
            ("family_vocabulary", tuple(self.family_vocabulary), _FAMILIES),
            (
                "top_level_terminals",
                tuple(self.top_level_terminals),
                TOP_LEVEL_TERMINALS,
            ),
            ("limitations", tuple(self.limitations), LIMITATIONS),
            (
                "carry_forward_requirement_ids",
                tuple(self.carry_forward_requirement_ids),
                CARRY_FORWARD_REQUIREMENT_IDS,
            ),
        )
        for field_name, actual, expected in exact:
            if actual != expected:
                raise ValueError(f"{field_name} changed from frozen protocol")
            object.__setattr__(self, field_name, actual)
        if self.participant_influence != _PARTICIPANT_INFLUENCE:
            raise ValueError("participant influence diagnostic changed")
        if self.bootstrap_seed != 43001:
            raise ValueError("bootstrap_seed changed from frozen protocol")
        if self.bootstrap_replicates != 10000:
            raise ValueError("bootstrap_replicates changed from frozen protocol")
        if self.percentile_rule != _PERCENTILE_RULE:
            raise ValueError("percentile_rule changed from frozen protocol")
        if self.relative_improvement_threshold != 0.01:
            raise ValueError("relative_improvement_threshold changed from frozen protocol")
        if self.external_registration is not False:
            raise ValueError("external_registration must remain false")

    def shared_identity_payload(self) -> dict[str, object]:
        return {
            "scientific_revision": self.scientific_revision,
            "source_identity_hash": self.source_identity_hash,
            "transform_identity_hash": self.transform_identity_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "candidate_hashes": list(self.candidate_hashes),
            "baseline_hash": self.baseline_hash,
            "final_commitment_hash": self.final_commitment_hash,
            "prediction_artifact_identity": self.prediction_artifact_identity,
            "train_seeds": list(self.train_seeds),
            "selection_seeds": list(self.selection_seeds),
            "final_seeds": list(self.final_seeds),
            "aggregations": list(self.aggregations),
            "task_strata": list(self.task_strata),
            "participant_influence": self.participant_influence,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_replicates": self.bootstrap_replicates,
            "percentile_rule": self.percentile_rule,
            "relative_improvement_threshold": self.relative_improvement_threshold,
            "family_vocabulary": list(self.family_vocabulary),
            "top_level_terminals": list(self.top_level_terminals),
            "limitations": list(self.limitations),
            "external_registration": self.external_registration,
            "carry_forward_requirement_ids": list(
                self.carry_forward_requirement_ids
            ),
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self.shared_identity_payload(),
            "score": self.score.value,
            "score_identity": self.score_identity,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_payload())

    @classmethod
    def from_payload(cls, payload: object) -> TransferProtocol:
        expected = tuple(
            list(
                (
                    "scientific_revision",
                    "source_identity_hash",
                    "transform_identity_hash",
                    "split_manifest_hash",
                    "candidate_hashes",
                    "baseline_hash",
                    "final_commitment_hash",
                    "prediction_artifact_identity",
                    "train_seeds",
                    "selection_seeds",
                    "final_seeds",
                    "aggregations",
                    "task_strata",
                    "participant_influence",
                    "bootstrap_seed",
                    "bootstrap_replicates",
                    "percentile_rule",
                    "relative_improvement_threshold",
                    "family_vocabulary",
                    "top_level_terminals",
                    "limitations",
                    "external_registration",
                    "carry_forward_requirement_ids",
                )
            )
            + ["score", "score_identity"]
        )
        values = _strict_fields(payload, expected=expected, label="transfer protocol")
        return cls(
            scientific_revision=values["scientific_revision"],
            source_identity_hash=values["source_identity_hash"],
            transform_identity_hash=values["transform_identity_hash"],
            split_manifest_hash=values["split_manifest_hash"],
            candidate_hashes=tuple(values["candidate_hashes"]),
            baseline_hash=values["baseline_hash"],
            final_commitment_hash=values["final_commitment_hash"],
            prediction_artifact_identity=values["prediction_artifact_identity"],
            score=values["score"],
            score_identity=values["score_identity"],
            train_seeds=tuple(values["train_seeds"]),
            selection_seeds=tuple(values["selection_seeds"]),
            final_seeds=tuple(values["final_seeds"]),
            aggregations=tuple(values["aggregations"]),
            task_strata=tuple(values["task_strata"]),
            participant_influence=values["participant_influence"],
            bootstrap_seed=values["bootstrap_seed"],
            bootstrap_replicates=values["bootstrap_replicates"],
            percentile_rule=values["percentile_rule"],
            relative_improvement_threshold=values[
                "relative_improvement_threshold"
            ],
            family_vocabulary=tuple(values["family_vocabulary"]),
            top_level_terminals=tuple(values["top_level_terminals"]),
            limitations=tuple(values["limitations"]),
            external_registration=values["external_registration"],
            carry_forward_requirement_ids=tuple(
                values["carry_forward_requirement_ids"]
            ),
        )


def build_transfer_protocol(
    *,
    scientific_revision: str,
    source_identity_hash: str,
    transform_identity_hash: str,
    split_manifest_hash: str,
    candidate_hashes: tuple[str, ...],
    baseline_hash: str,
    final_commitment_hash: str,
    prediction_artifact_identity: str,
    score: TransferScore,
    score_identity: str,
) -> TransferProtocol:
    return TransferProtocol(
        scientific_revision=scientific_revision,
        source_identity_hash=source_identity_hash,
        transform_identity_hash=transform_identity_hash,
        split_manifest_hash=split_manifest_hash,
        candidate_hashes=candidate_hashes,
        baseline_hash=baseline_hash,
        final_commitment_hash=final_commitment_hash,
        prediction_artifact_identity=prediction_artifact_identity,
        score=score,
        score_identity=score_identity,
    )


@dataclass(frozen=True)
class TransferScoreRelease:
    protocol: TransferProtocol
    release_receipt_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, TransferProtocol):
            raise TypeError("protocol must be TransferProtocol")
        object.__setattr__(
            self,
            "release_receipt_hash",
            _hash(self.release_receipt_hash, label="release_receipt_hash"),
        )

    @classmethod
    def create(
        cls,
        protocol: TransferProtocol,
        *,
        release_receipt_hash: str,
    ) -> TransferScoreRelease:
        return cls(
            protocol=protocol,
            release_receipt_hash=release_receipt_hash,
        )

    @property
    def score(self) -> TransferScore:
        return self.protocol.score

    @property
    def prediction_artifact_identity(self) -> str:
        return self.protocol.prediction_artifact_identity

    def shared_identity_payload(self) -> dict[str, object]:
        return self.protocol.shared_identity_payload()

    def to_payload(self) -> dict[str, object]:
        return {
            "protocol": self.protocol.to_payload(),
            "release_receipt_hash": self.release_receipt_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_payload())

    @classmethod
    def from_payload(cls, payload: object) -> TransferScoreRelease:
        values = _strict_fields(
            payload,
            expected=("protocol", "release_receipt_hash"),
            label="transfer score release",
        )
        return cls(
            protocol=TransferProtocol.from_payload(values["protocol"]),
            release_receipt_hash=values["release_receipt_hash"],
        )


@dataclass(frozen=True)
class DualTransferPreflight:
    scientific_revision: str
    shared_protocol_hash: str
    brier_release_hash: str
    log_release_hash: str
    prediction_artifact_identity: str
    ledger_genesis_hash: str
    ledger_head_hash: str
    final_started_count: int
    final_projection_openings: int
    completed_model_runs: int
    prediction_artifact_hashes: tuple[str, ...]
    score_artifact_hashes: tuple[str, ...]
    completed_at_utc: str
    authoritative_store: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scientific_revision",
            _revision(self.scientific_revision, label="scientific_revision"),
        )
        for field_name in (
            "shared_protocol_hash",
            "brier_release_hash",
            "log_release_hash",
            "prediction_artifact_identity",
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=field_name),
            )
        for field_name in ("ledger_genesis_hash", "ledger_head_hash"):
            object.__setattr__(
                self,
                field_name,
                _revision(getattr(self, field_name), label=field_name),
            )
        for field_name in (
            "final_started_count",
            "final_projection_openings",
            "completed_model_runs",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name in (
            "prediction_artifact_hashes",
            "score_artifact_hashes",
        ):
            values = tuple(
                _hash(value, label=field_name)
                for value in getattr(self, field_name)
            )
            object.__setattr__(self, field_name, values)
        object.__setattr__(
            self,
            "completed_at_utc",
            _timestamp(self.completed_at_utc, label="completed_at_utc"),
        )
        if self.authoritative_store is not True:
            raise ValueError("preflight must bind an authoritative store")
        if (
            self.final_started_count
            or self.final_projection_openings
            or self.completed_model_runs
            or self.prediction_artifact_hashes
            or self.score_artifact_hashes
        ):
            raise ValueError("dual preflight must prove zero FINAL history")

    def to_payload(self) -> dict[str, object]:
        return {
            "scientific_revision": self.scientific_revision,
            "shared_protocol_hash": self.shared_protocol_hash,
            "brier_release_hash": self.brier_release_hash,
            "log_release_hash": self.log_release_hash,
            "prediction_artifact_identity": self.prediction_artifact_identity,
            "ledger_genesis_hash": self.ledger_genesis_hash,
            "ledger_head_hash": self.ledger_head_hash,
            "final_started_count": self.final_started_count,
            "final_projection_openings": self.final_projection_openings,
            "completed_model_runs": self.completed_model_runs,
            "prediction_artifact_hashes": list(self.prediction_artifact_hashes),
            "score_artifact_hashes": list(self.score_artifact_hashes),
            "completed_at_utc": self.completed_at_utc,
            "authoritative_store": self.authoritative_store,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_payload())


def _require_exact_siblings(
    brier: TransferScoreRelease,
    log: TransferScoreRelease,
) -> None:
    if not isinstance(brier, TransferScoreRelease) or not isinstance(
        log,
        TransferScoreRelease,
    ):
        raise TypeError("preflight requires TransferScoreRelease siblings")
    if brier.score is not TransferScore.BRIER or log.score is not TransferScore.LOG:
        raise ValueError("score releases are not ordered BRIER then LOG")
    if brier.shared_identity_payload() != log.shared_identity_payload():
        raise ValueError("Brier and Log sibling shared identities differ")
    if brier.prediction_artifact_identity != log.prediction_artifact_identity:
        raise ValueError("score siblings do not share one prediction artifact identity")
    if brier.protocol.score_identity == log.protocol.score_identity:
        raise ValueError("score siblings require distinct score identities")


def preflight_transfer_releases(
    brier: TransferScoreRelease,
    log: TransferScoreRelease,
    *,
    store: object,
    completed_at_utc: str | None = None,
) -> DualTransferPreflight:
    _require_exact_siblings(brier, log)
    if not isinstance(store, GitTransferAttemptStore):
        raise TypeError("preflight requires GitTransferAttemptStore")
    head = store.head()
    history = store.history()
    if history.head_hash != head:
        raise ValueError("authoritative ledger history is not at the current head")
    if (
        history.events
        or history.final_started_count
        or history.final_projection_openings
        or history.completed_model_runs
        or history.prediction_artifact_hashes
        or history.score_artifact_hashes
    ):
        raise ValueError("authoritative ledger is not zero-FINAL at current head")
    timestamp = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        if completed_at_utc is None
        else completed_at_utc
    )
    return DualTransferPreflight(
        scientific_revision=brier.protocol.scientific_revision,
        shared_protocol_hash=stable_content_hash(brier.shared_identity_payload()),
        brier_release_hash=brier.content_hash,
        log_release_hash=log.content_hash,
        prediction_artifact_identity=brier.prediction_artifact_identity,
        ledger_genesis_hash=history.genesis_hash,
        ledger_head_hash=head,
        final_started_count=history.final_started_count,
        final_projection_openings=history.final_projection_openings,
        completed_model_runs=history.completed_model_runs,
        prediction_artifact_hashes=history.prediction_artifact_hashes,
        score_artifact_hashes=history.score_artifact_hashes,
        completed_at_utc=timestamp,
        authoritative_store=True,
    )

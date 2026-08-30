from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re

from narrative_dynamics.adapters.narrative_two_stage import (
    NarrativeTwoStageModelSource,
    create_narrative_two_stage_intentional_source,
    create_narrative_two_stage_planning_source,
    create_narrative_two_stage_reactive_source,
)
from narrative_dynamics.adapters.two_stage_metrics import (
    two_stage_brier_loss,
    two_stage_first_stage_policy_metrics,
    two_stage_log_loss,
)
from narrative_dynamics.attestation import RepositoryIdentity
from narrative_dynamics.candidate_execution import (
    CandidateExecutor,
    ProcessCandidateExecutor,
    SequentialCandidateExecutor,
)
from narrative_dynamics.contracts import Scenario, stable_content_hash
from narrative_dynamics.manifest import callable_identity, required_manifest_hash
from narrative_dynamics.measurement_validity import (
    ExactInvarianceFinding,
    MEASUREMENT_CLAIM_SCOPE,
    MeasurementAuditCase,
    MeasurementAuditInput,
    MeasurementCaseLoss,
    MeasurementModelPrediction,
    MeasurementPredictionArtifact,
    MeasurementRobustnessProfile,
    MeasurementScore,
    MeasurementSeedPrediction,
    MeasurementTerminalClass,
    MeasurementValidityProtocol,
    MeasurementValidityReport,
    MeasurementValidityStatus,
    average_seed_metrics,
    build_measurement_robustness_profile,
    permute_binary_metric_map,
    score_measurement_predictions,
)
from narrative_dynamics.losses import evaluate_metric_loss, metric_loss_identity
from narrative_dynamics.observations.dataset import ObservationPartitionRole
from narrative_dynamics.observations.preregistration import FrozenModelSpec
from narrative_dynamics.report_artifact import (
    AggregateReportArtifact,
    AttestedReport,
    attest_report,
)
from narrative_dynamics.simulation import SimulationRunner

from .feher_hare_two_stage_v1 import (
    PreparedFeherHareTwoStageV1,
    prepare_feher_hare_two_stage_v1,
)
from .two_stage_source import TwoStageSourceManifest


_TASKS = ("magic_carpet", "spaceship")
_MODELS = ("reactive", "intentional", "planning")
_ACTIONS = ("action_0", "action_1")
_CELL_ORDER = (
    (0, False),
    (0, True),
    (1, False),
    (1, True),
)
_SEEDS_BY_ROLE = {
    ObservationPartitionRole.TRAIN: (101, 102),
    ObservationPartitionRole.SELECTION_VALIDATION: (201, 202),
}
_DIAGNOSTIC_STATUSES = frozenset(
    {
        MeasurementValidityStatus.STABLE_UNDER_FROZEN_AUDIT,
        MeasurementValidityStatus.NOT_ESTABLISHED,
    }
)
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

FEHER_HARE_R3_LOCK_COMMIT = "f09d6afc96f0a720dd5e8e810f440cda0fe9f8a3"
_R3_SCIENTIFIC_REVISION = "d01232979cdfc9d902daab5f9e3e937079b56f69"
_R3_UPSTREAM_REVISION = "4567763780a2c596fd6510af720ec468a8214a8f"
_R3_IDENTITIES = {
    "source_manifest_hash": (
        "sha256:3609e980af172823cfef78290e7f2337fdb337d67f1f27641e17d473aeedd10d"
    ),
    "source_snapshot_hash": (
        "sha256:25bdc2e4bff4110f38b098b89e7d59aa38c9658727fcc89a38d50e107d243186"
    ),
    "transform_hash": (
        "sha256:2fb8ab6dc796a9e4ece5653a5485d865ef44dd0f5f7dff92d94a904932d6e541"
    ),
    "participant_assignment_hash": (
        "sha256:fc144c35f6713e27f75141d678872cc04ca44c7a0fd8e109e8618c72b4111ccf"
    ),
    "dataset_hash": (
        "sha256:17789130372d7eace05e1216a57bdae2ffbd519960333ffe814aee2d2d404781"
    ),
    "target_spec_hash": (
        "sha256:3134c9dc424418c87379f8e451420fb2defe81b8f30a45d67cd9b1f453349713"
    ),
    "train_partition_hash": (
        "sha256:71ce56243338eb23b9dd5ad6dd901e4b0d0be4989742b66bbe7ea4f025f207da"
    ),
    "selection_partition_hash": (
        "sha256:ecdcfa1681b58888ebf4419372d5de7b75be9624cdd3307144371c5486eb1347"
    ),
    "excluded_final_partition_hash": (
        "sha256:936ffe872644e888111007b300ea847484698be8fbfd7c2d54a177505a411047"
    ),
    "excluded_final_target_hash": (
        "sha256:927e1727d37993e9a5c887f79ac8712d07a155622deacf0a3122d41f90b16ed2"
    ),
    "train_selection_freeze_hash": (
        "sha256:77fa1bb80ab0c7ac737a09a8001f1d0c1432ce3bd991fd7191d07fdd0d8e05d5"
    ),
    "internal_lock_bundle_hash": (
        "sha256:96e553e557d6e7314eb0b9b1d0aaa8696e01d7a6ddec0abde949be8c8d45602f"
    ),
}
_R3_CANDIDATES = {
    "reactive": {
        "parameters": (("beta", 0.5),),
        "training_manifest_hash": (
            "sha256:5b607a4bc0a8082809c2446a8248fb8659b0d9ba178687ddad96dc74e3f19622"
        ),
        "selection_manifest_hash": (
            "sha256:e8414e301d19fc6acfd5accf055402ac995790f785402186c67ff95a48ee0c2e"
        ),
        "candidate_hash": (
            "sha256:42a84ef10c207159ff98397fe2bd86594df6c8be0f41b98e876f7ae33cba9456"
        ),
    },
    "intentional": {
        "parameters": (("beta", 2.0), ("memory_decay", 0.5)),
        "training_manifest_hash": (
            "sha256:47057311fe7450469c1710be49ba0e3b42aa7416fa2129f40761dbf8d237d4fc"
        ),
        "selection_manifest_hash": (
            "sha256:498a512cd931afe276f78bf135bd5c8269e051920d4b876177d0aa9ea250806d"
        ),
        "candidate_hash": (
            "sha256:68a01b5496e20095c1a603b30f1384bbe6a3ea22aac8cf74b7b97463c4935839"
        ),
    },
    "planning": {
        "parameters": (("beta", 4.0), ("memory_decay", 0.75)),
        "training_manifest_hash": (
            "sha256:ef0f3caf4f2a02b2d719bc302d7409fc8c2d937eb13517d906d83b38c29b76ac"
        ),
        "selection_manifest_hash": (
            "sha256:e347a3d7d523e2a35d0bb6b0f662343f5a01ddd1ff7d2efe95a8a5efa1986e74"
        ),
        "candidate_hash": (
            "sha256:c0ed956c1b285153075e5a22e7fb51f6d53dcf8ad326cce82e39e0221c2e877c"
        ),
    },
}


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _probability(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be a finite probability")
    return number


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return 0.0 if math.isclose(number, 0.0, rel_tol=0.0, abs_tol=1e-15) else number


def _status(
    value: MeasurementValidityStatus | str,
    *,
    label: str,
) -> MeasurementValidityStatus:
    try:
        status = (
            value
            if isinstance(value, MeasurementValidityStatus)
            else MeasurementValidityStatus(value)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is unsupported") from error
    if status not in _DIAGNOSTIC_STATUSES:
        raise ValueError(f"{label} is unsupported")
    return status


def _task(value: object, *, label: str) -> str:
    task = _text(value, label=label)
    if task not in _TASKS:
        raise ValueError(f"{label} is unsupported")
    return task


def _reward(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError("measurement diagnostic reward must be 0 or 1")
    return value


def _revision(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 40-character lowercase git revision")
    return value


def _content_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _parameters(value: object, *, label: str) -> tuple[tuple[str, float], ...]:
    try:
        rows = tuple(value)
    except TypeError as error:
        raise TypeError(f"{label} must be a sequence") from error
    canonical: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"{label} rows must be name/value pairs")
        name = _text(row[0], label=f"{label} name")
        if name in canonical:
            raise ValueError(f"{label} names must be unique")
        value_number = _finite(row[1], label=f"{label} {name!r}")
        canonical[name] = value_number
    if not canonical:
        raise ValueError(f"{label} must be non-empty")
    return tuple(sorted(canonical.items()))


@dataclass(frozen=True)
class FeherHareMeasurementCandidateRow:
    family: str
    parameters: tuple[tuple[str, float], ...]
    training_manifest_hash: str
    selection_manifest_hash: str
    candidate_hash: str

    def __post_init__(self) -> None:
        family = _text(
            self.family,
            label="Feher/Hare measurement candidate family",
        )
        if family not in _MODELS:
            raise ValueError("Feher/Hare measurement candidate family is unsupported")
        object.__setattr__(self, "family", family)
        parameters = _parameters(
            self.parameters,
            label=f"Feher/Hare {family} candidate parameters",
        )
        expected_names = (
            ("beta",)
            if family == "reactive"
            else ("beta", "memory_decay")
        )
        if tuple(name for name, _ in parameters) != expected_names:
            raise ValueError(
                f"Feher/Hare {family} candidate parameter schema changed"
            )
        object.__setattr__(self, "parameters", parameters)
        for attribute, label in (
            ("training_manifest_hash", "training manifest hash"),
            ("selection_manifest_hash", "selection manifest hash"),
            ("candidate_hash", "frozen candidate hash"),
        ):
            object.__setattr__(
                self,
                attribute,
                _content_hash(
                    getattr(self, attribute),
                    label=f"Feher/Hare {family} {label}",
                ),
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "parameters": [list(row) for row in self.parameters],
            "training_manifest_hash": self.training_manifest_hash,
            "selection_manifest_hash": self.selection_manifest_hash,
            "candidate_hash": self.candidate_hash,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "FeherHareMeasurementCandidateRow":
        if not isinstance(payload, Mapping):
            raise TypeError("Feher/Hare measurement candidate payload must be a mapping")
        expected = {
            "family",
            "parameters",
            "training_manifest_hash",
            "selection_manifest_hash",
            "candidate_hash",
        }
        missing = expected - set(payload)
        unknown = set(payload) - expected
        if missing:
            raise ValueError(
                "Feher/Hare measurement candidate payload has missing fields: "
                f"{tuple(sorted(missing))}"
            )
        if unknown:
            raise ValueError(
                "Feher/Hare measurement candidate payload has unknown fields: "
                f"{tuple(sorted(unknown))}"
            )
        return cls(
            family=payload["family"],
            parameters=tuple(tuple(row) for row in payload["parameters"]),
            training_manifest_hash=payload["training_manifest_hash"],
            selection_manifest_hash=payload["selection_manifest_hash"],
            candidate_hash=payload["candidate_hash"],
        )


@dataclass(frozen=True)
class FeherHareMeasurementAnchor:
    lock_commit: str
    scientific_repository_revision: str
    upstream_revision: str
    source_manifest_hash: str
    source_snapshot_hash: str
    transform_hash: str
    participant_assignment_hash: str
    dataset_hash: str
    target_spec_hash: str
    train_partition_hash: str
    selection_partition_hash: str
    excluded_final_partition_hash: str
    excluded_final_target_hash: str
    train_selection_freeze_hash: str
    internal_lock_bundle_hash: str
    candidate_rows: tuple[FeherHareMeasurementCandidateRow, ...]

    def __post_init__(self) -> None:
        for attribute, label in (
            ("lock_commit", "Feher/Hare measurement lock commit"),
            (
                "scientific_repository_revision",
                "Feher/Hare scientific repository revision",
            ),
            ("upstream_revision", "Feher/Hare upstream revision"),
        ):
            object.__setattr__(
                self,
                attribute,
                _revision(getattr(self, attribute), label=label),
            )
        for attribute in (
            "source_manifest_hash",
            "source_snapshot_hash",
            "transform_hash",
            "participant_assignment_hash",
            "dataset_hash",
            "target_spec_hash",
            "train_partition_hash",
            "selection_partition_hash",
            "excluded_final_partition_hash",
            "excluded_final_target_hash",
            "train_selection_freeze_hash",
            "internal_lock_bundle_hash",
        ):
            object.__setattr__(
                self,
                attribute,
                _content_hash(
                    getattr(self, attribute),
                    label=f"Feher/Hare measurement {attribute.replace('_', ' ')}",
                ),
            )
        rows = tuple(self.candidate_rows)
        if any(
            not isinstance(row, FeherHareMeasurementCandidateRow) for row in rows
        ):
            raise TypeError(
                "Feher/Hare measurement candidate rows have the wrong type"
            )
        rows = tuple(sorted(rows, key=lambda row: _MODELS.index(row.family)))
        if tuple(row.family for row in rows) != _MODELS:
            raise ValueError(
                "Feher/Hare measurement anchor requires all three candidate families"
            )
        if len({row.candidate_hash for row in rows}) != len(rows):
            raise ValueError(
                "Feher/Hare measurement candidate hashes must be unique"
            )
        object.__setattr__(self, "candidate_rows", rows)

    def to_payload(self) -> dict[str, object]:
        return {
            "lock_commit": self.lock_commit,
            "scientific_repository_revision": self.scientific_repository_revision,
            "upstream_revision": self.upstream_revision,
            "source_manifest_hash": self.source_manifest_hash,
            "source_snapshot_hash": self.source_snapshot_hash,
            "transform_hash": self.transform_hash,
            "participant_assignment_hash": self.participant_assignment_hash,
            "dataset_hash": self.dataset_hash,
            "target_spec_hash": self.target_spec_hash,
            "train_partition_hash": self.train_partition_hash,
            "selection_partition_hash": self.selection_partition_hash,
            "excluded_final_partition_hash": self.excluded_final_partition_hash,
            "excluded_final_target_hash": self.excluded_final_target_hash,
            "train_selection_freeze_hash": self.train_selection_freeze_hash,
            "internal_lock_bundle_hash": self.internal_lock_bundle_hash,
            "candidate_rows": [row.to_payload() for row in self.candidate_rows],
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "FeherHareMeasurementAnchor":
        if not isinstance(payload, Mapping):
            raise TypeError("Feher/Hare measurement anchor payload must be a mapping")
        expected = {
            "lock_commit",
            "scientific_repository_revision",
            "upstream_revision",
            "source_manifest_hash",
            "source_snapshot_hash",
            "transform_hash",
            "participant_assignment_hash",
            "dataset_hash",
            "target_spec_hash",
            "train_partition_hash",
            "selection_partition_hash",
            "excluded_final_partition_hash",
            "excluded_final_target_hash",
            "train_selection_freeze_hash",
            "internal_lock_bundle_hash",
            "candidate_rows",
        }
        missing = expected - set(payload)
        unknown = set(payload) - expected
        if missing:
            raise ValueError(
                "Feher/Hare measurement anchor payload has missing fields: "
                f"{tuple(sorted(missing))}"
            )
        if unknown:
            raise ValueError(
                "Feher/Hare measurement anchor payload has unknown fields: "
                f"{tuple(sorted(unknown))}"
            )
        return cls(
            lock_commit=payload["lock_commit"],
            scientific_repository_revision=payload[
                "scientific_repository_revision"
            ],
            upstream_revision=payload["upstream_revision"],
            source_manifest_hash=payload["source_manifest_hash"],
            source_snapshot_hash=payload["source_snapshot_hash"],
            transform_hash=payload["transform_hash"],
            participant_assignment_hash=payload["participant_assignment_hash"],
            dataset_hash=payload["dataset_hash"],
            target_spec_hash=payload["target_spec_hash"],
            train_partition_hash=payload["train_partition_hash"],
            selection_partition_hash=payload["selection_partition_hash"],
            excluded_final_partition_hash=payload[
                "excluded_final_partition_hash"
            ],
            excluded_final_target_hash=payload["excluded_final_target_hash"],
            train_selection_freeze_hash=payload[
                "train_selection_freeze_hash"
            ],
            internal_lock_bundle_hash=payload["internal_lock_bundle_hash"],
            candidate_rows=tuple(
                FeherHareMeasurementCandidateRow.from_payload(row)
                for row in payload["candidate_rows"]
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_payload())


def _mapping_field(
    payload: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing or invalid")
    return value


def _expect_identity(actual: object, expected: object, *, label: str) -> None:
    if actual != expected:
        raise ValueError(f"Feher/Hare {label} identity changed")


def _candidate_rows_from_lock(
    payload: Mapping[str, object],
) -> tuple[FeherHareMeasurementCandidateRow, ...]:
    freeze = _mapping_field(
        payload,
        "train_selection_freeze",
        label="Feher/Hare TRAIN/SELECTION freeze",
    )
    families = _mapping_field(
        freeze,
        "families",
        label="Feher/Hare TRAIN/SELECTION families",
    )
    if set(families) != set(_MODELS):
        raise ValueError("Feher/Hare frozen candidate family set changed")
    try:
        frozen_candidates = tuple(payload["frozen_candidates"])
    except (KeyError, TypeError) as error:
        raise ValueError("Feher/Hare frozen candidates are missing") from error
    by_name: dict[str, Mapping[str, object]] = {}
    for raw_candidate in frozen_candidates:
        if not isinstance(raw_candidate, Mapping):
            raise ValueError("Feher/Hare frozen candidate row is invalid")
        family = raw_candidate.get("name")
        if family in by_name or family not in _MODELS:
            raise ValueError("Feher/Hare frozen candidate family set changed")
        by_name[family] = raw_candidate
    if set(by_name) != set(_MODELS):
        raise ValueError("Feher/Hare frozen candidate family set changed")

    rows: list[FeherHareMeasurementCandidateRow] = []
    for family in _MODELS:
        family_row = families[family]
        if not isinstance(family_row, Mapping):
            raise ValueError(f"Feher/Hare {family} freeze row is invalid")
        frozen_row = by_name[family]
        family_parameters = _parameters(
            family_row.get("parameters"),
            label=f"Feher/Hare {family} freeze parameters",
        )
        frozen_parameters = _parameters(
            frozen_row.get("parameters"),
            label=f"Feher/Hare {family} frozen candidate parameters",
        )
        _expect_identity(
            frozen_parameters,
            family_parameters,
            label=f"{family} candidate parameters",
        )
        _expect_identity(
            frozen_row.get("selection_manifest_hash"),
            family_row.get("selection_manifest_hash"),
            label=f"{family} candidate selection manifest",
        )
        _expect_identity(
            frozen_row.get("content_hash"),
            family_row.get("frozen_candidate_hash"),
            label=f"{family} candidate",
        )
        expected = _R3_CANDIDATES[family]
        row = FeherHareMeasurementCandidateRow(
            family=family,
            parameters=family_parameters,
            training_manifest_hash=family_row.get("training_manifest_hash"),
            selection_manifest_hash=family_row.get("selection_manifest_hash"),
            candidate_hash=family_row.get("frozen_candidate_hash"),
        )
        for attribute in (
            "parameters",
            "training_manifest_hash",
            "selection_manifest_hash",
            "candidate_hash",
        ):
            _expect_identity(
                getattr(row, attribute),
                expected[attribute],
                label=f"{family} candidate {attribute.replace('_', ' ')}",
            )
        rows.append(row)
    return tuple(rows)


def load_feher_hare_r3_measurement_anchor(
    lock_path: Path,
    *,
    lock_commit: str,
) -> FeherHareMeasurementAnchor:
    if lock_commit != FEHER_HARE_R3_LOCK_COMMIT:
        raise ValueError("Feher/Hare measurement lock commit changed")
    path = Path(lock_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Feher/Hare R3 lock payload must be a mapping")

    _expect_identity(
        payload.get("scientific_repository_revision"),
        _R3_SCIENTIFIC_REVISION,
        label="scientific repository revision",
    )
    _expect_identity(
        payload.get("upstream_revision"),
        _R3_UPSTREAM_REVISION,
        label="upstream revision",
    )
    for field_name, label in (
        ("source_manifest_hash", "source manifest"),
        ("source_snapshot_hash", "source snapshot"),
        ("transform_hash", "transform"),
        ("participant_assignment_hash", "participant assignment"),
        ("dataset_hash", "dataset"),
        ("excluded_final_target_hash", "FINAL target"),
    ):
        source_field = (
            "final_target_hash"
            if field_name == "excluded_final_target_hash"
            else field_name
        )
        _expect_identity(
            payload.get(source_field),
            _R3_IDENTITIES[field_name],
            label=label,
        )

    freeze = _mapping_field(
        payload,
        "train_selection_freeze",
        label="Feher/Hare TRAIN/SELECTION freeze",
    )
    _expect_identity(
        freeze.get("artifact_payload_digest"),
        _R3_IDENTITIES["train_selection_freeze_hash"],
        label="TRAIN/SELECTION freeze",
    )
    brier = _mapping_field(
        payload,
        "brier_protocol",
        label="Feher/Hare Brier protocol",
    )
    protocol_fields = (
        ("dataset_hash", "dataset_hash", "dataset"),
        ("target_spec_hash", "target_spec_hash", "target spec"),
        ("train_partition_hash", "train_partition_hash", "TRAIN partition"),
        (
            "selection_partition_hash",
            "selection_partition_hash",
            "SELECTION partition",
        ),
        (
            "final_partition_hash",
            "excluded_final_partition_hash",
            "FINAL partition",
        ),
    )
    for source_field, anchor_field, label in protocol_fields:
        _expect_identity(
            brier.get(source_field),
            _R3_IDENTITIES[anchor_field],
            label=label,
        )
    log_protocol = payload.get("log_protocol")
    if log_protocol is not None:
        if not isinstance(log_protocol, Mapping):
            raise ValueError("Feher/Hare Log protocol is invalid")
        for source_field, _anchor_field, label in protocol_fields:
            _expect_identity(
                log_protocol.get(source_field),
                brier.get(source_field),
                label=f"Brier/Log {label}",
            )

    lock_bundle = _mapping_field(
        payload,
        "internal_lock_bundle",
        label="Feher/Hare internal lock bundle",
    )
    _expect_identity(
        lock_bundle.get("content_hash"),
        _R3_IDENTITIES["internal_lock_bundle_hash"],
        label="lock bundle",
    )
    repeated_bundle_fields = (
        ("dataset_hash", "dataset_hash", "dataset"),
        (
            "participant_assignment_hash",
            "participant_assignment_hash",
            "participant assignment",
        ),
        ("final_target_hash", "final_target_hash", "FINAL target"),
        (
            "scientific_repository_revision",
            "scientific_repository_revision",
            "scientific repository revision",
        ),
        ("source_manifest_hash", "source_manifest_hash", "source manifest"),
        ("source_snapshot_hash", "source_snapshot_hash", "source snapshot"),
        ("transform_hash", "transform_hash", "transform"),
        (
            "train_selection_freeze_digest",
            "train_selection_freeze",
            "TRAIN/SELECTION freeze",
        ),
    )
    for bundle_field, root_field, label in repeated_bundle_fields:
        if bundle_field in lock_bundle:
            if root_field == "train_selection_freeze":
                expected_value = freeze.get("artifact_payload_digest")
            else:
                expected_value = payload.get(root_field)
            _expect_identity(
                lock_bundle[bundle_field],
                expected_value,
                label=f"lock bundle {label}",
            )
    candidate_rows = _candidate_rows_from_lock(payload)
    if "frozen_candidate_hashes" in lock_bundle:
        _expect_identity(
            tuple(lock_bundle["frozen_candidate_hashes"]),
            tuple(row.candidate_hash for row in candidate_rows),
            label="lock bundle candidate",
        )

    return FeherHareMeasurementAnchor(
        lock_commit=lock_commit,
        scientific_repository_revision=payload[
            "scientific_repository_revision"
        ],
        upstream_revision=payload["upstream_revision"],
        source_manifest_hash=payload["source_manifest_hash"],
        source_snapshot_hash=payload["source_snapshot_hash"],
        transform_hash=payload["transform_hash"],
        participant_assignment_hash=payload["participant_assignment_hash"],
        dataset_hash=payload["dataset_hash"],
        target_spec_hash=brier["target_spec_hash"],
        train_partition_hash=brier["train_partition_hash"],
        selection_partition_hash=brier["selection_partition_hash"],
        excluded_final_partition_hash=brier["final_partition_hash"],
        excluded_final_target_hash=payload["final_target_hash"],
        train_selection_freeze_hash=freeze["artifact_payload_digest"],
        internal_lock_bundle_hash=lock_bundle["content_hash"],
        candidate_rows=candidate_rows,
    )


def freeze_feher_hare_measurement_candidates(
    anchor: FeherHareMeasurementAnchor,
) -> tuple[FrozenModelSpec, ...]:
    if not isinstance(anchor, FeherHareMeasurementAnchor):
        raise TypeError(
            "Feher/Hare measurement candidates require a measurement anchor"
        )
    sources = {
        "reactive": create_narrative_two_stage_reactive_source(),
        "intentional": create_narrative_two_stage_intentional_source(),
        "planning": create_narrative_two_stage_planning_source(),
    }
    candidates: list[FrozenModelSpec] = []
    for row in anchor.candidate_rows:
        candidate = FrozenModelSpec.freeze(
            name=row.family,
            model=sources[row.family],
            parameters=dict(row.parameters),
            selection_manifest_hash=row.selection_manifest_hash,
        )
        if candidate.content_hash != row.candidate_hash:
            raise ValueError(f"{row.family} frozen candidate identity changed")
        candidates.append(candidate)
    return tuple(candidates)


def _verify_prepared_anchor(
    prepared: PreparedFeherHareTwoStageV1,
    anchor: FeherHareMeasurementAnchor,
) -> None:
    checks = (
        (
            prepared.source_manifest.revision,
            anchor.upstream_revision,
            "upstream revision",
        ),
        (
            prepared.source_manifest.content_hash,
            anchor.source_manifest_hash,
            "source manifest",
        ),
        (
            prepared.transform_report.source_snapshot_hash,
            anchor.source_snapshot_hash,
            "source snapshot",
        ),
        (
            prepared.transform_report.content_hash,
            anchor.transform_hash,
            "transform",
        ),
        (
            prepared.assignment.content_hash,
            anchor.participant_assignment_hash,
            "participant assignment",
        ),
        (prepared.dataset.content_hash, anchor.dataset_hash, "dataset"),
        (
            prepared.target_spec.content_hash,
            anchor.target_spec_hash,
            "target spec",
        ),
        (
            prepared.dataset.partition(
                ObservationPartitionRole.TRAIN
            ).content_hash,
            anchor.train_partition_hash,
            "TRAIN partition",
        ),
        (
            prepared.dataset.partition(
                ObservationPartitionRole.SELECTION_VALIDATION
            ).content_hash,
            anchor.selection_partition_hash,
            "SELECTION partition",
        ),
        (
            prepared.dataset.partition(
                ObservationPartitionRole.FINAL_TEST
            ).content_hash,
            anchor.excluded_final_partition_hash,
            "FINAL partition",
        ),
        (
            prepared.final_targets.content_hash,
            anchor.excluded_final_target_hash,
            "FINAL target",
        ),
    )
    for actual, expected, label in checks:
        _expect_identity(actual, expected, label=label)
    target_checks = (
        (
            prepared.train_targets,
            ObservationPartitionRole.TRAIN,
            anchor.train_partition_hash,
            "TRAIN target",
        ),
        (
            prepared.selection_targets,
            ObservationPartitionRole.SELECTION_VALIDATION,
            anchor.selection_partition_hash,
            "SELECTION target",
        ),
        (
            prepared.final_targets,
            ObservationPartitionRole.FINAL_TEST,
            anchor.excluded_final_partition_hash,
            "FINAL target",
        ),
    )
    for report, role, partition_hash, label in target_checks:
        if report.role is not role:
            raise ValueError(f"Feher/Hare {label} role changed")
        _expect_identity(
            report.dataset_hash,
            anchor.dataset_hash,
            label=f"{label} dataset",
        )
        _expect_identity(
            report.spec_hash,
            anchor.target_spec_hash,
            label=f"{label} specification",
        )
        _expect_identity(
            report.partition_hash,
            partition_hash,
            label=f"{label} partition",
        )


def _project_prepared_measurement_input(
    prepared: PreparedFeherHareTwoStageV1,
    anchor: FeherHareMeasurementAnchor,
) -> MeasurementAuditInput:
    if not isinstance(prepared, PreparedFeherHareTwoStageV1):
        raise TypeError(
            "Feher/Hare measurement projection requires prepared Study V1 data"
        )
    if not isinstance(anchor, FeherHareMeasurementAnchor):
        raise TypeError(
            "Feher/Hare measurement projection requires a measurement anchor"
        )
    _verify_prepared_anchor(prepared, anchor)
    candidates = freeze_feher_hare_measurement_candidates(anchor)

    projected_cases: list[MeasurementAuditCase] = []
    reports = (
        prepared.train_targets,
        prepared.selection_targets,
    )
    for report in reports:
        partition = prepared.dataset.partition(report.role)
        records = {record.id: record for record in partition.records}
        if set(records) != {case.name for case in report.cases}:
            raise ValueError(
                f"Feher/Hare {report.role.value} target/record coverage changed"
            )
        for target_case in report.cases:
            record = records[target_case.name]
            if target_case.scenario.content_hash != record.scenario.content_hash:
                raise ValueError("Feher/Hare projected scenario identity changed")
            if target_case.record_hash != record.content_hash:
                raise ValueError("Feher/Hare projected record identity changed")
            task = _task(
                record.metadata.get("task_variant"),
                label="Feher/Hare projected task variant",
            )
            participant = _text(
                record.metadata.get("source_participant_id"),
                label="Feher/Hare projected participant",
            )
            trial_index = record.metadata.get("source_trial_id")
            if (
                isinstance(trial_index, bool)
                or not isinstance(trial_index, int)
                or trial_index < 0
            ):
                raise ValueError(
                    "Feher/Hare projected trial index must be non-negative"
                )
            if not isinstance(target_case.record_hash, str):
                raise ValueError("Feher/Hare projected record hash is missing")
            projected_cases.append(
                MeasurementAuditCase(
                    role=report.role,
                    scenario=Scenario(
                        id=(
                            "measurement-case-"
                            f"{target_case.record_hash.removeprefix('sha256:')}"
                        ),
                        payload=dict(target_case.scenario.payload),
                    ),
                    target=target_case.targets,
                    task_variant=task,
                    participant_group_hash=stable_content_hash(
                        (task, participant)
                    ),
                    trial_index=trial_index,
                    record_hash=target_case.record_hash,
                )
            )

    commitments = tuple(
        (
            role,
            stable_content_hash(
                tuple(
                    case.record_hash
                    for case in sorted(
                        (
                            item
                            for item in projected_cases
                            if item.role is role
                        ),
                        key=lambda item: item.case_hash,
                    )
                )
            ),
        )
        for role in (
            ObservationPartitionRole.TRAIN,
            ObservationPartitionRole.SELECTION_VALIDATION,
        )
    )
    return MeasurementAuditInput(
        claim_scope=MEASUREMENT_CLAIM_SCOPE,
        source_manifest_hash=anchor.source_manifest_hash,
        source_snapshot_hash=anchor.source_snapshot_hash,
        transform_hash=anchor.transform_hash,
        participant_assignment_hash=anchor.participant_assignment_hash,
        dataset_hash=anchor.dataset_hash,
        target_spec_hash=anchor.target_spec_hash,
        allowed_partition_hashes=(
            (ObservationPartitionRole.TRAIN, anchor.train_partition_hash),
            (
                ObservationPartitionRole.SELECTION_VALIDATION,
                anchor.selection_partition_hash,
            ),
        ),
        allowed_target_report_hashes=(
            (
                ObservationPartitionRole.TRAIN,
                prepared.train_targets.content_hash,
            ),
            (
                ObservationPartitionRole.SELECTION_VALIDATION,
                prepared.selection_targets.content_hash,
            ),
        ),
        allowed_case_commitments=commitments,
        frozen_candidates=candidates,
        excluded_final_partition_hash=anchor.excluded_final_partition_hash,
        excluded_final_target_hash=anchor.excluded_final_target_hash,
        cases=tuple(projected_cases),
    )


def provision_feher_hare_measurement_input(
    root: Path,
    manifest: TwoStageSourceManifest,
    anchor: FeherHareMeasurementAnchor,
) -> MeasurementAuditInput:
    if not isinstance(manifest, TwoStageSourceManifest):
        raise TypeError(
            "Feher/Hare measurement provisioner requires TwoStageSourceManifest"
        )
    prepared = prepare_feher_hare_two_stage_v1(
        root=Path(root),
        manifest=manifest,
    )
    return _project_prepared_measurement_input(prepared, anchor)


@dataclass(frozen=True)
class FeherHareMeasurementPredictionTask:
    family: str
    case_hash: str
    seed: int

    def __post_init__(self) -> None:
        family = _text(
            self.family,
            label="Feher/Hare measurement prediction task family",
        )
        if family not in _MODELS:
            raise ValueError(
                "Feher/Hare measurement prediction task family is unsupported"
            )
        object.__setattr__(self, "family", family)
        object.__setattr__(
            self,
            "case_hash",
            _content_hash(
                self.case_hash,
                label="Feher/Hare measurement prediction task case hash",
            ),
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError(
                "Feher/Hare measurement prediction task seed must be an integer"
            )
        if self.seed in (301, 302):
            raise ValueError(
                "Feher/Hare measurement prediction task cannot use FINAL seeds"
            )


@dataclass(frozen=True)
class _MeasurementWorkerCase:
    case_hash: str
    scenario: Scenario
    role: ObservationPartitionRole


@dataclass(frozen=True)
class _MeasurementWorkerContext:
    repository_identity: RepositoryIdentity
    family: str
    source: NarrativeTwoStageModelSource
    candidate: FrozenModelSpec
    cases: Mapping[str, _MeasurementWorkerCase]
    seeds_by_role: Mapping[ObservationPartitionRole, tuple[int, ...]]
    runner: SimulationRunner


_MEASUREMENT_WORKER_CONTEXT: _MeasurementWorkerContext | None = None


def _source_for_family(family: str) -> NarrativeTwoStageModelSource:
    sources = {
        "reactive": create_narrative_two_stage_reactive_source,
        "intentional": create_narrative_two_stage_intentional_source,
        "planning": create_narrative_two_stage_planning_source,
    }
    if family not in sources:
        raise ValueError("Feher/Hare measurement worker family is unsupported")
    return sources[family]()


def _thaw_worker_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_worker_value(item) for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_thaw_worker_value(item) for item in value)
    return value


def _measurement_worker_case_rows(
    audit_input: MeasurementAuditInput,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "case_hash": case.case_hash,
            "scenario_id": case.scenario.id,
            "scenario_payload": _thaw_worker_value(case.scenario.payload),
            "scenario_hash": case.scenario.content_hash,
            "role": case.role.value,
        }
        for case in audit_input.cases
    )


def _initialize_feher_hare_measurement_worker(
    repository_identity: RepositoryIdentity,
    family: str,
    parameters: tuple[tuple[str, float], ...],
    selection_manifest_hash: str,
    candidate_hash: str,
    case_rows: tuple[Mapping[str, object], ...],
    seeds_by_role: tuple[tuple[str, tuple[int, ...]], ...],
) -> None:
    if not isinstance(repository_identity, RepositoryIdentity):
        raise TypeError(
            "Feher/Hare measurement worker requires RepositoryIdentity"
        )
    source = _source_for_family(family)
    candidate = FrozenModelSpec.freeze(
        name=family,
        model=source,
        parameters=dict(parameters),
        selection_manifest_hash=selection_manifest_hash,
    )
    if candidate.content_hash != candidate_hash:
        raise ValueError(
            f"Feher/Hare measurement worker {family} candidate identity changed"
        )
    cases: dict[str, _MeasurementWorkerCase] = {}
    for raw_case in case_rows:
        if not isinstance(raw_case, Mapping):
            raise TypeError("Feher/Hare measurement worker case row is invalid")
        expected_fields = {
            "case_hash",
            "scenario_id",
            "scenario_payload",
            "scenario_hash",
            "role",
        }
        if set(raw_case) != expected_fields:
            raise ValueError("Feher/Hare measurement worker case schema changed")
        case_hash = _content_hash(
            raw_case["case_hash"],
            label="Feher/Hare measurement worker case hash",
        )
        scenario = Scenario(
            id=_text(
                raw_case["scenario_id"],
                label="Feher/Hare measurement worker scenario id",
            ),
            payload=raw_case["scenario_payload"],
        )
        if scenario.content_hash != raw_case["scenario_hash"]:
            raise ValueError(
                "Feher/Hare measurement worker scenario identity changed"
            )
        try:
            role = ObservationPartitionRole(raw_case["role"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Feher/Hare measurement worker role is unsupported"
            ) from error
        if role is ObservationPartitionRole.FINAL_TEST:
            raise ValueError("Feher/Hare measurement worker cannot receive FINAL cases")
        if case_hash in cases:
            raise ValueError("Feher/Hare measurement worker case hashes must be unique")
        cases[case_hash] = _MeasurementWorkerCase(
            case_hash=case_hash,
            scenario=scenario,
            role=role,
        )
    if not cases:
        raise ValueError("Feher/Hare measurement worker cases must be non-empty")

    seed_map: dict[ObservationPartitionRole, tuple[int, ...]] = {}
    for raw_role, raw_seeds in seeds_by_role:
        try:
            role = ObservationPartitionRole(raw_role)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Feher/Hare measurement worker seed role is unsupported"
            ) from error
        seeds = tuple(raw_seeds)
        if (
            role is ObservationPartitionRole.FINAL_TEST
            or not seeds
            or any(
                isinstance(seed, bool) or not isinstance(seed, int)
                for seed in seeds
            )
            or len(set(seeds)) != len(seeds)
            or any(seed in (301, 302) for seed in seeds)
        ):
            raise ValueError("Feher/Hare measurement worker seed plan changed")
        if role in seed_map:
            raise ValueError("Feher/Hare measurement worker seed roles must be unique")
        seed_map[role] = seeds
    if set(seed_map) != {
        ObservationPartitionRole.TRAIN,
        ObservationPartitionRole.SELECTION_VALIDATION,
    }:
        raise ValueError("Feher/Hare measurement worker seed coverage changed")

    global _MEASUREMENT_WORKER_CONTEXT
    _MEASUREMENT_WORKER_CONTEXT = _MeasurementWorkerContext(
        repository_identity=repository_identity,
        family=family,
        source=source,
        candidate=candidate,
        cases=cases,
        seeds_by_role=seed_map,
        runner=SimulationRunner(repository_identity=repository_identity),
    )


def _evaluate_feher_hare_measurement_prediction(
    task: FeherHareMeasurementPredictionTask,
) -> dict[str, object]:
    context = _MEASUREMENT_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("Feher/Hare measurement worker is not initialized")
    if not isinstance(task, FeherHareMeasurementPredictionTask):
        raise TypeError(
            "Feher/Hare measurement worker requires a prediction task"
        )
    if task.family != context.family:
        raise ValueError(
            "Feher/Hare measurement worker task family changed after initialization"
        )
    if task.case_hash not in context.cases:
        raise ValueError("Feher/Hare measurement worker task case is unknown")
    case = context.cases[task.case_hash]
    if task.seed not in context.seeds_by_role[case.role]:
        raise ValueError(
            "Feher/Hare measurement worker task seed changed for its role"
        )
    trace = context.runner.run_once(
        context.source,
        case.scenario,
        dict(context.candidate.parameters),
        seed=task.seed,
    )
    if trace.scenario_id != case.scenario.id:
        raise RuntimeError(
            "Feher/Hare measurement prediction trace scenario changed"
        )
    if trace.seed != task.seed:
        raise RuntimeError("Feher/Hare measurement prediction trace seed changed")
    if trace.parameters != context.candidate.parameters:
        raise RuntimeError(
            "Feher/Hare measurement prediction trace parameters changed"
        )
    metrics = tuple(sorted(two_stage_first_stage_policy_metrics(trace).items()))
    return {
        "repository_identity": context.repository_identity.manifest_identity(),
        "family": context.family,
        "candidate_hash": context.candidate.content_hash,
        "case_hash": case.case_hash,
        "scenario_hash": case.scenario.content_hash,
        "role": case.role.value,
        "seed": task.seed,
        "metrics": metrics,
        "run_manifest_hash": required_manifest_hash(
            trace,
            label="Feher/Hare measurement prediction trace",
        ),
    }


def _measurement_seed_prediction_from_payload(
    payload: object,
    *,
    repository_identity: RepositoryIdentity,
    expected_family: str,
    expected_candidate_hash: str,
) -> MeasurementSeedPrediction:
    if not isinstance(payload, Mapping):
        raise TypeError(
            "Feher/Hare measurement worker must return a mapping payload"
        )
    expected_fields = {
        "repository_identity",
        "family",
        "candidate_hash",
        "case_hash",
        "scenario_hash",
        "role",
        "seed",
        "metrics",
        "run_manifest_hash",
    }
    if set(payload) != expected_fields:
        raise ValueError("Feher/Hare measurement worker payload schema changed")
    if payload["repository_identity"] != repository_identity.manifest_identity():
        raise ValueError(
            "Feher/Hare measurement worker repository identity changed"
        )
    if payload["family"] != expected_family:
        raise ValueError("Feher/Hare measurement worker family changed")
    if payload["candidate_hash"] != expected_candidate_hash:
        raise ValueError("Feher/Hare measurement worker candidate changed")
    return MeasurementSeedPrediction(
        case_hash=payload["case_hash"],
        scenario_hash=payload["scenario_hash"],
        role=payload["role"],
        model_name=payload["family"],
        seed=payload["seed"],
        metrics=tuple(tuple(row) for row in payload["metrics"]),
        run_manifest_hash=payload["run_manifest_hash"],
    )


def execute_feher_hare_measurement_predictions(
    repository_identity: RepositoryIdentity,
    audit_input: MeasurementAuditInput,
    protocol: MeasurementValidityProtocol,
    executor: CandidateExecutor,
) -> MeasurementPredictionArtifact:
    if not isinstance(repository_identity, RepositoryIdentity):
        raise TypeError(
            "Feher/Hare measurement prediction requires RepositoryIdentity"
        )
    if not isinstance(audit_input, MeasurementAuditInput):
        raise TypeError(
            "Feher/Hare measurement prediction requires MeasurementAuditInput"
        )
    if not isinstance(protocol, MeasurementValidityProtocol):
        raise TypeError(
            "Feher/Hare measurement prediction requires MeasurementValidityProtocol"
        )
    execute = getattr(executor, "execute", None)
    if not callable(execute):
        raise TypeError(
            "Feher/Hare measurement prediction requires a CandidateExecutor"
        )
    if protocol.empirical_anchor_hash != audit_input.empirical_anchor_hash:
        raise ValueError("Feher/Hare measurement protocol empirical anchor changed")
    if protocol.candidate_hashes != tuple(
        candidate.content_hash for candidate in audit_input.frozen_candidates
    ):
        raise ValueError("Feher/Hare measurement protocol candidates changed")
    if protocol.allowed_roles != (
        ObservationPartitionRole.TRAIN,
        ObservationPartitionRole.SELECTION_VALIDATION,
    ):
        raise ValueError("Feher/Hare measurement protocol roles changed")
    seeds_by_role = dict(protocol.seeds_by_role)
    if any(seed in (301, 302) for seeds in seeds_by_role.values() for seed in seeds):
        raise ValueError("Feher/Hare measurement protocol contains FINAL seeds")

    candidates = {
        candidate.name: candidate for candidate in audit_input.frozen_candidates
    }
    if tuple(candidates) != _MODELS:
        raise ValueError("Feher/Hare measurement candidate family set changed")
    case_rows = _measurement_worker_case_rows(audit_input)
    seed_rows = tuple(
        (role.value, seeds) for role, seeds in protocol.seeds_by_role
    )
    expected_keys = {
        (case.case_hash, seed)
        for case in audit_input.cases
        for seed in seeds_by_role[case.role]
    }

    models: list[MeasurementModelPrediction] = []
    all_run_manifest_hashes: list[str] = []
    for family in _MODELS:
        candidate = candidates[family]
        tasks = tuple(
            FeherHareMeasurementPredictionTask(
                family=family,
                case_hash=case.case_hash,
                seed=seed,
            )
            for case in audit_input.cases
            for seed in seeds_by_role[case.role]
        )
        raw_results = tuple(
            execute(
                _evaluate_feher_hare_measurement_prediction,
                tasks,
                initializer=_initialize_feher_hare_measurement_worker,
                initargs=(
                    repository_identity,
                    family,
                    candidate.parameters,
                    candidate.selection_manifest_hash,
                    candidate.content_hash,
                    case_rows,
                    seed_rows,
                ),
            )
        )
        predictions = tuple(
            _measurement_seed_prediction_from_payload(
                payload,
                repository_identity=repository_identity,
                expected_family=family,
                expected_candidate_hash=candidate.content_hash,
            )
            for payload in raw_results
        )
        keys = tuple((row.case_hash, row.seed) for row in predictions)
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"Feher/Hare measurement {family} worker returned a duplicate row"
            )
        if set(keys) != expected_keys or len(keys) != len(expected_keys):
            raise ValueError(
                f"Feher/Hare measurement {family} worker coverage changed"
            )
        models.append(
            MeasurementModelPrediction(
                model_name=family,
                candidate_hash=candidate.content_hash,
                rows=predictions,
            )
        )
        all_run_manifest_hashes.extend(
            row.run_manifest_hash for row in predictions
        )
    return MeasurementPredictionArtifact(
        protocol_hash=protocol.content_hash,
        audit_input_hash=audit_input.content_hash,
        models=tuple(models),
        execution_manifest_hashes=tuple(all_run_manifest_hashes),
    )


_SEMANTIC_METRICS = (
    "first_stage.action_0",
    "first_stage.action_1",
)
_SEMANTIC_SEED = 41
_SEMANTIC_VARIANTS = ("original", "transformed")


@dataclass(frozen=True)
class FeherHareSemanticFixture:
    name: str
    original_scenario: Scenario
    transformed_scenario: Scenario
    inverse_metric_permutation: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _text(self.name, label="Feher/Hare semantic fixture name"),
        )
        if not isinstance(self.original_scenario, Scenario) or not isinstance(
            self.transformed_scenario,
            Scenario,
        ):
            raise TypeError("Feher/Hare semantic fixture requires Scenario values")
        try:
            raw_permutation = tuple(self.inverse_metric_permutation)
        except TypeError as error:
            raise TypeError(
                "Feher/Hare semantic metric permutation must be a sequence"
            ) from error
        permutation: dict[str, str] = {}
        for row in raw_permutation:
            if not isinstance(row, tuple) or len(row) != 2:
                raise ValueError(
                    "Feher/Hare semantic metric permutation rows must be pairs"
                )
            source = _text(
                row[0],
                label="Feher/Hare semantic metric permutation source",
            )
            destination = _text(
                row[1],
                label="Feher/Hare semantic metric permutation destination",
            )
            if source in permutation:
                raise ValueError(
                    "Feher/Hare semantic metric permutation sources must be unique"
                )
            permutation[source] = destination
        if set(permutation) != set(_SEMANTIC_METRICS) or set(
            permutation.values()
        ) != set(_SEMANTIC_METRICS):
            raise ValueError(
                "Feher/Hare semantic metric permutation must be a binary bijection"
            )
        object.__setattr__(
            self,
            "inverse_metric_permutation",
            tuple((name, permutation[name]) for name in _SEMANTIC_METRICS),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "original_scenario": {
                "id": self.original_scenario.id,
                "content_hash": self.original_scenario.content_hash,
            },
            "transformed_scenario": {
                "id": self.transformed_scenario.id,
                "content_hash": self.transformed_scenario.content_hash,
            },
            "inverse_metric_permutation": self.inverse_metric_permutation,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _frozen_semantic_history() -> tuple[dict[str, object], ...]:
    return (
        {
            "trial_id": 0,
            "first_stage_action": "action_0",
            "transition_common": True,
            "final_state": "state_0",
            "second_stage_action": "action_0",
            "reward": 1,
        },
        {
            "trial_id": 1,
            "first_stage_action": "action_1",
            "transition_common": False,
            "final_state": "state_0",
            "second_stage_action": "action_1",
            "reward": 0,
        },
        {
            "trial_id": 2,
            "first_stage_action": "action_0",
            "transition_common": False,
            "final_state": "state_1",
            "second_stage_action": "action_1",
            "reward": 1,
        },
    )


def _relabel_semantic_history(
    history: object,
    *,
    first_stage: bool = False,
    second_stage: bool = False,
    final_state: bool = False,
) -> tuple[dict[str, object], ...]:
    action_swap = {"action_0": "action_1", "action_1": "action_0"}
    state_swap = {"state_0": "state_1", "state_1": "state_0"}
    rows: list[dict[str, object]] = []
    for raw_row in tuple(history):
        if not isinstance(raw_row, Mapping):
            raise TypeError("Feher/Hare semantic history rows must be mappings")
        row = dict(raw_row)
        if first_stage:
            row["first_stage_action"] = action_swap[row["first_stage_action"]]
        if second_stage:
            row["second_stage_action"] = action_swap[row["second_stage_action"]]
        if final_state:
            row["final_state"] = state_swap[row["final_state"]]
        rows.append(row)
    return tuple(rows)


def _swapped_configuration(
    configuration: object,
    first_key: str,
    second_key: str,
) -> tuple[tuple[str, object], ...]:
    values = dict(tuple(configuration))
    if set(values) != {first_key, second_key}:
        raise ValueError("Feher/Hare semantic configuration schema changed")
    values[first_key], values[second_key] = values[second_key], values[first_key]
    return tuple((key, values[key]) for key in sorted(values))


def _semantic_scenario(
    fixture_name: str,
    variant: str,
    *,
    task_variant: str,
    configuration: tuple[tuple[str, object], ...],
    history: tuple[dict[str, object], ...],
) -> Scenario:
    return Scenario(
        id=f"measurement-semantic-{fixture_name}-{variant}",
        payload={
            "task_variant": task_variant,
            "first_stage_configuration": configuration,
            "history": history,
        },
    )


def frozen_feher_hare_semantic_fixtures() -> tuple[FeherHareSemanticFixture, ...]:
    history = _frozen_semantic_history()
    magic_configuration = (
        ("action_0_position", "left"),
        ("action_1_position", "right"),
    )
    swapped_magic_configuration = _swapped_configuration(
        magic_configuration,
        "action_0_position",
        "action_1_position",
    )
    spaceship_configuration = (("symbol0", 0), ("symbol1", 1))
    swapped_spaceship_configuration = _swapped_configuration(
        spaceship_configuration,
        "symbol0",
        "symbol1",
    )
    identity = tuple((name, name) for name in _SEMANTIC_METRICS)
    swap = (
        ("first_stage.action_0", "first_stage.action_1"),
        ("first_stage.action_1", "first_stage.action_0"),
    )

    coherent_name = "coherent_action_state_relabel"
    second_stage_name = "second_stage_action_relabel"
    magic_name = "magic_carpet_counterbalance"
    spaceship_name = "spaceship_symbol_order"
    post_choice_name = "post_choice_outcome_exclusion"
    post_choice_payload = {
        "task_variant": "magic_carpet",
        "first_stage_configuration": magic_configuration,
        "history": history,
    }
    return (
        FeherHareSemanticFixture(
            name=coherent_name,
            original_scenario=_semantic_scenario(
                coherent_name,
                "original",
                task_variant="magic_carpet",
                configuration=magic_configuration,
                history=history,
            ),
            transformed_scenario=_semantic_scenario(
                coherent_name,
                "transformed",
                task_variant="magic_carpet",
                configuration=swapped_magic_configuration,
                history=_relabel_semantic_history(
                    history,
                    first_stage=True,
                    second_stage=True,
                    final_state=True,
                ),
            ),
            inverse_metric_permutation=swap,
        ),
        FeherHareSemanticFixture(
            name=second_stage_name,
            original_scenario=_semantic_scenario(
                second_stage_name,
                "original",
                task_variant="magic_carpet",
                configuration=magic_configuration,
                history=history,
            ),
            transformed_scenario=_semantic_scenario(
                second_stage_name,
                "transformed",
                task_variant="magic_carpet",
                configuration=magic_configuration,
                history=_relabel_semantic_history(history, second_stage=True),
            ),
            inverse_metric_permutation=identity,
        ),
        FeherHareSemanticFixture(
            name=magic_name,
            original_scenario=_semantic_scenario(
                magic_name,
                "original",
                task_variant="magic_carpet",
                configuration=magic_configuration,
                history=history,
            ),
            transformed_scenario=_semantic_scenario(
                magic_name,
                "transformed",
                task_variant="magic_carpet",
                configuration=swapped_magic_configuration,
                history=history,
            ),
            inverse_metric_permutation=identity,
        ),
        FeherHareSemanticFixture(
            name=spaceship_name,
            original_scenario=_semantic_scenario(
                spaceship_name,
                "original",
                task_variant="spaceship",
                configuration=spaceship_configuration,
                history=history,
            ),
            transformed_scenario=_semantic_scenario(
                spaceship_name,
                "transformed",
                task_variant="spaceship",
                configuration=swapped_spaceship_configuration,
                history=history,
            ),
            inverse_metric_permutation=identity,
        ),
        FeherHareSemanticFixture(
            name=post_choice_name,
            original_scenario=Scenario(
                id=f"measurement-semantic-{post_choice_name}-before-outcome-a",
                payload=post_choice_payload,
            ),
            transformed_scenario=Scenario(
                id=f"measurement-semantic-{post_choice_name}-before-outcome-b",
                payload=post_choice_payload,
            ),
            inverse_metric_permutation=identity,
        ),
    )


@dataclass(frozen=True)
class _FeherHareSemanticExecutionTask:
    repository_identity: RepositoryIdentity
    family: str
    parameters: tuple[tuple[str, float], ...]
    selection_manifest_hash: str
    candidate_hash: str
    fixture_name: str
    variant: str
    scenario_id: str
    scenario_payload: Mapping[str, object]
    scenario_hash: str


_SEMANTIC_WORKER_CACHE: dict[
    tuple[str, str, str],
    tuple[NarrativeTwoStageModelSource, FrozenModelSpec, SimulationRunner],
] = {}


def _evaluate_feher_hare_semantic_task(
    task: _FeherHareSemanticExecutionTask,
) -> dict[str, object]:
    if not isinstance(task, _FeherHareSemanticExecutionTask):
        raise TypeError("Feher/Hare semantic worker requires a semantic task")
    if not isinstance(task.repository_identity, RepositoryIdentity):
        raise TypeError("Feher/Hare semantic worker requires RepositoryIdentity")
    if task.family not in _MODELS:
        raise ValueError("Feher/Hare semantic worker family changed")
    if task.variant not in _SEMANTIC_VARIANTS:
        raise ValueError("Feher/Hare semantic worker variant changed")
    cache_key = (
        task.repository_identity.content_hash,
        task.family,
        task.candidate_hash,
    )
    context = _SEMANTIC_WORKER_CACHE.get(cache_key)
    if context is None:
        source = _source_for_family(task.family)
        candidate = FrozenModelSpec.freeze(
            name=task.family,
            model=source,
            parameters=dict(task.parameters),
            selection_manifest_hash=task.selection_manifest_hash,
        )
        if candidate.content_hash != task.candidate_hash:
            raise ValueError(
                f"Feher/Hare semantic {task.family} candidate identity changed"
            )
        context = (
            source,
            candidate,
            SimulationRunner(repository_identity=task.repository_identity),
        )
        _SEMANTIC_WORKER_CACHE[cache_key] = context
    source, candidate, runner = context
    scenario = Scenario(
        id=task.scenario_id,
        payload=_thaw_worker_value(task.scenario_payload),
    )
    if scenario.content_hash != task.scenario_hash:
        raise ValueError("Feher/Hare semantic scenario identity changed")
    trace = runner.run_once(
        source,
        scenario,
        dict(candidate.parameters),
        seed=_SEMANTIC_SEED,
    )
    return {
        "repository_identity": task.repository_identity.manifest_identity(),
        "fixture_name": task.fixture_name,
        "variant": task.variant,
        "family": task.family,
        "candidate_hash": candidate.content_hash,
        "scenario_hash": scenario.content_hash,
        "metrics": tuple(
            sorted(two_stage_first_stage_policy_metrics(trace).items())
        ),
        "run_manifest_hash": required_manifest_hash(
            trace,
            label="Feher/Hare semantic trace",
        ),
    }


def _semantic_candidate_map(
    candidates: object,
) -> dict[str, FrozenModelSpec]:
    try:
        rows = tuple(candidates)
    except TypeError as error:
        raise TypeError("Feher/Hare semantic candidates must be a sequence") from error
    if len(rows) != len(_MODELS) or any(
        not isinstance(row, FrozenModelSpec) for row in rows
    ):
        raise TypeError(
            "Feher/Hare semantic candidates must contain three FrozenModelSpec values"
        )
    by_name = {row.name: row for row in rows}
    if set(by_name) != set(_MODELS) or len(by_name) != len(rows):
        raise ValueError(
            "Feher/Hare semantic candidates must be reactive, intentional, and planning"
        )
    for family in _MODELS:
        candidate = by_name[family]
        reconstructed = FrozenModelSpec.freeze(
            name=family,
            model=_source_for_family(family),
            parameters=dict(candidate.parameters),
            selection_manifest_hash=candidate.selection_manifest_hash,
        )
        if reconstructed.content_hash != candidate.content_hash:
            raise ValueError(
                f"Feher/Hare semantic {family} candidate identity changed"
            )
    return {family: by_name[family] for family in _MODELS}


def _semantic_result_rows(results: object) -> tuple[dict[str, object], ...]:
    try:
        raw_rows = tuple(results)
    except TypeError as error:
        raise TypeError("Feher/Hare semantic results must be a sequence") from error
    expected_fields = {
        "repository_identity",
        "fixture_name",
        "variant",
        "family",
        "candidate_hash",
        "scenario_hash",
        "metrics",
        "run_manifest_hash",
    }
    rows: list[dict[str, object]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping) or set(raw_row) != expected_fields:
            raise ValueError("Feher/Hare semantic worker result schema changed")
        row = dict(raw_row)
        if row["family"] not in _MODELS or row["variant"] not in _SEMANTIC_VARIANTS:
            raise ValueError("Feher/Hare semantic worker result coordinate changed")
        rows.append(row)
    coordinate_keys = tuple(
        (row["fixture_name"], row["family"], row["variant"])
        for row in rows
    )
    if len(set(coordinate_keys)) != len(coordinate_keys):
        raise ValueError("Feher/Hare semantic worker returned duplicate results")
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                str(row["fixture_name"]),
                _MODELS.index(str(row["family"])),
                _SEMANTIC_VARIANTS.index(str(row["variant"])),
            ),
        )
    )


def _canonical_semantic_scenario_payload(
    fixture: FeherHareSemanticFixture,
    scenario: Scenario,
    *,
    transformed: bool,
) -> object:
    payload = _thaw_worker_value(scenario.payload)
    if not isinstance(payload, dict):
        raise TypeError("Feher/Hare semantic scenario payload must be a mapping")
    if not transformed or fixture.name == "post_choice_outcome_exclusion":
        return payload
    if fixture.name == "coherent_action_state_relabel":
        payload["first_stage_configuration"] = _swapped_configuration(
            payload["first_stage_configuration"],
            "action_0_position",
            "action_1_position",
        )
        payload["history"] = _relabel_semantic_history(
            payload["history"],
            first_stage=True,
            second_stage=True,
            final_state=True,
        )
    elif fixture.name == "second_stage_action_relabel":
        payload["history"] = _relabel_semantic_history(
            payload["history"],
            second_stage=True,
        )
    elif fixture.name == "magic_carpet_counterbalance":
        payload["first_stage_configuration"] = _swapped_configuration(
            payload["first_stage_configuration"],
            "action_0_position",
            "action_1_position",
        )
    elif fixture.name == "spaceship_symbol_order":
        payload["first_stage_configuration"] = _swapped_configuration(
            payload["first_stage_configuration"],
            "symbol0",
            "symbol1",
        )
    else:
        raise ValueError("Feher/Hare semantic fixture is unsupported")
    return payload


def _semantic_loss_payload(metrics: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    target = {
        "first_stage.action_0": 1.0,
        "first_stage.action_1": 0.0,
    }
    return (
        (
            MeasurementScore.BRIER.value,
            evaluate_metric_loss(two_stage_brier_loss(), metrics, target),
        ),
        (
            MeasurementScore.LOG.value,
            evaluate_metric_loss(two_stage_log_loss(), metrics, target),
        ),
    )


def _exact_invariance_finding(
    check_name: str,
    original_payload: object,
    transformed_payload: object,
    *,
    details_payload: object,
) -> ExactInvarianceFinding:
    original_hash = stable_content_hash(original_payload)
    transformed_hash = stable_content_hash(transformed_payload)
    return ExactInvarianceFinding(
        check_name=check_name,
        status=(
            MeasurementValidityStatus.EXACT_INVARIANCE_MET
            if original_hash == transformed_hash
            else MeasurementValidityStatus.EXACT_INVARIANCE_FAILED
        ),
        original_hash=original_hash,
        transformed_hash=transformed_hash,
        details_hash=stable_content_hash(details_payload),
    )


def evaluate_feher_hare_semantic_invariance(
    repository_identity: RepositoryIdentity,
    candidates: object,
) -> tuple[ExactInvarianceFinding, ...]:
    if not isinstance(repository_identity, RepositoryIdentity):
        raise TypeError(
            "Feher/Hare semantic invariance requires RepositoryIdentity"
        )
    candidate_map = _semantic_candidate_map(candidates)
    fixtures = frozen_feher_hare_semantic_fixtures()
    if not fixtures or any(
        not isinstance(row, FeherHareSemanticFixture) for row in fixtures
    ):
        raise TypeError(
            "Feher/Hare semantic fixtures must contain FeherHareSemanticFixture values"
        )
    if len({row.name for row in fixtures}) != len(fixtures):
        raise ValueError("Feher/Hare semantic fixture names must be unique")

    tasks = tuple(
        _FeherHareSemanticExecutionTask(
            repository_identity=repository_identity,
            family=family,
            parameters=candidate_map[family].parameters,
            selection_manifest_hash=candidate_map[family].selection_manifest_hash,
            candidate_hash=candidate_map[family].content_hash,
            fixture_name=fixture.name,
            variant=variant,
            scenario_id=scenario.id,
            scenario_payload=_thaw_worker_value(scenario.payload),
            scenario_hash=scenario.content_hash,
        )
        for fixture in fixtures
        for family in _MODELS
        for variant, scenario in (
            ("original", fixture.original_scenario),
            ("transformed", fixture.transformed_scenario),
        )
    )
    sequential_rows = _semantic_result_rows(
        SequentialCandidateExecutor().execute(
            _evaluate_feher_hare_semantic_task,
            tasks,
        )
    )
    process_rows = _semantic_result_rows(
        ProcessCandidateExecutor(max_workers=2).execute(
            _evaluate_feher_hare_semantic_task,
            tasks,
        )
    )
    expected_coordinates = {
        (fixture.name, family, variant)
        for fixture in fixtures
        for family in _MODELS
        for variant in _SEMANTIC_VARIANTS
    }
    if {
        (row["fixture_name"], row["family"], row["variant"])
        for row in sequential_rows
    } != expected_coordinates or len(sequential_rows) != len(expected_coordinates):
        raise ValueError("Feher/Hare semantic sequential coverage changed")
    if {
        (row["fixture_name"], row["family"], row["variant"])
        for row in process_rows
    } != expected_coordinates or len(process_rows) != len(expected_coordinates):
        raise ValueError("Feher/Hare semantic process coverage changed")

    sequential_map = {
        (row["fixture_name"], row["family"], row["variant"]): row
        for row in sequential_rows
    }
    original_payload: list[dict[str, object]] = []
    transformed_payload: list[dict[str, object]] = []
    for fixture in fixtures:
        permutation = dict(fixture.inverse_metric_permutation)
        for family in _MODELS:
            original = sequential_map[(fixture.name, family, "original")]
            transformed = sequential_map[(fixture.name, family, "transformed")]
            original_metrics = dict(original["metrics"])
            transformed_metrics = dict(transformed["metrics"])
            canonical_transformed_metrics = {
                destination: transformed_metrics[source]
                for source, destination in permutation.items()
            }
            original_payload.append(
                {
                    "fixture_name": fixture.name,
                    "family": family,
                    "candidate_hash": original["candidate_hash"],
                    "scenario": _canonical_semantic_scenario_payload(
                        fixture,
                        fixture.original_scenario,
                        transformed=False,
                    ),
                    "metrics": tuple(sorted(original_metrics.items())),
                    "losses": _semantic_loss_payload(original_metrics),
                }
            )
            transformed_payload.append(
                {
                    "fixture_name": fixture.name,
                    "family": family,
                    "candidate_hash": transformed["candidate_hash"],
                    "scenario": _canonical_semantic_scenario_payload(
                        fixture,
                        fixture.transformed_scenario,
                        transformed=True,
                    ),
                    "metrics": tuple(
                        sorted(canonical_transformed_metrics.items())
                    ),
                    "losses": _semantic_loss_payload(
                        canonical_transformed_metrics
                    ),
                }
            )

    task_finding = _exact_invariance_finding(
        "task_canonicalization",
        tuple(original_payload),
        tuple(transformed_payload),
        details_payload={
            "semantic_seed": _SEMANTIC_SEED,
            "fixture_hashes": tuple(row.content_hash for row in fixtures),
            "candidate_hashes": tuple(
                candidate_map[family].content_hash for family in _MODELS
            ),
        },
    )
    executor_finding = _exact_invariance_finding(
        "semantic_executor_invariance",
        sequential_rows,
        process_rows,
        details_payload={
            "semantic_seed": _SEMANTIC_SEED,
            "sequential_executor": "SequentialCandidateExecutor",
            "process_executor": "ProcessCandidateExecutor(max_workers=2,spawn)",
            "task_count": len(tasks),
        },
    )
    return (task_finding, executor_finding)


def _validate_empirical_invariance_binding(
    audit_input: MeasurementAuditInput,
    prediction_artifact: MeasurementPredictionArtifact,
    protocol: MeasurementValidityProtocol,
) -> None:
    if not isinstance(audit_input, MeasurementAuditInput):
        raise TypeError(
            "Feher/Hare empirical invariance requires MeasurementAuditInput"
        )
    if not isinstance(prediction_artifact, MeasurementPredictionArtifact):
        raise TypeError(
            "Feher/Hare empirical invariance requires MeasurementPredictionArtifact"
        )
    if not isinstance(protocol, MeasurementValidityProtocol):
        raise TypeError(
            "Feher/Hare empirical invariance requires MeasurementValidityProtocol"
        )
    if prediction_artifact.protocol_hash != protocol.content_hash:
        raise ValueError("Feher/Hare empirical invariance protocol binding changed")
    if prediction_artifact.audit_input_hash != audit_input.content_hash:
        raise ValueError("Feher/Hare empirical invariance audit input binding changed")
    if protocol.empirical_anchor_hash != audit_input.empirical_anchor_hash:
        raise ValueError("Feher/Hare empirical invariance protocol anchor changed")
    if protocol.candidate_hashes != tuple(
        candidate.content_hash for candidate in audit_input.frozen_candidates
    ):
        raise ValueError("Feher/Hare empirical invariance protocol candidates changed")
    if dict(protocol.seeds_by_role) != _SEEDS_BY_ROLE:
        raise ValueError("Feher/Hare empirical invariance protocol seed plan changed")


def _measurement_profile_payload(
    case_losses: tuple[MeasurementCaseLoss, ...],
    profile_hash: str,
) -> dict[str, object]:
    return {
        "case_losses": tuple(row.identity_payload() for row in case_losses),
        "robustness_profile_hash": profile_hash,
    }


def _coordinate_permuted_case_losses(
    audit_input: MeasurementAuditInput,
    prediction_artifact: MeasurementPredictionArtifact,
    protocol: MeasurementValidityProtocol,
    original_losses: tuple[MeasurementCaseLoss, ...],
) -> tuple[MeasurementCaseLoss, ...]:
    seeds_by_role = dict(protocol.seeds_by_role)
    losses_by_score = {
        MeasurementScore.BRIER: two_stage_brier_loss(),
        MeasurementScore.LOG: two_stage_log_loss(),
    }
    transformed: dict[
        tuple[str, str, MeasurementScore],
        MeasurementCaseLoss,
    ] = {}
    for case in audit_input.cases:
        transformed_target = permute_binary_metric_map(dict(case.target))
        for model_name in _MODELS:
            model = prediction_artifact.model_map[model_name]
            averaged = dict(
                average_seed_metrics(
                    tuple(
                        model.row_map[(case.case_hash, seed)].metric_map
                        for seed in seeds_by_role[case.role]
                    )
                )
            )
            transformed_prediction = permute_binary_metric_map(averaged)
            for score in MeasurementScore:
                row = MeasurementCaseLoss(
                    case_hash=case.case_hash,
                    role=case.role,
                    task_variant=case.task_variant,
                    participant_group_hash=case.participant_group_hash,
                    model_name=model_name,
                    score=score,
                    value=evaluate_metric_loss(
                        losses_by_score[score],
                        transformed_prediction,
                        transformed_target,
                    ),
                )
                transformed[(case.case_hash, model_name, score)] = row
    return tuple(
        transformed[(row.case_hash, row.model_name, row.score)]
        for row in original_losses
    )


def evaluate_feher_hare_empirical_invariance(
    audit_input: MeasurementAuditInput,
    prediction_artifact: MeasurementPredictionArtifact,
    protocol: MeasurementValidityProtocol,
) -> tuple[ExactInvarianceFinding, ...]:
    _validate_empirical_invariance_binding(
        audit_input,
        prediction_artifact,
        protocol,
    )
    scoring_losses = (two_stage_brier_loss(), two_stage_log_loss())
    original_losses = score_measurement_predictions(
        audit_input,
        prediction_artifact,
        scoring_losses,
    )
    original_profile = build_measurement_robustness_profile(
        original_losses,
        protocol,
    )
    transformed_losses = _coordinate_permuted_case_losses(
        audit_input,
        prediction_artifact,
        protocol,
        original_losses,
    )
    transformed_profile = build_measurement_robustness_profile(
        transformed_losses,
        protocol,
    )
    coordinate_finding = _exact_invariance_finding(
        "categorical_coordinate_invariance",
        _measurement_profile_payload(
            original_losses,
            original_profile.content_hash,
        ),
        _measurement_profile_payload(
            transformed_losses,
            transformed_profile.content_hash,
        ),
        details_payload={
            "protocol_hash": protocol.content_hash,
            "audit_input_hash": audit_input.content_hash,
            "prediction_artifact_hash": prediction_artifact.content_hash,
            "transformation": "simultaneous_binary_target_prediction_swap",
        },
    )

    reordered_input = replace(
        audit_input,
        cases=tuple(reversed(audit_input.cases)),
    )
    reordered_models = tuple(
        MeasurementModelPrediction(
            model_name=model.model_name,
            candidate_hash=model.candidate_hash,
            rows=tuple(reversed(model.rows)),
        )
        for model in reversed(prediction_artifact.models)
    )
    reordered_artifact = MeasurementPredictionArtifact(
        protocol_hash=prediction_artifact.protocol_hash,
        audit_input_hash=reordered_input.content_hash,
        models=reordered_models,
        execution_manifest_hashes=tuple(
            reversed(prediction_artifact.execution_manifest_hashes)
        ),
    )
    reordered_losses = score_measurement_predictions(
        reordered_input,
        reordered_artifact,
        scoring_losses,
    )
    reordered_profile = build_measurement_robustness_profile(
        reordered_losses,
        protocol,
    )
    order_finding = _exact_invariance_finding(
        "record_order_batch_invariance",
        {
            "audit_input_hash": audit_input.content_hash,
            "prediction_artifact_hash": prediction_artifact.content_hash,
            **_measurement_profile_payload(
                original_losses,
                original_profile.content_hash,
            ),
        },
        {
            "audit_input_hash": reordered_input.content_hash,
            "prediction_artifact_hash": reordered_artifact.content_hash,
            **_measurement_profile_payload(
                reordered_losses,
                reordered_profile.content_hash,
            ),
        },
        details_payload={
            "protocol_hash": protocol.content_hash,
            "transformation": "reverse_cases_models_rows_and_manifest_index",
        },
    )
    return (coordinate_finding, order_finding)


def _ordered_exact_findings(
    findings: object,
    expected_names: tuple[str, ...],
    *,
    label: str,
) -> tuple[ExactInvarianceFinding, ...]:
    try:
        rows = tuple(findings)
    except TypeError as error:
        raise TypeError(f"{label} findings must be a sequence") from error
    if not rows or any(not isinstance(row, ExactInvarianceFinding) for row in rows):
        raise TypeError(f"{label} findings must contain ExactInvarianceFinding values")
    by_name = {row.check_name: row for row in rows}
    if len(by_name) != len(rows) or set(by_name) != set(expected_names):
        raise ValueError(f"{label} finding set changed")
    return tuple(by_name[name] for name in expected_names)


def combine_feher_hare_exact_invariance(
    semantic_findings: object,
    empirical_findings: object,
) -> ExactInvarianceFinding:
    semantic = _ordered_exact_findings(
        semantic_findings,
        ("task_canonicalization", "semantic_executor_invariance"),
        label="Feher/Hare semantic invariance",
    )
    empirical = _ordered_exact_findings(
        empirical_findings,
        (
            "categorical_coordinate_invariance",
            "record_order_batch_invariance",
        ),
        label="Feher/Hare empirical invariance",
    )
    rows = semantic + empirical
    expected_payload = tuple(
        (row.check_name, MeasurementValidityStatus.EXACT_INVARIANCE_MET.value)
        for row in rows
    )
    actual_payload = tuple(
        (row.check_name, row.status.value)
        for row in rows
    )
    return _exact_invariance_finding(
        "feher_hare_exact_invariance_gate",
        expected_payload,
        actual_payload,
        details_payload={
            "rule": "all_exact_invariance_findings_must_be_met",
            "finding_hashes": tuple(row.content_hash for row in rows),
        },
    )


@dataclass(frozen=True)
class StaySwitchCell:
    task_variant: str
    reward: int
    transition_common: bool
    observed_stay_probability: float | None
    model_expected_stay_probability: float | None
    count: int
    status: MeasurementValidityStatus | str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_variant",
            _task(
                self.task_variant,
                label="measurement diagnostic task variant",
            ),
        )
        object.__setattr__(self, "reward", _reward(self.reward))
        if not isinstance(self.transition_common, bool):
            raise ValueError(
                "measurement diagnostic transition common flag must be bool"
            )
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 0
        ):
            raise ValueError(
                "measurement diagnostic cell count must be a non-negative integer"
            )
        status = _status(self.status, label="measurement diagnostic cell status")
        object.__setattr__(self, "status", status)
        if self.count == 0:
            if (
                self.observed_stay_probability is not None
                or self.model_expected_stay_probability is not None
            ):
                raise ValueError(
                    "empty measurement diagnostic cells cannot contain probabilities"
                )
            if status is not MeasurementValidityStatus.NOT_ESTABLISHED:
                raise ValueError(
                    "empty measurement diagnostic cells must be not established"
                )
            return
        if status is not MeasurementValidityStatus.STABLE_UNDER_FROZEN_AUDIT:
            raise ValueError(
                "populated measurement diagnostic cells must be established"
            )
        object.__setattr__(
            self,
            "observed_stay_probability",
            _probability(
                self.observed_stay_probability,
                label="measurement observed stay probability",
            ),
        )
        object.__setattr__(
            self,
            "model_expected_stay_probability",
            _probability(
                self.model_expected_stay_probability,
                label="measurement model expected stay probability",
            ),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "task_variant": self.task_variant,
            "reward": self.reward,
            "transition_common": self.transition_common,
            "observed_stay_probability": self.observed_stay_probability,
            "model_expected_stay_probability": (
                self.model_expected_stay_probability
            ),
            "count": self.count,
            "status": self.status.value,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _cell_sort_key(cell: StaySwitchCell) -> tuple[int, int]:
    return (cell.reward, int(cell.transition_common))


def _interaction(
    cells: Mapping[tuple[int, bool], StaySwitchCell],
    *,
    attribute: str,
) -> float:
    values = {
        key: getattr(cell, attribute) for key, cell in cells.items()
    }
    if any(value is None for value in values.values()):
        raise ValueError("measurement diagnostic interaction support is incomplete")
    return _finite(
        values[(1, True)]
        - values[(0, True)]
        - values[(1, False)]
        + values[(0, False)],
        label="measurement diagnostic interaction",
    )


@dataclass(frozen=True)
class StaySwitchDiagnostic:
    task_variant: str
    model_name: str
    cells: tuple[StaySwitchCell, ...]
    observed_interaction: float | None
    model_interaction: float | None
    status: MeasurementValidityStatus | str

    def __post_init__(self) -> None:
        task = _task(
            self.task_variant,
            label="measurement diagnostic task variant",
        )
        object.__setattr__(self, "task_variant", task)
        model_name = _text(
            self.model_name,
            label="measurement diagnostic model name",
        )
        if model_name not in _MODELS:
            raise ValueError("measurement diagnostic model is unsupported")
        object.__setattr__(self, "model_name", model_name)
        cells = tuple(self.cells)
        if any(not isinstance(cell, StaySwitchCell) for cell in cells):
            raise TypeError(
                "measurement diagnostic cells must be StaySwitchCell values"
            )
        cells = tuple(sorted(cells, key=_cell_sort_key))
        if tuple((cell.reward, cell.transition_common) for cell in cells) != _CELL_ORDER:
            raise ValueError(
                "measurement diagnostic requires all reward/transition cells"
            )
        if any(cell.task_variant != task for cell in cells):
            raise ValueError("measurement diagnostic cell task changed")
        object.__setattr__(self, "cells", cells)
        status = _status(self.status, label="measurement diagnostic status")
        object.__setattr__(self, "status", status)
        by_cell = {
            (cell.reward, cell.transition_common): cell for cell in cells
        }
        if status is MeasurementValidityStatus.NOT_ESTABLISHED:
            if all(cell.count > 0 for cell in cells):
                raise ValueError(
                    "not-established measurement diagnostic has complete support"
                )
            if self.observed_interaction is not None or self.model_interaction is not None:
                raise ValueError(
                    "not-established measurement diagnostic cannot contain interactions"
                )
            return
        if any(cell.count == 0 for cell in cells):
            raise ValueError(
                "established measurement diagnostic has an empty cell"
            )
        observed = _interaction(
            by_cell,
            attribute="observed_stay_probability",
        )
        model = _interaction(
            by_cell,
            attribute="model_expected_stay_probability",
        )
        supplied_observed = _finite(
            self.observed_interaction,
            label="measurement observed interaction",
        )
        supplied_model = _finite(
            self.model_interaction,
            label="measurement model interaction",
        )
        if supplied_observed != observed or supplied_model != model:
            raise ValueError("measurement diagnostic interaction formula changed")
        object.__setattr__(self, "observed_interaction", supplied_observed)
        object.__setattr__(self, "model_interaction", supplied_model)

    def identity_payload(self) -> dict[str, object]:
        return {
            "task_variant": self.task_variant,
            "model_name": self.model_name,
            "cells": tuple(cell.identity_payload() for cell in self.cells),
            "observed_interaction": self.observed_interaction,
            "model_interaction": self.model_interaction,
            "status": self.status.value,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _observed_action(target: tuple[tuple[str, float], ...]) -> str:
    values = dict(target)
    selected = tuple(
        action
        for action in _ACTIONS
        if values[f"first_stage.{action}"] == 1.0
    )
    if len(selected) != 1 or any(
        values[f"first_stage.{action}"] not in (0.0, 1.0)
        for action in _ACTIONS
    ):
        raise ValueError(
            "measurement diagnostic requires a one-hot current action"
        )
    return selected[0]


def _latest_history(
    case,
    *,
    previous_case,
) -> Mapping[str, object] | None:
    if "history" not in case.scenario.payload:
        return None
    history = case.scenario.payload["history"]
    if history is None or history == ():
        return None
    if not isinstance(history, tuple):
        raise ValueError(
            "measurement diagnostic retained history must be a sequence"
        )
    latest = history[-1]
    if not isinstance(latest, Mapping):
        raise ValueError(
            "measurement diagnostic retained history row must be a mapping"
        )
    previous_trial = latest.get("trial_id")
    if (
        isinstance(previous_trial, bool)
        or not isinstance(previous_trial, int)
        or previous_trial < 0
        or previous_trial >= case.trial_index
    ):
        raise ValueError(
            "measurement diagnostic previous trial identity is invalid"
        )
    if previous_case is not None:
        if previous_trial != previous_case.trial_index:
            raise ValueError(
                "measurement diagnostic history is not the previous retained trial"
            )
        if _observed_action(previous_case.target) != latest.get(
            "first_stage_action"
        ):
            raise ValueError(
                "measurement diagnostic previous action disagrees with retained case"
            )
    return latest


def _validated_previous_fields(
    latest: Mapping[str, object],
) -> tuple[str, int, bool]:
    previous_action = _text(
        latest.get("first_stage_action"),
        label="measurement diagnostic previous action",
    )
    if previous_action not in _ACTIONS:
        raise ValueError("measurement diagnostic previous action is unsupported")
    reward = _reward(latest.get("reward"))
    transition_common = latest.get("transition_common")
    if not isinstance(transition_common, bool):
        raise ValueError(
            "measurement diagnostic transition common flag must be bool"
        )
    return previous_action, reward, transition_common


def _validate_prediction_binding(
    audit_input: MeasurementAuditInput,
    prediction_artifact: MeasurementPredictionArtifact,
) -> None:
    if prediction_artifact.audit_input_hash != audit_input.content_hash:
        raise ValueError("measurement diagnostic audit input changed")
    candidates = {
        candidate.name: candidate for candidate in audit_input.frozen_candidates
    }
    if tuple(prediction_artifact.model_map) != _MODELS:
        raise ValueError("measurement diagnostic candidate set changed")
    expected_coverage = {
        (
            case.case_hash,
            case.scenario.content_hash,
            case.role,
            seed,
        )
        for case in audit_input.cases
        for seed in _SEEDS_BY_ROLE[case.role]
    }
    for model_name in _MODELS:
        model = prediction_artifact.model_map[model_name]
        if model.candidate_hash != candidates[model_name].content_hash:
            raise ValueError(
                f"measurement diagnostic candidate changed for {model_name!r}"
            )
        coverage = {
            (row.case_hash, row.scenario_hash, row.role, row.seed)
            for row in model.rows
        }
        if coverage != expected_coverage or len(model.rows) != len(expected_coverage):
            raise ValueError(
                f"measurement diagnostic prediction coverage changed for {model_name!r}"
            )


def build_feher_hare_stay_switch_diagnostics(
    audit_input: MeasurementAuditInput,
    prediction_artifact: MeasurementPredictionArtifact,
) -> tuple[StaySwitchDiagnostic, ...]:
    if not isinstance(audit_input, MeasurementAuditInput):
        raise TypeError("measurement diagnostic requires MeasurementAuditInput")
    if not isinstance(prediction_artifact, MeasurementPredictionArtifact):
        raise TypeError(
            "measurement diagnostic requires MeasurementPredictionArtifact"
        )
    _validate_prediction_binding(audit_input, prediction_artifact)

    ordered_cases = tuple(
        sorted(
            audit_input.cases,
            key=lambda case: (
                _TASKS.index(case.task_variant),
                case.participant_group_hash,
                case.trial_index,
            ),
        )
    )
    observation_rows: dict[
        tuple[str, int, bool, str],
        list[tuple[float, float]],
    ] = {}
    previous_by_participant: dict[tuple[str, str], object] = {}
    for case in ordered_cases:
        participant_key = (case.task_variant, case.participant_group_hash)
        previous_case = previous_by_participant.get(participant_key)
        latest = _latest_history(case, previous_case=previous_case)
        previous_by_participant[participant_key] = case
        if latest is None:
            continue
        previous_action, reward, transition_common = _validated_previous_fields(
            latest
        )
        current_action = _observed_action(case.target)
        observed_stay = 1.0 if current_action == previous_action else 0.0
        metric_name = f"first_stage.{previous_action}"
        seeds = _SEEDS_BY_ROLE[case.role]
        for model_name in _MODELS:
            prediction_map = prediction_artifact.model_map[model_name].row_map
            try:
                averaged = dict(
                    average_seed_metrics(
                        tuple(
                            prediction_map[(case.case_hash, seed)].metric_map
                            for seed in seeds
                        )
                    )
                )
            except KeyError as error:
                raise ValueError(
                    "measurement diagnostic prediction coverage changed"
                ) from error
            if metric_name not in averaged:
                raise ValueError(
                    "measurement diagnostic stay probability metric is missing"
                )
            expected_stay = _probability(
                averaged[metric_name],
                label="measurement model expected stay probability",
            )
            observation_rows.setdefault(
                (case.task_variant, reward, transition_common, model_name),
                [],
            ).append((observed_stay, expected_stay))

    diagnostics: list[StaySwitchDiagnostic] = []
    for task in _TASKS:
        for model_name in _MODELS:
            cells: list[StaySwitchCell] = []
            for reward, transition_common in _CELL_ORDER:
                observations = observation_rows.get(
                    (task, reward, transition_common, model_name),
                    [],
                )
                if not observations:
                    cells.append(
                        StaySwitchCell(
                            task_variant=task,
                            reward=reward,
                            transition_common=transition_common,
                            observed_stay_probability=None,
                            model_expected_stay_probability=None,
                            count=0,
                            status=MeasurementValidityStatus.NOT_ESTABLISHED,
                        )
                    )
                    continue
                cells.append(
                    StaySwitchCell(
                        task_variant=task,
                        reward=reward,
                        transition_common=transition_common,
                        observed_stay_probability=(
                            math.fsum(row[0] for row in observations)
                            / len(observations)
                        ),
                        model_expected_stay_probability=(
                            math.fsum(row[1] for row in observations)
                            / len(observations)
                        ),
                        count=len(observations),
                        status=(
                            MeasurementValidityStatus.STABLE_UNDER_FROZEN_AUDIT
                        ),
                    )
                )
            cell_map = {
                (cell.reward, cell.transition_common): cell for cell in cells
            }
            established = all(cell.count > 0 for cell in cells)
            diagnostics.append(
                StaySwitchDiagnostic(
                    task_variant=task,
                    model_name=model_name,
                    cells=tuple(cells),
                    observed_interaction=(
                        _interaction(
                            cell_map,
                            attribute="observed_stay_probability",
                        )
                        if established
                        else None
                    ),
                    model_interaction=(
                        _interaction(
                            cell_map,
                            attribute="model_expected_stay_probability",
                        )
                        if established
                        else None
                    ),
                    status=(
                        MeasurementValidityStatus.STABLE_UNDER_FROZEN_AUDIT
                        if established
                        else MeasurementValidityStatus.NOT_ESTABLISHED
                    ),
                )
            )
    return tuple(diagnostics)


_SEMANTIC_EXACT_CHECKS = (
    "task_canonicalization",
    "semantic_executor_invariance",
)
_EMPIRICAL_EXACT_CHECKS = (
    "categorical_coordinate_invariance",
    "record_order_batch_invariance",
)
_COMPLETE_EXACT_CHECKS = (
    *_SEMANTIC_EXACT_CHECKS,
    *_EMPIRICAL_EXACT_CHECKS,
    "feher_hare_exact_invariance_gate",
)


def _report_exact_invariance_findings(
    value: object,
    *,
    prediction_performed: bool,
) -> tuple[ExactInvarianceFinding, ...]:
    try:
        rows = tuple(value)
    except TypeError as error:
        raise TypeError(
            "Feher/Hare measurement report invariance findings must be a sequence"
        ) from error
    if any(not isinstance(row, ExactInvarianceFinding) for row in rows):
        raise TypeError(
            "Feher/Hare measurement report invariance findings must contain "
            "ExactInvarianceFinding values"
        )
    expected_names = (
        _COMPLETE_EXACT_CHECKS
        if prediction_performed
        else _SEMANTIC_EXACT_CHECKS
    )
    by_name = {row.check_name: row for row in rows}
    if len(by_name) != len(rows) or set(by_name) != set(expected_names):
        raise ValueError(
            "Feher/Hare measurement report invariance finding set changed"
        )
    canonical = tuple(by_name[name] for name in expected_names)
    if not prediction_performed:
        if not any(
            row.status is MeasurementValidityStatus.EXACT_INVARIANCE_FAILED
            for row in canonical
        ):
            raise ValueError(
                "Feher/Hare pre-execution report requires a failed semantic gate"
            )
        return canonical

    gate = by_name["feher_hare_exact_invariance_gate"]
    component_failed = any(
        by_name[name].status
        is MeasurementValidityStatus.EXACT_INVARIANCE_FAILED
        for name in (*_SEMANTIC_EXACT_CHECKS, *_EMPIRICAL_EXACT_CHECKS)
    )
    expected_gate_status = (
        MeasurementValidityStatus.EXACT_INVARIANCE_FAILED
        if component_failed
        else MeasurementValidityStatus.EXACT_INVARIANCE_MET
    )
    if gate.status is not expected_gate_status:
        raise ValueError(
            "Feher/Hare measurement report combined invariance gate changed"
        )
    return canonical


def _measurement_report_counts(
    audit_input: MeasurementAuditInput,
) -> tuple[
    tuple[tuple[str, str, int], ...],
    tuple[tuple[str, str, int], ...],
]:
    record_counts: list[tuple[str, str, int]] = []
    participant_counts: list[tuple[str, str, int]] = []
    for role in (
        ObservationPartitionRole.TRAIN,
        ObservationPartitionRole.SELECTION_VALIDATION,
    ):
        for task in _TASKS:
            rows = tuple(
                case
                for case in audit_input.cases
                if case.role is role and case.task_variant == task
            )
            if not rows:
                raise ValueError(
                    "Feher/Hare measurement report role/task stratum is empty"
                )
            record_counts.append((role.value, task, len(rows)))
            participant_counts.append(
                (
                    role.value,
                    task,
                    len({case.participant_group_hash for case in rows}),
                )
            )
    return tuple(record_counts), tuple(participant_counts)


def _measurement_report_diagnostic_counts(
    diagnostics: tuple[StaySwitchDiagnostic, ...],
) -> tuple[tuple[str, str, int, bool, int], ...]:
    return tuple(
        (
            diagnostic.task_variant,
            diagnostic.model_name,
            cell.reward,
            cell.transition_common,
            cell.count,
        )
        for diagnostic in diagnostics
        for cell in diagnostic.cells
    )


def assemble_feher_hare_measurement_validity_report(
    *,
    audit_input: MeasurementAuditInput,
    protocol: MeasurementValidityProtocol,
    prediction_artifact: MeasurementPredictionArtifact | None,
    exact_invariance_findings: object,
    robustness_profile: MeasurementRobustnessProfile | None,
    stay_switch_diagnostics: object,
) -> MeasurementValidityReport:
    if not isinstance(audit_input, MeasurementAuditInput):
        raise TypeError(
            "Feher/Hare measurement report requires MeasurementAuditInput"
        )
    if not isinstance(protocol, MeasurementValidityProtocol):
        raise TypeError(
            "Feher/Hare measurement report requires MeasurementValidityProtocol"
        )
    if protocol.empirical_anchor_hash != audit_input.empirical_anchor_hash:
        raise ValueError("Feher/Hare measurement report empirical anchor changed")
    if protocol.candidate_hashes != tuple(
        candidate.content_hash for candidate in audit_input.frozen_candidates
    ):
        raise ValueError("Feher/Hare measurement report candidates changed")
    if protocol.allowed_roles != (
        ObservationPartitionRole.TRAIN,
        ObservationPartitionRole.SELECTION_VALIDATION,
    ):
        raise ValueError("Feher/Hare measurement report roles changed")
    if (
        protocol.excluded_final_partition_hash
        != audit_input.excluded_final_partition_hash
        or protocol.excluded_final_target_hash
        != audit_input.excluded_final_target_hash
    ):
        raise ValueError("Feher/Hare measurement report excluded FINAL identity changed")

    prediction_performed = prediction_artifact is not None
    if prediction_performed:
        if not isinstance(prediction_artifact, MeasurementPredictionArtifact):
            raise TypeError(
                "Feher/Hare measurement report prediction must be "
                "MeasurementPredictionArtifact or None"
            )
        _validate_empirical_invariance_binding(
            audit_input,
            prediction_artifact,
            protocol,
        )
    findings = _report_exact_invariance_findings(
        exact_invariance_findings,
        prediction_performed=prediction_performed,
    )
    failed = any(
        row.status is MeasurementValidityStatus.EXACT_INVARIANCE_FAILED
        for row in findings
    )
    terminal_class = (
        MeasurementTerminalClass.SCIENTIFIC_RED
        if failed
        else MeasurementTerminalClass.GREEN
    )

    try:
        diagnostics = tuple(stay_switch_diagnostics)
    except TypeError as error:
        raise TypeError(
            "Feher/Hare measurement report diagnostics must be a sequence"
        ) from error
    if any(not isinstance(row, StaySwitchDiagnostic) for row in diagnostics):
        raise TypeError(
            "Feher/Hare measurement report diagnostics must contain "
            "StaySwitchDiagnostic values"
        )

    if terminal_class is MeasurementTerminalClass.SCIENTIFIC_RED:
        if robustness_profile is not None:
            raise ValueError(
                "Feher/Hare scientific RED report cannot contain robustness"
            )
        if diagnostics:
            raise ValueError(
                "Feher/Hare scientific RED report cannot contain diagnostics"
            )
    else:
        if prediction_artifact is None:
            raise ValueError(
                "Feher/Hare GREEN report requires a prediction artifact"
            )
        if not isinstance(robustness_profile, MeasurementRobustnessProfile):
            raise TypeError(
                "Feher/Hare GREEN report requires MeasurementRobustnessProfile"
            )
        expected_profile = build_measurement_robustness_profile(
            score_measurement_predictions(
                audit_input,
                prediction_artifact,
                (two_stage_brier_loss(), two_stage_log_loss()),
            ),
            protocol,
        )
        if robustness_profile.content_hash != expected_profile.content_hash:
            raise ValueError(
                "Feher/Hare measurement report robustness identity changed"
            )
        expected_diagnostics = build_feher_hare_stay_switch_diagnostics(
            audit_input,
            prediction_artifact,
        )
        if tuple(row.content_hash for row in diagnostics) != tuple(
            row.content_hash for row in expected_diagnostics
        ):
            raise ValueError(
                "Feher/Hare measurement report diagnostic identity changed"
            )

    record_counts, participant_counts = _measurement_report_counts(audit_input)
    diagnostic_counts = _measurement_report_diagnostic_counts(diagnostics)
    return MeasurementValidityReport(
        terminal_class=terminal_class,
        claim_scope=MEASUREMENT_CLAIM_SCOPE,
        protocol_hash=protocol.content_hash,
        audit_input_hash=audit_input.content_hash,
        empirical_anchor_hash=audit_input.empirical_anchor_hash,
        allowed_partition_hashes=tuple(
            (role.value, content_hash)
            for role, content_hash in audit_input.allowed_partition_hashes
        ),
        allowed_target_report_hashes=tuple(
            (role.value, content_hash)
            for role, content_hash in audit_input.allowed_target_report_hashes
        ),
        excluded_final_partition_hash=audit_input.excluded_final_partition_hash,
        excluded_final_target_hash=audit_input.excluded_final_target_hash,
        candidate_hashes=tuple(
            candidate.content_hash for candidate in audit_input.frozen_candidates
        ),
        prediction_artifact_hash=(
            None
            if prediction_artifact is None
            else prediction_artifact.content_hash
        ),
        execution_manifest_hashes=(
            ()
            if prediction_artifact is None
            else prediction_artifact.execution_manifest_hashes
        ),
        exact_invariance_findings=findings,
        robustness_profile=robustness_profile,
        stay_switch_diagnostics=diagnostics,
        record_counts=record_counts,
        participant_counts=participant_counts,
        diagnostic_cell_counts=diagnostic_counts,
        parameter_training_performed=False,
        parameter_selection_performed=False,
        final_test_values_exposed_to_audit=False,
        final_test_outcomes_analyzed=False,
        final_model_execution=False,
    )


def _measurement_anchor_empirical_hash(
    anchor: FeherHareMeasurementAnchor,
) -> str:
    return stable_content_hash(
        {
            "claim_scope": MEASUREMENT_CLAIM_SCOPE,
            "source_manifest_hash": anchor.source_manifest_hash,
            "source_snapshot_hash": anchor.source_snapshot_hash,
            "transform_hash": anchor.transform_hash,
            "participant_assignment_hash": anchor.participant_assignment_hash,
            "dataset_hash": anchor.dataset_hash,
            "target_spec_hash": anchor.target_spec_hash,
            "allowed_partition_hashes": (
                (
                    ObservationPartitionRole.TRAIN.value,
                    anchor.train_partition_hash,
                ),
                (
                    ObservationPartitionRole.SELECTION_VALIDATION.value,
                    anchor.selection_partition_hash,
                ),
            ),
            "frozen_candidate_hashes": tuple(
                row.candidate_hash for row in anchor.candidate_rows
            ),
            "excluded_final_partition_hash": (
                anchor.excluded_final_partition_hash
            ),
            "excluded_final_target_hash": anchor.excluded_final_target_hash,
        }
    )


def _measurement_implementation_identities(
) -> tuple[tuple[str, Mapping[str, object]], ...]:
    return (
        (
            "anchor_loader",
            callable_identity(load_feher_hare_r3_measurement_anchor),
        ),
        (
            "provisioner",
            callable_identity(provision_feher_hare_measurement_input),
        ),
        (
            "candidate_freezer",
            callable_identity(freeze_feher_hare_measurement_candidates),
        ),
        (
            "semantic_evaluator",
            callable_identity(evaluate_feher_hare_semantic_invariance),
        ),
        (
            "prediction_executor",
            callable_identity(execute_feher_hare_measurement_predictions),
        ),
        (
            "empirical_invariance",
            callable_identity(evaluate_feher_hare_empirical_invariance),
        ),
        (
            "invariance_combiner",
            callable_identity(combine_feher_hare_exact_invariance),
        ),
        (
            "prediction_scorer",
            callable_identity(score_measurement_predictions),
        ),
        (
            "robustness_builder",
            callable_identity(build_measurement_robustness_profile),
        ),
        (
            "diagnostic_builder",
            callable_identity(build_feher_hare_stay_switch_diagnostics),
        ),
        (
            "report_assembler",
            callable_identity(assemble_feher_hare_measurement_validity_report),
        ),
        ("report_attestor", callable_identity(attest_report)),
        ("brier_loss", metric_loss_identity(two_stage_brier_loss())),
        ("log_loss", metric_loss_identity(two_stage_log_loss())),
    )


def build_feher_hare_measurement_protocol(
    anchor: FeherHareMeasurementAnchor,
) -> MeasurementValidityProtocol:
    if not isinstance(anchor, FeherHareMeasurementAnchor):
        raise TypeError(
            "Feher/Hare measurement protocol requires FeherHareMeasurementAnchor"
        )
    fixtures = frozen_feher_hare_semantic_fixtures()
    return MeasurementValidityProtocol(
        name="feher-hare-measurement-validity-v1",
        version="1",
        claim_scope=MEASUREMENT_CLAIM_SCOPE,
        empirical_anchor_hash=_measurement_anchor_empirical_hash(anchor),
        allowed_roles=(
            ObservationPartitionRole.TRAIN,
            ObservationPartitionRole.SELECTION_VALIDATION,
        ),
        candidate_hashes=tuple(
            row.candidate_hash for row in anchor.candidate_rows
        ),
        seeds_by_role=(
            (ObservationPartitionRole.TRAIN, (101, 102)),
            (
                ObservationPartitionRole.SELECTION_VALIDATION,
                (201, 202),
            ),
        ),
        semantic_fixture_hashes=tuple(
            fixture.content_hash for fixture in fixtures
        ),
        semantic_permutations=(
            "binary_coordinate_swap",
            "coherent_action_state_relabel",
            "second_stage_action_relabel",
            "magic_carpet_counterbalance",
            "spaceship_symbol_order",
            "post_choice_outcome_exclusion",
            "record_order_batch_permutation",
            "serial_spawn_executor_identity",
        ),
        task_strata=_TASKS,
        aggregation_rules=(
            "trial_equal",
            "participant_equal",
        ),
        scores=("brier", "log"),
        material_reversal_references=(
            (MeasurementScore.BRIER, 0.005),
            (MeasurementScore.LOG, 0.006931471805599453),
        ),
        participant_influence_rule=(
            "deterministic_leave_one_participant_out"
        ),
        stay_switch_definition=(
            "reward_by_transition_previous_retained_trial"
        ),
        excluded_final_partition_hash=anchor.excluded_final_partition_hash,
        excluded_final_target_hash=anchor.excluded_final_target_hash,
        implementation_identities=_measurement_implementation_identities(),
    )


def _require_measurement_implementation(
    protocol: MeasurementValidityProtocol,
    name: str,
    actual_identity: Mapping[str, object],
) -> None:
    expected = dict(protocol.implementation_identities).get(name)
    if expected is None:
        raise ValueError(
            f"Feher/Hare measurement protocol lacks {name!r} implementation identity"
        )
    if dict(expected) != dict(actual_identity):
        raise ValueError(
            f"Feher/Hare measurement {name!r} implementation identity changed"
        )


@dataclass(frozen=True)
class FeherHareMeasurementValidityResult:
    audit_input_hash: str
    protocol: MeasurementValidityProtocol
    prediction_hash: str | None
    report: MeasurementValidityReport
    artifact: AggregateReportArtifact
    attestation: AttestedReport[MeasurementValidityReport]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_input_hash",
            _content_hash(
                self.audit_input_hash,
                label="Feher/Hare measurement result audit input hash",
            ),
        )
        if not isinstance(self.protocol, MeasurementValidityProtocol):
            raise TypeError(
                "Feher/Hare measurement result protocol has the wrong type"
            )
        if self.prediction_hash is not None:
            object.__setattr__(
                self,
                "prediction_hash",
                _content_hash(
                    self.prediction_hash,
                    label="Feher/Hare measurement result prediction hash",
                ),
            )
        if not isinstance(self.report, MeasurementValidityReport):
            raise TypeError(
                "Feher/Hare measurement result report has the wrong type"
            )
        if not isinstance(self.artifact, AggregateReportArtifact):
            raise TypeError(
                "Feher/Hare measurement result artifact has the wrong type"
            )
        if not isinstance(self.attestation, AttestedReport):
            raise TypeError(
                "Feher/Hare measurement result attestation has the wrong type"
            )
        if self.report.audit_input_hash != self.audit_input_hash:
            raise ValueError(
                "Feher/Hare measurement result audit input identity changed"
            )
        if self.report.protocol_hash != self.protocol.content_hash:
            raise ValueError(
                "Feher/Hare measurement result protocol identity changed"
            )
        if self.report.prediction_artifact_hash != self.prediction_hash:
            raise ValueError(
                "Feher/Hare measurement result prediction identity changed"
            )
        if not self.artifact.matches(self.report):
            raise ValueError(
                "Feher/Hare measurement result artifact does not match report"
            )
        if self.attestation.report is not self.report:
            raise ValueError(
                "Feher/Hare measurement result attestation report changed"
            )
        if self.attestation.artifact != self.artifact:
            raise ValueError(
                "Feher/Hare measurement result attestation artifact changed"
            )

    def identity_payload(self) -> dict[str, object]:
        return {
            "audit_input_hash": self.audit_input_hash,
            "protocol_hash": self.protocol.content_hash,
            "prediction_hash": self.prediction_hash,
            "report_hash": self.report.content_hash,
            "artifact_hash": self.artifact.content_hash,
            "attestation_hash": self.attestation.content_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _finalize_feher_hare_measurement_result(
    *,
    audit_input: MeasurementAuditInput,
    protocol: MeasurementValidityProtocol,
    prediction_artifact: MeasurementPredictionArtifact | None,
    report: MeasurementValidityReport,
) -> FeherHareMeasurementValidityResult:
    _require_measurement_implementation(
        protocol,
        "report_attestor",
        callable_identity(attest_report),
    )
    attestation = attest_report(report)
    artifact = attestation.artifact
    return FeherHareMeasurementValidityResult(
        audit_input_hash=audit_input.content_hash,
        protocol=protocol,
        prediction_hash=(
            None
            if prediction_artifact is None
            else prediction_artifact.content_hash
        ),
        report=report,
        artifact=artifact,
        attestation=attestation,
    )


def run_feher_hare_measurement_validity_v1(
    *,
    root: Path,
    manifest: TwoStageSourceManifest,
    lock_path: Path,
    lock_commit: str,
    repository_identity: RepositoryIdentity,
    executor: CandidateExecutor,
) -> FeherHareMeasurementValidityResult:
    anchor = load_feher_hare_r3_measurement_anchor(
        Path(lock_path),
        lock_commit=lock_commit,
    )
    audit_input = provision_feher_hare_measurement_input(
        Path(root),
        manifest,
        anchor,
    )
    protocol = build_feher_hare_measurement_protocol(anchor)
    for name, function in (
        ("anchor_loader", load_feher_hare_r3_measurement_anchor),
        ("provisioner", provision_feher_hare_measurement_input),
        ("candidate_freezer", freeze_feher_hare_measurement_candidates),
        ("semantic_evaluator", evaluate_feher_hare_semantic_invariance),
        (
            "prediction_executor",
            execute_feher_hare_measurement_predictions,
        ),
        ("empirical_invariance", evaluate_feher_hare_empirical_invariance),
        ("invariance_combiner", combine_feher_hare_exact_invariance),
        ("prediction_scorer", score_measurement_predictions),
        ("robustness_builder", build_measurement_robustness_profile),
        (
            "diagnostic_builder",
            build_feher_hare_stay_switch_diagnostics,
        ),
        (
            "report_assembler",
            assemble_feher_hare_measurement_validity_report,
        ),
    ):
        _require_measurement_implementation(
            protocol,
            name,
            callable_identity(function),
        )
    if protocol.empirical_anchor_hash != audit_input.empirical_anchor_hash:
        raise ValueError(
            "Feher/Hare measurement protocol empirical anchor changed"
        )
    protocol.build_manifest(audit_input)

    candidates = freeze_feher_hare_measurement_candidates(anchor)
    if tuple(candidate.content_hash for candidate in candidates) != tuple(
        candidate.content_hash for candidate in audit_input.frozen_candidates
    ):
        raise ValueError(
            "Feher/Hare measurement provisioned candidate identities changed"
        )
    semantic_findings = evaluate_feher_hare_semantic_invariance(
        repository_identity,
        candidates,
    )
    if any(
        row.status is MeasurementValidityStatus.EXACT_INVARIANCE_FAILED
        for row in semantic_findings
    ):
        report = assemble_feher_hare_measurement_validity_report(
            audit_input=audit_input,
            protocol=protocol,
            prediction_artifact=None,
            exact_invariance_findings=semantic_findings,
            robustness_profile=None,
            stay_switch_diagnostics=(),
        )
        return _finalize_feher_hare_measurement_result(
            audit_input=audit_input,
            protocol=protocol,
            prediction_artifact=None,
            report=report,
        )

    prediction_artifact = execute_feher_hare_measurement_predictions(
        repository_identity,
        audit_input,
        protocol,
        executor,
    )
    empirical_findings = evaluate_feher_hare_empirical_invariance(
        audit_input,
        prediction_artifact,
        protocol,
    )
    complete_gate = combine_feher_hare_exact_invariance(
        semantic_findings,
        empirical_findings,
    )
    complete_findings = semantic_findings + empirical_findings + (complete_gate,)
    if complete_gate.status is MeasurementValidityStatus.EXACT_INVARIANCE_FAILED:
        report = assemble_feher_hare_measurement_validity_report(
            audit_input=audit_input,
            protocol=protocol,
            prediction_artifact=prediction_artifact,
            exact_invariance_findings=complete_findings,
            robustness_profile=None,
            stay_switch_diagnostics=(),
        )
        return _finalize_feher_hare_measurement_result(
            audit_input=audit_input,
            protocol=protocol,
            prediction_artifact=prediction_artifact,
            report=report,
        )

    brier_loss = two_stage_brier_loss()
    log_loss = two_stage_log_loss()
    _require_measurement_implementation(
        protocol,
        "brier_loss",
        metric_loss_identity(brier_loss),
    )
    _require_measurement_implementation(
        protocol,
        "log_loss",
        metric_loss_identity(log_loss),
    )
    case_losses = score_measurement_predictions(
        audit_input,
        prediction_artifact,
        (brier_loss, log_loss),
    )
    robustness_profile = build_measurement_robustness_profile(
        case_losses,
        protocol,
    )
    diagnostics = build_feher_hare_stay_switch_diagnostics(
        audit_input,
        prediction_artifact,
    )
    report = assemble_feher_hare_measurement_validity_report(
        audit_input=audit_input,
        protocol=protocol,
        prediction_artifact=prediction_artifact,
        exact_invariance_findings=complete_findings,
        robustness_profile=robustness_profile,
        stay_switch_diagnostics=diagnostics,
    )
    return _finalize_feher_hare_measurement_result(
        audit_input=audit_input,
        protocol=protocol,
        prediction_artifact=prediction_artifact,
        report=report,
    )


__all__ = [
    "FEHER_HARE_R3_LOCK_COMMIT",
    "FeherHareMeasurementAnchor",
    "FeherHareMeasurementCandidateRow",
    "FeherHareMeasurementPredictionTask",
    "FeherHareMeasurementValidityResult",
    "FeherHareSemanticFixture",
    "StaySwitchCell",
    "StaySwitchDiagnostic",
    "_project_prepared_measurement_input",
    "assemble_feher_hare_measurement_validity_report",
    "build_feher_hare_stay_switch_diagnostics",
    "build_feher_hare_measurement_protocol",
    "combine_feher_hare_exact_invariance",
    "evaluate_feher_hare_empirical_invariance",
    "evaluate_feher_hare_semantic_invariance",
    "execute_feher_hare_measurement_predictions",
    "freeze_feher_hare_measurement_candidates",
    "frozen_feher_hare_semantic_fixtures",
    "load_feher_hare_r3_measurement_anchor",
    "provision_feher_hare_measurement_input",
    "run_feher_hare_measurement_validity_v1",
]

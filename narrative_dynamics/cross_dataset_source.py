"""Source-neutral identity and semantic contracts for transfer Phase A."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Protocol
from urllib.parse import urlsplit

from narrative_dynamics.contracts import stable_content_hash


_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_SOURCE_PURPOSES = frozenset(
    {
        "behavioral_rows",
        "data_dictionary",
        "license_evidence",
        "source_analysis_code",
        "source_metadata",
        "source_readme",
    }
)
_STAGE_SEQUENCE = (
    "parse",
    "transform",
    "scenario",
    "predict",
    "score",
    "report",
)


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


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty without surrounding whitespace")
    return value


def _hash(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if _HASH_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{label} must be canonical sha256:<64 lowercase hex>")
    return text


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _canonical_date(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc
    if parsed.isoformat() != text:
        raise ValueError(f"{label} must be a canonical ISO date")
    return text


def _https_url(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    parsed = urlsplit(text)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(f"{label} must be a canonical HTTPS URL")
    return text


def _relative_path(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if "\\" in text:
        raise ValueError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative path")
    return text


def _pair_rows(value: object, *, label: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a sequence")
    rows: list[tuple[str, str]] = []
    for index, row in enumerate(value):
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError(f"{label}[{index}] must contain two fields")
        rows.append(
            (
                _text(row[0], label=f"{label}[{index}].source"),
                _text(row[1], label=f"{label}[{index}].target"),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class DatasetSourceFile:
    path: str
    locator: str
    byte_size: int
    sha256: str
    purpose: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path, label="path"))
        object.__setattr__(
            self,
            "locator",
            _https_url(self.locator, label="locator"),
        )
        object.__setattr__(
            self,
            "byte_size",
            _integer(self.byte_size, label="byte_size", minimum=1),
        )
        object.__setattr__(self, "sha256", _hash(self.sha256, label="sha256"))
        purpose = _text(self.purpose, label="purpose")
        if purpose not in _SOURCE_PURPOSES:
            raise ValueError(f"unknown source-file purpose: {purpose!r}")
        object.__setattr__(self, "purpose", purpose)

    def identity_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "locator": self.locator,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "purpose": self.purpose,
        }

    def to_payload(self) -> dict[str, object]:
        return self.identity_payload()

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    @classmethod
    def from_payload(cls, payload: object) -> DatasetSourceFile:
        expected = ("path", "locator", "byte_size", "sha256", "purpose")
        values = _strict_fields(payload, expected=expected, label="source file")
        return cls(
            path=values["path"],
            locator=values["locator"],
            byte_size=values["byte_size"],
            sha256=values["sha256"],
            purpose=values["purpose"],
        )


@dataclass(frozen=True)
class DatasetSourceManifest:
    selected_catalog_entry_hash: str
    name: str
    version: str
    study_reference: str
    public_locator: str
    release_date: str
    license_name: str
    license_reference: str
    files: tuple[DatasetSourceFile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_catalog_entry_hash",
            _hash(
                self.selected_catalog_entry_hash,
                label="selected_catalog_entry_hash",
            ),
        )
        for field_name in ("name", "version", "study_reference", "license_name"):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), label=field_name),
            )
        object.__setattr__(
            self,
            "public_locator",
            _https_url(self.public_locator, label="public_locator"),
        )
        object.__setattr__(
            self,
            "release_date",
            _canonical_date(self.release_date, label="release_date"),
        )
        object.__setattr__(
            self,
            "license_reference",
            _https_url(self.license_reference, label="license_reference"),
        )
        if not isinstance(self.files, (tuple, list)):
            raise TypeError("files must be a sequence")
        files = tuple(self.files)
        if not files:
            raise ValueError("files must not be empty")
        if not all(isinstance(row, DatasetSourceFile) for row in files):
            raise TypeError("files must contain DatasetSourceFile values")
        paths = tuple(row.path for row in files)
        if len(paths) != len(set(paths)):
            raise ValueError("source manifest paths must be unique")
        object.__setattr__(self, "files", tuple(sorted(files, key=lambda row: row.path)))

    def identity_payload(self) -> dict[str, object]:
        return {
            "selected_catalog_entry_hash": self.selected_catalog_entry_hash,
            "name": self.name,
            "version": self.version,
            "study_reference": self.study_reference,
            "public_locator": self.public_locator,
            "release_date": self.release_date,
            "license_name": self.license_name,
            "license_reference": self.license_reference,
            "files": [row.to_payload() for row in self.files],
        }

    def to_payload(self) -> dict[str, object]:
        return self.identity_payload()

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    @classmethod
    def from_payload(cls, payload: object) -> DatasetSourceManifest:
        expected = (
            "selected_catalog_entry_hash",
            "name",
            "version",
            "study_reference",
            "public_locator",
            "release_date",
            "license_name",
            "license_reference",
            "files",
        )
        values = _strict_fields(payload, expected=expected, label="source manifest")
        return cls(
            selected_catalog_entry_hash=values["selected_catalog_entry_hash"],
            name=values["name"],
            version=values["version"],
            study_reference=values["study_reference"],
            public_locator=values["public_locator"],
            release_date=values["release_date"],
            license_name=values["license_name"],
            license_reference=values["license_reference"],
            files=tuple(DatasetSourceFile.from_payload(row) for row in values["files"]),
        )


@dataclass(frozen=True)
class VerifiedSourceSnapshot:
    selected_catalog_entry_hash: str
    source_manifest_hash: str
    source_snapshot_hash: str
    files: tuple[tuple[str, int, str], ...]

    def __post_init__(self) -> None:
        for field_name in (
            "selected_catalog_entry_hash",
            "source_manifest_hash",
            "source_snapshot_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=field_name),
            )
        if not isinstance(self.files, (tuple, list)):
            raise TypeError("verified files must be a sequence")
        files: list[tuple[str, int, str]] = []
        for index, row in enumerate(self.files):
            if not isinstance(row, (tuple, list)) or len(row) != 3:
                raise ValueError(f"files[{index}] must contain three fields")
            files.append(
                (
                    _relative_path(row[0], label=f"files[{index}].path"),
                    _integer(row[1], label=f"files[{index}].byte_size", minimum=1),
                    _hash(row[2], label=f"files[{index}].sha256"),
                )
            )
        if not files:
            raise ValueError("verified files must not be empty")
        paths = tuple(row[0] for row in files)
        if len(paths) != len(set(paths)):
            raise ValueError("verified file paths must be unique")
        if tuple(files) != tuple(sorted(files)):
            raise ValueError("verified files must be in canonical path order")
        object.__setattr__(self, "files", tuple(files))

    @classmethod
    def create(
        cls,
        *,
        manifest: DatasetSourceManifest,
        files: tuple[tuple[str, int, str], ...],
    ) -> VerifiedSourceSnapshot:
        canonical_files = tuple(sorted(files))
        snapshot_hash = stable_content_hash(
            {
                "source_manifest_hash": manifest.content_hash,
                "files": [list(row) for row in canonical_files],
            }
        )
        return cls(
            selected_catalog_entry_hash=manifest.selected_catalog_entry_hash,
            source_manifest_hash=manifest.content_hash,
            source_snapshot_hash=snapshot_hash,
            files=canonical_files,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "selected_catalog_entry_hash": self.selected_catalog_entry_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "source_snapshot_hash": self.source_snapshot_hash,
            "files": [list(row) for row in self.files],
        }

    def to_payload(self) -> dict[str, object]:
        return self.identity_payload()

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    @classmethod
    def from_payload(cls, payload: object) -> VerifiedSourceSnapshot:
        expected = (
            "selected_catalog_entry_hash",
            "source_manifest_hash",
            "source_snapshot_hash",
            "files",
        )
        values = _strict_fields(payload, expected=expected, label="verified snapshot")
        return cls(
            selected_catalog_entry_hash=values["selected_catalog_entry_hash"],
            source_manifest_hash=values["source_manifest_hash"],
            source_snapshot_hash=values["source_snapshot_hash"],
            files=tuple(tuple(row) for row in values["files"]),
        )


def verify_source_snapshot(
    root: Path,
    manifest: DatasetSourceManifest,
) -> VerifiedSourceSnapshot:
    if not isinstance(root, Path):
        raise TypeError("root must be pathlib.Path")
    if not isinstance(manifest, DatasetSourceManifest):
        raise TypeError("manifest must be DatasetSourceManifest")
    try:
        resolved_root = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("missing source root") from exc
    if not resolved_root.is_dir():
        raise ValueError("source root must be a directory")
    verified: list[tuple[str, int, str]] = []
    for source_file in manifest.files:
        unresolved = resolved_root / source_file.path
        try:
            candidate = unresolved.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"missing source file: {source_file.path}") from exc
        if resolved_root not in candidate.parents or not candidate.is_file():
            raise ValueError(f"source file escapes verified root: {source_file.path}")
        payload = candidate.read_bytes()
        actual_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
        if len(payload) != source_file.byte_size or actual_hash != source_file.sha256:
            raise ValueError(f"source file identity mismatch: {source_file.path}")
        verified.append((source_file.path, len(payload), actual_hash))
    return VerifiedSourceSnapshot.create(manifest=manifest, files=tuple(verified))


@dataclass(frozen=True, repr=False)
class CanonicalTransferTrial:
    participant_key: str = field(repr=False)
    trial_id: int
    source_stratum: str
    first_stage_action: str
    transition_common: bool
    final_state: str
    second_stage_action: str
    reward: int
    row_commitment: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "participant_key",
            _text(self.participant_key, label="participant_key"),
        )
        object.__setattr__(
            self,
            "trial_id",
            _integer(self.trial_id, label="trial_id", minimum=1),
        )
        object.__setattr__(
            self,
            "source_stratum",
            _text(self.source_stratum, label="source_stratum"),
        )
        if self.first_stage_action not in {"action_0", "action_1"}:
            raise ValueError("first_stage_action must be action_0 or action_1")
        if not isinstance(self.transition_common, bool):
            raise TypeError("transition_common must be boolean")
        if self.final_state not in {"state_0", "state_1"}:
            raise ValueError("final_state must be state_0 or state_1")
        if self.second_stage_action not in {"second_0", "second_1"}:
            raise ValueError("second_stage_action must be second_0 or second_1")
        if isinstance(self.reward, bool) or self.reward not in {0, 1}:
            raise ValueError("reward must be integer 0 or 1")
        object.__setattr__(
            self,
            "row_commitment",
            _hash(self.row_commitment, label="row_commitment"),
        )

    def __repr__(self) -> str:
        return (
            "CanonicalTransferTrial(participant_key=<redacted>, "
            f"trial_id={self.trial_id}, source_stratum={self.source_stratum!r})"
        )


def validate_canonical_trials(
    trials: tuple[CanonicalTransferTrial, ...],
) -> tuple[CanonicalTransferTrial, ...]:
    if not isinstance(trials, (tuple, list)):
        raise TypeError("trials must be a sequence")
    canonical = tuple(trials)
    if not canonical:
        raise ValueError("trials must not be empty")
    if not all(isinstance(row, CanonicalTransferTrial) for row in canonical):
        raise TypeError("trials must contain CanonicalTransferTrial values")
    seen: set[tuple[str, int]] = set()
    last_by_participant: dict[str, int] = {}
    for trial in canonical:
        key = (trial.participant_key, trial.trial_id)
        if key in seen:
            raise ValueError("duplicate trial identity")
        seen.add(key)
        previous = last_by_participant.get(trial.participant_key)
        if previous is not None and trial.trial_id <= previous:
            raise ValueError("participant trials must be chronological")
        last_by_participant[trial.participant_key] = trial.trial_id
    return canonical


@dataclass(frozen=True)
class TransformReceipt:
    source_manifest_hash: str
    source_snapshot_hash: str
    adapter_identity_hash: str
    schema_version: str
    endpoint: str
    participant_count: int
    raw_trial_count: int
    retained_trial_count: int
    excluded_trial_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "source_manifest_hash",
            "source_snapshot_hash",
            "adapter_identity_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=field_name),
            )
        object.__setattr__(
            self,
            "schema_version",
            _text(self.schema_version, label="schema_version"),
        )
        endpoint = _text(self.endpoint, label="endpoint")
        if endpoint != "observed_first_stage_binary_choice":
            raise ValueError("endpoint must be observed_first_stage_binary_choice")
        for field_name, minimum in (
            ("participant_count", 1),
            ("raw_trial_count", 1),
            ("retained_trial_count", 1),
            ("excluded_trial_count", 0),
        ):
            object.__setattr__(
                self,
                field_name,
                _integer(getattr(self, field_name), label=field_name, minimum=minimum),
            )
        if self.raw_trial_count != self.retained_trial_count + self.excluded_trial_count:
            raise ValueError("raw trials must equal retained plus excluded trials")

    def identity_payload(self) -> dict[str, object]:
        return {
            "source_manifest_hash": self.source_manifest_hash,
            "source_snapshot_hash": self.source_snapshot_hash,
            "adapter_identity_hash": self.adapter_identity_hash,
            "schema_version": self.schema_version,
            "endpoint": self.endpoint,
            "participant_count": self.participant_count,
            "raw_trial_count": self.raw_trial_count,
            "retained_trial_count": self.retained_trial_count,
            "excluded_trial_count": self.excluded_trial_count,
        }

    def to_payload(self) -> dict[str, object]:
        return self.identity_payload()

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    @classmethod
    def from_payload(cls, payload: object) -> TransformReceipt:
        expected = (
            "source_manifest_hash",
            "source_snapshot_hash",
            "adapter_identity_hash",
            "schema_version",
            "endpoint",
            "participant_count",
            "raw_trial_count",
            "retained_trial_count",
            "excluded_trial_count",
        )
        values = _strict_fields(payload, expected=expected, label="transform receipt")
        return cls(**values)


@dataclass(frozen=True)
class SemanticRelabeling:
    name: str
    forward_map: tuple[tuple[str, str], ...]
    inverse_map: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="relabeling name"))
        forward = _pair_rows(self.forward_map, label="forward_map")
        inverse = _pair_rows(self.inverse_map, label="inverse_map")
        if self.name == "identity":
            if forward or inverse:
                raise ValueError("identity relabeling must have empty maps")
        else:
            if not forward or not inverse:
                raise ValueError("non-identity relabeling requires forward and inverse maps")
            if len({source for source, _ in forward}) != len(forward):
                raise ValueError("forward_map sources must be unique")
            if len({target for _, target in forward}) != len(forward):
                raise ValueError("forward_map targets must be unique")
            if dict(inverse) != {target: source for source, target in forward}:
                raise ValueError("inverse_map must exactly invert forward_map")
        object.__setattr__(self, "forward_map", forward)
        object.__setattr__(self, "inverse_map", inverse)

    @classmethod
    def identity(cls) -> SemanticRelabeling:
        return cls(name="identity", forward_map=(), inverse_map=())

    def identity_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "forward_map": [list(row) for row in self.forward_map],
            "inverse_map": [list(row) for row in self.inverse_map],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class SemanticStageReceipt:
    stage: str
    receipt_hash: str

    def __post_init__(self) -> None:
        stage = _text(self.stage, label="stage")
        if stage not in _STAGE_SEQUENCE:
            raise ValueError(f"unknown semantic stage: {stage!r}")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(
            self,
            "receipt_hash",
            _hash(self.receipt_hash, label="receipt_hash"),
        )

    def identity_payload(self) -> dict[str, object]:
        return {"stage": self.stage, "receipt_hash": self.receipt_hash}


@dataclass(frozen=True)
class SemanticPipelineResult:
    identity: str
    inverse_mapped_identity: str
    stage_receipts: tuple[SemanticStageReceipt, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _hash(self.identity, label="identity"))
        object.__setattr__(
            self,
            "inverse_mapped_identity",
            _hash(self.inverse_mapped_identity, label="inverse_mapped_identity"),
        )
        if not isinstance(self.stage_receipts, (tuple, list)) or not all(
            isinstance(row, SemanticStageReceipt) for row in self.stage_receipts
        ):
            raise TypeError("stage_receipts must contain SemanticStageReceipt values")
        object.__setattr__(self, "stage_receipts", tuple(self.stage_receipts))

    def identity_payload(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "inverse_mapped_identity": self.inverse_mapped_identity,
            "stage_receipts": [row.identity_payload() for row in self.stage_receipts],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


class SemanticPipeline(Protocol):
    def run(
        self,
        rows: tuple[Mapping[str, object], ...],
        relabeling: SemanticRelabeling,
    ) -> SemanticPipelineResult:
        raise NotImplementedError


@dataclass(frozen=True)
class SemanticInvarianceEvidence:
    reference_identity: str
    relabeling_count: int
    variant_result_hashes: tuple[tuple[str, str], ...]
    failed_relabelings: tuple[str, ...]
    passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_identity",
            _hash(self.reference_identity, label="reference_identity"),
        )
        count = _integer(self.relabeling_count, label="relabeling_count", minimum=1)
        if count != 5:
            raise ValueError("semantic invariance requires exactly five relabelings")
        variants = tuple(
            (
                _text(name, label=f"variant_result_hashes[{index}].name"),
                _hash(value, label=f"variant_result_hashes[{index}].hash"),
            )
            for index, (name, value) in enumerate(self.variant_result_hashes)
        )
        if len(variants) != count or len({name for name, _ in variants}) != count:
            raise ValueError("variant results must identify five unique relabelings")
        object.__setattr__(self, "variant_result_hashes", variants)
        failed = tuple(
            _text(value, label=f"failed_relabelings[{index}]")
            for index, value in enumerate(self.failed_relabelings)
        )
        if len(failed) != len(set(failed)):
            raise ValueError("failed_relabelings must be unique")
        if not set(failed).issubset({name for name, _ in variants}):
            raise ValueError("failed_relabelings must refer to known variants")
        object.__setattr__(self, "failed_relabelings", failed)
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be boolean")
        if self.passed != (not failed):
            raise ValueError("passed must agree with failed_relabelings")

    def identity_payload(self) -> dict[str, object]:
        return {
            "reference_identity": self.reference_identity,
            "relabeling_count": self.relabeling_count,
            "variant_result_hashes": [list(row) for row in self.variant_result_hashes],
            "failed_relabelings": list(self.failed_relabelings),
            "passed": self.passed,
        }

    def to_payload(self) -> dict[str, object]:
        return self.identity_payload()

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    @classmethod
    def from_payload(cls, payload: object) -> SemanticInvarianceEvidence:
        expected = (
            "reference_identity",
            "relabeling_count",
            "variant_result_hashes",
            "failed_relabelings",
            "passed",
        )
        values = _strict_fields(payload, expected=expected, label="semantic evidence")
        return cls(
            reference_identity=values["reference_identity"],
            relabeling_count=values["relabeling_count"],
            variant_result_hashes=tuple(
                tuple(row) for row in values["variant_result_hashes"]
            ),
            failed_relabelings=tuple(values["failed_relabelings"]),
            passed=values["passed"],
        )


def _require_complete_stages(result: SemanticPipelineResult) -> None:
    stages = tuple(receipt.stage for receipt in result.stage_receipts)
    if stages != _STAGE_SEQUENCE:
        raise ValueError(
            f"semantic pipeline stage sequence must be {_STAGE_SEQUENCE!r}; got {stages!r}"
        )


def evaluate_end_to_end_semantic_invariance(
    *,
    pipeline: SemanticPipeline,
    source_rows: tuple[Mapping[str, object], ...],
    relabelings: tuple[SemanticRelabeling, ...],
) -> SemanticInvarianceEvidence:
    if not isinstance(source_rows, (tuple, list)) or not source_rows:
        raise ValueError("source_rows must be a non-empty sequence")
    rows = tuple(source_rows)
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("source_rows must contain mappings")
    if not isinstance(relabelings, (tuple, list)):
        raise TypeError("relabelings must be a sequence")
    variants = tuple(relabelings)
    if len(variants) != 5 or not all(
        isinstance(row, SemanticRelabeling) for row in variants
    ):
        raise ValueError("semantic invariance requires five relabelings")
    names = tuple(row.name for row in variants)
    if "identity" in names or len(names) != len(set(names)):
        raise ValueError("semantic relabelings must be five unique non-identity variants")

    reference = pipeline.run(rows, SemanticRelabeling.identity())
    if not isinstance(reference, SemanticPipelineResult):
        raise TypeError("semantic pipeline must return SemanticPipelineResult")
    _require_complete_stages(reference)
    if reference.inverse_mapped_identity != reference.identity:
        raise ValueError("identity pipeline result must map to itself")

    results: list[tuple[SemanticRelabeling, SemanticPipelineResult]] = []
    for relabeling in variants:
        result = pipeline.run(rows, relabeling)
        if not isinstance(result, SemanticPipelineResult):
            raise TypeError("semantic pipeline must return SemanticPipelineResult")
        _require_complete_stages(result)
        results.append((relabeling, result))
    failed = tuple(
        relabeling.name
        for relabeling, result in results
        if result.inverse_mapped_identity != reference.identity
    )
    return SemanticInvarianceEvidence(
        reference_identity=reference.identity,
        relabeling_count=len(results),
        variant_result_hashes=tuple(
            (relabeling.name, result.content_hash)
            for relabeling, result in results
        ),
        failed_relabelings=failed,
        passed=not failed,
    )


__all__ = [
    "CanonicalTransferTrial",
    "DatasetSourceFile",
    "DatasetSourceManifest",
    "SemanticInvarianceEvidence",
    "SemanticPipeline",
    "SemanticPipelineResult",
    "SemanticRelabeling",
    "SemanticStageReceipt",
    "TransformReceipt",
    "VerifiedSourceSnapshot",
    "evaluate_end_to_end_semantic_invariance",
    "validate_canonical_trials",
    "verify_source_snapshot",
]

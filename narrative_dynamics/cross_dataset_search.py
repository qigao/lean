"""Governed, source-neutral dataset search contracts for transfer Phase A."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
import re
from typing import Mapping
from urllib.parse import urlsplit

from narrative_dynamics.contracts import stable_content_hash


CROSS_DATASET_TRANSFER_CLAIM_SCOPE = (
    "external_observational_cross_dataset_predictive_transfer_only"
)

_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_DEDUPLICATION_KEY = (
    "study_doi",
    "canonical_repository_locator",
    "immutable_release",
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


def _text(value: object, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    if value != value.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    if not allow_empty and not value:
        raise ValueError(f"{label} must not be empty")
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


def _canonical_hash(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if _HASH_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{label} must be canonical sha256:<64 lowercase hex>")
    return text


def _https_url(value: object, *, label: str, allow_empty: bool = False) -> str:
    text = _text(value, label=label, allow_empty=allow_empty)
    if allow_empty and not text:
        return text
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


def _integer(
    value: object,
    *,
    label: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _ordered_unique_text(
    values: object,
    *,
    label: str,
    nonempty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{label} must be a sequence")
    canonical = tuple(
        _text(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    )
    if nonempty and not canonical:
        raise ValueError(f"{label} must not be empty")
    if len(canonical) != len(set(canonical)):
        raise ValueError(f"{label} must not contain duplicates")
    if canonical != tuple(sorted(canonical)):
        raise ValueError(f"{label} must be in lexical order")
    return canonical


@dataclass(frozen=True)
class DatasetSearchProtocol:
    public_release_cutoff: str
    query_strings: tuple[str, ...]
    repository_families: tuple[str, ...]
    query_date: str
    interface_versions: tuple[tuple[str, str], ...]
    pagination_limit: int
    sort_order: str
    deduplication_key: tuple[str, ...]
    eligibility_fields: tuple[str, ...]
    stopping_rule: str
    candidate_model_code_available: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "public_release_cutoff",
            _canonical_date(
                self.public_release_cutoff,
                label="public_release_cutoff",
            ),
        )
        object.__setattr__(
            self,
            "query_strings",
            _ordered_unique_text(self.query_strings, label="query_strings"),
        )
        families = _ordered_unique_text(
            self.repository_families,
            label="repository_families",
        )
        object.__setattr__(self, "repository_families", families)
        object.__setattr__(
            self,
            "query_date",
            _canonical_date(self.query_date, label="query_date"),
        )
        if not isinstance(self.interface_versions, (tuple, list)):
            raise TypeError("interface_versions must be a sequence")
        versions: tuple[tuple[str, str], ...] = tuple(
            (
                _text(row[0], label=f"interface_versions[{index}].family"),
                _text(row[1], label=f"interface_versions[{index}].version"),
            )
            for index, row in enumerate(self.interface_versions)
            if isinstance(row, (tuple, list)) and len(row) == 2
        )
        if len(versions) != len(self.interface_versions):
            raise ValueError("every interface_versions row must contain two fields")
        if tuple(family for family, _ in versions) != families:
            raise ValueError(
                "interface_versions must cover repository_families in exact order"
            )
        object.__setattr__(self, "interface_versions", versions)
        object.__setattr__(
            self,
            "pagination_limit",
            _integer(self.pagination_limit, label="pagination_limit", minimum=1),
        )
        object.__setattr__(
            self,
            "sort_order",
            _text(self.sort_order, label="sort_order"),
        )
        if not isinstance(self.deduplication_key, (tuple, list)):
            raise TypeError("deduplication_key must be a sequence")
        deduplication_key = tuple(
            _text(value, label=f"deduplication_key[{index}]")
            for index, value in enumerate(self.deduplication_key)
        )
        if deduplication_key != _DEDUPLICATION_KEY:
            raise ValueError("deduplication_key must bind the exact V1 fields")
        object.__setattr__(self, "deduplication_key", deduplication_key)
        object.__setattr__(
            self,
            "eligibility_fields",
            _ordered_unique_text(self.eligibility_fields, label="eligibility_fields"),
        )
        object.__setattr__(
            self,
            "stopping_rule",
            _text(self.stopping_rule, label="stopping_rule"),
        )
        available = _boolean(
            self.candidate_model_code_available,
            label="candidate_model_code_available",
        )
        if available:
            raise ValueError("candidate-model code must be unavailable during search")

    @property
    def required_pages(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (family, page)
            for family in self.repository_families
            for page in range(1, self.pagination_limit + 1)
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "public_release_cutoff": self.public_release_cutoff,
            "query_strings": list(self.query_strings),
            "repository_families": list(self.repository_families),
            "query_date": self.query_date,
            "interface_versions": [list(row) for row in self.interface_versions],
            "pagination_limit": self.pagination_limit,
            "sort_order": self.sort_order,
            "deduplication_key": list(self.deduplication_key),
            "eligibility_fields": list(self.eligibility_fields),
            "stopping_rule": self.stopping_rule,
            "candidate_model_code_available": self.candidate_model_code_available,
        }

    def to_payload(self) -> dict[str, object]:
        return self.identity_payload()

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    @classmethod
    def from_payload(cls, payload: object) -> DatasetSearchProtocol:
        expected = (
            "public_release_cutoff",
            "query_strings",
            "repository_families",
            "query_date",
            "interface_versions",
            "pagination_limit",
            "sort_order",
            "deduplication_key",
            "eligibility_fields",
            "stopping_rule",
            "candidate_model_code_available",
        )
        values = _strict_fields(payload, expected=expected, label="search protocol")
        return cls(
            public_release_cutoff=values["public_release_cutoff"],
            query_strings=tuple(values["query_strings"]),
            repository_families=tuple(values["repository_families"]),
            query_date=values["query_date"],
            interface_versions=tuple(
                tuple(row) for row in values["interface_versions"]
            ),
            pagination_limit=values["pagination_limit"],
            sort_order=values["sort_order"],
            deduplication_key=tuple(values["deduplication_key"]),
            eligibility_fields=tuple(values["eligibility_fields"]),
            stopping_rule=values["stopping_rule"],
            candidate_model_code_available=values[
                "candidate_model_code_available"
            ],
        )


@dataclass(frozen=True)
class DatasetCatalogEntry:
    study_doi: str
    canonical_repository_locator: str
    immutable_release: str
    public_release_date: str
    eligible_participant_count: int
    repository_resolution_receipt_hash: str
    license_name: str
    license_reference: str
    consumed_file_count: int
    independent_collection: bool
    complete_consumed_file_inventory: bool
    binary_first_stage_choice: bool
    chronological_participant_sequence: bool
    transition_or_state_available: bool
    reward_available: bool
    source_declared_exclusions_available: bool
    independent_from_anchor: bool
    candidate_model_executions: int
    exclusion_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_doi", _text(self.study_doi, label="study_doi"))
        object.__setattr__(
            self,
            "canonical_repository_locator",
            _https_url(
                self.canonical_repository_locator,
                label="canonical_repository_locator",
            ),
        )
        object.__setattr__(
            self,
            "immutable_release",
            _text(self.immutable_release, label="immutable_release"),
        )
        object.__setattr__(
            self,
            "public_release_date",
            _canonical_date(self.public_release_date, label="public_release_date"),
        )
        object.__setattr__(
            self,
            "eligible_participant_count",
            _integer(
                self.eligible_participant_count,
                label="eligible_participant_count",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "repository_resolution_receipt_hash",
            _canonical_hash(
                self.repository_resolution_receipt_hash,
                label="repository_resolution_receipt_hash",
            ),
        )
        object.__setattr__(
            self,
            "license_name",
            _text(self.license_name, label="license_name", allow_empty=True),
        )
        object.__setattr__(
            self,
            "license_reference",
            _https_url(
                self.license_reference,
                label="license_reference",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "consumed_file_count",
            _integer(
                self.consumed_file_count,
                label="consumed_file_count",
                minimum=0,
            ),
        )
        for field in (
            "independent_collection",
            "complete_consumed_file_inventory",
            "binary_first_stage_choice",
            "chronological_participant_sequence",
            "transition_or_state_available",
            "reward_available",
            "source_declared_exclusions_available",
            "independent_from_anchor",
        ):
            object.__setattr__(
                self,
                field,
                _boolean(getattr(self, field), label=field),
            )
        executions = _integer(
            self.candidate_model_executions,
            label="candidate_model_executions",
            minimum=0,
        )
        if executions != 0:
            raise ValueError("catalog entries must have zero model executions")
        object.__setattr__(self, "candidate_model_executions", executions)
        object.__setattr__(
            self,
            "exclusion_codes",
            _ordered_unique_text(
                self.exclusion_codes,
                label="exclusion_codes",
                nonempty=False,
            ),
        )

    @property
    def identity_key(self) -> tuple[str, str, str]:
        return (
            self.study_doi,
            self.canonical_repository_locator,
            self.immutable_release,
        )

    def is_eligible_for(self, protocol: DatasetSearchProtocol) -> bool:
        return (
            self.public_release_date <= protocol.public_release_cutoff
            and self.eligible_participant_count >= 150
            and bool(self.license_name)
            and bool(self.license_reference)
            and self.consumed_file_count > 0
            and self.independent_collection
            and self.complete_consumed_file_inventory
            and self.binary_first_stage_choice
            and self.chronological_participant_sequence
            and self.transition_or_state_available
            and self.reward_available
            and self.source_declared_exclusions_available
            and self.independent_from_anchor
            and self.candidate_model_executions == 0
            and not self.exclusion_codes
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "study_doi": self.study_doi,
            "canonical_repository_locator": self.canonical_repository_locator,
            "immutable_release": self.immutable_release,
            "public_release_date": self.public_release_date,
            "eligible_participant_count": self.eligible_participant_count,
            "repository_resolution_receipt_hash": (
                self.repository_resolution_receipt_hash
            ),
            "license_name": self.license_name,
            "license_reference": self.license_reference,
            "consumed_file_count": self.consumed_file_count,
            "independent_collection": self.independent_collection,
            "complete_consumed_file_inventory": (
                self.complete_consumed_file_inventory
            ),
            "binary_first_stage_choice": self.binary_first_stage_choice,
            "chronological_participant_sequence": (
                self.chronological_participant_sequence
            ),
            "transition_or_state_available": self.transition_or_state_available,
            "reward_available": self.reward_available,
            "source_declared_exclusions_available": (
                self.source_declared_exclusions_available
            ),
            "independent_from_anchor": self.independent_from_anchor,
            "candidate_model_executions": self.candidate_model_executions,
            "exclusion_codes": list(self.exclusion_codes),
        }

    def to_payload(self) -> dict[str, object]:
        return self.identity_payload()

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    @classmethod
    def from_payload(cls, payload: object) -> DatasetCatalogEntry:
        expected = (
            "study_doi",
            "canonical_repository_locator",
            "immutable_release",
            "public_release_date",
            "eligible_participant_count",
            "repository_resolution_receipt_hash",
            "license_name",
            "license_reference",
            "consumed_file_count",
            "independent_collection",
            "complete_consumed_file_inventory",
            "binary_first_stage_choice",
            "chronological_participant_sequence",
            "transition_or_state_available",
            "reward_available",
            "source_declared_exclusions_available",
            "independent_from_anchor",
            "candidate_model_executions",
            "exclusion_codes",
        )
        values = _strict_fields(payload, expected=expected, label="catalog entry")
        return cls(
            study_doi=values["study_doi"],
            canonical_repository_locator=values["canonical_repository_locator"],
            immutable_release=values["immutable_release"],
            public_release_date=values["public_release_date"],
            eligible_participant_count=values["eligible_participant_count"],
            repository_resolution_receipt_hash=values[
                "repository_resolution_receipt_hash"
            ],
            license_name=values["license_name"],
            license_reference=values["license_reference"],
            consumed_file_count=values["consumed_file_count"],
            independent_collection=values["independent_collection"],
            complete_consumed_file_inventory=values[
                "complete_consumed_file_inventory"
            ],
            binary_first_stage_choice=values["binary_first_stage_choice"],
            chronological_participant_sequence=values[
                "chronological_participant_sequence"
            ],
            transition_or_state_available=values[
                "transition_or_state_available"
            ],
            reward_available=values["reward_available"],
            source_declared_exclusions_available=values[
                "source_declared_exclusions_available"
            ],
            independent_from_anchor=values["independent_from_anchor"],
            candidate_model_executions=values["candidate_model_executions"],
            exclusion_codes=tuple(values["exclusion_codes"]),
        )


@dataclass(frozen=True)
class DatasetCandidateCatalog:
    protocol: DatasetSearchProtocol
    pages_recorded: tuple[tuple[str, int], ...]
    resolved_repository_receipts: tuple[str, ...]
    entries: tuple[DatasetCatalogEntry, ...]
    candidate_model_executions: int

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, DatasetSearchProtocol):
            raise TypeError("protocol must be DatasetSearchProtocol")
        if not isinstance(self.pages_recorded, (tuple, list)):
            raise TypeError("pages_recorded must be a sequence")
        pages: list[tuple[str, int]] = []
        for index, row in enumerate(self.pages_recorded):
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError(f"pages_recorded[{index}] must have two fields")
            page = (
                _text(row[0], label=f"pages_recorded[{index}].family"),
                _integer(
                    row[1],
                    label=f"pages_recorded[{index}].page",
                    minimum=1,
                ),
            )
            pages.append(page)
        if len(pages) != len(set(pages)):
            raise ValueError("pages_recorded must not contain duplicates")
        unknown_pages = set(pages) - set(self.protocol.required_pages)
        if unknown_pages:
            raise ValueError(f"pages_recorded contains unknown pages: {unknown_pages!r}")
        canonical_pages = tuple(
            page for page in self.protocol.required_pages if page in set(pages)
        )
        if tuple(pages) != canonical_pages:
            raise ValueError("pages_recorded must follow protocol order")
        object.__setattr__(self, "pages_recorded", canonical_pages)

        receipts = tuple(
            _canonical_hash(value, label=f"resolved_repository_receipts[{index}]")
            for index, value in enumerate(self.resolved_repository_receipts)
        )
        if len(receipts) != len(set(receipts)):
            raise ValueError("resolved_repository_receipts must not contain duplicates")
        object.__setattr__(self, "resolved_repository_receipts", tuple(sorted(receipts)))

        if not isinstance(self.entries, (tuple, list)):
            raise TypeError("entries must be a sequence")
        entries = tuple(self.entries)
        if not all(isinstance(entry, DatasetCatalogEntry) for entry in entries):
            raise TypeError("entries must contain DatasetCatalogEntry values")
        identities = tuple(entry.identity_key for entry in entries)
        if len(identities) != len(set(identities)):
            raise ValueError("catalog entry identities must be unique")
        object.__setattr__(self, "entries", tuple(sorted(entries, key=lambda row: row.identity_key)))

        executions = _integer(
            self.candidate_model_executions,
            label="candidate_model_executions",
            minimum=0,
        )
        if executions != 0:
            raise ValueError("candidate catalog must have zero model executions")
        object.__setattr__(self, "candidate_model_executions", executions)

    @classmethod
    def create(
        cls,
        *,
        protocol: DatasetSearchProtocol,
        pages_recorded: tuple[tuple[str, int], ...],
        resolved_repository_receipts: tuple[str, ...],
        entries: tuple[DatasetCatalogEntry, ...],
        candidate_model_executions: int,
    ) -> DatasetCandidateCatalog:
        return cls(
            protocol=protocol,
            pages_recorded=pages_recorded,
            resolved_repository_receipts=resolved_repository_receipts,
            entries=entries,
            candidate_model_executions=candidate_model_executions,
        )

    @property
    def incomplete_reason_codes(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.pages_recorded != self.protocol.required_pages:
            reasons.append("CATALOG_INCOMPLETE")
        resolved = frozenset(self.resolved_repository_receipts)
        if any(
            entry.repository_resolution_receipt_hash not in resolved
            for entry in self.entries
        ):
            reasons.append("DOI_REPOSITORY_UNRESOLVED")
        return tuple(reasons)

    @property
    def is_complete(self) -> bool:
        return not self.incomplete_reason_codes

    def identity_payload(self) -> dict[str, object]:
        return {
            "protocol": self.protocol.to_payload(),
            "pages_recorded": [list(page) for page in self.pages_recorded],
            "resolved_repository_receipts": list(
                self.resolved_repository_receipts
            ),
            "entries": [entry.to_payload() for entry in self.entries],
            "candidate_model_executions": self.candidate_model_executions,
        }

    def to_payload(self) -> dict[str, object]:
        return self.identity_payload()

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    @classmethod
    def from_payload(cls, payload: object) -> DatasetCandidateCatalog:
        expected = (
            "protocol",
            "pages_recorded",
            "resolved_repository_receipts",
            "entries",
            "candidate_model_executions",
        )
        values = _strict_fields(payload, expected=expected, label="candidate catalog")
        return cls.create(
            protocol=DatasetSearchProtocol.from_payload(values["protocol"]),
            pages_recorded=tuple(tuple(row) for row in values["pages_recorded"]),
            resolved_repository_receipts=tuple(
                values["resolved_repository_receipts"]
            ),
            entries=tuple(
                DatasetCatalogEntry.from_payload(row) for row in values["entries"]
            ),
            candidate_model_executions=values["candidate_model_executions"],
        )


class DatasetSelectionStatus(str, Enum):
    SELECTED = "SELECTED"
    SCIENTIFIC_RED = "SCIENTIFIC_RED"
    INFRASTRUCTURE_INCOMPLETE = "INFRASTRUCTURE_INCOMPLETE"


@dataclass(frozen=True)
class DatasetSelectionDecision:
    status: DatasetSelectionStatus
    catalog_hash: str
    selected_entry_hash: str | None
    selected_locator: str | None
    reason_codes: tuple[str, ...]
    candidate_model_executions: int

    def __post_init__(self) -> None:
        if not isinstance(self.status, DatasetSelectionStatus):
            raise TypeError("status must be DatasetSelectionStatus")
        object.__setattr__(
            self,
            "catalog_hash",
            _canonical_hash(self.catalog_hash, label="catalog_hash"),
        )
        reasons = _ordered_unique_text(
            self.reason_codes,
            label="reason_codes",
            nonempty=False,
        )
        object.__setattr__(self, "reason_codes", reasons)
        executions = _integer(
            self.candidate_model_executions,
            label="candidate_model_executions",
            minimum=0,
        )
        if executions != 0:
            raise ValueError("selection decisions must have zero model executions")
        if self.status is DatasetSelectionStatus.SELECTED:
            if self.selected_entry_hash is None or self.selected_locator is None:
                raise ValueError("selected decision requires entry hash and locator")
            if reasons:
                raise ValueError("selected decision must not have reason codes")
            object.__setattr__(
                self,
                "selected_entry_hash",
                _canonical_hash(
                    self.selected_entry_hash,
                    label="selected_entry_hash",
                ),
            )
            object.__setattr__(
                self,
                "selected_locator",
                _https_url(self.selected_locator, label="selected_locator"),
            )
        else:
            if self.selected_entry_hash is not None or self.selected_locator is not None:
                raise ValueError("non-selected decision cannot identify an entry")
            if not reasons:
                raise ValueError("non-selected decision requires reason codes")

    @classmethod
    def selected(
        cls,
        *,
        catalog_hash: str,
        selected_entry_hash: str,
        selected_locator: str,
    ) -> DatasetSelectionDecision:
        return cls(
            status=DatasetSelectionStatus.SELECTED,
            catalog_hash=catalog_hash,
            selected_entry_hash=selected_entry_hash,
            selected_locator=selected_locator,
            reason_codes=(),
            candidate_model_executions=0,
        )

    @classmethod
    def scientific_red(
        cls,
        *,
        catalog_hash: str,
        reason_codes: tuple[str, ...],
    ) -> DatasetSelectionDecision:
        return cls(
            status=DatasetSelectionStatus.SCIENTIFIC_RED,
            catalog_hash=catalog_hash,
            selected_entry_hash=None,
            selected_locator=None,
            reason_codes=reason_codes,
            candidate_model_executions=0,
        )

    @classmethod
    def infrastructure_incomplete(
        cls,
        *,
        catalog_hash: str,
        reason_codes: tuple[str, ...],
    ) -> DatasetSelectionDecision:
        return cls(
            status=DatasetSelectionStatus.INFRASTRUCTURE_INCOMPLETE,
            catalog_hash=catalog_hash,
            selected_entry_hash=None,
            selected_locator=None,
            reason_codes=reason_codes,
            candidate_model_executions=0,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "catalog_hash": self.catalog_hash,
            "selected_entry_hash": self.selected_entry_hash,
            "selected_locator": self.selected_locator,
            "reason_codes": list(self.reason_codes),
            "candidate_model_executions": self.candidate_model_executions,
        }

    def to_payload(self) -> dict[str, object]:
        return self.identity_payload()

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())

    @classmethod
    def from_payload(cls, payload: object) -> DatasetSelectionDecision:
        expected = (
            "status",
            "catalog_hash",
            "selected_entry_hash",
            "selected_locator",
            "reason_codes",
            "candidate_model_executions",
        )
        values = _strict_fields(payload, expected=expected, label="selection decision")
        try:
            status = DatasetSelectionStatus(values["status"])
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown selection status") from exc
        return cls(
            status=status,
            catalog_hash=values["catalog_hash"],
            selected_entry_hash=values["selected_entry_hash"],
            selected_locator=values["selected_locator"],
            reason_codes=tuple(values["reason_codes"]),
            candidate_model_executions=values["candidate_model_executions"],
        )


def select_dataset_candidate(
    catalog: DatasetCandidateCatalog,
) -> DatasetSelectionDecision:
    if not isinstance(catalog, DatasetCandidateCatalog):
        raise TypeError("catalog must be DatasetCandidateCatalog")
    if not catalog.is_complete:
        return DatasetSelectionDecision.infrastructure_incomplete(
            catalog_hash=catalog.content_hash,
            reason_codes=catalog.incomplete_reason_codes,
        )
    eligible = tuple(
        entry for entry in catalog.entries if entry.is_eligible_for(catalog.protocol)
    )
    if not eligible:
        return DatasetSelectionDecision.scientific_red(
            catalog_hash=catalog.content_hash,
            reason_codes=("NO_ELIGIBLE_DATASET",),
        )
    selected = min(
        eligible,
        key=lambda entry: (
            -entry.eligible_participant_count,
            entry.public_release_date,
            entry.canonical_repository_locator,
        ),
    )
    return DatasetSelectionDecision.selected(
        catalog_hash=catalog.content_hash,
        selected_entry_hash=selected.content_hash,
        selected_locator=selected.canonical_repository_locator,
    )


__all__ = [
    "CROSS_DATASET_TRANSFER_CLAIM_SCOPE",
    "DatasetCandidateCatalog",
    "DatasetCatalogEntry",
    "DatasetSearchProtocol",
    "DatasetSelectionDecision",
    "DatasetSelectionStatus",
    "select_dataset_candidate",
]

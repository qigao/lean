"""Synthetic-only fixtures for Cross-Dataset Transfer V1 Phase A tests."""

from __future__ import annotations

import hashlib

from narrative_dynamics.cross_dataset_search import (
    DatasetCandidateCatalog,
    DatasetCatalogEntry,
    DatasetSearchProtocol,
)


def digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def search_protocol(**overrides: object) -> DatasetSearchProtocol:
    values: dict[str, object] = {
        "public_release_cutoff": "2026-08-30",
        "query_strings": (
            "sequential two-stage decision dataset",
            "two-step reinforcement learning public data",
        ),
        "repository_families": ("doi_archive", "paper_repository"),
        "query_date": "2026-08-30",
        "interface_versions": (
            ("doi_archive", "api-v1"),
            ("paper_repository", "ui-v2"),
        ),
        "pagination_limit": 2,
        "sort_order": "release_date_ascending_then_locator",
        "deduplication_key": (
            "study_doi",
            "canonical_repository_locator",
            "immutable_release",
        ),
        "eligibility_fields": (
            "binary_first_stage_choice",
            "chronological_participant_sequence",
            "complete_consumed_file_inventory",
            "independent_collection",
            "independent_from_anchor",
            "license_evidence",
            "reward_available",
            "source_declared_exclusions_available",
            "transition_or_state_available",
        ),
        "stopping_rule": "record_every_page_and_resolve_every_doi_repository",
        "candidate_model_code_available": False,
    }
    values.update(overrides)
    return DatasetSearchProtocol(**values)


def catalog_entry(
    slug: str,
    participant_count: int = 180,
    release_date: str = "2020-01-02",
    **overrides: object,
) -> DatasetCatalogEntry:
    values: dict[str, object] = {
        "study_doi": f"10.0000/example.{slug}",
        "canonical_repository_locator": f"https://example.invalid/{slug}",
        "immutable_release": f"release-{slug}",
        "public_release_date": release_date,
        "eligible_participant_count": participant_count,
        "repository_resolution_receipt_hash": digest(f"receipt-{slug}"),
        "license_name": "Synthetic Test License",
        "license_reference": f"https://example.invalid/{slug}/license",
        "consumed_file_count": 2,
        "independent_collection": True,
        "complete_consumed_file_inventory": True,
        "binary_first_stage_choice": True,
        "chronological_participant_sequence": True,
        "transition_or_state_available": True,
        "reward_available": True,
        "source_declared_exclusions_available": True,
        "independent_from_anchor": True,
        "candidate_model_executions": 0,
        "exclusion_codes": (),
    }
    values.update(overrides)
    return DatasetCatalogEntry(**values)


def eligible_entry(
    slug: str,
    participant_count: int = 180,
    release_date: str = "2020-01-02",
) -> DatasetCatalogEntry:
    return catalog_entry(slug, participant_count, release_date)


def ineligible_entry(slug: str = "too-small") -> DatasetCatalogEntry:
    return catalog_entry(slug, participant_count=149)


def complete_catalog(
    *,
    entries: tuple[DatasetCatalogEntry, ...] | None = None,
    protocol: DatasetSearchProtocol | None = None,
) -> DatasetCandidateCatalog:
    selected_protocol = search_protocol() if protocol is None else protocol
    selected_entries = (
        (eligible_entry("alpha"), eligible_entry("beta", 170))
        if entries is None
        else entries
    )
    return DatasetCandidateCatalog.create(
        protocol=selected_protocol,
        pages_recorded=selected_protocol.required_pages,
        resolved_repository_receipts=tuple(
            entry.repository_resolution_receipt_hash for entry in selected_entries
        ),
        entries=selected_entries,
        candidate_model_executions=0,
    )


def incomplete_catalog(
    *,
    missing_page: tuple[str, int] = ("doi_archive", 2),
) -> DatasetCandidateCatalog:
    protocol = search_protocol()
    entries = (eligible_entry("alpha"),)
    return DatasetCandidateCatalog.create(
        protocol=protocol,
        pages_recorded=tuple(
            page for page in protocol.required_pages if page != missing_page
        ),
        resolved_repository_receipts=(entries[0].repository_resolution_receipt_hash,),
        entries=entries,
        candidate_model_executions=0,
    )

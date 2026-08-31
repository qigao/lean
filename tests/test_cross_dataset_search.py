from __future__ import annotations

import unittest

from narrative_dynamics.cross_dataset_search import (
    CROSS_DATASET_TRANSFER_CLAIM_SCOPE,
    DatasetCandidateCatalog,
    DatasetCatalogEntry,
    DatasetSearchProtocol,
    DatasetSelectionDecision,
    DatasetSelectionStatus,
    select_dataset_candidate,
)
from tests.cross_dataset_transfer_fixtures import (
    catalog_entry,
    complete_catalog,
    digest,
    eligible_entry,
    incomplete_catalog,
    ineligible_entry,
    search_protocol,
)


class CrossDatasetSearchTests(unittest.TestCase):
    def test_catalog_selection_is_complete_zero_model_and_deterministic(self) -> None:
        catalog = complete_catalog(
            entries=(
                eligible_entry("b", 180, "2020-01-02"),
                eligible_entry("a", 180, "2020-01-02"),
            )
        )

        decision = select_dataset_candidate(catalog)

        self.assertIs(decision.status, DatasetSelectionStatus.SELECTED)
        self.assertEqual(decision.selected_locator, "https://example.invalid/a")
        self.assertEqual(decision.catalog_hash, catalog.content_hash)
        self.assertEqual(decision.candidate_model_executions, 0)

    def test_incomplete_catalog_is_infrastructure_incomplete(self) -> None:
        decision = select_dataset_candidate(incomplete_catalog())

        self.assertIs(
            decision.status,
            DatasetSelectionStatus.INFRASTRUCTURE_INCOMPLETE,
        )
        self.assertEqual(decision.reason_codes, ("CATALOG_INCOMPLETE",))
        self.assertIsNone(decision.selected_locator)

    def test_unresolved_doi_repository_is_infrastructure_incomplete(self) -> None:
        protocol = search_protocol()
        entry = eligible_entry("unresolved")
        catalog = DatasetCandidateCatalog.create(
            protocol=protocol,
            pages_recorded=protocol.required_pages,
            resolved_repository_receipts=(),
            entries=(entry,),
            candidate_model_executions=0,
        )

        decision = select_dataset_candidate(catalog)

        self.assertIs(
            decision.status,
            DatasetSelectionStatus.INFRASTRUCTURE_INCOMPLETE,
        )
        self.assertEqual(
            decision.reason_codes,
            ("DOI_REPOSITORY_UNRESOLVED",),
        )

    def test_no_eligible_entry_is_scientific_red_without_relaxation(self) -> None:
        decision = select_dataset_candidate(
            complete_catalog(entries=(ineligible_entry(),))
        )

        self.assertIs(decision.status, DatasetSelectionStatus.SCIENTIFIC_RED)
        self.assertEqual(decision.reason_codes, ("NO_ELIGIBLE_DATASET",))
        self.assertIsNone(decision.selected_entry_hash)

    def test_selection_prefers_count_then_release_then_locator(self) -> None:
        by_count = select_dataset_candidate(
            complete_catalog(
                entries=(
                    eligible_entry("older-small", 170, "2019-01-01"),
                    eligible_entry("newer-large", 180, "2020-01-01"),
                )
            )
        )
        by_release = select_dataset_candidate(
            complete_catalog(
                entries=(
                    eligible_entry("newer", 180, "2020-01-02"),
                    eligible_entry("older", 180, "2020-01-01"),
                )
            )
        )
        by_locator = select_dataset_candidate(
            complete_catalog(
                entries=(
                    eligible_entry("z", 180, "2020-01-01"),
                    eligible_entry("a", 180, "2020-01-01"),
                )
            )
        )

        self.assertEqual(
            by_count.selected_locator,
            "https://example.invalid/newer-large",
        )
        self.assertEqual(
            by_release.selected_locator,
            "https://example.invalid/older",
        )
        self.assertEqual(by_locator.selected_locator, "https://example.invalid/a")

    def test_search_protocol_rejects_empty_duplicate_or_unordered_queries(self) -> None:
        for query_strings in (
            (),
            ("same", "same"),
            ("z-query", "a-query"),
        ):
            with self.subTest(query_strings=query_strings):
                with self.assertRaises(ValueError):
                    search_protocol(query_strings=query_strings)

        with self.assertRaises(ValueError):
            search_protocol(repository_families=("paper_repository", "doi_archive"))
        with self.assertRaises(ValueError):
            search_protocol(candidate_model_code_available=True)
        with self.assertRaises(ValueError):
            search_protocol(public_release_cutoff="2026-8-30")
        with self.assertRaises(ValueError):
            search_protocol(query_date="not-a-date")

    def test_entry_requires_complete_inventory_license_and_same_family_endpoint(self) -> None:
        false_fields = (
            "independent_collection",
            "complete_consumed_file_inventory",
            "binary_first_stage_choice",
            "chronological_participant_sequence",
            "transition_or_state_available",
            "reward_available",
            "source_declared_exclusions_available",
            "independent_from_anchor",
        )
        for field in false_fields:
            with self.subTest(field=field):
                entry = catalog_entry(field, **{field: False})
                self.assertFalse(entry.is_eligible_for(search_protocol()))

        self.assertFalse(
            catalog_entry("no-license", license_name="").is_eligible_for(
                search_protocol()
            )
        )
        self.assertFalse(
            catalog_entry("no-files", consumed_file_count=0).is_eligible_for(
                search_protocol()
            )
        )
        self.assertFalse(
            catalog_entry("late", release_date="2026-08-31").is_eligible_for(
                search_protocol()
            )
        )

    def test_catalog_rejects_model_execution_and_duplicate_identity(self) -> None:
        with self.assertRaises(ValueError):
            catalog_entry("modeled", candidate_model_executions=1)
        with self.assertRaises(ValueError):
            DatasetCandidateCatalog.create(
                protocol=search_protocol(),
                pages_recorded=search_protocol().required_pages,
                resolved_repository_receipts=(),
                entries=(),
                candidate_model_executions=1,
            )
        duplicate = eligible_entry("duplicate")
        with self.assertRaises(ValueError):
            complete_catalog(entries=(duplicate, duplicate))

    def test_records_reject_noncanonical_scalar_values(self) -> None:
        invalid_entry_values = (
            {"canonical_repository_locator": "http://example.invalid/source"},
            {"canonical_repository_locator": "https://example.invalid/source "},
            {"public_release_date": "2020-1-01"},
            {"eligible_participant_count": True},
            {"eligible_participant_count": -1},
            {"repository_resolution_receipt_hash": "SHA256:bad"},
            {"study_doi": ""},
            {"immutable_release": " release"},
        )
        for values in invalid_entry_values:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    catalog_entry("invalid", **values)

    def test_all_payload_decoders_reject_unknown_or_missing_fields(self) -> None:
        records = (
            search_protocol(),
            eligible_entry("payload"),
            complete_catalog(),
            select_dataset_candidate(complete_catalog()),
        )
        decoders = (
            DatasetSearchProtocol.from_payload,
            DatasetCatalogEntry.from_payload,
            DatasetCandidateCatalog.from_payload,
            DatasetSelectionDecision.from_payload,
        )
        for record, decoder in zip(records, decoders, strict=True):
            payload = record.to_payload()
            self.assertEqual(decoder(payload), record)
            with self.subTest(record=type(record).__name__, drift="unknown"):
                with self.assertRaises(ValueError):
                    decoder({**payload, "unknown": "field"})
            with self.subTest(record=type(record).__name__, drift="missing"):
                first_key = next(iter(payload))
                with self.assertRaises(ValueError):
                    decoder({key: value for key, value in payload.items() if key != first_key})

    def test_claim_scope_is_closed(self) -> None:
        self.assertEqual(
            CROSS_DATASET_TRANSFER_CLAIM_SCOPE,
            "external_observational_cross_dataset_predictive_transfer_only",
        )
        self.assertTrue(digest("scope").startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()

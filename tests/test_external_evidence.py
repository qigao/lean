from __future__ import annotations

import unittest

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.observations import load_observation_dataset

from tests.external_validation_fixtures import declaration_kwargs, external_dataset

try:
    from narrative_dynamics.observations.external import (
        EXTERNAL_CLAIM_SCOPE,
        ExternalEvidenceDeclaration,
        ExternalValidationError,
    )
except ImportError as error:
    EXTERNAL_CLAIM_SCOPE = None
    ExternalEvidenceDeclaration = None
    ExternalValidationError = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


class ExternalEvidenceDeclarationTests(unittest.TestCase):
    def _api(self):
        if ExternalEvidenceDeclaration is None:
            self.fail(f"external evidence API is missing: {IMPORT_ERROR}")

    def _declaration(self, dataset=None, **overrides):
        self._api()
        kwargs = declaration_kwargs()
        kwargs.update(overrides)
        return ExternalEvidenceDeclaration.from_dataset(
            external_dataset() if dataset is None else dataset,
            **kwargs,
        )

    def test_positive_external_origin_builds_stable_declaration(self):
        dataset = external_dataset()
        first = self._declaration(dataset)
        second = self._declaration(dataset)
        self.assertEqual(first, second)
        self.assertEqual(first.dataset_hash, dataset.content_hash)
        self.assertEqual(first.evidence_origin, "external_observational")
        self.assertEqual(first.claim_scope, EXTERNAL_CLAIM_SCOPE)
        self.assertTrue(first.content_hash.startswith("sha256:"))
        first.require_matches(dataset)

    def test_absence_of_positive_external_origin_is_rejected(self):
        self._api()
        with self.assertRaises(ExternalValidationError):
            self._declaration(
                external_dataset(
                    source_kind="unknown",
                    external_observational=False,
                )
            )

    def test_prison_synthetic_fixture_is_rejected(self):
        self._api()
        dataset = load_observation_dataset(
            "fixtures/observations/prison_initial_choice_v1.json"
        )
        with self.assertRaises(ExternalValidationError):
            self._declaration(dataset)

    def test_p2_synthetic_identification_fixture_is_rejected(self):
        self._api()
        dataset = load_observation_dataset(
            "fixtures/observations/narrative_identification_v1.json"
        )
        with self.assertRaises(ExternalValidationError):
            self._declaration(dataset)

    def test_empirical_human_false_alone_does_not_mark_positive_external_data_synthetic(self):
        declaration = self._declaration(
            external_dataset(
                source_kind="external_observational",
                external_observational=True,
                empirical_human_data=False,
            )
        )
        self.assertEqual(declaration.evidence_origin, "external_observational")

    def test_partition_assignment_hash_is_order_stable_and_role_sensitive(self):
        first = self._declaration(external_dataset())
        reordered = self._declaration(external_dataset(reverse_partition_input=True))
        swapped = self._declaration(external_dataset(swap_train_selection=True))
        self.assertEqual(
            first.partition_assignment_hash,
            reordered.partition_assignment_hash,
        )
        self.assertNotEqual(
            first.partition_assignment_hash,
            swapped.partition_assignment_hash,
        )

    def test_dataset_source_transform_namespace_or_partition_drift_is_rejected(self):
        dataset = external_dataset()
        declaration = self._declaration(dataset)
        changed_dataset = external_dataset(swap_train_selection=True)
        with self.assertRaises(ExternalValidationError):
            declaration.require_matches(changed_dataset)
        with self.assertRaises(ExternalValidationError):
            declaration.require_matches(
                dataset,
                source_snapshot_hash=stable_content_hash({"snapshot": "drifted"}),
            )
        with self.assertRaises(ExternalValidationError):
            declaration.require_matches(
                dataset,
                transform_identity={"name": "drifted", "version": "2"},
            )
        with self.assertRaises(ExternalValidationError):
            declaration.require_matches(
                dataset,
                record_namespace="drifted-namespace",
            )


if __name__ == "__main__":
    unittest.main()

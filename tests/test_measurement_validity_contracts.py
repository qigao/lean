from __future__ import annotations

from dataclasses import replace
import math
import unittest

from narrative_dynamics import ExperimentStage
from narrative_dynamics.contracts import Scenario, stable_content_hash
from narrative_dynamics.measurement_validity import (
    MEASUREMENT_CLAIM_SCOPE,
    MeasurementAggregation,
    MeasurementAuditCase,
    MeasurementAuditInput,
    MeasurementScore,
    MeasurementTerminalClass,
    MeasurementValidityProtocol,
    MeasurementValidityStatus,
)
from narrative_dynamics.observations import ObservationPartitionRole

from tests.measurement_validity_fixtures import (
    digest,
    frozen_candidates,
    measurement_cases,
    measurement_input,
    measurement_protocol,
)


class MeasurementValidityContractTests(unittest.TestCase):
    def test_claim_status_and_terminal_vocabularies_are_closed(self) -> None:
        self.assertEqual(
            MEASUREMENT_CLAIM_SCOPE,
            "external_observational_measurement_audit_only",
        )
        self.assertEqual(
            {status.value for status in MeasurementValidityStatus},
            {
                "exact_invariance_met",
                "exact_invariance_failed",
                "stable_under_frozen_audit",
                "materially_measurement_dependent",
                "inconclusive_sensitivity",
                "not_established",
            },
        )
        self.assertEqual(
            tuple(status.value for status in MeasurementTerminalClass),
            ("green", "scientific_red", "infrastructure_incomplete"),
        )
        self.assertEqual(
            tuple(score.value for score in MeasurementScore),
            ("brier", "log"),
        )
        self.assertEqual(
            tuple(rule.value for rule in MeasurementAggregation),
            ("trial_equal", "participant_equal"),
        )

    def test_audit_input_is_canonical_immutable_and_final_free(self) -> None:
        cases = measurement_cases()
        candidates = frozen_candidates()
        audit_input = measurement_input(
            cases=tuple(reversed(cases)),
            candidates=tuple(reversed(candidates)),
        )
        canonical = measurement_input()

        self.assertEqual(audit_input.content_hash, canonical.content_hash)
        self.assertEqual(
            tuple(candidate.name for candidate in audit_input.frozen_candidates),
            ("reactive", "intentional", "planning"),
        )
        self.assertEqual(
            {case.role for case in audit_input.cases},
            {
                ObservationPartitionRole.TRAIN,
                ObservationPartitionRole.SELECTION_VALIDATION,
            },
        )
        payload_text = repr(audit_input.identity_payload())
        self.assertNotIn("participant_id", payload_text)
        self.assertNotIn("source_participant_id", payload_text)
        self.assertNotIn("final_test", tuple(role.value for role, _ in audit_input.allowed_partition_hashes))

        original_hash = audit_input.content_hash
        mutable_identity = {"name": "mutable", "version": "1"}
        protocol = measurement_protocol(
            implementation_identities=(("mutable", mutable_identity),)
        )
        protocol_hash = protocol.content_hash
        mutable_identity["version"] = "changed"
        self.assertEqual(protocol.content_hash, protocol_hash)
        self.assertEqual(audit_input.content_hash, original_hash)

    def test_case_rejects_final_direct_identity_and_invalid_targets(self) -> None:
        case = measurement_cases()[0]
        with self.assertRaisesRegex(ValueError, "TRAIN or SELECTION_VALIDATION"):
            replace(case, role=ObservationPartitionRole.FINAL_TEST)

        direct_identity_scenario = Scenario(
            id="direct-identity",
            payload={
                "task_variant": "magic_carpet",
                "source_participant_id": "human-1",
                "history": (),
            },
        )
        with self.assertRaisesRegex(ValueError, "direct participant identity"):
            replace(case, scenario=direct_identity_scenario)

        nested_identity_scenario = Scenario(
            id="nested-direct-identity",
            payload={
                "task_variant": "magic_carpet",
                "history": ({"participant_id": "human-1"},),
            },
        )
        with self.assertRaisesRegex(ValueError, "direct participant identity"):
            replace(case, scenario=nested_identity_scenario)

        invalid_rows = (
            (("first_stage.action_0", 0.8), ("first_stage.action_1", 0.3)),
            (("first_stage.action_0", 1.0),),
            (("first_stage.action_0", math.nan), ("first_stage.action_1", 0.0)),
            (("action_0", 1.0), ("action_1", 0.0)),
        )
        for target in invalid_rows:
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    replace(case, target=target)

        with self.assertRaisesRegex(ValueError, "task variant"):
            replace(case, task_variant="unknown")
        with self.assertRaisesRegex(ValueError, "opaque participant"):
            replace(case, participant_group_hash="human-1")

    def test_input_rejects_duplicate_or_incomplete_projection(self) -> None:
        cases = measurement_cases()
        duplicate_mutations = (
            cases + (cases[0],),
            cases + (replace(cases[0], record_hash=cases[1].record_hash),),
            cases
            + (
                replace(
                    cases[0],
                    participant_group_hash=digest("e"),
                    trial_index=99,
                    record_hash=digest("f"),
                ),
            ),
            cases + (
                (
                    replace(
                        cases[0],
                        record_hash=digest("f"),
                        scenario=Scenario(id="new", payload=cases[0].scenario.payload),
                    )
                ),
            ),
        )
        for values in duplicate_mutations:
            with self.subTest(size=len(values)):
                with self.assertRaises(ValueError):
                    measurement_input(cases=values)

        train_only = tuple(
            case for case in cases if case.role is ObservationPartitionRole.TRAIN
        )
        with self.assertRaisesRegex(ValueError, "both TRAIN and SELECTION_VALIDATION"):
            measurement_input(cases=train_only)

        candidate_rows = frozen_candidates()
        for values in ((), candidate_rows[:2], candidate_rows + (candidate_rows[0],)):
            with self.subTest(candidate_count=len(values)):
                with self.assertRaisesRegex(ValueError, "exactly three"):
                    measurement_input(candidates=values)

    def test_input_rejects_bad_commitments_and_final_lineage(self) -> None:
        audit_input = measurement_input()
        with self.assertRaisesRegex(ValueError, "case commitment"):
            replace(
                audit_input,
                allowed_case_commitments=(
                    (ObservationPartitionRole.TRAIN, digest("f")),
                    audit_input.allowed_case_commitments[1],
                ),
            )
        with self.assertRaisesRegex(ValueError, "excluded FINAL"):
            replace(
                audit_input,
                excluded_final_target_hash=audit_input.excluded_final_partition_hash,
            )

    def test_protocol_round_trip_is_exact_and_rejects_schema_drift(self) -> None:
        protocol = measurement_protocol()
        payload = protocol.to_payload()
        rebuilt = MeasurementValidityProtocol.from_payload(payload)
        self.assertEqual(rebuilt, protocol)
        self.assertEqual(rebuilt.content_hash, protocol.content_hash)

        unknown = dict(payload)
        unknown["scientific_conclusion"] = "planning wins"
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            MeasurementValidityProtocol.from_payload(unknown)

        missing = dict(payload)
        missing.pop("claim_scope")
        with self.assertRaisesRegex(ValueError, "missing fields"):
            MeasurementValidityProtocol.from_payload(missing)

    def test_protocol_rejects_final_seeds_duplicates_and_invalid_references(self) -> None:
        protocol = measurement_protocol()
        with self.assertRaisesRegex(ValueError, "FINAL seeds"):
            replace(
                protocol,
                seeds_by_role=(
                    (ObservationPartitionRole.TRAIN, (101, 102)),
                    (ObservationPartitionRole.SELECTION_VALIDATION, (301, 302)),
                ),
            )

        duplicate_cases = (
            ("semantic_permutations", ("swap", "swap")),
            ("task_strata", ("magic_carpet", "magic_carpet")),
            (
                "aggregation_rules",
                (
                    MeasurementAggregation.TRIAL_EQUAL,
                    MeasurementAggregation.TRIAL_EQUAL,
                ),
            ),
            ("scores", (MeasurementScore.BRIER, MeasurementScore.BRIER)),
            (
                "implementation_identities",
                (
                    ("same", {"name": "one"}),
                    ("same", {"name": "two"}),
                ),
            ),
        )
        for field, value in duplicate_cases:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    replace(protocol, **{field: value})

        for reference in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(reference=reference):
                with self.assertRaises(ValueError):
                    replace(
                        protocol,
                        material_reversal_references=(
                            (MeasurementScore.BRIER, reference),
                            (MeasurementScore.LOG, 0.006931471805599453),
                        ),
                    )

    def test_protocol_manifest_binds_measurement_stage_input_and_candidates(self) -> None:
        audit_input = measurement_input()
        protocol = measurement_protocol(audit_input)
        manifest = protocol.build_manifest(audit_input)
        self.assertIs(manifest.stage, ExperimentStage.MEASUREMENT_AUDIT)
        self.assertEqual(manifest.parent_hashes, (audit_input.content_hash,))
        self.assertEqual(
            manifest.inputs["candidate_hashes"],
            tuple(candidate.content_hash for candidate in audit_input.frozen_candidates),
        )

        changed_input = replace(audit_input, dataset_hash=digest("f"))
        with self.assertRaisesRegex(ValueError, "empirical anchor"):
            protocol.build_manifest(changed_input)

    def test_case_hash_and_commitment_are_derived_from_projection(self) -> None:
        case = measurement_cases()[0]
        self.assertEqual(
            case.case_hash,
            stable_content_hash(case.identity_payload()),
        )
        changed = replace(case, trial_index=case.trial_index + 1)
        self.assertNotEqual(changed.case_hash, case.case_hash)


if __name__ == "__main__":
    unittest.main()

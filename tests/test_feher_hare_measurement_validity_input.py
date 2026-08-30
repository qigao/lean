from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.measurement_validity import MeasurementAuditInput
from narrative_dynamics.observations import (
    ObservationDataset,
    ObservationPartitionRole,
)
from narrative_dynamics.studies.feher_hare_measurement_validity_v1 import (
    FEHER_HARE_R3_LOCK_COMMIT,
    FeherHareMeasurementAnchor,
    _project_prepared_measurement_input,
    freeze_feher_hare_measurement_candidates,
    load_feher_hare_r3_measurement_anchor,
    provision_feher_hare_measurement_input,
)
from narrative_dynamics.studies.feher_hare_two_stage_v1 import (
    PreparedFeherHareTwoStageV1,
    prepare_feher_hare_two_stage_v1,
)
from narrative_dynamics.studies.two_stage_source import TwoStageSourceManifest

from tests.test_two_stage_source import _files
from tests.two_stage_test_support import build_synthetic_two_stage_checkout


_SCIENTIFIC_REVISION = "d01232979cdfc9d902daab5f9e3e937079b56f69"
_UPSTREAM_REVISION = "4567763780a2c596fd6510af720ec468a8214a8f"
_SOURCE_MANIFEST_HASH = (
    "sha256:3609e980af172823cfef78290e7f2337fdb337d67f1f27641e17d473aeedd10d"
)
_SOURCE_SNAPSHOT_HASH = (
    "sha256:25bdc2e4bff4110f38b098b89e7d59aa38c9658727fcc89a38d50e107d243186"
)
_TRANSFORM_HASH = (
    "sha256:2fb8ab6dc796a9e4ece5653a5485d865ef44dd0f5f7dff92d94a904932d6e541"
)
_ASSIGNMENT_HASH = (
    "sha256:fc144c35f6713e27f75141d678872cc04ca44c7a0fd8e109e8618c72b4111ccf"
)
_DATASET_HASH = (
    "sha256:17789130372d7eace05e1216a57bdae2ffbd519960333ffe814aee2d2d404781"
)
_TARGET_SPEC_HASH = (
    "sha256:3134c9dc424418c87379f8e451420fb2defe81b8f30a45d67cd9b1f453349713"
)
_TRAIN_PARTITION_HASH = (
    "sha256:71ce56243338eb23b9dd5ad6dd901e4b0d0be4989742b66bbe7ea4f025f207da"
)
_SELECTION_PARTITION_HASH = (
    "sha256:ecdcfa1681b58888ebf4419372d5de7b75be9624cdd3307144371c5486eb1347"
)
_FINAL_PARTITION_HASH = (
    "sha256:936ffe872644e888111007b300ea847484698be8fbfd7c2d54a177505a411047"
)
_FINAL_TARGET_HASH = (
    "sha256:927e1727d37993e9a5c887f79ac8712d07a155622deacf0a3122d41f90b16ed2"
)
_FREEZE_HASH = (
    "sha256:77fa1bb80ab0c7ac737a09a8001f1d0c1432ce3bd991fd7191d07fdd0d8e05d5"
)
_LOCK_BUNDLE_HASH = (
    "sha256:96e553e557d6e7314eb0b9b1d0aaa8696e01d7a6ddec0abde949be8c8d45602f"
)

_CANDIDATES = {
    "reactive": {
        "parameters": [["beta", 0.5]],
        "selection_manifest_hash": (
            "sha256:e8414e301d19fc6acfd5accf055402ac995790f785402186c67ff95a48ee0c2e"
        ),
        "training_manifest_hash": (
            "sha256:5b607a4bc0a8082809c2446a8248fb8659b0d9ba178687ddad96dc74e3f19622"
        ),
        "frozen_candidate_hash": (
            "sha256:42a84ef10c207159ff98397fe2bd86594df6c8be0f41b98e876f7ae33cba9456"
        ),
    },
    "intentional": {
        "parameters": [["beta", 2.0], ["memory_decay", 0.5]],
        "selection_manifest_hash": (
            "sha256:498a512cd931afe276f78bf135bd5c8269e051920d4b876177d0aa9ea250806d"
        ),
        "training_manifest_hash": (
            "sha256:47057311fe7450469c1710be49ba0e3b42aa7416fa2129f40761dbf8d237d4fc"
        ),
        "frozen_candidate_hash": (
            "sha256:68a01b5496e20095c1a603b30f1384bbe6a3ea22aac8cf74b7b97463c4935839"
        ),
    },
    "planning": {
        "parameters": [["beta", 4.0], ["memory_decay", 0.75]],
        "selection_manifest_hash": (
            "sha256:e347a3d7d523e2a35d0bb6b0f662343f5a01ddd1ff7d2efe95a8a5efa1986e74"
        ),
        "training_manifest_hash": (
            "sha256:ef0f3caf4f2a02b2d719bc302d7409fc8c2d937eb13517d906d83b38c29b76ac"
        ),
        "frozen_candidate_hash": (
            "sha256:c0ed956c1b285153075e5a22e7fb51f6d53dcf8ad326cce82e39e0221c2e877c"
        ),
    },
}


def _lock_payload() -> dict[str, object]:
    return {
        "scientific_repository_revision": _SCIENTIFIC_REVISION,
        "upstream_revision": _UPSTREAM_REVISION,
        "source_manifest_hash": _SOURCE_MANIFEST_HASH,
        "source_snapshot_hash": _SOURCE_SNAPSHOT_HASH,
        "transform_hash": _TRANSFORM_HASH,
        "participant_assignment_hash": _ASSIGNMENT_HASH,
        "dataset_hash": _DATASET_HASH,
        "final_target_hash": _FINAL_TARGET_HASH,
        "train_selection_freeze": {
            "artifact_payload_digest": _FREEZE_HASH,
            "families": deepcopy(_CANDIDATES),
        },
        "frozen_candidates": [
            {
                "name": family,
                "parameters": deepcopy(row["parameters"]),
                "selection_manifest_hash": row["selection_manifest_hash"],
                "content_hash": row["frozen_candidate_hash"],
            }
            for family, row in _CANDIDATES.items()
        ],
        "brier_protocol": {
            "dataset_hash": _DATASET_HASH,
            "train_partition_hash": _TRAIN_PARTITION_HASH,
            "selection_partition_hash": _SELECTION_PARTITION_HASH,
            "final_partition_hash": _FINAL_PARTITION_HASH,
            "target_spec_hash": _TARGET_SPEC_HASH,
        },
        "log_protocol": {
            "dataset_hash": _DATASET_HASH,
            "train_partition_hash": _TRAIN_PARTITION_HASH,
            "selection_partition_hash": _SELECTION_PARTITION_HASH,
            "final_partition_hash": _FINAL_PARTITION_HASH,
            "target_spec_hash": _TARGET_SPEC_HASH,
        },
        "internal_lock_bundle": {
            "content_hash": _LOCK_BUNDLE_HASH,
            "dataset_hash": _DATASET_HASH,
            "participant_assignment_hash": _ASSIGNMENT_HASH,
            "final_target_hash": _FINAL_TARGET_HASH,
            "frozen_candidate_hashes": [
                _CANDIDATES[family]["frozen_candidate_hash"]
                for family in ("reactive", "intentional", "planning")
            ],
            "scientific_repository_revision": _SCIENTIFIC_REVISION,
            "source_manifest_hash": _SOURCE_MANIFEST_HASH,
            "source_snapshot_hash": _SOURCE_SNAPSHOT_HASH,
            "transform_hash": _TRANSFORM_HASH,
            "train_selection_freeze_digest": _FREEZE_HASH,
        },
    }


def _write_lock(root: Path, payload: object) -> Path:
    path = root / "measurement-lock.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _load_fixture_anchor(root: Path) -> FeherHareMeasurementAnchor:
    return load_feher_hare_r3_measurement_anchor(
        _write_lock(root, _lock_payload()),
        lock_commit=FEHER_HARE_R3_LOCK_COMMIT,
    )


class FeherHareMeasurementAnchorTests(unittest.TestCase):
    def test_exact_r3_lock_loads_and_anchor_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            anchor = _load_fixture_anchor(Path(temporary))
        self.assertEqual(FEHER_HARE_R3_LOCK_COMMIT, "f09d6afc96f0a720dd5e8e810f440cda0fe9f8a3")
        self.assertEqual(anchor.lock_commit, FEHER_HARE_R3_LOCK_COMMIT)
        self.assertEqual(anchor.scientific_repository_revision, _SCIENTIFIC_REVISION)
        self.assertEqual(anchor.upstream_revision, _UPSTREAM_REVISION)
        self.assertEqual(anchor.source_manifest_hash, _SOURCE_MANIFEST_HASH)
        self.assertEqual(anchor.source_snapshot_hash, _SOURCE_SNAPSHOT_HASH)
        self.assertEqual(anchor.transform_hash, _TRANSFORM_HASH)
        self.assertEqual(anchor.participant_assignment_hash, _ASSIGNMENT_HASH)
        self.assertEqual(anchor.dataset_hash, _DATASET_HASH)
        self.assertEqual(anchor.target_spec_hash, _TARGET_SPEC_HASH)
        self.assertEqual(anchor.train_partition_hash, _TRAIN_PARTITION_HASH)
        self.assertEqual(anchor.selection_partition_hash, _SELECTION_PARTITION_HASH)
        self.assertEqual(anchor.excluded_final_partition_hash, _FINAL_PARTITION_HASH)
        self.assertEqual(anchor.excluded_final_target_hash, _FINAL_TARGET_HASH)
        self.assertEqual(anchor.train_selection_freeze_hash, _FREEZE_HASH)
        self.assertEqual(anchor.internal_lock_bundle_hash, _LOCK_BUNDLE_HASH)
        self.assertEqual(
            tuple(row.family for row in anchor.candidate_rows),
            ("reactive", "intentional", "planning"),
        )
        self.assertEqual(
            anchor,
            FeherHareMeasurementAnchor.from_payload(anchor.to_payload()),
        )

    def test_loader_rejects_wrong_commit_or_any_frozen_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_lock(root, _lock_payload())
            with self.assertRaisesRegex(ValueError, "lock commit"):
                load_feher_hare_r3_measurement_anchor(
                    path,
                    lock_commit="0" * 40,
                )

            mutations = (
                (("scientific_repository_revision",), "scientific"),
                (("upstream_revision",), "upstream"),
                (("source_manifest_hash",), "source manifest"),
                (("source_snapshot_hash",), "source snapshot"),
                (("transform_hash",), "transform"),
                (("participant_assignment_hash",), "participant assignment"),
                (("dataset_hash",), "dataset"),
                (("final_target_hash",), "FINAL target"),
                (
                    ("train_selection_freeze", "artifact_payload_digest"),
                    "TRAIN/SELECTION freeze",
                ),
                (("brier_protocol", "target_spec_hash"), "target spec"),
                (("brier_protocol", "train_partition_hash"), "TRAIN partition"),
                (
                    ("brier_protocol", "selection_partition_hash"),
                    "SELECTION partition",
                ),
                (("brier_protocol", "final_partition_hash"), "FINAL partition"),
                (("internal_lock_bundle", "content_hash"), "lock bundle"),
            )
            for field_path, label in mutations:
                with self.subTest(field_path=field_path):
                    payload = _lock_payload()
                    cursor = payload
                    for key in field_path[:-1]:
                        cursor = cursor[key]
                    cursor[field_path[-1]] = (
                        "0" * 40
                        if field_path[-1].endswith("revision")
                        else "sha256:" + "0" * 64
                    )
                    changed = _write_lock(root, payload)
                    with self.assertRaisesRegex(ValueError, label):
                        load_feher_hare_r3_measurement_anchor(
                            changed,
                            lock_commit=FEHER_HARE_R3_LOCK_COMMIT,
                        )

    def test_loader_cross_checks_candidate_duplicates_and_anchor_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _lock_payload()
            payload["frozen_candidates"][0]["content_hash"] = "sha256:" + "0" * 64
            with self.assertRaisesRegex(ValueError, "candidate"):
                load_feher_hare_r3_measurement_anchor(
                    _write_lock(root, payload),
                    lock_commit=FEHER_HARE_R3_LOCK_COMMIT,
                )

            anchor = _load_fixture_anchor(root)
            anchor_payload = anchor.to_payload()
            missing = dict(anchor_payload)
            missing.pop("dataset_hash")
            with self.assertRaisesRegex(ValueError, "missing"):
                FeherHareMeasurementAnchor.from_payload(missing)
            unknown = dict(anchor_payload)
            unknown["winner"] = "planning"
            with self.assertRaisesRegex(ValueError, "unknown"):
                FeherHareMeasurementAnchor.from_payload(unknown)

    def test_candidates_are_reconstructed_without_training_or_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            anchor = _load_fixture_anchor(Path(temporary))
        forbidden = (
            "narrative_dynamics.calibration._grid_candidates",
            "narrative_dynamics.observations.training.fit_training_target_grid",
            "narrative_dynamics.validation.select_on_validation_suite",
            "narrative_dynamics.uncertainty.ParameterAcceptanceSet.from_parameters",
            "narrative_dynamics.observations.preregistration.FrozenModelSpec.from_selection",
        )
        with ExitStack() as stack:
            mocks = tuple(
                stack.enter_context(
                    patch(path, side_effect=AssertionError(f"forbidden call: {path}"))
                )
                for path in forbidden
            )
            candidates = freeze_feher_hare_measurement_candidates(anchor)
        for mocked in mocks:
            mocked.assert_not_called()
        self.assertEqual(
            tuple(candidate.name for candidate in candidates),
            ("reactive", "intentional", "planning"),
        )
        self.assertEqual(
            tuple(candidate.content_hash for candidate in candidates),
            tuple(
                _CANDIDATES[family]["frozen_candidate_hash"]
                for family in ("reactive", "intentional", "planning")
            ),
        )
        self.assertEqual(
            tuple(dict(candidate.parameters) for candidate in candidates),
            (
                {"beta": 0.5},
                {"beta": 2.0, "memory_decay": 0.5},
                {"beta": 4.0, "memory_decay": 0.75},
            ),
        )


class FeherHareMeasurementProvisionerTests(unittest.TestCase):
    def _prepared(self, root: Path) -> tuple[Path, TwoStageSourceManifest, PreparedFeherHareTwoStageV1]:
        checkout, revision = build_synthetic_two_stage_checkout(
            root,
            magic_n=6,
            spaceship_n=6,
        )
        manifest = TwoStageSourceManifest(
            name="synthetic-measurement-provisioner",
            version="1",
            repository="test/synthetic-two-stage",
            revision=revision,
            license_reference="test-only",
            files=_files(checkout),
        )
        return checkout, manifest, prepare_feher_hare_two_stage_v1(
            root=checkout,
            manifest=manifest,
        )

    def _synthetic_anchor(
        self,
        production: FeherHareMeasurementAnchor,
        prepared: PreparedFeherHareTwoStageV1,
    ) -> FeherHareMeasurementAnchor:
        return replace(
            production,
            upstream_revision=prepared.source_manifest.revision,
            source_manifest_hash=prepared.source_manifest.content_hash,
            source_snapshot_hash=prepared.transform_report.source_snapshot_hash,
            transform_hash=prepared.transform_report.content_hash,
            participant_assignment_hash=prepared.assignment.content_hash,
            dataset_hash=prepared.dataset.content_hash,
            target_spec_hash=prepared.target_spec.content_hash,
            train_partition_hash=prepared.dataset.partition(
                ObservationPartitionRole.TRAIN
            ).content_hash,
            selection_partition_hash=prepared.dataset.partition(
                ObservationPartitionRole.SELECTION_VALIDATION
            ).content_hash,
            excluded_final_partition_hash=prepared.dataset.partition(
                ObservationPartitionRole.FINAL_TEST
            ).content_hash,
            excluded_final_target_hash=prepared.final_targets.content_hash,
        )

    def test_projection_exposes_only_allowed_cases_and_opaque_participants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, _manifest, prepared = self._prepared(root)
            production = _load_fixture_anchor(root)
            anchor = self._synthetic_anchor(production, prepared)
            audit_input = _project_prepared_measurement_input(prepared, anchor)

            self.assertIsInstance(audit_input, MeasurementAuditInput)
            self.assertEqual(
                {case.role for case in audit_input.cases},
                {
                    ObservationPartitionRole.TRAIN,
                    ObservationPartitionRole.SELECTION_VALIDATION,
                },
            )
            self.assertEqual(
                len(audit_input.cases),
                len(prepared.train_targets.cases)
                + len(prepared.selection_targets.cases),
            )
            self.assertEqual(
                audit_input.excluded_final_partition_hash,
                prepared.dataset.partition(
                    ObservationPartitionRole.FINAL_TEST
                ).content_hash,
            )
            self.assertEqual(
                audit_input.excluded_final_target_hash,
                prepared.final_targets.content_hash,
            )
            self.assertFalse(hasattr(audit_input, "partition"))
            self.assertFalse(
                any(
                    isinstance(value, (ObservationDataset, PreparedFeherHareTwoStageV1))
                    for value in vars(audit_input).values()
                )
            )

            source_records = {
                record.id: record
                for role in (
                    ObservationPartitionRole.TRAIN,
                    ObservationPartitionRole.SELECTION_VALIDATION,
                )
                for record in prepared.dataset.partition(role).records
            }
            target_cases = {
                case.name: case
                for report in (prepared.train_targets, prepared.selection_targets)
                for case in report.cases
            }
            projected_by_record_hash = {
                case.record_hash: case for case in audit_input.cases
            }
            for name, target_case in target_cases.items():
                record = source_records[name]
                projected = projected_by_record_hash[target_case.record_hash]
                task = record.metadata["task_variant"]
                participant = record.metadata["source_participant_id"]
                self.assertEqual(
                    projected.participant_group_hash,
                    stable_content_hash((task, participant)),
                )
                self.assertEqual(projected.target, target_case.targets)
                self.assertEqual(projected.trial_index, record.metadata["source_trial_id"])
                self.assertNotEqual(projected.scenario.id, record.scenario.id)

            encoded = str(audit_input.identity_payload())
            self.assertNotIn(str(checkout), encoded)
            for record in source_records.values():
                self.assertNotIn(
                    str(record.metadata["source_participant_id"]),
                    encoded,
                )
            for final_case in prepared.final_targets.cases:
                self.assertNotIn(final_case.name, encoded)
                self.assertNotIn(str(final_case.record_hash), encoded)

    def test_every_full_identity_is_verified_before_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _checkout, _manifest, prepared = self._prepared(root)
            anchor = self._synthetic_anchor(_load_fixture_anchor(root), prepared)
            mutations = (
                ("upstream_revision", "0" * 40, "upstream"),
                ("source_manifest_hash", "sha256:" + "0" * 64, "source manifest"),
                ("source_snapshot_hash", "sha256:" + "0" * 64, "source snapshot"),
                ("transform_hash", "sha256:" + "0" * 64, "transform"),
                (
                    "participant_assignment_hash",
                    "sha256:" + "0" * 64,
                    "participant assignment",
                ),
                ("dataset_hash", "sha256:" + "0" * 64, "dataset"),
                ("target_spec_hash", "sha256:" + "0" * 64, "target spec"),
                (
                    "train_partition_hash",
                    "sha256:" + "0" * 64,
                    "TRAIN partition",
                ),
                (
                    "selection_partition_hash",
                    "sha256:" + "0" * 64,
                    "SELECTION partition",
                ),
                (
                    "excluded_final_partition_hash",
                    "sha256:" + "0" * 64,
                    "FINAL partition",
                ),
                (
                    "excluded_final_target_hash",
                    "sha256:" + "0" * 64,
                    "FINAL target",
                ),
            )
            for field_name, changed_value, label in mutations:
                with self.subTest(field_name=field_name):
                    with self.assertRaisesRegex(ValueError, label):
                        _project_prepared_measurement_input(
                            prepared,
                            replace(anchor, **{field_name: changed_value}),
                        )

    def test_public_provisioner_prepares_then_projects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, manifest, prepared = self._prepared(root)
            anchor = self._synthetic_anchor(_load_fixture_anchor(root), prepared)
            expected = _project_prepared_measurement_input(prepared, anchor)
            with patch(
                "narrative_dynamics.studies.feher_hare_measurement_validity_v1."
                "prepare_feher_hare_two_stage_v1",
                return_value=prepared,
            ) as prepare:
                actual = provision_feher_hare_measurement_input(
                    checkout,
                    manifest,
                    anchor,
                )
            prepare.assert_called_once_with(root=checkout, manifest=manifest)
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()

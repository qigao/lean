"""Synthetic-only fixtures for Cross-Dataset Transfer V1 Phase A tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from narrative_dynamics.cross_dataset_search import (
    DatasetCandidateCatalog,
    DatasetCatalogEntry,
    DatasetSearchProtocol,
)
from narrative_dynamics.cross_dataset_source import (
    CanonicalTransferTrial,
    DatasetSourceFile,
    DatasetSourceManifest,
    SemanticPipelineResult,
    SemanticRelabeling,
    SemanticStageReceipt,
)
from narrative_dynamics.cross_dataset_privacy import (
    PrivateParticipant,
    RestrictedStudySecret,
    assign_participant_roles,
)
from narrative_dynamics.cross_dataset_capabilities import (
    FinalUnlockGrant,
    provision_transfer_capabilities,
)
from narrative_dynamics.cross_dataset_candidates import (
    GridCandidateEvaluation,
    INTENTIONAL_GRID,
    PLANNING_GRID,
    REACTIVE_GRID,
    TransferFamily,
)
from narrative_dynamics.studies.feher_hare_measurement_validity_v1 import (
    FEHER_HARE_R3_LOCK_COMMIT,
    FeherHareMeasurementAnchor,
    FeherHareMeasurementCandidateRow,
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


def source_file(
    path: str,
    payload: bytes,
    purpose: str,
) -> DatasetSourceFile:
    return DatasetSourceFile(
        path=path,
        locator=f"https://example.invalid/files/{path}",
        byte_size=len(payload),
        sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        purpose=purpose,
    )


def source_manifest(
    *,
    files: tuple[DatasetSourceFile, ...] | None = None,
) -> DatasetSourceManifest:
    selected_files = (
        (
            source_file("behavior.csv", b"participant,trial,choice\np1,1,0\n", "behavioral_rows"),
            source_file("README.txt", b"synthetic source\n", "source_readme"),
        )
        if files is None
        else files
    )
    return DatasetSourceManifest(
        selected_catalog_entry_hash=eligible_entry("source").content_hash,
        name="Synthetic two-stage source",
        version="release-source",
        study_reference="10.0000/example.source",
        public_locator="https://example.invalid/source",
        release_date="2020-01-02",
        license_name="Synthetic Test License",
        license_reference="https://example.invalid/source/license",
        files=selected_files,
    )


def synthetic_source_tree(root: Path) -> tuple[Path, DatasetSourceManifest]:
    payloads = {
        "behavior.csv": b"participant,trial,choice\np1,1,0\n",
        "README.txt": b"synthetic source\n",
    }
    for relative, payload in payloads.items():
        (root / relative).write_bytes(payload)
    manifest = source_manifest(
        files=tuple(
            source_file(
                relative,
                payload,
                "behavioral_rows" if relative.endswith(".csv") else "source_readme",
            )
            for relative, payload in sorted(payloads.items())
        )
    )
    return root, manifest


def synthetic_rows() -> tuple[dict[str, object], ...]:
    return (
        {
            "participant": "private-p1",
            "trial": 1,
            "history_action": "action_0",
            "first_stage_action": "action_1",
            "final_state": "state_1",
            "second_stage_action": "second_0",
            "reward": 1,
        },
        {
            "participant": "private-p1",
            "trial": 2,
            "history_action": "action_1",
            "first_stage_action": "action_0",
            "final_state": "state_0",
            "second_stage_action": "second_1",
            "reward": 0,
        },
    )


def five_coherent_relabelings() -> tuple[SemanticRelabeling, ...]:
    coordinates = (
        ("first_stage_action", "action_0", "action_1"),
        ("history_action", "action_0", "action_1"),
        ("final_state", "state_0", "state_1"),
        ("second_stage_action", "second_0", "second_1"),
        ("transition_label", "common", "rare"),
    )
    return tuple(
        SemanticRelabeling(
            name=f"swap_{coordinate}",
            forward_map=((f"{coordinate}:{left}", f"{coordinate}:{right}"),
                         (f"{coordinate}:{right}", f"{coordinate}:{left}")),
            inverse_map=((f"{coordinate}:{left}", f"{coordinate}:{right}"),
                         (f"{coordinate}:{right}", f"{coordinate}:{left}")),
        )
        for coordinate, left, right in coordinates
    )


class RecordingSemanticPipeline:
    def __init__(self, *, drift_on: str | None = None, omit_report: bool = False) -> None:
        self.calls: tuple[str, ...] = ()
        self._drift_on = drift_on
        self._omit_report = omit_report

    def run(
        self,
        rows: tuple[dict[str, object], ...],
        relabeling: SemanticRelabeling,
    ) -> SemanticPipelineResult:
        del rows
        stages = (
            "parse",
            "transform",
            "scenario",
            "predict",
            "score",
            "report",
        )
        if self._omit_report:
            stages = stages[:-1]
        self.calls += stages
        identity = digest("semantic-reference")
        inverse_identity = (
            digest("semantic-drift")
            if relabeling.name == self._drift_on
            else identity
        )
        return SemanticPipelineResult(
            identity=identity,
            inverse_mapped_identity=inverse_identity,
            stage_receipts=tuple(
                SemanticStageReceipt(stage=stage, receipt_hash=digest(f"{relabeling.name}-{stage}"))
                for stage in stages
            ),
        )

    def scenario_hash(self, row: dict[str, object]) -> str:
        prechoice = {
            key: row[key]
            for key in (
                "participant",
                "trial",
                "history_action",
            )
        }
        return "sha256:" + hashlib.sha256(
            json.dumps(prechoice, sort_keys=True).encode("utf-8")
        ).hexdigest()


def recording_semantic_pipeline(
    *,
    drift_on: str | None = None,
    omit_report: bool = False,
) -> RecordingSemanticPipeline:
    return RecordingSemanticPipeline(drift_on=drift_on, omit_report=omit_report)


def same_prechoice_different_postchoice_rows() -> tuple[dict[str, object], dict[str, object]]:
    original = dict(synthetic_rows()[0])
    mutated = {
        **original,
        "first_stage_action": "action_0",
        "final_state": "state_0",
        "second_stage_action": "second_1",
        "reward": 0,
    }
    return original, mutated


def synthetic_stratified_participants(
    counts: tuple[int, ...] = (10, 15),
) -> tuple[PrivateParticipant, ...]:
    return tuple(
        PrivateParticipant(
            participant_id=f"private-s{stratum_index}-p{participant_index:03d}",
            source_stratum=f"s{stratum_index}",
        )
        for stratum_index, count in enumerate(counts)
        for participant_index in range(count)
    )


def assigned_split():
    secret = RestrictedStudySecret.from_bytes(bytes(range(32)))
    return assign_participant_roles(
        namespace="cross-dataset-transfer-v1",
        source_snapshot_hash=digest("snapshot"),
        participants=synthetic_stratified_participants(),
        secret=secret,
        transform_attestation_hash=digest("transform-attestation"),
    )


def synthetic_transfer_trials() -> tuple[CanonicalTransferTrial, ...]:
    return tuple(
        CanonicalTransferTrial(
            participant_key=participant.participant_id,
            trial_id=1,
            source_stratum=participant.source_stratum,
            first_stage_action="action_0" if index % 2 == 0 else "action_1",
            transition_common=index % 3 != 0,
            final_state="state_0" if index % 2 == 0 else "state_1",
            second_stage_action="second_0" if index % 2 == 0 else "second_1",
            reward=index % 2,
            row_commitment=digest(f"transfer-row-{index}"),
        )
        for index, participant in enumerate(synthetic_stratified_participants())
    )


def final_unlock_grant(**overrides: object) -> FinalUnlockGrant:
    values: dict[str, object] = {
        "scientific_revision": "a" * 40,
        "ledger_head_hash": digest("ledger-head"),
        "preflight_hash": digest("preflight"),
        "authorization_receipt_hash": digest("authorization"),
        "brier_release_hash": digest("brier-release"),
        "log_release_hash": digest("log-release"),
        "lock_commit": "b" * 40,
    }
    values.update(overrides)
    return FinalUnlockGrant(**values)


def prepared_transfer():
    private_index, public_manifest = assigned_split()
    prepared, backend = provision_transfer_capabilities(
        source_identity_hash=digest("source-identity"),
        trials=synthetic_transfer_trials(),
        role_index=private_index,
        split_manifest=public_manifest,
    )
    backend.bind_unlock_policy(prepared.final_vault_handle, final_unlock_grant())
    return prepared, backend


def r3_anchor() -> FeherHareMeasurementAnchor:
    candidates = (
        FeherHareMeasurementCandidateRow(
            family="reactive",
            parameters=(("beta", 0.5),),
            training_manifest_hash=digest("r3-reactive-training"),
            selection_manifest_hash=digest("r3-reactive-selection"),
            candidate_hash="sha256:42a84ef10c207159ff98397fe2bd86594df6c8be0f41b98e876f7ae33cba9456",
        ),
        FeherHareMeasurementCandidateRow(
            family="intentional",
            parameters=(("beta", 2.0), ("memory_decay", 0.5)),
            training_manifest_hash=digest("r3-intentional-training"),
            selection_manifest_hash=digest("r3-intentional-selection"),
            candidate_hash="sha256:68a01b5496e20095c1a603b30f1384bbe6a3ea22aac8cf74b7b97463c4935839",
        ),
        FeherHareMeasurementCandidateRow(
            family="planning",
            parameters=(("beta", 4.0), ("memory_decay", 0.75)),
            training_manifest_hash=digest("r3-planning-training"),
            selection_manifest_hash=digest("r3-planning-selection"),
            candidate_hash="sha256:c0ed956c1b285153075e5a22e7fb51f6d53dcf8ad326cce82e39e0221c2e877c",
        ),
    )
    return FeherHareMeasurementAnchor(
        lock_commit=FEHER_HARE_R3_LOCK_COMMIT,
        scientific_repository_revision="d01232979cdfc9d902daab5f9e3e937079b56f69",
        upstream_revision="4567763780a2c596fd6510af720ec468a8214a8f",
        source_manifest_hash=digest("r3-source-manifest"),
        source_snapshot_hash=digest("r3-source-snapshot"),
        transform_hash=digest("r3-transform"),
        participant_assignment_hash=digest("r3-participant-assignment"),
        dataset_hash=digest("r3-dataset"),
        target_spec_hash=digest("r3-target-spec"),
        train_partition_hash=digest("r3-train-partition"),
        selection_partition_hash=digest("r3-selection-partition"),
        excluded_final_partition_hash=digest("r3-final-partition"),
        excluded_final_target_hash=digest("r3-final-target"),
        train_selection_freeze_hash=digest("r3-freeze"),
        internal_lock_bundle_hash=digest("r3-lock-bundle"),
        candidate_rows=candidates,
    )


def new_source_lineage() -> dict[str, str]:
    prepared, _backend = prepared_transfer()
    return {
        "source_identity_hash": prepared.source_identity_hash,
        "split_manifest_hash": prepared.split_manifest.content_hash,
        "train_projection_hash": prepared.train.projection_hash,
        "selection_projection_hash": prepared.selection_validation.projection_hash,
    }


def builder_identities() -> tuple[tuple[str, str], ...]:
    return tuple(
        (family.value, digest(f"{family.value}-builder"))
        for family in TransferFamily
    )


def complete_36_point_evaluations(
    *,
    with_tie: bool = False,
) -> tuple[GridCandidateEvaluation, ...]:
    prepared, _backend = prepared_transfer()
    builders = dict(builder_identities())
    grids = (
        (TransferFamily.REACTIVE, REACTIVE_GRID),
        (TransferFamily.INTENTIONAL, INTENTIONAL_GRID),
        (TransferFamily.PLANNING, PLANNING_GRID),
    )
    rows: list[GridCandidateEvaluation] = []
    for family, grid in grids:
        for index, parameters in enumerate(grid):
            loss = 0.1 + index / 1000.0
            if with_tie and index in {0, 1}:
                loss = 0.1
            rows.append(
                GridCandidateEvaluation(
                    family=family,
                    parameters=parameters,
                    train_projection_hash=prepared.train.projection_hash,
                    selection_projection_hash=prepared.selection_validation.projection_hash,
                    target_spec_hash=digest("transfer-target-spec"),
                    split_manifest_hash=prepared.split_manifest.content_hash,
                    train_seeds=(101, 102),
                    selection_seeds=(201, 202),
                    score="BRIER",
                    brier_loss_identity=digest("brier"),
                    builder_identity=builders[family.value],
                    simulation_identity=digest("simulation"),
                    selection_brier_loss=loss,
                    evaluation_receipt_hash=digest(
                        f"evaluation-{family.value}-{parameters!r}"
                    ),
                )
            )
    return tuple(rows)

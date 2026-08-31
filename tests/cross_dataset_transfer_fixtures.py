"""Synthetic-only fixtures for Cross-Dataset Transfer V1 Phase A tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

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
from narrative_dynamics.cross_dataset_inference import (
    _ParticipantLossBlock,
    fit_train_base_rate,
)
from narrative_dynamics.cross_dataset_ledger import (
    GitTransferAttemptStore,
    TransferAttemptEvent,
    TransferAttemptEventType,
)
from narrative_dynamics.cross_dataset_release import (
    DualTransferPreflight,
    TransferScore,
    TransferScoreRelease,
    build_transfer_protocol,
    preflight_transfer_releases,
)
from narrative_dynamics.cross_dataset_authorization import AUTHORIZATION_HEADER
from narrative_dynamics.cross_dataset_prediction import TransferRunProgress
from narrative_dynamics.cross_dataset_reporting import (
    CarryForwardSensitivityStatus,
)
from narrative_dynamics.studies.cross_dataset_transfer_locked_final import (
    TransferFinalInfrastructureError,
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
            training_manifest_hash="sha256:5b607a4bc0a8082809c2446a8248fb8659b0d9ba178687ddad96dc74e3f19622",
            selection_manifest_hash="sha256:e8414e301d19fc6acfd5accf055402ac995790f785402186c67ff95a48ee0c2e",
            candidate_hash="sha256:42a84ef10c207159ff98397fe2bd86594df6c8be0f41b98e876f7ae33cba9456",
        ),
        FeherHareMeasurementCandidateRow(
            family="intentional",
            parameters=(("beta", 2.0), ("memory_decay", 0.5)),
            training_manifest_hash="sha256:47057311fe7450469c1710be49ba0e3b42aa7416fa2129f40761dbf8d237d4fc",
            selection_manifest_hash="sha256:498a512cd931afe276f78bf135bd5c8269e051920d4b876177d0aa9ea250806d",
            candidate_hash="sha256:68a01b5496e20095c1a603b30f1384bbe6a3ea22aac8cf74b7b97463c4935839",
        ),
        FeherHareMeasurementCandidateRow(
            family="planning",
            parameters=(("beta", 4.0), ("memory_decay", 0.75)),
            training_manifest_hash="sha256:ef0f3caf4f2a02b2d719bc302d7409fc8c2d937eb13517d906d83b38c29b76ac",
            selection_manifest_hash="sha256:e347a3d7d523e2a35d0bb6b0f662343f5a01ddd1ff7d2efe95a8a5efa1986e74",
            candidate_hash="sha256:c0ed956c1b285153075e5a22e7fb51f6d53dcf8ad326cce82e39e0221c2e877c",
        ),
    )
    return FeherHareMeasurementAnchor(
        lock_commit=FEHER_HARE_R3_LOCK_COMMIT,
        scientific_repository_revision="d01232979cdfc9d902daab5f9e3e937079b56f69",
        upstream_revision="4567763780a2c596fd6510af720ec468a8214a8f",
        source_manifest_hash="sha256:3609e980af172823cfef78290e7f2337fdb337d67f1f27641e17d473aeedd10d",
        source_snapshot_hash="sha256:25bdc2e4bff4110f38b098b89e7d59aa38c9658727fcc89a38d50e107d243186",
        transform_hash="sha256:2fb8ab6dc796a9e4ece5653a5485d865ef44dd0f5f7dff92d94a904932d6e541",
        participant_assignment_hash="sha256:fc144c35f6713e27f75141d678872cc04ca44c7a0fd8e109e8618c72b4111ccf",
        dataset_hash="sha256:17789130372d7eace05e1216a57bdae2ffbd519960333ffe814aee2d2d404781",
        target_spec_hash="sha256:3134c9dc424418c87379f8e451420fb2defe81b8f30a45d67cd9b1f453349713",
        train_partition_hash="sha256:71ce56243338eb23b9dd5ad6dd901e4b0d0be4989742b66bbe7ea4f025f207da",
        selection_partition_hash="sha256:ecdcfa1681b58888ebf4419372d5de7b75be9624cdd3307144371c5486eb1347",
        excluded_final_partition_hash="sha256:936ffe872644e888111007b300ea847484698be8fbfd7c2d54a177505a411047",
        excluded_final_target_hash="sha256:927e1727d37993e9a5c887f79ac8712d07a155622deacf0a3122d41f90b16ed2",
        train_selection_freeze_hash="sha256:77fa1bb80ab0c7ac737a09a8001f1d0c1432ce3bd991fd7191d07fdd0d8e05d5",
        internal_lock_bundle_hash="sha256:96e553e557d6e7314eb0b9b1d0aaa8696e01d7a6ddec0abde949be8c8d45602f",
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


def train_projection(
    actions: tuple[str, ...] = ("action_1", "action_1", "action_0"),
):
    if len(actions) != 3:
        raise ValueError("synthetic TRAIN fixture requires exactly three actions")
    participants = synthetic_stratified_participants((5,))
    secret = RestrictedStudySecret.from_bytes(bytes(range(32)))
    role_index, split_manifest = assign_participant_roles(
        namespace="cross-dataset-transfer-v1-inference-fixture",
        source_snapshot_hash=digest("inference-snapshot"),
        participants=participants,
        secret=secret,
        transform_attestation_hash=digest("inference-transform"),
    )
    action_iter = iter(actions)
    trials = tuple(
        CanonicalTransferTrial(
            participant_key=participant.participant_id,
            trial_id=1,
            source_stratum=participant.source_stratum,
            first_stage_action=(
                next(action_iter)
                if role_index.role_for(
                    participant.participant_id,
                    participant.source_stratum,
                ).value
                == "TRAIN"
                else "action_0"
            ),
            transition_common=True,
            final_state="state_0",
            second_stage_action="second_0",
            reward=0,
            row_commitment=digest(f"inference-{participant.participant_id}"),
        )
        for participant in participants
    )
    prepared, _backend = provision_transfer_capabilities(
        source_identity_hash=digest("inference-source"),
        trials=trials,
        role_index=role_index,
        split_manifest=split_manifest,
    )
    return prepared.train


def participant_blocks(
    means: tuple[float, ...],
) -> tuple[_ParticipantLossBlock, ...]:
    return tuple(
        _ParticipantLossBlock(
            token=digest(f"private-loss-token-{index}"),
            losses=(value,),
        )
        for index, value in enumerate(means)
    )


def local_git_attempt_store(root: Path) -> GitTransferAttemptStore:
    root.mkdir(parents=True, exist_ok=True)
    remote = root / "attempt-ledger.git"
    if not remote.exists():
        subprocess.run(
            ("git", "init", "--bare", str(remote)),
            check=True,
            capture_output=True,
            text=True,
        )
    client_index = len(tuple(root.glob("ledger-client-*")))
    return GitTransferAttemptStore(
        remote=remote,
        workspace=root / f"ledger-client-{client_index}",
    )


def two_git_store_clients(
    root: Path,
) -> tuple[GitTransferAttemptStore, GitTransferAttemptStore]:
    first = local_git_attempt_store(root)
    second = local_git_attempt_store(root)
    return first, second


def attempt_event(
    event_type: TransferAttemptEventType,
    *,
    parent: str,
    **details: object,
) -> TransferAttemptEvent:
    return TransferAttemptEvent.create(
        event_type=event_type,
        parent_ledger_head=parent,
        scientific_revision="a" * 40,
        attempt_id="attempt-synthetic-v1",
        timestamp_utc="2026-08-30T12:00:00Z",
        repository_receipt_hash=digest("repository-receipt"),
        run_receipt_hash=digest("run-receipt"),
        job_receipt_hash=digest("job-receipt"),
        details=details,
    )


def final_started_event(*, parent: str) -> TransferAttemptEvent:
    return attempt_event(
        TransferAttemptEventType.FINAL_STARTED,
        parent=parent,
        authorization_receipt_hash=digest("authorization-receipt"),
    )


def failure_event(*, parent: str) -> TransferAttemptEvent:
    return attempt_event(
        TransferAttemptEventType.REVISION_REQUIRED,
        parent=parent,
        failure_class="SCHEMA",
    )


def sibling_releases() -> tuple[TransferScoreRelease, TransferScoreRelease]:
    shared: dict[str, object] = {
        "scientific_revision": "a" * 40,
        "source_identity_hash": digest("release-source"),
        "transform_identity_hash": digest("release-transform"),
        "split_manifest_hash": digest("release-split"),
        "candidate_hashes": tuple(
            digest(f"release-candidate-{index}") for index in range(6)
        ),
        "baseline_hash": digest("release-baseline"),
        "final_commitment_hash": digest("release-final-commitment"),
        "prediction_artifact_identity": digest("prediction-artifact-schema"),
    }
    brier_protocol = build_transfer_protocol(
        score=TransferScore.BRIER,
        score_identity=digest("brier-score"),
        **shared,
    )
    log_protocol = build_transfer_protocol(
        score=TransferScore.LOG,
        score_identity=digest("log-score"),
        **shared,
    )
    return (
        TransferScoreRelease.create(
            brier_protocol,
            release_receipt_hash=digest("brier-release-receipt"),
        ),
        TransferScoreRelease.create(
            log_protocol,
            release_receipt_hash=digest("log-release-receipt"),
        ),
    )


def dual_preflight(
    root: Path,
) -> tuple[DualTransferPreflight, GitTransferAttemptStore]:
    store = local_git_attempt_store(root)
    preflight = preflight_transfer_releases(
        *sibling_releases(),
        store=store,
        completed_at_utc="2026-08-30T13:00:00Z",
    )
    return preflight, store


def authorization_comment(
    preflight: DualTransferPreflight,
    *,
    lock_commit: str = "b" * 40,
    **overrides: object,
) -> dict[str, object]:
    body = "\n".join(
        (
            AUTHORIZATION_HEADER,
            f"scientific_sha={preflight.scientific_revision}",
            f"lock_commit={lock_commit}",
            f"preflight_hash={preflight.content_hash}",
            f"brier_release_hash={preflight.brier_release_hash}",
            f"log_release_hash={preflight.log_release_hash}",
        )
    )
    values: dict[str, object] = {
        "comment_id": 123456789,
        "html_url": "https://github.com/qigao/lean/issues/43#issuecomment-123456789",
        "author_login": "qigao",
        "created_at": "2026-08-30T14:00:00Z",
        "updated_at": "2026-08-30T14:00:00Z",
        "body": body,
        "deleted": False,
    }
    values.update(overrides)
    return values


def zero_shot_freeze():
    from narrative_dynamics.cross_dataset_candidates import freeze_zero_shot_candidates

    return freeze_zero_shot_candidates(r3_anchor(), new_source_lineage())


def refit_freeze():
    from narrative_dynamics.cross_dataset_candidates import freeze_refit_candidates

    prepared, _backend = prepared_transfer()
    return freeze_refit_candidates(
        train=prepared.train,
        selection=prepared.selection_validation,
        evaluations=complete_36_point_evaluations(),
        brier_loss_identity=digest("brier"),
        builder_identities=builder_identities(),
        simulation_identity=digest("simulation"),
        tie_break_identity="lexical_parameters_v1",
        target_spec_hash=digest("transfer-target-spec"),
        split_manifest_hash=prepared.split_manifest.content_hash,
    )


def final_worker_projection(case_count: int = 4):
    from narrative_dynamics.cross_dataset_capabilities import FinalWorkerProjection

    return FinalWorkerProjection(
        rows=synthetic_transfer_trials()[:case_count],
        commitment_hash=digest(f"synthetic-final-commitment-{case_count}"),
    )


class RecordingTransferEvaluator:
    def __init__(self, calls: list[tuple[object, ...]], *, invalid=None) -> None:
        self.calls = calls
        self.invalid = invalid
        self.inputs: list[object] = []

    def __call__(self, model_input, candidate, seed):
        self.inputs.append(model_input)
        key = (
            model_input.case_token,
            candidate.path,
            candidate.family.value,
            candidate.parameters,
            seed,
        )
        self.calls.append(key)
        if isinstance(self.invalid, BaseException):
            raise self.invalid
        if self.invalid is not None:
            return self.invalid
        return (0.4, 0.6)


def recording_evaluator(calls: list[tuple[object, ...]], *, invalid=None):
    return RecordingTransferEvaluator(calls, invalid=invalid)


class RecordingProgress:
    def __init__(self, calls: list[tuple[object, ...]]) -> None:
        self.calls = calls
        self.rows: list[TransferRunProgress] = []

    def __call__(self, row: TransferRunProgress) -> None:
        self.rows.append(row)
        if row.completed_model_runs != len(self.calls):
            raise AssertionError("progress was not persisted before the next run")


def recording_progress(calls: list[tuple[object, ...]]) -> RecordingProgress:
    return RecordingProgress(calls)


def sealed_prediction_artifact(
    probabilities: tuple[float, float] = (0.4, 0.6),
):
    from narrative_dynamics.cross_dataset_prediction import execute_transfer_final_predictions

    calls: list[tuple[object, ...]] = []
    projection = final_worker_projection(4)
    return execute_transfer_final_predictions(
        projection=projection,
        zero_shot=zero_shot_freeze(),
        refit=refit_freeze(),
        seeds=(301, 302),
        evaluator=recording_evaluator(calls, invalid=probabilities),
        progress=recording_progress(calls),
        expected_final_commitment_hash=projection.commitment_hash,
        prediction_artifact_identity=sibling_releases()[0].prediction_artifact_identity,
    )


def carry_forward_statuses(
    status: CarryForwardSensitivityStatus = CarryForwardSensitivityStatus.COMPARABLE,
):
    from narrative_dynamics.cross_dataset_release import CARRY_FORWARD_REQUIREMENT_IDS

    return tuple((requirement_id, status) for requirement_id in CARRY_FORWARD_REQUIREMENT_IDS)


def valid_negative_scoring_input() -> dict[str, object]:
    artifact = sealed_prediction_artifact((0.4, 0.6))
    baseline = fit_train_base_rate(train_projection())
    original_brier, original_log = sibling_releases()
    brier = TransferScoreRelease.create(
        replace(
            original_brier.protocol,
            baseline_hash=baseline.content_hash,
            final_commitment_hash=artifact.final_commitment_hash,
        ),
        release_receipt_hash=original_brier.release_receipt_hash,
    )
    log = TransferScoreRelease.create(
        replace(
            original_log.protocol,
            baseline_hash=baseline.content_hash,
            final_commitment_hash=artifact.final_commitment_hash,
        ),
        release_receipt_hash=original_log.release_receipt_hash,
    )
    return {
        "artifact": artifact,
        "baseline": baseline,
        "brier_release": brier,
        "log_release": log,
        "semantic_invariance_receipt_hash": digest("semantic-invariance"),
        "semantic_invariance_pass": True,
        "carry_forward_statuses": carry_forward_statuses(),
        "task_condition_strata": ("s0", "synthetic-condition"),
        "participant_influence_hash": digest("participant-influence"),
        "forbidden_archive_values": ("private-p1", "private-p2"),
    }


class RecordingEvidenceSink:
    def __init__(self, order: list[str], *, fail: bool = False) -> None:
        self.order = order
        self.fail = fail
        self.snapshots: list[dict[str, object]] = []

    def persist_aggregate_state(self, history) -> None:
        self.order.append("evidence:persist")
        if self.fail:
            self.fail = False
            raise RuntimeError("synthetic evidence persistence failure")
        self.snapshots.append(
            {
                "final_started_count": history.final_started_count,
                "final_projection_openings": history.final_projection_openings,
                "completed_model_runs": history.completed_model_runs,
                "prediction_artifact_hashes": history.prediction_artifact_hashes,
                "score_artifact_hashes": history.score_artifact_hashes,
                "terminal_count": history.terminal_count,
            }
        )


def locked_final_inputs(
    root: Path,
    *,
    order: list[str] | None = None,
    fail_at: str | None = None,
    infrastructure_failure: bool = True,
    evidence_failure: bool = False,
) -> dict[str, object]:
    selected_order = [] if order is None else order
    participants = synthetic_stratified_participants((5,))
    secret = RestrictedStudySecret.from_bytes(bytes(range(32)))
    role_index, split_manifest = assign_participant_roles(
        namespace="locked-final-fixture",
        source_snapshot_hash=digest("locked-snapshot"),
        participants=participants,
        secret=secret,
        transform_attestation_hash=digest("locked-transform"),
    )
    trials = tuple(
        CanonicalTransferTrial(
            participant_key=participant.participant_id,
            trial_id=1,
            source_stratum=participant.source_stratum,
            first_stage_action="action_1" if index % 2 else "action_0",
            transition_common=True,
            final_state="state_0",
            second_stage_action="second_0",
            reward=index % 2,
            row_commitment=digest(f"locked-row-{index}"),
        )
        for index, participant in enumerate(participants)
    )
    prepared, backend = provision_transfer_capabilities(
        source_identity_hash=digest("locked-source"),
        trials=trials,
        role_index=role_index,
        split_manifest=split_manifest,
    )
    zero = zero_shot_freeze()
    refit = refit_freeze()
    baseline = fit_train_base_rate(prepared.train)
    candidate_hashes = tuple(
        row.content_hash for row in zero.candidates + refit.selected_candidates
    )
    shared: dict[str, object] = {
        "scientific_revision": "a" * 40,
        "source_identity_hash": prepared.source_identity_hash,
        "transform_identity_hash": digest("locked-transform"),
        "split_manifest_hash": prepared.split_manifest.content_hash,
        "candidate_hashes": candidate_hashes,
        "baseline_hash": baseline.content_hash,
        "final_commitment_hash": prepared.final_commitment.content_hash,
        "prediction_artifact_identity": digest("prediction-artifact-schema"),
    }
    brier = TransferScoreRelease.create(
        build_transfer_protocol(
            score=TransferScore.BRIER,
            score_identity=digest("locked-brier-score"),
            **shared,
        ),
        release_receipt_hash=digest("locked-brier-release"),
    )
    log = TransferScoreRelease.create(
        build_transfer_protocol(
            score=TransferScore.LOG,
            score_identity=digest("locked-log-score"),
            **shared,
        ),
        release_receipt_hash=digest("locked-log-release"),
    )
    store = local_git_attempt_store(root)
    preflight = preflight_transfer_releases(
        brier,
        log,
        store=store,
        completed_at_utc="2026-08-30T13:00:00Z",
    )
    from narrative_dynamics.cross_dataset_authorization import parse_transfer_authorization

    authorization = parse_transfer_authorization(
        issue_number=43,
        comment=authorization_comment(preflight),
        authorized_owner="qigao",
        preflight=preflight,
        lock_commit="b" * 40,
        store=store,
    )
    backend.bind_unlock_policy(
        prepared.final_vault_handle,
        FinalUnlockGrant(
            scientific_revision=preflight.scientific_revision,
            ledger_head_hash=preflight.ledger_head_hash,
            preflight_hash=preflight.content_hash,
            authorization_receipt_hash=authorization.content_hash,
            brier_release_hash=preflight.brier_release_hash,
            log_release_hash=preflight.log_release_hash,
            lock_commit=authorization.lock_commit,
        ),
    )
    calls: list[tuple[object, ...]] = []

    def stage_hook(stage: str) -> None:
        selected_order.append(stage)
        if stage == fail_at:
            if infrastructure_failure:
                raise TransferFinalInfrastructureError(f"synthetic failure at {stage}")
            raise ValueError(f"synthetic schema failure at {stage}")

    return {
        "store": store,
        "preflight": preflight,
        "authorization": authorization,
        "vault_backend": backend,
        "vault_handle": prepared.final_vault_handle,
        "zero_shot": zero,
        "refit": refit,
        "baseline": baseline,
        "brier_release": brier,
        "log_release": log,
        "evaluator": recording_evaluator(calls),
        "semantic_invariance_receipt_hash": digest("locked-semantic"),
        "carry_forward_statuses": carry_forward_statuses(),
        "task_condition_strata": ("s0", "synthetic-condition"),
        "participant_influence_hash": digest("locked-influence"),
        "evidence_sink": RecordingEvidenceSink(
            selected_order,
            fail=evidence_failure,
        ),
        "stage_hook": stage_hook,
    }

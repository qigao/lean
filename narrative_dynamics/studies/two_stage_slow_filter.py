from __future__ import annotations

import hashlib
from pathlib import Path

from . import two_stage_transform as _base
from .two_stage_source import TwoStageSourceManifest, VerifiedTwoStageSnapshot


def _implementation_identity() -> dict[str, object]:
    identity = dict(_base._implementation_identity())
    base_bytes = Path(_base.__file__).read_bytes()
    correction_bytes = Path(__file__).read_bytes()
    identity["implementation_sha256"] = hashlib.sha256(
        base_bytes + b"\0" + correction_bytes
    ).hexdigest()
    return identity


def transform_two_stage_snapshot(
    snapshot: VerifiedTwoStageSnapshot,
    manifest: TwoStageSourceManifest,
) -> _base.TwoStageTransformReport:
    """Transform retained trials after applying the frozen source slow rule.

    Feher/Hare source rows with ``slow == 1`` may contain ``-1`` sentinels for
    responses and reaction times that never occurred.  The preregistered rule
    excludes those rows before target/history construction, so only ``trial``
    ordering and the binary ``slow`` flag are required before exclusion.
    Retained rows still pass through the original strict task-specific parser.
    """
    if not isinstance(snapshot, VerifiedTwoStageSnapshot):
        raise TypeError("two-stage transform requires a verified source snapshot")
    if not isinstance(manifest, TwoStageSourceManifest):
        raise TypeError("two-stage transform requires TwoStageSourceManifest")
    if snapshot.manifest_hash != manifest.content_hash:
        raise ValueError("source snapshot and manifest identity differ")

    root = Path(snapshot.root)
    verified = _base.verify_two_stage_snapshot(root, manifest)
    if verified.source_snapshot_hash != snapshot.source_snapshot_hash:
        raise ValueError("source snapshot identity changed before transform")

    files = {item.path: item for item in manifest.files}
    retained: list[_base.CanonicalTwoStageTrial] = []
    eligible: list[tuple[str, str]] = []
    exclusions: list[tuple[str, str, str]] = []
    slow_counts: list[tuple[str, str, int]] = []
    retained_counts: list[tuple[str, str, int]] = []

    evidence_files = tuple(
        item for item in manifest.files if item.purpose == "scientific_evidence"
    )
    for source in evidence_files:
        participant = source.source_participant_id
        assert participant is not None
        source_path = root / source.path

        if source.task_variant == "magic_carpet":
            _base._validate_magic_config(root, participant, files)
            rows = _base._read_rows(source_path, _base._MAGIC_HEADER)
            parser = _base._parse_magic_trial
        else:
            _base._validate_spaceship_info(root, participant, files)
            rows = _base._read_rows(source_path, _base._SPACESHIP_HEADER)
            parser = _base._parse_spaceship_trial

        _base._validate_trial_order(rows, path=source_path)
        participant_retained: list[_base.CanonicalTwoStageTrial] = []
        slow_count = 0
        for row in rows:
            slow = _base._binary(row["slow"], label=f"{source.path}: slow")
            if slow == 1:
                slow_count += 1
                continue
            trial, parsed_slow = parser(
                row,
                participant=participant,
                path=source.path,
            )
            if parsed_slow != 0:
                raise RuntimeError("retained two-stage parser returned a slow trial")
            participant_retained.append(trial)

        slow_counts.append((source.task_variant, participant, slow_count))
        retained_counts.append(
            (source.task_variant, participant, len(participant_retained))
        )
        if not participant_retained:
            exclusions.append(
                (source.task_variant, participant, "no_retained_scorable_trials")
            )
            continue
        eligible.append((source.task_variant, participant))
        retained.extend(participant_retained)

    if not retained:
        raise ValueError("two-stage transform produced no retained scorable trials")

    return _base.TwoStageTransformReport(
        source_manifest_hash=manifest.content_hash,
        source_snapshot_hash=snapshot.source_snapshot_hash,
        repository_revision=snapshot.repository_revision,
        transform_identity=_implementation_identity(),
        trials=tuple(retained),
        eligible_participants=tuple(eligible),
        structural_exclusions=tuple(exclusions),
        slow_counts=tuple(slow_counts),
        retained_counts=tuple(retained_counts),
    )


__all__ = ["transform_two_stage_snapshot"]

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


def _parse_retained_spaceship_trial(
    row: dict[str, str],
    *,
    participant: str,
    path: str,
) -> tuple[_base.CanonicalTwoStageTrial, int]:
    """Parse one retained Spaceship row using the frozen upstream encoding.

    Upstream ``symbol0`` and ``symbol1`` are independent binary indices for
    spaceship and planet presentation, so all four 00/01/10/11 combinations
    are valid.  The first-stage target remains the upstream relative-choice
    recoding from transition type and final state.
    """
    trial_id = _base._int_text(row["trial"], label=f"{path}: trial")
    common = _base._binary(row["common"], label=f"{path}: common")
    reward = _base._binary(row["reward"], label=f"{path}: reward")
    slow = _base._binary(row["slow"], label=f"{path}: slow")
    latent = tuple(
        _base._probability(row[name], label=f"{path}: {name}")
        for name in ("rwrd_prob0", "rwrd_prob1", "rwrd_prob2", "rwrd_prob3")
    )
    symbol0 = _base._binary(row["symbol0"], label=f"{path}: symbol0")
    symbol1 = _base._binary(row["symbol1"], label=f"{path}: symbol1")
    _base._binary(row["choice1"], label=f"{path}: choice1")
    _base._float_text(row["rt1"], label=f"{path}: rt1", minimum=0.0)
    _base._float_text(row["rt2"], label=f"{path}: rt2", minimum=0.0)
    final_state = _base._binary(row["final_state"], label=f"{path}: final_state")
    choice2 = _base._binary(row["choice2"], label=f"{path}: choice2")
    relative = final_state + 1 if common else 2 - final_state
    return (
        _base.CanonicalTwoStageTrial(
            pre_choice=_base.TwoStagePreChoiceView(
                task_variant="spaceship",
                source_participant_id=participant,
                trial_id=trial_id,
                first_stage_configuration=(("symbol0", symbol0), ("symbol1", symbol1)),
            ),
            outcome=_base.TwoStageObservedOutcome(
                first_stage_action=_base._canonical_action_from_one_two(
                    relative,
                    label="spaceship relative choice",
                ),
                transition_common=bool(common),
                final_state=_base._canonical_state_from_zero_one(final_state),
                second_stage_action=_base._canonical_action_from_zero_one(
                    choice2,
                    label="spaceship choice2",
                ),
                reward=reward,
            ),
            source_path=path,
            audit_latent_reward_probabilities=latent,
        ),
        slow,
    )


def transform_two_stage_snapshot(
    snapshot: VerifiedTwoStageSnapshot,
    manifest: TwoStageSourceManifest,
) -> _base.TwoStageTransformReport:
    """Transform retained trials after applying the frozen source slow rule.

    Feher/Hare source rows with ``slow == 1`` may contain ``-1`` sentinels for
    responses and reaction times that never occurred.  The preregistered rule
    excludes those rows before target/history construction, so only ``trial``
    ordering and the binary ``slow`` flag are required before exclusion.
    Retained rows still pass through strict task-specific validation.
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
            parser = _parse_retained_spaceship_trial

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

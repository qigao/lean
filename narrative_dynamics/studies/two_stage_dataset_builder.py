from __future__ import annotations

from . import two_stage_transform as _base


def build_two_stage_observation_dataset(
    report: _base.TwoStageTransformReport,
    assignment: _base.TwoStageParticipantAssignment,
) -> _base.ObservationDataset:
    """Build the canonical dataset without repeatedly hashing full lineage.

    The dataset payload is intentionally identical to the original builder. The
    only change is that lineage hashes are computed once before record
    construction and then reused for every record and provenance binding.
    """
    if not isinstance(report, _base.TwoStageTransformReport):
        raise TypeError("dataset construction requires TwoStageTransformReport")
    if not isinstance(assignment, _base.TwoStageParticipantAssignment):
        raise TypeError("dataset construction requires TwoStageParticipantAssignment")

    transform_report_hash = report.content_hash
    participant_assignment_hash = assignment.content_hash

    role_by_participant = {
        (task, participant): role for task, participant, role in assignment.assignments
    }
    if set(role_by_participant) != set(report.eligible_participants):
        raise ValueError(
            "participant assignment must cover every eligible participant exactly once"
        )

    grouped: dict[tuple[str, str], list[_base.CanonicalTwoStageTrial]] = {}
    for trial in report.trials:
        key = (
            trial.pre_choice.task_variant,
            trial.pre_choice.source_participant_id,
        )
        grouped.setdefault(key, []).append(trial)

    records_by_role: dict[str, list[_base.ObservationRecord]] = {
        role: [] for role in _base._ROLE_ORDER
    }
    for key in sorted(grouped):
        task, participant = key
        trials = sorted(grouped[key], key=lambda item: item.pre_choice.trial_id)
        history: list[dict[str, object]] = []
        for trial in trials:
            history_tuple = tuple(history)
            history_hash = _base.stable_content_hash(history_tuple)
            scenario_identity = _base.stable_content_hash(
                {
                    "task_variant": task,
                    "participant_source_identity": _base.stable_content_hash(
                        (task, participant)
                    ),
                    "trial_id": trial.pre_choice.trial_id,
                    "pre_choice": trial.pre_choice.identity_payload(),
                    "causal_history_hash": history_hash,
                }
            )
            scenario = _base.Scenario(
                id=f"two-stage:{scenario_identity.removeprefix('sha256:')}",
                payload={
                    "task_variant": task,
                    "first_stage_configuration": trial.pre_choice.first_stage_configuration,
                    "history": history_tuple,
                },
            )
            action = trial.outcome.first_stage_action
            counts = {
                "action_0": 1 if action == "action_0" else 0,
                "action_1": 1 if action == "action_1" else 0,
            }
            record_id = f"{task}/{participant}/{trial.pre_choice.trial_id}"
            role = role_by_participant[key]
            records_by_role[role].append(
                _base.ObservationRecord(
                    id=record_id,
                    scenario=scenario,
                    counts=counts,
                    metadata={
                        "task_variant": task,
                        "source_participant_id": participant,
                        "source_trial_id": trial.pre_choice.trial_id,
                        "source_path": trial.source_path,
                        "causal_history_hash": history_hash,
                        "participant_assignment_hash": participant_assignment_hash,
                        "transform_report_hash": transform_report_hash,
                    },
                )
            )
            history.append(_base._history_payload(trial))

    if any(not records_by_role[role] for role in _base._ROLE_ORDER):
        raise ValueError(
            "two-stage dataset requires non-empty train, selection, and final partitions"
        )

    transform_identity = {
        **dict(report.transform_identity),
        "participant_assignment_hash": participant_assignment_hash,
        "source_manifest_hash": report.source_manifest_hash,
        "source_snapshot_hash": report.source_snapshot_hash,
    }
    return _base.ObservationDataset(
        name="feher-hare-two-stage-v1",
        version="1",
        source={
            "kind": "external_observational",
            "source_snapshot_hash": report.source_snapshot_hash,
            "repository_revision": report.repository_revision,
            "source_manifest_hash": report.source_manifest_hash,
        },
        provenance={
            "external_observational": True,
            "transform_identity": transform_identity,
            "participant_assignment_hash": participant_assignment_hash,
            "transform_report_hash": transform_report_hash,
        },
        partitions=tuple(
            _base.ObservationPartition(
                name=role,
                role=_base._ROLE_ENUM[role],
                records=tuple(records_by_role[role]),
            )
            for role in _base._ROLE_ORDER
        ),
    )


__all__ = ["build_two_stage_observation_dataset"]

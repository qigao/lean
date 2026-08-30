from __future__ import annotations

from collections.abc import Mapping

from narrative_dynamics.contracts import Scenario, stable_content_hash
from narrative_dynamics.measurement_validity import (
    MEASUREMENT_CLAIM_SCOPE,
    MeasurementAggregation,
    MeasurementAuditCase,
    MeasurementAuditInput,
    MeasurementScore,
    MeasurementValidityProtocol,
)
from narrative_dynamics.observations import ObservationPartitionRole
from narrative_dynamics.observations.preregistration import FrozenModelSpec


def digest(character: str) -> str:
    if len(character) != 1 or character not in "0123456789abcdef":
        raise ValueError("fixture digest character must be hexadecimal")
    return f"sha256:{character * 64}"


class FixtureModel:
    def __init__(self, family: str) -> None:
        self.family = family

    def manifest_identity(self) -> Mapping[str, object]:
        return {
            "name": f"fixture-{self.family}",
            "version": "1",
            "family": self.family,
        }


def frozen_candidates() -> tuple[FrozenModelSpec, ...]:
    rows = (
        ("reactive", {"beta": 0.5}, digest("1")),
        (
            "intentional",
            {"beta": 2.0, "memory_decay": 0.5},
            digest("2"),
        ),
        (
            "planning",
            {"beta": 4.0, "memory_decay": 0.75},
            digest("3"),
        ),
    )
    return tuple(
        FrozenModelSpec.freeze(
            name=family,
            model=FixtureModel(family),
            parameters=parameters,
            selection_manifest_hash=selection_hash,
        )
        for family, parameters, selection_hash in rows
    )


def measurement_cases() -> tuple[MeasurementAuditCase, ...]:
    rows = (
        (
            ObservationPartitionRole.TRAIN,
            "magic_carpet",
            digest("4"),
            1,
            "action_0",
            digest("8"),
        ),
        (
            ObservationPartitionRole.TRAIN,
            "spaceship",
            digest("5"),
            1,
            "action_1",
            digest("9"),
        ),
        (
            ObservationPartitionRole.SELECTION_VALIDATION,
            "magic_carpet",
            digest("6"),
            2,
            "action_1",
            digest("a"),
        ),
        (
            ObservationPartitionRole.SELECTION_VALIDATION,
            "spaceship",
            digest("7"),
            2,
            "action_0",
            digest("b"),
        ),
    )
    cases: list[MeasurementAuditCase] = []
    for index, (role, task, participant_hash, trial_index, action, record_hash) in enumerate(rows):
        other_action = "action_1" if action == "action_0" else "action_0"
        cases.append(
            MeasurementAuditCase(
                role=role,
                scenario=Scenario(
                    id=f"measurement-fixture-{index}",
                    payload={
                        "task_variant": task,
                        "first_stage_configuration": (
                            ("action_0_position", "left"),
                            ("action_1_position", "right"),
                        ),
                        "history": (
                            {
                                "trial_id": trial_index - 1,
                                "first_stage_action": other_action,
                                "transition_common": bool(index % 2),
                                "final_state": f"state_{index % 2}",
                                "second_stage_action": action,
                                "reward": index % 2,
                            },
                        ),
                    },
                ),
                target=(
                    ("first_stage.action_0", 1.0 if action == "action_0" else 0.0),
                    ("first_stage.action_1", 1.0 if action == "action_1" else 0.0),
                ),
                task_variant=task,
                participant_group_hash=participant_hash,
                trial_index=trial_index,
                record_hash=record_hash,
            )
        )
    return tuple(cases)


def measurement_input(
    *,
    cases: tuple[MeasurementAuditCase, ...] | None = None,
    candidates: tuple[FrozenModelSpec, ...] | None = None,
) -> MeasurementAuditInput:
    selected_cases = measurement_cases() if cases is None else cases
    commitments = tuple(
        (
            role,
            stable_content_hash(
                tuple(
                    case.record_hash
                    for case in sorted(
                        (item for item in selected_cases if item.role is role),
                        key=lambda item: item.case_hash,
                    )
                )
            ),
        )
        for role in (
            ObservationPartitionRole.TRAIN,
            ObservationPartitionRole.SELECTION_VALIDATION,
        )
    )
    return MeasurementAuditInput(
        claim_scope=MEASUREMENT_CLAIM_SCOPE,
        source_manifest_hash=digest("c"),
        source_snapshot_hash=digest("d"),
        transform_hash=digest("e"),
        participant_assignment_hash=digest("f"),
        dataset_hash=digest("0"),
        target_spec_hash=digest("1"),
        allowed_partition_hashes=(
            (ObservationPartitionRole.TRAIN, digest("2")),
            (ObservationPartitionRole.SELECTION_VALIDATION, digest("3")),
        ),
        allowed_target_report_hashes=(
            (ObservationPartitionRole.TRAIN, digest("4")),
            (ObservationPartitionRole.SELECTION_VALIDATION, digest("5")),
        ),
        allowed_case_commitments=commitments,
        frozen_candidates=(
            frozen_candidates() if candidates is None else candidates
        ),
        excluded_final_partition_hash=digest("6"),
        excluded_final_target_hash=digest("7"),
        cases=selected_cases,
    )


def measurement_protocol(
    audit_input: MeasurementAuditInput | None = None,
    *,
    implementation_identities: tuple[tuple[str, Mapping[str, object]], ...] | None = None,
) -> MeasurementValidityProtocol:
    selected_input = measurement_input() if audit_input is None else audit_input
    return MeasurementValidityProtocol(
        name="feher-hare-measurement-validity-v1",
        version="1",
        claim_scope=MEASUREMENT_CLAIM_SCOPE,
        empirical_anchor_hash=selected_input.empirical_anchor_hash,
        allowed_roles=(
            ObservationPartitionRole.TRAIN,
            ObservationPartitionRole.SELECTION_VALIDATION,
        ),
        candidate_hashes=tuple(
            candidate.content_hash for candidate in selected_input.frozen_candidates
        ),
        seeds_by_role=(
            (ObservationPartitionRole.TRAIN, (101, 102)),
            (ObservationPartitionRole.SELECTION_VALIDATION, (201, 202)),
        ),
        semantic_fixture_hashes=(digest("8"), digest("9")),
        semantic_permutations=(
            "binary_coordinate_swap",
            "magic_carpet_counterbalance",
            "spaceship_symbol_order",
        ),
        task_strata=("magic_carpet", "spaceship"),
        aggregation_rules=(
            MeasurementAggregation.TRIAL_EQUAL,
            MeasurementAggregation.PARTICIPANT_EQUAL,
        ),
        scores=(MeasurementScore.BRIER, MeasurementScore.LOG),
        material_reversal_references=(
            (MeasurementScore.BRIER, 0.005),
            (MeasurementScore.LOG, 0.006931471805599453),
        ),
        participant_influence_rule="deterministic_leave_one_participant_out",
        stay_switch_definition="reward_by_transition_previous_retained_trial",
        excluded_final_partition_hash=selected_input.excluded_final_partition_hash,
        excluded_final_target_hash=selected_input.excluded_final_target_hash,
        implementation_identities=(
            (
                (
                    "project",
                    {"name": "fixture-project", "version": "1"},
                ),
                (
                    "score",
                    {"name": "fixture-score", "version": "1"},
                ),
            )
            if implementation_identities is None
            else implementation_identities
        ),
    )


__all__ = [
    "FixtureModel",
    "digest",
    "frozen_candidates",
    "measurement_cases",
    "measurement_input",
    "measurement_protocol",
]

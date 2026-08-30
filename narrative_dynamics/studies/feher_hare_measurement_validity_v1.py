from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.measurement_validity import (
    MeasurementAuditInput,
    MeasurementPredictionArtifact,
    MeasurementValidityStatus,
    average_seed_metrics,
)
from narrative_dynamics.observations.dataset import ObservationPartitionRole


_TASKS = ("magic_carpet", "spaceship")
_MODELS = ("reactive", "intentional", "planning")
_ACTIONS = ("action_0", "action_1")
_CELL_ORDER = (
    (0, False),
    (0, True),
    (1, False),
    (1, True),
)
_SEEDS_BY_ROLE = {
    ObservationPartitionRole.TRAIN: (101, 102),
    ObservationPartitionRole.SELECTION_VALIDATION: (201, 202),
}
_DIAGNOSTIC_STATUSES = frozenset(
    {
        MeasurementValidityStatus.STABLE_UNDER_FROZEN_AUDIT,
        MeasurementValidityStatus.NOT_ESTABLISHED,
    }
)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _probability(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be a finite probability")
    return number


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return 0.0 if math.isclose(number, 0.0, rel_tol=0.0, abs_tol=1e-15) else number


def _status(
    value: MeasurementValidityStatus | str,
    *,
    label: str,
) -> MeasurementValidityStatus:
    try:
        status = (
            value
            if isinstance(value, MeasurementValidityStatus)
            else MeasurementValidityStatus(value)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is unsupported") from error
    if status not in _DIAGNOSTIC_STATUSES:
        raise ValueError(f"{label} is unsupported")
    return status


def _task(value: object, *, label: str) -> str:
    task = _text(value, label=label)
    if task not in _TASKS:
        raise ValueError(f"{label} is unsupported")
    return task


def _reward(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError("measurement diagnostic reward must be 0 or 1")
    return value


@dataclass(frozen=True)
class StaySwitchCell:
    task_variant: str
    reward: int
    transition_common: bool
    observed_stay_probability: float | None
    model_expected_stay_probability: float | None
    count: int
    status: MeasurementValidityStatus | str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_variant",
            _task(
                self.task_variant,
                label="measurement diagnostic task variant",
            ),
        )
        object.__setattr__(self, "reward", _reward(self.reward))
        if not isinstance(self.transition_common, bool):
            raise ValueError(
                "measurement diagnostic transition common flag must be bool"
            )
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 0
        ):
            raise ValueError(
                "measurement diagnostic cell count must be a non-negative integer"
            )
        status = _status(self.status, label="measurement diagnostic cell status")
        object.__setattr__(self, "status", status)
        if self.count == 0:
            if (
                self.observed_stay_probability is not None
                or self.model_expected_stay_probability is not None
            ):
                raise ValueError(
                    "empty measurement diagnostic cells cannot contain probabilities"
                )
            if status is not MeasurementValidityStatus.NOT_ESTABLISHED:
                raise ValueError(
                    "empty measurement diagnostic cells must be not established"
                )
            return
        if status is not MeasurementValidityStatus.STABLE_UNDER_FROZEN_AUDIT:
            raise ValueError(
                "populated measurement diagnostic cells must be established"
            )
        object.__setattr__(
            self,
            "observed_stay_probability",
            _probability(
                self.observed_stay_probability,
                label="measurement observed stay probability",
            ),
        )
        object.__setattr__(
            self,
            "model_expected_stay_probability",
            _probability(
                self.model_expected_stay_probability,
                label="measurement model expected stay probability",
            ),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "task_variant": self.task_variant,
            "reward": self.reward,
            "transition_common": self.transition_common,
            "observed_stay_probability": self.observed_stay_probability,
            "model_expected_stay_probability": (
                self.model_expected_stay_probability
            ),
            "count": self.count,
            "status": self.status.value,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _cell_sort_key(cell: StaySwitchCell) -> tuple[int, int]:
    return (cell.reward, int(cell.transition_common))


def _interaction(
    cells: Mapping[tuple[int, bool], StaySwitchCell],
    *,
    attribute: str,
) -> float:
    values = {
        key: getattr(cell, attribute) for key, cell in cells.items()
    }
    if any(value is None for value in values.values()):
        raise ValueError("measurement diagnostic interaction support is incomplete")
    return _finite(
        values[(1, True)]
        - values[(0, True)]
        - values[(1, False)]
        + values[(0, False)],
        label="measurement diagnostic interaction",
    )


@dataclass(frozen=True)
class StaySwitchDiagnostic:
    task_variant: str
    model_name: str
    cells: tuple[StaySwitchCell, ...]
    observed_interaction: float | None
    model_interaction: float | None
    status: MeasurementValidityStatus | str

    def __post_init__(self) -> None:
        task = _task(
            self.task_variant,
            label="measurement diagnostic task variant",
        )
        object.__setattr__(self, "task_variant", task)
        model_name = _text(
            self.model_name,
            label="measurement diagnostic model name",
        )
        if model_name not in _MODELS:
            raise ValueError("measurement diagnostic model is unsupported")
        object.__setattr__(self, "model_name", model_name)
        cells = tuple(self.cells)
        if any(not isinstance(cell, StaySwitchCell) for cell in cells):
            raise TypeError(
                "measurement diagnostic cells must be StaySwitchCell values"
            )
        cells = tuple(sorted(cells, key=_cell_sort_key))
        if tuple((cell.reward, cell.transition_common) for cell in cells) != _CELL_ORDER:
            raise ValueError(
                "measurement diagnostic requires all reward/transition cells"
            )
        if any(cell.task_variant != task for cell in cells):
            raise ValueError("measurement diagnostic cell task changed")
        object.__setattr__(self, "cells", cells)
        status = _status(self.status, label="measurement diagnostic status")
        object.__setattr__(self, "status", status)
        by_cell = {
            (cell.reward, cell.transition_common): cell for cell in cells
        }
        if status is MeasurementValidityStatus.NOT_ESTABLISHED:
            if all(cell.count > 0 for cell in cells):
                raise ValueError(
                    "not-established measurement diagnostic has complete support"
                )
            if self.observed_interaction is not None or self.model_interaction is not None:
                raise ValueError(
                    "not-established measurement diagnostic cannot contain interactions"
                )
            return
        if any(cell.count == 0 for cell in cells):
            raise ValueError(
                "established measurement diagnostic has an empty cell"
            )
        observed = _interaction(
            by_cell,
            attribute="observed_stay_probability",
        )
        model = _interaction(
            by_cell,
            attribute="model_expected_stay_probability",
        )
        supplied_observed = _finite(
            self.observed_interaction,
            label="measurement observed interaction",
        )
        supplied_model = _finite(
            self.model_interaction,
            label="measurement model interaction",
        )
        if supplied_observed != observed or supplied_model != model:
            raise ValueError("measurement diagnostic interaction formula changed")
        object.__setattr__(self, "observed_interaction", supplied_observed)
        object.__setattr__(self, "model_interaction", supplied_model)

    def identity_payload(self) -> dict[str, object]:
        return {
            "task_variant": self.task_variant,
            "model_name": self.model_name,
            "cells": tuple(cell.identity_payload() for cell in self.cells),
            "observed_interaction": self.observed_interaction,
            "model_interaction": self.model_interaction,
            "status": self.status.value,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _observed_action(target: tuple[tuple[str, float], ...]) -> str:
    values = dict(target)
    selected = tuple(
        action
        for action in _ACTIONS
        if values[f"first_stage.{action}"] == 1.0
    )
    if len(selected) != 1 or any(
        values[f"first_stage.{action}"] not in (0.0, 1.0)
        for action in _ACTIONS
    ):
        raise ValueError(
            "measurement diagnostic requires a one-hot current action"
        )
    return selected[0]


def _latest_history(
    case,
    *,
    previous_case,
) -> Mapping[str, object] | None:
    if "history" not in case.scenario.payload:
        return None
    history = case.scenario.payload["history"]
    if history is None or history == ():
        return None
    if not isinstance(history, tuple):
        raise ValueError(
            "measurement diagnostic retained history must be a sequence"
        )
    latest = history[-1]
    if not isinstance(latest, Mapping):
        raise ValueError(
            "measurement diagnostic retained history row must be a mapping"
        )
    previous_trial = latest.get("trial_id")
    if (
        isinstance(previous_trial, bool)
        or not isinstance(previous_trial, int)
        or previous_trial < 0
        or previous_trial >= case.trial_index
    ):
        raise ValueError(
            "measurement diagnostic previous trial identity is invalid"
        )
    if previous_case is not None:
        if previous_trial != previous_case.trial_index:
            raise ValueError(
                "measurement diagnostic history is not the previous retained trial"
            )
        if _observed_action(previous_case.target) != latest.get(
            "first_stage_action"
        ):
            raise ValueError(
                "measurement diagnostic previous action disagrees with retained case"
            )
    return latest


def _validated_previous_fields(
    latest: Mapping[str, object],
) -> tuple[str, int, bool]:
    previous_action = _text(
        latest.get("first_stage_action"),
        label="measurement diagnostic previous action",
    )
    if previous_action not in _ACTIONS:
        raise ValueError("measurement diagnostic previous action is unsupported")
    reward = _reward(latest.get("reward"))
    transition_common = latest.get("transition_common")
    if not isinstance(transition_common, bool):
        raise ValueError(
            "measurement diagnostic transition common flag must be bool"
        )
    return previous_action, reward, transition_common


def _validate_prediction_binding(
    audit_input: MeasurementAuditInput,
    prediction_artifact: MeasurementPredictionArtifact,
) -> None:
    if prediction_artifact.audit_input_hash != audit_input.content_hash:
        raise ValueError("measurement diagnostic audit input changed")
    candidates = {
        candidate.name: candidate for candidate in audit_input.frozen_candidates
    }
    if tuple(prediction_artifact.model_map) != _MODELS:
        raise ValueError("measurement diagnostic candidate set changed")
    expected_coverage = {
        (
            case.case_hash,
            case.scenario.content_hash,
            case.role,
            seed,
        )
        for case in audit_input.cases
        for seed in _SEEDS_BY_ROLE[case.role]
    }
    for model_name in _MODELS:
        model = prediction_artifact.model_map[model_name]
        if model.candidate_hash != candidates[model_name].content_hash:
            raise ValueError(
                f"measurement diagnostic candidate changed for {model_name!r}"
            )
        coverage = {
            (row.case_hash, row.scenario_hash, row.role, row.seed)
            for row in model.rows
        }
        if coverage != expected_coverage or len(model.rows) != len(expected_coverage):
            raise ValueError(
                f"measurement diagnostic prediction coverage changed for {model_name!r}"
            )


def build_feher_hare_stay_switch_diagnostics(
    audit_input: MeasurementAuditInput,
    prediction_artifact: MeasurementPredictionArtifact,
) -> tuple[StaySwitchDiagnostic, ...]:
    if not isinstance(audit_input, MeasurementAuditInput):
        raise TypeError("measurement diagnostic requires MeasurementAuditInput")
    if not isinstance(prediction_artifact, MeasurementPredictionArtifact):
        raise TypeError(
            "measurement diagnostic requires MeasurementPredictionArtifact"
        )
    _validate_prediction_binding(audit_input, prediction_artifact)

    ordered_cases = tuple(
        sorted(
            audit_input.cases,
            key=lambda case: (
                _TASKS.index(case.task_variant),
                case.participant_group_hash,
                case.trial_index,
            ),
        )
    )
    observation_rows: dict[
        tuple[str, int, bool, str],
        list[tuple[float, float]],
    ] = {}
    previous_by_participant: dict[tuple[str, str], object] = {}
    for case in ordered_cases:
        participant_key = (case.task_variant, case.participant_group_hash)
        previous_case = previous_by_participant.get(participant_key)
        latest = _latest_history(case, previous_case=previous_case)
        previous_by_participant[participant_key] = case
        if latest is None:
            continue
        previous_action, reward, transition_common = _validated_previous_fields(
            latest
        )
        current_action = _observed_action(case.target)
        observed_stay = 1.0 if current_action == previous_action else 0.0
        metric_name = f"first_stage.{previous_action}"
        seeds = _SEEDS_BY_ROLE[case.role]
        for model_name in _MODELS:
            prediction_map = prediction_artifact.model_map[model_name].row_map
            try:
                averaged = dict(
                    average_seed_metrics(
                        tuple(
                            prediction_map[(case.case_hash, seed)].metric_map
                            for seed in seeds
                        )
                    )
                )
            except KeyError as error:
                raise ValueError(
                    "measurement diagnostic prediction coverage changed"
                ) from error
            if metric_name not in averaged:
                raise ValueError(
                    "measurement diagnostic stay probability metric is missing"
                )
            expected_stay = _probability(
                averaged[metric_name],
                label="measurement model expected stay probability",
            )
            observation_rows.setdefault(
                (case.task_variant, reward, transition_common, model_name),
                [],
            ).append((observed_stay, expected_stay))

    diagnostics: list[StaySwitchDiagnostic] = []
    for task in _TASKS:
        for model_name in _MODELS:
            cells: list[StaySwitchCell] = []
            for reward, transition_common in _CELL_ORDER:
                observations = observation_rows.get(
                    (task, reward, transition_common, model_name),
                    [],
                )
                if not observations:
                    cells.append(
                        StaySwitchCell(
                            task_variant=task,
                            reward=reward,
                            transition_common=transition_common,
                            observed_stay_probability=None,
                            model_expected_stay_probability=None,
                            count=0,
                            status=MeasurementValidityStatus.NOT_ESTABLISHED,
                        )
                    )
                    continue
                cells.append(
                    StaySwitchCell(
                        task_variant=task,
                        reward=reward,
                        transition_common=transition_common,
                        observed_stay_probability=(
                            math.fsum(row[0] for row in observations)
                            / len(observations)
                        ),
                        model_expected_stay_probability=(
                            math.fsum(row[1] for row in observations)
                            / len(observations)
                        ),
                        count=len(observations),
                        status=(
                            MeasurementValidityStatus.STABLE_UNDER_FROZEN_AUDIT
                        ),
                    )
                )
            cell_map = {
                (cell.reward, cell.transition_common): cell for cell in cells
            }
            established = all(cell.count > 0 for cell in cells)
            diagnostics.append(
                StaySwitchDiagnostic(
                    task_variant=task,
                    model_name=model_name,
                    cells=tuple(cells),
                    observed_interaction=(
                        _interaction(
                            cell_map,
                            attribute="observed_stay_probability",
                        )
                        if established
                        else None
                    ),
                    model_interaction=(
                        _interaction(
                            cell_map,
                            attribute="model_expected_stay_probability",
                        )
                        if established
                        else None
                    ),
                    status=(
                        MeasurementValidityStatus.STABLE_UNDER_FROZEN_AUDIT
                        if established
                        else MeasurementValidityStatus.NOT_ESTABLISHED
                    ),
                )
            )
    return tuple(diagnostics)


__all__ = [
    "StaySwitchCell",
    "StaySwitchDiagnostic",
    "build_feher_hare_stay_switch_diagnostics",
]

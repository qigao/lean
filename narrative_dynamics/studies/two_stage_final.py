from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
import re
from statistics import fmean

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.external_prediction import ExternalFinalPredictionArtifact
from narrative_dynamics.observations.dataset import (
    ObservationDataset,
    ObservationPartitionRole,
    ObservationRecord,
)


_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ACTIONS = ("action_0", "action_1")
_TASKS = ("magic_carpet", "spaceship")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _hashes(
    values: object,
    *,
    label: str,
    require_nonempty: bool,
) -> tuple[str, ...]:
    try:
        hashes = tuple(_hash(value, label=label) for value in values)
    except TypeError as error:
        raise TypeError(f"{label}s must be iterable") from error
    if require_nonempty and not hashes:
        raise ValueError(f"{label}s must be non-empty")
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"{label}s must be unique")
    return tuple(sorted(hashes))


def _probability(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must be a finite probability")
    return number


def _binary_reward(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError("two-stage previous reward must be 0 or 1")
    return value


def _observed_action(record: ObservationRecord) -> str:
    counts = record.count_map
    if set(counts) != set(_ACTIONS) or sum(counts.values()) != 1:
        raise ValueError(
            "two-stage stay/switch diagnostic requires one observed first-stage action"
        )
    selected = tuple(name for name in _ACTIONS if counts[name] == 1)
    if len(selected) != 1:
        raise ValueError(
            "two-stage stay/switch diagnostic requires one-hot first-stage counts"
        )
    return selected[0]


def _record_identity(record: ObservationRecord) -> tuple[str, str, int]:
    task = _text(
        record.metadata.get("task_variant"),
        label="two-stage diagnostic task variant",
    )
    if task not in _TASKS:
        raise ValueError("two-stage diagnostic task variant is unsupported")
    participant = _text(
        record.metadata.get("source_participant_id"),
        label="two-stage diagnostic participant",
    )
    trial = record.metadata.get("source_trial_id")
    if isinstance(trial, bool) or not isinstance(trial, int):
        raise ValueError("two-stage diagnostic source trial id must be an integer")
    if record.scenario.payload.get("task_variant") != task:
        raise ValueError("two-stage diagnostic scenario task identity changed")
    return task, participant, trial


@dataclass(frozen=True)
class TwoStageStaySwitchCell:
    task_variant: str
    previous_reward: int
    previous_transition_common: bool
    observation_count: int
    observed_stay_rate: float
    predicted_stay_probability_by_model: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        task = _text(self.task_variant, label="two-stage diagnostic task variant")
        if task not in _TASKS:
            raise ValueError("two-stage diagnostic task variant is unsupported")
        object.__setattr__(self, "task_variant", task)
        object.__setattr__(self, "previous_reward", _binary_reward(self.previous_reward))
        if not isinstance(self.previous_transition_common, bool):
            raise ValueError(
                "two-stage previous transition common flag must be bool"
            )
        if (
            isinstance(self.observation_count, bool)
            or not isinstance(self.observation_count, int)
            or self.observation_count <= 0
        ):
            raise ValueError(
                "two-stage diagnostic observation count must be a positive integer"
            )
        object.__setattr__(
            self,
            "observed_stay_rate",
            _probability(
                self.observed_stay_rate,
                label="two-stage observed stay rate",
            ),
        )
        rows = tuple(
            sorted(
                (
                    _text(name, label="two-stage diagnostic model name"),
                    _probability(
                        probability,
                        label="two-stage predicted stay probability",
                    ),
                )
                for name, probability in self.predicted_stay_probability_by_model
            )
        )
        if not rows:
            raise ValueError(
                "two-stage diagnostic requires predicted stay probabilities"
            )
        if len({name for name, _ in rows}) != len(rows):
            raise ValueError("two-stage diagnostic model names must be unique")
        object.__setattr__(self, "predicted_stay_probability_by_model", rows)

    def identity_payload(self) -> dict[str, object]:
        return {
            "task_variant": self.task_variant,
            "previous_reward": self.previous_reward,
            "previous_transition_common": self.previous_transition_common,
            "observation_count": self.observation_count,
            "observed_stay_rate": self.observed_stay_rate,
            "predicted_stay_probability_by_model":
                self.predicted_stay_probability_by_model,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class TwoStageStaySwitchDiagnostic:
    prediction_artifact_hash: str
    cells: tuple[TwoStageStaySwitchCell, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prediction_artifact_hash",
            _hash(
                self.prediction_artifact_hash,
                label="two-stage diagnostic prediction artifact hash",
            ),
        )
        cells = tuple(
            sorted(
                tuple(self.cells),
                key=lambda item: (
                    item.task_variant,
                    item.previous_reward,
                    item.previous_transition_common,
                ),
            )
        )
        if not cells or any(
            not isinstance(item, TwoStageStaySwitchCell) for item in cells
        ):
            raise ValueError("two-stage diagnostic requires stay/switch cells")
        keys = tuple(
            (
                item.task_variant,
                item.previous_reward,
                item.previous_transition_common,
            )
            for item in cells
        )
        if len(set(keys)) != len(keys):
            raise ValueError("two-stage diagnostic cell identities must be unique")
        model_sets = tuple(
            tuple(name for name, _ in item.predicted_stay_probability_by_model)
            for item in cells
        )
        if any(names != model_sets[0] for names in model_sets[1:]):
            raise ValueError(
                "two-stage diagnostic cells must share one model set"
            )
        object.__setattr__(self, "cells", cells)

    def identity_payload(self) -> dict[str, object]:
        return {
            "prediction_artifact_hash": self.prediction_artifact_hash,
            "cells": tuple(item.identity_payload() for item in self.cells),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def build_two_stage_stay_switch_diagnostic(
    *,
    dataset: ObservationDataset,
    artifact: ExternalFinalPredictionArtifact,
) -> TwoStageStaySwitchDiagnostic:
    if not isinstance(dataset, ObservationDataset):
        raise TypeError("two-stage diagnostic requires ObservationDataset")
    if not isinstance(artifact, ExternalFinalPredictionArtifact):
        raise TypeError(
            "two-stage diagnostic requires ExternalFinalPredictionArtifact"
        )
    final_partition = dataset.partition(ObservationPartitionRole.FINAL_TEST)
    if artifact.dataset_hash != dataset.content_hash:
        raise ValueError("two-stage diagnostic dataset identity changed")
    if artifact.final_partition_hash != final_partition.content_hash:
        raise ValueError("two-stage diagnostic final partition identity changed")

    grouped_records: dict[
        tuple[str, str],
        list[tuple[int, ObservationRecord]],
    ] = defaultdict(list)
    for record in final_partition.records:
        task, participant, trial = _record_identity(record)
        grouped_records[(task, participant)].append((trial, record))

    observations: dict[
        tuple[str, int, bool],
        list[tuple[float, dict[str, float]]],
    ] = defaultdict(list)
    model_names = tuple(model.model_name for model in artifact.model_predictions)
    if not model_names:
        raise ValueError("two-stage diagnostic prediction artifact has no models")

    for (task, _participant), rows in sorted(grouped_records.items()):
        ordered = tuple(sorted(rows, key=lambda item: item[0]))
        for (_previous_trial, previous), (current_trial, current) in zip(
            ordered,
            ordered[1:],
        ):
            history = current.scenario.payload.get("history")
            if not isinstance(history, tuple) or not history:
                raise ValueError(
                    "two-stage diagnostic current trial is missing retained history"
                )
            latest = history[-1]
            if not isinstance(latest, Mapping):
                raise ValueError(
                    "two-stage diagnostic retained history row must be a mapping"
                )
            if latest.get("trial_id") != previous.metadata.get("source_trial_id"):
                raise ValueError(
                    "two-stage diagnostic history is not the previous retained trial"
                )
            previous_action = _text(
                latest.get("first_stage_action"),
                label="two-stage previous first-stage action",
            )
            if previous_action not in _ACTIONS:
                raise ValueError(
                    "two-stage previous first-stage action is unsupported"
                )
            if _observed_action(previous) != previous_action:
                raise ValueError(
                    "two-stage previous observed action disagrees with retained history"
                )
            current_action = _observed_action(current)
            previous_reward = _binary_reward(latest.get("reward"))
            transition_common = latest.get("transition_common")
            if not isinstance(transition_common, bool):
                raise ValueError(
                    "two-stage previous transition common flag must be bool"
                )

            predicted: dict[str, float] = {}
            metric_name = f"first_stage.{previous_action}"
            for model in artifact.model_predictions:
                prediction_map = model.prediction_map
                sealed = []
                for seed in artifact.simulation_seeds:
                    try:
                        prediction = prediction_map[(current.id, seed)]
                    except KeyError as error:
                        raise ValueError(
                            "two-stage diagnostic sealed prediction coverage changed"
                        ) from error
                    if prediction.scenario_id != current.scenario.id:
                        raise ValueError(
                            "two-stage diagnostic sealed scenario identity changed"
                        )
                    if metric_name not in prediction.metric_map:
                        raise ValueError(
                            "two-stage diagnostic sealed policy metric is missing"
                        )
                    sealed.append(
                        _probability(
                            prediction.metric_map[metric_name],
                            label="two-stage sealed stay probability",
                        )
                    )
                predicted[model.model_name] = fmean(sealed)

            if tuple(sorted(predicted)) != tuple(sorted(model_names)):
                raise ValueError(
                    "two-stage diagnostic model prediction set changed"
                )
            key = (task, previous_reward, transition_common)
            observations[key].append(
                (1.0 if current_action == previous_action else 0.0, predicted)
            )

    if not observations:
        raise ValueError(
            "two-stage diagnostic requires adjacent retained FINAL trial pairs"
        )

    cells = []
    for (task, reward, transition_common), rows in sorted(observations.items()):
        per_model = tuple(
            (
                model_name,
                fmean(row[1][model_name] for row in rows),
            )
            for model_name in sorted(model_names)
        )
        cells.append(
            TwoStageStaySwitchCell(
                task_variant=task,
                previous_reward=reward,
                previous_transition_common=transition_common,
                observation_count=len(rows),
                observed_stay_rate=fmean(row[0] for row in rows),
                predicted_stay_probability_by_model=per_model,
            )
        )
    return TwoStageStaySwitchDiagnostic(
        prediction_artifact_hash=artifact.content_hash,
        cells=tuple(cells),
    )


class TwoStageFinalAttemptStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"
    REVISION_REQUIRED = "revision_required"


@dataclass(frozen=True)
class TwoStageFinalAttempt:
    attempt_id: str
    preregistration_hash: str
    preflight_hash: str
    repository_revision: str
    dataset_hash: str
    final_target_hash: str
    frozen_candidate_hashes: tuple[str, ...]
    brier_release_hash: str
    log_release_hash: str
    started_at: str
    status: TwoStageFinalAttemptStatus
    failure_class: str | None = None
    completed_run_manifest_hashes: tuple[str, ...] = ()
    result_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attempt_id",
            _text(self.attempt_id, label="two-stage FINAL attempt id"),
        )
        for field_name, label in (
            ("preregistration_hash", "two-stage FINAL preregistration hash"),
            ("preflight_hash", "two-stage FINAL preflight hash"),
            ("dataset_hash", "two-stage FINAL dataset hash"),
            ("final_target_hash", "two-stage FINAL target hash"),
            ("brier_release_hash", "two-stage FINAL Brier release hash"),
            ("log_release_hash", "two-stage FINAL Log release hash"),
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=label),
            )
        object.__setattr__(
            self,
            "repository_revision",
            _text(
                self.repository_revision,
                label="two-stage FINAL repository revision",
            ),
        )
        object.__setattr__(
            self,
            "frozen_candidate_hashes",
            _hashes(
                self.frozen_candidate_hashes,
                label="two-stage FINAL frozen candidate hash",
                require_nonempty=True,
            ),
        )
        object.__setattr__(
            self,
            "started_at",
            _text(self.started_at, label="two-stage FINAL start time"),
        )
        if not isinstance(self.status, TwoStageFinalAttemptStatus):
            raise ValueError("two-stage FINAL attempt status is invalid")
        run_hashes = _hashes(
            self.completed_run_manifest_hashes,
            label="two-stage FINAL completed run manifest hash",
            require_nonempty=False,
        )
        object.__setattr__(
            self,
            "completed_run_manifest_hashes",
            run_hashes,
        )
        if self.result_hash is not None:
            object.__setattr__(
                self,
                "result_hash",
                _hash(
                    self.result_hash,
                    label="two-stage FINAL result hash",
                ),
            )

        if self.status is TwoStageFinalAttemptStatus.STARTED:
            if (
                self.failure_class is not None
                or run_hashes
                or self.result_hash is not None
            ):
                raise ValueError(
                    "started two-stage FINAL attempt cannot contain terminal data"
                )
        elif self.status is TwoStageFinalAttemptStatus.COMPLETED:
            if self.failure_class is not None:
                raise ValueError(
                    "completed two-stage FINAL attempt cannot contain failure class"
                )
            if not run_hashes or self.result_hash is None:
                raise ValueError(
                    "completed two-stage FINAL attempt requires run and result hashes"
                )
        else:
            failure = _text(
                self.failure_class,
                label="two-stage FINAL failure class",
            )
            object.__setattr__(self, "failure_class", failure)
            if run_hashes or self.result_hash is not None:
                raise ValueError(
                    "failed two-stage FINAL attempt cannot contain completed result data"
                )

    def scientific_identity_payload(self) -> dict[str, object]:
        return {
            "preregistration_hash": self.preregistration_hash,
            "preflight_hash": self.preflight_hash,
            "repository_revision": self.repository_revision,
            "dataset_hash": self.dataset_hash,
            "final_target_hash": self.final_target_hash,
            "frozen_candidate_hashes": self.frozen_candidate_hashes,
            "brier_release_hash": self.brier_release_hash,
            "log_release_hash": self.log_release_hash,
        }

    def identity_payload(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            **self.scientific_identity_payload(),
            "started_at": self.started_at,
            "status": self.status.value,
            "failure_class": self.failure_class,
            "completed_run_manifest_hashes": self.completed_run_manifest_hashes,
            "result_hash": self.result_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class TwoStageFinalAttemptLedger:
    attempts: tuple[TwoStageFinalAttempt, ...]

    def __post_init__(self) -> None:
        attempts = tuple(self.attempts)
        if any(not isinstance(item, TwoStageFinalAttempt) for item in attempts):
            raise TypeError(
                "two-stage FINAL attempt ledger requires attempt records"
            )
        started_by_id: dict[str, TwoStageFinalAttempt] = {}
        terminal_ids: set[str] = set()
        open_attempt_id: str | None = None
        for item in attempts:
            if item.status is TwoStageFinalAttemptStatus.STARTED:
                if item.attempt_id in started_by_id:
                    raise ValueError(
                        "two-stage FINAL attempt id cannot be reused"
                    )
                if open_attempt_id is not None:
                    raise ValueError(
                        "two-stage FINAL attempt ledger cannot have concurrent attempts"
                    )
                started_by_id[item.attempt_id] = item
                open_attempt_id = item.attempt_id
                continue
            started = started_by_id.get(item.attempt_id)
            if started is None:
                raise ValueError(
                    "two-stage FINAL terminal record requires a started attempt"
                )
            if item.attempt_id in terminal_ids:
                raise ValueError(
                    "two-stage FINAL terminal attempt cannot be overwritten or reopened"
                )
            if (
                item.scientific_identity_payload()
                != started.scientific_identity_payload()
                or item.started_at != started.started_at
            ):
                raise ValueError(
                    "two-stage FINAL terminal record changed attempt identity"
                )
            if open_attempt_id != item.attempt_id:
                raise ValueError(
                    "two-stage FINAL terminal record is out of append order"
                )
            terminal_ids.add(item.attempt_id)
            open_attempt_id = None
        object.__setattr__(self, "attempts", attempts)

    def identity_payload(self) -> dict[str, object]:
        return {
            "attempts": tuple(item.identity_payload() for item in self.attempts)
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _scientific_identity(
    *,
    preregistration_hash: str,
    preflight_hash: str,
    repository_revision: str,
    dataset_hash: str,
    final_target_hash: str,
    frozen_candidate_hashes: tuple[str, ...],
    brier_release_hash: str,
    log_release_hash: str,
) -> dict[str, object]:
    return {
        "preregistration_hash": _hash(
            preregistration_hash,
            label="two-stage FINAL preregistration hash",
        ),
        "preflight_hash": _hash(
            preflight_hash,
            label="two-stage FINAL preflight hash",
        ),
        "repository_revision": _text(
            repository_revision,
            label="two-stage FINAL repository revision",
        ),
        "dataset_hash": _hash(
            dataset_hash,
            label="two-stage FINAL dataset hash",
        ),
        "final_target_hash": _hash(
            final_target_hash,
            label="two-stage FINAL target hash",
        ),
        "frozen_candidate_hashes": _hashes(
            frozen_candidate_hashes,
            label="two-stage FINAL frozen candidate hash",
            require_nonempty=True,
        ),
        "brier_release_hash": _hash(
            brier_release_hash,
            label="two-stage FINAL Brier release hash",
        ),
        "log_release_hash": _hash(
            log_release_hash,
            label="two-stage FINAL Log release hash",
        ),
    }


def start_final_attempt(
    ledger: TwoStageFinalAttemptLedger,
    *,
    attempt_id: str,
    started_at: str,
    preregistration_hash: str,
    preflight_hash: str,
    repository_revision: str,
    dataset_hash: str,
    final_target_hash: str,
    frozen_candidate_hashes: tuple[str, ...],
    brier_release_hash: str,
    log_release_hash: str,
) -> TwoStageFinalAttemptLedger:
    if not isinstance(ledger, TwoStageFinalAttemptLedger):
        raise TypeError(
            "two-stage FINAL start requires TwoStageFinalAttemptLedger"
        )
    canonical_id = _text(attempt_id, label="two-stage FINAL attempt id")
    if any(item.attempt_id == canonical_id for item in ledger.attempts):
        raise ValueError("two-stage FINAL attempt id cannot be reused")
    if ledger.attempts and ledger.attempts[-1].status is TwoStageFinalAttemptStatus.STARTED:
        raise ValueError("two-stage FINAL attempt is already started")
    identity = _scientific_identity(
        preregistration_hash=preregistration_hash,
        preflight_hash=preflight_hash,
        repository_revision=repository_revision,
        dataset_hash=dataset_hash,
        final_target_hash=final_target_hash,
        frozen_candidate_hashes=frozen_candidate_hashes,
        brier_release_hash=brier_release_hash,
        log_release_hash=log_release_hash,
    )
    attempt = TwoStageFinalAttempt(
        attempt_id=canonical_id,
        started_at=started_at,
        status=TwoStageFinalAttemptStatus.STARTED,
        **identity,
    )
    return TwoStageFinalAttemptLedger(ledger.attempts + (attempt,))


def _started_attempt(
    ledger: TwoStageFinalAttemptLedger,
    *,
    attempt_id: str,
) -> TwoStageFinalAttempt:
    if not isinstance(ledger, TwoStageFinalAttemptLedger):
        raise TypeError(
            "two-stage FINAL transition requires TwoStageFinalAttemptLedger"
        )
    canonical_id = _text(attempt_id, label="two-stage FINAL attempt id")
    matching = tuple(
        item for item in ledger.attempts if item.attempt_id == canonical_id
    )
    if not matching:
        raise ValueError("two-stage FINAL attempt was not started")
    if matching[-1].status is not TwoStageFinalAttemptStatus.STARTED:
        raise ValueError(
            "two-stage FINAL terminal attempt cannot be overwritten or reopened"
        )
    if ledger.attempts[-1] is not matching[-1]:
        raise ValueError("two-stage FINAL attempt is not the active append head")
    return matching[-1]


def mark_infrastructure_failed(
    ledger: TwoStageFinalAttemptLedger,
    *,
    attempt_id: str,
    failure_class: str,
) -> TwoStageFinalAttemptLedger:
    started = _started_attempt(ledger, attempt_id=attempt_id)
    terminal = TwoStageFinalAttempt(
        attempt_id=started.attempt_id,
        **started.scientific_identity_payload(),
        started_at=started.started_at,
        status=TwoStageFinalAttemptStatus.INFRASTRUCTURE_FAILED,
        failure_class=failure_class,
    )
    return TwoStageFinalAttemptLedger(ledger.attempts + (terminal,))


def mark_revision_required(
    ledger: TwoStageFinalAttemptLedger,
    *,
    attempt_id: str,
    failure_class: str,
) -> TwoStageFinalAttemptLedger:
    started = _started_attempt(ledger, attempt_id=attempt_id)
    terminal = TwoStageFinalAttempt(
        attempt_id=started.attempt_id,
        **started.scientific_identity_payload(),
        started_at=started.started_at,
        status=TwoStageFinalAttemptStatus.REVISION_REQUIRED,
        failure_class=failure_class,
    )
    return TwoStageFinalAttemptLedger(ledger.attempts + (terminal,))


def complete_final_attempt(
    ledger: TwoStageFinalAttemptLedger,
    *,
    attempt_id: str,
    completed_run_manifest_hashes: tuple[str, ...],
    result_hash: str,
) -> TwoStageFinalAttemptLedger:
    started = _started_attempt(ledger, attempt_id=attempt_id)
    terminal = TwoStageFinalAttempt(
        attempt_id=started.attempt_id,
        **started.scientific_identity_payload(),
        started_at=started.started_at,
        status=TwoStageFinalAttemptStatus.COMPLETED,
        completed_run_manifest_hashes=completed_run_manifest_hashes,
        result_hash=result_hash,
    )
    return TwoStageFinalAttemptLedger(ledger.attempts + (terminal,))


def require_exact_retry(
    ledger: TwoStageFinalAttemptLedger,
    *,
    preregistration_hash: str,
    preflight_hash: str,
    repository_revision: str,
    dataset_hash: str,
    final_target_hash: str,
    frozen_candidate_hashes: tuple[str, ...],
    brier_release_hash: str,
    log_release_hash: str,
) -> TwoStageFinalAttempt:
    if not isinstance(ledger, TwoStageFinalAttemptLedger):
        raise TypeError(
            "two-stage FINAL retry requires TwoStageFinalAttemptLedger"
        )
    if not ledger.attempts:
        raise ValueError("two-stage FINAL retry requires a failed attempt")
    previous = ledger.attempts[-1]
    if previous.status is not TwoStageFinalAttemptStatus.INFRASTRUCTURE_FAILED:
        raise ValueError(
            "two-stage FINAL retry is allowed only after infrastructure failure"
        )
    expected = _scientific_identity(
        preregistration_hash=preregistration_hash,
        preflight_hash=preflight_hash,
        repository_revision=repository_revision,
        dataset_hash=dataset_hash,
        final_target_hash=final_target_hash,
        frozen_candidate_hashes=frozen_candidate_hashes,
        brier_release_hash=brier_release_hash,
        log_release_hash=log_release_hash,
    )
    if previous.scientific_identity_payload() != expected:
        raise ValueError(
            "two-stage FINAL retry changed a frozen scientific identity"
        )
    return previous


__all__ = [
    "TwoStageFinalAttempt",
    "TwoStageFinalAttemptLedger",
    "TwoStageFinalAttemptStatus",
    "TwoStageStaySwitchCell",
    "TwoStageStaySwitchDiagnostic",
    "build_two_stage_stay_switch_diagnostic",
    "complete_final_attempt",
    "mark_infrastructure_failed",
    "mark_revision_required",
    "require_exact_retry",
    "start_final_attempt",
]

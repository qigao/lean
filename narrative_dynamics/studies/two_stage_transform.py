from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import csv
import hashlib
import math
from pathlib import Path
import re

from narrative_dynamics.contracts import Scenario, stable_content_hash
from narrative_dynamics.observations import (
    ObservationDataset,
    ObservationPartition,
    ObservationPartitionRole,
    ObservationRecord,
)

from .two_stage_source import (
    TwoStageSourceFile,
    TwoStageSourceManifest,
    VerifiedTwoStageSnapshot,
    verify_two_stage_snapshot,
)


_MAGIC_HEADER = (
    "trial",
    "common",
    "reward.1.1",
    "reward.1.2",
    "reward.2.1",
    "reward.2.2",
    "isymbol_lft",
    "isymbol_rgt",
    "rt1",
    "choice1",
    "final_state",
    "fsymbol_lft",
    "fsymbol_rgt",
    "rt2",
    "choice2",
    "reward",
    "slow",
)
_SPACESHIP_HEADER = (
    "trial",
    "rwrd_prob0",
    "rwrd_prob1",
    "rwrd_prob2",
    "rwrd_prob3",
    "symbol0",
    "symbol1",
    "common",
    "choice1",
    "rt1",
    "final_state",
    "choice2",
    "rt2",
    "reward",
    "slow",
)
_ROLE_ORDER = ("train", "selection_validation", "final_test")
_ROLE_ENUM = {
    "train": ObservationPartitionRole.TRAIN,
    "selection_validation": ObservationPartitionRole.SELECTION_VALIDATION,
    "final_test": ObservationPartitionRole.FINAL_TEST,
}
_MAGIC_CONFIG_PATTERN = re.compile(
    r"Common transitions:\s*1\s*->\s*([^\s]+)\s*->\s*\((\d+)\s*,\s*(\d+)\)\s*;\s*"
    r"2\s*->\s*([^\s]+)\s*->\s*\((\d+)\s*,\s*(\d+)\)\s*;?\s*"
)
_TRANSFORM_VERSION = "1.0.0"
_SLOW_RULE_VERSION = "source-slow-equals-one-v1"
_CHOICE_NORMALIZATION_VERSION = "feher-hare-first-stage-v1"
_INFORMATION_FIREWALL_VERSION = "prechoice-plus-retained-history-v1"


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _int_text(value: object, *, label: str) -> int:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be an integer field")
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{label} must be an integer field") from error
    if str(parsed) != value and not (value.startswith("+") and str(parsed) == value[1:]):
        raise ValueError(f"{label} must use canonical integer text")
    return parsed


def _float_text(value: object, *, label: str, minimum: float | None = None) -> float:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a finite number")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return parsed


def _binary(value: object, *, label: str) -> int:
    parsed = _int_text(value, label=label)
    if parsed not in (0, 1):
        raise ValueError(f"{label} must be 0 or 1")
    return parsed


def _probability(value: object, *, label: str) -> float:
    parsed = _float_text(value, label=label)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return parsed


def _canonical_action_from_one_two(value: int, *, label: str) -> str:
    if value not in (1, 2):
        raise ValueError(f"{label} must be 1 or 2")
    return f"action_{value - 1}"


def _canonical_action_from_zero_one(value: int, *, label: str) -> str:
    if value not in (0, 1):
        raise ValueError(f"{label} must be 0 or 1")
    return f"action_{value}"


def _canonical_state_from_one_two(value: int) -> str:
    if value not in (1, 2):
        raise ValueError("magic final_state must be 1 or 2")
    return f"state_{value - 1}"


def _canonical_state_from_zero_one(value: int) -> str:
    if value not in (0, 1):
        raise ValueError("spaceship final_state must be 0 or 1")
    return f"state_{value}"


def _implementation_identity() -> Mapping[str, object]:
    data = Path(__file__).read_bytes()
    return {
        "name": "two_stage_observational_transform",
        "version": _TRANSFORM_VERSION,
        "module": __name__,
        "implementation_sha256": hashlib.sha256(data).hexdigest(),
        "slow_rule_version": _SLOW_RULE_VERSION,
        "choice_normalization_version": _CHOICE_NORMALIZATION_VERSION,
        "information_firewall_version": _INFORMATION_FIREWALL_VERSION,
    }


@dataclass(frozen=True)
class TwoStagePreChoiceView(Mapping[str, object]):
    task_variant: str
    source_participant_id: str
    trial_id: int
    first_stage_configuration: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        if self.task_variant not in ("magic_carpet", "spaceship"):
            raise ValueError("two-stage pre-choice task variant is unsupported")
        object.__setattr__(
            self,
            "source_participant_id",
            _text(self.source_participant_id, label="two-stage participant"),
        )
        if isinstance(self.trial_id, bool) or not isinstance(self.trial_id, int) or self.trial_id < 0:
            raise ValueError("two-stage trial id must be a non-negative integer")
        values = tuple(self.first_stage_configuration)
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            for item in values
        ):
            raise ValueError("first-stage configuration must contain named pairs")
        keys = tuple(item[0] for item in values)
        if len(keys) != len(set(keys)):
            raise ValueError("first-stage configuration keys must be unique")
        object.__setattr__(self, "first_stage_configuration", tuple(sorted(values)))

    def identity_payload(self) -> dict[str, object]:
        return {
            "task_variant": self.task_variant,
            "source_participant_id": self.source_participant_id,
            "trial_id": self.trial_id,
            "first_stage_configuration": self.first_stage_configuration,
        }

    def __getitem__(self, key: str) -> object:
        return self.identity_payload()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.identity_payload())

    def __len__(self) -> int:
        return 4


@dataclass(frozen=True)
class TwoStageObservedOutcome:
    first_stage_action: str
    transition_common: bool
    final_state: str
    second_stage_action: str
    reward: int

    def __post_init__(self) -> None:
        if self.first_stage_action not in ("action_0", "action_1"):
            raise ValueError("first-stage action is unsupported")
        if not isinstance(self.transition_common, bool):
            raise TypeError("transition_common must be bool")
        if self.final_state not in ("state_0", "state_1"):
            raise ValueError("final state is unsupported")
        if self.second_stage_action not in ("action_0", "action_1"):
            raise ValueError("second-stage action is unsupported")
        if self.reward not in (0, 1):
            raise ValueError("reward must be binary")

    def identity_payload(self) -> dict[str, object]:
        return {
            "first_stage_action": self.first_stage_action,
            "transition_common": self.transition_common,
            "final_state": self.final_state,
            "second_stage_action": self.second_stage_action,
            "reward": self.reward,
        }


@dataclass(frozen=True)
class CanonicalTwoStageTrial:
    pre_choice: TwoStagePreChoiceView
    outcome: TwoStageObservedOutcome
    source_path: str
    audit_latent_reward_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pre_choice, TwoStagePreChoiceView):
            raise TypeError("canonical two-stage trial requires TwoStagePreChoiceView")
        if not isinstance(self.outcome, TwoStageObservedOutcome):
            raise TypeError("canonical two-stage trial requires TwoStageObservedOutcome")
        object.__setattr__(self, "source_path", _text(self.source_path, label="source path"))
        values = tuple(float(value) for value in self.audit_latent_reward_probabilities)
        if len(values) != 4 or any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            raise ValueError("audit latent reward probabilities must contain four probabilities")
        object.__setattr__(self, "audit_latent_reward_probabilities", values)

    def identity_payload(self) -> dict[str, object]:
        return {
            "pre_choice": self.pre_choice.identity_payload(),
            "outcome": self.outcome.identity_payload(),
            "source_path": self.source_path,
            "audit_latent_reward_probabilities": self.audit_latent_reward_probabilities,
        }


@dataclass(frozen=True)
class TwoStageTransformReport:
    source_manifest_hash: str
    source_snapshot_hash: str
    repository_revision: str
    transform_identity: Mapping[str, object]
    trials: tuple[CanonicalTwoStageTrial, ...]
    eligible_participants: tuple[tuple[str, str], ...]
    structural_exclusions: tuple[tuple[str, str, str], ...]
    slow_counts: tuple[tuple[str, str, int], ...]
    retained_counts: tuple[tuple[str, str, int], ...]

    def __post_init__(self) -> None:
        trials = tuple(sorted(
            self.trials,
            key=lambda item: (
                item.pre_choice.task_variant,
                item.pre_choice.source_participant_id,
                item.pre_choice.trial_id,
            ),
        ))
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "eligible_participants", tuple(sorted(self.eligible_participants)))
        object.__setattr__(self, "structural_exclusions", tuple(sorted(self.structural_exclusions)))
        object.__setattr__(self, "slow_counts", tuple(sorted(self.slow_counts)))
        object.__setattr__(self, "retained_counts", tuple(sorted(self.retained_counts)))
        object.__setattr__(self, "transform_identity", dict(self.transform_identity))

    @property
    def slow_trial_count(self) -> int:
        return sum(count for _, _, count in self.slow_counts)

    @property
    def retained_trial_count(self) -> int:
        return sum(count for _, _, count in self.retained_counts)

    @property
    def content_hash(self) -> str:
        return stable_content_hash({
            "source_manifest_hash": self.source_manifest_hash,
            "source_snapshot_hash": self.source_snapshot_hash,
            "repository_revision": self.repository_revision,
            "transform_identity": self.transform_identity,
            "trials": tuple(trial.identity_payload() for trial in self.trials),
            "eligible_participants": self.eligible_participants,
            "structural_exclusions": self.structural_exclusions,
            "slow_counts": self.slow_counts,
            "retained_counts": self.retained_counts,
        })


@dataclass(frozen=True)
class TwoStageParticipantSplitPlan:
    namespace: str = "feher-hare-two-stage-v1"
    version: str = "1"
    ratios: tuple[tuple[str, int], ...] = (
        ("train", 60),
        ("selection_validation", 20),
        ("final_test", 20),
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _text(self.namespace, label="split namespace"))
        object.__setattr__(self, "version", _text(self.version, label="split version"))
        ratios = tuple(self.ratios)
        if tuple(role for role, _ in ratios) != _ROLE_ORDER:
            raise ValueError("two-stage split roles must be train, selection_validation, final_test")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for _, value in ratios):
            raise ValueError("two-stage split ratios must be positive integers")
        if sum(value for _, value in ratios) != 100:
            raise ValueError("two-stage split ratios must sum to 100")
        object.__setattr__(self, "ratios", ratios)

    @property
    def content_hash(self) -> str:
        return stable_content_hash({
            "namespace": self.namespace,
            "version": self.version,
            "ratios": self.ratios,
        })


@dataclass(frozen=True)
class TwoStageParticipantAssignment:
    assignments: tuple[tuple[str, str, str], ...]
    split_plan_hash: str = ""

    def __post_init__(self) -> None:
        assignments = tuple(sorted(self.assignments))
        identities: set[tuple[str, str]] = set()
        for task, participant, role in assignments:
            if task not in ("magic_carpet", "spaceship"):
                raise ValueError("participant assignment task is unsupported")
            _text(participant, label="participant assignment identity")
            if role not in _ROLE_ORDER:
                raise ValueError("participant assignment role is unsupported")
            identity = (task, participant)
            if identity in identities:
                raise ValueError("participant may appear in exactly one partition")
            identities.add(identity)
        object.__setattr__(self, "assignments", assignments)

    @property
    def content_hash(self) -> str:
        return stable_content_hash({
            "split_plan_hash": self.split_plan_hash,
            "assignments": self.assignments,
        })


def _read_rows(path: Path, expected_header: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != expected_header:
                raise ValueError(f"unexpected two-stage CSV schema: {path}")
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise ValueError(f"unable to read two-stage CSV: {path}") from error
    if not rows:
        raise ValueError(f"included two-stage main-task file is empty: {path}")
    return rows


def _validate_trial_order(rows: list[dict[str, str]], *, path: Path) -> tuple[int, ...]:
    trial_ids = tuple(_int_text(row["trial"], label=f"{path}: trial") for row in rows)
    if any(trial < 0 for trial in trial_ids):
        raise ValueError(f"trial ids must be non-negative: {path}")
    if any(right <= left for left, right in zip(trial_ids, trial_ids[1:])):
        raise ValueError(f"trial ids must be strictly increasing and unique: {path}")
    return trial_ids


def _validate_magic_config(root: Path, participant: str, files: Mapping[str, TwoStageSourceFile]) -> None:
    relative = f"results/magic_carpet/choices/{participant}_config.txt"
    metadata = files.get(relative)
    if metadata is None or metadata.purpose != "transform_metadata":
        raise ValueError(f"magic participant is missing pinned config metadata: {participant}")
    text = (root / relative).read_text(encoding="utf-8")
    match = _MAGIC_CONFIG_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(f"magic config metadata is malformed: {participant}")
    first_color, a, b, second_color, c, d = match.groups()
    if first_color == second_color or len({int(a), int(b), int(c), int(d)}) != 4:
        raise ValueError(f"magic config metadata is not structurally discriminating: {participant}")


def _validate_spaceship_info(root: Path, participant: str, files: Mapping[str, TwoStageSourceFile]) -> None:
    relative = f"results/spaceship/choices/{participant}_info.txt"
    metadata = files.get(relative)
    if metadata is None or metadata.purpose != "transform_metadata":
        raise ValueError(f"spaceship participant is missing pinned info metadata: {participant}")
    if not (root / relative).read_text(encoding="utf-8").strip():
        raise ValueError(f"spaceship info metadata is empty: {participant}")


def _parse_magic_trial(row: dict[str, str], *, participant: str, path: str) -> tuple[CanonicalTwoStageTrial, int]:
    trial_id = _int_text(row["trial"], label=f"{path}: trial")
    common = _binary(row["common"], label=f"{path}: common")
    reward = _binary(row["reward"], label=f"{path}: reward")
    slow = _binary(row["slow"], label=f"{path}: slow")
    latent = tuple(
        _probability(row[name], label=f"{path}: {name}")
        for name in ("reward.1.1", "reward.1.2", "reward.2.1", "reward.2.2")
    )
    left_symbol = _int_text(row["isymbol_lft"], label=f"{path}: isymbol_lft")
    right_symbol = _int_text(row["isymbol_rgt"], label=f"{path}: isymbol_rgt")
    if {left_symbol, right_symbol} != {1, 2}:
        raise ValueError(f"magic first-stage symbols must be exactly 1 and 2: {path}")
    _float_text(row["rt1"], label=f"{path}: rt1", minimum=0.0)
    _float_text(row["rt2"], label=f"{path}: rt2", minimum=0.0)
    choice1 = _int_text(row["choice1"], label=f"{path}: choice1")
    final_state = _int_text(row["final_state"], label=f"{path}: final_state")
    choice2 = _int_text(row["choice2"], label=f"{path}: choice2")
    _int_text(row["fsymbol_lft"], label=f"{path}: fsymbol_lft")
    _int_text(row["fsymbol_rgt"], label=f"{path}: fsymbol_rgt")
    action0_position = "left" if left_symbol == 1 else "right"
    action1_position = "left" if left_symbol == 2 else "right"
    return (
        CanonicalTwoStageTrial(
            pre_choice=TwoStagePreChoiceView(
                task_variant="magic_carpet",
                source_participant_id=participant,
                trial_id=trial_id,
                first_stage_configuration=(
                    ("action_0_position", action0_position),
                    ("action_1_position", action1_position),
                ),
            ),
            outcome=TwoStageObservedOutcome(
                first_stage_action=_canonical_action_from_one_two(choice1, label="magic choice1"),
                transition_common=bool(common),
                final_state=_canonical_state_from_one_two(final_state),
                second_stage_action=_canonical_action_from_one_two(choice2, label="magic choice2"),
                reward=reward,
            ),
            source_path=path,
            audit_latent_reward_probabilities=latent,
        ),
        slow,
    )


def _parse_spaceship_trial(row: dict[str, str], *, participant: str, path: str) -> tuple[CanonicalTwoStageTrial, int]:
    trial_id = _int_text(row["trial"], label=f"{path}: trial")
    common = _binary(row["common"], label=f"{path}: common")
    reward = _binary(row["reward"], label=f"{path}: reward")
    slow = _binary(row["slow"], label=f"{path}: slow")
    latent = tuple(
        _probability(row[name], label=f"{path}: {name}")
        for name in ("rwrd_prob0", "rwrd_prob1", "rwrd_prob2", "rwrd_prob3")
    )
    symbol0 = _binary(row["symbol0"], label=f"{path}: symbol0")
    symbol1 = _binary(row["symbol1"], label=f"{path}: symbol1")
    if symbol0 == symbol1:
        raise ValueError(f"spaceship first-stage symbols must differ: {path}")
    _binary(row["choice1"], label=f"{path}: choice1")
    _float_text(row["rt1"], label=f"{path}: rt1", minimum=0.0)
    _float_text(row["rt2"], label=f"{path}: rt2", minimum=0.0)
    final_state = _binary(row["final_state"], label=f"{path}: final_state")
    choice2 = _binary(row["choice2"], label=f"{path}: choice2")
    relative = final_state + 1 if common else 2 - final_state
    return (
        CanonicalTwoStageTrial(
            pre_choice=TwoStagePreChoiceView(
                task_variant="spaceship",
                source_participant_id=participant,
                trial_id=trial_id,
                first_stage_configuration=(("symbol0", symbol0), ("symbol1", symbol1)),
            ),
            outcome=TwoStageObservedOutcome(
                first_stage_action=_canonical_action_from_one_two(relative, label="spaceship relative choice"),
                transition_common=bool(common),
                final_state=_canonical_state_from_zero_one(final_state),
                second_stage_action=_canonical_action_from_zero_one(choice2, label="spaceship choice2"),
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
) -> TwoStageTransformReport:
    if not isinstance(snapshot, VerifiedTwoStageSnapshot):
        raise TypeError("two-stage transform requires a verified source snapshot")
    if not isinstance(manifest, TwoStageSourceManifest):
        raise TypeError("two-stage transform requires TwoStageSourceManifest")
    if snapshot.manifest_hash != manifest.content_hash:
        raise ValueError("source snapshot and manifest identity differ")
    root = Path(snapshot.root)
    verified = verify_two_stage_snapshot(root, manifest)
    if verified.source_snapshot_hash != snapshot.source_snapshot_hash:
        raise ValueError("source snapshot identity changed before transform")

    files = {item.path: item for item in manifest.files}
    retained: list[CanonicalTwoStageTrial] = []
    eligible: list[tuple[str, str]] = []
    exclusions: list[tuple[str, str, str]] = []
    slow_counts: list[tuple[str, str, int]] = []
    retained_counts: list[tuple[str, str, int]] = []

    evidence_files = tuple(item for item in manifest.files if item.purpose == "scientific_evidence")
    for source in evidence_files:
        participant = source.source_participant_id
        assert participant is not None
        source_path = root / source.path
        if source.task_variant == "magic_carpet":
            _validate_magic_config(root, participant, files)
            rows = _read_rows(source_path, _MAGIC_HEADER)
            _validate_trial_order(rows, path=source_path)
            parsed = tuple(
                _parse_magic_trial(row, participant=participant, path=source.path)
                for row in rows
            )
        else:
            _validate_spaceship_info(root, participant, files)
            rows = _read_rows(source_path, _SPACESHIP_HEADER)
            _validate_trial_order(rows, path=source_path)
            parsed = tuple(
                _parse_spaceship_trial(row, participant=participant, path=source.path)
                for row in rows
            )

        participant_retained = tuple(trial for trial, slow in parsed if slow == 0)
        slow_count = sum(slow for _, slow in parsed)
        slow_counts.append((source.task_variant, participant, slow_count))
        retained_counts.append((source.task_variant, participant, len(participant_retained)))
        if not participant_retained:
            exclusions.append((source.task_variant, participant, "no_retained_scorable_trials"))
            continue
        eligible.append((source.task_variant, participant))
        retained.extend(participant_retained)

    if not retained:
        raise ValueError("two-stage transform produced no retained scorable trials")

    return TwoStageTransformReport(
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


def _apportion(count: int, plan: TwoStageParticipantSplitPlan) -> dict[str, int]:
    total = sum(value for _, value in plan.ratios)
    allocated: dict[str, int] = {}
    remainders: list[tuple[int, int, str]] = []
    used = 0
    for index, (role, ratio) in enumerate(plan.ratios):
        quotient, remainder = divmod(count * ratio, total)
        allocated[role] = quotient
        used += quotient
        remainders.append((-remainder, index, role))
    for _, _, role in sorted(remainders)[: count - used]:
        allocated[role] += 1
    return allocated


def assign_two_stage_participants(
    report: TwoStageTransformReport,
    plan: TwoStageParticipantSplitPlan,
) -> TwoStageParticipantAssignment:
    if not isinstance(report, TwoStageTransformReport):
        raise TypeError("participant assignment requires TwoStageTransformReport")
    if not isinstance(plan, TwoStageParticipantSplitPlan):
        raise TypeError("participant assignment requires TwoStageParticipantSplitPlan")
    assignments: list[tuple[str, str, str]] = []
    for task in ("magic_carpet", "spaceship"):
        participants = [participant for item_task, participant in report.eligible_participants if item_task == task]
        participants.sort(
            key=lambda participant: (
                stable_content_hash((plan.namespace, plan.version, task, participant)),
                participant,
            )
        )
        counts = _apportion(len(participants), plan)
        offset = 0
        for role in _ROLE_ORDER:
            limit = offset + counts[role]
            assignments.extend((task, participant, role) for participant in participants[offset:limit])
            offset = limit
        if offset != len(participants):
            raise RuntimeError("participant split apportionment did not consume stratum")
    if not assignments:
        raise ValueError("participant split requires eligible participants")
    return TwoStageParticipantAssignment(
        assignments=tuple(assignments),
        split_plan_hash=plan.content_hash,
    )


def _history_payload(trial: CanonicalTwoStageTrial) -> dict[str, object]:
    return {
        "trial_id": trial.pre_choice.trial_id,
        "first_stage_action": trial.outcome.first_stage_action,
        "transition_common": trial.outcome.transition_common,
        "final_state": trial.outcome.final_state,
        "second_stage_action": trial.outcome.second_stage_action,
        "reward": trial.outcome.reward,
    }


def build_two_stage_observation_dataset(
    report: TwoStageTransformReport,
    assignment: TwoStageParticipantAssignment,
) -> ObservationDataset:
    if not isinstance(report, TwoStageTransformReport):
        raise TypeError("dataset construction requires TwoStageTransformReport")
    if not isinstance(assignment, TwoStageParticipantAssignment):
        raise TypeError("dataset construction requires TwoStageParticipantAssignment")
    role_by_participant = {
        (task, participant): role for task, participant, role in assignment.assignments
    }
    if set(role_by_participant) != set(report.eligible_participants):
        raise ValueError("participant assignment must cover every eligible participant exactly once")

    grouped: dict[tuple[str, str], list[CanonicalTwoStageTrial]] = {}
    for trial in report.trials:
        key = (trial.pre_choice.task_variant, trial.pre_choice.source_participant_id)
        grouped.setdefault(key, []).append(trial)

    records_by_role: dict[str, list[ObservationRecord]] = {role: [] for role in _ROLE_ORDER}
    for key in sorted(grouped):
        task, participant = key
        trials = sorted(grouped[key], key=lambda item: item.pre_choice.trial_id)
        history: list[dict[str, object]] = []
        for trial in trials:
            history_tuple = tuple(history)
            history_hash = stable_content_hash(history_tuple)
            scenario_identity = stable_content_hash({
                "task_variant": task,
                "participant_source_identity": stable_content_hash((task, participant)),
                "trial_id": trial.pre_choice.trial_id,
                "pre_choice": trial.pre_choice.identity_payload(),
                "causal_history_hash": history_hash,
            })
            scenario = Scenario(
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
                ObservationRecord(
                    id=record_id,
                    scenario=scenario,
                    counts=counts,
                    metadata={
                        "task_variant": task,
                        "source_participant_id": participant,
                        "source_trial_id": trial.pre_choice.trial_id,
                        "source_path": trial.source_path,
                        "causal_history_hash": history_hash,
                        "participant_assignment_hash": assignment.content_hash,
                        "transform_report_hash": report.content_hash,
                    },
                )
            )
            history.append(_history_payload(trial))

    if any(not records_by_role[role] for role in _ROLE_ORDER):
        raise ValueError("two-stage dataset requires non-empty train, selection, and final partitions")

    transform_identity = {
        **dict(report.transform_identity),
        "participant_assignment_hash": assignment.content_hash,
        "source_manifest_hash": report.source_manifest_hash,
        "source_snapshot_hash": report.source_snapshot_hash,
    }
    return ObservationDataset(
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
            "participant_assignment_hash": assignment.content_hash,
            "transform_report_hash": report.content_hash,
        },
        partitions=tuple(
            ObservationPartition(
                name=role,
                role=_ROLE_ENUM[role],
                records=tuple(records_by_role[role]),
            )
            for role in _ROLE_ORDER
        ),
    )


__all__ = [
    "CanonicalTwoStageTrial",
    "TwoStageObservedOutcome",
    "TwoStageParticipantAssignment",
    "TwoStageParticipantSplitPlan",
    "TwoStagePreChoiceView",
    "TwoStageTransformReport",
    "assign_two_stage_participants",
    "build_two_stage_observation_dataset",
    "transform_two_stage_snapshot",
]

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from statistics import fmean

from narrative_dynamics.abm.calibration import (
    EmpiricalABMCalibrationReport,
    apply_calibration_candidate,
    observe_dynamic_role_state,
)
from narrative_dynamics.abm.calibration_contracts import (
    ABMCalibrationCandidate,
    CalibrationSplit,
    EmpiricalABMDataset,
    ObservedABMSnapshot,
)
from narrative_dynamics.abm.contracts import _hash, _text
from narrative_dynamics.abm.role_contracts import (
    DynamicRoleModel,
    initialize_dynamic_role_population,
)
from narrative_dynamics.abm.roles import simulate_dynamic_role_population
from narrative_dynamics.contracts import stable_content_hash


_PRIMARY_METRICS = {
    "active_share",
    "mean_active_belief",
    "mean_trust",
    "learned_edge_rate",
    "effective_active_edge_rate",
    "rewired_edge_rate",
    "verification_rate",
    "active_sharing_rate",
    "transition_rate",
    "role_entropy",
}


class ExperimentObjective(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True)
class ABMExperimentArm:
    arm_id: str
    candidate: ABMCalibrationCandidate

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", _text(self.arm_id, label="experiment arm id"))
        if not isinstance(self.candidate, ABMCalibrationCandidate):
            raise TypeError("experiment arm requires ABMCalibrationCandidate")

    def to_dict(self) -> dict[str, object]:
        return {"arm_id": self.arm_id, "candidate": self.candidate.to_dict()}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ABMExperimentProtocol:
    protocol_id: str
    version: str
    baseline_arm_id: str
    primary_metric: str
    objective: ExperimentObjective
    arms: tuple[ABMExperimentArm, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_id", _text(self.protocol_id, label="experiment protocol id"))
        object.__setattr__(self, "version", _text(self.version, label="experiment protocol version"))
        object.__setattr__(self, "baseline_arm_id", _text(self.baseline_arm_id, label="experiment baseline arm id"))
        metric = _text(self.primary_metric, label="experiment primary metric")
        if metric not in _PRIMARY_METRICS:
            raise ValueError("experiment primary metric is not in snapshot schema")
        object.__setattr__(self, "primary_metric", metric)
        if not isinstance(self.objective, ExperimentObjective):
            raise TypeError("experiment objective must be ExperimentObjective")
        if not isinstance(self.arms, tuple) or len(self.arms) < 2:
            raise ValueError("experiment protocol requires at least two arms")
        if any(not isinstance(item, ABMExperimentArm) for item in self.arms):
            raise TypeError("experiment protocol has invalid arms")
        arm_ids = tuple(item.arm_id for item in self.arms)
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("experiment arm ids must be unique")
        if self.baseline_arm_id not in arm_ids:
            raise ValueError("experiment baseline arm must exist")
        candidates = tuple(item.candidate for item in self.arms)
        if len(set(candidates)) != len(candidates):
            raise ValueError("experiment arm candidate values must be unique")
        object.__setattr__(self, "arms", tuple(sorted(self.arms, key=lambda item: item.arm_id)))

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_id": self.protocol_id,
            "version": self.version,
            "baseline_arm_id": self.baseline_arm_id,
            "primary_metric": self.primary_metric,
            "objective": self.objective.value,
            "arms": [item.to_dict() for item in self.arms],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class ABMExperimentCaseResult:
    arm_id: str
    case_id: str
    final_snapshot: ObservedABMSnapshot
    final_state_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", _text(self.arm_id, label="experiment case arm id"))
        object.__setattr__(self, "case_id", _text(self.case_id, label="experiment case id"))
        if not isinstance(self.final_snapshot, ObservedABMSnapshot):
            raise TypeError("experiment case requires final snapshot")
        object.__setattr__(self, "final_state_hash", _hash(self.final_state_hash, label="experiment final state hash"))

    def to_dict(self) -> dict[str, object]:
        return {
            "arm_id": self.arm_id,
            "case_id": self.case_id,
            "final_snapshot": self.final_snapshot.to_dict(),
            "final_state_hash": self.final_state_hash,
        }


@dataclass(frozen=True)
class ABMExperimentArmResult:
    arm: ABMExperimentArm
    case_results: tuple[ABMExperimentCaseResult, ...]
    mean_outcome: float
    delta_from_baseline: float

    def __post_init__(self) -> None:
        if not isinstance(self.arm, ABMExperimentArm):
            raise TypeError("experiment arm result requires arm")
        if not isinstance(self.case_results, tuple) or not self.case_results:
            raise ValueError("experiment arm result requires case results")
        if any(not isinstance(item, ABMExperimentCaseResult) for item in self.case_results):
            raise TypeError("experiment arm result has invalid case results")
        if any(item.arm_id != self.arm.arm_id for item in self.case_results):
            raise ValueError("experiment case results must bind exact arm")
        ids = tuple(item.case_id for item in self.case_results)
        if len(set(ids)) != len(ids):
            raise ValueError("experiment case result ids must be unique")
        object.__setattr__(self, "case_results", tuple(sorted(self.case_results, key=lambda item: item.case_id)))
        object.__setattr__(self, "mean_outcome", _finite(self.mean_outcome, label="experiment mean outcome"))
        object.__setattr__(self, "delta_from_baseline", _finite(self.delta_from_baseline, label="experiment baseline delta"))

    def to_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm.to_dict(),
            "case_results": [item.to_dict() for item in self.case_results],
            "mean_outcome": self.mean_outcome,
            "delta_from_baseline": self.delta_from_baseline,
        }


@dataclass(frozen=True)
class CalibratedABMExperimentReport:
    base_model_hash: str
    dataset_hash: str
    calibration_report_hash: str
    protocol: ABMExperimentProtocol
    selected_candidate: ABMCalibrationCandidate
    arm_results: tuple[ABMExperimentArmResult, ...]
    ranking: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("base_model_hash", "dataset_hash", "calibration_report_hash"):
            object.__setattr__(self, name, _hash(getattr(self, name), label=f"experiment {name}"))
        if not isinstance(self.protocol, ABMExperimentProtocol):
            raise TypeError("experiment report requires protocol")
        if not isinstance(self.selected_candidate, ABMCalibrationCandidate):
            raise TypeError("experiment report requires selected candidate")
        baseline_arm = next(item for item in self.protocol.arms if item.arm_id == self.protocol.baseline_arm_id)
        if baseline_arm.candidate != self.selected_candidate:
            raise ValueError("experiment baseline must equal selected calibration candidate")
        if not isinstance(self.arm_results, tuple):
            raise TypeError("experiment arm results must be a tuple")
        results = tuple(self.arm_results)
        by_id = {item.arm.arm_id: item for item in results if isinstance(item, ABMExperimentArmResult)}
        expected_ids = {item.arm_id for item in self.protocol.arms}
        if len(by_id) != len(results) or set(by_id) != expected_ids:
            raise ValueError("experiment results must cover exact protocol arms")
        case_sets = {tuple(item.case_id for item in result.case_results) for result in results}
        if len(case_sets) != 1:
            raise ValueError("experiment arms must cover identical paired cases")
        for result in results:
            expected_mean = fmean(getattr(item.final_snapshot, self.protocol.primary_metric) for item in result.case_results)
            if not math.isclose(result.mean_outcome, expected_mean, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError("experiment mean outcome must match case results")
        baseline = by_id[self.protocol.baseline_arm_id].mean_outcome
        for result in results:
            expected_delta = result.mean_outcome - baseline
            if not math.isclose(result.delta_from_baseline, expected_delta, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError("experiment baseline deltas must match arm means")
        canonical = tuple(sorted(results, key=lambda item: item.arm.arm_id))
        object.__setattr__(self, "arm_results", canonical)
        reverse = self.protocol.objective is ExperimentObjective.MAXIMIZE
        expected_ranking = tuple(
            item.arm.arm_id
            for item in sorted(
                results,
                key=lambda item: ((-item.mean_outcome if reverse else item.mean_outcome), item.arm.arm_id),
            )
        )
        if self.ranking != expected_ranking:
            raise ValueError("experiment ranking must match declared objective")

    def to_dict(self) -> dict[str, object]:
        return {
            "base_model_hash": self.base_model_hash,
            "dataset_hash": self.dataset_hash,
            "calibration_report_hash": self.calibration_report_hash,
            "protocol": self.protocol.to_dict(),
            "selected_candidate": self.selected_candidate.to_dict(),
            "arm_results": [item.to_dict() for item in self.arm_results],
            "ranking": self.ranking,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def run_calibrated_abm_experiment(
    base_model: DynamicRoleModel,
    dataset: EmpiricalABMDataset,
    calibration: EmpiricalABMCalibrationReport,
    protocol: ABMExperimentProtocol,
) -> CalibratedABMExperimentReport:
    if not isinstance(base_model, DynamicRoleModel):
        raise TypeError("calibrated experiment requires DynamicRoleModel")
    if not isinstance(dataset, EmpiricalABMDataset):
        raise TypeError("calibrated experiment requires EmpiricalABMDataset")
    if not isinstance(calibration, EmpiricalABMCalibrationReport):
        raise TypeError("calibrated experiment requires calibration report")
    if not isinstance(protocol, ABMExperimentProtocol):
        raise TypeError("calibrated experiment requires protocol")
    if calibration.base_model_hash != base_model.content_hash:
        raise ValueError("calibrated experiment base model does not match calibration")
    if calibration.dataset_hash != dataset.content_hash:
        raise ValueError("calibrated experiment dataset does not match calibration")
    baseline = next(item for item in protocol.arms if item.arm_id == protocol.baseline_arm_id)
    if baseline.candidate != calibration.selected_candidate:
        raise ValueError("experiment baseline must equal selected calibration candidate")
    holdout = tuple(item for item in dataset.cases if item.split is CalibrationSplit.HOLDOUT)
    raw: list[tuple[ABMExperimentArm, tuple[ABMExperimentCaseResult, ...], float]] = []
    for arm in protocol.arms:
        model = apply_calibration_candidate(base_model, arm.candidate)
        cases = []
        for case in holdout:
            try:
                initial = initialize_dynamic_role_population(model, beliefs=dict(case.initial_beliefs))
                trajectory = simulate_dynamic_role_population(
                    model,
                    initial,
                    environment_event_schedule=case.environment_event_schedule,
                    truth_observation_schedule=case.truth_observation_schedule,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"experiment arm {arm.arm_id!r} cannot replay case {case.case_id!r}: {error}") from error
            cases.append(ABMExperimentCaseResult(arm.arm_id, case.case_id, observe_dynamic_role_state(model, trajectory.final_state), trajectory.final_state.content_hash))
        case_tuple = tuple(cases)
        mean = fmean(getattr(item.final_snapshot, protocol.primary_metric) for item in case_tuple)
        raw.append((arm, case_tuple, mean))
    baseline_mean = next(mean for arm, _cases, mean in raw if arm.arm_id == protocol.baseline_arm_id)
    results = tuple(ABMExperimentArmResult(arm, cases, mean, mean - baseline_mean) for arm, cases, mean in raw)
    reverse = protocol.objective is ExperimentObjective.MAXIMIZE
    ranking = tuple(item.arm.arm_id for item in sorted(results, key=lambda item: ((-item.mean_outcome if reverse else item.mean_outcome), item.arm.arm_id)))
    return CalibratedABMExperimentReport(base_model.content_hash, dataset.content_hash, calibration.content_hash, protocol, calibration.selected_candidate, results, ranking)


__all__ = (
    "ExperimentObjective",
    "ABMExperimentArm",
    "ABMExperimentProtocol",
    "ABMExperimentCaseResult",
    "ABMExperimentArmResult",
    "CalibratedABMExperimentReport",
    "run_calibrated_abm_experiment",
)

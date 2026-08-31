from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean

from narrative_dynamics.abm.autonomy_contracts import AutonomousNetworkModel
from narrative_dynamics.abm.autonomy_metrics import measure_agent_autonomy
from narrative_dynamics.abm.calibration_contracts import (
    ABMCalibrationCandidate,
    ABMCalibrationWeights,
    CalibrationSplit,
    EmpiricalABMCase,
    EmpiricalABMDataset,
    ObservedABMSnapshot,
)
from narrative_dynamics.abm.contracts import _hash, _text
from narrative_dynamics.abm.evolving_contracts import EvolvingNetworkModel
from narrative_dynamics.abm.evolving_metrics import measure_evolving_system
from narrative_dynamics.abm.role_contracts import (
    DynamicRoleModel,
    DynamicRolePopulationState,
    initialize_dynamic_role_population,
)
from narrative_dynamics.abm.role_metrics import measure_role_dynamics
from narrative_dynamics.abm.roles import (
    _validate_dynamic_role_model_state,
    simulate_dynamic_role_population,
)
from narrative_dynamics.contracts import stable_content_hash


def observe_dynamic_role_state(
    model: DynamicRoleModel,
    state: DynamicRolePopulationState,
) -> ObservedABMSnapshot:
    """Project one positive V7 state through the declared calibration metrics."""

    _validate_dynamic_role_model_state(model, state)
    if state.round_index == 0:
        raise ValueError("calibration observation requires a positive round")
    evolving = measure_evolving_system(
        model.autonomy_model.evolving_model,
        state.autonomy_state.evolving_state,
    )
    autonomy = measure_agent_autonomy(
        model.autonomy_model,
        state.autonomy_state,
    )
    roles = measure_role_dynamics(model, state)
    topology = state.autonomy_state.evolving_state.edge_topology
    rewired_edge_rate = sum(item.rewiring_count > 0 for item in topology) / len(
        topology
    )
    return ObservedABMSnapshot(
        state.round_index,
        evolving.active_share,
        (
            0.0
            if evolving.mean_active_belief is None
            else evolving.mean_active_belief
        ),
        evolving.mean_trust,
        evolving.learned_edge_rate,
        evolving.effective_active_edge_rate,
        rewired_edge_rate,
        autonomy.verification_rate,
        autonomy.active_sharing_rate,
        roles.transition_rate,
        roles.role_entropy,
    )


def apply_calibration_candidate(
    base_model: DynamicRoleModel,
    candidate: ABMCalibrationCandidate,
) -> DynamicRoleModel:
    """Rebuild a V7 model with only the three declared V8 parameters changed."""

    if not isinstance(base_model, DynamicRoleModel):
        raise TypeError("ABM calibration requires DynamicRoleModel")
    if not isinstance(candidate, ABMCalibrationCandidate):
        raise TypeError("ABM calibration requires ABMCalibrationCandidate")
    prior_evolving = base_model.autonomy_model.evolving_model
    evolving = EvolvingNetworkModel(
        prior_evolving.model_id,
        prior_evolving.version,
        prior_evolving.base_model,
        prior_evolving.initial_active_agent_ids,
        candidate.learning_rate,
        prior_evolving.initial_trust,
        candidate.dissolution_similarity,
        candidate.formation_similarity,
    )
    autonomy = AutonomousNetworkModel(
        base_model.autonomy_model.model_id,
        base_model.autonomy_model.version,
        evolving,
        base_model.autonomy_model.role_policies,
    )
    return DynamicRoleModel(
        base_model.model_id,
        base_model.version,
        autonomy,
        base_model.transition_rules,
    )


def _finite_loss(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class ABMCaseCalibrationFit:
    candidate: ABMCalibrationCandidate
    case_id: str
    split: CalibrationSplit
    loss: float
    observed_snapshot_hashes: tuple[str, ...]
    predicted_snapshot_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ABMCalibrationCandidate):
            raise TypeError("ABM case fit requires calibration candidate")
        object.__setattr__(
            self,
            "case_id",
            _text(self.case_id, label="ABM calibration fit case id"),
        )
        if not isinstance(self.split, CalibrationSplit):
            raise TypeError("ABM case fit split must be CalibrationSplit")
        object.__setattr__(
            self,
            "loss",
            _finite_loss(self.loss, label="ABM case calibration loss"),
        )
        for field_name in (
            "observed_snapshot_hashes",
            "predicted_snapshot_hashes",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"ABM case fit {field_name} must be a nonempty tuple")
            object.__setattr__(
                self,
                field_name,
                tuple(
                    _hash(item, label=f"ABM case fit {field_name} item")
                    for item in values
                ),
            )
        if len(self.observed_snapshot_hashes) != len(
            self.predicted_snapshot_hashes
        ):
            raise ValueError("ABM case fit snapshot hash counts must match")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "case_id": self.case_id,
            "split": self.split.value,
            "loss": self.loss,
            "observed_snapshot_hashes": self.observed_snapshot_hashes,
            "predicted_snapshot_hashes": self.predicted_snapshot_hashes,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ABMCandidateCalibrationFit:
    candidate: ABMCalibrationCandidate
    case_fits: tuple[ABMCaseCalibrationFit, ...]
    training_loss: float
    holdout_loss: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ABMCalibrationCandidate):
            raise TypeError("ABM candidate fit requires calibration candidate")
        if not isinstance(self.case_fits, tuple):
            raise TypeError("ABM candidate case fits must be a tuple")
        fits = tuple(self.case_fits)
        if any(not isinstance(item, ABMCaseCalibrationFit) for item in fits):
            raise TypeError("ABM candidate fit has invalid case fits")
        if any(item.candidate != self.candidate for item in fits):
            raise ValueError("ABM case fits must bind exact candidate")
        case_ids = tuple(item.case_id for item in fits)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("ABM candidate case fit ids must be unique")
        split_losses = {
            split: tuple(item.loss for item in fits if item.split is split)
            for split in CalibrationSplit
        }
        if any(not values for values in split_losses.values()):
            raise ValueError("ABM candidate fit requires TRAIN and HOLDOUT cases")
        object.__setattr__(
            self,
            "case_fits",
            tuple(sorted(fits, key=lambda item: item.case_id)),
        )
        for field_name, split in (
            ("training_loss", CalibrationSplit.TRAIN),
            ("holdout_loss", CalibrationSplit.HOLDOUT),
        ):
            actual = _finite_loss(
                getattr(self, field_name),
                label=f"ABM candidate {field_name}",
            )
            expected = fmean(split_losses[split])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"ABM candidate {field_name} must match case losses")
            object.__setattr__(self, field_name, actual)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "case_fits": [item.to_dict() for item in self.case_fits],
            "training_loss": self.training_loss,
            "holdout_loss": self.holdout_loss,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class EmpiricalABMCalibrationReport:
    base_model_hash: str
    dataset_hash: str
    weights: ABMCalibrationWeights
    candidates: tuple[ABMCalibrationCandidate, ...]
    candidate_fits: tuple[ABMCandidateCalibrationFit, ...]
    ranking: tuple[ABMCalibrationCandidate, ...]
    selected_candidate: ABMCalibrationCandidate

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_model_hash",
            _hash(self.base_model_hash, label="ABM calibration base model hash"),
        )
        object.__setattr__(
            self,
            "dataset_hash",
            _hash(self.dataset_hash, label="ABM calibration dataset hash"),
        )
        if not isinstance(self.weights, ABMCalibrationWeights):
            raise TypeError("ABM calibration report requires weights")
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ValueError("ABM calibration report requires candidate tuple")
        if any(not isinstance(item, ABMCalibrationCandidate) for item in self.candidates):
            raise TypeError("ABM calibration report has invalid candidates")
        candidate_set = set(self.candidates)
        if len(candidate_set) != len(self.candidates):
            raise ValueError("ABM calibration candidates must be unique")
        canonical_candidates = tuple(
            sorted(self.candidates, key=lambda item: item.canonical_tuple)
        )
        object.__setattr__(self, "candidates", canonical_candidates)
        if not isinstance(self.candidate_fits, tuple):
            raise TypeError("ABM calibration candidate fits must be a tuple")
        fits = tuple(self.candidate_fits)
        if any(not isinstance(item, ABMCandidateCalibrationFit) for item in fits):
            raise TypeError("ABM calibration report has invalid candidate fits")
        fit_by_candidate = {item.candidate: item for item in fits}
        if len(fit_by_candidate) != len(fits) or set(fit_by_candidate) != candidate_set:
            raise ValueError("ABM calibration fits must cover exact candidate grid")
        canonical_fits = tuple(
            sorted(fits, key=lambda item: item.candidate.canonical_tuple)
        )
        object.__setattr__(self, "candidate_fits", canonical_fits)
        expected_ranking = tuple(
            item.candidate
            for item in sorted(
                fits,
                key=lambda item: (
                    item.training_loss,
                    item.candidate.canonical_tuple,
                ),
            )
        )
        if not isinstance(self.ranking, tuple) or self.ranking != expected_ranking:
            raise ValueError("ABM calibration ranking must use training loss only")
        if self.selected_candidate != expected_ranking[0]:
            raise ValueError("ABM calibration selected candidate must lead ranking")

    def to_dict(self) -> dict[str, object]:
        return {
            "base_model_hash": self.base_model_hash,
            "dataset_hash": self.dataset_hash,
            "weights": self.weights.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "candidate_fits": [item.to_dict() for item in self.candidate_fits],
            "ranking": [item.to_dict() for item in self.ranking],
            "selected_candidate": self.selected_candidate.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _trajectory_loss(
    observed: tuple[ObservedABMSnapshot, ...],
    predicted: tuple[ObservedABMSnapshot, ...],
    weights: ABMCalibrationWeights,
) -> float:
    if len(observed) != len(predicted):
        raise ValueError("ABM calibration trajectories must have equal lengths")
    weight_map = dict(weights.metric_items)
    denominator = sum(weight_map.values()) * len(observed)
    total = 0.0
    for expected, actual in zip(observed, predicted, strict=True):
        if expected.round_index != actual.round_index:
            raise ValueError("ABM calibration trajectory rounds must match")
        expected_metrics = dict(expected.metric_items)
        actual_metrics = dict(actual.metric_items)
        total += sum(
            weight_map[name] * (actual_metrics[name] - expected_metrics[name]) ** 2
            for name in weight_map
        )
    return total / denominator


def _evaluate_case(
    base_model: DynamicRoleModel,
    candidate: ABMCalibrationCandidate,
    case: EmpiricalABMCase,
    weights: ABMCalibrationWeights,
) -> ABMCaseCalibrationFit:
    model = apply_calibration_candidate(base_model, candidate)
    try:
        initial = initialize_dynamic_role_population(
            model,
            beliefs=dict(case.initial_beliefs),
        )
        trajectory = simulate_dynamic_role_population(
            model,
            initial,
            environment_event_schedule=case.environment_event_schedule,
            truth_observation_schedule=case.truth_observation_schedule,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"candidate {candidate.canonical_tuple} cannot replay case {case.case_id!r}: {error}"
        ) from error
    predicted = tuple(
        observe_dynamic_role_state(model, item.next_state)
        for item in trajectory.rounds
    )
    return ABMCaseCalibrationFit(
        candidate,
        case.case_id,
        case.split,
        _trajectory_loss(case.observations, predicted, weights),
        tuple(item.content_hash for item in case.observations),
        tuple(item.content_hash for item in predicted),
    )


def calibrate_dynamic_role_model(
    base_model: DynamicRoleModel,
    dataset: EmpiricalABMDataset,
    *,
    candidates: tuple[ABMCalibrationCandidate, ...],
    weights: ABMCalibrationWeights,
) -> EmpiricalABMCalibrationReport:
    """Exhaustively rank a frozen V8 grid by training trajectory loss."""

    if not isinstance(base_model, DynamicRoleModel):
        raise TypeError("empirical ABM calibration requires DynamicRoleModel")
    if not isinstance(dataset, EmpiricalABMDataset):
        raise TypeError("empirical ABM calibration requires EmpiricalABMDataset")
    if not isinstance(candidates, tuple) or not candidates:
        raise ValueError("empirical ABM calibration requires candidate tuple")
    if any(not isinstance(item, ABMCalibrationCandidate) for item in candidates):
        raise TypeError("empirical ABM calibration has invalid candidates")
    if len(set(candidates)) != len(candidates):
        raise ValueError("empirical ABM calibration candidates must be unique")
    if not isinstance(weights, ABMCalibrationWeights):
        raise TypeError("empirical ABM calibration requires weights")
    catalog_ids = set(
        base_model.autonomy_model.evolving_model.base_model.network.agent_ids
    )
    for case in dataset.cases:
        if not {agent_id for agent_id, _ in case.initial_beliefs} <= catalog_ids:
            raise ValueError(
                f"empirical ABM case {case.case_id!r} has unknown initial belief agent"
            )
    canonical_candidates = tuple(
        sorted(candidates, key=lambda item: item.canonical_tuple)
    )
    candidate_fits: list[ABMCandidateCalibrationFit] = []
    for candidate in canonical_candidates:
        case_fits = tuple(
            _evaluate_case(base_model, candidate, case, weights)
            for case in dataset.cases
        )
        training = tuple(
            item.loss for item in case_fits if item.split is CalibrationSplit.TRAIN
        )
        holdout = tuple(
            item.loss for item in case_fits if item.split is CalibrationSplit.HOLDOUT
        )
        candidate_fits.append(
            ABMCandidateCalibrationFit(
                candidate,
                case_fits,
                fmean(training),
                fmean(holdout),
            )
        )
    ranking = tuple(
        item.candidate
        for item in sorted(
            candidate_fits,
            key=lambda item: (
                item.training_loss,
                item.candidate.canonical_tuple,
            ),
        )
    )
    return EmpiricalABMCalibrationReport(
        base_model.content_hash,
        dataset.content_hash,
        weights,
        canonical_candidates,
        tuple(candidate_fits),
        ranking,
        ranking[0],
    )


__all__ = (
    "observe_dynamic_role_state",
    "apply_calibration_candidate",
    "ABMCaseCalibrationFit",
    "ABMCandidateCalibrationFit",
    "EmpiricalABMCalibrationReport",
    "calibrate_dynamic_role_model",
)

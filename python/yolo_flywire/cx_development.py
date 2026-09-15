from __future__ import annotations

from dataclasses import dataclass
import math
import re
from statistics import fmean
from typing import Iterable

from .cx_protocol import CxProtocol
from .ntu_skeleton import split_for_subject_gate1


_SEEDS = (7, 11, 19, 23, 31)
_ARMS = ("real", "degree", "block")
_OBSERVATION_RATIOS = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
_EARLY_EFFECT = 0.02
_TEMPORAL_EFFECT = 0.05
_POSITIVE_SEED_COUNT = 4
_DIGEST = re.compile(r"^[0-9a-f]{64}$")

# Inner development validation is intentionally independent of the official
# X-Sub held-out side. The final-test side remains inaccessible. This subset is
# frozen for Gate 1 development and must not be tuned from task outcomes.
_VALIDATION_SUBJECTS = frozenset({2, 14, 31, 46, 55, 70, 82, 85, 94, 103})


def _finite(name: str, value: object) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


def _digest(name: str, value: object) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class CxDevelopmentBudget:
    optimizer: str
    learning_rate: float
    epochs: int
    max_updates: int
    batch_size: int
    early_stopping: str
    retention_horizon: int
    perturbation_fraction: float
    recovery_horizon: int

    def __post_init__(self) -> None:
        if self.optimizer != "adam":
            raise ValueError("optimizer must remain frozen at 'adam'")
        learning_rate = _finite("learning_rate", self.learning_rate)
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if type(self.epochs) is not int or self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if type(self.max_updates) is not int or self.max_updates <= 0:
            raise ValueError("max_updates must be positive")
        if type(self.batch_size) is not int or self.batch_size != 1:
            raise ValueError("batch_size must remain frozen at 1")
        if self.early_stopping != "best-validation-macro-f1-over-all-epochs":
            raise ValueError("early_stopping policy is not the frozen Gate 1 policy")
        if type(self.retention_horizon) is not int or self.retention_horizon <= 0:
            raise ValueError("retention_horizon must be positive")
        fraction = _finite("perturbation_fraction", self.perturbation_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("perturbation_fraction must be in (0,1)")
        if type(self.recovery_horizon) is not int or self.recovery_horizon <= 0:
            raise ValueError("recovery_horizon must be positive")


@dataclass(frozen=True)
class CxDevelopmentBinding:
    dataset_content_hash: str
    split_hash: str
    event_encoder_hash: str
    dynamics_hash: str
    input_node_fingerprint: str
    output_node_fingerprint: str
    real_graph_fingerprint: str
    degree_graph_fingerprints: tuple[tuple[int, str], ...]
    block_graph_fingerprints: tuple[tuple[int, str], ...]
    budget: CxDevelopmentBudget
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "dataset_content_hash",
            "split_hash",
            "event_encoder_hash",
            "dynamics_hash",
            "input_node_fingerprint",
            "output_node_fingerprint",
            "real_graph_fingerprint",
        ):
            _digest(name, getattr(self, name))
        if type(self.budget) is not CxDevelopmentBudget:
            raise ValueError("budget must be a CxDevelopmentBudget")
        if type(self.seeds) is not tuple or self.seeds != _SEEDS:
            raise ValueError("seeds must remain frozen at 7,11,19,23,31")
        self._validate_graph_map("degree_graph_fingerprints", self.degree_graph_fingerprints)
        self._validate_graph_map("block_graph_fingerprints", self.block_graph_fingerprints)

    def _validate_graph_map(self, name: str, value: object) -> None:
        if type(value) is not tuple or len(value) != len(self.seeds):
            raise ValueError(f"{name} must contain exactly the frozen seeds")
        seed_values: list[int] = []
        for item in value:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError(f"{name} entries must be (seed,fingerprint) pairs")
            seed, fingerprint = item
            if type(seed) is not int:
                raise ValueError(f"{name} seed must be an integer")
            _digest(f"{name}[{seed}]", fingerprint)
            seed_values.append(seed)
        if tuple(seed_values) != self.seeds:
            raise ValueError(f"{name} must follow the exact frozen seed order")

    def graph_fingerprint_for(self, arm: str, seed: int) -> str:
        if arm == "real":
            return self.real_graph_fingerprint
        if arm == "degree":
            return dict(self.degree_graph_fingerprints)[seed]
        if arm == "block":
            return dict(self.block_graph_fingerprints)[seed]
        raise ValueError(f"unknown development arm: {arm!r}")


@dataclass(frozen=True)
class CxScoreboard:
    full_macro_f1: float
    early_macro_f1_by_ratio: tuple[tuple[float, float], ...]
    early_prediction_auc: float
    retention_auc: float
    retention_t50: float
    recovery_rate: float
    median_recovery_steps: float
    spike_rate: float
    active_neuron_fraction: float
    synaptic_events_per_frame: float
    parameter_count: int

    def __post_init__(self) -> None:
        full = _finite("full_macro_f1", self.full_macro_f1)
        if not 0.0 <= full <= 1.0:
            raise ValueError("full_macro_f1 must be in [0,1]")
        if type(self.early_macro_f1_by_ratio) is not tuple:
            raise ValueError("early_macro_f1_by_ratio must be a tuple")
        ratios: list[float] = []
        for pair in self.early_macro_f1_by_ratio:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("early_macro_f1_by_ratio entries must be (ratio,macro_f1)")
            ratio = _finite("observation ratio", pair[0])
            score = _finite("early macro_f1", pair[1])
            if not 0.0 <= score <= 1.0:
                raise ValueError("early macro_f1 must be in [0,1]")
            ratios.append(ratio)
        if tuple(ratios) != _OBSERVATION_RATIOS:
            raise ValueError("early_macro_f1_by_ratio must use the frozen observation ratios")
        early_auc = _finite("early_prediction_auc", self.early_prediction_auc)
        if not 0.0 <= early_auc <= 1.0:
            raise ValueError("early_prediction_auc must be in [0,1]")
        _finite("retention_auc", self.retention_auc)
        if _finite("retention_t50", self.retention_t50) < 0.0:
            raise ValueError("retention_t50 must be non-negative")
        recovery = _finite("recovery_rate", self.recovery_rate)
        if not 0.0 <= recovery <= 1.0:
            raise ValueError("recovery_rate must be in [0,1]")
        if _finite("median_recovery_steps", self.median_recovery_steps) < 0.0:
            raise ValueError("median_recovery_steps must be non-negative")
        if _finite("spike_rate", self.spike_rate) < 0.0:
            raise ValueError("spike_rate must be non-negative")
        active = _finite("active_neuron_fraction", self.active_neuron_fraction)
        if not 0.0 <= active <= 1.0:
            raise ValueError("active_neuron_fraction must be in [0,1]")
        if _finite("synaptic_events_per_frame", self.synaptic_events_per_frame) < 0.0:
            raise ValueError("synaptic_events_per_frame must be non-negative")
        if type(self.parameter_count) is not int or self.parameter_count <= 0:
            raise ValueError("parameter_count must be a positive integer")


@dataclass(frozen=True)
class CxDevelopmentResult:
    arm: str
    seed: int
    graph_fingerprint: str
    dataset_content_hash: str
    split_hash: str
    event_encoder_hash: str
    dynamics_hash: str
    input_node_fingerprint: str
    output_node_fingerprint: str
    budget: CxDevelopmentBudget
    scoreboard: CxScoreboard

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise ValueError(f"arm must be one of {_ARMS}")
        if type(self.seed) is not int or self.seed not in _SEEDS:
            raise ValueError("seed is not a frozen Gate 1 seed")
        for name in (
            "graph_fingerprint",
            "dataset_content_hash",
            "split_hash",
            "event_encoder_hash",
            "dynamics_hash",
            "input_node_fingerprint",
            "output_node_fingerprint",
        ):
            _digest(name, getattr(self, name))
        if type(self.budget) is not CxDevelopmentBudget:
            raise ValueError("budget must be a CxDevelopmentBudget")
        if type(self.scoreboard) is not CxScoreboard:
            raise ValueError("scoreboard must be a CxScoreboard")


@dataclass(frozen=True)
class CxGateDecision:
    comparator_arm: str
    status: str
    early_mean_difference: float
    positive_early_seeds: int
    negative_early_seeds: int
    retention_mean_difference: float
    positive_retention_seeds: int
    recovery_mean_difference: float
    positive_recovery_seeds: int


@dataclass(frozen=True)
class CxGate1Summary:
    gate1a: CxGateDecision
    gate1b: CxGateDecision


def development_split_for_subject(subject: int) -> str:
    outer = split_for_subject_gate1(subject)
    if outer == "final_test":
        return "final_test"
    return "validation" if subject in _VALIDATION_SUBJECTS else "train"


def assert_split_access(protocol: CxProtocol, split: str) -> None:
    if type(protocol) is not CxProtocol:
        raise ValueError("protocol must be a CxProtocol")
    if split in ("train", "validation"):
        return
    if split == "final_test":
        if not protocol.final_test_enabled:
            raise ValueError("final test is sealed")
        return
    raise ValueError(f"unknown split: {split!r}")


def validate_comparison_results(
    binding: CxDevelopmentBinding,
    results: tuple[CxDevelopmentResult, ...],
) -> tuple[CxDevelopmentResult, ...]:
    if type(binding) is not CxDevelopmentBinding:
        raise ValueError("binding must be a CxDevelopmentBinding")
    if type(results) is not tuple or any(type(row) is not CxDevelopmentResult for row in results):
        raise ValueError("results must be a tuple of CxDevelopmentResult values")

    expected_cells = {(arm, seed) for seed in binding.seeds for arm in _ARMS}
    actual_cells = [(row.arm, row.seed) for row in results]
    if len(actual_cells) != len(expected_cells) or set(actual_cells) != expected_cells:
        raise ValueError("results must contain the complete arm/seed matrix exactly once")

    pin_names = (
        "dataset_content_hash",
        "split_hash",
        "event_encoder_hash",
        "dynamics_hash",
        "input_node_fingerprint",
        "output_node_fingerprint",
    )
    for row in results:
        for name in pin_names:
            if getattr(row, name) != getattr(binding, name):
                raise ValueError(f"comparison {name} does not match development binding")
        if row.budget != binding.budget:
            raise ValueError("comparison budget does not match development binding")
        expected_graph = binding.graph_fingerprint_for(row.arm, row.seed)
        if row.graph_fingerprint != expected_graph:
            raise ValueError("comparison graph_fingerprint does not match development binding")
    return results


def _decision(
    binding: CxDevelopmentBinding,
    rows: dict[tuple[str, int], CxDevelopmentResult],
    comparator: str,
) -> CxGateDecision:
    early_diffs: list[float] = []
    retention_diffs: list[float] = []
    recovery_diffs: list[float] = []
    for seed in binding.seeds:
        real = rows[("real", seed)].scoreboard
        null = rows[(comparator, seed)].scoreboard
        early_diffs.append(real.early_prediction_auc - null.early_prediction_auc)
        retention_diffs.append(real.retention_auc - null.retention_auc)
        recovery_diffs.append(real.recovery_rate - null.recovery_rate)

    early_mean = fmean(early_diffs)
    retention_mean = fmean(retention_diffs)
    recovery_mean = fmean(recovery_diffs)
    positive_early = sum(value > 0.0 for value in early_diffs)
    negative_early = sum(value < 0.0 for value in early_diffs)
    positive_retention = sum(value > 0.0 for value in retention_diffs)
    positive_recovery = sum(value > 0.0 for value in recovery_diffs)

    primary_pass = early_mean >= _EARLY_EFFECT and positive_early >= _POSITIVE_SEED_COUNT
    temporal_pass = (
        retention_mean >= _TEMPORAL_EFFECT and positive_retention >= _POSITIVE_SEED_COUNT
    ) or (
        recovery_mean >= _TEMPORAL_EFFECT and positive_recovery >= _POSITIVE_SEED_COUNT
    )
    reverse_primary = early_mean <= -_EARLY_EFFECT and negative_early >= _POSITIVE_SEED_COUNT
    status = "pass" if primary_pass and temporal_pass else "negative" if reverse_primary else "null"

    return CxGateDecision(
        comparator_arm=comparator,
        status=status,
        early_mean_difference=early_mean,
        positive_early_seeds=positive_early,
        negative_early_seeds=negative_early,
        retention_mean_difference=retention_mean,
        positive_retention_seeds=positive_retention,
        recovery_mean_difference=recovery_mean,
        positive_recovery_seeds=positive_recovery,
    )


def summarize_gate1(
    binding: CxDevelopmentBinding,
    results: tuple[CxDevelopmentResult, ...],
) -> CxGate1Summary:
    validated = validate_comparison_results(binding, results)
    rows = {(row.arm, row.seed): row for row in validated}
    return CxGate1Summary(
        gate1a=_decision(binding, rows, "degree"),
        gate1b=_decision(binding, rows, "block"),
    )

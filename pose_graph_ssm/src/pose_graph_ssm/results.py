from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .protocol import ExperimentProtocol
from .training import RunResult, architecture_fingerprint, training_protocol_fingerprint


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DevelopmentCell:
    run: RunResult
    binding_fingerprint: str
    dataset_content_hash: str
    split_hash: str
    feature_stats_hash: str
    training_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.run) is not RunResult:
            raise ValueError("development cell run must be a RunResult")
        for name in (
            "binding_fingerprint",
            "dataset_content_hash",
            "split_hash",
            "feature_stats_hash",
            "training_fingerprint",
        ):
            value = getattr(self, name)
            if type(value) is not str or _DIGEST.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class DevelopmentSummary:
    cell_count: int
    primary_pass: bool
    graph_contribution_pass: bool
    ssm_contribution_pass: bool
    disposition: str
    primary_early_mean: float
    primary_early_positive_seeds: int
    retention_mean: float
    retention_positive_seeds: int
    recovery_mean: float
    recovery_positive_seeds: int
    graph_early_mean: float
    graph_positive_seeds: int
    ssm_early_mean: float
    ssm_positive_seeds: int


def require_development_split(split: str) -> str:
    if split == "final_test":
        raise ValueError("final test is sealed")
    if split not in ("train", "validation"):
        raise ValueError(f"unsupported development split: {split!r}")
    return split


def _paired_differences(
    cells: dict[tuple[str, int], DevelopmentCell],
    protocol: ExperimentProtocol,
    left: str,
    right: str,
    field: str,
) -> tuple[float, ...]:
    result: list[float] = []
    for seed in protocol.seeds:
        left_run = cells[(left, seed)].run
        right_run = cells[(right, seed)].run
        result.append(float(getattr(left_run, field)) - float(getattr(right_run, field)))
    return tuple(result)


def _mean(values: tuple[float, ...]) -> float:
    return float(sum(values) / len(values))


def _positive(values: tuple[float, ...]) -> int:
    return sum(value > 0.0 for value in values)


def _effect_pass(values: tuple[float, ...], threshold: float, positive_seed_count: int) -> bool:
    return _mean(values) >= threshold and _positive(values) >= positive_seed_count


def _validate_matrix(cells: Iterable[DevelopmentCell], protocol: ExperimentProtocol) -> dict[tuple[str, int], DevelopmentCell]:
    values = list(cells)
    expected_count = len(protocol.model_kinds) * len(protocol.seeds)
    if len(values) != expected_count:
        raise ValueError(f"development matrix must contain exactly {expected_count} cells")

    matrix: dict[tuple[str, int], DevelopmentCell] = {}
    for cell in values:
        if type(cell) is not DevelopmentCell:
            raise ValueError("development matrix contains a non-cell value")
        key = (cell.run.model_kind, cell.run.seed)
        if key in matrix:
            raise ValueError(f"duplicate development matrix cell: {key}")
        matrix[key] = cell

    expected_keys = {(kind, seed) for kind in protocol.model_kinds for seed in protocol.seeds}
    if set(matrix) != expected_keys:
        raise ValueError("development matrix does not match frozen model/seed roster")

    reference = values[0]
    for cell in values[1:]:
        if cell.binding_fingerprint != reference.binding_fingerprint:
            raise ValueError("binding fingerprint drift across development matrix")
        if cell.dataset_content_hash != reference.dataset_content_hash:
            raise ValueError("dataset-content pin drift across development matrix")
        if cell.split_hash != reference.split_hash:
            raise ValueError("split pin drift across development matrix")
        if cell.feature_stats_hash != reference.feature_stats_hash:
            raise ValueError("feature-statistics pin drift across development matrix")
        if cell.training_fingerprint != reference.training_fingerprint:
            raise ValueError("training pin drift across development matrix")

    expected_training = training_protocol_fingerprint(protocol)
    if reference.training_fingerprint != expected_training:
        raise ValueError("training fingerprint does not match frozen protocol")

    for (kind, seed), cell in matrix.items():
        run = cell.run
        if run.architecture_fingerprint != architecture_fingerprint(kind, protocol):
            raise ValueError(f"architecture fingerprint mismatch for {kind}")
        if run.parameter_count > protocol.parameter_ceiling:
            raise ValueError(f"parameter ceiling exceeded for {kind}")
        if run.update_count <= 0 or run.update_count > protocol.training.max_updates:
            raise ValueError(f"invalid update count for {kind}/{seed}")

    for seed in protocol.seeds:
        seed_cells = [matrix[(kind, seed)] for kind in protocol.model_kinds]
        orders = {cell.run.sample_order_fingerprint for cell in seed_cells}
        if len(orders) != 1:
            raise ValueError(f"sample-order pin drift for seed {seed}")
        updates = {cell.run.update_count for cell in seed_cells}
        if len(updates) != 1:
            raise ValueError(f"update-budget drift for seed {seed}")

    return matrix


def reduce_development_matrix(
    cells: Iterable[DevelopmentCell],
    protocol: ExperimentProtocol,
) -> DevelopmentSummary:
    if type(protocol) is not ExperimentProtocol:
        raise ValueError("protocol must be an ExperimentProtocol")
    matrix = _validate_matrix(cells, protocol)

    primary_early = _paired_differences(matrix, protocol, "graph_ssm", "gru", "early_prediction_auc")
    retention = _paired_differences(matrix, protocol, "graph_ssm", "gru", "retention_auc")
    recovery = _paired_differences(matrix, protocol, "graph_ssm", "gru", "dropout_recovery_rate")
    graph_early = _paired_differences(matrix, protocol, "graph_ssm", "ssm_only", "early_prediction_auc")
    ssm_early = _paired_differences(matrix, protocol, "graph_ssm", "graph_tcn", "early_prediction_auc")

    primary_early_pass = _effect_pass(primary_early, protocol.primary_early_effect, protocol.positive_seed_count)
    retention_pass = _effect_pass(retention, protocol.temporal_effect, protocol.positive_seed_count)
    recovery_pass = _effect_pass(recovery, protocol.temporal_effect, protocol.positive_seed_count)
    primary_pass = primary_early_pass and (retention_pass or recovery_pass)
    graph_pass = _effect_pass(graph_early, protocol.attribution_effect, protocol.positive_seed_count)
    ssm_pass = _effect_pass(ssm_early, protocol.attribution_effect, protocol.positive_seed_count)

    if not primary_pass:
        disposition = "stop_primary_null"
    elif graph_pass and ssm_pass:
        disposition = "continue_gate2"
    elif not graph_pass and ssm_pass:
        disposition = "primary_pass_no_graph_attribution"
    elif graph_pass and not ssm_pass:
        disposition = "primary_pass_no_ssm_attribution"
    else:
        disposition = "primary_pass_no_specific_attribution"

    return DevelopmentSummary(
        cell_count=len(matrix),
        primary_pass=primary_pass,
        graph_contribution_pass=graph_pass,
        ssm_contribution_pass=ssm_pass,
        disposition=disposition,
        primary_early_mean=_mean(primary_early),
        primary_early_positive_seeds=_positive(primary_early),
        retention_mean=_mean(retention),
        retention_positive_seeds=_positive(retention),
        recovery_mean=_mean(recovery),
        recovery_positive_seeds=_positive(recovery),
        graph_early_mean=_mean(graph_early),
        graph_positive_seeds=_positive(graph_early),
        ssm_early_mean=_mean(ssm_early),
        ssm_positive_seeds=_positive(ssm_early),
    )

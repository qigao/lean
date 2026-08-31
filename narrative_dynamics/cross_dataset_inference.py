"""TRAIN-only base rates and aggregate paired inference for transfer V1."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import re
from typing import Mapping

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.cross_dataset_capabilities import TrainProjection


_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_BOOTSTRAP_SEED = 43001
_BOOTSTRAP_REPLICATES = 10000
_THRESHOLD = 0.01
_PERCENTILE_RULE = "linear_order_statistic_v1"
_SCORES = frozenset({"BRIER", "CLIPPED_LOG"})
_AGGREGATIONS = frozenset({"PARTICIPANT_EQUAL", "TRIAL_EQUAL"})


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical sha256:<64 lowercase hex>")
    return value


def _finite(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return number


@dataclass(frozen=True)
class FrozenBaseRateCell:
    source_stratum: str
    action_0_count: int
    action_1_count: int
    probability_action_0: float
    probability_action_1: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_stratum, str)
            or not self.source_stratum
            or self.source_stratum != self.source_stratum.strip()
        ):
            raise ValueError("source_stratum must be non-empty trimmed text")
        for field_name in ("action_0_count", "action_1_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.action_0_count + self.action_1_count < 1:
            raise ValueError("base-rate cell must contain at least one TRAIN row")
        p0 = _finite(self.probability_action_0, label="probability_action_0")
        p1 = _finite(self.probability_action_1, label="probability_action_1")
        expected_p1 = (self.action_1_count + 1) / (
            self.action_0_count + self.action_1_count + 2
        )
        if p1 != expected_p1 or p0 != 1.0 - expected_p1:
            raise ValueError("base-rate probabilities do not use Laplace-one smoothing")
        object.__setattr__(self, "probability_action_0", p0)
        object.__setattr__(self, "probability_action_1", p1)

    def to_payload(self) -> dict[str, object]:
        return {
            "source_stratum": self.source_stratum,
            "action_0_count": self.action_0_count,
            "action_1_count": self.action_1_count,
            "probability_action_0": self.probability_action_0,
            "probability_action_1": self.probability_action_1,
        }


@dataclass(frozen=True)
class FrozenBaseRateComparator:
    train_projection_hash: str
    cells: tuple[FrozenBaseRateCell, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "train_projection_hash",
            _hash(self.train_projection_hash, label="train_projection_hash"),
        )
        cells = tuple(self.cells)
        if not cells or not all(isinstance(cell, FrozenBaseRateCell) for cell in cells):
            raise ValueError("base-rate cells must contain FrozenBaseRateCell values")
        ordered = tuple(sorted(cells, key=lambda cell: cell.source_stratum))
        if len({cell.source_stratum for cell in ordered}) != len(ordered):
            raise ValueError("base-rate source strata must be unique")
        object.__setattr__(self, "cells", ordered)

    def probabilities_for(self, source_stratum: str) -> tuple[float, float]:
        matches = tuple(
            cell for cell in self.cells if cell.source_stratum == source_stratum
        )
        if len(matches) != 1:
            raise KeyError("unknown source stratum; no baseline fallback is permitted")
        cell = matches[0]
        return cell.probability_action_0, cell.probability_action_1

    def identity_payload(self) -> dict[str, object]:
        return {
            "train_projection_hash": self.train_projection_hash,
            "cells": [cell.to_payload() for cell in self.cells],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True, repr=False)
class _ParticipantLossBlock:
    token: str = field(repr=False)
    losses: tuple[float, ...] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token", _hash(self.token, label="private token"))
        if not isinstance(self.losses, (tuple, list)):
            raise TypeError("private losses must be a sequence")
        losses = tuple(
            _finite(value, label="private loss", minimum=0.0)
            for value in self.losses
        )
        if not losses:
            raise ValueError("private loss block must be non-empty")
        object.__setattr__(self, "losses", losses)

    @property
    def mean_loss(self) -> float:
        return math.fsum(self.losses) / len(self.losses)

    def __repr__(self) -> str:
        return "_ParticipantLossBlock(token=<redacted>, losses=<redacted>)"

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("participant-level loss blocks are not serializable")


def _blocks(value: object) -> tuple[_ParticipantLossBlock, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("participant loss blocks must be a sequence")
    blocks = tuple(value)
    if not blocks or not all(isinstance(block, _ParticipantLossBlock) for block in blocks):
        raise ValueError("participant loss blocks must be non-empty and private")
    tokens = tuple(block.token for block in blocks)
    if len(tokens) != len(set(tokens)):
        raise ValueError("participant loss blocks contain a duplicate token")
    return tuple(sorted(blocks, key=lambda block: block.token))


def fit_train_base_rate(train: TrainProjection) -> FrozenBaseRateComparator:
    if not isinstance(train, TrainProjection):
        raise TypeError("train must be TrainProjection")
    counts: dict[str, list[int]] = {}
    for row in train.consume_for_refit():
        cell = counts.setdefault(row.source_stratum, [0, 0])
        cell[0 if row.first_stage_action == "action_0" else 1] += 1
    cells = tuple(
        FrozenBaseRateCell(
            source_stratum=stratum,
            action_0_count=counts_by_action[0],
            action_1_count=counts_by_action[1],
            probability_action_1=(counts_by_action[1] + 1)
            / (sum(counts_by_action) + 2),
            probability_action_0=1.0
            - (counts_by_action[1] + 1) / (sum(counts_by_action) + 2),
        )
        for stratum, counts_by_action in sorted(counts.items())
    )
    return FrozenBaseRateComparator(
        train_projection_hash=train.projection_hash,
        cells=cells,
    )


def participant_equal_loss(blocks: object) -> float:
    canonical = _blocks(blocks)
    return math.fsum(block.mean_loss for block in canonical) / len(canonical)


def trial_equal_loss(blocks: object) -> float:
    canonical = _blocks(blocks)
    losses = tuple(loss for block in canonical for loss in block.losses)
    return math.fsum(losses) / len(losses)


def relative_improvement(candidate_loss: float, comparator_loss: float) -> float:
    candidate = _finite(candidate_loss, label="candidate_loss", minimum=0.0)
    comparator = _finite(comparator_loss, label="comparator_loss", minimum=0.0)
    if comparator <= 0.0:
        raise ValueError("comparator loss must be positive")
    return (comparator - candidate) / comparator


def _linear_percentile(values: tuple[float, ...], q: float) -> float:
    if not values:
        raise ValueError("percentile values must be non-empty")
    quantile = _finite(q, label="percentile quantile")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile quantile must be in [0, 1]")
    ordered = sorted(_finite(value, label="percentile value") for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass(frozen=True)
class TransferInferenceEvidence:
    candidate_identity: str
    comparator_identity: str
    score: str
    aggregation: str
    candidate_loss: float
    comparator_loss: float
    relative_improvement: float
    lower_95: float
    upper_95: float
    bootstrap_seed: int
    bootstrap_replicates: int
    percentile_rule: str
    participant_count: int
    threshold: float
    pass_status: bool
    source_stratum_identity: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("candidate_identity", "comparator_identity"):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=field_name),
            )
        if self.score not in _SCORES:
            raise ValueError("unknown transfer score")
        if self.aggregation not in _AGGREGATIONS:
            raise ValueError("unknown transfer aggregation")
        for field_name in (
            "candidate_loss",
            "comparator_loss",
            "relative_improvement",
            "lower_95",
            "upper_95",
            "threshold",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite(getattr(self, field_name), label=field_name),
            )
        if self.bootstrap_seed != _BOOTSTRAP_SEED:
            raise ValueError("bootstrap seed changed from frozen protocol")
        if self.bootstrap_replicates != _BOOTSTRAP_REPLICATES:
            raise ValueError("bootstrap replicate count changed from frozen protocol")
        if self.percentile_rule != _PERCENTILE_RULE:
            raise ValueError("bootstrap percentile rule changed")
        if (
            isinstance(self.participant_count, bool)
            or not isinstance(self.participant_count, int)
            or self.participant_count < 1
        ):
            raise ValueError("participant_count must be a positive integer")
        if self.threshold != _THRESHOLD:
            raise ValueError("relative-improvement threshold changed")
        expected_pass = (
            self.relative_improvement >= self.threshold and self.lower_95 > 0.0
        )
        if self.pass_status is not expected_pass:
            raise ValueError("inference pass status does not follow frozen rule")
        if self.source_stratum_identity is not None:
            object.__setattr__(
                self,
                "source_stratum_identity",
                _hash(
                    self.source_stratum_identity,
                    label="source_stratum_identity",
                ),
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate_identity": self.candidate_identity,
            "comparator_identity": self.comparator_identity,
            "score": self.score,
            "aggregation": self.aggregation,
            "candidate_loss": self.candidate_loss,
            "comparator_loss": self.comparator_loss,
            "relative_improvement": self.relative_improvement,
            "lower_95": self.lower_95,
            "upper_95": self.upper_95,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_replicates": self.bootstrap_replicates,
            "percentile_rule": self.percentile_rule,
            "participant_count": self.participant_count,
            "threshold": self.threshold,
            "pass_status": self.pass_status,
            "source_stratum_identity": self.source_stratum_identity,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_payload())


def _loss_for_indices(
    blocks: tuple[_ParticipantLossBlock, ...],
    indices: tuple[int, ...],
    aggregation: str,
) -> float:
    selected = tuple(blocks[index] for index in indices)
    if aggregation == "PARTICIPANT_EQUAL":
        return math.fsum(block.mean_loss for block in selected) / len(selected)
    losses = tuple(loss for block in selected for loss in block.losses)
    return math.fsum(losses) / len(losses)


def paired_participant_bootstrap(
    *,
    candidate: tuple[_ParticipantLossBlock, ...],
    comparator: tuple[_ParticipantLossBlock, ...],
    score: str,
    aggregation: str,
    candidate_identity: str,
    comparator_identity: str,
    seed: int,
    replicates: int,
    source_stratum_identity: str | None = None,
) -> TransferInferenceEvidence:
    candidate_blocks = _blocks(candidate)
    comparator_blocks = _blocks(comparator)
    if tuple(block.token for block in candidate_blocks) != tuple(
        block.token for block in comparator_blocks
    ):
        raise ValueError("paired blocks must contain the same private tokens")
    if any(
        len(candidate_block.losses) != len(comparator_block.losses)
        for candidate_block, comparator_block in zip(
            candidate_blocks,
            comparator_blocks,
            strict=True,
        )
    ):
        raise ValueError("paired blocks must contain complete matching trial rows")
    if score not in _SCORES:
        raise ValueError("score must be BRIER or CLIPPED_LOG")
    if aggregation not in _AGGREGATIONS:
        raise ValueError("aggregation must be PARTICIPANT_EQUAL or TRIAL_EQUAL")
    if seed != _BOOTSTRAP_SEED or replicates != _BOOTSTRAP_REPLICATES:
        raise ValueError("bootstrap seed and replicates are frozen")

    aggregate = (
        participant_equal_loss
        if aggregation == "PARTICIPANT_EQUAL"
        else trial_equal_loss
    )
    candidate_loss = aggregate(candidate_blocks)
    comparator_loss = aggregate(comparator_blocks)
    point = relative_improvement(candidate_loss, comparator_loss)

    rng = random.Random(seed)
    count = len(candidate_blocks)
    improvements: list[float] = []
    for _ in range(replicates):
        indices = tuple(rng.randrange(count) for _ in range(count))
        sampled_candidate = _loss_for_indices(
            candidate_blocks,
            indices,
            aggregation,
        )
        sampled_comparator = _loss_for_indices(
            comparator_blocks,
            indices,
            aggregation,
        )
        improvements.append(
            relative_improvement(sampled_candidate, sampled_comparator)
        )
    lower = _linear_percentile(tuple(improvements), 0.025)
    upper = _linear_percentile(tuple(improvements), 0.975)
    return TransferInferenceEvidence(
        candidate_identity=candidate_identity,
        comparator_identity=comparator_identity,
        score=score,
        aggregation=aggregation,
        candidate_loss=candidate_loss,
        comparator_loss=comparator_loss,
        relative_improvement=point,
        lower_95=lower,
        upper_95=upper,
        bootstrap_seed=seed,
        bootstrap_replicates=replicates,
        percentile_rule=_PERCENTILE_RULE,
        participant_count=count,
        threshold=_THRESHOLD,
        pass_status=point >= _THRESHOLD and lower > 0.0,
        source_stratum_identity=source_stratum_identity,
    )

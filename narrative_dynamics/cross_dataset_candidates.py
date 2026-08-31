"""Exact zero-shot and complete-grid refit candidate freezes for transfer V1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Mapping

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.cross_dataset_capabilities import (
    SelectionProjection,
    TrainProjection,
)
from narrative_dynamics.studies.feher_hare_measurement_validity_v1 import (
    FEHER_HARE_R3_LOCK_COMMIT,
    FeherHareMeasurementAnchor,
)


_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_TRAIN_SEEDS = (101, 102)
_SELECTION_SEEDS = (201, 202)
_TIE_BREAK = "lexical_parameters_v1"


class TransferFamily(str, Enum):
    REACTIVE = "reactive"
    INTENTIONAL = "intentional"
    PLANNING = "planning"


REACTIVE_GRID = tuple((("beta", beta),) for beta in (0.5, 1.0, 2.0, 4.0))
MEMORY_GRID = tuple(
    (("beta", beta), ("memory_decay", decay))
    for beta in (0.5, 1.0, 2.0, 4.0)
    for decay in (0.5, 0.75, 0.9, 1.0)
)
INTENTIONAL_GRID = MEMORY_GRID
PLANNING_GRID = MEMORY_GRID


_R3_IDENTITY = {
    "lock_commit": FEHER_HARE_R3_LOCK_COMMIT,
    "scientific_repository_revision": "d01232979cdfc9d902daab5f9e3e937079b56f69",
    "upstream_revision": "4567763780a2c596fd6510af720ec468a8214a8f",
    "source_manifest_hash": "sha256:3609e980af172823cfef78290e7f2337fdb337d67f1f27641e17d473aeedd10d",
    "source_snapshot_hash": "sha256:25bdc2e4bff4110f38b098b89e7d59aa38c9658727fcc89a38d50e107d243186",
    "transform_hash": "sha256:2fb8ab6dc796a9e4ece5653a5485d865ef44dd0f5f7dff92d94a904932d6e541",
    "participant_assignment_hash": "sha256:fc144c35f6713e27f75141d678872cc04ca44c7a0fd8e109e8618c72b4111ccf",
    "dataset_hash": "sha256:17789130372d7eace05e1216a57bdae2ffbd519960333ffe814aee2d2d404781",
    "target_spec_hash": "sha256:3134c9dc424418c87379f8e451420fb2defe81b8f30a45d67cd9b1f453349713",
    "train_partition_hash": "sha256:71ce56243338eb23b9dd5ad6dd901e4b0d0be4989742b66bbe7ea4f025f207da",
    "selection_partition_hash": "sha256:ecdcfa1681b58888ebf4419372d5de7b75be9624cdd3307144371c5486eb1347",
    "excluded_final_partition_hash": "sha256:936ffe872644e888111007b300ea847484698be8fbfd7c2d54a177505a411047",
    "excluded_final_target_hash": "sha256:927e1727d37993e9a5c887f79ac8712d07a155622deacf0a3122d41f90b16ed2",
    "train_selection_freeze_hash": "sha256:77fa1bb80ab0c7ac737a09a8001f1d0c1432ce3bd991fd7191d07fdd0d8e05d5",
    "internal_lock_bundle_hash": "sha256:96e553e557d6e7314eb0b9b1d0aaa8696e01d7a6ddec0abde949be8c8d45602f",
}
_R3_CANDIDATES = (
    (
        TransferFamily.REACTIVE,
        (("beta", 0.5),),
        "sha256:5b607a4bc0a8082809c2446a8248fb8659b0d9ba178687ddad96dc74e3f19622",
        "sha256:e8414e301d19fc6acfd5accf055402ac995790f785402186c67ff95a48ee0c2e",
        "sha256:42a84ef10c207159ff98397fe2bd86594df6c8be0f41b98e876f7ae33cba9456",
    ),
    (
        TransferFamily.INTENTIONAL,
        (("beta", 2.0), ("memory_decay", 0.5)),
        "sha256:47057311fe7450469c1710be49ba0e3b42aa7416fa2129f40761dbf8d237d4fc",
        "sha256:498a512cd931afe276f78bf135bd5c8269e051920d4b876177d0aa9ea250806d",
        "sha256:68a01b5496e20095c1a603b30f1384bbe6a3ea22aac8cf74b7b97463c4935839",
    ),
    (
        TransferFamily.PLANNING,
        (("beta", 4.0), ("memory_decay", 0.75)),
        "sha256:ef0f3caf4f2a02b2d719bc302d7409fc8c2d937eb13517d906d83b38c29b76ac",
        "sha256:e347a3d7d523e2a35d0bb6b0f662343f5a01ddd1ff7d2efe95a8a5efa1986e74",
        "sha256:c0ed956c1b285153075e5a22e7fb51f6d53dcf8ad326cce82e39e0221c2e877c",
    ),
)
_LINEAGE_FIELDS = (
    "source_identity_hash",
    "split_manifest_hash",
    "train_projection_hash",
    "selection_projection_hash",
)


def _strict_fields(
    payload: object,
    *,
    expected: tuple[str, ...],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must be a mapping")
    actual = frozenset(payload)
    required = frozenset(expected)
    if actual != required:
        missing = tuple(sorted(required - actual))
        unknown = tuple(sorted(actual - required))
        raise ValueError(
            f"{label} fields mismatch: missing={missing!r}, unknown={unknown!r}"
        )
    return payload


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical sha256:<64 lowercase hex>")
    return value


def _family(value: object) -> TransferFamily:
    if isinstance(value, TransferFamily):
        return value
    try:
        return TransferFamily(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown transfer family") from exc


def _parameters(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("parameters must be a sequence")
    rows: list[tuple[str, float]] = []
    for index, row in enumerate(value):
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError(f"parameters[{index}] must contain two fields")
        name, raw_number = row
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("parameter names must be non-empty trimmed text")
        if isinstance(raw_number, bool) or not isinstance(raw_number, (int, float)):
            raise TypeError("parameter values must be numeric")
        number = float(raw_number)
        if not math.isfinite(number):
            raise ValueError("parameter values must be finite")
        rows.append((name, number))
    canonical = tuple(rows)
    if not canonical or len({name for name, _ in canonical}) != len(canonical):
        raise ValueError("parameters must be non-empty with unique names")
    if tuple(name for name, _ in canonical) not in {
        ("beta",),
        ("beta", "memory_decay"),
    }:
        raise ValueError("parameter schema is unsupported")
    return canonical


def _seeds(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a sequence")
    seeds = tuple(value)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError(f"{label} must contain integer seeds")
    return seeds


@dataclass(frozen=True)
class FrozenTransferCandidate:
    family: TransferFamily
    parameters: tuple[tuple[str, float], ...]
    source_candidate_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _family(self.family))
        object.__setattr__(self, "parameters", _parameters(self.parameters))
        object.__setattr__(
            self,
            "source_candidate_hash",
            _hash(self.source_candidate_hash, label="source_candidate_hash"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "parameters": [list(row) for row in self.parameters],
            "source_candidate_hash": self.source_candidate_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_payload())


@dataclass(frozen=True)
class ZeroShotCandidateFreeze:
    r3_anchor_hash: str
    new_source_lineage: tuple[tuple[str, str], ...]
    candidates: tuple[FrozenTransferCandidate, ...]
    new_data_training: bool = False
    new_data_selection: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "r3_anchor_hash", _hash(self.r3_anchor_hash, label="r3_anchor_hash"))
        lineage = tuple(self.new_source_lineage)
        if tuple(name for name, _ in lineage) != _LINEAGE_FIELDS:
            raise ValueError("new-source lineage fields changed")
        object.__setattr__(
            self,
            "new_source_lineage",
            tuple((name, _hash(value, label=name)) for name, value in lineage),
        )
        candidates = tuple(self.candidates)
        if tuple(row.family for row in candidates) != tuple(TransferFamily):
            raise ValueError("zero-shot candidate family set changed")
        object.__setattr__(self, "candidates", candidates)
        if self.new_data_training is not False or self.new_data_selection is not False:
            raise ValueError("zero-shot freeze forbids new-data fitting or selection")

    def identity_payload(self) -> dict[str, object]:
        return {
            "r3_anchor_hash": self.r3_anchor_hash,
            "new_source_lineage": [list(row) for row in self.new_source_lineage],
            "candidates": [row.to_payload() for row in self.candidates],
            "new_data_training": self.new_data_training,
            "new_data_selection": self.new_data_selection,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


@dataclass(frozen=True)
class GridCandidateEvaluation:
    family: TransferFamily
    parameters: tuple[tuple[str, float], ...]
    train_projection_hash: str
    selection_projection_hash: str
    target_spec_hash: str
    split_manifest_hash: str
    train_seeds: tuple[int, ...]
    selection_seeds: tuple[int, ...]
    score: str
    brier_loss_identity: str
    builder_identity: str
    simulation_identity: str
    selection_brier_loss: float
    evaluation_receipt_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _family(self.family))
        object.__setattr__(self, "parameters", _parameters(self.parameters))
        for field_name in (
            "train_projection_hash",
            "selection_projection_hash",
            "target_spec_hash",
            "split_manifest_hash",
            "brier_loss_identity",
            "builder_identity",
            "simulation_identity",
            "evaluation_receipt_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), label=field_name),
            )
        object.__setattr__(self, "train_seeds", _seeds(self.train_seeds, label="TRAIN seeds"))
        object.__setattr__(
            self,
            "selection_seeds",
            _seeds(self.selection_seeds, label="SELECTION seeds"),
        )
        if not isinstance(self.score, str) or not self.score:
            raise ValueError("score must be non-empty text")
        if isinstance(self.selection_brier_loss, bool) or not isinstance(
            self.selection_brier_loss,
            (int, float),
        ):
            raise TypeError("selection_brier_loss must be numeric")
        loss = float(self.selection_brier_loss)
        if not math.isfinite(loss):
            raise ValueError("selection_brier_loss must be finite")
        if loss < 0.0:
            raise ValueError("selection_brier_loss must be non-negative")
        object.__setattr__(self, "selection_brier_loss", loss)

    def to_payload(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "parameters": [list(row) for row in self.parameters],
            "train_projection_hash": self.train_projection_hash,
            "selection_projection_hash": self.selection_projection_hash,
            "target_spec_hash": self.target_spec_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "train_seeds": list(self.train_seeds),
            "selection_seeds": list(self.selection_seeds),
            "score": self.score,
            "brier_loss_identity": self.brier_loss_identity,
            "builder_identity": self.builder_identity,
            "simulation_identity": self.simulation_identity,
            "selection_brier_loss": self.selection_brier_loss,
            "evaluation_receipt_hash": self.evaluation_receipt_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_payload())

    @classmethod
    def from_payload(cls, payload: object) -> GridCandidateEvaluation:
        expected = (
            "family",
            "parameters",
            "train_projection_hash",
            "selection_projection_hash",
            "target_spec_hash",
            "split_manifest_hash",
            "train_seeds",
            "selection_seeds",
            "score",
            "brier_loss_identity",
            "builder_identity",
            "simulation_identity",
            "selection_brier_loss",
            "evaluation_receipt_hash",
        )
        values = _strict_fields(payload, expected=expected, label="grid evaluation")
        return cls(
            family=values["family"],
            parameters=tuple(tuple(row) for row in values["parameters"]),
            train_projection_hash=values["train_projection_hash"],
            selection_projection_hash=values["selection_projection_hash"],
            target_spec_hash=values["target_spec_hash"],
            split_manifest_hash=values["split_manifest_hash"],
            train_seeds=tuple(values["train_seeds"]),
            selection_seeds=tuple(values["selection_seeds"]),
            score=values["score"],
            brier_loss_identity=values["brier_loss_identity"],
            builder_identity=values["builder_identity"],
            simulation_identity=values["simulation_identity"],
            selection_brier_loss=values["selection_brier_loss"],
            evaluation_receipt_hash=values["evaluation_receipt_hash"],
        )


@dataclass(frozen=True)
class RefitCandidateFreeze:
    train_projection_hash: str
    selection_projection_hash: str
    target_spec_hash: str
    split_manifest_hash: str
    brier_loss_identity: str
    builder_identities: tuple[tuple[str, str], ...]
    simulation_identity: str
    tie_break_identity: str
    evaluations: tuple[GridCandidateEvaluation, ...]
    selected_candidates: tuple[FrozenTransferCandidate, ...]
    final_projection_opened: bool = False
    final_model_execution: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "train_projection_hash",
            "selection_projection_hash",
            "target_spec_hash",
            "split_manifest_hash",
            "brier_loss_identity",
            "simulation_identity",
        ):
            object.__setattr__(self, field_name, _hash(getattr(self, field_name), label=field_name))
        if self.tie_break_identity != _TIE_BREAK:
            raise ValueError("refit tie-break identity changed")
        if self.final_projection_opened is not False or self.final_model_execution is not False:
            raise ValueError("refit freeze cannot contain FINAL execution state")

    def identity_payload(self) -> dict[str, object]:
        return {
            "train_projection_hash": self.train_projection_hash,
            "selection_projection_hash": self.selection_projection_hash,
            "target_spec_hash": self.target_spec_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "brier_loss_identity": self.brier_loss_identity,
            "builder_identities": [list(row) for row in self.builder_identities],
            "simulation_identity": self.simulation_identity,
            "tie_break_identity": self.tie_break_identity,
            "evaluations": [row.to_payload() for row in self.evaluations],
            "selected_candidates": [row.to_payload() for row in self.selected_candidates],
            "final_projection_opened": self.final_projection_opened,
            "final_model_execution": self.final_model_execution,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())


def _validate_r3_anchor(anchor: object) -> FeherHareMeasurementAnchor:
    if not isinstance(anchor, FeherHareMeasurementAnchor):
        raise TypeError("anchor must be FeherHareMeasurementAnchor")
    for field_name, expected in _R3_IDENTITY.items():
        if getattr(anchor, field_name) != expected:
            raise ValueError(f"R3 anchor {field_name} changed")
    actual_candidates = tuple(
        (
            _family(row.family),
            row.parameters,
            row.training_manifest_hash,
            row.selection_manifest_hash,
            row.candidate_hash,
        )
        for row in anchor.candidate_rows
    )
    if actual_candidates != _R3_CANDIDATES:
        raise ValueError("R3 anchor candidates changed")
    return anchor


def freeze_zero_shot_candidates(
    anchor: FeherHareMeasurementAnchor,
    new_source_lineage: Mapping[str, object],
) -> ZeroShotCandidateFreeze:
    canonical_anchor = _validate_r3_anchor(anchor)
    lineage = _strict_fields(
        new_source_lineage,
        expected=_LINEAGE_FIELDS,
        label="new-source lineage fields",
    )
    canonical_lineage = tuple(
        (field_name, _hash(lineage[field_name], label=field_name))
        for field_name in _LINEAGE_FIELDS
    )
    candidates = tuple(
        FrozenTransferCandidate(
            family=family,
            parameters=parameters,
            source_candidate_hash=candidate_hash,
        )
        for family, parameters, _train, _selection, candidate_hash in _R3_CANDIDATES
    )
    return ZeroShotCandidateFreeze(
        r3_anchor_hash=stable_content_hash(canonical_anchor.to_payload()),
        new_source_lineage=canonical_lineage,
        candidates=candidates,
    )


def _canonical_builders(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("builder_identities must be a sequence")
    rows = tuple(value)
    try:
        mapping = {str(name): _hash(identity, label=f"{name} builder identity") for name, identity in rows}
    except (TypeError, ValueError) as exc:
        raise ValueError("builder identities are invalid") from exc
    expected_names = tuple(family.value for family in TransferFamily)
    if len(mapping) != len(rows) or frozenset(mapping) != frozenset(expected_names):
        raise ValueError("builder identities must cover each transfer family exactly")
    return tuple((name, mapping[name]) for name in expected_names)


def _expected_grid(family: TransferFamily) -> tuple[tuple[tuple[str, float], ...], ...]:
    if family is TransferFamily.REACTIVE:
        return REACTIVE_GRID
    if family is TransferFamily.INTENTIONAL:
        return INTENTIONAL_GRID
    return PLANNING_GRID


def _select_candidate(rows: tuple[GridCandidateEvaluation, ...]) -> GridCandidateEvaluation:
    return min(rows, key=lambda row: (row.selection_brier_loss, row.parameters))


def freeze_refit_candidates(
    *,
    train: TrainProjection,
    selection: SelectionProjection,
    evaluations: tuple[GridCandidateEvaluation, ...],
    brier_loss_identity: str,
    builder_identities: tuple[tuple[str, str], ...],
    simulation_identity: str,
    tie_break_identity: str,
    target_spec_hash: str,
    split_manifest_hash: str,
    final_projection: object | None = None,
    prior_final_result: object | None = None,
) -> RefitCandidateFreeze:
    if final_projection is not None or prior_final_result is not None:
        raise ValueError("refit candidate freeze rejects every FINAL input")
    if not isinstance(train, TrainProjection):
        raise TypeError("train must be TrainProjection")
    if not isinstance(selection, SelectionProjection):
        raise TypeError("selection must be SelectionProjection")
    loss_identity = _hash(brier_loss_identity, label="brier_loss_identity")
    simulation_hash = _hash(simulation_identity, label="simulation_identity")
    target_hash = _hash(target_spec_hash, label="target_spec_hash")
    split_hash = _hash(split_manifest_hash, label="split_manifest_hash")
    if tie_break_identity != _TIE_BREAK:
        raise ValueError("refit tie-break identity must be lexical_parameters_v1")
    builders = _canonical_builders(builder_identities)
    builder_map = dict(builders)
    if not isinstance(evaluations, (tuple, list)):
        raise TypeError("evaluations must be a sequence")
    rows = tuple(evaluations)
    if not rows or not all(isinstance(row, GridCandidateEvaluation) for row in rows):
        raise ValueError("evaluations must contain GridCandidateEvaluation values")

    keys = tuple((row.family, row.parameters) for row in rows)
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate grid evaluation")
    for family in TransferFamily:
        actual = {row.parameters for row in rows if row.family is family}
        expected = set(_expected_grid(family))
        if actual != expected:
            raise ValueError(f"{family.value} evaluations must contain the complete exact grid")

    for row in rows:
        if row.train_projection_hash != train.projection_hash:
            raise ValueError("evaluation TRAIN projection identity changed")
        if row.selection_projection_hash != selection.projection_hash:
            raise ValueError("evaluation SELECTION projection identity changed")
        if row.target_spec_hash != target_hash:
            raise ValueError("evaluation target spec identity changed")
        if row.split_manifest_hash != split_hash:
            raise ValueError("evaluation split manifest identity changed")
        if row.train_seeds != _TRAIN_SEEDS:
            raise ValueError("evaluation TRAIN seeds must be (101, 102)")
        if row.selection_seeds != _SELECTION_SEEDS:
            raise ValueError("evaluation SELECTION seeds must be (201, 202)")
        if row.score != "BRIER":
            raise ValueError("refit selection score must be BRIER")
        if row.brier_loss_identity != loss_identity:
            raise ValueError("evaluation Brier loss identity changed")
        if row.builder_identity != builder_map[row.family.value]:
            raise ValueError("evaluation builder identity changed")
        if row.simulation_identity != simulation_hash:
            raise ValueError("evaluation simulation identity changed")

    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                tuple(TransferFamily).index(row.family),
                row.parameters,
            ),
        )
    )
    selected_evaluations = tuple(
        _select_candidate(tuple(row for row in ordered if row.family is family))
        for family in TransferFamily
    )
    selected = tuple(
        FrozenTransferCandidate(
            family=row.family,
            parameters=row.parameters,
            source_candidate_hash=row.evaluation_receipt_hash,
        )
        for row in selected_evaluations
    )
    return RefitCandidateFreeze(
        train_projection_hash=train.projection_hash,
        selection_projection_hash=selection.projection_hash,
        target_spec_hash=target_hash,
        split_manifest_hash=split_hash,
        brier_loss_identity=loss_identity,
        builder_identities=builders,
        simulation_identity=simulation_hash,
        tie_break_identity=tie_break_identity,
        evaluations=ordered,
        selected_candidates=selected,
    )

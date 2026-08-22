from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
from statistics import fmean, pstdev

from narrative_dynamics.calibration import CalibrationResult, calibrate_grid
from narrative_dynamics.contracts import (
    ExperimentManifest,
    ExperimentStage,
    Scenario,
    _validated_content_hash,
)
from narrative_dynamics.manifest import required_manifest_hash, stable_content_hash
from narrative_dynamics.metrics import MetricExtractor
from narrative_dynamics.simulation import ModelSource, SimulationRunner


ParameterTuple = tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ParameterAcceptanceSet:
    """Canonical finite candidates plus the manifests that produced them."""

    parameters: tuple[ParameterTuple, ...]
    source_manifest_hashes: tuple[str, ...] = field(
        default=(),
        compare=False,
    )

    def __post_init__(self) -> None:
        canonical = tuple(
            sorted({_canonical_parameter_tuple(candidate) for candidate in self.parameters})
        )
        if canonical:
            schema = tuple(name for name, _ in canonical[0])
            for candidate in canonical[1:]:
                if tuple(name for name, _ in candidate) != schema:
                    raise ValueError("accepted parameter tuples must share one schema")
        sources = tuple(
            sorted(
                {
                    _validated_content_hash(
                        source_hash,
                        label="accepted-parameter source manifest hash",
                    )
                    for source_hash in self.source_manifest_hashes
                }
            )
        )
        object.__setattr__(self, "parameters", canonical)
        object.__setattr__(self, "source_manifest_hashes", sources)

    @classmethod
    def from_parameters(
        cls,
        parameters: Iterable[Iterable[tuple[str, float]]],
        *,
        source_manifest_hashes: Iterable[str] = (),
    ) -> "ParameterAcceptanceSet":
        return cls(
            parameters=tuple(tuple(candidate) for candidate in parameters),
            source_manifest_hashes=tuple(source_manifest_hashes),
        )

    @property
    def count(self) -> int:
        return len(self.parameters)

    @property
    def parameter_maps(self) -> tuple[dict[str, float], ...]:
        return tuple(dict(parameters) for parameters in self.parameters)

    @property
    def content_hash(self) -> str:
        return stable_content_hash(
            {
                "parameters": self.parameters,
                "source_manifest_hashes": self.source_manifest_hashes,
            }
        )

    def __iter__(self):
        return iter(self.parameters)

    def __len__(self) -> int:
        return len(self.parameters)


@dataclass(frozen=True)
class CandidateStability:
    """How one finite-grid candidate behaves across independent seed blocks."""

    parameters: ParameterTuple
    wins: int
    accepted_blocks: int
    mean_loss: float
    loss_stddev: float
    block_losses: tuple[float, ...]

    @property
    def parameter_map(self) -> dict[str, float]:
        return dict(self.parameters)

    @property
    def acceptance_fraction(self) -> float:
        return self.accepted_blocks / len(self.block_losses)


@dataclass(frozen=True)
class RepeatedCalibrationReport:
    """Repeated calibrations plus a stability-derived accepted parameter set."""

    blocks: tuple[CalibrationResult, ...]
    stability: tuple[CandidateStability, ...]
    accepted_parameters: tuple[ParameterTuple, ...]
    manifest: ExperimentManifest | None = None

    @property
    def acceptance_set(self) -> ParameterAcceptanceSet:
        source_hashes = (
            ()
            if self.manifest is None
            else (self.manifest.content_hash,)
        )
        return ParameterAcceptanceSet.from_parameters(
            self.accepted_parameters,
            source_manifest_hashes=source_hashes,
        )


@dataclass(frozen=True)
class SeedBlockVariant:
    """One deterministic translation of every seed in every calibration block."""

    offset: int
    seed_blocks: tuple[tuple[int, ...], ...]
    calibration: RepeatedCalibrationReport


@dataclass(frozen=True)
class SeedBlockVariationReport:
    """Calibration robustness under deterministic whole-block seed translations."""

    variants: tuple[SeedBlockVariant, ...]
    accepted_union: ParameterAcceptanceSet
    accepted_intersection: ParameterAcceptanceSet
    manifest: ExperimentManifest | None = None


@dataclass(frozen=True)
class ParameterIdentifiability:
    """Values retained for one parameter after uncertainty filtering."""

    name: str
    values: tuple[float, ...]
    identified: bool


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Per-parameter identifiability over an explicit accepted set."""

    accepted_parameters: tuple[ParameterTuple, ...]
    parameters: tuple[ParameterIdentifiability, ...]

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_parameters)

    @property
    def fully_identified(self) -> bool:
        return all(parameter.identified for parameter in self.parameters)


def _validated_nonnegative_finite(value: float, *, label: str) -> float:
    try:
        validated = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(validated) or validated < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return validated


def _validated_fraction(value: float, *, label: str) -> float:
    try:
        validated = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(validated) or not 0.0 < validated <= 1.0:
        raise ValueError(f"{label} must be finite and in (0, 1]")
    return validated


def _materialized_parameter_grid(
    parameter_grid: Mapping[str, Iterable[float]],
) -> dict[str, tuple[float, ...]]:
    """Consume each grid dimension once so every later calibration can replay it."""

    return {name: tuple(values) for name, values in parameter_grid.items()}


def repeated_grid_calibration(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    scenario: Scenario,
    parameter_grid: Mapping[str, Iterable[float]],
    seed_blocks: Iterable[Iterable[int]],
    extractor: MetricExtractor,
    target: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
    acceptance_loss_delta: float = 0.0,
    min_acceptance_fraction: float = 1.0,
) -> RepeatedCalibrationReport:
    """Repeat finite-grid calibration over independent seed blocks.

    A candidate is accepted in a block when its loss is no more than
    ``acceptance_loss_delta`` above that block's best loss. The final accepted
    set contains candidates accepted in at least ``min_acceptance_fraction`` of
    blocks. This deliberately reports stability rather than treating one best
    point as a uniquely true parameter.
    """

    blocks = tuple(tuple(block) for block in seed_blocks)
    if not blocks:
        raise ValueError("repeated calibration requires at least one seed block")
    if any(not block for block in blocks):
        raise ValueError("every repeated-calibration seed block must be non-empty")

    loss_delta = _validated_nonnegative_finite(
        acceptance_loss_delta,
        label="acceptance loss delta",
    )
    minimum_fraction = _validated_fraction(
        min_acceptance_fraction,
        label="minimum acceptance fraction",
    )
    replayable_grid = _materialized_parameter_grid(parameter_grid)

    calibrations = tuple(
        calibrate_grid(
            runner=runner,
            model=model,
            scenario=scenario,
            parameter_grid=replayable_grid,
            seeds=block,
            extractor=extractor,
            target=target,
            weights=weights,
        )
        for block in blocks
    )

    first_candidates = tuple(
        candidate.parameters for candidate in calibrations[0].ranking
    )
    first_candidate_set = set(first_candidates)
    for calibration in calibrations[1:]:
        if {candidate.parameters for candidate in calibration.ranking} != first_candidate_set:
            raise ValueError("repeated calibration changed its parameter candidate set")

    stability_entries: list[CandidateStability] = []
    accepted_parameters: list[ParameterTuple] = []
    block_count = len(calibrations)

    for parameters in sorted(first_candidate_set):
        losses: list[float] = []
        wins = 0
        accepted_blocks = 0
        for calibration in calibrations:
            candidate = next(
                item for item in calibration.ranking if item.parameters == parameters
            )
            losses.append(candidate.loss)
            if calibration.best.parameters == parameters:
                wins += 1
            if candidate.loss <= calibration.best.loss + loss_delta:
                accepted_blocks += 1

        loss_tuple = tuple(losses)
        entry = CandidateStability(
            parameters=parameters,
            wins=wins,
            accepted_blocks=accepted_blocks,
            mean_loss=fmean(loss_tuple),
            loss_stddev=pstdev(loss_tuple),
            block_losses=loss_tuple,
        )
        stability_entries.append(entry)
        if accepted_blocks / block_count >= minimum_fraction:
            accepted_parameters.append(parameters)

    block_hashes = tuple(
        required_manifest_hash(block, label="repeated-calibration block")
        for block in calibrations
    )
    manifest = ExperimentManifest(
        stage=ExperimentStage.REPEATED_CALIBRATION,
        inputs={
            "seed_blocks": blocks,
            "acceptance_loss_delta": loss_delta,
            "min_acceptance_fraction": minimum_fraction,
        },
        parent_hashes=block_hashes,
    )
    return RepeatedCalibrationReport(
        blocks=calibrations,
        stability=tuple(stability_entries),
        accepted_parameters=tuple(accepted_parameters),
        manifest=manifest,
    )


def calibrate_seed_block_variants(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    scenario: Scenario,
    parameter_grid: Mapping[str, Iterable[float]],
    seed_blocks: Iterable[Iterable[int]],
    seed_offsets: Iterable[int],
    extractor: MetricExtractor,
    target: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
    acceptance_loss_delta: float = 0.0,
    min_acceptance_fraction: float = 1.0,
) -> SeedBlockVariationReport:
    """Re-run repeated calibration after translating complete seed blocks.

    Each offset is added to every seed while preserving block membership and
    block size. The report exposes both the union and intersection of parameter
    candidates accepted across the resulting calibrations.
    """

    base_blocks = tuple(tuple(block) for block in seed_blocks)
    if not base_blocks:
        raise ValueError("seed-block variation requires at least one seed block")
    if any(not block for block in base_blocks):
        raise ValueError("every seed-block variation block must be non-empty")
    if any(not isinstance(seed, int) for block in base_blocks for seed in block):
        raise TypeError("seed-block variation seeds must be integers")

    offsets = tuple(seed_offsets)
    if not offsets:
        raise ValueError("seed-block variation requires at least one offset")
    if any(not isinstance(offset, int) for offset in offsets):
        raise TypeError("seed-block variation offsets must be integers")
    if len(set(offsets)) != len(offsets):
        raise ValueError("seed-block variation offsets must be unique")

    replayable_grid = _materialized_parameter_grid(parameter_grid)
    variants: list[SeedBlockVariant] = []
    accepted_sets: list[set[ParameterTuple]] = []
    for offset in offsets:
        mutated_blocks = tuple(
            tuple(seed + offset for seed in block) for block in base_blocks
        )
        calibration = repeated_grid_calibration(
            runner=runner,
            model=model,
            scenario=scenario,
            parameter_grid=replayable_grid,
            seed_blocks=mutated_blocks,
            extractor=extractor,
            target=target,
            weights=weights,
            acceptance_loss_delta=acceptance_loss_delta,
            min_acceptance_fraction=min_acceptance_fraction,
        )
        variants.append(
            SeedBlockVariant(
                offset=offset,
                seed_blocks=mutated_blocks,
                calibration=calibration,
            )
        )
        accepted_sets.append(set(calibration.accepted_parameters))

    accepted_union = set().union(*accepted_sets)
    accepted_intersection = set.intersection(*accepted_sets)
    variant_hashes = tuple(
        required_manifest_hash(
            variant.calibration,
            label="seed-block calibration variant",
        )
        for variant in variants
    )
    manifest = ExperimentManifest(
        stage=ExperimentStage.SEED_BLOCK_VARIATION,
        inputs={
            "base_seed_blocks": base_blocks,
            "seed_offsets": offsets,
        },
        parent_hashes=variant_hashes,
    )
    source_hashes = (manifest.content_hash,)
    return SeedBlockVariationReport(
        variants=tuple(variants),
        accepted_union=ParameterAcceptanceSet.from_parameters(
            accepted_union,
            source_manifest_hashes=source_hashes,
        ),
        accepted_intersection=ParameterAcceptanceSet.from_parameters(
            accepted_intersection,
            source_manifest_hashes=source_hashes,
        ),
        manifest=manifest,
    )


def _canonical_parameter_tuple(parameters: Iterable[tuple[str, float]]) -> ParameterTuple:
    canonical: list[tuple[str, float]] = []
    seen: set[str] = set()
    for name, raw_value in parameters:
        if not isinstance(name, str) or not name:
            raise ValueError("parameter names must be non-empty strings")
        if name in seen:
            raise ValueError("accepted parameter tuples cannot repeat a name")
        seen.add(name)
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"parameter {name!r} must be numeric") from error
        if not math.isfinite(value):
            raise ValueError(f"parameter {name!r} must be finite")
        canonical.append((name, value))
    if not canonical:
        raise ValueError("accepted parameter tuples must be non-empty")
    canonical.sort(key=lambda item: item[0])
    return tuple(canonical)


def diagnose_identifiability(
    accepted_parameters: Iterable[Iterable[tuple[str, float]]],
) -> IdentifiabilityReport:
    """Diagnose coordinate-wise identifiability over an accepted set.

    A coordinate is identified only when every accepted candidate retains the
    same value. This is deliberately weaker than claiming global structural
    identifiability of the underlying simulator.
    """

    canonical = tuple(
        sorted({_canonical_parameter_tuple(parameters) for parameters in accepted_parameters})
    )
    if not canonical:
        raise ValueError("identifiability requires a non-empty accepted parameter set")

    names = tuple(name for name, _ in canonical[0])
    name_set = set(names)
    for parameters in canonical[1:]:
        if {name for name, _ in parameters} != name_set:
            raise ValueError("accepted parameter tuples must share one schema")

    diagnostics = tuple(
        ParameterIdentifiability(
            name=name,
            values=tuple(sorted({dict(parameters)[name] for parameters in canonical})),
            identified=len({dict(parameters)[name] for parameters in canonical}) == 1,
        )
        for name in names
    )
    return IdentifiabilityReport(
        accepted_parameters=canonical,
        parameters=diagnostics,
    )

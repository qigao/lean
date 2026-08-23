from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import random
import re
from types import MappingProxyType
from typing import Protocol


MANIFEST_SCHEMA_VERSION = 1
_CONTENT_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _freeze_canonical_value(value: object, *, label: str) -> object:
    """Detach and recursively freeze JSON-like runtime boundary data."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} keys must be non-empty strings")
            frozen[key] = _freeze_canonical_value(item, label=f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_canonical_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{label} values must be canonical scalars, mappings, lists, or tuples")


def _freeze_mapping(value: Mapping[str, object], *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen = _freeze_canonical_value(value, label=label)
    assert isinstance(frozen, Mapping)
    return frozen


def _canonical_hash_value(value: object, *, label: str) -> object:
    """Encode canonical values with explicit type tags for stable hashing."""

    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} numbers must be finite")
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, Mapping):
        keys = tuple(value)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError(f"{label} keys must be non-empty strings")
        items: list[tuple[str, object]] = []
        for key in sorted(keys):
            items.append((key, _canonical_hash_value(value[key], label=f"{label}.{key}")))
        return ("mapping", tuple(items))
    if isinstance(value, (list, tuple)):
        return (
            "sequence",
            tuple(
                _canonical_hash_value(item, label=f"{label}[{index}]")
                for index, item in enumerate(value)
            ),
        )
    raise TypeError(f"{label} values must be canonical scalars, mappings, lists, or tuples")


def stable_content_hash(value: object) -> str:
    """Return a deterministic typed SHA-256 digest for canonical runtime data."""

    canonical = _canonical_hash_value(value, label="content")
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _validated_content_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


class ExperimentStage(str, Enum):
    """Stable manifest stages for the Python research runtime."""

    SIMULATION_RUN = "simulation_run"
    GRID_CALIBRATION = "grid_calibration"
    REPEATED_CALIBRATION = "repeated_calibration"
    SEED_BLOCK_VARIATION = "seed_block_variation"
    SYNTHETIC_RECOVERY = "synthetic_recovery"
    HELD_OUT_VALIDATION = "held_out_validation"
    ACCEPTANCE_VALIDATION = "acceptance_validation"
    SELECTION_VALIDATION = "selection_validation"
    FINAL_TEST = "final_test"
    LOCAL_SENSITIVITY = "local_sensitivity"
    MODEL_MISSPECIFICATION = "model_misspecification"
    INTERACTION_SENSITIVITY = "interaction_sensitivity"
    FINAL_TEST_COVERAGE = "final_test_coverage"
    TARGET_CONSTRUCTION = "target_construction"
    ALTERNATIVE_MODEL_COMPARISON = "alternative_model_comparison"
    OBSERVATION_TARGET_CONSTRUCTION = "observation_target_construction"
    MODEL_COMPARISON = "model_comparison"


@dataclass(frozen=True)
class ExperimentManifest:
    """Versioned immutable identity and parent chain for one experiment stage."""

    stage: ExperimentStage | str
    inputs: Mapping[str, object]
    parent_hashes: tuple[str, ...] = ()
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version <= 0
        ):
            raise ValueError("manifest schema version must be a positive integer")
        try:
            stage = self.stage if isinstance(self.stage, ExperimentStage) else ExperimentStage(self.stage)
        except (TypeError, ValueError) as error:
            raise ValueError("manifest stage must be supported") from error
        parent_hashes = tuple(sorted({
            _validated_content_hash(parent_hash, label="manifest parent hash")
            for parent_hash in self.parent_hashes
        }))
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "inputs", _freeze_mapping(self.inputs, label="manifest inputs"))
        object.__setattr__(self, "parent_hashes", parent_hashes)

    @property
    def content_hash(self) -> str:
        return stable_content_hash({
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "inputs": self.inputs,
            "parent_hashes": self.parent_hashes,
        })


@dataclass(frozen=True)
class Scenario:
    """Opaque model input identified by a stable experiment id."""

    id: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("scenario id must be non-empty")
        object.__setattr__(self, "payload", _freeze_mapping(self.payload, label="scenario payload"))

    @property
    def content_hash(self) -> str:
        return stable_content_hash({"id": self.id, "payload": self.payload})


@dataclass(frozen=True)
class TraceEvent:
    """One ordered event emitted by a simulation model."""

    tick: int
    kind: str
    data: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise ValueError("event tick must be non-negative")
        if not self.kind:
            raise ValueError("event kind must be non-empty")
        object.__setattr__(self, "data", _freeze_mapping(self.data, label="event data"))


@dataclass(frozen=True)
class ModelRun:
    """Model-owned output before trusted run metadata is attached."""

    events: tuple[TraceEvent, ...]
    outcome: Mapping[str, object]

    def __post_init__(self) -> None:
        events = tuple(self.events)
        if any(not isinstance(event, TraceEvent) for event in events):
            raise TypeError("model run events must contain TraceEvent values")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "outcome", _freeze_mapping(self.outcome, label="model outcome"))


@dataclass(frozen=True)
class ExecutionCapture:
    """Trusted execution metadata captured outside the model result."""

    isolated: bool
    stdout: str
    stderr: str
    return_code: int
    duration_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.isolated, bool):
            raise TypeError("execution isolated flag must be boolean")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("execution stdout and stderr must be strings")
        if not isinstance(self.return_code, int) or isinstance(self.return_code, bool):
            raise TypeError("execution return code must be an integer")
        if (
            not isinstance(self.duration_seconds, (int, float))
            or isinstance(self.duration_seconds, bool)
            or not math.isfinite(float(self.duration_seconds))
            or float(self.duration_seconds) < 0.0
        ):
            raise ValueError("execution duration must be finite and non-negative")
        object.__setattr__(self, "duration_seconds", float(self.duration_seconds))


@dataclass(frozen=True)
class SimulationTrace:
    """Canonical result used by metrics, calibration, and replay."""

    model_name: str
    scenario_id: str
    parameters: tuple[tuple[str, float], ...]
    seed: int
    events: tuple[TraceEvent, ...]
    outcome: Mapping[str, object]
    manifest: ExperimentManifest | None = None
    execution: ExecutionCapture | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        events = tuple(self.events)
        if any(not isinstance(event, TraceEvent) for event in events):
            raise TypeError("simulation trace events must contain TraceEvent values")
        if self.manifest is not None and not isinstance(self.manifest, ExperimentManifest):
            raise TypeError("simulation trace manifest must be an ExperimentManifest")
        if self.execution is not None and not isinstance(self.execution, ExecutionCapture):
            raise TypeError("simulation trace execution must be an ExecutionCapture")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "outcome", _freeze_mapping(self.outcome, label="trace outcome"))

    @property
    def parameter_map(self) -> dict[str, float]:
        return dict(self.parameters)

    @property
    def manifest_hash(self) -> str | None:
        return None if self.manifest is None else self.manifest.content_hash


class SimulatorModel(Protocol):
    """Boundary implemented by existing, learned, or black-box models."""

    name: str

    def simulate(
        self,
        scenario: Scenario,
        parameters: Mapping[str, float],
        rng: random.Random,
    ) -> ModelRun:
        ...

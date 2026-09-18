"""Exact input model for the PathN exposure consensus analyzer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
import re
import tomllib


class ModelInputError(ValueError):
    """Raised when a model document is outside the Phase 1 input language."""


_RAT_RE = re.compile(r"^[+-]?\d+(?:/[1-9]\d*)?$")


@dataclass(frozen=True)
class ExactRat:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator == 0:
            raise ValueError("rational denominator must be nonzero")
        value = Fraction(self.numerator, self.denominator)
        object.__setattr__(self, "numerator", value.numerator)
        object.__setattr__(self, "denominator", value.denominator)


class DecayTarget(Enum):
    ZERO = "zero"
    ONE = "one"


@dataclass(frozen=True)
class HarmonicSchedule:
    c: ExactRat
    offset: int
    target: DecayTarget


@dataclass(frozen=True)
class PolynomialSchedule:
    c: ExactRat
    p: int
    offset: int
    target: DecayTarget


@dataclass(frozen=True)
class ExponentialSchedule:
    c: ExactRat
    base: ExactRat
    offset: int
    target: DecayTarget


@dataclass(frozen=True)
class PeriodicSchedule:
    values: tuple[ExactRat, ...]


@dataclass(frozen=True)
class AlternatingSchedule:
    a: ExactRat
    b: ExactRat


@dataclass(frozen=True)
class ConstantSchedule:
    value: ExactRat


@dataclass(frozen=True)
class PiecewiseSchedule:
    default: ExactRat
    points: tuple[tuple[int, ExactRat], ...]


@dataclass(frozen=True)
class NamedSchedule:
    schedule_id: str


Schedule = (
    ConstantSchedule
    | PiecewiseSchedule
    | NamedSchedule
    | HarmonicSchedule
    | PolynomialSchedule
    | ExponentialSchedule
    | PeriodicSchedule
    | AlternatingSchedule
)


@dataclass(frozen=True)
class PathModel:
    n: int
    beliefs: tuple[ExactRat, ...]
    exposures: tuple[int, ...]
    threshold: ExactRat
    schedule: Schedule


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ModelInputError(f"{context} must be a table")
    return value


def _field(table: Mapping[str, object], key: str, context: str) -> object:
    try:
        return table[key]
    except KeyError as exc:
        raise ModelInputError(f"missing {context}.{key}") from exc


def _string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ModelInputError(f"{context} must be a string")
    return value


def _natural(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ModelInputError(f"{context} must be a natural number")
    return value


def _exact_rat(value: object, context: str) -> ExactRat:
    if not isinstance(value, str) or _RAT_RE.fullmatch(value) is None:
        raise ModelInputError(
            f"{context} must be an exact integer or fraction string"
        )
    if "/" in value:
        numerator_text, denominator_text = value.split("/", 1)
        numerator = int(numerator_text)
        denominator = int(denominator_text)
    else:
        numerator = int(value)
        denominator = 1
    return ExactRat(numerator, denominator)


def _list(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ModelInputError(f"{context} must be an array")
    return value


def _positive_exact_rat(value: object, context: str) -> ExactRat:
    result = _exact_rat(value, context)
    if result.numerator <= 0:
        raise ModelInputError(f"{context} must be positive")
    return result


def _positive_natural(value: object, context: str) -> int:
    result = _natural(value, context)
    if result == 0:
        raise ModelInputError(f"{context} must be at least 1")
    return result


def _decay_target(value: object, context: str) -> DecayTarget:
    target = _string(value, context)
    try:
        return DecayTarget(target)
    except ValueError as exc:
        raise ModelInputError(f"unsupported {context}: {target}") from exc


def _parse_schedule(value: object) -> Schedule:
    schedule = _mapping(value, "schedule")
    kind = _string(_field(schedule, "kind", "schedule"), "schedule.kind")


    if kind == "harmonic":
        return HarmonicSchedule(
            c=_positive_exact_rat(_field(schedule, "c", "schedule"), "schedule.c"),
            offset=_positive_natural(
                _field(schedule, "offset", "schedule"), "schedule.offset"
            ),
            target=_decay_target(
                _field(schedule, "target", "schedule"), "schedule.target"
            ),
        )

    if kind == "polynomial":
        return PolynomialSchedule(
            c=_positive_exact_rat(_field(schedule, "c", "schedule"), "schedule.c"),
            p=_positive_natural(_field(schedule, "p", "schedule"), "schedule.p"),
            offset=_positive_natural(
                _field(schedule, "offset", "schedule"), "schedule.offset"
            ),
            target=_decay_target(
                _field(schedule, "target", "schedule"), "schedule.target"
            ),
        )

    if kind == "exponential":
        c = _positive_exact_rat(_field(schedule, "c", "schedule"), "schedule.c")
        base = _positive_exact_rat(
            _field(schedule, "base", "schedule"), "schedule.base"
        )
        if Fraction(base.numerator, base.denominator) >= 1:
            raise ModelInputError("schedule.base must be strictly less than 1")
        return ExponentialSchedule(
            c=c,
            base=base,
            offset=_natural(
                _field(schedule, "offset", "schedule"), "schedule.offset"
            ),
            target=_decay_target(
                _field(schedule, "target", "schedule"), "schedule.target"
            ),
        )

    if kind == "periodic":
        raw_values = _list(
            _field(schedule, "values", "schedule"), "schedule.values"
        )
        if not raw_values:
            raise ModelInputError("schedule.values must not be empty")
        return PeriodicSchedule(
            tuple(
                _exact_rat(item, f"schedule.values[{index}]")
                for index, item in enumerate(raw_values)
            )
        )

    if kind == "alternating":
        return AlternatingSchedule(
            a=_exact_rat(_field(schedule, "a", "schedule"), "schedule.a"),
            b=_exact_rat(_field(schedule, "b", "schedule"), "schedule.b"),
        )

    if kind == "constant":
        return ConstantSchedule(
            _exact_rat(_field(schedule, "value", "schedule"), "schedule.value")
        )

    if kind == "piecewise":
        default = _exact_rat(
            _field(schedule, "default", "schedule"), "schedule.default"
        )
        raw_points = _list(
            _field(schedule, "points", "schedule"), "schedule.points"
        )
        points: list[tuple[int, ExactRat]] = []
        seen: set[int] = set()
        for index, raw_point in enumerate(raw_points):
            point = _mapping(raw_point, f"schedule.points[{index}]")
            exposure = _natural(
                _field(point, "exposure", f"schedule.points[{index}]"),
                f"schedule.points[{index}].exposure",
            )
            if exposure in seen:
                raise ModelInputError(
                    f"duplicate piecewise exposure index: {exposure}"
                )
            seen.add(exposure)
            point_value = _exact_rat(
                _field(point, "value", f"schedule.points[{index}]"),
                f"schedule.points[{index}].value",
            )
            points.append((exposure, point_value))
        points.sort(key=lambda item: item[0])
        return PiecewiseSchedule(default=default, points=tuple(points))

    if kind == "named":
        schedule_id = _string(
            _field(schedule, "id", "schedule"), "schedule.id"
        )
        if not schedule_id:
            raise ModelInputError("schedule.id must not be empty")
        return NamedSchedule(schedule_id)

    raise ModelInputError(f"unsupported schedule kind: {kind}")


def parse_model(document: Mapping[str, object]) -> PathModel:
    root = _mapping(document, "model")
    topology = _mapping(_field(root, "topology", "model"), "topology")
    initial = _mapping(_field(root, "initial", "model"), "initial")
    dynamics = _mapping(_field(root, "dynamics", "model"), "dynamics")

    topology_kind = _string(
        _field(topology, "kind", "topology"), "topology.kind"
    )
    if topology_kind != "path":
        raise ModelInputError(f"unsupported topology kind: {topology_kind}")

    n = _natural(_field(topology, "n", "topology"), "topology.n")
    if n < 2:
        raise ModelInputError("topology.n must be at least 2")

    raw_beliefs = _list(_field(initial, "beliefs", "initial"), "initial.beliefs")
    beliefs = tuple(
        _exact_rat(value, f"initial.beliefs[{index}]")
        for index, value in enumerate(raw_beliefs)
    )
    if len(beliefs) != n:
        raise ModelInputError(
            f"initial.beliefs length {len(beliefs)} does not match topology.n {n}"
        )

    raw_exposures = _list(
        _field(initial, "exposures", "initial"), "initial.exposures"
    )
    exposures = tuple(
        _natural(value, f"initial.exposures[{index}]")
        for index, value in enumerate(raw_exposures)
    )
    if len(exposures) != n:
        raise ModelInputError(
            f"initial.exposures length {len(exposures)} does not match topology.n {n}"
        )

    threshold = _exact_rat(
        _field(dynamics, "threshold", "dynamics"), "dynamics.threshold"
    )
    schedule = _parse_schedule(_field(root, "schedule", "model"))

    return PathModel(
        n=n,
        beliefs=beliefs,
        exposures=exposures,
        threshold=threshold,
        schedule=schedule,
    )


def load_model(path: Path) -> PathModel:
    try:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ModelInputError(f"invalid TOML: {exc}") from exc
    return parse_model(document)

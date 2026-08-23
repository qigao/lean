from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import re
from types import MappingProxyType

from motivation import (
    AgentMotivation,
    Goal,
    bayes_posterior,
    effective_pressure,
    learn_instrumentality,
)


REFERENCE_SCHEMA_VERSION = 1
REFERENCE_GENERATOR = "NarrativeDynamics.Conformance.ReferenceVectors"
REFERENCE_NUMERIC_ENCODING = "reduced_fraction_v1"
REFERENCE_DEFINITIONS = (
    "NarrativeDynamics.bayesPosterior",
    "NarrativeDynamics.learnInstrumentality",
    "NarrativeDynamics.effectivePressure",
    "NarrativeDynamics.goalScore",
)
CONFORMANCE_REL_TOL = 1e-12
CONFORMANCE_ABS_TOL = 1e-12

_TOP_LEVEL_KEYS = (
    "schema_version",
    "generator",
    "numeric_encoding",
    "definitions",
    "vectors",
)
_VECTOR_KEYS = ("id", "operation", "inputs", "expected")
_OPERATION_INPUTS = MappingProxyType(
    {
        "bayes_posterior": (
            "prior",
            "likelihood_h",
            "likelihood_not_h",
        ),
        "effective_pressure2": (
            "p_self",
            "p_other",
            "boundary",
        ),
        "goal_score_single_drive": (
            "pressure",
            "instrumentality",
            "cost",
            "risk",
        ),
        "learn_instrumentality": (
            "old",
            "observed",
            "rate",
        ),
    }
)
_FRACTION_PATTERN = re.compile(r"^(0|-?[1-9][0-9]*)/([1-9][0-9]*)$")
_VECTOR_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class ConformanceDefinitionError(ValueError):
    """The committed reference corpus violates its versioned contract."""

    def __init__(self, message: str, *, path: str = "$") -> None:
        self.path = path
        super().__init__(f"conformance definition error at {path}: {message}")


class ConformanceMismatch(AssertionError):
    """One Python production result differs from its Lean reference value."""

    def __init__(
        self,
        *,
        vector_id: str,
        operation: str,
        expected: float,
        actual: object,
        message: str | None = None,
    ) -> None:
        self.vector_id = vector_id
        self.operation = operation
        self.expected = expected
        self.actual = actual
        detail = message or "Python result differs from the Lean reference"
        super().__init__(
            f"{detail}: vector={vector_id!r}, operation={operation!r}, "
            f"expected={expected!r}, actual={actual!r}"
        )


@dataclass(frozen=True)
class ExactRational:
    """Canonical reduced rational transported from the Lean corpus."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if not isinstance(self.numerator, int) or isinstance(self.numerator, bool):
            raise TypeError("exact rational numerator must be an integer")
        if not isinstance(self.denominator, int) or isinstance(self.denominator, bool):
            raise TypeError("exact rational denominator must be an integer")
        if self.denominator <= 0:
            raise ValueError("exact rational denominator must be positive")
        if math.gcd(abs(self.numerator), self.denominator) != 1:
            raise ValueError("exact rational must be reduced")

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def to_float(self) -> float:
        return float(self.fraction)

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass(frozen=True)
class ReferenceVector:
    """One named operation invocation and exact Lean-certified result."""

    id: str
    operation: str
    inputs: Mapping[str, ExactRational]
    expected: ExactRational

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or _VECTOR_ID_PATTERN.fullmatch(self.id) is None:
            raise ValueError("reference vector id must be a canonical snake-case identifier")
        if self.operation not in _OPERATION_INPUTS:
            raise ValueError("reference vector operation must be supported")
        frozen_inputs = dict(self.inputs)
        if tuple(frozen_inputs) != _OPERATION_INPUTS[self.operation]:
            raise ValueError("reference vector inputs do not match the operation contract")
        if any(not isinstance(value, ExactRational) for value in frozen_inputs.values()):
            raise TypeError("reference vector inputs must be ExactRational values")
        if not isinstance(self.expected, ExactRational):
            raise TypeError("reference vector expected value must be ExactRational")
        object.__setattr__(self, "inputs", MappingProxyType(frozen_inputs))


@dataclass(frozen=True)
class ReferenceSuite:
    """Immutable versioned corpus emitted from the Lean sidecar."""

    schema_version: int
    generator: str
    numeric_encoding: str
    definitions: tuple[str, ...]
    vectors: tuple[ReferenceVector, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "definitions", tuple(self.definitions))
        object.__setattr__(self, "vectors", tuple(self.vectors))


OperationEvaluator = Callable[[Mapping[str, float]], float]


def _definition_error(message: str, *, path: str) -> None:
    raise ConformanceDefinitionError(message, path=path)


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _definition_error("must be an object", path=path)
    for key in value:
        if not isinstance(key, str) or not key:
            _definition_error("object keys must be non-empty strings", path=path)
    return value


def _array(value: object, *, path: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        _definition_error("must be an array", path=path)
    return tuple(value)


def _require_exact_keys(
    value: Mapping[str, object],
    expected: tuple[str, ...],
    *,
    path: str,
) -> None:
    actual = tuple(value)
    unknown = tuple(key for key in actual if key not in expected)
    missing = tuple(key for key in expected if key not in value)
    if unknown:
        _definition_error(f"contains unsupported key {unknown[0]!r}", path=path)
    if missing:
        _definition_error(f"is missing required key {missing[0]!r}", path=path)
    if actual != expected:
        _definition_error(
            f"keys must appear in canonical order {expected!r}",
            path=path,
        )


def _text(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        _definition_error("must be a non-empty string", path=path)
    return value


def _parse_exact_rational(value: object, *, path: str) -> ExactRational:
    if not isinstance(value, str):
        _definition_error("must be a reduced fraction string", path=path)
    matched = _FRACTION_PATTERN.fullmatch(value)
    if matched is None:
        _definition_error(
            "must use canonical numerator/positive-denominator syntax",
            path=path,
        )
    numerator = int(matched.group(1))
    denominator = int(matched.group(2))
    if math.gcd(abs(numerator), denominator) != 1:
        _definition_error("fraction must be reduced", path=path)
    return ExactRational(numerator=numerator, denominator=denominator)


def _parse_vector(value: object, *, index: int) -> ReferenceVector:
    path = f"$.vectors[{index}]"
    raw = _mapping(value, path=path)
    _require_exact_keys(raw, _VECTOR_KEYS, path=path)

    vector_id = _text(raw["id"], path=f"{path}.id")
    if _VECTOR_ID_PATTERN.fullmatch(vector_id) is None:
        _definition_error(
            "must be a canonical snake-case identifier",
            path=f"{path}.id",
        )

    operation = _text(raw["operation"], path=f"{path}.operation")
    expected_inputs = _OPERATION_INPUTS.get(operation)
    if expected_inputs is None:
        _definition_error(
            f"unsupported operation {operation!r}",
            path=f"{path}.operation",
        )

    raw_inputs = _mapping(raw["inputs"], path=f"{path}.inputs")
    _require_exact_keys(raw_inputs, expected_inputs, path=f"{path}.inputs")
    parsed_inputs = {
        name: _parse_exact_rational(
            raw_inputs[name],
            path=f"{path}.inputs.{name}",
        )
        for name in expected_inputs
    }
    expected = _parse_exact_rational(raw["expected"], path=f"{path}.expected")

    vector = ReferenceVector(
        id=vector_id,
        operation=operation,
        inputs=parsed_inputs,
        expected=expected,
    )
    if operation == "bayes_posterior":
        prior = vector.inputs["prior"].fraction
        likelihood_h = vector.inputs["likelihood_h"].fraction
        likelihood_not_h = vector.inputs["likelihood_not_h"].fraction
        denominator = (
            prior * likelihood_h
            + (Fraction(1, 1) - prior) * likelihood_not_h
        )
        if denominator <= 0:
            _definition_error(
                "Bayesian reference denominator must be strictly positive",
                path=f"{path}.inputs",
            )
    elif operation == "goal_score_single_drive":
        if vector.inputs["pressure"].fraction < 0:
            _definition_error(
                "single-drive pressure must be non-negative",
                path=f"{path}.inputs.pressure",
            )
    return vector


def _decode_json(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConformanceDefinitionError(
            f"cannot read reference fixture: {error}",
            path="$",
        ) from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ConformanceDefinitionError(
            f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}",
            path="$",
        ) from error


def load_reference_suite(path: str | Path) -> ReferenceSuite:
    """Load and strictly validate one Lean-generated reference corpus."""

    source = Path(path)
    raw = _mapping(_decode_json(source), path="$")
    _require_exact_keys(raw, _TOP_LEVEL_KEYS, path="$")

    schema_version = raw["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != REFERENCE_SCHEMA_VERSION
    ):
        _definition_error(
            f"must equal {REFERENCE_SCHEMA_VERSION}",
            path="$.schema_version",
        )

    generator = _text(raw["generator"], path="$.generator")
    if generator != REFERENCE_GENERATOR:
        _definition_error(
            f"must equal {REFERENCE_GENERATOR!r}",
            path="$.generator",
        )

    numeric_encoding = _text(
        raw["numeric_encoding"],
        path="$.numeric_encoding",
    )
    if numeric_encoding != REFERENCE_NUMERIC_ENCODING:
        _definition_error(
            f"must equal {REFERENCE_NUMERIC_ENCODING!r}",
            path="$.numeric_encoding",
        )

    definitions = tuple(
        _text(value, path=f"$.definitions[{index}]")
        for index, value in enumerate(
            _array(raw["definitions"], path="$.definitions")
        )
    )
    if definitions != REFERENCE_DEFINITIONS:
        _definition_error(
            f"must equal the canonical definition list {REFERENCE_DEFINITIONS!r}",
            path="$.definitions",
        )

    raw_vectors = _array(raw["vectors"], path="$.vectors")
    if not raw_vectors:
        _definition_error("must contain at least one reference vector", path="$.vectors")
    vectors = tuple(
        _parse_vector(value, index=index)
        for index, value in enumerate(raw_vectors)
    )
    vector_ids = tuple(vector.id for vector in vectors)
    if len(set(vector_ids)) != len(vector_ids):
        _definition_error("vector ids must be unique", path="$.vectors")
    if vector_ids != tuple(sorted(vector_ids)):
        _definition_error(
            "vectors must be ordered lexicographically by id",
            path="$.vectors",
        )

    return ReferenceSuite(
        schema_version=schema_version,
        generator=generator,
        numeric_encoding=numeric_encoding,
        definitions=definitions,
        vectors=vectors,
    )


def _evaluate_bayes(inputs: Mapping[str, float]) -> float:
    return bayes_posterior(
        inputs["prior"],
        inputs["likelihood_h"],
        inputs["likelihood_not_h"],
    )


def _evaluate_learning(inputs: Mapping[str, float]) -> float:
    return learn_instrumentality(
        inputs["old"],
        inputs["observed"],
        inputs["rate"],
    )


def _evaluate_effective_pressure(inputs: Mapping[str, float]) -> float:
    return effective_pressure(
        (1.0, inputs["boundary"]),
        (inputs["p_self"], inputs["p_other"]),
    )


def _evaluate_goal_score(inputs: Mapping[str, float]) -> float:
    model = AgentMotivation(
        pressures={"drive": inputs["pressure"]},
        goals={
            "goal": Goal(
                {"drive": inputs["instrumentality"]},
                cost=inputs["cost"],
                risk=inputs["risk"],
            )
        },
    )
    return model.score("goal")


def default_operation_evaluators() -> dict[str, OperationEvaluator]:
    """Return mutable dispatch entries backed by existing production code."""

    return {
        "bayes_posterior": _evaluate_bayes,
        "effective_pressure2": _evaluate_effective_pressure,
        "goal_score_single_drive": _evaluate_goal_score,
        "learn_instrumentality": _evaluate_learning,
    }


def assert_reference_conformance(
    suite: ReferenceSuite,
    *,
    evaluators: Mapping[str, OperationEvaluator] | None = None,
) -> None:
    """Raise a typed mismatch when Python drifts from any Lean reference."""

    if not isinstance(suite, ReferenceSuite):
        raise TypeError("suite must be a ReferenceSuite")
    dispatch = default_operation_evaluators() if evaluators is None else evaluators
    if not isinstance(dispatch, Mapping):
        raise TypeError("evaluators must be a mapping or None")

    for vector in suite.vectors:
        evaluator = dispatch.get(vector.operation)
        if not callable(evaluator):
            raise ConformanceDefinitionError(
                f"missing callable evaluator for {vector.operation!r}",
                path=f"$.vectors[{vector.id!r}].operation",
            )
        inputs = MappingProxyType(
            {
                name: exact.to_float()
                for name, exact in vector.inputs.items()
            }
        )
        expected = vector.expected.to_float()
        try:
            raw_actual = evaluator(inputs)
        except Exception as error:
            raise ConformanceMismatch(
                vector_id=vector.id,
                operation=vector.operation,
                expected=expected,
                actual=f"{type(error).__name__}: {error}",
                message="Python evaluator raised an exception",
            ) from error
        if isinstance(raw_actual, bool):
            raise ConformanceMismatch(
                vector_id=vector.id,
                operation=vector.operation,
                expected=expected,
                actual=raw_actual,
                message="Python evaluator returned a boolean",
            )
        try:
            actual = float(raw_actual)
        except (TypeError, ValueError) as error:
            raise ConformanceMismatch(
                vector_id=vector.id,
                operation=vector.operation,
                expected=expected,
                actual=raw_actual,
                message="Python evaluator returned a non-numeric value",
            ) from error
        if not math.isfinite(actual):
            raise ConformanceMismatch(
                vector_id=vector.id,
                operation=vector.operation,
                expected=expected,
                actual=actual,
                message="Python evaluator returned a non-finite value",
            )
        if not math.isclose(
            actual,
            expected,
            rel_tol=CONFORMANCE_REL_TOL,
            abs_tol=CONFORMANCE_ABS_TOL,
        ):
            raise ConformanceMismatch(
                vector_id=vector.id,
                operation=vector.operation,
                expected=expected,
                actual=actual,
            )


__all__ = [
    "CONFORMANCE_ABS_TOL",
    "CONFORMANCE_REL_TOL",
    "ConformanceDefinitionError",
    "ConformanceMismatch",
    "ExactRational",
    "REFERENCE_DEFINITIONS",
    "REFERENCE_GENERATOR",
    "REFERENCE_NUMERIC_ENCODING",
    "REFERENCE_SCHEMA_VERSION",
    "ReferenceSuite",
    "ReferenceVector",
    "assert_reference_conformance",
    "default_operation_evaluators",
    "load_reference_suite",
]

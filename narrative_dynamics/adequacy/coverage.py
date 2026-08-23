from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
import math

from narrative_dynamics.contracts import ExperimentManifest, ExperimentStage
from narrative_dynamics.manifest import (
    required_manifest_hash,
    scenario_identity,
    stable_content_hash,
)
from narrative_dynamics.validation import (
    EvaluationRole,
    FinalTestReport,
    HeldOutSuite,
)


def _validated_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot contain surrounding whitespace")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


@dataclass(frozen=True)
class CoverageBin:
    name: str
    lower: float
    upper: float
    include_upper: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _validated_name(self.name, label="coverage bin name"),
        )
        lower = _finite_number(self.lower, label="coverage bin lower bound")
        upper = _finite_number(self.upper, label="coverage bin upper bound")
        if upper < lower:
            raise ValueError("coverage bin upper bound must be at least its lower bound")
        if upper == lower and not self.include_upper:
            raise ValueError("point coverage bins must include their upper bound")
        if not isinstance(self.include_upper, bool):
            raise TypeError("coverage bin include_upper must be boolean")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def contains(self, value: float) -> bool:
        number = _finite_number(value, label="coverage value")
        if self.include_upper:
            return self.lower <= number <= self.upper
        return self.lower <= number < self.upper

    def manifest_identity(self) -> dict[str, object]:
        return {
            "name": self.name,
            "lower": self.lower,
            "upper": self.upper,
            "include_upper": self.include_upper,
        }


@dataclass(frozen=True)
class CoverageAxis:
    name: str
    field: str
    bins: tuple[CoverageBin, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _validated_name(self.name, label="coverage axis name"),
        )
        object.__setattr__(
            self,
            "field",
            _validated_name(self.field, label="coverage axis field"),
        )
        bins = tuple(self.bins)
        if not bins:
            raise ValueError("coverage axis requires at least one bin")
        if any(not isinstance(item, CoverageBin) for item in bins):
            raise TypeError("coverage axis bins must be CoverageBin values")
        if len({item.name for item in bins}) != len(bins):
            raise ValueError("coverage axis bin names must be unique")

        ordered = tuple(sorted(bins, key=lambda item: (item.lower, item.upper, item.name)))
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.lower < previous.upper:
                raise ValueError("coverage axis bins cannot overlap")
            if current.lower == previous.upper and previous.include_upper:
                raise ValueError("coverage axis bins cannot overlap at a closed boundary")
        object.__setattr__(self, "bins", bins)

    def manifest_identity(self) -> dict[str, object]:
        return {
            "name": self.name,
            "field": self.field,
            "bins": tuple(item.manifest_identity() for item in self.bins),
        }


ScenarioCoverageAxis = CoverageAxis


@dataclass(frozen=True)
class AxisCoverage:
    axis_name: str
    field: str
    counts: tuple[tuple[str, int], ...]
    missing_bins: tuple[str, ...]
    uncovered_cases: tuple[str, ...]

    @property
    def count_map(self) -> dict[str, int]:
        return dict(self.counts)

    @property
    def complete(self) -> bool:
        return not self.missing_bins and not self.uncovered_cases


@dataclass(frozen=True)
class FinalTestCoverageReport:
    suite_name: str
    case_count: int
    total_seed_count: int
    unique_seed_count: int
    reused_seeds: tuple[int, ...]
    axes: tuple[AxisCoverage, ...]
    complete: bool
    manifest: ExperimentManifest

    @property
    def axis_map(self) -> dict[str, AxisCoverage]:
        return {axis.axis_name: axis for axis in self.axes}

    @property
    def total_seeds(self) -> int:
        return self.total_seed_count

    @property
    def unique_seeds(self) -> int:
        return self.unique_seed_count


def _axis_coverage(axis: CoverageAxis, suite: HeldOutSuite) -> AxisCoverage:
    counts = {item.name: 0 for item in axis.bins}
    uncovered: list[str] = []
    for case in suite.cases:
        payload = case.scenario.payload
        if axis.field not in payload:
            raise ValueError(
                f"coverage axis field {axis.field!r} is missing from case "
                f"{case.effective_name!r}"
            )
        value = _finite_number(
            payload[axis.field],
            label=f"coverage field {axis.field!r}",
        )
        matches = tuple(item for item in axis.bins if item.contains(value))
        if len(matches) > 1:
            raise RuntimeError("validated coverage bins matched one value more than once")
        if not matches:
            uncovered.append(case.effective_name)
        else:
            counts[matches[0].name] += 1
    missing = tuple(item.name for item in axis.bins if counts[item.name] == 0)
    return AxisCoverage(
        axis_name=axis.name,
        field=axis.field,
        counts=tuple((item.name, counts[item.name]) for item in axis.bins),
        missing_bins=missing,
        uncovered_cases=tuple(uncovered),
    )


def diagnose_final_test_coverage(
    *,
    final_report: FinalTestReport,
    suite: HeldOutSuite,
    axes: Iterable[CoverageAxis],
) -> FinalTestCoverageReport:
    """Report declared scenario-stratum coverage and seed reuse.

    This is design coverage only. It is not confidence-interval coverage and it
    does not establish population representativeness.
    """

    if not isinstance(final_report, FinalTestReport):
        raise TypeError("coverage diagnosis requires a FinalTestReport")
    if not isinstance(suite, HeldOutSuite):
        raise TypeError("coverage diagnosis suite must be a HeldOutSuite")
    if suite.role is not EvaluationRole.FINAL_TEST:
        raise ValueError("coverage diagnosis requires a final-test suite")
    if final_report.role is not EvaluationRole.FINAL_TEST:
        raise ValueError("coverage diagnosis requires a final-test report")
    if final_report.suite_name != suite.name:
        raise ValueError("final-test report and coverage suite names must match")

    expected_cases = tuple(
        (case.effective_name, case.scenario.id) for case in suite.cases
    )
    reported_cases = tuple(
        (case.name, case.scenario_id) for case in final_report.validation.cases
    )
    if expected_cases != reported_cases:
        raise ValueError("final-test report cases do not match the supplied suite")

    validation_manifest = final_report.validation.manifest
    if validation_manifest is None:
        raise ValueError("final-test validation is missing its manifest")
    recorded_cases = validation_manifest.inputs.get("cases")
    expected_case_inputs = tuple(
        {
            "name": case.effective_name,
            "scenario": scenario_identity(case.scenario),
            "seeds": tuple(case.seeds),
            "target_hash": stable_content_hash(case.target),
            "weights_hash": (
                None
                if case.weights is None
                else stable_content_hash(case.weights)
            ),
        }
        for case in suite.cases
    )
    if recorded_cases != expected_case_inputs:
        raise ValueError(
            "final-test validation inputs do not match the supplied suite"
        )

    declared_axes = tuple(axes)
    if not declared_axes:
        raise ValueError("coverage diagnosis requires at least one axis")
    if any(not isinstance(axis, CoverageAxis) for axis in declared_axes):
        raise TypeError("coverage axes must be CoverageAxis values")
    if len({axis.name for axis in declared_axes}) != len(declared_axes):
        raise ValueError("coverage axis names must be unique")

    evaluated_axes = tuple(_axis_coverage(axis, suite) for axis in declared_axes)
    seed_counter = Counter(
        seed
        for case in suite.cases
        for seed in case.seeds
    )
    reused = tuple(sorted(seed for seed, count in seed_counter.items() if count > 1))
    complete = all(axis.complete for axis in evaluated_axes)
    final_hash = required_manifest_hash(
        final_report,
        label="final-test coverage source report",
    )
    manifest = ExperimentManifest(
        stage=ExperimentStage.FINAL_TEST_COVERAGE,
        inputs={
            "suite_name": suite.name,
            "cases": tuple(
                {
                    "name": case.effective_name,
                    "scenario": scenario_identity(case.scenario),
                    "seeds": tuple(case.seeds),
                }
                for case in suite.cases
            ),
            "axes": tuple(axis.manifest_identity() for axis in declared_axes),
            "case_count": len(suite.cases),
            "total_seed_count": sum(seed_counter.values()),
            "unique_seed_count": len(seed_counter),
            "reused_seeds": reused,
            "complete": complete,
        },
        parent_hashes=(final_hash,),
    )
    return FinalTestCoverageReport(
        suite_name=suite.name,
        case_count=len(suite.cases),
        total_seed_count=sum(seed_counter.values()),
        unique_seed_count=len(seed_counter),
        reused_seeds=reused,
        axes=evaluated_axes,
        complete=complete,
        manifest=manifest,
    )


__all__ = [
    "AxisCoverage",
    "CoverageAxis",
    "CoverageBin",
    "FinalTestCoverageReport",
    "ScenarioCoverageAxis",
    "diagnose_final_test_coverage",
]

"""Closed registry for theorem-backed named receptivity schedules."""

from __future__ import annotations

from dataclasses import dataclass

from .model import ExactRat, ModelInputError, NamedSchedule, PathModel
from .result import ClaimStatus


LEAN_NAMESPACE = "NarrativeDynamics.FitnessABMPathNExposureConvergence"
FIXED_PATH2_CLAIM = "path2_consensus"


@dataclass(frozen=True)
class FixedFixtureRoute:
    schedule_id: str
    claim_id: str
    expected_status: ClaimStatus
    theorem: str
    n: int
    beliefs: tuple[ExactRat, ...]
    exposures: tuple[int, ...]
    threshold: ExactRat
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class NamedScheduleEntry:
    schedule_id: str
    lean_definition: str
    fixed_fixture: FixedFixtureRoute


def _qualified(name: str) -> str:
    return f"{LEAN_NAMESPACE}.{name}"


_SPLIT2_BELIEFS = (ExactRat(1, 1), ExactRat(0, 1))
_SPLIT2_EXPOSURES = (0, 0)
_ZERO_THRESHOLD = ExactRat(0, 1)
_FIXED_ASSUMPTIONS = (
    "n = 2",
    "initial beliefs = (1, 0)",
    "initial exposures = (0, 0)",
    "threshold = 0",
    "named schedule equals production fixture",
)


def _entry(
    schedule_id: str,
    theorem: str,
    status: ClaimStatus,
) -> NamedScheduleEntry:
    return NamedScheduleEntry(
        schedule_id=schedule_id,
        lean_definition=_qualified(schedule_id),
        fixed_fixture=FixedFixtureRoute(
            schedule_id=schedule_id,
            claim_id=FIXED_PATH2_CLAIM,
            expected_status=status,
            theorem=_qualified(theorem),
            n=2,
            beliefs=_SPLIT2_BELIEFS,
            exposures=_SPLIT2_EXPOSURES,
            threshold=_ZERO_THRESHOLD,
            assumptions=_FIXED_ASSUMPTIONS,
        ),
    )


_REGISTRY: dict[str, NamedScheduleEntry] = {
    "slowZeroSchedule": _entry(
        "slowZeroSchedule",
        "slowZero_not_consensus",
        ClaimStatus.DISPROVED,
    ),
    "nearOneSchedule": _entry(
        "nearOneSchedule",
        "nearOne_not_convergent",
        ClaimStatus.DISPROVED,
    ),
    "harmonicSchedule": _entry(
        "harmonicSchedule",
        "harmonic_consensus",
        ClaimStatus.PROVED,
    ),
}


def resolve_named_schedule(schedule_id: str) -> NamedScheduleEntry:
    try:
        return _REGISTRY[schedule_id]
    except KeyError as exc:
        raise ModelInputError(f"unsupported named schedule: {schedule_id}") from exc


def fixed_fixture_route(model: PathModel) -> FixedFixtureRoute | None:
    if not isinstance(model.schedule, NamedSchedule):
        return None
    entry = resolve_named_schedule(model.schedule.schedule_id)
    route = entry.fixed_fixture
    if model.n != route.n:
        return None
    if model.beliefs != route.beliefs:
        return None
    if model.exposures != route.exposures:
        return None
    if model.threshold != route.threshold:
        return None
    return route

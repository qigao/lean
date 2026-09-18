"""Syntactic recognition for the closed schedule family language."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from .model import (
    AlternatingSchedule,
    ConstantSchedule,
    ExactRat,
    ExponentialSchedule,
    HarmonicSchedule,
    NamedSchedule,
    PeriodicSchedule,
    PiecewiseSchedule,
    PolynomialSchedule,
    Schedule,
)


class FamilyRecognitionStatus(Enum):
    RECOGNIZED = "RECOGNIZED"
    UNRECOGNIZED = "UNRECOGNIZED"


@dataclass(frozen=True)
class ScheduleFamilyInfo:
    status: FamilyRecognitionStatus
    family_id: str
    canonical_family_id: str
    exact_parameters: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exact_parameters",
            MappingProxyType(dict(self.exact_parameters)),
        )


def _rat(value: ExactRat) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def recognize_family(schedule: Schedule) -> ScheduleFamilyInfo:
    if isinstance(schedule, HarmonicSchedule):
        return ScheduleFamilyInfo(
            status=FamilyRecognitionStatus.RECOGNIZED,
            family_id="harmonic",
            canonical_family_id="polynomial",
            exact_parameters={
                "c": _rat(schedule.c),
                "offset": str(schedule.offset),
                "p": "1",
                "target": schedule.target.value,
            },
        )

    if isinstance(schedule, PolynomialSchedule):
        return ScheduleFamilyInfo(
            status=FamilyRecognitionStatus.RECOGNIZED,
            family_id="polynomial",
            canonical_family_id="polynomial",
            exact_parameters={
                "c": _rat(schedule.c),
                "offset": str(schedule.offset),
                "p": str(schedule.p),
                "target": schedule.target.value,
            },
        )

    if isinstance(schedule, ExponentialSchedule):
        return ScheduleFamilyInfo(
            status=FamilyRecognitionStatus.RECOGNIZED,
            family_id="exponential",
            canonical_family_id="exponential",
            exact_parameters={
                "base": _rat(schedule.base),
                "c": _rat(schedule.c),
                "offset": str(schedule.offset),
                "target": schedule.target.value,
            },
        )

    if isinstance(schedule, PeriodicSchedule):
        return ScheduleFamilyInfo(
            status=FamilyRecognitionStatus.RECOGNIZED,
            family_id="periodic",
            canonical_family_id="periodic",
            exact_parameters={"values": ",".join(_rat(value) for value in schedule.values)},
        )

    if isinstance(schedule, AlternatingSchedule):
        return ScheduleFamilyInfo(
            status=FamilyRecognitionStatus.RECOGNIZED,
            family_id="alternating",
            canonical_family_id="periodic",
            exact_parameters={"a": _rat(schedule.a), "b": _rat(schedule.b)},
        )

    if isinstance(schedule, PiecewiseSchedule):
        return ScheduleFamilyInfo(
            status=FamilyRecognitionStatus.RECOGNIZED,
            family_id="piecewise",
            canonical_family_id="piecewise_constant_tail",
            exact_parameters={
                "default": _rat(schedule.default),
                "points": ",".join(
                    f"{exposure}:{_rat(value)}"
                    for exposure, value in schedule.points
                ),
            },
        )

    if isinstance(schedule, ConstantSchedule):
        return ScheduleFamilyInfo(
            status=FamilyRecognitionStatus.RECOGNIZED,
            family_id="constant",
            canonical_family_id="constant",
            exact_parameters={"value": _rat(schedule.value)},
        )

    if isinstance(schedule, NamedSchedule):
        return ScheduleFamilyInfo(
            status=FamilyRecognitionStatus.RECOGNIZED,
            family_id="named",
            canonical_family_id="named",
            exact_parameters={"id": schedule.schedule_id},
        )

    return ScheduleFamilyInfo(
        status=FamilyRecognitionStatus.UNRECOGNIZED,
        family_id=type(schedule).__name__,
        canonical_family_id="unrecognized",
        exact_parameters={},
    )

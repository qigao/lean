import unittest

from narrative_analyzer.families import (
    FamilyRecognitionStatus,
    recognize_family,
)
from narrative_analyzer.model import (
    AlternatingSchedule,
    ConstantSchedule,
    DecayTarget,
    ExactRat,
    ExponentialSchedule,
    HarmonicSchedule,
    NamedSchedule,
    PeriodicSchedule,
    PiecewiseSchedule,
    PolynomialSchedule,
)


class FamilyRecognitionTests(unittest.TestCase):
    def test_closed_families_have_stable_canonical_ids(self) -> None:
        cases = (
            (
                HarmonicSchedule(ExactRat(1, 2), 1, DecayTarget.ZERO),
                "harmonic",
                "polynomial",
                {"c": "1/2", "offset": "1", "p": "1", "target": "zero"},
            ),
            (
                PolynomialSchedule(ExactRat(1, 3), 2, 2, DecayTarget.ONE),
                "polynomial",
                "polynomial",
                {"c": "1/3", "offset": "2", "p": "2", "target": "one"},
            ),
            (
                ExponentialSchedule(
                    ExactRat(1, 4), ExactRat(1, 2), 0, DecayTarget.ZERO
                ),
                "exponential",
                "exponential",
                {"base": "1/2", "c": "1/4", "offset": "0", "target": "zero"},
            ),
            (
                PeriodicSchedule((ExactRat(1, 4), ExactRat(3, 4))),
                "periodic",
                "periodic",
                {"values": "1/4,3/4"},
            ),
            (
                AlternatingSchedule(ExactRat(1, 4), ExactRat(3, 4)),
                "alternating",
                "periodic",
                {"a": "1/4", "b": "3/4"},
            ),
            (
                PiecewiseSchedule(
                    ExactRat(1, 4),
                    ((0, ExactRat(1, 3)), (2, ExactRat(1, 5))),
                ),
                "piecewise",
                "piecewise_constant_tail",
                {"default": "1/4", "points": "0:1/3,2:1/5"},
            ),
            (
                ConstantSchedule(ExactRat(1, 4)),
                "constant",
                "constant",
                {"value": "1/4"},
            ),
            (
                NamedSchedule("harmonicSchedule"),
                "named",
                "named",
                {"id": "harmonicSchedule"},
            ),
        )
        for schedule, family_id, canonical, parameters in cases:
            with self.subTest(family_id=family_id):
                info = recognize_family(schedule)
                self.assertEqual(info.status, FamilyRecognitionStatus.RECOGNIZED)
                self.assertEqual(info.family_id, family_id)
                self.assertEqual(info.canonical_family_id, canonical)
                self.assertEqual(dict(info.exact_parameters), parameters)
                with self.assertRaises(TypeError):
                    info.exact_parameters["x"] = "y"  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()

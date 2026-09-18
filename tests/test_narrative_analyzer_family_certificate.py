import unittest

from narrative_analyzer.family_certificate import _render_schedule
from narrative_analyzer.model import (
    AlternatingSchedule,
    DecayTarget,
    ExactRat,
    ExponentialSchedule,
    HarmonicSchedule,
    ModelInputError,
    PeriodicSchedule,
    PolynomialSchedule,
    parse_model,
)


def model_document(schedule: dict[str, object]) -> dict[str, object]:
    return {
        "topology": {"kind": "path", "n": 2},
        "initial": {"beliefs": ["1", "0"], "exposures": [0, 0]},
        "dynamics": {"threshold": "0"},
        "schedule": schedule,
    }


class FamilyScheduleRenderingTests(unittest.TestCase):
    def test_closed_family_renderers_use_only_fixed_lean_forms(self) -> None:
        cases = (
            (
                HarmonicSchedule(ExactRat(1, 2), 1, DecayTarget.ZERO),
                "harmonicReceptivity ((1 : Rat) / 2) 1 DecayTarget.zero",
            ),
            (
                PolynomialSchedule(ExactRat(1, 3), 2, 2, DecayTarget.ONE),
                "polynomialReceptivity ((1 : Rat) / 3) 2 2 DecayTarget.one",
            ),
            (
                ExponentialSchedule(
                    ExactRat(1, 4), ExactRat(1, 2), 0, DecayTarget.ZERO
                ),
                "exponentialReceptivity ((1 : Rat) / 4) ((1 : Rat) / 2) 0 DecayTarget.zero",
            ),
            (
                AlternatingSchedule(ExactRat(1, 4), ExactRat(3, 4)),
                "alternatingReceptivity ((1 : Rat) / 4) ((3 : Rat) / 4)",
            ),
            (
                PeriodicSchedule((ExactRat(1, 4), ExactRat(3, 4))),
                "periodicReceptivity 2 (by norm_num) ![((1 : Rat) / 4), ((3 : Rat) / 4)]",
            ),
        )
        for schedule, expected in cases:
            with self.subTest(schedule=type(schedule).__name__):
                rendered = _render_schedule(schedule)
                self.assertEqual(rendered, expected)
                for forbidden in (
                    "0.25",
                    "0.5",
                    "import ",
                    "axiom ",
                    "theorem ",
                    "sorry",
                    "admit",
                    "unsafe",
                    "native_decide",
                ):
                    self.assertNotIn(forbidden, rendered)

    def test_renderer_uses_canonical_exact_rationals(self) -> None:
        rendered = _render_schedule(
            PolynomialSchedule(ExactRat(2, 4), 2, 3, DecayTarget.ZERO)
        )
        self.assertEqual(
            rendered,
            "polynomialReceptivity ((1 : Rat) / 2) 2 3 DecayTarget.zero",
        )

    def test_source_like_family_kind_fails_before_rendering(self) -> None:
        with self.assertRaises(ModelInputError):
            parse_model(
                model_document(
                    {
                        "kind": "polynomial; import Evil",
                        "c": "1/3",
                        "p": 2,
                        "offset": 2,
                        "target": "zero",
                    }
                )
            )

    def test_source_like_target_fails_before_rendering(self) -> None:
        with self.assertRaises(ModelInputError):
            parse_model(
                model_document(
                    {
                        "kind": "exponential",
                        "c": "1/4",
                        "base": "1/2",
                        "offset": 0,
                        "target": "zero); axiom hacked : False",
                    }
                )
            )


if __name__ == "__main__":
    unittest.main()

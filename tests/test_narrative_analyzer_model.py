import tempfile
import unittest
from pathlib import Path

from narrative_analyzer.model import (
    ConstantSchedule,
    ExactRat,
    ModelInputError,
    NamedSchedule,
    PiecewiseSchedule,
    load_model,
    parse_model,
)


def base_document() -> dict[str, object]:
    return {
        "topology": {"kind": "path", "n": 3},
        "initial": {
            "beliefs": ["1", "1/2", "0"],
            "exposures": [0, 0, 0],
        },
        "dynamics": {"threshold": "0"},
        "schedule": {"kind": "constant", "value": "1/4"},
    }


class ExactRatTests(unittest.TestCase):
    def test_accepts_integer_and_fraction_strings(self) -> None:
        model = parse_model(base_document())
        self.assertEqual(model.beliefs[0], ExactRat(1, 1))
        self.assertEqual(model.beliefs[1], ExactRat(1, 2))
        self.assertEqual(model.threshold, ExactRat(0, 1))

    def test_canonicalizes_sign_and_gcd(self) -> None:
        document = base_document()
        document["initial"] = {
            "beliefs": ["+2/4", "-6/8", "0"],
            "exposures": [0, 0, 0],
        }
        model = parse_model(document)
        self.assertEqual(model.beliefs[0], ExactRat(1, 2))
        self.assertEqual(model.beliefs[1], ExactRat(-3, 4))

    def test_rejects_toml_float(self) -> None:
        document = base_document()
        document["dynamics"] = {"threshold": 0.5}
        with self.assertRaises(ModelInputError):
            parse_model(document)

    def test_rejects_decimal_string(self) -> None:
        document = base_document()
        document["schedule"] = {"kind": "constant", "value": "0.25"}
        with self.assertRaises(ModelInputError):
            parse_model(document)

    def test_rejects_scientific_notation(self) -> None:
        document = base_document()
        document["schedule"] = {"kind": "constant", "value": "1e-3"}
        with self.assertRaises(ModelInputError):
            parse_model(document)

    def test_rejects_zero_denominator(self) -> None:
        document = base_document()
        document["schedule"] = {"kind": "constant", "value": "1/0"}
        with self.assertRaises(ModelInputError):
            parse_model(document)


class PathModelTests(unittest.TestCase):
    def test_constant_schedule_parses(self) -> None:
        model = parse_model(base_document())
        self.assertEqual(model.n, 3)
        self.assertEqual(model.exposures, (0, 0, 0))
        self.assertEqual(model.schedule, ConstantSchedule(ExactRat(1, 4)))

    def test_piecewise_points_are_sorted(self) -> None:
        document = base_document()
        document["schedule"] = {
            "kind": "piecewise",
            "default": "1/4",
            "points": [
                {"exposure": 9, "value": "1/5"},
                {"exposure": 0, "value": "1/3"},
                {"exposure": 4, "value": "2/5"},
            ],
        }
        model = parse_model(document)
        self.assertEqual(
            model.schedule,
            PiecewiseSchedule(
                default=ExactRat(1, 4),
                points=(
                    (0, ExactRat(1, 3)),
                    (4, ExactRat(2, 5)),
                    (9, ExactRat(1, 5)),
                ),
            ),
        )

    def test_duplicate_piecewise_exposure_is_rejected(self) -> None:
        document = base_document()
        document["schedule"] = {
            "kind": "piecewise",
            "default": "1/4",
            "points": [
                {"exposure": 4, "value": "1/3"},
                {"exposure": 4, "value": "2/5"},
            ],
        }
        with self.assertRaises(ModelInputError):
            parse_model(document)

    def test_named_schedule_parses_as_data_only(self) -> None:
        document = base_document()
        document["schedule"] = {"kind": "named", "id": "harmonicSchedule"}
        model = parse_model(document)
        self.assertEqual(model.schedule, NamedSchedule("harmonicSchedule"))

    def test_n_less_than_two_is_rejected(self) -> None:
        document = base_document()
        document["topology"] = {"kind": "path", "n": 1}
        document["initial"] = {"beliefs": ["1"], "exposures": [0]}
        with self.assertRaises(ModelInputError):
            parse_model(document)

    def test_belief_length_mismatch_is_rejected(self) -> None:
        document = base_document()
        document["initial"] = {"beliefs": ["1", "0"], "exposures": [0, 0, 0]}
        with self.assertRaises(ModelInputError):
            parse_model(document)

    def test_exposure_length_mismatch_is_rejected(self) -> None:
        document = base_document()
        document["initial"] = {"beliefs": ["1", "1/2", "0"], "exposures": [0, 0]}
        with self.assertRaises(ModelInputError):
            parse_model(document)

    def test_negative_exposure_is_rejected(self) -> None:
        document = base_document()
        document["initial"] = {"beliefs": ["1", "1/2", "0"], "exposures": [0, -1, 0]}
        with self.assertRaises(ModelInputError):
            parse_model(document)

    def test_unsupported_topology_is_rejected(self) -> None:
        document = base_document()
        document["topology"] = {"kind": "cycle", "n": 3}
        with self.assertRaises(ModelInputError):
            parse_model(document)

    def test_unsupported_schedule_kind_is_rejected(self) -> None:
        document = base_document()
        document["schedule"] = {"kind": "formula", "value": "1/4"}
        with self.assertRaises(ModelInputError):
            parse_model(document)

    def test_load_model_reads_toml(self) -> None:
        source = '''
[topology]
kind = "path"
n = 2

[initial]
beliefs = ["1", "0"]
exposures = [0, 0]

[dynamics]
threshold = "0"

[schedule]
kind = "constant"
value = "1/3"
'''
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.toml"
            path.write_text(source, encoding="utf-8")
            model = load_model(path)
        self.assertEqual(model.n, 2)
        self.assertEqual(model.schedule, ConstantSchedule(ExactRat(1, 3)))


if __name__ == "__main__":
    unittest.main()

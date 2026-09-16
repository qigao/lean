"""RED consumers for the exact BB rational conformance schema."""

from fractions import Fraction
import json
import unittest

from tools.bb_rational_conformance import (
    SCHEMA,
    canonical_json_line,
    parse_rat,
    rat,
)


class BBRationalConformanceSchemaTests(unittest.TestCase):
    def test_rational_codec_is_structural_and_reduced(self):
        self.assertEqual(SCHEMA, "bb-rational-conformance/v1")
        self.assertEqual(rat(Fraction(-6, 16)), {"num": -3, "den": 8})
        self.assertEqual(rat(0), {"num": 0, "den": 1})
        self.assertEqual(parse_rat({"num": -3, "den": 8}), Fraction(-3, 8))

    def test_parse_rat_rejects_float(self):
        with self.assertRaises((TypeError, ValueError)):
            parse_rat(0.5)

    def test_parse_rat_rejects_boolean_components(self):
        for value in (
            {"num": True, "den": 1},
            {"num": 1, "den": True},
        ):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    parse_rat(value)

    def test_parse_rat_rejects_invalid_shape(self):
        for value in (
            {"num": 1},
            {"den": 1},
            {"num": 1, "den": 2, "extra": 0},
            [1, 2],
        ):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    parse_rat(value)

    def test_parse_rat_rejects_nonpositive_denominator(self):
        for value in (
            {"num": 1, "den": 0},
            {"num": 1, "den": -2},
        ):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    parse_rat(value)

    def test_parse_rat_rejects_nonreduced_pair(self):
        with self.assertRaises((TypeError, ValueError)):
            parse_rat({"num": 2, "den": 4})

    def test_canonical_json_line_is_compact_sorted_and_terminated(self):
        line = canonical_json_line(
            {"schema": SCHEMA, "z": 1, "a": rat(Fraction(1, 2))}
        )
        self.assertTrue(line.endswith("\n"))
        self.assertNotIn(" ", line)
        self.assertEqual(
            json.loads(line),
            {"a": {"den": 2, "num": 1}, "schema": SCHEMA, "z": 1},
        )
        self.assertLess(line.index('"a"'), line.index('"schema"'))
        self.assertLess(line.index('"schema"'), line.index('"z"'))


if __name__ == "__main__":
    unittest.main()

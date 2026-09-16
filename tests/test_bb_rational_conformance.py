"""TDD consumers for the exact BB rational conformance contract."""

from fractions import Fraction
import json
import unittest

from tools.bb_rational_conformance import (
    SCHEMA,
    canonical_json_line,
    evaluate_named,
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


class BBRationalConformanceEvaluatorTests(unittest.TestCase):
    def test_inclusive_threshold_broadcasts_and_updates_receiver_exactly(self):
        out = evaluate_named("inclusive-threshold")
        self.assertEqual(out["expected"]["broadcasting"], [True, False])
        self.assertEqual(
            out["expected"]["beliefs"],
            [rat(Fraction(1, 2)), rat(Fraction(1, 4))],
        )
        self.assertEqual(out["expected"]["exposures"], [0, 1])

    def test_no_broadcast_preserves_beliefs_and_exposures(self):
        out = evaluate_named("no-broadcast")
        self.assertEqual(
            out["expected"]["beliefs"],
            [rat(Fraction(1, 4)), rat(Fraction(1, 8))],
        )
        self.assertEqual(out["expected"]["exposures"], [0, 0])

    def test_nontrivial_mean_is_exact(self):
        out = evaluate_named("nontrivial-mean")
        self.assertEqual(parse_rat(out["expected"]["beliefs"][1]), Fraction(2, 3))
        for value in out["expected"]["beliefs"]:
            self.assertIsInstance(parse_rat(value), Fraction)

    def test_successive_birth_mass_is_one_eighth(self):
        out = evaluate_named("bb-successive-births")
        self.assertEqual(parse_rat(out["expected"]["trace_mass"]), Fraction(1, 8))
        self.assertEqual(out["expected"]["node_count"], 4)

    def test_duplicate_target_has_stable_error(self):
        out = evaluate_named("duplicate-target-error")
        self.assertEqual(
            out["expected"]["error"],
            {"stage": "tick_network", "code": "duplicateTarget", "field": "targets"},
        )


if __name__ == "__main__":
    unittest.main()

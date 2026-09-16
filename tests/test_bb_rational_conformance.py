"""TDD consumers for the exact BB rational conformance contract."""

from fractions import Fraction
import json
from pathlib import Path
import tempfile
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


class BBRationalConformanceGeneratorTests(unittest.TestCase):
    def test_contract_hash_is_stable_sha256(self):
        from tools.bb_rational_conformance import contract_hash

        first = contract_hash()
        second = contract_hash()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertTrue(all(ch in "0123456789abcdef" for ch in first))

    def test_generator_is_byte_deterministic_and_provenanced(self):
        from tools.generate_bb_rational_conformance import generate_records, write_corpus

        records = generate_records()
        self.assertEqual(len(records), 5)
        self.assertEqual(
            [record["case_id"] for record in records],
            sorted(record["case_id"] for record in records),
        )

        for record in records:
            self.assertEqual(
                set(record),
                {"schema", "case_id", "kind", "provenance", "input", "expected"},
            )
            self.assertEqual(record["schema"], SCHEMA)
            provenance = record["provenance"]
            self.assertEqual(provenance["schema"], SCHEMA)
            self.assertEqual(
                provenance["generator"],
                "tools/generate_bb_rational_conformance.py",
            )
            self.assertEqual(
                provenance["lean_checker"],
                "NarrativeDynamics/Conformance/BBRationalConformance.lean",
            )
            self.assertEqual(provenance["case_set"], "v1")
            self.assertEqual(len(provenance["generator_contract"]), 64)
            self.assertTrue(
                all(ch in "0123456789abcdef" for ch in provenance["generator_contract"])
            )

        by_id = {record["case_id"]: record for record in records}
        for case_id in ("inclusive-threshold", "no-broadcast", "nontrivial-mean"):
            self.assertEqual(
                set(by_id[case_id]["input"]),
                {"node_count", "edges", "alpha", "threshold", "beliefs", "exposures"},
            )

        replay_input = by_id["bb-successive-births"]["input"]
        self.assertEqual(set(replay_input), {"m", "seed_fitness", "seed_edges", "births"})
        self.assertEqual(replay_input["m"], 1)
        self.assertEqual(
            replay_input["births"],
            [
                {"fitness": rat(1), "targets": [1]},
                {"fitness": rat(1), "targets": [2]},
            ],
        )
        self.assertEqual(
            {key: by_id["bb-successive-births"]["expected"][key]
             for key in ("node_count", "birth_count", "tick_count")},
            {"node_count": 4, "birth_count": 2, "tick_count": 2},
        )

        error_input = by_id["duplicate-target-error"]["input"]
        self.assertEqual(set(error_input), {"m", "seed_fitness", "seed_edges", "birth"})
        self.assertEqual(error_input["m"], 2)
        self.assertEqual(error_input["birth"]["targets"], [0, 0])

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            write_corpus(first)
            write_corpus(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertTrue(first.read_bytes().endswith(b"\n"))
            self.assertEqual(len(first.read_text(encoding="utf-8").splitlines()), 5)


if __name__ == "__main__":
    unittest.main()

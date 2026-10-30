"""Native metrics must preserve family-specific units and real source hashes."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

from narrative_dynamics.abm.native_metric_protocol import (
    NativeMetric, NativeMetricProtocol, build_native_metric_protocol,
    metric_pair_eligibility,
)
from tests.test_network_model_family_comparison import build_family_fixture
from tools.render_native_metric_protocol_vectors import render_vector_corpus


class NativeMetricProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = build_family_fixture()
        cls.audit = build_native_metric_protocol(cls.source)
        cls.data = cls.audit.to_dict()

    def test_native_denominators_are_not_a_shared_metric(self):
        d = self.data
        self.assertEqual(d["source_report_hash"], self.source.content_hash)
        self.assertEqual(d["normalization"], "NONE")
        self.assertEqual(d["cross_family_numeric_ranking"], "FORBIDDEN")
        self.assertEqual(d["sample_size_per_native_run"], 1)
        self.assertEqual(d["confidence_interval"], "NOT_ESTIMABLE")
        bb = d["families"]["bb_fitness_growth"]
        rew = d["families"]["directed_rewiring"]
        evo = d["families"]["lifecycle_feedback"]
        self.assertEqual([(x["numerator"], x["denominator"], x["exact_fraction"])
                          for x in bb], [(2, 3, "2/3"), (2, 3, "2/3")])
        self.assertEqual([x["exact_fraction"] for x in rew], ["1", "1"])
        self.assertEqual([x["exact_fraction"] for x in evo], ["1", "1"])
        self.assertEqual(rew[0]["denominator_kind"], "FIXED_DIRECTED_CANDIDATE_CATALOG")
        self.assertEqual(evo[0]["denominator_kind"],
                         "ALL_NATIVE_CATALOG_MEMBERS_INCLUDING_INACTIVE")
        self.assertNotEqual(bb[0]["unit"], rew[0]["unit"])
        self.assertNotEqual(rew[0]["unit"], evo[0]["unit"])

    def test_only_same_model_run_unit_is_supported_descriptively(self):
        groups = [[NativeMetric(**x) for x in self.data["families"][family]]
                  for family in ("bb_fitness_growth", "directed_rewiring",
                                 "lifecycle_feedback")]
        for rows in groups:
            self.assertEqual(metric_pair_eligibility(rows[0], rows[1]), "SUPPORTED")
        for left, right in ((groups[0][0], groups[1][0]),
                            (groups[0][0], groups[2][0]),
                            (groups[1][0], groups[2][0])):
            self.assertEqual(metric_pair_eligibility(left, right), "NOT_COMPARABLE")
        self.assertTrue(all(x["status"] == "NOT_COMPARABLE"
                            for x in self.data["comparison_eligibility"].values()))
        self.assertEqual(self.data["v19_reference"]["metric_status"],
                         "NOT_MEASURABLE_FROM_M4B_REFERENCE")

    def test_bad_units_mappings_and_zero_denominators_fail_closed(self):
        bb = NativeMetric(**self.data["families"]["bb_fitness_growth"][0])
        rew = NativeMetric(**self.data["families"]["directed_rewiring"][0])
        with self.assertRaisesRegex(ValueError, "approved mapping"):
            metric_pair_eligibility(bb, rew, mapping_version="fake-v1")
        for change in (
            {"numerator": 4}, {"denominator": 0}, {"exact_fraction": "0.99"},
            {"unit": "universal_edge"}, {"phase": "V19_STORY"},
            {"run_source_hash": "fake"}, {"native_time": -1},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(bb, **change)
        with self.assertRaises(TypeError):
            metric_pair_eligibility(object(), rew)

    def test_manifest_rejects_forged_comparability(self):
        for key, val in (
            ("normalization", "UNIVERSAL_SCORE"),
            ("cross_family_numeric_ranking", "BB_WINS"),
            ("confidence_interval", "95%"),
            ("sample_size_per_native_run", 10),
        ):
            x = self.audit.to_dict()
            x[key] = val
            with self.subTest(key=key), self.assertRaises(ValueError):
                NativeMetricProtocol(json.dumps(x, sort_keys=True))
        x = self.audit.to_dict()
        x["comparison_eligibility"]["bb_vs_rewiring"]["status"] = "SUPPORTED"
        with self.assertRaises(ValueError):
            NativeMetricProtocol(json.dumps(x))
        x = self.audit.to_dict()
        x["families"]["bb_fitness_growth"][0]["unit"] = "directed_trust"
        with self.assertRaises(ValueError):
            NativeMetricProtocol(json.dumps(x))
        with self.assertRaises(TypeError):
            build_native_metric_protocol(object())

    def test_source_regenerated_vector_is_reviewed_exact_bytes(self):
        pinned = (Path(__file__).resolve().parents[1] /
                  "conformance/native_metric_protocol_v1.json")
        self.assertEqual(pinned.read_bytes(), render_vector_corpus().encode("utf-8"))
        self.assertEqual(self.audit.to_dict(),
                         build_native_metric_protocol(build_family_fixture()).to_dict())
        corpus = json.dumps(self.data, sort_keys=True)
        self.assertNotIn('"message":', corpus)


if __name__ == "__main__":
    unittest.main()

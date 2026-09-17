import shutil
import unittest
from pathlib import Path

from narrative_analyzer.analyze import analyze_model
from narrative_analyzer.certificate import Certificate, CertificateBuilder
from narrative_analyzer.model import ModelInputError, load_model, parse_model
from narrative_analyzer.result import CertificateCompileError, ClaimStatus
from narrative_analyzer.runner import LeanCertificateRunner

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "narrative_analyzer"
HAS_LEAN_CHECKOUT = shutil.which("lake") is not None and (ROOT / "lakefile.toml").exists()


def claims(result):
    return {claim.claim_id: claim for claim in result.claims}


def constant_path2():
    return parse_model({
        "topology": {"kind": "path", "n": 2},
        "initial": {"beliefs": ["1", "0"], "exposures": [0, 0]},
        "dynamics": {"threshold": "0"},
        "schedule": {"kind": "constant", "value": "1/4"},
    })


@unittest.skipUnless(HAS_LEAN_CHECKOUT, "real certificate golden tests require Lean checkout")
class RealCertificateGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = LeanCertificateRunner(ROOT)

    def analyze_fixture(self, name):
        return analyze_model(load_model(FIXTURES / name), runner=self.runner)

    def test_constant_path2_generic_consensus_certificate_compiles(self):
        result = analyze_model(constant_path2(), runner=self.runner)
        by_id = claims(result)
        self.assertIs(by_id["pathn_consensus_exists"].status, ClaimStatus.PROVED)
        self.assertIs(by_id["path2_consensus"].status, ClaimStatus.UNKNOWN)

    def test_constant_pathn_consensus_certificate_compiles(self):
        result = self.analyze_fixture("constant_path7.toml")
        self.assertIs(claims(result)["pathn_consensus_exists"].status, ClaimStatus.PROVED)

    def test_piecewise_pathn_consensus_certificate_compiles(self):
        result = self.analyze_fixture("piecewise_path7.toml")
        by_id = claims(result)
        self.assertIs(by_id["reachable_interior"].status, ClaimStatus.PROVED)
        self.assertEqual(by_id["reachable_interior"].exact_values["eps"], "1/4")

    def test_post_incoming_lookup_certificate_compiles(self):
        model = load_model(FIXTURES / "constant_path7.toml")
        certificate = CertificateBuilder().build_structural_positive(
            model, ("effective_alpha_lookup",)
        )
        evidence = self.runner.compile(certificate)
        self.assertEqual(evidence.returncode, 0)
        self.assertIn("(k + 1) * FitnessABMPathN.degree", certificate.source)
        self.assertEqual(
            certificate.claims[0].exact_values["lookup"],
            "e0(i) + (k+1) * degree(i)",
        )

    def test_harmonic_path2_positive_fixture_compiles(self):
        result = self.analyze_fixture("harmonic_path2.toml")
        by_id = claims(result)
        self.assertIs(by_id["path2_consensus"].status, ClaimStatus.PROVED)
        self.assertEqual(by_id["consensus_value_known"].exact_values["value"], "1/2")

    def test_slow_zero_path2_negative_fixture_compiles(self):
        result = self.analyze_fixture("slow_zero_path2.toml")
        self.assertIs(claims(result)["path2_consensus"].status, ClaimStatus.DISPROVED)

    def test_near_one_path2_negative_fixture_compiles(self):
        result = self.analyze_fixture("near_one_path2.toml")
        self.assertIs(claims(result)["path2_consensus"].status, ClaimStatus.DISPROVED)

    def test_unknown_named_path2_remains_unknown(self):
        result = self.analyze_fixture("unknown_path2.toml")
        by_id = claims(result)
        self.assertIs(by_id["path2_consensus"].status, ClaimStatus.UNKNOWN)
        self.assertIs(by_id["pathn_consensus_exists"].status, ClaimStatus.UNKNOWN)

    def test_noninterior_sufficient_failure_is_not_disproved(self):
        model = parse_model({
            "topology": {"kind": "path", "n": 3},
            "initial": {"beliefs": ["1", "1", "1"], "exposures": [0, 0, 0]},
            "dynamics": {"threshold": "0"},
            "schedule": {"kind": "constant", "value": "0"},
        })
        result = analyze_model(model, runner=self.runner)
        self.assertIs(claims(result)["pathn_consensus_exists"].status, ClaimStatus.UNKNOWN)

    def test_generic_pathn_consensus_value_stays_unknown(self):
        result = self.analyze_fixture("constant_path7.toml")
        self.assertIs(claims(result)["consensus_value_known"].status, ClaimStatus.UNKNOWN)

    def test_corrupted_certificate_is_failure_not_unknown(self):
        bad = Certificate(
            "import NarrativeDynamics.Core.FitnessABMPathNExposureConvergence\nexample : False := by exact True.intro\n",
            (),
        )
        with self.assertRaises(CertificateCompileError):
            self.runner.compile(bad)


class GoldenSecurityAndInputTests(unittest.TestCase):
    def test_toml_float_is_rejected(self):
        with self.assertRaises(ModelInputError):
            parse_model({
                "topology": {"kind": "path", "n": 2},
                "initial": {"beliefs": ["1", 0.5], "exposures": [0, 0]},
                "dynamics": {"threshold": "0"},
                "schedule": {"kind": "constant", "value": "1/4"},
            })

    def test_malformed_rational_and_length_mismatch_are_rejected(self):
        for beliefs in (["1", "1e-3"], ["1"]):
            with self.subTest(beliefs=beliefs), self.assertRaises(ModelInputError):
                parse_model({
                    "topology": {"kind": "path", "n": 2},
                    "initial": {"beliefs": beliefs, "exposures": [0, 0]},
                    "dynamics": {"threshold": "0"},
                    "schedule": {"kind": "constant", "value": "1/4"},
                })

    def test_unsupported_and_duplicate_schedule_forms_are_rejected(self):
        documents = [
            {"kind": "formula", "value": "1/4"},
            {"kind": "piecewise", "default": "1/4", "points": [
                {"exposure": 1, "value": "1/3"},
                {"exposure": 1, "value": "1/5"},
            ]},
        ]
        for schedule in documents:
            with self.subTest(schedule=schedule), self.assertRaises(ModelInputError):
                parse_model({
                    "topology": {"kind": "path", "n": 2},
                    "initial": {"beliefs": ["1", "0"], "exposures": [0, 0]},
                    "dynamics": {"threshold": "0"},
                    "schedule": schedule,
                })

    def test_source_like_named_schedule_is_rejected_before_rendering(self):
        model = parse_model({
            "topology": {"kind": "path", "n": 2},
            "initial": {"beliefs": ["1", "0"], "exposures": [0, 0]},
            "dynamics": {"threshold": "0"},
            "schedule": {"kind": "named", "id": "harmonicSchedule; axiom bad : False"},
        })
        with self.assertRaises(ModelInputError):
            CertificateBuilder().build_structural_positive(
                model, ("parameters_valid",)
            )

    def test_generated_source_has_no_proof_escape_hatch(self):
        certificate = CertificateBuilder().build_structural_positive(
            constant_path2(),
            (
                "parameters_valid",
                "initial_all_broadcast",
                "exposure_law",
                "effective_alpha_lookup",
            ),
        )
        for forbidden in ("sorry", "admit", "axiom", "unsafe", "native_decide"):
            self.assertNotIn(forbidden, certificate.source)


if __name__ == "__main__":
    unittest.main()

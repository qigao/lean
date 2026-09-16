import re
import unittest

from narrative_analyzer.certificate import CertificateBuilder
from narrative_analyzer.model import ModelInputError, parse_model
from narrative_analyzer.result import ClaimStatus


STRUCTURAL = (
    "parameters_valid",
    "initial_all_broadcast",
    "exposure_law",
    "effective_alpha_lookup",
)


def model_document(
    *,
    n: int = 3,
    beliefs: list[str] | None = None,
    exposures: list[int] | None = None,
    threshold: str = "0",
    schedule: dict[str, object] | None = None,
):
    if beliefs is None:
        beliefs = ["1"] * n
    if exposures is None:
        exposures = [0] * n
    if schedule is None:
        schedule = {"kind": "constant", "value": "1/4"}
    return {
        "topology": {"kind": "path", "n": n},
        "initial": {"beliefs": beliefs, "exposures": exposures},
        "dynamics": {"threshold": threshold},
        "schedule": schedule,
    }


class CertificateRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = CertificateBuilder()

    def test_exact_rationals_render_canonically_without_floats(self) -> None:
        model = parse_model(
            model_document(beliefs=["1", "-6/8", "0"], threshold="-1/2")
        )
        certificate = self.builder.build_structural_positive(
            model, ("initial_all_broadcast",)
        )
        self.assertIn("((-3 : Rat) / 4)", certificate.source)
        self.assertIn("((-1 : Rat) / 2)", certificate.source)
        self.assertNotIn("0.75", certificate.source)
        self.assertNotIn("-0.5", certificate.source)

    def test_generated_state_vector_has_exactly_n_entries(self) -> None:
        model = parse_model(model_document(n=5))
        source = self.builder.build_structural_positive(
            model, ("parameters_valid",)
        ).source
        state_line = next(
            line for line in source.splitlines() if ": State 5 := ![" in line
        )
        self.assertEqual(state_line.count("⟨"), 5)

    def test_piecewise_rendering_is_sorted_and_deterministic(self) -> None:
        schedule_a = {
            "kind": "piecewise",
            "default": "1/4",
            "points": [
                {"exposure": 9, "value": "1/5"},
                {"exposure": 0, "value": "1/3"},
                {"exposure": 4, "value": "2/5"},
            ],
        }
        schedule_b = {
            "kind": "piecewise",
            "default": "1/4",
            "points": list(reversed(schedule_a["points"])),
        }
        source_a = self.builder.build_structural_positive(
            parse_model(model_document(schedule=schedule_a)),
            ("parameters_valid",),
        ).source
        source_b = self.builder.build_structural_positive(
            parse_model(model_document(schedule=schedule_b)),
            ("parameters_valid",),
        ).source
        self.assertEqual(source_a, source_b)
        self.assertLess(source_a.index("e = 0"), source_a.index("e = 4"))
        self.assertLess(source_a.index("e = 4"), source_a.index("e = 9"))

    def test_certificate_import_is_fixed_production_module(self) -> None:
        source = self.builder.build_structural_positive(
            parse_model(model_document()),
            ("parameters_valid",),
        ).source
        imports = [line for line in source.splitlines() if line.startswith("import ")]
        self.assertEqual(
            imports,
            ["import NarrativeDynamics.Core.FitnessABMPathNExposureConvergence"],
        )

    def test_source_like_named_schedule_never_reaches_rendering(self) -> None:
        model = parse_model(
            model_document(
                n=2,
                beliefs=["1", "0"],
                schedule={
                    "kind": "named",
                    "id": "harmonicSchedule; axiom hacked : False",
                },
            )
        )
        with self.assertRaises(ModelInputError):
            self.builder.build_structural_positive(model, ("parameters_valid",))

    def test_generated_source_has_no_proof_escape_hatches(self) -> None:
        source = self.builder.build_structural_positive(
            parse_model(model_document()),
            STRUCTURAL,
        ).source
        for forbidden in ("sorry", "admit", "axiom", "unsafe", "native_decide"):
            self.assertNotIn(forbidden, source)

    def test_internal_names_are_deterministic_hash_names(self) -> None:
        model = parse_model(model_document())
        first = self.builder.build_structural_positive(model, STRUCTURAL)
        second = self.builder.build_structural_positive(model, STRUCTURAL)
        self.assertEqual(first.source, second.source)
        self.assertRegex(first.source, r"AnalyzerParams_[0-9a-f]{12}")
        self.assertRegex(first.source, r"AnalyzerState_[0-9a-f]{12}")


class StructuralCertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = CertificateBuilder()

    def test_positive_structural_certificate_records_exact_provenance(self) -> None:
        certificate = self.builder.build_structural_positive(
            parse_model(model_document()),
            STRUCTURAL,
        )
        self.assertEqual(tuple(c.claim_id for c in certificate.claims), STRUCTURAL)
        self.assertTrue(all(c.expected_status is ClaimStatus.PROVED for c in certificate.claims))
        by_id = {claim.claim_id: claim for claim in certificate.claims}
        self.assertIn("ExposureParameters.Valid", by_id["parameters_valid"].theorem)
        self.assertIn("allBroadcast", by_id["initial_all_broadcast"].theorem)
        self.assertIn("exposure_iterate", by_id["exposure_law"].theorem)
        self.assertIn("exposure_iterate", by_id["effective_alpha_lookup"].theorem)
        self.assertIn("(k + 1) * FitnessABMPathN.degree", certificate.source)

    def test_negative_parameter_certificate_is_structural_only(self) -> None:
        model = parse_model(
            model_document(schedule={"kind": "constant", "value": "5/4"})
        )
        certificate = self.builder.build_structural_negative(
            model, ("parameters_valid",)
        )
        self.assertEqual(len(certificate.claims), 1)
        claim = certificate.claims[0]
        self.assertEqual(claim.claim_id, "parameters_valid")
        self.assertIs(claim.expected_status, ClaimStatus.DISPROVED)
        self.assertIn("¬", certificate.source)
        self.assertNotIn("path2_consensus", certificate.source)
        self.assertNotIn("pathn_consensus_exists", certificate.source)

    def test_negative_all_broadcast_certificate_uses_concrete_index_witness(self) -> None:
        model = parse_model(
            model_document(
                n=2,
                beliefs=["1", "0"],
                threshold="1/2",
            )
        )
        certificate = self.builder.build_structural_negative(
            model, ("initial_all_broadcast",)
        )
        claim = certificate.claims[0]
        self.assertEqual(claim.claim_id, "initial_all_broadcast")
        self.assertIs(claim.expected_status, ClaimStatus.DISPROVED)
        self.assertIn("Fin 2", certificate.source)
        self.assertIn("allBroadcast", claim.theorem)

    def test_negative_builder_rejects_consensus_claim(self) -> None:
        model = parse_model(model_document())
        with self.assertRaises(ValueError):
            self.builder.build_structural_negative(model, ("pathn_consensus_exists",))


if __name__ == "__main__":
    unittest.main()

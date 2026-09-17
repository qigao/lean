import re
import unittest

from narrative_analyzer.certificate import CertificateBuilder
from narrative_analyzer.model import ExactRat, ModelInputError, parse_model
from narrative_analyzer.named_schedules import fixed_fixture_route
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


class GlobalInteriorCertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = CertificateBuilder()

    def test_constant_candidate_is_exact_minimum_margin(self) -> None:
        from narrative_analyzer.certificate import candidate_global_interior

        model = parse_model(model_document(schedule={"kind": "constant", "value": "1/4"}))
        self.assertEqual(candidate_global_interior(model), ExactRat(1, 4))

    def test_piecewise_candidate_includes_default_and_all_branches(self) -> None:
        from narrative_analyzer.certificate import candidate_global_interior

        model = parse_model(
            model_document(
                schedule={
                    "kind": "piecewise",
                    "default": "1/4",
                    "points": [
                        {"exposure": 9, "value": "4/5"},
                        {"exposure": 0, "value": "1/3"},
                        {"exposure": 4, "value": "2/5"},
                    ],
                }
            )
        )
        self.assertEqual(candidate_global_interior(model), ExactRat(1, 5))

    def test_nonpositive_margin_returns_no_route_not_negative_evidence(self) -> None:
        from narrative_analyzer.certificate import candidate_global_interior

        for value in ("0", "1", "5/4"):
            model = parse_model(model_document(schedule={"kind": "constant", "value": value}))
            self.assertIsNone(candidate_global_interior(model))

    def test_constant_pathn_certificate_records_global_interior_and_consensus(self) -> None:
        from narrative_analyzer.certificate import candidate_global_interior

        model = parse_model(model_document(n=7, schedule={"kind": "constant", "value": "1/4"}))
        eps = candidate_global_interior(model)
        self.assertEqual(eps, ExactRat(1, 4))
        assert eps is not None
        certificate = self.builder.build_pathn_consensus(model, eps)
        self.assertEqual(
            tuple(claim.claim_id for claim in certificate.claims),
            ("reachable_interior", "pathn_consensus_exists"),
        )
        self.assertTrue(all(c.expected_status is ClaimStatus.PROVED for c in certificate.claims))
        by_id = {claim.claim_id: claim for claim in certificate.claims}
        self.assertEqual(by_id["reachable_interior"].exact_values["eps"], "1/4")
        self.assertIn("ReachableInterior", by_id["reachable_interior"].theorem)
        self.assertIn(
            "trajectory_consensus_exists_of_global_interior",
            by_id["pathn_consensus_exists"].theorem,
        )
        self.assertIn("∀ e", certificate.source)
        self.assertIn("1 -", certificate.source)
        self.assertNotIn("consensus_value_known", certificate.source)

    def test_piecewise_certificate_uses_finite_exact_branch_proof(self) -> None:
        from narrative_analyzer.certificate import candidate_global_interior

        model = parse_model(
            model_document(
                n=7,
                schedule={
                    "kind": "piecewise",
                    "default": "1/4",
                    "points": [
                        {"exposure": 4, "value": "2/5"},
                        {"exposure": 0, "value": "1/3"},
                    ],
                },
            )
        )
        eps = candidate_global_interior(model)
        assert eps is not None
        source = self.builder.build_pathn_consensus(model, eps).source
        self.assertIn("split_ifs <;> norm_num", source)
        self.assertIn("e = 0", source)
        self.assertIn("e = 4", source)


class NamedPath2CertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = CertificateBuilder()

    def _model(self, schedule_id: str, **overrides):
        document = model_document(
            n=overrides.pop("n", 2),
            beliefs=overrides.pop("beliefs", ["1", "0"]),
            exposures=overrides.pop("exposures", [0, 0]),
            threshold=overrides.pop("threshold", "0"),
            schedule={"kind": "named", "id": schedule_id},
        )
        self.assertFalse(overrides)
        return parse_model(document)

    def test_harmonic_fixture_certificate_proves_consensus_and_value(self) -> None:
        model = self._model("harmonicSchedule")
        route = fixed_fixture_route(model)
        self.assertIsNotNone(route)
        assert route is not None
        certificate = self.builder.build_named_path2(model, route)
        self.assertEqual(
            tuple(claim.claim_id for claim in certificate.claims),
            ("path2_consensus", "consensus_value_known"),
        )
        self.assertTrue(all(c.expected_status is ClaimStatus.PROVED for c in certificate.claims))
        by_id = {claim.claim_id: claim for claim in certificate.claims}
        self.assertIn("harmonic_consensus", by_id["path2_consensus"].theorem)
        self.assertEqual(by_id["consensus_value_known"].exact_values["value"], "1/2")
        self.assertIn("= harmonicSchedule", certificate.source)
        self.assertIn("AnalyzerState_", certificate.source)

    def test_slow_zero_fixture_certificate_is_theorem_backed_negative(self) -> None:
        model = self._model("slowZeroSchedule")
        route = fixed_fixture_route(model)
        assert route is not None
        certificate = self.builder.build_named_path2(model, route)
        self.assertEqual(len(certificate.claims), 1)
        claim = certificate.claims[0]
        self.assertEqual(claim.claim_id, "path2_consensus")
        self.assertIs(claim.expected_status, ClaimStatus.DISPROVED)
        self.assertIn("slowZero_not_consensus", claim.theorem)
        self.assertIn("slowZero_not_consensus", certificate.source)

    def test_near_one_fixture_certificate_is_theorem_backed_negative(self) -> None:
        model = self._model("nearOneSchedule")
        route = fixed_fixture_route(model)
        assert route is not None
        certificate = self.builder.build_named_path2(model, route)
        claim = certificate.claims[0]
        self.assertIs(claim.expected_status, ClaimStatus.DISPROVED)
        self.assertIn("nearOne_not_convergent", claim.theorem)
        self.assertIn("¬ ∃ c : Real", certificate.source)

    def test_builder_rejects_route_from_different_fixture(self) -> None:
        harmonic = self._model("harmonicSchedule")
        slow = self._model("slowZeroSchedule")
        route = fixed_fixture_route(harmonic)
        assert route is not None
        with self.assertRaises(ValueError):
            self.builder.build_named_path2(slow, route)

    def test_changed_fixture_has_no_fixed_theorem_route(self) -> None:
        changed = self._model("harmonicSchedule", exposures=[0, 1])
        self.assertIsNone(fixed_fixture_route(changed))


if __name__ == "__main__":
    unittest.main()

import unittest

from narrative_analyzer.result import (
    AnalysisResult,
    AnalyzerError,
    CertificateCompileError,
    CertificateGenerationError,
    CertificateTimeoutError,
    ClaimResult,
    ClaimStatus,
    ProvenanceMismatchError,
)


def claim(
    claim_id: str,
    status: ClaimStatus,
    *,
    theorem: str | None = None,
    note: str | None = None,
) -> ClaimResult:
    return ClaimResult(
        claim_id=claim_id,
        status=status,
        theorem=theorem,
        assumptions=(),
        exact_values={},
        note=note,
    )


class ClaimResultTests(unittest.TestCase):
    def test_proved_requires_theorem_provenance(self) -> None:
        with self.assertRaises(ValueError):
            claim("parameters_valid", ClaimStatus.PROVED)

    def test_disproved_requires_theorem_provenance(self) -> None:
        with self.assertRaises(ValueError):
            claim("path2_consensus", ClaimStatus.DISPROVED)

    def test_unknown_requires_explanatory_note(self) -> None:
        with self.assertRaises(ValueError):
            claim("consensus_value_known", ClaimStatus.UNKNOWN)

    def test_valid_claim_snapshots_exact_values(self) -> None:
        values = {"eps": "1/4"}
        result = ClaimResult(
            claim_id="reachable_interior",
            status=ClaimStatus.PROVED,
            theorem="Example.theorem",
            assumptions=("parameters valid",),
            exact_values=values,
            note=None,
        )
        values["eps"] = "1/3"
        self.assertEqual(result.exact_values["eps"], "1/4")
        with self.assertRaises(TypeError):
            result.exact_values["eps"] = "1/2"  # type: ignore[index]


class AnalysisResultTests(unittest.TestCase):
    def test_claim_ids_must_be_unique(self) -> None:
        duplicate = claim(
            "pathn_consensus_exists",
            ClaimStatus.UNKNOWN,
            note="no route",
        )
        with self.assertRaises(ValueError):
            AnalysisResult(model_summary={}, claims=(duplicate, duplicate))

    def test_consensus_status_prefers_path2_exact_claim(self) -> None:
        result = AnalysisResult(
            model_summary={"topology": "Path 2"},
            claims=(
                claim(
                    "pathn_consensus_exists",
                    ClaimStatus.PROVED,
                    theorem="Generic.consensus",
                ),
                claim(
                    "path2_consensus",
                    ClaimStatus.DISPROVED,
                    theorem="Path2.not_consensus",
                ),
            ),
        )
        self.assertEqual(result.consensus_status, ClaimStatus.DISPROVED)
        self.assertEqual(result.consensus_claim.claim_id, "path2_consensus")

    def test_consensus_status_falls_back_to_pathn_claim(self) -> None:
        result = AnalysisResult(
            model_summary={},
            claims=(
                claim(
                    "path2_consensus",
                    ClaimStatus.UNKNOWN,
                    note="iff assumptions unavailable",
                ),
                claim(
                    "pathn_consensus_exists",
                    ClaimStatus.PROVED,
                    theorem="Generic.consensus",
                ),
            ),
        )
        self.assertEqual(result.consensus_status, ClaimStatus.PROVED)
        self.assertEqual(result.consensus_claim.claim_id, "pathn_consensus_exists")

    def test_consensus_status_is_unknown_without_exact_consensus_result(self) -> None:
        result = AnalysisResult(
            model_summary={},
            claims=(
                claim(
                    "pathn_consensus_exists",
                    ClaimStatus.UNKNOWN,
                    note="no supported theorem route",
                ),
            ),
        )
        self.assertEqual(result.consensus_status, ClaimStatus.UNKNOWN)
        self.assertIsNone(result.consensus_claim)

    def test_model_summary_is_snapshotted(self) -> None:
        summary = {"topology": "Path 7"}
        result = AnalysisResult(model_summary=summary, claims=())
        summary["topology"] = "cycle"
        self.assertEqual(result.model_summary["topology"], "Path 7")
        with self.assertRaises(TypeError):
            result.model_summary["topology"] = "Path 2"  # type: ignore[index]


class FailureTaxonomyTests(unittest.TestCase):
    def test_analyzer_failures_are_exceptions_not_statuses(self) -> None:
        self.assertEqual(
            set(ClaimStatus),
            {ClaimStatus.PROVED, ClaimStatus.DISPROVED, ClaimStatus.UNKNOWN},
        )
        for error_type in (
            CertificateGenerationError,
            CertificateCompileError,
            CertificateTimeoutError,
            ProvenanceMismatchError,
        ):
            self.assertTrue(issubclass(error_type, AnalyzerError))
            self.assertTrue(issubclass(error_type, RuntimeError))


if __name__ == "__main__":
    unittest.main()

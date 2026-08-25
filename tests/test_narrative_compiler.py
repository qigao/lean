from __future__ import annotations

from dataclasses import replace
import json
import unittest

from narrative_dynamics.narrative.compiler import (
    CandidateBundle,
    CandidateClaim,
    CandidateDecision,
    CandidateEntity,
    CandidateObservation,
    CandidateProposition,
    CandidateReception,
    CandidateValue,
    ResolutionRecord,
    SourceSpan,
)

_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.compiler import (
        CompilationDiagnostic,
        CompilationResult,
        compile_candidates,
    )
except ImportError as error:
    _IMPORT_ERROR = error

from tests.narrative_compiler_test_support import (
    make_candidate_bundle,
    make_source_bundle,
    whole_span,
)
from tests.narrative_test_support import make_test_domain


class DeterministicCompilerTests(unittest.TestCase):
    def require_compiler(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"deterministic narrative compiler is missing: {_IMPORT_ERROR}")

    def test_incomplete_then_resolved(self) -> None:
        self.require_compiler()
        source = make_source_bundle()
        candidates = make_candidate_bundle(source, first_event_time=None)
        domain = make_test_domain()

        incomplete = compile_candidates(source, candidates, (), domain)
        self.assertEqual(incomplete.status, "incomplete")
        self.assertIsNone(incomplete.canonical_scenario)
        self.assertIsNone(incomplete.canonical_hash)

        resolved = compile_candidates(
            source,
            candidates,
            (
                ResolutionRecord(
                    "cev1",
                    "accepted",
                    {"logical_time": 1},
                    "human:test",
                    "first event",
                ),
            ),
            domain,
        )
        self.assertEqual(resolved.status, "canonical")
        self.assertIsNotNone(resolved.canonical_scenario)
        self.assertEqual(resolved.canonical_hash, resolved.canonical_scenario.content_hash)

    def test_provenance_can_change_without_semantic_identity_change(self) -> None:
        self.require_compiler()
        source_a = make_source_bundle("wording A")
        source_b = make_source_bundle("wording B")
        domain = make_test_domain()

        first = compile_candidates(
            source_a,
            make_candidate_bundle(source_a, confidence=0.8),
            (),
            domain,
        )
        second = compile_candidates(
            source_b,
            make_candidate_bundle(source_b, confidence=0.99),
            (),
            domain,
        )

        self.assertEqual(first.status, "canonical")
        self.assertEqual(second.status, "canonical")
        self.assertNotEqual(first.source_bundle_hash, second.source_bundle_hash)
        self.assertNotEqual(first.candidate_bundle_hash, second.candidate_bundle_hash)
        self.assertEqual(first.canonical_hash, second.canonical_hash)

    def test_unknown_resolution_field_is_rejected(self) -> None:
        self.require_compiler()
        source = make_source_bundle()
        candidates = make_candidate_bundle(source)
        result = compile_candidates(
            source,
            candidates,
            (
                ResolutionRecord(
                    "cev1",
                    "accepted",
                    {"unsupported_field": "x"},
                    "human:test",
                    "probe",
                ),
            ),
            make_test_domain(),
        )
        self.assertEqual(result.status, "rejected")
        self.assertIsNone(result.canonical_scenario)
        self.assertIsNone(result.canonical_hash)

    def test_conflicting_claims_can_compile_as_canonical(self) -> None:
        self.require_compiler()
        source = make_source_bundle()
        base = make_candidate_bundle(source, claim_value="recovered")
        span = whole_span(source)
        decision = next(
            item for item in base.candidates if isinstance(item, CandidateDecision)
        )
        candidates = tuple(
            item for item in base.candidates if not isinstance(item, CandidateDecision)
        ) + (
            CandidateEntity("ce-carol", "carol", "Agent", (span,), 0.9),
            CandidateObservation("co3", "o3", "ce-carol", "cev2", (span,), 0.9),
            CandidateClaim(
                "cc2",
                "c2",
                4,
                "ce-carol",
                CandidateProposition(
                    "ce-svc",
                    "service.health",
                    "equals",
                    CandidateValue("HealthState", "failed"),
                ),
                ("cev2",),
                (span,),
                0.9,
            ),
            CandidateReception("cr2", "r2", "cc2", "ce-bob", (span,), 0.9),
            replace(decision, logical_time=5),
        )
        result = compile_candidates(
            source,
            CandidateBundle(base.extractor, candidates),
            (),
            make_test_domain(),
        )
        self.assertEqual(result.status, "canonical")
        self.assertIsNotNone(result.canonical_scenario)
        self.assertEqual(len(result.canonical_scenario.claims), 2)

    def test_forged_source_span_rejects_before_unresolved_semantics(self) -> None:
        self.require_compiler()
        source = make_source_bundle()
        base = make_candidate_bundle(source, first_event_time=None)
        span = whole_span(source)
        forged = SourceSpan(span.document_id, span.start, span.end, "sha256:forged")
        first = base.candidates[0]
        self.assertIsInstance(first, CandidateEntity)
        candidates = (replace(first, source_spans=(forged,)),) + base.candidates[1:]

        result = compile_candidates(
            source,
            CandidateBundle(base.extractor, candidates),
            (),
            make_test_domain(),
        )
        self.assertEqual(result.status, "rejected")
        self.assertIsNone(result.canonical_scenario)
        self.assertIsNone(result.canonical_hash)

    def test_canonical_payload_excludes_compiler_provenance(self) -> None:
        self.require_compiler()
        source = make_source_bundle("source wording must not enter canonical payload")
        candidates = make_candidate_bundle(source, first_event_time=None, confidence=0.73)
        result = compile_candidates(
            source,
            candidates,
            (
                ResolutionRecord(
                    "cev1",
                    "accepted",
                    {"logical_time": 1},
                    "human:compiler-test",
                    "resolved source ambiguity",
                ),
            ),
            make_test_domain(),
        )
        self.assertEqual(result.status, "canonical")
        payload = json.dumps(result.canonical_scenario.to_dict(), sort_keys=True)
        self.assertNotIn("source wording must not enter canonical payload", payload)
        self.assertNotIn("confidence", payload)
        self.assertNotIn("test-extractor", payload)
        self.assertNotIn("human:compiler-test", payload)
        self.assertNotIn("diagnostics", payload)


if __name__ == "__main__":
    unittest.main()

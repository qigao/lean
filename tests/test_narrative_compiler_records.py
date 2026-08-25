from __future__ import annotations

import unittest

from narrative_dynamics.contracts import stable_content_hash

_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.compiler import (
        CandidateBundle,
        CandidateEntity,
        CandidateEntityRef,
        CandidateEvent,
        CandidateValue,
        ExtractorIdentity,
        ResolutionRecord,
        SourceBundle,
        SourceDocument,
        SourceSpan,
    )
except ImportError as error:
    _IMPORT_ERROR = error

if _IMPORT_ERROR is None:
    from tests.narrative_compiler_test_support import (
        make_candidate_bundle,
        make_source_bundle,
        whole_span,
    )


class CompilerRecordTests(unittest.TestCase):
    def require_compiler_records(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"narrative compiler provenance records are missing: {_IMPORT_ERROR}")

    def test_span_confidence_and_distinct_identity_layers(self) -> None:
        self.require_compiler_records()
        doc = SourceDocument("d1", "service failed", "text", "memory:d1")
        span = SourceSpan.from_document(doc, 0, 7)
        source = SourceBundle((doc,))
        source.validate_span(span)

        with self.assertRaisesRegex(ValueError, "confidence"):
            CandidateEntity("ce", "svc", "Service", (span,), float("nan"))

        candidates = make_candidate_bundle(source)
        resolution = ResolutionRecord(
            "ce-bob", "accepted", {}, "human:test", "entity confirmed"
        )
        self.assertNotEqual(source.content_hash, candidates.content_hash)
        self.assertTrue(
            stable_content_hash((resolution.to_dict(),)).startswith("sha256:")
        )

    def test_candidate_record_boundaries_are_fail_closed_and_frozen(self) -> None:
        self.require_compiler_records()
        source = make_source_bundle()
        span = whole_span(source)

        with self.assertRaisesRegex(ValueError, "source span"):
            CandidateEntity("ce-empty", "svc", "Service", (), 0.9)
        with self.assertRaisesRegex(ValueError, "decision"):
            ResolutionRecord("ce-x", "maybe", {}, "human:test", "probe")

        arguments = {
            "service": CandidateValue("ServiceRef", CandidateEntityRef("ce-svc")),
            "health": CandidateValue("HealthState", "failed"),
        }
        event = CandidateEvent(
            "cev-x", "e-x", "SetHealth", 1, None, arguments, (span,), 0.9
        )
        arguments["health"] = CandidateValue("HealthState", "recovered")
        self.assertEqual(event.arguments["health"].value, "failed")

        candidates = make_candidate_bundle(source)
        with self.assertRaisesRegex(ValueError, "candidate"):
            CandidateBundle(
                ExtractorIdentity("test-extractor", "1"),
                candidates.candidates + (candidates.candidates[0],),
            )


if __name__ == "__main__":
    unittest.main()

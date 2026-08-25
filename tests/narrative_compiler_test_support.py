from __future__ import annotations

from narrative_dynamics.narrative.compiler import (
    CandidateActionOption,
    CandidateBundle,
    CandidateClaim,
    CandidateDecision,
    CandidateEntity,
    CandidateEntityRef,
    CandidateEvent,
    CandidateObservation,
    CandidateProposition,
    CandidateReception,
    CandidateStateCellRef,
    CandidateValue,
    ExtractorIdentity,
    SourceBundle,
    SourceDocument,
    SourceSpan,
)


def make_source_bundle(
    text: str = "service failed then recovered; Alice told Bob recovered",
) -> SourceBundle:
    return SourceBundle((SourceDocument("d1", text, "text", "memory:d1"),))


def whole_span(source: SourceBundle) -> SourceSpan:
    doc = source.documents[0]
    return SourceSpan.from_document(doc, 0, len(doc.content))


def make_candidate_bundle(
    source: SourceBundle,
    first_event_time: int | None = 1,
    claim_value: str = "recovered",
    confidence: float = 0.9,
) -> CandidateBundle:
    span = whole_span(source)
    return CandidateBundle(
        ExtractorIdentity("test-extractor", "1"),
        (
            CandidateEntity("ce-bob", "bob", "Agent", (span,), confidence),
            CandidateEntity("ce-alice", "alice", "Agent", (span,), confidence),
            CandidateEntity("ce-svc", "svc", "Service", (span,), confidence),
            CandidateEvent(
                "cev1",
                "e1",
                "SetHealth",
                first_event_time,
                None,
                {
                    "service": CandidateValue(
                        "ServiceRef", CandidateEntityRef("ce-svc")
                    ),
                    "health": CandidateValue("HealthState", "failed"),
                },
                (span,),
                confidence,
            ),
            CandidateEvent(
                "cev2",
                "e2",
                "SetHealth",
                2,
                None,
                {
                    "service": CandidateValue(
                        "ServiceRef", CandidateEntityRef("ce-svc")
                    ),
                    "health": CandidateValue("HealthState", "recovered"),
                },
                (span,),
                confidence,
            ),
            CandidateObservation(
                "co1", "o1", "ce-bob", "cev1", (span,), confidence
            ),
            CandidateObservation(
                "co2", "o2", "ce-alice", "cev2", (span,), confidence
            ),
            CandidateClaim(
                "cc1",
                "c1",
                3,
                "ce-alice",
                CandidateProposition(
                    "ce-svc",
                    "service.health",
                    "equals",
                    CandidateValue("HealthState", claim_value),
                ),
                ("cev2",),
                (span,),
                confidence,
            ),
            CandidateReception(
                "cr1", "r1", "cc1", "ce-bob", (span,), confidence
            ),
            CandidateDecision(
                "cd1",
                "d1",
                4,
                "ce-bob",
                "service-response",
                (CandidateStateCellRef("ce-svc", "service.health"),),
                (
                    CandidateActionOption("restart", "service-action", {}),
                    CandidateActionOption("leave", "service-action", {}),
                ),
                (span,),
                confidence,
            ),
        ),
    )

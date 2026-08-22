from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from narrative_dynamics.attestation._common import validated_hash
from narrative_dynamics.contracts import TraceEvent, stable_content_hash


RESULT_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResultArtifact:
    """Content identity for every canonical event and the complete outcome."""

    events_hash: str
    outcome_hash: str
    model_run_hash: str
    schema_version: int = RESULT_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version <= 0
        ):
            raise ValueError("result artifact schema version must be positive")
        for attribute, label in (
            ("events_hash", "result events hash"),
            ("outcome_hash", "result outcome hash"),
            ("model_run_hash", "result model-run hash"),
        ):
            object.__setattr__(
                self,
                attribute,
                validated_hash(getattr(self, attribute), label=label),
            )

    @staticmethod
    def _event_payload(events: Sequence[TraceEvent]) -> tuple[dict[str, object], ...]:
        values = tuple(events)
        if any(not isinstance(event, TraceEvent) for event in values):
            raise TypeError("result artifact events must contain TraceEvent values")
        return tuple(
            {"tick": event.tick, "kind": event.kind, "data": event.data}
            for event in values
        )

    @classmethod
    def from_result(
        cls,
        events: Sequence[TraceEvent],
        outcome: Mapping[str, object],
    ) -> "ResultArtifact":
        if not isinstance(outcome, Mapping):
            raise TypeError("result artifact outcome must be a mapping")
        event_payload = cls._event_payload(events)
        return cls(
            events_hash=stable_content_hash(event_payload),
            outcome_hash=stable_content_hash(outcome),
            model_run_hash=stable_content_hash(
                {"events": event_payload, "outcome": outcome}
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_content_hash(
            {
                "schema_version": self.schema_version,
                "events_hash": self.events_hash,
                "outcome_hash": self.outcome_hash,
                "model_run_hash": self.model_run_hash,
            }
        )

    def manifest_identity(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "events_hash": self.events_hash,
            "outcome_hash": self.outcome_hash,
            "model_run_hash": self.model_run_hash,
            "content_hash": self.content_hash,
        }

    def matches(
        self,
        events: Sequence[TraceEvent],
        outcome: Mapping[str, object],
    ) -> bool:
        return self == self.from_result(events, outcome)


__all__ = [
    "RESULT_ARTIFACT_SCHEMA_VERSION",
    "ResultArtifact",
]

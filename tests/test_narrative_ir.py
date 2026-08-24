from __future__ import annotations

import json
import unittest

_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from narrative_dynamics.narrative.ir import (
        ActionOption,
        Claim,
        Decision,
        Entity,
        EntityRef,
        GenericNarrative,
        NarrativeEvent,
        Observation,
        Proposition,
        Reception,
        StateCellRef,
        TypedValue,
    )
except ModuleNotFoundError as error:
    _IMPORT_ERROR = error


class GenericNarrativeIRTests(unittest.TestCase):
    def test_records_are_immutable_and_hash_stable(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"generic narrative IR is missing: {_IMPORT_ERROR}")

        service = EntityRef("svc", "Service")
        story = GenericNarrative(
            domain_id="service-incident",
            domain_version="1",
            domain_spec_hash="sha256:" + "1" * 64,
            entities=(Entity("svc", "Service"), Entity("bob", "Agent")),
            events=(
                NarrativeEvent(
                    "e1",
                    1,
                    "SetHealth",
                    None,
                    {
                        "service": TypedValue("ServiceRef", service),
                        "health": TypedValue("HealthState", "failed"),
                    },
                ),
            ),
            observations=(Observation("o1", "bob", "e1"),),
            claims=(
                Claim(
                    "c1",
                    2,
                    "bob",
                    Proposition(
                        service,
                        "service.health",
                        "equals",
                        TypedValue("HealthState", "failed"),
                    ),
                    ("e1",),
                ),
            ),
            receptions=(Reception("r1", "c1", "bob"),),
            decisions=(
                Decision(
                    "d1",
                    3,
                    "bob",
                    "service-response",
                    (StateCellRef(service, "service.health"),),
                    (ActionOption("restart", "service-action", {}),),
                ),
            ),
        )
        payload = story.to_dict()
        self.assertEqual(json.loads(json.dumps(payload, sort_keys=True)), payload)
        self.assertEqual(story.content_hash, story.content_hash)
        with self.assertRaises(TypeError):
            story.events[0].arguments["health"] = TypedValue(
                "HealthState", "healthy"
            )

    def test_entity_reference_is_not_plain_text(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"generic narrative IR is missing: {_IMPORT_ERROR}")

        entity = TypedValue("ServiceRef", EntityRef("svc", "Service"))
        text = TypedValue("ServiceRef", "svc")
        self.assertNotEqual(entity.to_dict(), text.to_dict())


if __name__ == "__main__":
    unittest.main()

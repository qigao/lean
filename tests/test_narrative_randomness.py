from __future__ import annotations

from dataclasses import replace
import math
import unittest

_RANDOMNESS_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.randomness import (
        RANDOM_DERIVATION_VERSION,
        derive_stream_hash,
        draw_u64_for_stream,
        sample_categorical,
        validate_root_seed,
    )
except ImportError as error:
    _RANDOMNESS_IMPORT_ERROR = error


class NarrativeRandomnessTests(unittest.TestCase):
    def require_randomness(self) -> None:
        if _RANDOMNESS_IMPORT_ERROR is not None:
            self.fail(
                "narrative randomness API is missing: "
                f"{_RANDOMNESS_IMPORT_ERROR}"
            )

    def _common(self):
        return {
            "root_seed": 42,
            "namespace": "world.transition",
            "step_index": 1,
            "source_hash": "sha256:" + "1" * 64,
            "component_hash": "sha256:" + "2" * 64,
            "component_key": ("a1", "d1", "act"),
        }

    def test_seed_validation_and_namespaces_are_closed(self):
        self.require_randomness()
        self.assertEqual(validate_root_seed(0), 0)
        self.assertEqual(validate_root_seed(-7), -7)
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    validate_root_seed(bad)
        with self.assertRaises((TypeError, ValueError)):
            derive_stream_hash(
                **{**self._common(), "namespace": "decision.choice"}
            )

    def test_stream_and_draw_are_exact_and_replay(self):
        self.require_randomness()
        first = derive_stream_hash(**self._common())
        second = derive_stream_hash(**self._common())
        self.assertEqual(first, second)
        self.assertEqual(
            draw_u64_for_stream(first),
            draw_u64_for_stream(second),
        )
        self.assertEqual(
            RANDOM_DERIVATION_VERSION,
            "narrative-hash-categorical-v1",
        )

    def test_component_key_change_changes_stream_identity(self):
        self.require_randomness()
        first = derive_stream_hash(**self._common())
        second = derive_stream_hash(
            **{
                **self._common(),
                "component_key": ("a2", "d2", "act"),
            }
        )
        self.assertNotEqual(first, second)

    def test_outcome_input_order_does_not_change_selection(self):
        self.require_randomness()
        common = {
            **self._common(),
            "namespace": "observation.projection",
            "component_key": ("a1", "vision"),
            "distribution_hash": "sha256:" + "5" * 64,
        }
        outcomes = (
            ("drop", 0.5, "sha256:" + "6" * 64),
            ("emit", 0.5, "sha256:" + "7" * 64),
        )
        first = sample_categorical(outcomes=outcomes, **common)
        second = sample_categorical(
            outcomes=tuple(reversed(outcomes)),
            **common,
        )
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_distribution_validation_rejects_invalid_probabilities(self):
        self.require_randomness()
        common = {
            **self._common(),
            "distribution_hash": "sha256:" + "3" * 64,
        }
        bad_sets = (
            (("only", 1.0, "sha256:" + "4" * 64),),
            (
                ("a", 0.0, "sha256:" + "4" * 64),
                ("b", 1.0, "sha256:" + "5" * 64),
            ),
            (
                ("a", 0.4, "sha256:" + "4" * 64),
                ("b", 0.4, "sha256:" + "5" * 64),
            ),
            (
                ("a", 0.5, "sha256:" + "4" * 64),
                ("a", 0.5, "sha256:" + "5" * 64),
            ),
            (
                ("a", True, "sha256:" + "4" * 64),
                ("b", 0.5, "sha256:" + "5" * 64),
            ),
            (
                ("a", math.nan, "sha256:" + "4" * 64),
                ("b", 1.0, "sha256:" + "5" * 64),
            ),
            (
                ("a", math.inf, "sha256:" + "4" * 64),
                ("b", 1.0, "sha256:" + "5" * 64),
            ),
        )
        for outcomes in bad_sets:
            with self.subTest(outcomes=outcomes):
                with self.assertRaises((TypeError, ValueError)):
                    sample_categorical(outcomes=outcomes, **common)

    def test_random_sample_record_forgery_rejects(self):
        self.require_randomness()
        record = sample_categorical(
            **self._common(),
            distribution_hash="sha256:" + "3" * 64,
            outcomes=(
                ("a", 0.5, "sha256:" + "4" * 64),
                ("b", 0.5, "sha256:" + "5" * 64),
            ),
        )
        for field, value in (
            ("stream_hash", "sha256:" + "f" * 64),
            ("draw_u64", record.draw_u64 ^ 1),
            ("namespace", "decision.choice"),
        ):
            with self.subTest(field=field):
                with self.assertRaises((TypeError, ValueError)):
                    replace(record, **{field: value})


if __name__ == "__main__":
    unittest.main()

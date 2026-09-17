import unittest

from narrative_analyzer.model import ModelInputError, parse_model
from narrative_analyzer.named_schedules import (
    FIXED_PATH2_CLAIM,
    LEAN_NAMESPACE,
    fixed_fixture_route,
    resolve_named_schedule,
)
from narrative_analyzer.result import ClaimStatus


def named_model(
    schedule_id: str,
    *,
    n: int = 2,
    beliefs: list[str] | None = None,
    exposures: list[int] | None = None,
    threshold: str = "0",
):
    if beliefs is None:
        beliefs = ["1", "0"] if n == 2 else ["1"] + ["0"] * (n - 1)
    if exposures is None:
        exposures = [0] * n
    return parse_model(
        {
            "topology": {"kind": "path", "n": n},
            "initial": {"beliefs": beliefs, "exposures": exposures},
            "dynamics": {"threshold": threshold},
            "schedule": {"kind": "named", "id": schedule_id},
        }
    )


class NamedScheduleRegistryTests(unittest.TestCase):
    def test_all_approved_names_resolve_to_predeclared_lean_definitions(self) -> None:
        expected = {
            "slowZeroSchedule": "slowZeroSchedule",
            "nearOneSchedule": "nearOneSchedule",
            "harmonicSchedule": "harmonicSchedule",
        }
        for schedule_id, lean_tail in expected.items():
            entry = resolve_named_schedule(schedule_id)
            self.assertEqual(entry.schedule_id, schedule_id)
            self.assertEqual(
                entry.lean_definition,
                f"{LEAN_NAMESPACE}.{lean_tail}",
            )

    def test_unknown_name_is_rejected_before_certificate_generation(self) -> None:
        with self.assertRaises(ModelInputError):
            resolve_named_schedule("notRegistered")

    def test_source_like_name_does_not_resolve(self) -> None:
        payload = "slowZeroSchedule; axiom hacked : False"
        model = named_model(payload)
        with self.assertRaises(ModelInputError):
            resolve_named_schedule(model.schedule.schedule_id)

    def test_fixed_fixture_routes_have_only_predeclared_theorems(self) -> None:
        expected = {
            "slowZeroSchedule": ("slowZero_not_consensus", ClaimStatus.DISPROVED),
            "nearOneSchedule": ("nearOne_not_convergent", ClaimStatus.DISPROVED),
            "harmonicSchedule": ("harmonic_consensus", ClaimStatus.PROVED),
        }
        for schedule_id, (theorem_tail, status) in expected.items():
            route = fixed_fixture_route(named_model(schedule_id))
            self.assertIsNotNone(route)
            assert route is not None
            self.assertEqual(route.claim_id, FIXED_PATH2_CLAIM)
            self.assertEqual(route.expected_status, status)
            self.assertEqual(route.theorem, f"{LEAN_NAMESPACE}.{theorem_tail}")

    def test_fixed_fixture_requires_exact_n(self) -> None:
        self.assertIsNone(fixed_fixture_route(named_model("harmonicSchedule", n=3)))

    def test_fixed_fixture_requires_exact_beliefs(self) -> None:
        self.assertIsNone(
            fixed_fixture_route(
                named_model("harmonicSchedule", beliefs=["3/4", "1/4"])
            )
        )

    def test_fixed_fixture_requires_exact_exposures(self) -> None:
        self.assertIsNone(
            fixed_fixture_route(
                named_model("slowZeroSchedule", exposures=[1, 1])
            )
        )

    def test_fixed_fixture_requires_exact_threshold(self) -> None:
        self.assertIsNone(
            fixed_fixture_route(
                named_model("nearOneSchedule", threshold="1/4")
            )
        )

    def test_fixed_fixture_requires_matching_named_schedule(self) -> None:
        model = named_model("harmonicSchedule")
        route = fixed_fixture_route(model)
        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.schedule_id, "harmonicSchedule")
        self.assertNotEqual(route.theorem, f"{LEAN_NAMESPACE}.slowZero_not_consensus")


if __name__ == "__main__":
    unittest.main()

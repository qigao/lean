import unittest

from motivation import AgentMotivation, Goal


class MotivationTests(unittest.TestCase):
    def test_spartacus_prefers_escape_when_agency_pressure_is_high(self):
        model = AgentMotivation(
            pressures={"agency": 0.9, "status": 0.1, "security": 0.3},
            goals={
                "escape": Goal({"agency": 1.0}, cost=0.25, risk=0.20),
                "submit": Goal({"security": 0.25}, cost=0.05, risk=0.02),
            },
        )
        self.assertGreater(model.score("escape"), model.score("submit"))

    def test_general_prefers_victory_without_glory_goal_being_primitive(self):
        model = AgentMotivation(
            pressures={"status": 0.85, "security": 0.15},
            goals={
                "military_victory": Goal({"status": 0.9}, cost=0.20, risk=0.25),
                "avoid_campaign": Goal({"security": 0.35}, cost=0.05, risk=0.03),
            },
        )
        self.assertGreater(model.score("military_victory"), model.score("avoid_campaign"))

    def test_batiatus_prefers_wealth_as_multi_drive_instrument(self):
        model = AgentMotivation(
            pressures={"status": 0.8, "security": 0.5, "agency": 0.6},
            goals={
                "wealth": Goal({"status": 0.7, "security": 0.6, "agency": 0.5}, cost=0.25, risk=0.10),
                "idle": Goal({}, cost=0.0, risk=0.0),
            },
        )
        self.assertGreater(model.score("wealth"), model.score("idle"))


if __name__ == "__main__":
    unittest.main()

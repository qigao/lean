from copy import deepcopy
from dataclasses import fields, replace
from fractions import Fraction
import unittest
from unittest.mock import patch

import narrative_dynamics.abm as abm
from narrative_dynamics.abm import bb_runtime
from narrative_dynamics.abm.bb_runtime import replay_bb_population
from narrative_dynamics.abm.bb_runtime_contracts import (
    BBRuntimeAgent,
    BBRuntimeBirth,
    BBRuntimeError,
    BBRuntimeNewborn,
    BBRuntimeRawSeed,
    BBRuntimeReplay,
    BBRuntimeSeed,
    BBRuntimeTick,
)
from narrative_dynamics.abm.contracts import NetworkAgentState
from narrative_dynamics.abm.simulation import PopulationTrajectory
from tests.bb_runtime_fixtures import birth, seed


class BBRuntimeReplayTests(unittest.TestCase):
    def assert_error(self, result, *, stage, code, field,
                     agent_index=None, tick_index=None, birth_index=None,
                     expected=None, actual=None, bb_cause=None):
        self.assertIsInstance(result, BBRuntimeError)
        self.assertEqual(
            (result.stage, result.code, result.field),
            (stage, code, field),
        )
        self.assertEqual(
            (result.agent_index, result.tick_index, result.birth_index),
            (agent_index, tick_index, birth_index),
        )
        self.assertEqual((result.expected, result.actual), (expected, actual))
        self.assertEqual(result.bb_cause, bb_cause)

    def test_successive_births_and_idle_keep_two_clocks(self):
        out = replay_bb_population(seed(), 1,
                                   (birth("2", (1,)), birth("3", (2,)), BBRuntimeTick()))
        self.assertIsInstance(out, BBRuntimeReplay)
        frames = [t.next_frame for t in out.transitions]
        self.assertEqual([f.tick_count for f in frames], [1, 2, 3])
        self.assertEqual([f.birth_count for f in frames], [1, 2, 2])
        self.assertEqual([f.population.round_index for f in frames], [1, 1, 2])
        self.assertEqual([a.belief for a in frames[1].population.agents], [1, 1, 1, 0])
        self.assertEqual([a.exposure_count for a in frames[1].population.agents],
                         [1, 2, 1, 0])
        self.assertEqual([a.belief for a in out.final.population.agents], [1, 1, 1, 1])
        self.assertEqual(out.final.trace_mass, Fraction(1, 8))
        self.assertEqual(out.final.agent_ids, ("0", "1", "2", "3"))
        self.assertEqual(out.final.topology.edges, ((0, 1), (1, 2), (2, 3)))

    def test_empty_still_validates_and_does_not_propagate(self):
        with patch("narrative_dynamics.abm.bb_runtime.simulate_round") as operator:
            out = replay_bb_population(seed(), 1, ())
            invalid = replay_bb_population(seed(), 0, ())
        operator.assert_not_called()
        self.assertEqual(out.initial, out.final)
        self.assertEqual(out.transitions, ())
        self.assertEqual(out.final.trace_mass, Fraction(1))
        self.assertEqual((invalid.stage, invalid.code), ("initial_m", "initialM"))

    def test_first_error_and_two_indices_are_atomic(self):
        original = seed()
        bad = birth("2", (2,))
        requests = (BBRuntimeTick(), BBRuntimeTick(), bad)
        expected = replay_bb_population(original, 1, requests)
        actual = replay_bb_population(original, 1, requests + (object(),))
        self.assertIsInstance(actual, BBRuntimeError)
        self.assertEqual(actual, expected)
        self.assertEqual((actual.tick_index, actual.birth_index), (2, 0))
        self.assertEqual(actual.bb_cause, "targetOutOfRange")
        self.assertEqual(original, seed())
        self.assertEqual(requests[-1], bad)
        self.assertFalse(hasattr(actual, "transitions"))

    def test_numeric_registry_crosses_nine_to_ten_without_state_reassociation(self):
        raw = BBRuntimeSeed(
            BBRuntimeRawSeed(
                10,
                (1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                 (5, 6), (6, 7), (7, 8), (8, 9)),
            ),
            tuple(
                BBRuntimeAgent(str(i), "peer", 0, 1, i / 16, i)
                for i in range(10)
            ),
        )
        out = replay_bb_population(raw, 1, (birth("10", (9,), receptivity=0),))
        self.assertIsInstance(out, BBRuntimeReplay)
        step = out.transitions[0]
        by_id = {state.agent_id: state for state in step.post_growth_population.agents}
        self.assertEqual(out.final.agent_ids,
                         ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"))
        self.assertEqual(tuple(agent.agent_id for agent in out.final.model.agents),
                         ("0", "1", "10", "2", "3", "4", "5", "6", "7", "8", "9"))
        self.assertEqual(
            tuple(by_id[str(i)] for i in range(10)),
            (
                NetworkAgentState("0", 0.0, 0, False),
                NetworkAgentState("1", 0.0625, 1, False),
                NetworkAgentState("2", 0.125, 2, False),
                NetworkAgentState("3", 0.1875, 3, False),
                NetworkAgentState("4", 0.25, 4, False),
                NetworkAgentState("5", 0.3125, 5, False),
                NetworkAgentState("6", 0.375, 6, False),
                NetworkAgentState("7", 0.4375, 7, False),
                NetworkAgentState("8", 0.5, 8, False),
                NetworkAgentState("9", 0.5625, 9, False),
            ),
        )
        self.assertEqual(out.final.trace_mass, Fraction(1, 18))

    def test_repeated_and_equivalent_exact_inputs_have_identical_content(self):
        integer_input = seed(fitness=(1, 1))
        rational_input = seed(fitness=(Fraction(1), Fraction(1)))
        integer_ticks = (birth("2", (1,), fitness=1), BBRuntimeTick())
        rational_ticks = (birth("2", (1,), fitness=Fraction(1)), BBRuntimeTick())
        first = replay_bb_population(integer_input, 1, integer_ticks)
        second = replay_bb_population(integer_input, 1, integer_ticks)
        normalized = replay_bb_population(rational_input, 1, rational_ticks)
        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(first, normalized)
        self.assertEqual(first.content_hash, normalized.content_hash)

    def test_ordered_targets_are_part_of_history_and_exact_mass(self):
        left = replay_bb_population(seed(), 2, (birth("2", (0, 1)),))
        right = replay_bb_population(seed(), 2, (birth("2", (1, 0)),))
        self.assertEqual(left.final.model, right.final.model)
        self.assertEqual(left.final.population, right.final.population)
        self.assertEqual((left.final.trace_mass, right.final.trace_mass),
                         (Fraction(1, 2), Fraction(1, 2)))
        self.assertNotEqual(left.transitions[0].tick, right.transitions[0].tick)
        self.assertNotEqual(left.final.content_hash, right.final.content_hash)
        self.assertNotEqual(left.content_hash, right.content_hash)

        weighted_left = replay_bb_population(
            seed(fitness=(1, 3)), 2, (birth("2", (0, 1)),),
        )
        weighted_right = replay_bb_population(
            seed(fitness=(1, 3)), 2, (birth("2", (1, 0)),),
        )
        self.assertEqual((weighted_left.final.trace_mass, weighted_right.final.trace_mass),
                         (Fraction(1, 4), Fraction(3, 4)))

    def test_idle_only_keeps_epoch_data_and_advances_local_and_global_rounds(self):
        out = replay_bb_population(seed(), 1, (BBRuntimeTick(), BBRuntimeTick()))
        self.assertEqual(
            tuple((step.next_frame.tick_count, step.next_frame.population.round_index)
                  for step in out.transitions),
            ((1, 1), (2, 2)),
        )
        self.assertEqual(out.final.birth_count, 0)
        self.assertEqual(out.final.trace_mass, Fraction(1))
        self.assertEqual(out.final.topology, out.initial.topology)
        self.assertEqual(out.final.model, out.initial.model)
        self.assertEqual(out.final.topology.fitness, (Fraction(1), Fraction(1)))

        invalid = replay_bb_population(seed(), 0, [object()])
        self.assert_error(invalid, stage="initial_m", code="initialM", field="m")

    def test_birth_after_idles_opens_local_round_one_without_rewriting_old_exposure(self):
        out = replay_bb_population(
            seed(exposures=(7, 3)), 1,
            (BBRuntimeTick(), BBRuntimeTick(), birth("2", (1,), receptivity=0)),
        )
        step = out.transitions[2]
        before = {state.agent_id: state.exposure_count
                  for state in out.transitions[1].next_frame.population.agents}
        post = {state.agent_id: state.exposure_count
                for state in step.post_growth_population.agents}
        self.assertEqual((step.global_round, step.next_frame.population.round_index,
                          step.next_frame.epoch), (3, 1, 1))
        self.assertEqual((post["0"], post["1"]), (before["0"], before["1"]))

    def test_late_failures_report_separate_tick_and_birth_indices(self):
        late = replay_bb_population(
            seed(), 1,
            (BBRuntimeTick(), birth("2", (1,)), BBRuntimeTick(), birth("3", (3,))),
        )
        first_birth = replay_bb_population(
            seed(), 1, (BBRuntimeTick(), BBRuntimeTick(), birth("2", (2,))),
        )
        self.assert_error(
            late, stage="tick_network", code="targetOutOfRange", field="targets",
            tick_index=3, birth_index=1, bb_cause="targetOutOfRange",
        )
        self.assert_error(
            first_birth, stage="tick_network", code="targetOutOfRange", field="targets",
            tick_index=2, birth_index=0, bb_cause="targetOutOfRange",
        )

    def test_iteration_stops_at_first_agent_or_network_failure(self):
        invalid_agent = birth("2", (1,), threshold=2)
        bad_network = birth("3", (1,), fitness=0)
        arbitrary = object()
        for later in (bad_network, arbitrary):
            with self.subTest(later=type(later).__name__):
                result = replay_bb_population(seed(), 1, (invalid_agent, later))
                self.assert_error(
                    result, stage="tick_agent", code="invalidAgentValue",
                    field="broadcast_threshold", tick_index=0, birth_index=0,
                )
        reversed_result = replay_bb_population(seed(), 1, (bad_network, invalid_agent))
        self.assert_error(
            reversed_result, stage="tick_network", code="nonpositiveFitness",
            field="fitness", tick_index=0, birth_index=0,
            bb_cause="nonpositiveFitness",
        )

    def test_seed_validation_precedence_is_left_to_right(self):
        invalid_network = replace(seed(), network=object(), agents=(object(),))
        self.assert_error(
            replay_bb_population(invalid_network, 0, object()),
            stage="seed_network", code="invalidType", field="network",
            bb_cause=None,
        )

        wrong_count = replace(seed(), agents=(object(),))
        self.assert_error(
            replay_bb_population(wrong_count, 0, object()),
            stage="initial_m", code="initialM", field="m",
        )
        self.assert_error(
            replay_bb_population(wrong_count, 1, object()),
            stage="seed_agents", code="seedAgentCount", field="agents",
            expected=2, actual=1,
        )

        malformed = replace(seed(), agents=(object(), object()))
        self.assert_error(
            replay_bb_population(malformed, 1, ()),
            stage="seed_agent", code="invalidType", field="agent", agent_index=0,
        )

        bad_scalars = replace(
            seed(),
            agents=(
                BBRuntimeAgent("", "", 2, 2, 2, -1, 2),
                BBRuntimeAgent("1", "peer", True, 0.5, 0),
            ),
        )
        self.assert_error(
            replay_bb_population(bad_scalars, 1, ()),
            stage="seed_agent", code="invalidAgentValue", field="receptivity",
            agent_index=0,
        )
        exposure_first = replace(
            seed(),
            agents=(BBRuntimeAgent("", "", 1, 0.5, 0, -1, 2), seed().agents[1]),
        )
        self.assert_error(
            replay_bb_population(exposure_first, 1, ()),
            stage="seed_agent", code="invalidExposure", field="exposure_count",
            agent_index=0,
        )

    def test_public_boundary_rejects_wrong_shapes_and_identity_fields(self):
        cases = (
            (object(), 1, (), "seed_network", "invalidType", "seed"),
            (replace(seed(), network=replace(seed().network, fitness=[1, 1])),
             1, (), "seed_network", "invalidType", "fitness"),
            (replace(seed(), network=replace(seed().network, edges=[(0, 1)])),
             1, (), "seed_network", "invalidType", "edges"),
            (replace(seed(), agents=list(seed().agents)), 1, (),
             "seed_agents", "invalidType", "agents"),
            (replace(seed(), agents=(object(), seed().agents[1])), 1, (),
             "seed_agent", "invalidType", "agent"),
            (replace(seed(), model_id=7, version=""), 1, (),
             "seed_identity", "invalidType", "model_id"),
            (replace(seed(), model_id="", version=7), 1, (),
             "seed_identity", "invalidText", "model_id"),
            (replace(seed(), version=7), 1, (),
             "seed_identity", "invalidType", "version"),
            (seed(), 1, [], "ticks", "invalidType", "ticks"),
            (seed(), 1, (BBRuntimeTick(True),), "tick", "invalidType", "kind"),
            (seed(), 1, (BBRuntimeTick("later"),), "tick", "invalidTickKind", "kind"),
            (seed(), 1, (BBRuntimeTick("idle", object()),),
             "tick", "unexpectedBirthData", "birth"),
            (seed(), 1, (BBRuntimeTick("birth"),),
             "tick", "missingBirthData", "birth"),
            (seed(), 1, (BBRuntimeTick("birth", object()),),
             "tick", "invalidType", "birth"),
            (seed(), 1, (replace(birth("2", (1,)),
                                 birth=replace(birth("2", (1,)).birth, agent=object())),),
             "tick_agent", "invalidType", "agent"),
            (seed(), 1, (birth("2", [1]),),
             "tick_network", "invalidType", "targets"),
            (replace(seed(), agents=(replace(seed().agents[0], agent_id=""), seed().agents[1])),
             1, (), "seed_agent", "invalidText", "agent_id"),
            (replace(seed(), agents=(seed().agents[0], replace(seed().agents[1], agent_id="0"))),
             1, (), "seed_agent", "duplicateAgentId", "agent_id"),
            (replace(seed(), agents=(replace(seed().agents[0], role=""), seed().agents[1])),
             1, (), "seed_agent", "invalidText", "role"),
            (replace(seed(), agents=(replace(seed().agents[0], adoption_threshold=2),
                                    seed().agents[1])),
             1, (), "seed_agent", "invalidAdoptionThreshold", "adoption_threshold"),
        )
        for raw_seed, m, ticks, stage, code, field in cases:
            with self.subTest(stage=stage, code=code, field=field):
                result = replay_bb_population(raw_seed, m, ticks)
                self.assertEqual((result.stage, result.code, result.field),
                                 (stage, code, field))
        self.assertNotIn("exposure_count", {item.name for item in fields(BBRuntimeNewborn)})
        with self.assertRaises(TypeError):
            BBRuntimeNewborn("2", "peer", 1, 0.5, 0, exposure_count=1)

    def test_execution_module_is_the_only_public_import_location(self):
        self.assertEqual(bb_runtime.__all__, ["replay_bb_population"])
        self.assertNotIn("replay_bb_population", abm.__all__)
        self.assertFalse(hasattr(abm, "replay_bb_population"))

    def test_requests_are_fresh_and_failure_does_not_reserve_an_id(self):
        original_seed = seed()
        valid = birth("2", (1,))
        malformed_targets = [1]
        malformed = BBRuntimeTick(
            "birth",
            BBRuntimeBirth(1, malformed_targets,
                           BBRuntimeNewborn("3", "peer", 1, 0.5, 0)),
        )
        requests = (valid, malformed)
        copied_seed = deepcopy(original_seed)
        copied_requests = deepcopy(requests)
        result = replay_bb_population(original_seed, 1, requests)
        self.assert_error(
            result, stage="tick_network", code="invalidType", field="targets",
            tick_index=1, birth_index=1,
        )
        self.assertEqual(original_seed, copied_seed)
        self.assertEqual(requests, copied_requests)
        self.assertEqual(malformed_targets, [1])
        self.assertFalse(hasattr(result, "transitions"))

        corrected = replay_bb_population(
            original_seed, 1, (valid, birth("3", (2,))),
        )
        self.assertIsInstance(corrected, BBRuntimeReplay)
        self.assertEqual(corrected.final.agent_ids, ("0", "1", "2", "3"))

    def test_v1_round_and_trajectory_fixed_roster_contracts_still_reject(self):
        out = replay_bb_population(seed(), 1, (birth("2", (1,)),))
        step = out.transitions[0]
        with self.assertRaises(ValueError):
            replace(step.round_result, prior_state=out.initial.population)
        with self.assertRaises(ValueError):
            PopulationTrajectory(
                out.initial.model.model_id,
                out.initial.model.content_hash,
                out.initial.population,
                (),
                out.initial.population,
            )


if __name__ == "__main__":
    unittest.main()

from dataclasses import replace
from fractions import Fraction
import unittest
from unittest.mock import patch

from narrative_dynamics.abm.bb_runtime import _advance_tick, _parse_agent
from narrative_dynamics.abm.bb_runtime_contracts import (
    BBRuntimeAgent,
    BBRuntimeBirth,
    BBRuntimeError,
    BBRuntimeNewborn,
    BBRuntimeTick,
)
from narrative_dynamics.abm.contracts import (
    NetworkABMModel,
    NetworkAgentSpec,
    NetworkAgentState,
    PopulationState,
    SocialEdge,
    SocialNetwork,
)
from narrative_dynamics.abm.simulation import simulate_round
from tests.bb_runtime_fixtures import birth, genesis_frame


class BBRuntimeRoundTests(unittest.TestCase):
    def assert_error(self, result, *, stage, code, field,
                     tick_index=None, birth_index=None, agent_index=None):
        self.assertIsInstance(result, BBRuntimeError)
        self.assertEqual((result.stage, result.code, result.field),
                         (stage, code, field))
        self.assertEqual((result.tick_index, result.birth_index, result.agent_index),
                         (tick_index, birth_index, agent_index))

    def test_birth_retains_old_state_before_exactly_one_v1_round(self):
        initial = genesis_frame(exposures=(7, 3))
        with patch("narrative_dynamics.abm.bb_runtime.simulate_round",
                   wraps=simulate_round) as operator:
            step = _advance_tick(initial, birth("2", (1,)))
        operator.assert_called_once_with(step.next_frame.model,
                                         step.post_growth_population)
        post = {a.agent_id: a for a in step.post_growth_population.agents}
        old = {a.agent_id: a for a in initial.population.agents}
        self.assertEqual({k: post[k] for k in old}, old)
        self.assertEqual((post["2"].belief, post["2"].exposure_count), (0.0, 0))
        self.assertEqual(step.post_growth_population.round_index, 0)
        self.assertIsNone(step.post_growth_population.parent_state_hash)
        self.assertEqual(step.round_result.round_index, 1)
        self.assertEqual(step.next_frame.trace_mass, Fraction(1, 2))
        self.assertEqual(step.next_frame.population.parent_state_hash,
                         step.post_growth_population.content_hash)
        self.assertEqual(step.next_frame.parent_frame_hash, initial.content_hash)

    def test_new_receiver_relays_only_after_next_global_tick(self):
        first = _advance_tick(genesis_frame(), birth("2", (1,)))
        self.assertEqual([a.belief for a in first.next_frame.population.agents],
                         [1.0, 1.0, 0.0])
        second = _advance_tick(first.next_frame, BBRuntimeTick())
        self.assertEqual(second.post_growth_population, first.next_frame.population)
        self.assertEqual([a.belief for a in second.next_frame.population.agents],
                         [1.0, 1.0, 1.0])
        self.assertEqual([a.exposure_count for a in second.next_frame.population.agents],
                         [1, 2, 1])
        self.assertEqual(second.round_result.round_index, 2)
        self.assertEqual(second.global_round, 2)
        self.assertEqual(second.tick_mass, Fraction(1))

    def test_attach_source_informs_both_old_neighbor_and_newborn(self):
        step = _advance_tick(genesis_frame(), birth("2", (0,)))
        self.assertEqual([a.belief for a in step.next_frame.population.agents],
                         [1.0, 1.0, 1.0])
        self.assertEqual([a.exposure_count for a in step.next_frame.population.agents],
                         [0, 1, 1])

    def test_zero_receptivity_still_counts_incoming_exposure(self):
        step = _advance_tick(
            genesis_frame(), birth("2", (0,), receptivity=0),
        )
        newborn = {a.agent_id: a for a in step.next_frame.population.agents}["2"]
        self.assertEqual((newborn.belief, newborn.exposure_count), (0.0, 1))

    def test_silent_population_emits_no_transmissions(self):
        initial = genesis_frame()
        silent_population = PopulationState(
            initial.model.model_id,
            initial.model.content_hash,
            0,
            None,
            tuple(NetworkAgentState(agent_id, 0.0, index + 2, False)
                  for index, agent_id in enumerate(initial.agent_ids)),
        )
        silent = replace(initial, population=silent_population)
        step = _advance_tick(silent, BBRuntimeTick())
        self.assertEqual(step.round_result.transmissions, ())
        self.assertEqual([a.exposure_count for a in step.next_frame.population.agents],
                         [2, 3])

    def test_zero_threshold_newborn_broadcasts_zero_belief_immediately(self):
        initial = genesis_frame()
        silent_population = PopulationState(
            initial.model.model_id,
            initial.model.content_hash,
            0,
            None,
            tuple(NetworkAgentState(agent_id, 0.0, 0, False)
                  for agent_id in initial.agent_ids),
        )
        silent = replace(initial, population=silent_population)
        step = _advance_tick(silent, birth("2", (1,), threshold=0))
        sent = tuple((item.source_agent_id, item.target_agent_id, item.signal)
                     for item in step.round_result.transmissions)
        self.assertEqual(sent, (("2", "1", 0.0),))
        self.assertEqual([a.exposure_count for a in step.next_frame.population.agents],
                         [0, 1, 0])

    def test_initially_broadcasting_newborn_emits_on_birth_tick(self):
        step = _advance_tick(
            genesis_frame(), birth("2", (1,), belief=1, threshold=0.5),
        )
        self.assertIn(
            ("2", "1", 1.0),
            tuple((item.source_agent_id, item.target_agent_id, item.signal)
                  for item in step.round_result.transmissions),
        )

    def test_birth_preserves_old_role_and_adoption_threshold(self):
        initial = genesis_frame(role="archivist", adoption_threshold=0.875)
        step = _advance_tick(initial, birth("2", (1,)))
        old = {profile.agent_id: profile for profile in initial.model.agents}
        grown = {profile.agent_id: profile for profile in step.next_frame.model.agents}
        self.assertEqual({key: grown[key] for key in old}, old)

    def test_duplicate_id_allocates_nothing_and_does_not_propagate(self):
        initial = genesis_frame()
        with patch("narrative_dynamics.abm.bb_runtime._apply_birth") as growth, \
             patch("narrative_dynamics.abm.bb_runtime.simulate_round") as operator:
            result = _advance_tick(initial, birth("1", (0,)))
        self.assert_error(
            result, stage="tick_agent", code="duplicateAgentId",
            field="agent_id", tick_index=0, birth_index=0,
        )
        growth.assert_not_called()
        operator.assert_not_called()

    def test_bb_error_precedes_conflicting_newborn_error(self):
        result = _advance_tick(
            genesis_frame(), birth("2", (0,), fitness=0, belief=2),
        )
        self.assert_error(
            result, stage="tick_network", code="nonpositiveFitness",
            field="fitness", tick_index=0, birth_index=0,
        )
        self.assertEqual(result.bb_cause, "nonpositiveFitness")

    def test_non_dyadic_threshold_near_case_equals_independent_v1_round(self):
        initial = genesis_frame()
        profiles = (
            NetworkAgentSpec("0", "peer", 0.3, 0.5, 0.1 + 0.2),
            NetworkAgentSpec("1", "peer", 0.3, 0.5, 0.1 + 0.2),
        )
        model = NetworkABMModel(
            "bb-runtime", "1", profiles,
            SocialNetwork(
                ("0", "1"),
                (
                    SocialEdge("0", "1", "bb", 1.0, True),
                    SocialEdge("1", "0", "bb", 1.0, True),
                ),
            ),
        )
        population = PopulationState(
            model.model_id, model.content_hash, 0, None,
            (
                NetworkAgentState("0", 0.1, 0, False),
                NetworkAgentState("1", 0.2, 0, False),
            ),
        )
        prior = replace(initial, model=model, population=population)
        raw = BBRuntimeTick(
            "birth",
            BBRuntimeBirth(
                1,
                (1,),
                BBRuntimeNewborn("2", "peer", 0.3, 0.3, 0.3),
            ),
        )

        expected_profiles = profiles + (
            NetworkAgentSpec("2", "peer", 0.3, 0.5, 0.3),
        )
        expected_model = NetworkABMModel(
            "bb-runtime", "1", expected_profiles,
            SocialNetwork(
                ("0", "1", "2"),
                (
                    SocialEdge("0", "1", "bb", 1.0, True),
                    SocialEdge("1", "0", "bb", 1.0, True),
                    SocialEdge("1", "2", "bb", 1.0, True),
                    SocialEdge("2", "1", "bb", 1.0, True),
                ),
            ),
        )
        expected_post_growth = PopulationState(
            expected_model.model_id, expected_model.content_hash, 0, None,
            (
                NetworkAgentState("0", 0.1, 0, False),
                NetworkAgentState("1", 0.2, 0, False),
                NetworkAgentState("2", 0.3, 0, True),
            ),
        )
        expected = simulate_round(expected_model, expected_post_growth)

        actual = _advance_tick(prior, raw)
        self.assertEqual(actual.round_result, expected)

    def test_each_agent_scalar_rejects_nonfinite_values_and_booleans(self):
        base = BBRuntimeNewborn("2", "peer", 1, 0.5, 0, 0.5)
        codes = {
            "receptivity": "invalidAgentValue",
            "broadcast_threshold": "invalidAgentValue",
            "belief": "invalidAgentValue",
            "adoption_threshold": "invalidAdoptionThreshold",
        }
        for field, semantic_code in codes.items():
            for value in (float("nan"), float("inf"), float("-inf"), True):
                with self.subTest(field=field, value=value):
                    candidate = replace(base, **{field: value})
                    if field != "adoption_threshold":
                        candidate = replace(candidate, agent_id="", role="")
                    raw = BBRuntimeTick(
                        "birth", BBRuntimeBirth(1, (0,), candidate),
                    )
                    result = _advance_tick(genesis_frame(), raw)
                    self.assert_error(
                        result,
                        stage="tick_agent",
                        code="invalidType" if value is True else semantic_code,
                        field=field,
                        tick_index=0,
                        birth_index=0,
                    )

    def test_agent_validation_uses_semantic_field_order(self):
        malformed = BBRuntimeNewborn(
            "", "", float("nan"), float("inf"), float("-inf"), True,
        )
        result = _advance_tick(
            genesis_frame(), BBRuntimeTick("birth", BBRuntimeBirth(1, (0,), malformed)),
        )
        self.assert_error(
            result, stage="tick_agent", code="invalidAgentValue",
            field="receptivity", tick_index=0, birth_index=0,
        )

        result = _advance_tick(
            genesis_frame(), BBRuntimeTick(
                "birth", BBRuntimeBirth(
                    1, (0,), replace(malformed, receptivity=1),
                ),
            ),
        )
        self.assert_error(
            result, stage="tick_agent", code="invalidAgentValue",
            field="broadcast_threshold", tick_index=0, birth_index=0,
        )

        result = _advance_tick(
            genesis_frame(), BBRuntimeTick(
                "birth", BBRuntimeBirth(
                    1, (0,), replace(
                        malformed, receptivity=1, broadcast_threshold=0.5,
                    ),
                ),
            ),
        )
        self.assert_error(
            result, stage="tick_agent", code="invalidAgentValue",
            field="belief", tick_index=0, birth_index=0,
        )

    def test_seed_exposure_precedes_identity_metadata(self):
        raw = BBRuntimeAgent("", "", 1, 0.5, 0, True, 2)
        result = _parse_agent(
            raw, newborn=False, used_ids=frozenset(), stage="seed_agent",
            agent_index=4,
        )
        self.assert_error(
            result, stage="seed_agent", code="invalidType",
            field="exposure_count", agent_index=4,
        )

    def test_identity_duplicate_role_and_adoption_follow_scalar_checks(self):
        cases = (
            (BBRuntimeNewborn(7, "", 1, 0.5, 0, 2), frozenset(),
             "invalidType", "agent_id"),
            (BBRuntimeNewborn("", "", 1, 0.5, 0, 2), frozenset(),
             "invalidText", "agent_id"),
            (BBRuntimeNewborn("2", "", 1, 0.5, 0, 2), frozenset({"2"}),
             "duplicateAgentId", "agent_id"),
            (BBRuntimeNewborn("2", "", 1, 0.5, 0, 2), frozenset(),
             "invalidText", "role"),
            (BBRuntimeNewborn("2", "peer", 1, 0.5, 0, 2), frozenset(),
             "invalidAdoptionThreshold", "adoption_threshold"),
        )
        for raw, used_ids, code, field in cases:
            with self.subTest(code=code):
                result = _parse_agent(
                    raw, newborn=True, used_ids=used_ids, stage="tick_agent",
                    tick_index=8, birth_index=3,
                )
                self.assert_error(
                    result, stage="tick_agent", code=code, field=field,
                    tick_index=8, birth_index=3,
                )

    def test_huge_integer_probability_overflow_is_a_field_error(self):
        result = _parse_agent(
            BBRuntimeNewborn("2", "peer", 10 ** 1000, 0.5, 0),
            newborn=True,
            used_ids=frozenset(),
            stage="tick_agent",
            tick_index=2,
            birth_index=1,
        )
        self.assert_error(
            result, stage="tick_agent", code="invalidAgentValue",
            field="receptivity", tick_index=2, birth_index=1,
        )

    def test_current_tick_record_and_tag_errors_have_current_indices(self):
        cases = (
            (object(), "invalidType", "tick"),
            (BBRuntimeTick(True), "invalidType", "kind"),
            (BBRuntimeTick("later"), "invalidTickKind", "kind"),
            (BBRuntimeTick("idle", object()), "unexpectedBirthData", "birth"),
            (BBRuntimeTick("birth"), "missingBirthData", "birth"),
            (BBRuntimeTick("birth", object()), "invalidType", "birth"),
        )
        for raw, code, field in cases:
            with self.subTest(code=code):
                result = _advance_tick(genesis_frame(), raw)
                self.assert_error(
                    result, stage="tick", code=code, field=field,
                    tick_index=0, birth_index=0,
                )


if __name__ == "__main__":
    unittest.main()

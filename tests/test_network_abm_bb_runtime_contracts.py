from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction
import json
import unittest

from narrative_dynamics.abm.bb_runtime_contracts import (
    BBRuntimeAgent, BBRuntimeBirth, BBRuntimeError,
    BBRuntimeNewborn, BBRuntimeRawSeed, BBRuntimeReplay, BBRuntimeSeed,
    BBRuntimeTick, BBRuntimeTopology,
)
from narrative_dynamics.abm.contracts import (
    NetworkAgentState, SocialNetwork,
)
from narrative_dynamics.abm.simulation import PopulationTrajectory, simulate_round
from narrative_dynamics.contracts import stable_content_hash
from tests.bb_runtime_fixtures import (
    birth, genesis_frame, manual_birth, manual_idle, seed,
)


class BBRuntimeContractsTests(unittest.TestCase):
    def test_raw_requests_defer_domain_checks(self):
        values = [False, object()]
        network = BBRuntimeRawSeed(False, values, "invalid")
        agent = BBRuntimeAgent("", None, False, float("nan"), 2, -1, object())
        newborn = BBRuntimeNewborn("", None, False, -1, 2, object())
        request = BBRuntimeSeed(network, (agent,), None, False)
        tick = BBRuntimeTick("invalid", BBRuntimeBirth(0, values, newborn))
        self.assertIs(request.network.fitness, values)
        self.assertIs(tick.birth.targets, values)
        self.assertEqual(len(values), 2)
        self.assertEqual(seed().network.node_count, 2)
        self.assertEqual(birth("2", (1,)).birth.targets, (1,))

    def test_newborn_has_no_exposure_override(self):
        self.assertNotIn("exposure_count", {f.name for f in fields(BBRuntimeNewborn)})
        with self.assertRaises(TypeError):
            BBRuntimeNewborn("2", "peer", 1, 0.5, 0, exposure_count=1)

    def test_empty_wrapper_and_numeric_registry(self):
        frame = genesis_frame(ids=("z", "a"), exposures=(7, 3))
        replay = BBRuntimeReplay(frame, (), frame)
        self.assertEqual(replay.final, frame)
        self.assertEqual(frame.agent_ids, ("z", "a"))
        self.assertEqual(tuple(a.agent_id for a in frame.model.agents), ("a", "z"))
        self.assertEqual(frame.trace_mass, Fraction(1))
        self.assertEqual(frame.population.round_index, 0)
        self.assertEqual(frame.epoch, 0)
        with self.assertRaises(FrozenInstanceError):
            frame.tick_count = 1

    def test_config_rejects_invalid_naturals_and_seed_counts(self):
        valid = genesis_frame().config
        for changes in (
            {"m": 0}, {"m": 3}, {"m": True}, {"seed_node_count": 1},
            {"seed_node_count": True}, {"seed_edge_count": 0},
            {"seed_edge_count": 2}, {"seed_edge_count": True},
            {"model_id": ""}, {"version": " "},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(valid, **changes)

    def test_topology_rejects_implicit_numeric_and_edge_repairs(self):
        for fitness, edges in (
            ((1, Fraction(1)), ((0, 1),)),
            ((True, Fraction(1)), ((0, 1),)),
            ((Fraction(0), Fraction(1)), ((0, 1),)),
            ((Fraction(-1), Fraction(1)), ((0, 1),)),
            ((Fraction(1),), ()),
            ((Fraction(1), Fraction(1)), ((1, 0),)),
            ((Fraction(1), Fraction(1)), ((0, 1), (0, 1))),
            ((Fraction(1), Fraction(1)), ((0, 0),)),
            ((Fraction(1), Fraction(1)), ((0, 2),)),
            ((Fraction(1), Fraction(1)), ((False, 1),)),
            ((Fraction(1), Fraction(1)), [(0, 1)]),
            ([Fraction(1), Fraction(1)], ((0, 1),)),
            ((Fraction(1),) * 3, ((1, 2), (0, 1))),
        ):
            with self.subTest(fitness=fitness, edges=edges):
                with self.assertRaises((TypeError, ValueError)):
                    BBRuntimeTopology(fitness, edges)

    def test_exact_rational_serialization(self):
        frame = genesis_frame(fitness=(Fraction(1, 3), 2))
        self.assertEqual(frame.topology.to_dict(), {
            "node_count": 2, "fitness": ["1/3", "2"], "edges": [[0, 1]],
        })
        self.assertEqual(frame.to_dict()["trace_mass"], "1")
        step = manual_birth(fitness=Fraction(7, 3))
        self.assertEqual(step.to_dict()["tick_mass"], "1/2")
        self.assertEqual(step.to_dict()["tick"]["birth"]["fitness"], "7/3")

    def test_rejects_forged_model_and_roster_bindings(self):
        frame = genesis_frame()
        wrong_hash = replace(frame.population, model_hash="sha256:" + "0" * 64)
        for changes in (
            {"population": wrong_hash},
            {"agent_ids": ("0", "ghost")},
            {"agent_ids": ("0", "0")},
            {"agent_ids": ["0", "1"]},
            {"agent_ids": ("0",)},
            {"trace_mass": Fraction(1, 2)},
            {"parent_frame_hash": "sha256:" + "1" * 64},
            {"applied_tick": BBRuntimeTick()},
            {"config": replace(frame.config, version="2")},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    replace(frame, **changes)

    def test_frame_rejects_wrong_roster_and_broadcasting(self):
        frame = genesis_frame()
        population = replace(frame.population, agents=frame.population.agents[:1])
        with self.assertRaises(ValueError):
            replace(frame, population=population)
        changed = replace(frame.population.agents[0], broadcasting=False)
        population = replace(frame.population, agents=(changed, frame.population.agents[1]))
        with self.assertRaises(ValueError):
            replace(frame, population=population)

    def test_frame_requires_exact_unit_bidirectional_bb_channels(self):
        frame = genesis_frame()
        for edges in (
            frame.model.network.edges[:1],
            tuple(replace(e, influence=0.5) for e in frame.model.network.edges),
            tuple(replace(e, active=False) for e in frame.model.network.edges),
            tuple(replace(e, relation_type="peer") for e in frame.model.network.edges),
        ):
            model = replace(frame.model, network=SocialNetwork(frame.agent_ids, edges))
            population = replace(frame.population, model_hash=model.content_hash)
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                replace(frame, model=model, population=population)

    def test_frame_checks_counts_and_exact_mass_domain(self):
        frame = manual_birth().next_frame
        for changes in (
            {"tick_count": True}, {"birth_count": True}, {"birth_count": 2},
            {"tick_count": 0}, {"trace_mass": 0.5}, {"trace_mass": True},
            {"trace_mass": Fraction(0)}, {"trace_mass": Fraction(-1)},
            {"trace_mass": Fraction(2)}, {"parent_frame_hash": None},
            {"applied_tick": None},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    replace(frame, **changes)

    def test_applied_tick_is_normalized_and_immutable(self):
        step = manual_birth()
        raw = step.tick.birth
        for tick in (
            BBRuntimeTick("unknown"),
            BBRuntimeTick("idle", raw),
            BBRuntimeTick("birth"),
            BBRuntimeTick("birth", replace(raw, fitness=1)),
            BBRuntimeTick("birth", replace(raw, targets=[1])),
            BBRuntimeTick("birth", replace(raw, targets=(True,))),
            BBRuntimeTick("birth", replace(raw, targets=(2,))),
            BBRuntimeTick("birth", replace(raw, targets=())),
            BBRuntimeTick("birth", replace(raw, agent=replace(raw.agent, belief=0))),
            BBRuntimeTick("birth", replace(raw, agent=replace(raw.agent, belief=float("nan")))),
        ):
            with self.subTest(tick=tick), self.assertRaises((TypeError, ValueError)):
                replace(step.next_frame, applied_tick=tick)

    def test_birth_stage_preserves_old_states_and_metadata(self):
        initial = genesis_frame(ids=("z", "a"), exposures=(7, 3),
                                role="observer", adoption_threshold=0.25)
        step = manual_birth(initial, agent_id="new")
        old = {a.agent_id: a for a in initial.population.agents}
        post = {a.agent_id: a for a in step.post_growth_population.agents}
        profiles = {p.agent_id: p for p in step.next_frame.model.agents}
        self.assertEqual({k: post[k] for k in old}, old)
        self.assertEqual(post["new"], NetworkAgentState("new", 0, 0, False))
        for profile in initial.model.agents:
            self.assertEqual(profiles[profile.agent_id], profile)
        self.assertEqual(step.next_frame.agent_ids, ("z", "a", "new"))
        self.assertEqual(step.post_growth_population.round_index, 0)
        self.assertIsNone(step.post_growth_population.parent_state_hash)
        self.assertEqual(step.round_result.round_index, 1)
        self.assertEqual(step.next_frame.population.parent_state_hash,
                         step.post_growth_population.content_hash)
        self.assertEqual(step.next_frame.parent_frame_hash, initial.content_hash)

    def with_post_growth(self, step, post, *, model=None):
        model = step.next_frame.model if model is None else model
        result = simulate_round(model, post)
        next_frame = replace(step.next_frame, model=model, population=result.next_state)
        return replace(step, post_growth_population=post, round_result=result,
                       next_frame=next_frame)

    def test_transition_rejects_coherent_but_changed_old_states(self):
        step = manual_birth(genesis_frame(exposures=(7, 3)))
        old = step.post_growth_population.agents[0]
        for changed in (replace(old, belief=0.75), replace(old, exposure_count=8)):
            states = (changed,) + step.post_growth_population.agents[1:]
            post = replace(step.post_growth_population, agents=states)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.with_post_growth(step, post)

    def test_transition_rejects_nonzero_newborn_exposure_or_wrong_initial_belief(self):
        step = manual_birth()
        for newborn in (
            NetworkAgentState("2", 0, 1, False),
            NetworkAgentState("2", 0.75, 0, True),
        ):
            post = replace(step.post_growth_population,
                           agents=step.post_growth_population.agents[:2] + (newborn,))
            with self.subTest(newborn=newborn), self.assertRaises(ValueError):
                self.with_post_growth(step, post)

    def test_transition_rejects_changed_old_profile_in_new_model(self):
        step = manual_birth()
        for changes in ({"role": "other"}, {"adoption_threshold": 0.25},
                        {"receptivity": 0.5}):
            profiles = (replace(step.next_frame.model.agents[0], **changes),) + \
                step.next_frame.model.agents[1:]
            model = replace(step.next_frame.model, agents=profiles)
            post = replace(step.post_growth_population, model_hash=model.content_hash)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.with_post_growth(step, post, model=model)

    def test_transition_requires_exact_parent_clock_and_mass(self):
        step = manual_birth()
        for changes in (
            {"parent_frame_hash": "sha256:" + "f" * 64},
            {"tick_count": 2},
            {"trace_mass": Fraction(1, 4)},
        ):
            candidate = replace(step.next_frame, **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(step, next_frame=candidate)
        with self.assertRaises(ValueError):
            replace(step, tick_mass=Fraction(1, 4))
        with self.assertRaises(TypeError):
            replace(step, tick_mass=0.5)

    def test_transition_rejects_mismatched_round_input_and_output(self):
        step = manual_birth()
        other = manual_birth(targets=(0,))
        with self.assertRaises(ValueError):
            replace(step, post_growth_population=other.post_growth_population)
        with self.assertRaises(ValueError):
            replace(step, round_result=other.round_result)

    def test_birth_rejects_wrong_id_fitness_profile_and_target_binding(self):
        step = manual_birth()
        raw = step.tick.birth
        for changed in (
            replace(raw, targets=(0,)),
            replace(raw, fitness=Fraction(2)),
            replace(raw, agent=replace(raw.agent, agent_id="0")),
            replace(raw, agent=replace(raw.agent, agent_id="ghost")),
            replace(raw, agent=replace(raw.agent, role="other")),
        ):
            tick = BBRuntimeTick("birth", changed)
            with self.subTest(tick=tick), self.assertRaises(ValueError):
                next_frame = replace(step.next_frame, applied_tick=tick)
                replace(step, tick=tick, next_frame=next_frame)

    def test_idle_continues_epoch_and_birth_opens_another(self):
        first = manual_birth()
        idle = manual_idle(first.next_frame)
        second = manual_birth(idle.next_frame, targets=(2,),
                              tick_mass=Fraction(1, 4))
        replay = BBRuntimeReplay(first.prior, (first, idle, second), second.next_frame)
        self.assertEqual([s.global_round for s in replay.transitions], [1, 2, 3])
        self.assertEqual([s.tick_index for s in replay.transitions], [0, 1, 2])
        self.assertEqual([s.birth_index for s in replay.transitions], [0, 1, 1])
        self.assertEqual([s.next_frame.epoch for s in replay.transitions], [1, 1, 2])
        self.assertEqual([s.round_result.round_index for s in replay.transitions], [1, 2, 1])
        self.assertEqual(idle.post_growth_population, first.next_frame.population)
        self.assertEqual(second.next_frame.trace_mass, Fraction(1, 8))

    def test_idle_rejects_topology_change_and_probability_factor(self):
        step = manual_idle(genesis_frame())
        topology = replace(step.next_frame.topology, fitness=(Fraction(2), Fraction(1)))
        candidate = replace(step.next_frame, topology=topology)
        with self.assertRaises(ValueError):
            replace(step, next_frame=candidate)
        candidate = replace(step.next_frame, trace_mass=Fraction(1, 2))
        with self.assertRaises(ValueError):
            replace(step, tick_mass=Fraction(1, 2), next_frame=candidate)

    def test_birth_cannot_continue_an_old_v1_local_clock(self):
        step = manual_birth()
        post = replace(step.post_growth_population, round_index=1,
                       parent_state_hash="sha256:" + "a" * 64)
        with self.assertRaises(ValueError):
            self.with_post_growth(step, post)

    def test_replay_rejects_discontinuous_initial_tail_and_order(self):
        first = manual_birth()
        idle = manual_idle(first.next_frame)
        other = manual_birth(targets=(0,))
        for initial, transitions, final in (
            (first.prior, (), first.next_frame),
            (first.next_frame, (), first.next_frame),
            (first.prior, (idle,), idle.next_frame),
            (first.prior, (first, idle), first.next_frame),
            (first.prior, (first, other), other.next_frame),
            (first.prior, [first], first.next_frame),
        ):
            with self.subTest(transitions=transitions):
                with self.assertRaises((TypeError, ValueError)):
                    BBRuntimeReplay(initial, transitions, final)

    def test_equal_outcomes_retain_distinct_ordered_history(self):
        initial = genesis_frame(m=2)
        left = manual_birth(initial, targets=(0, 1))
        right = manual_birth(initial, targets=(1, 0))
        self.assertEqual(left.next_frame.model, right.next_frame.model)
        self.assertEqual(left.next_frame.population, right.next_frame.population)
        self.assertEqual(left.next_frame.trace_mass, right.next_frame.trace_mass)
        self.assertNotEqual(left.next_frame.content_hash, right.next_frame.content_hash)
        self.assertNotEqual(left.content_hash, right.content_hash)
        a = BBRuntimeReplay(initial, (left,), left.next_frame)
        b = BBRuntimeReplay(initial, (right,), right.next_frame)
        self.assertNotEqual(a.content_hash, b.content_hash)
        self.assertEqual(left.to_dict()["tick"]["birth"]["targets"], [0, 1])
        self.assertEqual(right.to_dict()["tick"]["birth"]["targets"], [1, 0])

    def test_derived_enumeration_does_not_change_registry_or_hash(self):
        left = manual_birth(genesis_frame(ids=("z", "a"), m=2), targets=(0, 1))
        right = manual_birth(genesis_frame(ids=("z", "a"), m=2, reverse=True),
                             targets=(0, 1), reverse=True)
        self.assertEqual(left.next_frame.agent_ids, ("z", "a", "2"))
        self.assertEqual(left.content_hash, right.content_hash)
        self.assertEqual(
            BBRuntimeReplay(left.prior, (left,), left.next_frame).content_hash,
            BBRuntimeReplay(right.prior, (right,), right.next_frame).content_hash,
        )

    def test_registry_crosses_lexical_nine_to_ten_boundary(self):
        ids = tuple(str(i) for i in range(10))
        initial = genesis_frame(ids=ids, exposures=tuple(range(10)), fitness=(1,) * 10)
        step = manual_birth(initial, targets=(9,), tick_mass=Fraction(1, 18))
        self.assertEqual(step.next_frame.agent_ids, ids + ("10",))
        self.assertEqual(tuple(a.agent_id for a in step.next_frame.model.agents)[:4],
                         ("0", "1", "10", "2"))
        post = {a.agent_id: a for a in step.post_growth_population.agents}
        for old in initial.population.agents:
            self.assertEqual(post[old.agent_id], old)

    def test_serialization_is_detached_complete_and_uses_existing_hash(self):
        step = manual_birth()
        replay = BBRuntimeReplay(step.prior, (step,), step.next_frame)
        original = replay.content_hash
        document = replay.to_dict()
        self.assertEqual(original, stable_content_hash(document))
        self.assertEqual(document["schema"], "bb-abm-runtime-v1")
        self.assertEqual(document["initial"]["schema"], "bb-abm-runtime-v1")
        self.assertEqual(document["transitions"][0]["next_frame"], step.next_frame.to_dict())
        self.assertEqual(document["transitions"][0]["transition"], step.to_dict())
        self.assertEqual(document["final_frame_hash"], step.next_frame.content_hash)
        self.assertEqual(json.loads(json.dumps(document)), document)
        document["initial"]["agent_ids"].append("foreign")
        document["transitions"][0]["next_frame"]["model"]["agents"][0]["role"] = "changed"
        document["transitions"][0]["transition"]["tick"]["birth"]["targets"].clear()
        self.assertEqual(replay.content_hash, original)
        self.assertNotIn("transition_hash", step.next_frame.to_dict())

    def test_replay_binds_content_hash_when_float_equality_is_insufficient(self):
        step = manual_birth()
        original = step.prior
        states = (original.population.agents[0],
                  replace(original.population.agents[1], belief=-0.0))
        alternative = replace(original, population=replace(original.population, agents=states))
        self.assertEqual(original, alternative)
        self.assertNotEqual(original.content_hash, alternative.content_hash)
        with self.assertRaises(ValueError):
            BBRuntimeReplay(alternative, (step,), step.next_frame)

    def test_old_signed_zero_cannot_be_rewritten_at_a_birth(self):
        step = manual_birth()
        states = list(step.post_growth_population.agents)
        states[1] = replace(states[1], belief=-0.0)
        post = replace(step.post_growth_population, agents=tuple(states))
        self.assertEqual(post, step.post_growth_population)
        self.assertNotEqual(post.content_hash, step.post_growth_population.content_hash)
        with self.assertRaises(ValueError):
            self.with_post_growth(step, post)

    def test_error_has_no_success_prefix_payload(self):
        error = BBRuntimeError(
            "tick_network", "targetOutOfRange", field="targets",
            tick_index=2, birth_index=0, bb_cause="targetOutOfRange",
        )
        self.assertEqual((error.tick_index, error.birth_index), (2, 0))
        self.assertEqual(set(error.to_dict()), {
            "stage", "code", "field", "agent_index", "tick_index", "birth_index",
            "bb_cause", "expected", "actual",
        })
        self.assertFalse({"final", "transitions", "trace_mass", "population"}
                         & error.to_dict().keys())
        self.assertEqual(error.to_dict()["bb_cause"], "targetOutOfRange")
        with self.assertRaises(FrozenInstanceError):
            error.tick_index = 3

    def test_error_indices_and_cause_are_structured_values(self):
        error = BBRuntimeError(
            "seed_agents", "seedAgentCount", field="agents", expected=2, actual=1,
        )
        self.assertEqual((error.expected, error.actual), (2, 1))
        for changes in ({"expected": True}, {"actual": -1}, {"field": []}):
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    replace(error, **changes)
        with self.assertRaises(ValueError):
            BBRuntimeError("tick_network", "duplicateTarget", tick_index=-1, birth_index=0)

    def test_v1_empty_trajectory_and_cross_model_round_still_reject(self):
        step = manual_birth()
        frame = step.prior
        with self.assertRaises(ValueError):
            PopulationTrajectory(frame.model.model_id, frame.model.content_hash,
                                 frame.population, (), frame.population)
        with self.assertRaises(ValueError):
            replace(step.round_result, prior_state=frame.population)


if __name__ == "__main__":
    unittest.main()

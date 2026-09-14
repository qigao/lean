from dataclasses import replace
from fractions import Fraction
import unittest

from narrative_dynamics.abm.bb_runtime import (
    _apply_birth,
    _check_birth,
    _check_m,
    _parse_bb_seed,
)
from narrative_dynamics.abm.bb_runtime_contracts import (
    BBRuntimeError,
    BBRuntimeRawSeed,
)
from tests.bb_runtime_fixtures import birth, seed


class BBRuntimeKernelTests(unittest.TestCase):
    def assert_error(self, result, *, stage, code, field):
        self.assertIsInstance(result, BBRuntimeError)
        self.assertEqual((result.stage, result.code, result.field),
                         (stage, code, field))
        self.assertEqual(
            result.bb_cause,
            code if stage in ("seed_network", "tick_network")
            and code != "invalidType" else None,
        )
        self.assertIsNone(result.expected)
        self.assertIsNone(result.actual)

    def test_ordered_weighted_birth_uses_frozen_undirected_degrees(self):
        topology = _parse_bb_seed(seed(fitness=(1, 3)).network)
        checks = [
            _check_birth(topology, 2, birth("2", order).birth,
                         tick_index=0, birth_index=0)
            for order in ((0, 1), (1, 0))
        ]
        self.assertEqual([c.tick_mass for c in checks],
                         [Fraction(1, 4), Fraction(3, 4)])
        left, right = (_apply_birth(topology, c) for c in checks)
        self.assertEqual(left, right)
        self.assertEqual(left.edges, ((0, 1), (0, 2), (1, 2)))
        self.assertEqual(topology.edges, ((0, 1),))

    def test_second_birth_uses_new_degrees_only_on_the_next_tick(self):
        topology = _parse_bb_seed(seed().network)
        first = _check_birth(topology, 1, birth("2", (1,)).birth,
                             tick_index=0, birth_index=0)
        grown = _apply_birth(topology, first)
        second = _check_birth(grown, 1, birth("3", (2,)).birth,
                              tick_index=1, birth_index=1)
        self.assertEqual((first.tick_mass, second.tick_mass),
                         (Fraction(1, 2), Fraction(1, 4)))
        self.assertEqual(first.tick_mass * second.tick_mass, Fraction(1, 8))

    def test_network_fields_fail_before_any_newborn_agent_field(self):
        topology = _parse_bb_seed(seed().network)
        raw = birth("2", (0, 0), fitness=0, belief=2).birth
        error = _check_birth(topology, 2, raw, tick_index=4, birth_index=1)
        self.assertIsInstance(error, BBRuntimeError)
        self.assertEqual((error.stage, error.code, error.bb_cause),
                         ("tick_network", "nonpositiveFitness", "nonpositiveFitness"))
        self.assertEqual((error.tick_index, error.birth_index), (4, 1))

    def test_seed_validation_precedence_and_error_fields(self):
        valid = seed().network
        cases = (
            ("record type", object(), "invalidType", "network"),
            ("node type", replace(valid, node_count=True), "invalidType", "node_count"),
            ("node minimum", replace(valid, node_count=1), "invalidNodeCount", "node_count"),
            ("fitness container", replace(valid, fitness=[1, 1]), "invalidType", "fitness"),
            ("node before fitness size",
             BBRuntimeRawSeed(1, (1, 1), ((0, 1),)), "invalidNodeCount", "node_count"),
            ("fitness size", replace(valid, fitness=(1,)),
             "fitnessSizeMismatch", "fitness"),
            ("fitness element type", replace(valid, fitness=(1.0, 1)),
             "invalidType", "fitness[0]"),
            ("fitness bool", replace(valid, fitness=(True, 1)),
             "invalidType", "fitness[0]"),
            ("zero fitness before edges",
             BBRuntimeRawSeed(2, (0, 1), ((0, 2),)),
             "nonpositiveFitness", "fitness"),
            ("negative fitness before edges",
             BBRuntimeRawSeed(2, (-1, 1), ((0, 2),)),
             "nonpositiveFitness", "fitness"),
            ("edges container", replace(valid, edges=[(0, 1)]),
             "invalidType", "edges"),
            ("edge pair type", replace(valid, edges=([0, 1],)),
             "invalidType", "edges[0]"),
            ("edge pair shape", replace(valid, edges=((0,),)),
             "invalidEdge", "edges"),
            ("endpoint type", replace(valid, edges=((0, True),)),
             "invalidType", "edges[0][1]"),
            ("self edge", replace(valid, edges=((0, 0),)),
             "invalidEdge", "edges"),
            ("edge bounds before reversed duplicate",
             BBRuntimeRawSeed(3, (1, 1, 1), ((0, 1), (1, 0), (2, 3))),
             "invalidEdge", "edges"),
            ("duplicate edges before connectedness",
             BBRuntimeRawSeed(3, (1, 1, 1), ((0, 1), (1, 0))),
             "duplicateEdge", "edges"),
            ("disconnected seed",
             BBRuntimeRawSeed(4, (1, 1, 1, 1), ((0, 1), (2, 3))),
             "disconnectedSeed", "edges"),
        )
        for label, raw, code, field in cases:
            with self.subTest(label=label):
                self.assert_error(_parse_bb_seed(raw), stage="seed_network",
                                  code=code, field=field)

    def test_seed_success_canonicalizes_edges_and_retains_exact_fitness(self):
        raw = BBRuntimeRawSeed(
            3,
            (Fraction(1, 3), 2, Fraction(5, 7)),
            ((2, 1), (1, 0)),
        )
        topology = _parse_bb_seed(raw)
        self.assertEqual(topology.fitness,
                         (Fraction(1, 3), Fraction(2), Fraction(5, 7)))
        self.assertEqual(topology.edges, ((0, 1), (1, 2)))

    def test_m_validation_covers_initial_and_tick_stages(self):
        cases = (
            (0, "initial_m", "initialM"),
            (3, "initial_m", "initialM"),
            (0, "tick_network", "invalidM"),
            (3, "tick_network", "invalidM"),
        )
        for value, stage, code in cases:
            with self.subTest(value=value, stage=stage):
                result = _check_m(value, 2, stage=stage, tick_index=5, birth_index=3)
                self.assert_error(result, stage=stage, code=code, field="m")
        for stage in ("initial_m", "tick_network"):
            with self.subTest(stage=stage):
                result = _check_m(True, 2, stage=stage, tick_index=5, birth_index=3)
                self.assert_error(result, stage=stage, code="invalidType", field="m")

    def test_birth_validation_precedence_and_error_fields(self):
        topology = _parse_bb_seed(seed().network)
        valid = birth("2", (0,)).birth
        cases = (
            ("m before fitness", 0, replace(valid, fitness=0),
             "invalidM", "m"),
            ("fitness float", 1, replace(valid, fitness=1.0),
             "invalidType", "fitness"),
            ("fitness bool", 1, replace(valid, fitness=True),
             "invalidType", "fitness"),
            ("fitness positive", 1, replace(valid, fitness=0),
             "nonpositiveFitness", "fitness"),
            ("targets container", 1, replace(valid, targets=[0]),
             "invalidType", "targets"),
            ("target count before bounds", 2, replace(valid, targets=(0, 9, 1)),
             "targetCountMismatch", "targets"),
            ("target type", 1, replace(valid, targets=(True,)),
             "invalidType", "targets[0]"),
            ("negative target", 1, replace(valid, targets=(-1,)),
             "targetOutOfRange", "targets"),
            ("duplicate target", 2, replace(valid, targets=(0, 0)),
             "duplicateTarget", "targets"),
        )
        for label, m, raw, code, field in cases:
            with self.subTest(label=label):
                result = _check_birth(topology, m, raw,
                                      tick_index=4, birth_index=1)
                self.assert_error(result, stage="tick_network", code=code, field=field)
                self.assertEqual((result.tick_index, result.birth_index), (4, 1))

        three = _parse_bb_seed(BBRuntimeRawSeed(
            3, (1, 1, 1), ((0, 1), (1, 2))))
        result = _check_birth(
            three, 3, replace(valid, targets=(0, 0, 3)),
            tick_index=4, birth_index=1,
        )
        self.assert_error(result, stage="tick_network",
                          code="targetOutOfRange", field="targets")

    def test_birth_accepts_m_equal_to_node_count(self):
        topology = _parse_bb_seed(seed().network)
        checked = _check_birth(topology, 2, birth("2", (0, 1)).birth,
                               tick_index=0, birth_index=0)
        self.assertEqual(checked.tick_mass, Fraction(1, 2))

    def test_nonuniform_path_uses_frozen_weighted_sampling_without_replacement(self):
        topology = _parse_bb_seed(BBRuntimeRawSeed(
            3, (2, 3, 5), ((0, 1), (1, 2))))
        checks = tuple(
            _check_birth(topology, 2, birth("3", targets).birth,
                         tick_index=0, birth_index=0)
            for targets in ((1, 0), (0, 1))
        )
        self.assertEqual(tuple(item.tick_mass for item in checks),
                         (Fraction(12, 91), Fraction(12, 143)))
        left, right = (_apply_birth(topology, item) for item in checks)
        self.assertEqual(left, right)
        self.assertEqual(left.edges,
                         ((0, 1), (0, 3), (1, 2), (1, 3)))


if __name__ == "__main__":
    unittest.main()

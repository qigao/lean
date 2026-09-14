"""Complete authored runtime histories exported by the actual Lean replay."""

import json
from fractions import Fraction
from pathlib import Path
import unittest

from narrative_dynamics.abm.bb_runtime import replay_bb_population
from narrative_dynamics.abm.bb_runtime_contracts import (
    BBRuntimeAgent, BBRuntimeBirth, BBRuntimeError, BBRuntimeNewborn,
    BBRuntimeRawSeed, BBRuntimeReplay, BBRuntimeSeed, BBRuntimeTick,
)


CORPUS = Path(__file__).resolve().parents[1] / "conformance" / "bb_abm_runtime_v1.json"
SUCCESS_IDS = {
    "empty", "idle-two", "attach-source", "attach-relay", "relay-idle",
    "successive-births", "successive-idle", "weighted-source", "ordered-01",
    "ordered-10", "zero-receptive", "silent", "zero-threshold",
    "broadcast-newborn", "retained-exposures", "half-receptive",
}
ERROR_IDS = {
    "seed-nodes", "seed-size", "seed-fitness", "seed-edge", "seed-duplicate",
    "seed-disconnected", "initial-m-zero", "initial-m-too-large",
    "seed-agent-count", "seed-agent-r", "seed-agent-threshold", "seed-agent-belief",
    "birth-fitness", "birth-target-count", "birth-target-range",
    "birth-target-duplicate", "birth-agent-threshold", "late-birth",
    "first-failure", "late-first-birth",
}


def decode_input(raw):
    """Decode the finite Lean corpus domain, retaining invalid raw scalars."""
    network = raw["seed"]
    agents = tuple(
        BBRuntimeAgent(str(i), "peer", float(Fraction(a["receptivity"])),
                       float(Fraction(a["threshold"])), float(Fraction(a["belief"])),
                       a["exposures"], 0.5)
        for i, a in enumerate(raw["agents"])
    )
    seed = BBRuntimeSeed(
        BBRuntimeRawSeed(network["node_count"],
                         tuple(Fraction(x) for x in network["fitness"]),
                         tuple(tuple(e) for e in network["edges"])),
        agents, "bb-runtime-conformance", "1",
    )
    ticks = []
    births = 0
    for tick in raw["ticks"]:
        if tick is None:
            ticks.append(BBRuntimeTick())
        else:
            newborn = BBRuntimeNewborn(
                str(network["node_count"] + births), "peer",
                float(Fraction(tick["receptivity"])),
                float(Fraction(tick["threshold"])), float(Fraction(tick["belief"])),
                0.5,
            )
            ticks.append(BBRuntimeTick("birth", BBRuntimeBirth(
                Fraction(tick["fitness"]), tuple(tick["targets"]), newborn)))
            births += 1
    return seed, raw["m"], tuple(ticks)


class BBRuntimeConformanceTests(unittest.TestCase):
    def assert_state(self, frame, population, expected):
        self.assertEqual(frame.topology.node_count, expected["node_count"])
        self.assertEqual([list(edge) for edge in frame.topology.edges], expected["edges"])
        self.assertEqual(frame.topology.fitness, tuple(map(Fraction, expected["fitness"])))
        profiles = {a.agent_id: a for a in frame.model.agents}
        states = {a.agent_id: a for a in population.agents}
        self.assertEqual(set(profiles), set(frame.agent_ids))
        self.assertEqual(set(states), set(frame.agent_ids))
        self.assertEqual([profiles[i].receptivity for i in frame.agent_ids],
                         [float(Fraction(x)) for x in expected["receptivity"]])
        self.assertEqual([profiles[i].broadcast_threshold for i in frame.agent_ids],
                         [float(Fraction(x)) for x in expected["thresholds"]])
        self.assertEqual([states[i].belief for i in frame.agent_ids],
                         [float(Fraction(x)) for x in expected["beliefs"]])
        self.assertEqual([states[i].exposure_count for i in frame.agent_ids],
                         expected["exposures"])
        self.assertEqual([states[i].broadcasting for i in frame.agent_ids],
                         expected["broadcasting"])

    def test_complete_lean_runtime_histories(self):
        corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
        self.assertEqual(corpus["schema"], "bb-abm-runtime-v1")
        self.assertEqual(len(corpus["success"]), 16)
        self.assertEqual(len(corpus["errors"]), 20)
        self.assertEqual({case["id"] for case in corpus["success"]}, SUCCESS_IDS)
        self.assertEqual({case["id"] for case in corpus["errors"]}, ERROR_IDS)
        for case in corpus["success"]:
            with self.subTest(case=case["id"]):
                out = replay_bb_population(*decode_input(case["input"]))
                self.assertIsInstance(out, BBRuntimeReplay)
                frames = (out.initial,) + tuple(t.next_frame for t in out.transitions)
                self.assertEqual(len(frames), len(case["prefixes"]))
                self.assertEqual(len(out.transitions), len(case["transitions"]))
                for frame, expected in zip(frames, case["prefixes"]):
                    self.assertEqual(frame.tick_count, expected["tick_count"])
                    self.assertEqual(frame.birth_count, expected["birth_count"])
                    self.assertEqual(frame.trace_mass, Fraction(expected["trace_mass"]))
                    self.assert_state(frame, frame.population, expected["state"])
                for step, expected in zip(out.transitions, case["transitions"]):
                    self.assertEqual(step.tick_index, expected["tick_index"])
                    self.assertEqual(step.birth_index, expected["birth_index"])
                    self.assertEqual(step.tick_mass, Fraction(expected["tick_mass"]))
                    self.assert_state(step.next_frame, step.post_growth_population,
                                      expected["post_growth"])
                    ids = {agent_id: i for i, agent_id in enumerate(step.next_frame.agent_ids)}
                    actual = sorted((ids[t.source_agent_id], ids[t.target_agent_id], t.signal)
                                    for t in step.round_result.transmissions)
                    wanted = [(t["source"], t["target"], float(Fraction(t["signal"])))
                              for t in expected["transmissions"]]
                    self.assertEqual(actual, wanted)
                    for transmission in step.round_result.transmissions:
                        self.assertEqual(transmission.round_index, step.round_result.round_index)
                        self.assertEqual(transmission.relation_type, "bb")
                        self.assertEqual(transmission.influence, 1.0)
        for case in corpus["errors"]:
            with self.subTest(case=case["id"]):
                out = replay_bb_population(*decode_input(case["input"]))
                self.assertIsInstance(out, BBRuntimeError)
                self.assertEqual(out.to_dict(), case["expected_error"])

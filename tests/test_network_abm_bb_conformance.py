import json
from fractions import Fraction
from pathlib import Path
import unittest

from narrative_dynamics.abm.contracts import (
    NetworkABMModel, NetworkAgentSpec, NetworkAgentState,
    PopulationState, SocialEdge, SocialNetwork,
)
from narrative_dynamics.abm.simulation import simulate_round

CORPUS = Path(__file__).resolve().parents[1] / "conformance" / "bb_abm_v1.json"

def number(text):
    return float(Fraction(text))

class BBPropagationConformanceTests(unittest.TestCase):
    def test_shared_vectors_match_existing_v1_round(self):
        corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
        self.assertEqual(corpus["schema"], "bb-abm-v1")
        self.assertEqual(len(corpus["vectors"]), 8)
        self.assertEqual({v["id"] for v in corpus["vectors"]}, {
            "attach-source", "attach-relay", "relay-next-round", "second-birth",
            "half-receptive", "zero-receptive", "silent", "zero-threshold",
        })
        for vector in corpus["vectors"]:
            for reverse in (False, True):
                with self.subTest(case=vector["id"], reverse=reverse):
                    raw, expected = vector["input"], vector["expected"]
                    ids = tuple(str(i) for i in range(len(raw["beliefs"])))
                    profiles = tuple(
                        NetworkAgentSpec(
                            agent_id, "peer", number(raw["receptivity"][i]),
                            0.5, number(raw["thresholds"][i]),
                        ) for i, agent_id in enumerate(ids)
                    )
                    edges = tuple(
                        SocialEdge(str(source), str(target), "peer", 1.0)
                        for u, v in raw["edges"]
                        for source, target in ((u, v), (v, u))
                    )
                    model = NetworkABMModel(
                        "bb-conformance:" + vector["id"], "1",
                        profiles[::-1] if reverse else profiles,
                        SocialNetwork(
                            ids[::-1] if reverse else ids,
                            edges[::-1] if reverse else edges,
                        ),
                    )
                    states = tuple(
                        NetworkAgentState(
                            agent_id, number(raw["beliefs"][i]), raw["exposures"][i],
                            number(raw["beliefs"][i]) >= number(raw["thresholds"][i]),
                        ) for i, agent_id in enumerate(ids)
                    )
                    prior = PopulationState(
                        model.model_id, model.content_hash, 0, None,
                        states[::-1] if reverse else states,
                    )
                    result = simulate_round(model, prior)
                    actual = sorted(result.next_state.agents, key=lambda a: int(a.agent_id))
                    self.assertEqual([a.belief for a in actual], list(map(number, expected["beliefs"])))
                    self.assertEqual([a.exposure_count for a in actual], expected["exposures"])
                    self.assertEqual([a.broadcasting for a in actual], expected["broadcasting"])
                    delivered = sorted(
                        [int(t.source_agent_id), int(t.target_agent_id)]
                        for t in result.transmissions
                    )
                    self.assertEqual(delivered, expected["transmissions"])

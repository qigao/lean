import unittest

from typed_graph import NodeKind, TypedNode
from provenance import ProofRecord
from belief_support import BeliefKB, belief_supported
from world_graph import WorldState
from observation_admission import observation_evidence_admissible


class ObservationAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.event = "panic"
        self.agent = "hamlet"
        self.event_fact = TypedNode(NodeKind.EVENT, self.event)
        graph = {
            "obs": ProofRecord(self.event_fact, None, (), 0),
        }
        self.kb = BeliefKB(owner=self.agent, graph=graph, active={"obs"})

    def world(self, with_path: bool = True) -> WorldState:
        info_edges = {"event:panic": {"agent:hamlet"}} if with_path else {}
        return WorldState(
            alive={self.agent},
            can_act={self.agent},
            info_edges=info_edges,
            event_nodes={self.event: "event:panic"},
            agent_nodes={self.agent: "agent:hamlet"},
            observations={(self.event, self.agent)},
            causal_edges=set(),
            event_time={self.event: 1},
        )

    def test_observed_event_with_valid_root_proof_is_admissible_evidence(self):
        world = self.world(with_path=True)
        self.assertTrue(observation_evidence_admissible(world, self.kb, self.event, "obs"))
        self.assertTrue(belief_supported(self.kb, self.event_fact))

    def test_forged_observation_without_information_path_is_rejected(self):
        world = self.world(with_path=False)
        self.assertFalse(observation_evidence_admissible(world, self.kb, self.event, "obs"))

    def test_non_observation_root_cannot_be_admitted_as_raw_perception(self):
        bad_graph = dict(self.kb.graph)
        bad_graph["derived"] = ProofRecord(self.event_fact, None, ("obs",), 1)
        bad_kb = BeliefKB(owner=self.agent, graph=bad_graph, active={"obs", "derived"})
        world = self.world(with_path=True)
        self.assertFalse(observation_evidence_admissible(world, bad_kb, self.event, "derived"))


if __name__ == "__main__":
    unittest.main()

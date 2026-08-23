from __future__ import annotations

from typed_graph import NodeKind, TypedNode
from belief_support import BeliefKB
from truth_maintenance import proof_valid
from world_graph import WorldState, invariant_holds


def observation_evidence_admissible(
    world: WorldState,
    kb: BeliefKB,
    event: str,
    proof_id: str,
) -> bool:
    """Admit only world-observed, valid, raw event evidence into perception."""
    if not invariant_holds(world):
        return False
    if (event, kb.owner) not in world.observations:
        return False
    if proof_id not in kb.graph:
        return False
    if not proof_valid(kb.graph, kb.active, proof_id):
        return False

    record = kb.graph[proof_id]
    if record.fact != TypedNode(NodeKind.EVENT, event):
        return False
    if record.origin is not None:
        return False
    if record.parents:
        return False
    return True

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Callable, Mapping, Iterable

from epistemic import info_reachable


@dataclass
class WorldState:
    alive: set[str]
    can_act: set[str]
    info_edges: Mapping[str, Iterable[str]]
    event_nodes: Mapping[str, str]
    agent_nodes: Mapping[str, str]
    observations: set[tuple[str, str]]
    causal_edges: set[tuple[str, str]]
    event_time: Mapping[str, int]


def invariant_holds(world: WorldState) -> bool:
    """Check the minimal world invariants used by the Lean model."""
    if not world.can_act.issubset(world.alive):
        return False

    for event, agent in world.observations:
        if event not in world.event_nodes or agent not in world.agent_nodes:
            return False
        if not info_reachable(
            world.info_edges,
            world.event_nodes[event],
            world.agent_nodes[agent],
        ):
            return False

    for source, target in world.causal_edges:
        if source not in world.event_time or target not in world.event_time:
            return False
        if not world.event_time[source] < world.event_time[target]:
            return False

    return True


def kill_agent(world: WorldState, agent: str) -> WorldState:
    """A legal death rewrite removes both liveness and action capability."""
    alive = set(world.alive)
    can_act = set(world.can_act)
    alive.discard(agent)
    can_act.discard(agent)
    return replace(world, alive=alive, can_act=can_act)


def add_observation(world: WorldState, event: str, agent: str) -> WorldState:
    """Insert an observation only when a permitted information path exists."""
    if event not in world.event_nodes or agent not in world.agent_nodes:
        raise ValueError("unknown event or agent node")
    if not info_reachable(
        world.info_edges,
        world.event_nodes[event],
        world.agent_nodes[agent],
    ):
        raise ValueError("observation has no information path")

    observations = set(world.observations)
    observations.add((event, agent))
    return replace(world, observations=observations)


def add_causal_edge(world: WorldState, source: str, target: str) -> WorldState:
    """Insert a causal edge only when event time strictly increases."""
    if source not in world.event_time or target not in world.event_time:
        raise ValueError("unknown event time")
    if not world.event_time[source] < world.event_time[target]:
        raise ValueError("causal edge must strictly increase time")

    causal_edges = set(world.causal_edges)
    causal_edges.add((source, target))
    return replace(world, causal_edges=causal_edges)


def apply_rewrites(
    world: WorldState,
    rewrites: list[Callable[[WorldState], WorldState]],
) -> WorldState:
    """Apply only rewrites that preserve the invariant at every step."""
    if not invariant_holds(world):
        raise ValueError("initial world violates invariant")

    state = world
    for rewrite in rewrites:
        state = rewrite(state)
        if not invariant_holds(state):
            raise ValueError("rewrite violated world invariant")
    return state

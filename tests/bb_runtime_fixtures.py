"""Literal record fixtures; these do not implement the future BB executor."""

from fractions import Fraction

from narrative_dynamics.abm.bb_runtime_contracts import (
    BBRuntimeAgent, BBRuntimeBirth, BBRuntimeConfig, BBRuntimeFrame,
    BBRuntimeNewborn, BBRuntimeRawSeed, BBRuntimeSeed, BBRuntimeTick,
    BBRuntimeTopology, BBRuntimeTransition,
)
from narrative_dynamics.abm.contracts import (
    NetworkABMModel, NetworkAgentSpec, NetworkAgentState, PopulationState,
    SocialEdge, SocialNetwork,
)
from narrative_dynamics.abm.simulation import simulate_round


def seed(*, fitness=(1, 1), ids=("0", "1"), exposures=(0, 0)):
    return BBRuntimeSeed(
        BBRuntimeRawSeed(len(ids), fitness, tuple((i, i + 1) for i in range(len(ids) - 1))),
        tuple(BBRuntimeAgent(agent_id, "peer", 1, 0.5, int(i == 0), exposures[i])
              for i, agent_id in enumerate(ids)),
    )


def birth(agent_id, targets, *, fitness=1, belief=0, receptivity=1, threshold=0.5):
    return BBRuntimeTick(
        "birth", BBRuntimeBirth(
            fitness, targets,
            BBRuntimeNewborn(agent_id, "peer", receptivity, threshold, belief),
        ),
    )


def fixture_model(config, ids, topology, profiles, *, reverse=False):
    channels = tuple(
        SocialEdge(ids[source], ids[target], "bb", 1.0, True)
        for u, v in topology.edges
        for source, target in ((u, v), (v, u))
    )
    return NetworkABMModel(
        config.model_id, config.version, profiles[::-1] if reverse else profiles,
        SocialNetwork(ids[::-1] if reverse else ids,
                      channels[::-1] if reverse else channels),
    )


def genesis_frame(*, ids=("0", "1"), exposures=(0, 0), m=1,
                  fitness=(1, 1), reverse=False, role="peer",
                  adoption_threshold=0.5):
    edges = tuple((i, i + 1) for i in range(len(ids) - 1))
    topology = BBRuntimeTopology(tuple(Fraction(f) for f in fitness), edges)
    config = BBRuntimeConfig("bb-runtime", "1", m, len(ids), len(edges))
    profiles = tuple(NetworkAgentSpec(i, role, 1, adoption_threshold, 0.5) for i in ids)
    model = fixture_model(config, ids, topology, profiles, reverse=reverse)
    states = tuple(NetworkAgentState(agent_id, int(i == 0), exposures[i], i == 0)
                   for i, agent_id in enumerate(ids))
    population = PopulationState(
        model.model_id, model.content_hash, 0, None,
        states[::-1] if reverse else states,
    )
    return BBRuntimeFrame(
        config, 0, 0, ids, topology, model, population, Fraction(1), None, None,
    )


def manual_birth(prior=None, *, targets=(1,), agent_id=None, fitness=1,
                 belief=0, receptivity=1, threshold=0.5, role="peer",
                 adoption_threshold=0.5, tick_mass=Fraction(1, 2), reverse=False):
    """Author one known record using a literal mass and the existing V1 round."""
    if prior is None:
        prior = genesis_frame()
    n = prior.topology.node_count
    agent_id = str(n) if agent_id is None else agent_id
    newborn = BBRuntimeNewborn(
        agent_id, role, float(receptivity), float(threshold), float(belief),
        float(adoption_threshold),
    )
    tick = BBRuntimeTick("birth", BBRuntimeBirth(Fraction(fitness), targets, newborn))
    ids = prior.agent_ids + (agent_id,)
    topology = BBRuntimeTopology(
        prior.topology.fitness + (Fraction(fitness),),
        tuple(sorted(prior.topology.edges + tuple((t, n) for t in targets))),
    )
    profile = NetworkAgentSpec(agent_id, role, receptivity, adoption_threshold, threshold)
    model = fixture_model(prior.config, ids, topology, prior.model.agents + (profile,),
                          reverse=reverse)
    states = prior.population.agents + (
        NetworkAgentState(agent_id, belief, 0, belief >= threshold),
    )
    post = PopulationState(model.model_id, model.content_hash, 0, None,
                           states[::-1] if reverse else states)
    result = simulate_round(model, post)
    next_frame = BBRuntimeFrame(
        prior.config, prior.tick_count + 1, prior.birth_count + 1, ids, topology,
        model, result.next_state, prior.trace_mass * tick_mass,
        prior.content_hash, tick,
    )
    return BBRuntimeTransition(prior, tick, post, result, tick_mass, next_frame)


def manual_idle(prior):
    tick = BBRuntimeTick()
    result = simulate_round(prior.model, prior.population)
    next_frame = BBRuntimeFrame(
        prior.config, prior.tick_count + 1, prior.birth_count, prior.agent_ids,
        prior.topology, prior.model, result.next_state, prior.trace_mass,
        prior.content_hash, tick,
    )
    return BBRuntimeTransition(
        prior, tick, prior.population, result, Fraction(1), next_frame,
    )

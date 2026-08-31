# Network Interaction and Emergence V1 Design

## Objective

Turn the existing fixed-roster cognitive multi-agent runtime into a minimal population ABM in which information moves only through an explicit social network, changes persistent per-agent state over synchronous rounds, and produces reproducible macro-level measurements and intervention contrasts.

V1 keeps the population roster and role labels fixed. Population birth, death, entry, exit, network rewiring, and learned model parameters are explicitly deferred. An agent's behavior still changes: its belief, cumulative exposure count, adoption status, and broadcasting status evolve as information arrives.

## Scope

V1 provides four tightly coupled capabilities:

1. Immutable agent, directed social-edge, network, model, and population-state contracts with canonical ordering and stable content hashes.
2. Deterministic synchronous local propagation where round `t` reads one shared snapshot and commits every update to round `t + 1`.
3. Population measurements for mean belief, adoption, broadcasting, informed reach, consensus, and polarization.
4. Declarative edge and seed interventions with paired baseline/treatment trajectories and metric deltas.

The implementation is dependency-free Python and lives under `narrative_dynamics.abm`. It does not modify the generic narrative scheduler in V1; the ABM runtime is a focused composition boundary that can later adapt narrative evidence into network signals.

## Contracts

### Agent profile

`NetworkAgentSpec` contains:

- `agent_id`: stable non-empty identifier;
- `role`: stable non-empty role label;
- `receptivity`: finite value in `[0, 1]` controlling assimilation;
- `adoption_threshold`: finite value in `[0, 1]` used by macro metrics;
- `broadcast_threshold`: finite value in `[0, 1]` controlling whether the agent sends during the next round.

Roles are fixed classifications in V1. Belief-dependent behavior is dynamic.

### Social edge

`SocialEdge` is directed from `source_agent_id` to `target_agent_id` and contains a relation type, influence in `[0, 1]`, and active flag. Self-edges and duplicate `(source, target, relation_type)` identities are rejected. Zero-influence and inactive edges do not transmit.

### Model and state

`NetworkABMModel` binds an exact set of profiles to a `SocialNetwork`. Both are canonicalized by identifiers before hashing.

`NetworkAgentState` contains belief in `[0, 1]`, cumulative non-negative exposure count, and current broadcasting status. `PopulationState` binds those states to an exact model hash, round index, and parent-state hash. Initial states have no parent; subsequent states bind the exact previous state.

## Synchronous propagation

For every active positive-influence edge whose source is broadcasting in the prior snapshot, one `InformationTransmission` is emitted. Its signal is the source's prior belief. Transmissions are collected before any agent changes state.

For recipient `i` with incoming transmissions:

```text
neighbor_signal = sum(edge_weight * source_belief) / sum(edge_weight)
assimilation = receptivity_i * min(1, sum(edge_weight))
next_belief_i = belief_i + assimilation * (neighbor_signal - belief_i)
```

The result is clamped only for floating-point boundary noise; validated inputs make the mathematical value remain in `[0, 1]`. Exposure count increases by the number of received transmissions. Broadcasting for the next round is `next_belief >= broadcast_threshold`.

Agents without incoming transmissions preserve belief and exposure count. Because every transmission reads the prior snapshot, information cannot travel more than one graph hop per round and input ordering cannot change semantics.

## Emergence measurements

For `N` agents with beliefs `b_i`:

- `mean_belief`: arithmetic mean of `b_i`;
- `adoption_rate`: fraction where `b_i >= adoption_threshold_i`;
- `broadcasting_rate`: fraction currently broadcasting;
- `informed_rate`: fraction with positive cumulative exposure count;
- `consensus`: `1 - (max(b_i) - min(b_i))`;
- `polarization`: `4 * mean((b_i - mean_belief)^2)`, bounded in `[0, 1]` for beliefs in `[0, 1]`.

Consensus and polarization describe distribution shape; informed rate prevents an all-uninformed population from being mistaken for successful diffusion.

## Interventions

`NetworkIntervention` may disable named edges, override edge influence, and seed agent beliefs. Every target must exist and an edge cannot be both disabled and overridden. Applying an intervention creates a distinct model identity and a rebound initial state; it does not mutate baseline objects.

`compare_intervention` runs the baseline and treatment for the same number of deterministic synchronous rounds, measures both final populations, and reports treatment-minus-baseline deltas. Disabling every edge is the V1 no-propagation null model.

## Failure and reproducibility rules

- All public contracts reject wrong types, non-finite values, invalid ranges, duplicate identities, and model/state mismatch.
- Collections are detached into immutable canonical tuples.
- State and trajectory lineage is validated before execution.
- No partial next state is returned on validation failure.
- Equivalent input order produces the same model, state, trajectory, and comparison hashes.

## Acceptance scenarios

1. In `A -> B -> C`, with only `A` initially broadcasting, `B` changes in round one and `C` only in round two.
2. A disconnected agent remains unchanged.
3. Zero-influence, inactive, and disabled edges carry no transmission.
4. Reversing input agent and edge order preserves exact hashes and results.
5. Replaying a trajectory produces the same content hash.
6. Hand-computed population metrics match exact expected values.
7. Disabling `B -> C` changes only the treatment path and yields the expected metric deltas.
8. The no-propagation null preserves every non-seeded agent state.

## Deferred work

- stochastic contact and paired random streams;
- dynamic network rewiring;
- parameter learning and trust adaptation;
- population birth, death, entry, and exit;
- multiple information topics and contradiction resolution;
- degree-preserving and complete-mixing null networks;
- adapter from narrative testimony/evidence ledgers into ABM transmissions.


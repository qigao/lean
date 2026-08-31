# Population Lifecycle V3 Design

## Objective

Extend the network ABM so the effective population changes between rounds. Catalogued agents may enter, exit, or die; lifecycle events apply at the round boundary, and only agents active after those events participate in that round's local propagation.

V3 keeps the identity/profile catalog and social graph fixed and auditable. It changes active membership, not the existence or meaning of identities. Creation of previously unknown identities, reproduction, profile evolution, and topology rewiring remain deferred.

## Round semantics

For round `t + 1`:

1. Validate and canonically order that round's lifecycle events.
2. Apply every event atomically to the prior lifecycle snapshot.
3. Build the induced network containing only agents now marked active and edges whose endpoints are both active.
4. Run one synchronous V1 propagation step over that induced network.
5. Recombine propagated active states with unchanged inactive and dead states.

An entering agent participates immediately. An exiting or dying agent cannot send or receive in the same round. All active transmissions still read one post-event, pre-propagation snapshot, so information travels at most one edge per round.

## Status transitions

Each catalogued agent has exactly one status:

- `inactive`: outside the effective population but eligible to enter;
- `active`: inside the effective population and eligible to send and receive;
- `dead`: permanently outside the effective population.

Allowed transitions are `inactive -> active` by `ENTER`, `active -> inactive` by `EXIT`, and `inactive|active -> dead` by `DEATH`. All other transitions are rejected. At most one event may target an agent in one round.

An `ENTER` event may provide a belief seed. Without a seed, a returning agent preserves its prior belief and exposure count. With a seed, belief is replaced and exposure count increases by one. Entry recomputes broadcasting from the catalogued profile threshold. Exit and death force broadcasting off. Initial active agents have one recorded entry; initial inactive agents have none.

## Contracts

`PopulationLifecycleModel` binds an exact `NetworkABMModel` catalog and a canonical non-empty set of initially active agent ids.

`LifecycleMemberState` stores agent id, belief, exposure count, broadcasting, status, entry count, and exit count. The state tuple always covers the complete catalog, including inactive and dead agents.

`PopulationLifecycleState` binds the exact lifecycle model, round index, parent-state hash, and complete member-state tuple.

`PopulationLifecycleEvent` identifies one catalogued agent, an `ENTER`, `EXIT`, or `DEATH` kind, and an optional entry belief. Entry belief is forbidden for exit and death.

`LifecycleRoundResult` binds the exact prior state, canonical applied events, transmissions among active agents, and next lifecycle state. `LifecycleTrajectory` validates a complete multi-round chain.

All public contracts are frozen, canonically ordered, serializable with `to_dict()`, and content-addressed with the repository's stable SHA-256 hashing.

## Empty active population

The catalog must remain non-empty, but the effective active population may be empty. Such a round is valid: it produces no transmissions, advances the round and parent-state chain, and preserves every member except for applied lifecycle changes. The V1 kernel is bypassed because a V1 `PopulationState` intentionally requires at least one agent.

## Population view and metrics

`active_population_view` projects active lifecycle members to a V1-compatible `PopulationState` over an induced model. It returns `None` when no agent is active.

`PopulationLifecycleMetrics` reports:

- catalog population size;
- active, inactive, and dead counts;
- active and dead shares of the catalog;
- cumulative entries;
- cumulative exits.

Death is represented by status and does not increment the voluntary exit counter. This keeps exit and mortality analytically distinct.

## Acceptance scenarios

1. In catalog line `A -> B -> C`, with only A initially active, entering B lets A influence B immediately while C remains unchanged.
2. Entering C in the next round lets the belief B acquired in the prior round influence C; no signal crosses two edges in one round.
3. Exiting B before propagation removes the bridge and blocks all A-to-C transmission.
4. Killing B removes it before propagation and any later attempt to re-enter B is rejected.
5. A state with no active agents advances deterministically with zero transmissions.
6. Reversing event input order preserves exact results and content hashes.
7. A returning agent preserves learned state unless its entry event supplies a new belief seed.
8. Lifecycle metrics match hand-computed catalog counts and cumulative event counts.

## Deferred work

- births and creation of identities outside the catalog;
- reproduction, inheritance, and profile mutation;
- endogenous entry, exit, and mortality hazards;
- topology rewiring when membership changes;
- lifecycle-aware adaptive trust retention policies;
- multiple topics and role evolution.

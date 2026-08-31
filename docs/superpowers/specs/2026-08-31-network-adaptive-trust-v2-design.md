# Network Adaptive Trust V2 Design

## Objective

Extend network interaction V1 so agents learn which neighboring sources are reliable. Information changes recipient belief immediately; later truth feedback changes edge-local trust, and the learned trust changes the strength of future information from that source.

V2 retains the V1 fixed population, fixed role labels, directed topology, synchronous rounds, and scalar belief topic. Population lifecycle, topology rewiring, multiple topics, and endogenous truth discovery remain deferred.

## Learning semantics

Every directed social edge has persistent trust `T` in `[0, 1]` and a non-negative feedback count. During propagation, the effective influence of an edge is:

```text
effective_influence = base_edge_influence * current_trust
```

All transmissions in round `t` read the same prior agent and trust snapshot. Truth feedback is evaluated only after those transmissions and therefore affects round `t + 1`, never the current round.

For a transmitted signal `s` and externally verified truth `y`, both in `[0, 1]`:

```text
accuracy = 1 - abs(s - y)
next_trust = trust + learning_rate * (accuracy - trust)
```

This is the existing prediction-error learning rule applied to source accuracy. Accurate sources move upward, inaccurate sources move downward, rates zero and one have their usual no-learning and full-update meanings, and all results remain bounded.

## Contracts

`AdaptiveTrustModel` binds one exact `NetworkABMModel`, a learning rate, and an initial trust shared by every edge. Its content hash is distinct from the base model.

`EdgeTrustState` identifies one exact edge and stores trust plus feedback count. `AdaptivePopulationState` stores the canonical agent states and all edge trust states under one adaptive model identity and parent-state chain.

`TruthFeedback` identifies a transmitted edge and an observed truth. Feedback edge identities must be unique within a round and must correspond to a transmission that actually occurred in that round.

`EdgeTrustUpdate` records old trust, transmitted signal, observed truth, derived accuracy, and new trust. `AdaptiveRoundResult` binds the exact prior state, effective transmissions, feedback, updates, and next state.

## Execution

`initialize_adaptive_population` reuses V1 belief initialization and creates one trust state per social edge.

`simulate_adaptive_round` derives a temporary effective topology from prior trust, delegates the synchronous belief transition to the V1 propagation kernel, applies feedback after propagation, and returns an adaptive next state bound to the prior adaptive state hash.

`simulate_adaptive_population` accepts one feedback tuple per round. An empty tuple means no truth feedback in that round. Input order is canonical and cannot affect hashes.

## Metrics

`AdaptiveTrustMetrics` reports:

- `mean_trust`;
- `min_trust`;
- `max_trust`;
- `learned_edge_rate`, the fraction of edges with at least one feedback observation.

V1 `measure_emergence` continues to measure population belief outcomes after the adaptive agent states are projected to a V1-compatible population snapshot.

## Acceptance scenarios

1. With learning rate `0.5` and prior trust `0.5`, perfectly accurate feedback moves trust to `0.75` and perfectly inaccurate feedback moves it to `0.25`.
2. Both sources have equal effective influence in the feedback round; the updated trust affects only the following round.
3. After feedback, a target exposed to accurate belief `1` and inaccurate belief `0` moves from `0.5` to `0.75` because the accurate source receives three times the trust weight.
4. No-feedback rounds preserve every trust value and count.
5. Feedback for an edge that did not transmit is rejected before a next adaptive state is returned.
6. Reversing feedback input order preserves exact state, trajectory, and hash identity.
7. Learning rate zero reproduces fixed-trust propagation across rounds.
8. Trust metrics match hand-computed values.

## Deferred work

- multiple claims and topic-specific trust;
- conflicting objective feedback sources;
- agent-generated verification actions;
- dynamic edge creation and deletion;
- population birth, death, entry, and exit.

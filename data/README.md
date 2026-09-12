# FlyWire V0 data contract

V0 consumes a **local, user-supplied CSV export**. The runtime does not embed FlyWire credentials and does not silently query a remote service.

Required columns:

- `pre_id`: presynaptic FlyWire neuron/root identifier
- `post_id`: postsynaptic FlyWire neuron/root identifier
- `synapse_count`: positive integer synapse count used as the directed edge weight

Optional annotation columns:

- `region`
- `cell_type`

Every experimental selection must record both:

1. the exact FlyWire release/data identity (`release_id`), and
2. the explicit extraction/selection rule (`selection_rule`).

These provenance strings are mandatory and must be carried into experiment manifests. Neuron identifiers are deterministically remapped by sorted original ID when constructing the model graph. Parallel rows for the same directed pair are aggregated by synapse count.

For V0, prefer a bounded visual/motion-related subgraph that is small enough to train under the frozen compute budget. The selection rule must be fixed before final-test evaluation; do not choose neurons, regions, cell types, thresholds, or graph size using final-test results.

The topology-specific comparison requires a matched rewired control derived from the same selected graph. Rewiring must preserve node count, edge count, directed in/out degree sequence, and the edge-weight multiset while changing the graph fingerprint.

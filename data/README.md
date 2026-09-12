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

## Frozen YOLO/Pose observations

Model-comparison runs consume a pre-exported JSONL file; they do **not** invoke YOLO at training/evaluation time. Freeze and record the detector/Pose version before producing the comparison dataset.

Each JSONL record represents one frame and requires:

- `sample_id`: stable sequence identifier;
- `frame_index`: non-negative frame index within the sequence;
- `label`: dedicated target label only;
- `body_keypoints`: ordered `[x, y, confidence]` triples.

Optional fields are `hand_keypoints`, `bbox`, and `detector_confidence`. Their presence and keypoint counts are part of the frozen observation schema and must remain constant throughout one dataset. Missing or uncertain keypoints should be represented through their confidence values rather than by changing keypoint count or order.

The loader sorts by `(sample_id, frame_index)`, rejects duplicate frame identities and schema drift, and computes an observation-schema hash from the declared field layout/counts rather than observation values. Unknown feature payloads are rejected so target labels cannot be copied into model inputs.

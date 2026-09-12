# Pinned four-arm pose development runner

Tracked by #68. `pose_comparison.run_pose_comparison` composes existing input,
provenance, model and training contracts. It actually executes GRU, random sparse
graph, matched rewired graph and selected FlyWire graph arms. It does not extend
the old `compare` CLI or expose a final-test evaluator.

## Inputs and preflight

The entry requires the pose bundle path and every argument required by
`load_pose_development`, plus an independently retained prepared-binding SHA-256.
It reloads the source files and recomputes that binding, not merely accepting a
caller-constructed partition or a hash-shaped declaration.

Graph inputs are the topology protocol JSON, connectivity CSV and pre-generated
control-bundle JSON, with independent protocol and control-file SHA-256 pins.
The existing provenance context verifies CSV SHA-256/Git-blob identity, T4/T5
selection, graph statistics and original fingerprint. All supplied controls must
match the exact generator/runtime/protocol identity, frozen seed fingerprints,
directed degree sequence, global weight multiset and fixed diagonal edges. The
runner never regenerates invalid controls, retries with fewer swaps or substitutes
the original graph. JSON is parsed with the existing strict reader.

`PoseComparisonSpec` explicitly supplies ordered seeds, epochs, learning rate,
batch size, exact update budget, parameter ceiling, GRU hidden width and graph
node width. Seeds and epoch/update/parameter budget must agree with the pinned
topology protocol. For the complete training roster:

```
max_updates == epochs * ceil(number_of_training_samples / batch_size)
```

An incompatible budget is an error before optimization, not early stopping,
dropping samples or automatically increasing the budget. Widths are not selected
from validation scores. All model/partition/configuration checks complete before
the first training call. CPU float32 is the supported execution condition.

## Shared conditions, distinct controls

Each seed runs the same four families, in the fixed order GRU, random graph,
rewired, FlyWire. All consume the same pinned observation/label/split/class-map
binding, optimizer configuration, epoch permutations and full update budget.
Only validation macro-F1 selects each arm's earliest best checkpoint, through the
existing padded trainer. Padding masks and real missing frames retain their
existing meanings.

The three graph models have identical node widths and learned-parameter counts;
the GRU only shares the parameter ceiling, not an exact parameter count. The
random baseline uses the existing loop-free, unit-weight `random_sparse_graph`
with the original node/edge counts and the run seed. It is NOT a degree/weight
matched control; only the separately verified rewired graph serves that role.
Graph weights are consumed by the existing recurrence without normalization or
any equation change.

Every arm records its graph fingerprint, actual float32 adjacency hash (graph
adjacency is nonpersistent in model state), learned-state hash, parameter count,
selected epoch, validation metrics, optimizer count and epoch-order hashes.
Order evidence is checked against an independently generated expected permutation
sequence, not just against the first arm. Inputs, selected state and adjacency
are rechecked around execution. A failed arm prevents a comparison report from
being returned; already executed optimization is not rolled back or retried.
Caller RNG/deterministic settings are restored by scoped construction/training.

## Usage

```python
from yolo_flywire.pose_comparison import PoseComparisonSpec, run_pose_comparison

report = run_pose_comparison(
    pose_bundle_path,
    root=rgb_root, inventory=rgb_inventory,
    extraction_spec=frozen_extraction_spec, feature_spec=frozen_feature_spec,
    classes=frozen_ordered_classes,
    expected_manifest_sha256=frozen_manifest_sha256,
    expected_encoder_hash=frozen_encoder_hash,
    expected_binding_sha256=frozen_prepared_binding_sha256,
    topology_protocol=topology_protocol_path, connectivity=connectivity_csv_path,
    controls=control_bundle_path,
    expected_topology_sha256=frozen_topology_protocol_sha256,
    expected_controls_sha256=frozen_control_bundle_sha256,
    config=PoseComparisonSpec(**frozen_development_execution_config),
)
```

Paths, pins and configuration are independently supplied, not inferred from the
current files. Fixture tests deliberately prepare their own independent pins;
that is not a real-corpus freeze procedure. Output is a JSON-compatible report;
the caller chooses persistence. No output file is published on a failed call.

## Evidence and remaining boundaries

Reports are always `development_validation_only`, with `final_test_evaluated`
and `topology_claim_evaluated` false. Paired FlyWire-minus-rewired validation
differences may be positive, zero or negative. No confirmatory threshold is
applied: these same validation scores helped choose checkpoints. Complete
execution does not imply useful recognition or any biological-topology advantage.
Source hashes bind bytes, not honest inference, approved pretrained origin or
authentic authorship. Runtime descriptions/source hashes are not a container or
hardware attestation. Use a trusted process and private read-only input trees;
this is not a concurrent-filesystem or hostile-Python security boundary.

Full padded partitions remain materialized in RAM. The runner is not a streaming
large-corpus loader. The frozen real preflight's 20 epochs/40 updates requires a
compatible full-roster batch size; it must not be treated as an arbitrary default.
Likewise, graph widths must fit its existing parameter ceiling. This runner never
edits those protocol values to make a run succeed.

Unit tests use generated pose/graph files and actual optimizer execution. The
actual-library integration uses generated video, real PyAV and an untrained
local YOLO checkpoint, then the pinned preparation and four-arm entry twice.
Its T4/T5-named CSV is generated, not actual FAFB. The separate provenance job
continues to verify real v783 graph/control bytes. Neither fixture check runs a
real NTU recognition experiment or accesses final-test video observations.

Real asset approval/freeze, full runtime/execution configuration, bounded-memory
corpus policy and sealed confirmatory evaluation remain separate gates. Existing
real protocols, graph fingerprints, seeds, budgets and thresholds are unchanged.

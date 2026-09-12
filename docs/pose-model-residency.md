# One-model-at-a-time pose comparison

Tracked by #68. This is a resource-lifetime fix to the existing file-bound
`run_pose_comparison`, not a new trainer or a streaming corpus implementation.

## What was retained unnecessarily

The previous `_preflight_models` kept every initialized model in its returned
plan list. All four arms for all seeds, including completed models, remained
reachable until the whole comparison returned. Five seeds retained twenty
models and their learned parameters and dense adjacency buffers.

Preflight now returns frozen `_ArmPlan` records containing construction inputs,
verified graph declarations, training configuration, expected order hashes,
parameter count, topology fingerprint, adjacency hash and initial-state hash.
These records contain no model or tensor handles. Each temporary preflight model
is released before constructing the next one.

**Complete preflight still precedes every optimizer call.** This is not a lazy
validation generator: a failure in the last seed's last model prevents all
training. Once all plans pass, a scoped helper reconstructs one model under the
same CPU float32 seeded condition, repeats model/partition/configuration checks,
and compares its initialized state and adjacency with its preflight record.
Drift is an error before that arm's training, not a reset-based repair.

The helper runs the existing trainer and all previous update/order/input/state/
adjacency/metric checks, then returns only the scalar/JSON-compatible arm record.
The outer loop never retains a trained-run or model handle. Completed models
can be reclaimed before the next constructor. Production does not force garbage
collection or flush allocator caches. External hooks that retain objects and
retained exception tracebacks are outside this lifetime guarantee.

## Equivalence and tests

`tests/test_pose_model_residency.py` uses weak references to observe reachable
initialized models and their parameter/buffer tensors with one, two and five
seeds. The test invokes GC to exclude unreachable cycles; it does not measure
process RSS, optimizer temporaries or native allocator retention. On the old
implementation, CI #104 observed peaks of four, eight and twenty live models.
The five-seed generated fixture retained 154,720 parameter/buffer bytes at the
last constructor, versus 8,216 bytes for its largest single model.

A separate reference constructs all models in the previous eager order and
trains them with the actual unchanged trainer. It compares every arm output
exactly: learned-state and adjacency hashes, selected epochs, metrics, parameter
counts, topology fingerprints, optimizer counts and sample-order hashes.
Caller RNG and deterministic-state regression tests remain in the existing suite.
The complete report's execution-source hash necessarily changes with source code;
this must not be hidden or treated as byte-identical report provenance.

Run `python -m pytest tests/test_pose_model_residency.py -q -s`, the complete unit
suite, and the existing actual-library integration. All inputs in these tests
are generated fixtures, not real NTU recognition evidence.

## Remaining memory work

Only model residency is bounded independently of seed count. Immutable arm
records, random/rewired graph edge tuples and output reports still grow with the
number of seeds. Full padded development partitions and the trainer's defensive
partition snapshots still occupy RAM. Training activations, optimizer state and
checkpoint snapshots for the active arm are also unchanged. This fix does not
make total memory constant or make large real-corpus batches fit automatically.

The next corpus change needs a separately specified disk-backed sequence/index
boundary and per-minibatch materialization, preserving byte pins, complete
rosters, actual frames, masks and the existing sample order. With the real frozen
20-epoch/40-update protocol, complete-roster execution still means exactly two
optimizer batches per epoch; reducing the physical batch by increasing optimizer
updates would change the experiment. Microbatch gradient accumulation, if later
adopted, needs its own explicit contract and numerical-equivalence evidence.
Neither streaming nor accumulation is implemented by this change. No real
protocol, control, width, seed, budget, threshold or final-test access is changed.

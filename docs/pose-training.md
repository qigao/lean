# Padding-aware development training and validation

Tracked by #68. `python/yolo_flywire/pose_training.py` connects the explicit
[padding-aware models](padded-models.md) to Adam training and validation-only
checkpoint selection. The fixed-length synthetic `train_model` / `evaluate`
remain separate; the new path never delegates to them or discards lengths/masks.
This is an in-memory development primitive, not the real four-arm experiment
runner or authorization to open a final test.

## Inputs and ownership

`PosePartition` contains `observations: PoseBatch`, aligned CPU int64 `targets`,
an explicitly ordered tuple of `classes`, aligned `sample_ids` and positive
integer `subjects`, and a `split` role. The class tuple determines the meaning of
each output column and must match the classifier's output width. It is never
inferred from a minibatch's observed labels. The API accepts the existing GRU
and graph recurrent classifier implementations with the CPU float32 / 121-channel
batch contract; there is no device migration, coercion or alternate model path.

Before resetting learned parameters or doing recurrent work, training validates
both complete partitions, the existing batch/model runtime contract, target
shape/type/range, metadata lengths/uniqueness, and the full training config.
Training requires the `train` role, validation requires `validation`, their
ordered class tuples must agree, and their sample-ID and subject sets must be
disjoint. Duplicate subjects *within* a partition are expected; duplicate sample
IDs are not. Final-test and unknown roles are rejected. The evaluator accepts
only the validation role.

**Declared metadata is not authenticated provenance.** A caller can fabricate
consistent IDs, labels, subjects and roles. A frozen dataclass also does not make
its tensor storage immutable. Construct partitions from `load_development_bundle`
with independently supplied byte pins and the common frozen encoder. The future
real runner must bind that verified source, roster, class mapping and complete
configuration; this module does not do so and does not fill real protocol hashes.
It cannot detect a final-test tensor deceptively relabeled as training data.
No file access, video decoding or YOLO inference occurs here.

Training makes detached copies of observations and targets before updates, and
minibatches use one common index selection for all observation fields and labels.
Caller tensor storage/gradients are not changed. Callers must not mutate inputs,
model state, global RNG state or deterministic settings concurrently with calls.
The supplied model is deliberately reset and trained in place; parameter objects
and the graph's nonpersistent adjacency are not replaced. Compute failures raise
and return no run; preceding model resets/updates are **not rolled back**.

## Budget, randomization and checkpoint selection

The existing `TrainConfig` is required. Epoch count, batch size and parameter
ceiling must be positive integers; the seed must be an integer in `[0, 2**32)`;
the learning rate must be finite and positive. Booleans are not valid numeric
configuration values. An over-ceiling model is rejected before its reset.

Adam uses the configured learning rate. Each epoch visits every training sample
exactly once in a CPU-generator-seeded permutation; the last partial batch is
retained. Only unused *padding* beyond a minibatch's longest observed clip is
trimmed. No observation is truncated, resampled or dropped, including actual
missing-person frames. Model calls receive the complete `PoseBatch` through
`forward_padded`. Nonfinite logits, losses, gradients or updated parameters are
errors, not a skipped batch, repaired value or zero-score result.

All configured epochs are executed, with one validation pass after each epoch.
The highest validation macro-F1 selects a deep-copied checkpoint; strict `>`
comparison retains the earliest epoch on ties. Validation has no optimizer step
and its labels cannot change the training trajectory, although they can change
the selected checkpoint. There is no early stopping or access to a test partition.
The selected model is returned in evaluation mode with parameter gradients cleared.

`PaddedTrainedRun` records the model, learned-state hash, best validation macro-F1,
one-based selected epoch, actual optimizer-step count, and one order hash per
epoch. The latter hashes UTF-8 compact JSON of that epoch's ordered sample IDs,
with `json.dumps(..., separators=(",", ":"))` and SHA-256. With `N` samples,
`optimizer_steps = epochs * ceil(N / batch_size)`. The hash from the existing
state serializer covers learned `state_dict` contents, **not** the nonpersistent
graph adjacency or complete data/runtime provenance. A graph fingerprint and
independently frozen experiment manifest remain necessary.

Training temporarily seeds Python, NumPy and PyTorch CPU randomness and enables
strict deterministic algorithms; their prior RNG states and deterministic/warn
flags are restored even on failure. This does not establish cross-version,
cross-hardware, GPU, mixed-precision or concurrent-thread reproducibility.

## Validation metrics and mode handling

Validation aggregates a confusion matrix across the entire partition, not the
mean of per-minibatch F1 scores. Macro-F1 and mean class recall (the returned
`balanced_accuracy`) both average over **every declared class**, with zero
contribution when the corresponding denominator is zero. Thus a class absent
from a fixture's validation labels still appears in the fixed denominator. This
is the explicit project convention, not an inference of the vocabulary from
validation labels. The real study's class coverage is an upstream protocol gate.

Evaluation runs under `no_grad` with every module in evaluation mode, then restores
each individual module's prior training flag in `finally`, including on errors.
It does not reset weights, clear caller parameter gradients or update an optimizer.
The returned object is the existing `MetricBundle`.

## Usage after upstream verification

The caller supplies the frozen class tuple, aligned targets and separate
partitions; labels/IDs/subjects never become observation columns:

```python
from yolo_flywire.models import GRUClassifier
from yolo_flywire.pose_training import PosePartition, train_padded_model, evaluate_padded
from yolo_flywire.train import TrainConfig

# train_partition and validation_partition are PosePartition objects assembled
# from independently verified bundles and a single frozen class-to-target map.
model = GRUClassifier(121, hidden_dim=32, num_classes=len(train_partition.classes))
run = train_padded_model(model, train_partition, validation_partition, config)
validation_metrics = evaluate_padded(run.model, validation_partition, batch_size=config.batch_size)
print(run.best_epoch, run.optimizer_steps, run.best_validation_macro_f1)
```

`config` is an independently fixed `TrainConfig`, not a newly selected experiment
budget. This primitive materializes padded partitions and copies them for training;
its memory scales with the entire supplied partition and longest clip, plus
minibatch/autograd/checkpoint storage. It is **not** a bounded-memory corpus loader.
A streaming/large-corpus execution policy must be separately specified and frozen.

## Verification and remaining gate

```bash
python -m pytest tests/test_pose_training.py -q -W error
python -m pytest tests -q
python -m pytest integration/test_pose_backend.py -q -s
```

Tests use generated feature fixtures and actual optimizer updates for both models.
They check repeatability, added-padding invariance of selected weights, identical
row/label/length selection, exact epoch/update budgets, fixed-vocabulary metrics,
earliest-tie checkpoint restoration, mode/RNG preservation and rejection before
reset. Generated-video integration extracts and verifies two independent pose
bundles, encodes and collates separate development splits, then repeats short
training runs and compares learned-state/order hashes. The YOLO checkpoint is
untrained and the graph is a tiny fixture; this is plumbing evidence only.

Next: bind verified real development inputs and source/configuration hashes into
a four-arm runner with common budgets and graph/control provenance. Authorized
NTU media, independently approved pretrained weights, the complete freeze and
sealed final-test execution remain outstanding. No real recognition accuracy or
FlyWire topology advantage is established by these tests.

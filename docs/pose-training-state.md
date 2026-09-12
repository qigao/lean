# Development model-state preflight

The [padded development trainer](pose-training.md) validates model state before
reset or inference on every call. Finite logits do not establish finite model
state: sigmoid/tanh saturation can turn infinite recurrent biases into finite
outputs and plausible validation metrics. Every parameter and buffer (including
nonpersistent graph adjacency) must therefore be dense and finite. Mutating a
model after a successful call does not bypass the next validation. Corrupted
state is rejected, never repaired by the deterministic reset.

Training also requires every learned parameter to be trainable. Partially
freezing a classifier would silently change the experimental training condition,
so it is rejected before reset rather than implicitly unfreezing parameters or
counting Adam calls as full-model training. This restriction is training-only:
validation accepts finite frozen models and retains their flags.

`tests/test_pose_training_state.py` exercises both classifiers, infinite recurrent
biases, NaN readout/adjacency, rejection before mutation/computation, frozen-model
validation and restoration of caller RNG/determinism state after compute errors.
The existing training alignment tests count completed Adam steps, not functional
loss-wrapper entries, because CPU device dispatch can re-enter the latter.

These checks neither authenticate dataset/graph provenance nor change recurrence,
Adam settings, checkpoint selection, sample order, budgets or final-test policy.
Invalid runs produce errors rather than research evidence. Valid generated-fixture
runs remain engineering checks, not NTU recognition or FlyWire advantage results.

# Verified pose development-input binding

Tracked by #68. `pose_development.py` composes the existing pinned file reader,
timed COCO17 encoder, collator and `PosePartition`. It is a development-input
boundary, not another trainer, a four-arm runner, or a final-test evaluator.

## Entry and class mapping

`load_pose_development` requires a bundle path, RGB root/inventory, an explicit
`ExtractionSpec`, an explicit `PoseFeatureSpec`, an independently supplied
manifest SHA-256, an independently supplied encoder hash, and an ordered tuple
of class names. There is no object-based shortcut accepting a constructed
`VerifiedPoseBundle`, no inferred vocabulary, sample filter or split override.

The encoder pin is checked before file reading. The existing bundle reader then
verifies the inventory and actual RGB bytes, full manifest, observation and timing
sidecars, extraction identity and complete development roster before any feature
encoding. The encoder receives only observations, PTS, time bases and detector
confidences. Each sample retains every actual frame, including missing detections.

Both development splits must contain exactly the declared class set. The supplied
tuple defines target indices, without sorting or normalization. Reordering it is
a different binding and changes target indices, not observations. Sample order
within each split is the order returned by the verified reader. Labels, subjects,
IDs and split roles remain outside the 121 feature columns.

## Result and integrity record

`PreparedPoseDevelopment` contains `train` and `validation` partitions plus
canonical `binding_json` and its SHA-256 `binding_sha256`. `descriptor()` returns
a fresh decoded record, so editing that dictionary cannot edit the stored record.

The record includes all source bundle identity fields and output hashes, the
explicit encoder hash/descriptor, ordered class names, the preparation source
file hashes, Python/NumPy/PyTorch and platform descriptions, and each partition's
ordered IDs, subjects, class map, split role and tensor identities. Tensor hashes
include shape and fixed little-endian dtype metadata followed by contiguous
logical values for features, lengths, time masks and targets. They do not depend
on tensor storage addresses or filesystem location.

`verify()` recomputes the record digest and the current partition identities.
It detects changes to features, labels, lengths, masks, row metadata or the record
against the saved digest. Call it before and after handing a prepared dataset to
the development trainer. It does not re-read the input corpus: it checks the
materialized snapshot. The loader separately rejects changes in encoder/preparation-code
identity observed during preparation.

**Integrity is not authority.** Python dataclasses and hashes are not an access
control mechanism. A caller able to fabricate objects and recompute all records
can fabricate an internally consistent result. Keep the binding digest separately
when freezing an experiment; the future runner must compare against that external
pin, enforce source loading and holdout rules, and bind its models and budgets.
Neither the record nor file hashes attest that poses were generated honestly,
that weights are pretrained/approved, or that a model recognizes real actions.

## Use with the existing trainer

```python
from yolo_flywire.pose_development import load_pose_development
from yolo_flywire.pose_training import train_padded_model

prepared = load_pose_development(
    bundle_path, root=rgb_root, inventory=rgb_inventory,
    extraction_spec=extraction_spec, feature_spec=feature_spec,
    classes=frozen_classes,
    expected_manifest_sha256=frozen_manifest_sha256,
    expected_encoder_hash=frozen_encoder_hash,
)
prepared.verify()
run = train_padded_model(model, prepared.train, prepared.validation, train_config)
prepared.verify()
```

The configuration, classes, model, paths and pins above are caller-supplied frozen
inputs, not defaults selected from validation outcomes. Do not compute an
"expected" pin from an untrusted current file in a real confirmatory run.
The existing trainer still uses validation-only earliest-best checkpoint selection
and the complete configured epoch/update budget.

## Scope and verification

This implementation materializes the verified bundle and full padded partitions.
Memory grows with corpus size and each split's longest sequence; hashing may make
contiguous copies of noncontiguous tensors. This is not a bounded-memory streaming
loader. No observation truncation or subsampling is used to hide that limit.

Run `python -m pytest tests/test_pose_development.py -q` and the full CI suite.
Tests compose the actual file verifier, encoder, collator and Adam trainer over
generated fixtures; decoder/predictor seams are test doubles, not recognition
evidence. The existing actual-library video integration remains a separate CI
check. No actual NTU media or pretrained checkpoint is acquired by these tests.
No workflow, real experiment protocol/hash, graph/control, budget, seed, success
threshold or final-test authorization is changed.

Next: consume the externally pinned prepared identity in a common-budget four-arm
development runner. Large-corpus memory policy, authorized real assets, the full
runtime/model freeze and sealed final-test execution remain separate gates.

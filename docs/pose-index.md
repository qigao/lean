# Verified on-demand pose sidecar reads

Tracked by #68. This checkpoint adds a disk-backed observation read boundary to
`pose_index.py`. It does not replace the tensor preparation, trainer, four-arm
runner, or final-test gate. Existing extraction files and schemas are unchanged.

## Complete verification first

`index_development_bundle` takes exactly the bundle, independent manifest pin,
RGB root/inventory and extraction specification needed at the existing file
boundary. It uses the existing RGB inventory verifier, manifest validator,
strict JSON parser and geometry/timing frame validator. Every development row is
parsed and checked before the index is returned, including the last sample.
Whole-file sidecar digests must match the pinned manifest. Extra rows, gaps,
misjoined times, changed source media, unknown fields and invalid missing-person
counts fail; indexing is not a hash-only shortcut or an eager-loader fallback.

The returned frozen `IndexedPoseBundle.samples` contains scalar
`PoseSampleIndex` records: identity, split, label, subject, dimensions, real frame
count, missing-person count and two `ByteSpan` records. Spans use half-open byte
ranges in the original geometry/timing JSONL files and SHA-256 over the exact
range. They cover those files contiguously in the verified roster order.
No frame/sequence/tensor objects, open file handles or materialized samples are
retained by the index. No new observation files or persisted index cache are
written.

The canonical `descriptor()` binds all sample metadata, spans, source hashes and
development-only scope. `index_sha256` identifies that descriptor independently
of the input directory's location. A relocated copy must be freshly indexed and
verified, not trusted by replacing the directory field. Runtime filesystem stat
signatures are a separate change detector, not part of the portable identity.

## Requested sequences only

`read_samples(split="train", indices=(9, 0, 4))` uses explicit **split-local**
indices and preserves their order. Only `train` and `validation` are accepted;
empty, repeated, negative, out-of-range, boolean and coerced indices are errors.
The caller, not this reader, controls epoch order and optimizer batch membership.
There is no shuffle, sorting, truncation, resampling, frame interpolation or
selection from final-test observations.

Before reading, the saved index descriptor and current bundle file state are
checked. Each requested span is sought directly and read within its recorded
end boundary. Its bytes, row joins, frame count and missing-person count are
checked again. Only then can that sample enter the internal result tuple. A
failure in any requested sample or the final file-state check prevents returning
the entire batch; no partially verified iterator is exposed. File handles close
on success or error. A changed span is an error, not automatic reindexing.

Returned `TimedPoseSample` values have the same immutable sequence, original PTS,
rational time bases, detector/joint confidence and actual missing frames as the
existing eager reader. Labels and sample identities remain metadata, never
encoder inputs.

```python
from yolo_flywire.pose_index import index_development_bundle
from yolo_flywire.pose_features import encode_timed_pose
from yolo_flywire.pose_batches import collate_pose_features

index = index_development_bundle(
    bundle_path, root=rgb_root, inventory=verified_inventory,
    spec=extraction_spec, expected_manifest_sha256=frozen_manifest_sha256,
)
# Independently retain/check index.index_sha256 at the experiment boundary.
samples = index.read_samples(split="train", indices=(9, 0, 4))
features = tuple(encode_timed_pose(
    sample.sequence, pts=sample.pts, time_bases=sample.time_bases,
    detector_confidences=sample.detector_confidences, spec=feature_spec,
) for sample in samples)
batch = collate_pose_features(features)
```

## Memory, integrity and evidence limits

Index construction retains O(number of samples) roster/index metadata and only
row-sized observation working state, instead of all corpus frames. Manifest and
RGB inventory metadata still reside in memory; an arbitrarily large malformed
JSON row is not subject to a new hard byte limit. Reading retains the requested
samples, so asking for the entire split can still materialize it. Feature
encoding and padding require memory for the chosen batch's sequences and longest
clip. Index integrity checks currently traverse O(number of samples) metadata
per call; no bounded total RSS or corpus throughput claim is made.

Read instrumentation measures bytes returned by Python file reads and frames
parsed, not operating-system read-ahead, page cache or storage-device traffic.
Frame weak-reference tests demonstrate lack of retained corpus frame objects;
they are not process-memory benchmarks.

`verify()` checks the saved index hash and bundle stat signatures. It does not
rehash every sidecar or rescan the RGB corpus on each call. Requested span hashes
are checked on every read. Source media were checked when the index was built;
use a trusted process and private read-only trees. This is not an atomic snapshot,
hostile-filesystem defense, cryptographic signature or access token. Deliberately
fabricating both an index and its saved digest is not prevented. The independent
source/experiment pins and honest inference remain necessary.

Actual-library integration compares indexed/encoded generated-video minibatches
with existing eager prepared tensors exactly, including lengths and padding
masks. It still runs the unchanged eager four-arm path separately. These are
engineering checks using generated inputs and an untrained YOLO model, not real
NTU recognition or evidence of a FlyWire topology advantage.

Next: bind per-batch features/labels to this verified index and adapt development
training without whole-partition tensors or defensive corpus copies. The real
20-epoch/40-update budget remains unchanged; physical microbatch accumulation
requires its own explicit contract and numerical-equivalence tests. Neither
streaming training nor accumulation is implemented here.

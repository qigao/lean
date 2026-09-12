# File-bound indexed development batches

Tracked by #68. This is the explicit index-to-feature/target binding boundary;
existing eager training and the four-arm runner are not switched by this change.
It is not a second encoder, a compatibility fallback or a persisted cache format.

## Preparation and independent pins

`load_indexed_pose_development` takes the same explicit file, inventory,
extraction-specification, feature-specification, ordered class vocabulary,
manifest SHA-256 and encoder hash inputs as eager development preparation.
Passing a constructed index instead of a file path is rejected.

The existing indexer verifies the complete development roster, original RGB
inventory and all geometry/timing rows and sidecar hashes **before encoding**.
Then preparation reads and encodes each complete clip separately, performs the
existing strict float64-to-float32 conversion, and saves its unpadded feature
hash with its label index and row metadata. It exposes no prepared source until
all clips pass, including the last validation clip. No frames are truncated,
resampled, filtered or replaced. A finite float64 value that cannot survive the
float32 conversion still fails here rather than halfway through optimization.

The binding descriptor contains the full index identity, ordered classes,
encoder identity and policy, preparation-source/runtime identity, and complete
ordered per-clip records. The hash of each `[frames,121]` float32 matrix uses the
existing canonical shape/dtype plus little-endian logical tensor-value rule.
The index identity already commits to the source metadata and exact raw spans.
No source path, inode or file timestamp participates in the content binding.

Persist the resulting binding digest independently as part of a reviewed
experiment configuration. Methods require that external digest explicitly; they
do not silently take the object's own digest as the caller's approval. For a
generated smoke fixture, retaining the result in a separate variable is useful
for integrity testing, but is not a real-data protocol freeze.

```python
from yolo_flywire.pose_indexed_development import load_indexed_pose_development

source = load_indexed_pose_development(
    bundle_path, root=rgb_root, inventory=inventory,
    extraction_spec=extraction_spec, feature_spec=feature_spec, classes=classes,
    expected_manifest_sha256=manifest_pin, expected_encoder_hash=encoder_pin,
)
# binding_pin must be supplied from the independently reviewed experiment record.
source.verify(expected_binding_sha256=binding_pin)
part = source.read_partition(
    split="train", indices=(9, 0, 4), expected_binding_sha256=binding_pin,
)
source.verify_partition(
    part, split="train", indices=(9, 0, 4), expected_binding_sha256=binding_pin,
)
```

## Requested batches

Selectors must be nonempty tuples of unique nonnegative split-local integer
indices for `train` or `validation`, in the required order. Invalid selectors
are rejected before opening any file. Requests do not choose their own class
vocabulary or encoder. A request need not contain every class; the full source
must cover exactly the declared vocabulary in each development split.

Only the requested raw spans are reread and encoded. `read_partition` returns
the existing `PosePartition`, with labels outside `PoseBatch` observations.
It verifies feature hashes, targets, row identities/subjects, true lengths,
time masks and zero trailing padding, and ending source identity before return.
Actual all-missing frames are true time steps, not padding. The producer uses
canonical batch-local padding to the longest requested clip; its integrity
checker rejects shape/mask changes even though the model supports extra padding.

Returned tensor storage is mutable and fresh per request. `verify_partition`
detects changed contents or row interpretation against the original source and
external binding pin without rereading/encoding other observations. It must not
be treated as an authorization token. A trusted process and private read-only
input files remain required; hashes do not stop an actor replacing both code
and all independent pins.

## Resource and filesystem limits

Preparation retains one complete clip's observations/features/tensors at a
time, plus O(number of clips) scalar metadata. It does **not** bound the longest
clip, malicious JSON-row size, allocator caches or process RSS. The extra
feature-validation pass reads the sidecars again, but does not decode RGB or run
YOLO. A scoped helper releases clip objects before the next one is encoded.

Requested output memory is proportional to batch size times its longest clip,
with temporary raw and encoded requested clips. Callers can still request a
whole split. Integrity checking traverses the scalar roster on each call; this
is not an O(1) metadata lookup or a performance benchmark. Per-clip preparation
uses the existing internal span reader in a guarded complete pass, rather than
rehashing the full index metadata once per clip.

Index creation verifies original RGB bytes; later reads check the index's
recorded bundle state and rehash requested spans. These are observed stat
checks, **not an atomic filesystem snapshot**, and do not rehash unrequested
sidecars or the RGB inventory on every minibatch. Encoder/preparation source
and recorded runtime changes fail closed. This descriptive runtime record is
not a complete native-library/hardware reproducibility freeze. No automatic
reindex or pin repair.

## Tests and remaining execution gate

Unit tests cover eager equivalence, complete vocabulary/label ordering,
requested-only reads and encoding, no retained observation tensors, independent
pins, late numerical failures, changed files/identities, output mutation and
reader reordering. Actual PyAV/untrained-YOLO generated-video integration checks
variable-length batches against eager tensors and row metadata in both splits.
The existing four-arm smoke still runs separately, not through a streaming
optimizer. These generated fixtures establish no real recognition advantage.

Next, adapt development training to consume this source without full-partition
tensors and preserve the entire epoch roster, seeded order, validation-only
selection and exact optimizer budget. The real 20-epoch/40-update condition is
unchanged. Microbatch gradient accumulation is not implemented or assumed.

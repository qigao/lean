# Variable-length pose tensor batches

This is the observation-only collation step after the verified [bundle reader](pose-bundle.md) and [timed COCO17 encoder](pose-features.md), tracked by #68. It prepares tensors; it does not load a dataset, select training examples, fit a model, or establish recognition accuracy.

## Contract

`python/yolo_flywire/pose_batches.py` exposes `collate_pose_features(sequences)` and the returned `PoseBatch`. The input must be a nonempty **tuple** of nonempty NumPy `float64` arrays, each `[frames,121]`, produced under one independently frozen `PoseFeatureSpec` and encoder identity. Different clips may have different frame counts. Read-only, strided and Fortran-ordered arrays are accepted without modifying their storage; array subclasses and implicit list/dtype coercions are not.

For `B` clips and `T` equal to the longest input clip:

| Field | Shape and type | Meaning |
| --- | --- | --- |
| `features` | `[B,T,121]`, CPU `torch.float32` | Original rows converted once, followed only by trailing zero padding |
| `lengths` | `[B]`, CPU `torch.int64` | Original number of observed frames, not visible-person frames |
| `time_mask` | `[B,T]`, CPU `torch.bool` | `True` exactly when the frame existed in the input clip |

The input order is preserved. There is no sorting by length, cropping, resampling, dropping empty detections, interpolation, sample duplication or alternate encoder. Every output tensor has independent storage, is contiguous and has no gradient history. Device and dtype are explicit, regardless of PyTorch defaults. The frozen dataclass prevents field rebinding; **its tensors remain mutable**. Mutating one returned batch cannot affect source arrays or another batch, but downstream consumers are responsible for not corrupting their own tensors.

## Padding is not missing observation

Joint position/velocity masks remain the existing columns `85:102` and `102:119`. `time_mask` is separate and is never derived from them or from zero feature values.

For a one-frame missing-person clip padded to three frames:

```text
features[0,:,:] = three all-zero rows
time_mask[0,:] = [True, False, False]
lengths[0] = 1
```

The first row is an observation with no detected person; the other two rows never happened. For later all-missing frames, the original positive `dt` is also retained. Neither missing detections nor unusable reference geometry shorten a clip or bridge a velocity gap.

## Validation and float32 conversion

Before a batch is returned, each sequence must have the 121-column shape and the existing feature invariants: finite numbers, confidences in `[0,1]`, binary validity masks, zero values for masked geometry/velocity, no first-frame velocity, velocity masks matching both adjacent position masks, a zero first interval and positive later intervals. Zero person confidence requires all non-time channels to be zero. Visible joint positions cannot have zero joint confidence.

Masks are copied, **not recomputed from rounded float32 confidences**. In particular, a confidence just below a float64 threshold can round to the same float32 value as the threshold; its already-computed invalid mask must stay invalid.

Ordinary finite float32 rounding is expected and is not bitwise equality with float64. Overflow and any nonzero value that would become zero during conversion are errors, including an interval or confidence that would disappear. Representable subnormal values are accepted. No clipping, nominal-FPS repair, dtype fallback or silent sample rejection is provided. A malformed later clip rejects the whole call rather than producing a partial batch.

These checks do not recompute the encoder or authenticate an array. A 121-column fabricated matrix can satisfy structural checks. File provenance, the common encoder/configuration, sample roster, subject isolation and final-test access remain the caller's responsibility, enforced by the upstream verified reader and the future experiment runner. The collation API intentionally has no labels, IDs, paths, subjects or split parameters.

## Development wiring

Starting with a `VerifiedPoseBundle` returned by `load_development_bundle` using an independently supplied manifest pin:

```python
from yolo_flywire.pose_features import PoseFeatureSpec, encode_timed_pose
from yolo_flywire.pose_batches import collate_pose_features

feature_spec = PoseFeatureSpec(confidence_threshold=0.05, scale_epsilon=1e-6)
# This example takes at most eight training clips; it does not define the
# experiment's sampling policy. The future runner owns frozen batching/sampling.
selected = tuple(sample for sample in bundle.samples if sample.split == "train")[:8]
encoded = tuple(encode_timed_pose(
    sample.sequence, pts=sample.pts, time_bases=sample.time_bases,
    detector_confidences=sample.detector_confidences, spec=feature_spec,
) for sample in selected)
batch = collate_pose_features(encoded)
labels = tuple(sample.label for sample in selected)  # Never feature columns.
# Repeat separately for validation; preserve this exact sample order when
# applying the independently frozen class-to-target mapping.
```

The output allocation is proportional to `B * max(frames) * 121`, plus temporary converted input arrays. This is an in-memory single-batch utility, not a streaming loader or a bounded-memory corpus pipeline. The caller must select sensible batch sizes before calling it. The existing `pose_encoder_hash` still describes the float64 encoder only; a future real-run freeze must also bind this collation policy/source and the actual NumPy/PyTorch runtime. No real protocol hash is filled by this utility.

## Verification and next gate

```bash
python -m pytest tests/test_pose_batches.py -q -W error
python -m pytest tests -q
python -m pytest integration/test_pose_backend.py -q -s
```

Focused tests cover unsorted variable lengths, nonzero kinematic features, nonuniform `dt`, observed all-zero versus padded rows, mask retention around gaps and rounded confidence thresholds, numeric/schema rejection, independent storage and explicit CPU/dtype behavior. Actual-library integration uses generated 1/2/3-frame videos and an **untrained local YOLO checkpoint**. It extracts and reads twice, encodes, then collates train and validation separately, with network connections blocked and final-test fixture decoding forbidden. That verifies library wiring, not useful human detections or behavior recognition.

**Do not feed only `batch.features` into the old training/model path and declare variable-length support complete.** The existing models do not yet consume these lengths/masks. The next gate is padding-aware model ingestion, including invariance to extra trailing padding and preservation of observed missing frames, before implementing the real four-arm development runner. This slice does not change model equations, graph/control fingerprints, training/rewiring budgets, seeds, success thresholds or final-test authorization. Authorized NTU bytes, independently approved pretrained-weight provenance and the complete runtime/data freeze remain outstanding.

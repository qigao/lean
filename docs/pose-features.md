# Timed COCO17 pose features

This is the development encoding seam after the verified [pose-bundle reader](pose-bundle.md), tracked by #68. It is not a real NTU recognition result, an approved final experiment freeze, or evidence of a FlyWire advantage. The synthetic `features.encode_sequence` remains unchanged: its synthetic point ordering and frame differences are not the COCO17/seconds contract below.

## Observation-only API

File callers must first use `load_development_bundle` to verify the independently pinned manifest, actual input inventory, extraction specification, sidecar hashes, subjects and train/validation membership. Do not obtain the expected manifest digest from an untrusted bundle and treat that as independent verification. The pure encoder validates geometry and timing, not file provenance or split membership.

```python
from yolo_flywire.pose_bundle import load_development_bundle
from yolo_flywire.pose_features import (
    PoseFeatureSpec, encode_timed_pose, pose_encoder_hash,
)

# bundle_path, rgb_root, inventory, extraction_spec and manifest_pin come from
# the independently approved local development inputs described in pose-bundle.md.
bundle = load_development_bundle(
    bundle_path, root=rgb_root, inventory=inventory, spec=extraction_spec,
    expected_manifest_sha256=manifest_pin,
)
feature_spec = PoseFeatureSpec(confidence_threshold=0.05, scale_epsilon=1e-6)
encoder_identity = pose_encoder_hash(feature_spec)
for sample in bundle.samples:
    features = encode_timed_pose(
        sample.sequence,
        pts=sample.pts,
        time_bases=sample.time_bases,
        detector_confidences=sample.detector_confidences,
        spec=feature_spec,
    )
    assert features.shape == (len(sample.sequence.frames), 121)
    # Keep labels/IDs/subjects/splits outside these observation features.
```

The API requires immutable `PointSequence`/`KeypointFrame` observations, exactly 17 ordered `(x_pixels, y_pixels, confidence)` tuples per frame, integer PTS tuples, positive `Fraction` time-base tuples, detector-confidence tuples and an explicit `PoseFeatureSpec`. Strings, booleans, non-finite numbers, malformed schemas, mismatched lengths and invalid timing are errors, not coercion or fallback opportunities. A zero detector confidence requires the extractor's all-zero missing-person observation.

## Geometry, timing and masks

COCO17 zero-based shoulders are 5/6 and hips are 11/12; the nose at 0 is not a torso anchor. Reference ordering: [Ultralytics COCO pose documentation](https://docs.ultralytics.com/datasets/pose/coco/).

For frame `i`, let `p[i,j]` be the two pixel coordinates of joint `j`, `c[i,j]` its confidence, `d[i]` detector confidence, `tau` the joint threshold and `epsilon` the minimum reference span in pixels. The candidate policy is:

```text
anchor[i] = (p[i,11] + p[i,12]) / 2
span[i]   = ||p[i,5] - p[i,6]||_2
reference_valid[i] = d[i] > 0
                     and all(c[i,k] >= tau for k in (5,6,11,12))
                     and span[i] > epsilon
position_mask[i,j] = reference_valid[i] and c[i,j] >= tau
position[i,j] = (p[i,j] - anchor[i]) / span[i]  when valid, otherwise 0

time[i] = pts[i] * time_base[i]               (exact Fraction)
dt[0]   = 0
dt[i]   = float(time[i] - time[i-1])          for i > 0
velocity_mask[0,j] = false
velocity_mask[i,j] = position_mask[i,j] and position_mask[i-1,j]
velocity[i,j] = (position[i,j] - position[i-1,j]) / dt[i]
                when velocity_mask[i,j], otherwise 0
```

The confidence threshold is inclusive and lies in `(0,1]`; `epsilon` is strictly positive. Missing/low-confidence reference points or a collapsed shoulder span invalidate the entire frame's geometry, not the raw confidence or timing. There is no fallback to the nose, an eye pair, one hip, an earlier frame or a unit scale. Invalid joint coordinates never enter normalization arithmetic, but even masked inputs must be finite numeric values. Intermediate geometry/velocity overflow or a non-representable positive interval is rejected rather than clipped.

Timestamps must strictly increase. Rational subtraction occurs before float conversion, so a large common PTS offset does not erase a small interval. Nonuniform frame intervals and changing positive rational time bases are supported; no nominal FPS is assumed. The first frame has zero velocity, zero velocity masks and zero initial interval, not a fabricated duration. Every input frame remains present. In particular, a missing joint/person at frame `i` prevents velocity at both `i` and `i+1`; velocity resumes only when two adjacent observations are valid. There is no interpolation, sorting, resampling or missing-frame bridging.

Positions are body-relative coordinates normalized separately at each frame; their velocity is the derivative of that representation per second, **not physical/world velocity or raw pixel velocity divided by a single scale**. Global image translation and positive uniform scale are removed in the nonsingular regime (subject to floating-point precision and the pixel epsilon boundary). There is no rotation normalization. This policy discards global body translation/scale motion, and shoulder foreshortening or detector jitter can still affect the representation. Recognition usefulness must be evaluated, not inferred from these invariants.

## Fixed layout

The result is a fresh C-contiguous finite `float64` array `[frames, 121]`. Each xy block is COCO17 joint-major (`x0,y0,x1,y1,...`). Half-open column ranges are:

| Columns | Contents |
| --- | --- |
| `0:34` | normalized joint xy |
| `34:68` | normalized-coordinate velocity xy per second |
| `68:85` | raw joint confidences |
| `85:102` | joint position masks as 0/1 |
| `102:119` | joint velocity masks as 0/1 |
| `119:120` | raw detector confidence |
| `120:121` | actual adjacent interval in seconds; first 0 |

Raw confidence is retained even when reference geometry is unusable. It is a detector score, **not a calibrated probability or a quantified uncertainty interval**. An invalid zero position/velocity is distinguished from a measured zero by its separate mask. All-missing frames still retain their timing and are not sequence padding.

`PoseFeatureSpec.descriptor()` returns a fresh description of the ordering, dimensions, policies and parameter values. `pose_encoder_hash(spec)` hashes canonical JSON containing that descriptor and SHA-256 of the actual local `pose_features.py` and `schema.py` source bytes. It binds this candidate encoder implementation and parameters, not the full operating system, NumPy wheel or runtime. Producer provenance remains the bundle reader's responsibility. No real protocol `encoder_hash` is filled automatically.

## Verification and remaining work

The focused analytic tests require known nonzero geometry/velocity, exact rational intervals, time dilation, translation/scale invariance, reference and joint masking, no gap bridging, strict numeric rejection, fresh outputs and descriptor/source hash binding. The existing actual-library integration also passes generated-video observations through extraction, verified timed reading and this encoder twice, with network connections blocked and no additional decoding during encoding. Its local YOLO checkpoint is deliberately untrained and does not establish useful detections. Analytic fixtures establish the nonzero kinematic calculations; the real-library smoke establishes the wiring and repeatability on generated inputs.

```bash
python -m pytest tests/test_pose_features.py -q -W error
python -m pytest tests -q
python -m pytest integration/test_pose_backend.py -q -s
```

Per-sample encoding materializes features in memory; this slice does not implement batching, padding masks, float32 tensor conversion, model ingestion, classifier training or final-test evaluation. The next development slice must preserve these validity and timing semantics across variable-length batches and distinguish padded steps from observed all-missing frames. Authorized NTU RGB bytes, independently approved pretrained-weight provenance, a complete runtime/encoder freeze, and an executed real four-arm runner remain separate gates. #68 stays open.

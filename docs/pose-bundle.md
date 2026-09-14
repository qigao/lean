# Reading verified timed pose bundles

The development extractor writes three files: `observations.jsonl`, `timing.jsonl`, and the final commit marker `manifest.json`. Use `yolo_flywire.pose_bundle.load_development_bundle` to consume that complete bundle. The geometry-only `load_pose_jsonl` is not a timed real-data reader and is not a fallback.

## Inputs and trust boundary

Provide the bundle directory, the original read-only NTU root and inventory, the independently frozen `ExtractionSpec`, and the SHA-256 of `manifest.json` recorded at trusted publication time. A digest copied from the same untrusted bundle does not authenticate it. The reader requires the pin explicitly; it never substitutes an automatically accepted manifest hash.

The reader checks the marker against that pin, rebuilds/reverifies the NTU inventory from actual local bytes, and binds the extraction report to that inventory's content, split, sample IDs, labels and whole-subject partitions. It checks the report's extraction spec, producer schema, versions and code identity. It then streams the two JSONL sidecars together, hashing the exact bytes it parses, and exposes no result until both files are completely consumed and validated.

The current extractor-code identity covers `pose_extract.py`, `pose_backend.py` and `ntu_io.py`. A different producer implementation is rejected rather than accepted by a compatibility path. Recorded inference package versions must agree with the supplied extraction spec; reading does not require installing or running those inference packages. This identity check does not attest that a particular computation actually ran. It also does not pin every wheel, OS or FFmpeg build.

Only the three defined regular files are accepted. Missing markers, extra files, symlinked paths, unknown fields, duplicate JSON keys, non-finite numbers, changed checksums and structural inconsistencies are errors. Manifest paths never determine which files are opened.

## Python API

```python
import json
from pathlib import Path
from yolo_flywire.pose_extract import ExtractionSpec
from yolo_flywire.pose_bundle import load_development_bundle

inventory = json.loads(Path("/private-run/ntu-inventory.json").read_text())
spec = ExtractionSpec(**json.loads(Path("/private-run/extraction-spec.json").read_text()))
# Read the trusted digest from your separately recorded run/provenance record.
manifest_pin = Path("/private-run/trusted-manifest.sha256").read_text().strip()
bundle = load_development_bundle(
    "/private-run/poses", root="/datasets/ntu120", inventory=inventory,
    spec=spec, expected_manifest_sha256=manifest_pin,
)
for sample in bundle.samples:
    assert sample.split in ("train", "validation")
    for frame, timestamp, missing in zip(
        sample.sequence.frames, sample.timestamps, sample.missing_person
    ):
        # timestamp is fractions.Fraction; frame.points retains (x,y,confidence).
        pass
```

Every `TimedPoseSample` retains immutable tuples containing the original integer PTS, rational time bases, exact derived timestamps, person confidences and COCO17 geometry. Sample IDs, labels, subject, partition and image dimensions are explicit metadata, not classifier input features. Individual zero-confidence joints are preserved even when their coordinates are nonzero. A missing person must have all 17 points zeroed, as specified by the producer contract; the missing-person count must match the manifest.

Row order is part of the producer contract: the complete development roster in inventory order, and contiguous zero-based frame indices within each sample. Geometry and timing must agree exactly on every `(sample_id, frame_index)`. The reader rejects rather than sorts, interpolates or renumbers malformed observations. Rational timestamps must strictly increase within each sample; negative initial PTS and nonuniform frame intervals are preserved, not repaired.

## Read-only CLI

```bash
PYTHONPATH=python python -m yolo_flywire.pose_bundle \
  --bundle /private-run/poses \
  --root /datasets/ntu120 \
  --inventory /private-run/ntu-inventory.json \
  --config /private-run/extraction-spec.json \
  --manifest-sha256 "$TRUSTED_MANIFEST_SHA256"
```

Success prints a JSON summary with `samples`, `frames`, provenance hashes and `evidence_scope=development_bundle_verified_only`. Rejection exits nonzero, prints no successful summary and writes no files. There is no final-test flag or output directory option.

## Limits and next stage

No decoder, detector, tracker, depth estimator, model fitting or evaluation is invoked by reading. Input-inventory verification integrity-hashes final-test files, but the bundle contains only train/validation observations. This is an application-level boundary, not an operating-system sandbox. Filesystems must remain private and read-only throughout verification; stat and hash checks are not an atomic adversarial snapshot.

Sidecar parsing is streaming, but the returned immutable dataset is materialized in memory. Time-aware feature extraction, chunking/batching, normalization, uncertainty-threshold selection and a verified four-arm real runner are separate work. Do not feed these observations to the existing frame-difference encoder while claiming that seconds-based dynamics or arbitrary video timing were already handled.

Tests use generated opaque input files and extraction doubles; the existing integration additionally uses real PyAV and a locally constructed untrained YOLO26 checkpoint. Neither supplies official NTU data or independently approved pretrained-weight provenance. A verified bundle proves the tested integrity/format contract, not useful detections, recognition accuracy or a FlyWire topology advantage. Real protocol hashes stay null until actual authorized inputs are frozen.

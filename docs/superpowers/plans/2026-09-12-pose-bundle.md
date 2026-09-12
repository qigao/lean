# Verified timed pose-bundle reader implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume the existing development extraction output as verified, immutable, timed observations rather than silently dropping its timing sidecar.

**Architecture:** Add one reader between `pose_extract.extract_development` and future temporal feature preparation. Bind a caller-pinned manifest SHA-256 to actual sidecar bytes, an independently supplied extraction spec, the current extractor code identity, and a freshly verified NTU inventory. Join rows by exact sample/frame identity in producer order and return immutable geometry, confidences and rational timing. This is a continuation of #68's approved real input/extraction path, not a new model or benchmark.

**Tech Stack:** Python 3.11 standard library plus existing extraction/schema modules; pytest; existing GitHub Actions unit and real-library integration jobs.

**Spec:** `docs/phase2-real-data-freeze.md`, `docs/pose-extraction.md`, and Issue #68's extraction completion/next-gate contract.

## Global constraints

- Development partitions only: train and validation. No final-test override, decoder call, inference, model training or evaluation in the reader.
- Preserve the existing COCO17 producer format. No geometry-only fallback, reordering, interpolation, frame renumbering or replacement timestamps.
- Require an independently supplied manifest SHA-256. Hash consistency is integrity evidence, not authentication of a dataset release, a pretrained checkpoint or scientific validity.
- Read/rehash actual NTU input bytes with `verify_rgb_manifest`; never follow paths supplied by an unverified bundle.
- Preserve fractional PTS/time bases and all keypoint/person confidences. Normalization, thresholds, derivatives and tensorization remain a separate next task.
- Keep real protocol null hashes, graph/control fingerprints, seeds, swap budgets, training budgets and success thresholds unchanged.
- CI inputs are generated fixtures. Do not acquire or publish NTU media or pretrained weights.

## Task 1: Verified reader and read-only CLI

**Files:** create `python/yolo_flywire/pose_bundle.py`, `tests/test_pose_bundle.py`, and `docs/pose-bundle.md`; extend `integration/test_pose_backend.py` only to consume its existing generated extraction output.

**Interfaces:**
- `load_development_bundle(bundle, *, root, inventory, spec, expected_manifest_sha256) -> VerifiedPoseBundle`.
- `VerifiedPoseBundle.samples` is a tuple of `TimedPoseSample`; each includes sample/label/subject/split, dimensions, a `PointSequence`, original integer `pts`, tuple of `Fraction` time bases, and tuple of detector confidences. `timestamps` and `missing_person` expose exact times and explicit person masks without discarding geometry confidence.
- `python -m yolo_flywire.pose_bundle --bundle ... --root ... --inventory ... --config ... --manifest-sha256 ...` verifies and prints a summary, creates no outputs, and exits nonzero on invalid inputs.

- [ ] Write RED tests that generate bundles through the existing extraction fixture; verify immutable round trips, exact nonuniform timing, confidence preservation, and absence of decoder/inference calls during reading.
- [ ] Require rejection of missing commit marker, modified bytes, symlinks/nonregular entries, extra files, malformed/duplicate-key JSON, inconsistent spec/schema/code/input hashes, roster/label/split/count changes, noncanonical indices, missing/extra/desynchronized rows, invalid numbers/timing and inconsistent missing-person masks.
- [ ] Extend actual PyAV/untrained-YOLO integration to read both generated bundles and verify identical timed samples; retain socket blocking and final-test decoding guard.
- [ ] Push test-only RED plus this plan on the existing research branch. Read exact-head CI failures before writing production code.
- [ ] Implement minimal manifest and streaming row validation with fixed filenames, strict JSON types and immutable outputs. Do not call the legacy geometry-only loader or any model API.
- [ ] Run exact-head full CI, including existing extraction integration and FlyWire cache verification. Repair actual regressions without weakening tests.
- [ ] Review the changed scope and record exact RED/GREEN commits, run IDs, test counts, limitations and next task in #68.

## Acceptance and limits

A GREEN result proves the reader accepts the current producer output and rejects the tested corruptions before returning observations. It does not prove the fixtures are NTU data, that untrained detections are useful, that the filesystem is an adversarially safe snapshot, or that FlyWire improves recognition. Use private read-only inputs. The reader materializes immutable observations in memory; a streaming/sharded training dataset and time-aware feature encoder are not delivered by this task.

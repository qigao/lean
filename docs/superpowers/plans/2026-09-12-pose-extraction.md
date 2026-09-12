# NTU YOLO/Pose Development Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Connect the verified NTU RGB inventory to local, version- and weight-bound YOLO/Pose extraction without opening final-test videos.

**Architecture:** `pose_extract.py` owns configuration validation, input verification, streaming serialization and transactional publication. `pose_backend.py` owns PyAV decoding and the Ultralytics adapter. Existing `ntu_io.verify_rgb_manifest` supplies the only accepted roster; existing pose JSONL remains the geometry format, with a separately hashed rational-PTS timing sidecar. The CLI exposes development extraction only; there is no final-test override.

**Tech Stack:** Python 3.11+, NumPy, PyAV, Ultralytics/PyTorch, pytest, existing GitHub Actions. The integration environment uses Ultralytics 8.4.146 and PyAV 14.4.0; actual extraction requires explicit exact versions of all five numerical/media packages and a local weight SHA-256.

**Spec:** `docs/phase2-real-data-freeze.md`; continuation of the approved inventory -> fixed YOLO/Pose -> four-arm experiment boundary. This slice implements extraction, not training or empirical effectiveness.

## Fixed extraction behavior

- Accept only a verified local NTU inventory and an existing regular `yolo26n-pose.pt` with the supplied SHA-256. No weight/model/data download, alternate detector, tracking, depth, skeleton, or RGB embedding fallback.
- Before model construction check exact installed versions of ultralytics, torch, numpy, av and opencv-python. Record Python/platform and FFmpeg library versions as additional environment provenance; this does not assert bitwise reproducibility across machines.
- Fixed prediction options: CPU float32, batch one, imgsz 640, conf 0.25, iou 0.7, rect false, augment false, max_det 300, classes [0], no saving/plotting/stream dropping. Pin package version instead of allowing implicit upgrades.
- Only train/validation rows are decoded. Final-test bytes may be integrity-hashed by inventory verification but never decoded, inferred, displayed or evaluated.
- Decode one video stream through PyAV; retain every decoded frame and its original integer PTS and rational time base. Missing/nonincreasing timestamps, empty streams, corrupt decode, dimension drift and non-uint8 three-channel images are errors. No guessed frame rate or timestamp repair.
- Output 17 COCO keypoints as original-image pixel x/y/confidence plus person confidence. No detections means exactly 17 zero-confidence zero-coordinate points. Multiple detections or a malformed pose tensor means failure; no largest-box/highest-score identity heuristic.
- `observations.jsonl` conforms to the existing `yolo_io` loader. `timing.jsonl` joins it by sample_id/frame_index and preserves PTS/time_base. Downstream real temporal ingestion must consume timing or explicitly freeze resampling; the old geometry-only loader alone is not a timed real runner.
- Stream rows, hash both output files, record per-sample frame counts, input content/split/inventory hashes, extraction spec/code hashes and environment. Reverify inputs and weights before publishing the manifest commit marker.
- Refuse existing output directories. Reserve a fresh directory exclusively; clean owned files on failure. A directory without a final manifest is incomplete and must not be consumed. Crash/kill recovery never treats partial files as success.
- Keep real preflight null hashes, graph/control fingerprints, classifier budgets/seeds and success criterion unchanged. Development fixture hashes are never promoted to real NTU provenance.

## Task 1: Core extraction and adapters

**Files:** Create `tests/test_pose_extract.py`, `python/yolo_flywire/pose_extract.py`, `python/yolo_flywire/pose_backend.py`, `docs/pose-extraction.md`; add explicit optional extraction dependencies to `python/pyproject.toml`.

**Interfaces:** `ExtractionSpec`, `extract_development(root, inventory, weights, spec, output) -> dict`, CLI `python -m yolo_flywire.pose_extract --root ROOT --inventory MANIFEST --weights WEIGHTS --config SPEC --output DIR`; backend `decode_video(path)` yields `(pts, Fraction time_base, uint8 BGR image)`; `load_predictor(weights, spec)` returns a callable yielding `(keypoints[N,17,3], confidence[N])`.

- [ ] Commit acceptance tests and plan without implementation. Require exact-head CI RED only for missing extraction APIs; existing 119 tests and Lean remain passing.
- [ ] Implement validation and dependency-lazy adapters after observing RED. Validate actual byte/roster binding before following inventory paths.
- [ ] Implement streaming train/validation extraction and non-overwriting manifest-last output. Fail closed on ambiguous detections, bad timing, corrupted/changed inputs and publication failures.
- [ ] Run all unit tests, review scope and record exact-head GREEN in #68.

## Task 2: Real library/decode integration gate

**Files:** Create `integration/test_pose_backend.py`; extend `.github/workflows/yolo-flywire-ci.yml` with a required extraction integration job after unit tests.

- [ ] Test real PyAV encode/decode and exact PTS ordering on a tiny generated lossless video; test corrupt input rejection.
- [ ] Build and save an untrained YOLO26n-pose model from the pinned package's local YAML, then load that explicit byte-hashed file through the real adapter and perform a CPU forward pass. No pretrained weight download, NTU media or performance claim.
- [ ] Run extraction twice on generated video fixtures, compare serialized observations/timing and verify final-test rows are excluded. Upload no media or weights.
- [ ] Check authoritative GitHub CI jobs and update #68 with measured results and remaining real-data blockers. No final-test experiment is authorized by these integration fixtures.

## Verification

Unit: `python -m pytest tests -q`. Integration: `python -m pytest integration/test_pose_backend.py -q`. Lean and graph provenance remain part of the existing exact-head workflow. Test doubles cover orchestration only; the separate integration job is necessary evidence for the actual decoder/model API, still not evidence about real gesture accuracy.

API references: https://docs.ultralytics.com/tasks/pose ; https://docs.ultralytics.com/modes/predict ; https://pyav.org/docs/stable/api/video.html ; https://pypi.org/project/ultralytics/8.4.146/ .

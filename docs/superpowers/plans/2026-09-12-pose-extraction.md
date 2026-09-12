# NTU YOLO/Pose Development Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Connect the verified NTU RGB inventory to local, version- and weight-bound YOLO/Pose extraction without opening final-test videos.

**Architecture:** `pose_extract.py` owns configuration validation, input verification, streaming serialization and manifest-last publication. `pose_backend.py` owns PyAV decoding and the Ultralytics adapter. Existing `ntu_io.verify_rgb_manifest` supplies the only accepted roster; existing pose JSONL remains the geometry format, with a separately hashed rational-PTS timing sidecar. The CLI exposes development extraction only, with no final-test override.

**Tech Stack:** Python 3.11+, NumPy, PyAV, Ultralytics/PyTorch, pytest, existing GitHub Actions. Integration pins Ultralytics 8.4.146 and PyAV 15.1.0. Actual extraction requires explicit exact versions of all five numerical/media packages and an independently approved local weight SHA-256.

**Spec:** `docs/phase2-real-data-freeze.md`; continuation of the approved inventory -> fixed YOLO/Pose -> four-arm experiment boundary. This slice implements extraction, not training or empirical effectiveness.

## Fixed extraction behavior

- Accept only a verified local NTU inventory and an existing regular `yolo26n-pose.pt` with the supplied SHA-256. No weight/model/data download, alternate detector, tracking, depth, skeleton, or RGB embedding fallback.
- Before model construction check exact installed versions of ultralytics, torch, numpy, av and opencv-python. Record Python/platform in the manifest as well. This is not a hermetic wheel/OS/FFmpeg-build freeze and does not assert bitwise reproducibility across machines; that remains a confirmatory-environment requirement.
- Fixed prediction options: CPU float32, one image, imgsz 640, conf 0.25, iou 0.7, rect false, augment false, max_det 300, classes [0], no saving/plotting/stream dropping. Dependencies are identified explicitly, not upgraded inside extraction.
- Only train/validation rows are decoded. Final-test bytes may be integrity-hashed by inventory verification but never decoded, inferred, displayed or evaluated.
- Decode one AVI video stream through PyAV; retain every decoded frame and original integer PTS/rational time base. Missing/nonincreasing timestamps, empty/corrupt decode, dimension drift and non-uint8 three-channel images are errors. No guessed frame rate or timestamp repair.
- Output 17 COCO keypoints as original-image pixel x/y/confidence plus person confidence. No detections means 17 zero-confidence zero-coordinate points. Multiple detections or malformed pose tensors mean failure; no largest-box/highest-score identity heuristic.
- `observations.jsonl` conforms to `yolo_io`. `timing.jsonl` joins it by sample_id/frame_index and preserves PTS/time_base. Downstream real temporal ingestion must consume timing or explicitly freeze resampling; the old geometry-only loader alone is not a timed real runner.
- Stream rows, hash both outputs, record per-sample frame counts, input content/split/inventory hashes, extraction spec/code hashes and environment. Reverify inputs and weights before publishing the manifest commit marker.
- Refuse existing output directories. Reserve a fresh directory exclusively; clean owned files on failure. A directory without a final manifest is incomplete and must not be consumed. Crash/kill recovery never treats partial files as success.
- Keep real preflight null hashes, graph/control fingerprints, classifier budgets/seeds and success criterion unchanged. Fixture hashes are never promoted to real NTU provenance.

## Task 1: Core extraction and adapters

**Files:** Create `tests/test_pose_extract.py`, `python/yolo_flywire/pose_extract.py`, `python/yolo_flywire/pose_backend.py`, `docs/pose-extraction.md`; add optional extraction dependencies to `python/pyproject.toml`.

**Interfaces:** `ExtractionSpec`; `extract_development(root, inventory, weights, spec, output) -> dict`; CLI `python -m yolo_flywire.pose_extract --root ROOT --inventory MANIFEST --weights WEIGHTS --config SPEC --output DIR`; backend `decode_video(path)` yields `(pts, Fraction time_base, uint8 BGR image)`; `load_predictor(weights, spec)` returns a callable yielding `(keypoints[N,17,3], confidence[N])`.

- [x] Commit tests before implementation. CI #80 / `34691747174` at `48d9a4513ac7a49e2caf6f84a1181e2f45167a83`: 30 expected missing-module failures, prior 119 tests passed, Lean passed.
- [ ] Implement dependency-lazy adapters and validate actual input binding before following inventory paths.
- [ ] Implement streaming train/validation extraction and exclusive manifest-last output. Fail closed on ambiguous detections, bad timing, changed inputs and publication failures.
- [ ] Run all unit tests, review scope and record exact-head GREEN in #68.

## Task 2: Actual library integration gate

**Files:** Create `integration/test_pose_backend.py`; extend `.github/workflows/yolo-flywire-ci.yml` with an independent, required extraction integration job following Lean. It may run in parallel with unit tests so its missing-API RED is observable.

- [ ] Test real PyAV encode/decode and exact PTS ordering on a tiny generated lossless video; test corrupt input rejection.
- [ ] Build/save an untrained YOLO26n-pose model from the installed package YAML, then load the explicit byte-hashed file through the actual adapter and perform CPU inference. No pretrained download, NTU media or performance claim.
- [ ] Run extraction twice on generated fixtures, compare serialized observations/timing and require final-test rows to remain excluded. Upload no media or weights.
- [ ] Check all exact-head CI jobs and update #68 with measured results and remaining blockers. No final-test experiment is authorized by these fixtures.

Environment correction: CI #81 failed **before integration tests** because PyAV 14.4.0 selected a source distribution and lacked FFmpeg development libraries. Integration now explicitly requests `av==15.1.0` with `--only-binary=av`; this is a pre-execution dependency correction, not a fallback or a result-dependent experiment change. Opaque unit-test version strings remain test doubles, not a real dependency lock.

## Verification

Unit: `python -m pytest tests -q`. Integration: `python -m pytest integration/test_pose_backend.py -q -s`. Lean and graph provenance remain in the exact-head workflow. Test doubles cover orchestration only; actual decoder/model integration is separate evidence, still not recognition accuracy. See #68 for authoritative completion evidence; no unchecked task is implied complete here.

API references: https://docs.ultralytics.com/tasks/pose ; https://docs.ultralytics.com/modes/predict ; https://pyav.org/docs/stable/api/video.html ; https://pypi.org/project/ultralytics/8.4.146/ ; https://pypi.org/project/av/15.1.0/ .

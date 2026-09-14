# YOLO11 Hosted-CI Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second real-data development path that runs the frozen 10-class NTU120 experiment on standard GitHub-hosted runners using `yolo11n-pose.pt`, bounded per-clip streaming, deterministic shard aggregation, and the existing indexed four-arm trainer/comparison core.

**Architecture:** Keep the existing YOLO26 local-root/self-hosted path behavior unchanged. Add an explicit remote-manifest source boundary, stream one verified clip at a time in extraction shards, aggregate only train/validation pose sidecars plus final-test byte-verification evidence, bind the aggregated bundle without requiring deleted RGB bytes, then reuse the existing four-arm comparison execution core.

**Tech Stack:** Python 3.11, PyTorch, Ultralytics 8.4.146, PyAV 15.1.0, GitHub Actions, canonical JSON/SHA-256 provenance.

**Spec:** `docs/superpowers/specs/2026-09-13-yolo11-hosted-ci-streaming-design.md`

## Global Constraints

- Existing `v0-real-ntu120-preflight` / `yolo26n-pose.pt` behavior and evidence meaning remain unchanged.
- New protocol ID is `v1-hosted-yolo11n-preflight`; model basename is exactly `yolo11n-pose.pt`.
- Scientific source identity excludes transport URLs; `transport_manifest_hash` is run provenance only.
- Final-test clips may be downloaded and byte-verified but must never be decoded or produce pose observations.
- No local/remote fallback path. Every public entry chooses one source contract explicitly.
- Seeds stay `[7, 11, 19, 23, 31]`; budget stays 20 epochs / 40 updates / 50,000 parameters.
- Existing indexed trainer and four-arm arm execution are reused; no duplicate optimizer/model-comparison implementation.
- User explicitly requested direct implementation rather than RED-first TDD. Tests are added/run as verification after each implementation slice.

---

### Task 1: Generalize explicit pose-model identity without changing YOLO26 behavior

**Files:**
- Modify: `python/yolo_flywire/pose_extract.py`
- Modify: `python/yolo_flywire/pose_backend.py`
- Test: `tests/test_pose_extract.py`

**Interfaces:**
- `ExtractionSpec(..., model: str = "yolo26n-pose.pt")`
- `_check_weights(path, spec)` verifies `path.name == spec.model`.
- `load_predictor(weights, spec)` verifies the same explicit basename plus task=`pose` and `(17,3)` keypoint shape.

- [ ] Add explicit model identity to `ExtractionSpec` with the existing YOLO26 basename as the default so old descriptors remain byte-for-byte equivalent.
- [ ] Replace hard-coded YOLO26 basename checks with `spec.model`; do not accept aliases or alternate names.
- [ ] Add verification that a YOLO11 extraction spec produces a distinct descriptor/hash while old YOLO26 tests remain unchanged.
- [ ] Run focused extraction/backend tests and full Python suite.

### Task 2: Remote scientific source and hosted protocol freeze

**Files:**
- Create: `python/yolo_flywire/remote_rgb.py`
- Create: `python/yolo_flywire/hosted_freeze.py`
- Create: `protocols/v1-hosted-yolo11n-preflight.json`
- Test: `tests/test_remote_rgb.py`
- Test: `tests/test_hosted_freeze.py`

**Interfaces:**
- `validate_remote_manifest(value: dict[str, Any]) -> VerifiedRemoteManifest`
- `VerifiedRemoteManifest.inventory` is a URL-free inventory record compatible with the existing scientific dataset/split hash rules.
- `VerifiedRemoteManifest.source_manifest_hash` excludes locators.
- `VerifiedRemoteManifest.transport_manifest_hash` includes locators.
- `freeze_hosted_inputs(protocol, *, remote_manifest, weights) -> dict[str, Any]` binds YOLO11 checkpoint/runtime/schema/encoder and scientific source hashes.

- [ ] Validate canonical NTU names and exact declared setup/camera/subject/repetition/action/sample/label/split fields.
- [ ] Recompute `dataset_content_hash` and `split_hash` using the same canonical payloads as `build_rgb_manifest`.
- [ ] Reject duplicate sample IDs/content, malformed hashes/sizes, split drift and incomplete 10-class coverage in train/validation/final-test.
- [ ] Keep locator text out of scientific hashes; compute a separate transport hash over the complete private manifest.
- [ ] Freeze `v1-hosted-yolo11n-preflight` using exact `yolo11n-pose.pt` basename/hash and pinned runtime versions.
- [ ] Run focused remote/freeze tests plus all existing YOLO26 freeze tests.

### Task 3: One-clip-at-a-time shard extraction and deterministic aggregation

**Files:**
- Create: `python/yolo_flywire/hosted_extract.py`
- Create: `python/yolo_flywire/hosted_aggregate.py`
- Test: `tests/test_hosted_extract.py`
- Test: `tests/test_hosted_aggregate.py`

**Interfaces:**
- `extract_remote_shard(remote_manifest, *, shard_index, shard_count, weights, spec, output) -> dict[str, Any]`
- `aggregate_remote_shards(remote_manifest, shard_paths, *, spec, output) -> dict[str, Any]`

- [ ] Deterministically assign samples by frozen manifest order modulo shard count; shard count affects scheduling only.
- [ ] Materialize each HTTPS locator to a fresh regular file, stream-hash while writing, then require exact size/SHA before decode.
- [ ] For final-test rows: record byte verification and unlink immediately; decoder/predictor is never called.
- [ ] For train/validation rows: decode all frames, run the explicit YOLO11 predictor, write per-sample geometry/timing outputs and hashes, fsync, then unlink RGB before continuing.
- [ ] Shard reports contain no URL text and include source/extraction/schema/runtime identities.
- [ ] Aggregation rejects missing/duplicate/unexpected samples, identity drift, final-test sidecars and hash mismatch; reconstruct order from the frozen source manifest.
- [ ] Aggregator emits standard `manifest.json`, `observations.jsonl`, `timing.jsonl` logical bundle files for development samples only.
- [ ] Run focused streaming/aggregation tests using generated AVI fixtures and an untrained local pose checkpoint.

### Task 4: Remote indexed binding and shared four-arm execution core

**Files:**
- Modify: `python/yolo_flywire/pose_index.py`
- Modify: `python/yolo_flywire/pose_indexed_development.py`
- Modify: `python/yolo_flywire/pose_comparison.py`
- Create: `python/yolo_flywire/hosted_development.py`
- Test: `tests/test_remote_indexed_development.py`
- Test: `tests/test_hosted_development.py`

**Interfaces:**
- Extract internal `pose_index._index_verified_bundle(bundle, verified_inventory, spec, expected_manifest_sha256)`.
- Existing `index_development_bundle(...)` remains local-root and calls the helper only after `verify_rgb_manifest`.
- New `index_remote_development_bundle(bundle, remote_manifest, spec, expected_manifest_sha256)` validates the remote manifest then calls the same helper.
- Extract `pose_indexed_development._bind_indexed_source(indexed, feature_spec, classes, expected_encoder_hash)`.
- Extract `pose_comparison.run_bound_pose_comparison(source, ..., expected_binding_sha256, ...)`; existing `run_pose_comparison(...)` loads local source then delegates.
- `run_hosted_development(...)` loads remote indexed source then delegates to the same comparison core.

- [ ] Refactor only at verified-source boundaries; preserve existing local signatures and reports.
- [ ] Remote indexed loading must never call `verify_rgb_manifest` or require RGB files.
- [ ] Verify remote binding produces the same partition tensor semantics as an equivalent local fixture bundle.
- [ ] Reuse `_preflight_models` / `_execute_arm` unchanged through the shared bound-source comparison core.
- [ ] Hosted development report stays validation-only with `final_test_evaluated=false` and `topology_claim_evaluated=false`.
- [ ] Run focused remote indexed/hosted development tests, existing comparison tests, and full Python suite.

### Task 5: GitHub-hosted sharded workflow, documentation and exact-head verification

**Files:**
- Create: `.github/workflows/yolo-flywire-hosted-real.yml`
- Create: `tests/test_hosted_real_workflow.py`
- Modify: `docs/phase2-real-data-freeze.md`
- Modify: `docs/real-input-freeze.md`

**Interfaces:**
- Manual-only workflow input: shard count and development hyperparameters.
- Secret: `NTU120_REMOTE_MANIFEST_URL` points to the private authorized transport manifest; locator contents are never printed/uploaded.

- [ ] Coordinator on `ubuntu-latest`: fetch private manifest, freeze hosted protocol, download/hash official YOLO11n, produce URL-redacted scientific freeze evidence and deterministic shard matrix.
- [ ] Matrix jobs on `ubuntu-latest`: independently download/hash the exact YOLO11n checkpoint and run one bounded remote shard each.
- [ ] Aggregation job downloads only shard pose/evidence artifacts, aggregates the exact roster, verifies final-test byte evidence/no pose, builds remote indexed binding and executes the existing four-arm comparison core.
- [ ] Upload only frozen protocol/hash records, development pose bundle/binding, graph provenance, execution config and development report; never upload RGB, URLs, credentials or model bytes.
- [ ] Keep existing `.github/workflows/yolo-flywire-real.yml` unchanged for YOLO26 self-hosted execution.
- [ ] Run ordinary workflow-contract tests, full Python suite, Lean build, actual-library integration and FlyWire provenance.
- [ ] Require fresh exact-head GitHub Actions success before calling the hosted implementation complete.

# Hosted YOLO11 real NTU development experiment

Tracked by #68 and PR #69. This is a second real-data protocol for standard GitHub-hosted runners. It does not replace or mutate the existing YOLO26/self-hosted protocol.

## Purpose

`.github/workflows/yolo-flywire-hosted-real.yml` runs the frozen 10-class NTU RGB+D 120 development experiment with `yolo11n-pose.pt` on `ubuntu-latest` without downloading the full NTU RGB archives onto one runner.

The workflow is manual-only. It uses a coordinator, a bounded extraction matrix, and one aggregate/comparison job. Each extraction shard downloads one authorized RGB clip at a time, verifies exact byte size and SHA-256, extracts train/validation pose observations, then deletes the RGB clip before moving to the next one.

Final-test clips are downloaded only to verify their frozen byte identity. They are deleted without opening the decoder or constructing pose observations. There is no final-test selector or classifier evaluator in this workflow.

## Required repository secret

Configure one Actions secret:

```text
NTU120_REMOTE_MANIFEST_URL
```

The value is an authorized HTTPS URL to a private transport manifest. The URL itself and all per-sample authorized locators stay private and are never uploaded as artifacts.

The transport manifest has this top-level form:

```json
{
  "format_version": 1,
  "kind": "ntu120_rgb_remote_transport",
  "task_labels": ["... frozen 10 labels ..."],
  "split_policy": {"... exact frozen split policy ..."},
  "samples": [
    {
      "filename": "S001C001P001R001A008_rgb.avi",
      "sample_id": "S001C001P001R001A008",
      "setup": 1,
      "camera": 1,
      "subject": 1,
      "repetition": 1,
      "action": 8,
      "label": "A8:sitting_down",
      "split": "train",
      "size_bytes": 123456,
      "sha256": "<64 lowercase hex characters>",
      "locator": "https://<authorized per-clip location>"
    }
  ]
}
```

Every row is validated against the canonical NTU filename parser, the frozen X-Sub120 subject boundary, and deterministic training-side validation split. Train, validation, and final-test partitions must each cover all ten frozen actions. Duplicate sample IDs, duplicate byte hashes, split drift, malformed hashes, non-positive sizes, or non-HTTPS locators fail closed.

## Scientific source identity vs transport identity

The hosted path derives four relevant hashes:

- `dataset_content_hash`: ordered `{sample_id,size_bytes,sha256}` identity, using the same rule as the local NTU inventory;
- `split_hash`: the same dataset/split/task-label identity rule as the local inventory;
- `source_manifest_hash`: URL-free scientific source identity over all non-transport sample metadata;
- `transport_manifest_hash`: current private locator-set identity for run provenance only.

Rotating expiring signed URLs may change `transport_manifest_hash`, but it must leave `dataset_content_hash`, `split_hash`, and `source_manifest_hash` unchanged.

## YOLO11 extractor identity

The hosted protocol is `protocols/v1-hosted-yolo11n-preflight.json`. It freezes:

```text
model = yolo11n-pose.pt
ultralytics = 8.4.146
av = 15.1.0
```

The checkpoint is downloaded explicitly from the official Ultralytics GitHub assets release:

```text
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt
```

The workflow does not use `YOLO("yolo11n-pose.pt")` as a download mechanism. The coordinator fetches the fixed GitHub release asset with `curl`, hashes the downloaded bytes, and binds that SHA-256 into the frozen hosted protocol. Every extraction shard fetches the same official release asset independently and refuses to run unless its SHA-256 matches the coordinator freeze.

The workflow also pins the exact PyTorch, NumPy, OpenCV, PyAV, and Ultralytics package versions used by extraction. Model basename, checkpoint hash, prediction options, runtime versions, observation schema, and pose encoder are all bound explicitly. There is no alternate-model fallback.

## Bounded GitHub-hosted execution

The default manual input uses 16 extraction shards. `shard_count` may be changed without changing scientific sample order or source identity. Membership is deterministic from the frozen source order modulo shard count.

Each shard runs on `ubuntu-latest` and keeps only its current RGB clip plus growing pose sidecars. GitHub-hosted jobs are capped at six hours, so all hosted workflow jobs use `timeout-minutes <= 360`. If a shard is too large for that execution limit, increase the shard count; do not reduce the sample roster, frame sampling, or extraction contract.

The aggregate job accepts all deterministic shards exactly once, rejects missing/duplicate/unexpected samples, verifies final-test byte evidence has no pose sidecar, reconstructs train/validation observations in frozen source order, and binds them through the existing `IndexedPoseDevelopment` type.

## Four-arm comparison

The hosted path does not implement a second trainer. After remote indexed binding it reuses the existing four-arm model construction and execution core:

```text
GRU
random sparse graph
matched rewired FlyWire graph
original FlyWire graph
```

Seeds remain `7, 11, 19, 23, 31`. The budget remains 20 epochs / 40 optimizer updates / 50,000 parameter ceiling. Development checkpoint selection remains validation-only.

The resulting comparison still records:

```text
evidence_scope = development_validation_only
final_test_evaluated = false
topology_claim_evaluated = false
```

A positive FlyWire-minus-rewired validation difference is development evidence only. It is not a confirmatory topology claim.

## Artifact policy

The workflow never uploads or caches:

- NTU RGB media or archives;
- the private transport manifest or authorized locators;
- YOLO checkpoint bytes;
- final-test pose observations.

Shard artifacts contain only train/validation pose sidecars plus URL-free byte-verification reports. Final evidence contains frozen protocol/hash records, the URL-free scientific source record, aggregate manifest, FlyWire provenance, execution config, and development report.

## Running

After this workflow exists on the repository default branch and `NTU120_REMOTE_MANIFEST_URL` is configured, open Actions → `yolo-flywire-hosted-real` → Run workflow.

The workflow can then run the real NTU development experiment on standard GitHub-hosted runners. Until a successful run produces the final development artifact, no real YOLO11 NTU recognition result or FlyWire-vs-rewired empirical advantage should be claimed.

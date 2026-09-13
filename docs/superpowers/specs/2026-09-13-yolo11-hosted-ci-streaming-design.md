# YOLO11 Hosted-CI Streaming Real Experiment Design

Tracked by #68 and PR #69.

## Goal

Add a second real-data development protocol that can execute on standard GitHub-hosted `ubuntu-latest` runners without requiring a 260+ GB local NTU RGB tree or a self-hosted runner. The new protocol uses `yolo11n-pose.pt` and bounded sharded streaming of authorized NTU RGB clips while preserving exact byte provenance, the existing train/validation/final-test boundary, the frozen FlyWire topology/control contract, and the five-seed four-arm development comparison.

The existing YOLO26 real protocol remains unchanged and keeps its current evidence meaning. The hosted path is a separate protocol family and must never silently substitute for it.

## Non-goals

This design does not create or authorize a final-test evaluator. It does not reinterpret development validation as confirmatory evidence. It does not add a compatibility/fallback path between local-root and remote-manifest sources. It does not weaken existing byte/hash checks, graph/control invariants, seed/budget constraints, or topology-claim boundaries. It does not redistribute NTU media or make unauthorized download URLs public.

## Protocol identity

Create a distinct protocol, `v1-hosted-yolo11n`, whose extractor identity is frozen independently from the existing YOLO26 protocol.

The hosted protocol keeps the existing scientific task and control contract:

- dataset: NTU RGB+D 120 RGB-only, same ten action labels A8/A9/A22/A23/A26/A27/A31/A34/A35/A36;
- split: official X-Sub120 outer split plus the existing deterministic training-side validation subject hash;
- final test: sealed from decoding, model selection, threshold selection, or classifier evaluation;
- pose representation: COCO17 xy/confidence plus person confidence, same 121-dimensional timed feature encoder;
- FlyWire release/source/topology selection/fingerprints/rewiring algorithm: unchanged;
- seeds: `[7, 11, 19, 23, 31]`;
- budget: 20 epochs / 40 updates / 50,000 parameter ceiling;
- primary metric: macro-F1;
- development evidence scope: validation-only and never topology-confirmatory.

The extractor-specific fields change explicitly:

- model family: `yolo11n-pose.pt`;
- checkpoint SHA-256: derived from the exact downloaded checkpoint bytes and frozen into the protocol snapshot;
- Ultralytics/runtime versions: exact and frozen in the extraction descriptor;
- extraction descriptor/model identity hash: distinct from the YOLO26 protocol.

No code may infer model identity from a filename alone. Model basename, hash, task, keypoint shape and runtime identity must all agree with the frozen hosted protocol.

## Authorized remote source manifest

The hosted protocol consumes a private, explicit remote-source manifest rather than a local corpus root. The manifest is not a list of arbitrary URLs accepted at runtime; it is the complete frozen NTU roster for this experiment.

Each row records exactly:

- canonical NTU RGB filename;
- parsed NTU setup/camera/subject/repetition/action identity;
- canonical sample ID;
- label;
- frozen split assignment;
- byte size;
- SHA-256;
- an opaque authorized download URL or provider locator needed by the workflow.

The manifest also records the same split policy descriptor and ordered task labels as the local inventory format. URLs/locators are transport metadata and are excluded from scientific dataset identity hashes. Expiring signed URLs may be rotated only by producing a separately verified transport manifest that maps to the exact same frozen sample identities, sizes and SHA-256 values.

Three digests are distinct:

1. `dataset_content_hash`: same canonical content identity rule as the local NTU inventory — ordered `{sample_id,size_bytes,sha256}` records;
2. `split_hash`: same canonical task-label, split-policy and sample-assignment rule as the local inventory;
3. `source_manifest_hash`: canonical scientific source manifest hash over all non-transport fields, excluding URL/provider locator text.

The current run also records `transport_manifest_hash` over the complete private transport manifest, including the opaque locators. `transport_manifest_hash` is run provenance only: it is not part of scientific protocol identity and may change when signed URLs rotate while `dataset_content_hash`, `split_hash` and `source_manifest_hash` remain unchanged.

The remote manifest must fail closed on malformed NTU names, duplicate sample IDs, duplicate content hashes, missing classes in any required partition, split drift, unexpected labels, non-canonical hashes, non-positive sizes, or any mismatch between parsed filename identity and declared row fields.

## Source boundary

Introduce an explicit source abstraction rather than teaching existing local-root functions to guess between local and remote modes.

The two supported source types are intentionally separate:

- `LocalRgbSource`: existing local-root + `verify_rgb_manifest` semantics, unchanged;
- `RemoteRgbManifestSource`: frozen remote manifest plus verified per-clip materialization semantics.

There is no fallback from one source kind to the other. Callers choose one explicit source contract and mismatched source/config combinations are errors.

A remote source is considered verified only after every frozen manifest row has been byte-checked against declared size and SHA-256 during the current workflow. The final-test rows participate in source identity verification but never in pose decoding.

## Sharded hosted-CI execution

The real hosted workflow is manual-only and runs on `ubuntu-latest`. It does not require a self-hosted runner.

Execution has four stages.

### 1. Freeze/validate transport and protocol identity

A lightweight coordinator job loads the private authorized remote manifest and validates its complete roster, split policy, ten-class coverage, `dataset_content_hash`, `split_hash` and `source_manifest_hash`. It separately records `transport_manifest_hash` for the current authorized locator set. It downloads the official `yolo11n-pose.pt`, records its SHA-256 and verifies the exact pose task / 17-keypoint model contract under the pinned Ultralytics runtime.

The coordinator produces a frozen hosted real-input protocol snapshot that binds:

- remote `dataset_content_hash`;
- `split_hash`;
- `source_manifest_hash` with transport fields excluded;
- YOLO11n checkpoint hash;
- extraction/runtime descriptor and hash;
- observation schema hash;
- pose encoder hash;
- existing graph/control design fields.

The current `transport_manifest_hash` is stored in run evidence beside the frozen protocol, not folded into the protocol's scientific identity.

The coordinator never decodes NTU media and never runs a classifier.

### 2. Matrix extraction shards

The workflow deterministically partitions the complete remote roster into bounded shards. Shard membership is a pure function of the frozen scientific source manifest and workflow shard count; changing shard count may change scheduling but must not change scientific sample order or source identity.

Each shard processes one clip at a time:

1. download one frozen row to a fresh temporary path;
2. verify regular-file semantics, exact byte size and SHA-256 before any decode;
3. if the row is `final_test`, record byte-verification evidence and delete the RGB file without opening a decoder;
4. if the row is `train` or `validation`, run the exact frozen YOLO11n extraction on all decoded frames;
5. write one per-sample geometry/timing sidecar plus per-sample extraction metadata and output hashes;
6. fsync/close outputs;
7. delete the RGB clip immediately;
8. assert the temporary clip no longer exists before moving to the next row.

At most the configured shard's current clip plus its growing pose output is resident on disk. No shard downloads a full NTU archive and no shard preserves licensed RGB media as an artifact or cache.

### 3. Deterministic aggregation and indexed binding

An aggregation job downloads only shard evidence/pose artifacts. It verifies that:

- every development sample from the frozen remote manifest appears exactly once;
- every final-test sample has byte-verification evidence but no pose sidecar;
- no unexpected sample exists;
- per-sample source SHA-256 equals the frozen remote manifest;
- shard extraction spec/schema/encoder/runtime identities all equal the frozen protocol;
- all sidecar byte hashes match shard reports;
- sample order is reconstructed from the frozen source manifest, never artifact arrival order.

The aggregator emits the same logical development pose bundle contract used by downstream indexing, but its source provenance is `remote_manifest_verified` rather than `local_root_verified`.

A remote-source indexed loader verifies the frozen source manifest and the aggregated pose bundle directly. It must not call `verify_rgb_manifest` or require deleted RGB files. It reuses the existing pose row/schema/geometry/timing validators and indexed batching/training logic. There is no local-root fallback inside this loader.

### 4. Four-arm development comparison

The existing GRU / random graph / degree-and-weight-matched rewired / FlyWire comparison remains the only development comparison implementation. The hosted path supplies a verified remote-source indexed binding plus the same graph protocol/control pins and explicit execution config.

Seeds, epoch/update/parameter budget, graph widths and selected development hyperparameters obey the same explicit validation rules as the existing real-development path. Reports remain:

- `evidence_scope = development_validation_only` at comparison level;
- `final_test_evaluated = false`;
- `topology_claim_evaluated = false`.

No `FlyWire > rewired` final threshold is applied to development validation scores.

## Final-test seal

The remote-source manifest includes final-test clip byte identities so dataset/split provenance covers the complete frozen experimental corpus. Hosted extraction shards are forbidden to decode final-test clips.

The implementation must make this observable and testable:

- final-test shard rows may be downloaded only for byte verification;
- decoder/predictor calls on a final-test row are an immediate error;
- no final-test observation artifact may exist;
- aggregation requires final-test byte-verification evidence and rejects final-test pose output;
- no hosted workflow CLI/input exposes a `--test`, `--final-test`, or equivalent selector.

A separate future confirmatory design is required before any final-test decode or classifier evaluation.

## Checkpoint and runtime handling

The hosted workflow downloads `yolo11n-pose.pt` from the official Ultralytics mechanism under an exact pinned Ultralytics package version. After download, it computes the checkpoint SHA-256 and binds it into the hosted freeze snapshot.

Subsequent shard jobs do not trust an unfrozen filename. They must independently download the official checkpoint and verify it against the frozen SHA-256. Model bytes are not retained as repository artifacts or caches.

Every shard must use the same exact versions for Ultralytics, PyTorch, NumPy, PyAV and OpenCV. Runtime drift is a hard failure.

## Artifact and privacy policy

Never upload or cache:

- NTU RGB clips or archives;
- authorized URLs or provider credentials;
- final-test pose observations;
- model bytes.

Allowed workflow artifacts are restricted to:

- frozen protocol/hash records with transport URLs redacted or excluded;
- run-level `transport_manifest_hash` without locator contents;
- development pose geometry/timing sidecars for train/validation only;
- per-shard source verification reports containing sample IDs, sizes and hashes but no URLs;
- extraction manifests/index/binding records;
- FlyWire provenance/control records;
- execution config and development comparison report.

Artifacts are evidence, not authorization tokens.

## Failure and retry semantics

All execution is fail-closed. There is no automatic source substitution, alternate model, reduced sample roster, reduced frame sampling, relaxed hash check, different split, lower rewiring budget, shorter training budget or final-test compatibility path.

A shard retry may repeat the same deterministic shard against the same frozen protocol/source manifest. The retry must re-download and re-verify every clip in that shard; it may not trust partial prior files. Aggregation accepts exactly one successful evidence record per frozen sample and rejects duplicates with different identities.

If one shard fails, the experiment has no complete development evidence and the comparison stage must not run.

## Testing strategy

Use TDD and keep ordinary CI independent of licensed NTU media.

### Remote manifest unit tests

Generated NTU-style fixture bytes must test:

- canonical remote roster parsing;
- equality of remote `dataset_content_hash` / `split_hash` with an equivalent local `build_rgb_manifest` inventory;
- duplicate sample/content rejection;
- split/label/name mismatch rejection;
- incomplete class coverage rejection;
- transport URL exclusion from `dataset_content_hash`, `split_hash` and `source_manifest_hash`;
- URL rotation changes `transport_manifest_hash` but leaves all scientific source hashes unchanged.

### Streaming extraction tests

Use generated AVI fixtures and an untrained local pose checkpoint. Require:

- one clip materialized at a time;
- source size/SHA verified before decoder construction;
- train/validation clips decoded and emitted;
- final-test clips hashed but decoder/predictor never called;
- temporary RGB deletion before the next sample;
- per-sample output/source hashes recorded;
- no fallback to local-root inventory verification.

### Aggregation/index tests

Require rejection of missing/duplicate shards, unexpected samples, source-hash drift, extraction-spec drift, final-test sidecars, output-hash mismatch and artifact-order dependence. Verify the resulting remote indexed source produces the same partition tensors/binding behavior as an equivalent local fixture bundle.

### Workflow contract tests

The hosted real workflow must be `workflow_dispatch` only, use `ubuntu-latest`, contain no self-hosted label or 350-GiB requirement, use a bounded extraction matrix, never download the full NTU archives, never upload RGB/model/authorized-URL artifacts, and expose no final-test selector.

The existing YOLO26 self-hosted workflow and protocol remain unchanged and retain their current tests.

## Evidence interpretation

A successful hosted run establishes a real NTU development result for the distinct YOLO11n observation-generator protocol. It is not numerically interchangeable with the YOLO26 protocol and must be reported separately.

Positive FlyWire-minus-rewired development differences are exploratory/development evidence only because validation data participate in checkpoint selection. Negative or null differences are equally valid outcomes and must be retained.

No topology-specific confirmatory claim exists until a separately reviewed final-test execution is frozen and run.

## Acceptance criteria

The design is implemented only when all of the following are true:

1. Existing YOLO26 protocol/workflow behavior remains unchanged and green.
2. A separate YOLO11n hosted protocol validates exact model/runtime/source identity.
3. Remote and equivalent local manifests produce identical scientific dataset/split hashes.
4. Signed URL rotation changes only transport provenance, not scientific source identity.
5. Hosted shards use `ubuntu-latest` and bounded one-clip-at-a-time materialization.
6. Final-test clips are byte-verified but never decoded and never produce pose observations.
7. Aggregation reconstructs the exact frozen development roster independent of shard/artifact order.
8. Remote indexed loading verifies source and pose evidence without requiring deleted RGB bytes.
9. Existing indexed trainer/four-arm comparison code is reused with no compatibility fallback.
10. Ordinary fixture CI is fully green without real NTU access.
11. The manual hosted real workflow can run from authorized remote per-clip access without a self-hosted runner or full-corpus local disk.

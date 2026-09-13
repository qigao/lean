# KTH YOLO11 Public Real-CI Design

Tracked by #68 and PR #69.

## Goal

Add a second public real-data development benchmark that can execute end-to-end on standard GitHub-hosted `ubuntu-latest` runners without repository secrets, self-hosted runners, or private dataset credentials. The benchmark uses the official KTH Human Action Database, `yolo11n-pose.pt`, the existing COCO17 temporal feature encoder, and the existing GRU / random-graph / rewired / FlyWire four-arm comparison core.

This protocol is **development evidence only**. It does not replace the existing NTU RGB+D 120 protocol, does not inherit NTU confirmatory thresholds, and does not authorize any NTU or KTH final-test classifier evaluation.

## Why KTH

KTH is a public, compact RGB action benchmark that matches the immediate CI requirement:

- official source: `https://www.csc.kth.se/cvap/actions/`;
- public for non-commercial use;
- six action classes: walking, jogging, running, boxing, hand waving, hand clapping;
- 25 subjects in four scenarios;
- 600 AVI files containing 2,391 official action subsequences;
- 160x120 pixels at 25 fps;
- six official DivX-compressed ZIP archives totaling about 1.15 GB;
- official `00sequences.txt` supplies exact subsequence frame ranges and the original 8/8/9 subject split.

The exact official split frozen here is:

- train: persons 11, 12, 13, 14, 15, 16, 17, 18;
- validation: persons 19, 20, 21, 23, 24, 25, 01, 04;
- final test: persons 22, 02, 03, 05, 06, 07, 08, 09, 10.

## Evidence boundary

KTH and NTU are separate evidence families.

A successful KTH run may establish:

- real RGB bytes were downloaded from the official KTH source;
- YOLO11n pose inference ran on KTH train/validation action subsequences;
- the existing four model families completed the fixed development budget;
- validation macro-F1 and paired FlyWire-minus-rewired development differences were measured.

It may **not** establish:

- an NTU result;
- an NTU topology-specific confirmatory claim;
- a KTH final-test result;
- a generic human behavior-recognition claim beyond this benchmark/protocol;
- direct biological correspondence between FlyWire and human neural circuitry.

The comparison report remains `development_validation_only`, with `final_test_evaluated=false` and `topology_claim_evaluated=false`.

## Official dataset inputs

The workflow uses only these fixed KTH URLs:

- `https://www.csc.kth.se/cvap/actions/walking.zip`
- `https://www.csc.kth.se/cvap/actions/jogging.zip`
- `https://www.csc.kth.se/cvap/actions/running.zip`
- `https://www.csc.kth.se/cvap/actions/boxing.zip`
- `https://www.csc.kth.se/cvap/actions/handwaving.zip`
- `https://www.csc.kth.se/cvap/actions/handclapping.zip`
- `https://www.csc.kth.se/cvap/actions/00sequences.txt`

There is no mirror fallback. A failed official download fails the run.

`00sequences.txt` is downloaded and hashed before extraction. Its parser must accept only the exact KTH grammar needed for the frozen 600-video roster and 2,391 official subsequences. The parsed subject split must equal the frozen split above. Unknown actions, subjects, scenarios, duplicate video keys, overlapping/invalid ranges, or unexpected roster shape fail closed.

## Dataset identity

KTH scientific identity is byte-backed and independent of workflow scheduling.

The coordinator records:

- SHA-256 and size of each of the six official ZIP archives;
- SHA-256 of `00sequences.txt`;
- an ordered manifest of the 600 extracted AVI members with action, subject, scenario, byte size, and SHA-256;
- an ordered manifest of all 2,391 subsequences with parent AVI identity, official frame interval, label, subject, split, and deterministic subsequence ID;
- `dataset_content_hash` over the exact parent AVI byte identities plus the exact official sequence-file hash;
- `split_hash` over dataset identity, frozen class order, frozen subject policy, and all subsequence split assignments;
- `source_manifest_hash` over the URL-free KTH scientific manifest.

Archive URLs are source locators, not scientific identity fields. The downloaded bytes determine identity.

Because this is development evidence rather than a pre-registered confirmatory test, the first successful official-source run may establish the archive/member SHA-256 pins in its evidence artifact. A subsequent protocol-freeze commit may promote those observed hashes into static expected values; until then, every run still records and binds exact bytes before inference.

## Action subsequence semantics

The unit of classification is the official KTH subsequence, not the whole AVI file.

Each AVI contains several action intervals. The extractor decodes a development AVI once and routes frames into the official intervals from `00sequences.txt`. A deterministic sample ID is derived from:

`<video-key>#<1-based-official-range-index>`

For example:

`person11_boxing_d1#01`

Only frames within an official interval are emitted to that sample. Frames outside all official intervals are decoded only as needed to advance the stream and are not supplied to the pose predictor. This prevents idle/inter-sequence frames from entering classifier features while avoiding temporary clipped-video files.

PTS/time-base values remain those of the parent AVI. Each subsequence begins at frame index zero in the emitted pose sidecar but retains increasing source timestamps. The existing timed pose feature encoder is reused unchanged.

## Final-test seal

The KTH final-test subjects are part of source-byte verification but are never decoded by the YOLO/Pose extractor.

For final-test parent AVIs:

- ZIP/member byte identity is verified;
- official sequence ranges are parsed and bound into the source manifest;
- no `decode_video` call is allowed;
- no pose predictor is constructed for that video;
- no final-test geometry/timing sidecar is emitted;
- aggregate validation rejects any final-test pose observation.

A future separately reviewed protocol is required before any KTH final-test decode or classifier evaluation.

## YOLO11 extractor identity

The pose checkpoint is downloaded directly from the fixed Ultralytics official GitHub Release asset:

`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt`

No Ultralytics implicit model-download API is used.

The KTH protocol independently freezes:

- model basename `yolo11n-pose.pt`;
- downloaded checkpoint SHA-256;
- Ultralytics `8.4.146`;
- PyTorch `2.14.0`;
- NumPy `2.4.6`;
- PyAV `15.1.0`;
- OpenCV `5.0.0.93`;
- existing prediction options;
- COCO17 observation schema;
- existing pose feature spec (`confidence_threshold=0.05`, `scale_epsilon=1e-06`).

The checkpoint SHA is frozen by the coordinator and every action shard must independently download the same GitHub asset and match the coordinator SHA before inference.

## Hosted execution topology

Use a dedicated manual/push workflow: `.github/workflows/yolo-flywire-kth-real.yml`.

It supports:

- `workflow_dispatch` for explicit reruns;
- `push` only on branch `research/yolo-flywire-behavior-v0` and only when KTH real-CI source/protocol/workflow files change.

This push trigger exists specifically so the research branch can execute a real-data experiment without merging PR #69 into `master` first.

The workflow has three stages.

### 1. Coordinator

`ubuntu-latest`, <= 60 minutes.

The coordinator:

1. checks out the exact branch head;
2. installs the pinned runtime;
3. downloads and hashes official `00sequences.txt`;
4. validates the frozen split/range grammar;
5. downloads `yolo11n-pose.pt` from the official GitHub Release and hashes it;
6. emits a URL-free protocol/runtime/source-plan record;
7. builds a six-entry action matrix.

The coordinator does not decode KTH video and does not train a classifier.

### 2. Six action extraction shards

One `ubuntu-latest` matrix job per KTH action archive, max-parallel 6, each <= 360 minutes.

Each shard:

1. downloads its one official ZIP archive;
2. verifies it is a regular nonempty ZIP and records archive SHA-256/size;
3. enumerates exactly the expected 100 action/scenario/subject AVI members for that class;
4. validates member names and hashes each extracted AVI;
5. for final-test subjects, records byte evidence and never opens a decoder;
6. for train/validation subjects, decodes each parent AVI once and emits one pose sample per official interval;
7. deletes each extracted AVI immediately after its evidence/extraction is committed;
8. deletes the ZIP before artifact upload;
9. uploads URL-free source evidence plus train/validation pose sidecars only.

There is no reduced-roster fallback, frame subsampling fallback, alternate mirror, or alternate model.

### 3. Aggregate + four-arm development

`ubuntu-latest`, <= 360 minutes.

The aggregate job:

1. downloads coordinator evidence and all six shard artifacts;
2. reconstructs the exact 600-video/2,391-subsequence scientific manifest independent of artifact arrival order;
3. requires complete byte evidence for final-test videos but rejects final-test pose sidecars;
4. assembles train/validation pose sidecars in canonical subsequence order;
5. creates an `IndexedPoseBundle` and then an `IndexedPoseDevelopment` binding without revisiting deleted RGB bytes;
6. downloads/verifies pinned FlyWire connectivity and controls under the existing topology protocol;
7. derives a complete-roster minibatch size that satisfies the fixed 20-epoch / 40-update budget;
8. executes the existing `_preflight_models` / `_execute_arm` four-arm core;
9. publishes development report + KTH source/protocol evidence only.

No second trainer or second graph-comparison implementation is introduced.

## Source/index boundary

Do not extend NTU filename parsing or NTU remote-manifest validation with KTH conditionals.

Create KTH-specific source/extraction validation modules and share only dataset-agnostic pose/index primitives:

- `IndexedPoseBundle`
- `PoseSampleIndex`
- `_frame` / JSONL validation
- `IndexedPoseDevelopment`
- pose feature encoding
- indexed training/evaluation
- four-arm model execution

The KTH indexed loader must consume a verified KTH source record plus an aggregated pose bundle directly. It must not call `verify_rgb_manifest`, `validate_remote_manifest`, or any NTU parser.

## Model/training condition

Class order is frozen as:

1. `boxing`
2. `handclapping`
3. `handwaving`
4. `jogging`
5. `running`
6. `walking`

The comparison keeps the existing five seeds `(7, 11, 19, 23, 31)`, 20 epochs, 40 optimizer updates, and 50,000 parameter ceiling. Development hyperparameters remain explicit workflow inputs with the same defaults as the hosted NTU path unless the parameter ceiling rejects them.

The FlyWire graph, matched rewiring generator, random-graph policy, adjacency handling, checkpoint-selection rule, primary macro-F1 metric, and secondary robustness metrics remain unchanged.

KTH does **not** inherit the NTU confirmatory success rule (`mean FlyWire-rewired >= 0.02 and 4/5 positive`). The KTH report may display paired validation differences but cannot label the original NTU criterion as passed or failed.

## Artifacts and licensing

KTH is public for non-commercial use, but the workflow still avoids republishing the dataset.

Never upload:

- KTH ZIP archives;
- KTH AVI files;
- YOLO checkpoint bytes;
- KTH final-test pose observations.

Allowed artifacts:

- archive/member hashes and frozen KTH source manifests;
- official sequence-file hash and parsed interval records;
- train/validation pose geometry/timing sidecars;
- aggregate/index/binding records;
- FlyWire provenance/controls;
- execution config;
- development report.

The documentation must retain the KTH non-commercial-use note and the Schuldt/Laptev/Caputo ICPR 2004 citation requirement.

## Failure policy

Fail closed on:

- official KTH source unavailable;
- archive corruption or non-ZIP payload;
- wrong archive/member roster;
- malformed or changed official sequence grammar;
- wrong frozen split;
- duplicate video/subsequence identity;
- invalid or overlapping subsequence ranges;
- YOLO checkpoint/runtime drift;
- multiple-person predictor output where the current pose contract forbids selection;
- any final-test decode/pose output;
- missing action shard;
- source hash/spec/schema/encoder mismatch across shards;
- sidecar hash mismatch;
- parameter-budget/update-budget mismatch.

No automatic fallback may convert a failed real run into fixture evidence.

## Testing and verification

Per the user's explicit direction for this repository phase, implementation may be written directly rather than using a RED-first cycle. Verification still must be fresh and exact-head before any completion claim.

Ordinary CI tests use generated fixture AVI/ZIP bytes and must verify:

- KTH `00sequences.txt` split/range parsing;
- canonical 600-video source roster construction;
- deterministic 2,391-subsequence IDs/order;
- test-subject decode prohibition;
- one-parent-video decode feeding several official subsequences;
- KTH aggregate/index binding with no NTU parser dependency;
- workflow uses only official KTH URLs and official Ultralytics GitHub asset;
- no secrets/self-hosted runner;
- no uploaded source AVI/ZIP/model/final-test pose bytes.

The actual KTH real workflow is the acceptance gate for real-data evidence. A successful ordinary fixture CI is not a substitute.

## Acceptance criteria

1. Existing NTU YOLO26 and hosted-NTU YOLO11 protocols remain unchanged in evidence meaning.
2. New protocol `v1-kth-yolo11n-real-ci` has its own dataset/model/source identity.
3. KTH official six archives + `00sequences.txt` are the only RGB/sequence sources.
4. No repository secret or self-hosted runner is required.
5. The workflow can trigger on the research branch without merging PR #69.
6. Classification units are official KTH subsequences, not whole AVI files.
7. KTH final-test subjects are byte-verified but never decoded/evaluated.
8. Development pose output binds into the existing indexed training type with no eager/fallback path.
9. Four-arm execution reuses the existing graph/model/training core.
10. Exact-head ordinary CI is green after implementation.
11. A real `.github/workflows/yolo-flywire-kth-real.yml` run downloads official KTH RGB bytes, downloads official YOLO11n weights, completes the six action shards, runs the four-arm development comparison, and publishes real KTH validation metrics.
12. The final report explicitly labels those metrics as KTH development evidence, not NTU or confirmatory evidence.

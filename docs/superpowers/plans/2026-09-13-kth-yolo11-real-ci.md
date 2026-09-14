# KTH YOLO11 Public Real-CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a no-secret GitHub-hosted real RGB benchmark using official KTH action videos, YOLO11n pose, and the existing indexed/FlyWire four-arm development core, then trigger the real workflow on the research branch.

**Architecture:** KTH-specific code owns official source parsing, archive/member verification, official subsequence slicing, final-test sealing, and pose-shard aggregation. After aggregation, data is converted into the existing `IndexedPoseDevelopment` contract so training/evaluation/model-comparison code is reused unchanged. NTU modules remain benchmark-specific and are not extended with KTH branches.

**Tech Stack:** Python 3.11, PyAV 15.1.0, Ultralytics 8.4.146, PyTorch 2.14.0, NumPy 2.4.6, OpenCV 5.0.0.93, GitHub Actions `ubuntu-latest`.

**Spec:** `docs/superpowers/specs/2026-09-13-kth-yolo11-real-ci-design.md`

## Global Constraints

- Official KTH inputs only: six fixed `https://www.csc.kth.se/cvap/actions/<action>.zip` archives plus `00sequences.txt`; no mirrors or fallback.
- Class order: `boxing, handclapping, handwaving, jogging, running, walking`.
- Official split: train `11..18`; validation `19,20,21,23,24,25,01,04`; final test `22,02,03,05,06,07,08,09,10`.
- Classification units are the 2,391 official frame intervals, not whole AVI files.
- Final-test parent AVIs are byte-verified only; never decoded and never emit pose sidecars.
- YOLO checkpoint source is exactly `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt`.
- Runtime pins: Ultralytics `8.4.146`, PyTorch `2.14.0`, NumPy `2.4.6`, PyAV `15.1.0`, OpenCV `5.0.0.93`.
- Existing five seeds `(7,11,19,23,31)`, 20 epochs, 40 optimizer updates, 50,000 parameter ceiling.
- KTH result is `development_validation_only`; `final_test_evaluated=false`; `topology_claim_evaluated=false`.
- No secret, self-hosted runner, source-media artifact, model artifact, final-test observation artifact, or compatibility fallback.
- Per user direction, implementation is direct with post-implementation verification rather than RED-first TDD.

---

### Task 1: KTH official source contract

**Files:**
- Create: `python/yolo_flywire/kth_source.py`
- Create: `protocols/v1-kth-yolo11n-real-ci.json`
- Create: `tests/test_kth_source.py`

**Interfaces:**
- Produces `parse_sequence_file(text: str) -> KthSequencePlan`.
- Produces `build_kth_source(archives: dict[str, Path], sequence_file: Path) -> dict[str, Any]`.
- `KthSequencePlan` exposes canonical 600 parent-video records, 2,391 official subsequence records, frozen class order/split policy, and deterministic `<video-key>#<NN>` sample IDs.
- Scientific source record contains archive/member/sequence hashes, `dataset_content_hash`, `split_hash`, and `source_manifest_hash` with no filesystem paths.

- [ ] Implement strict parser for the exact subject split and `personNN_action_dM frames a-b,...` grammar from official `00sequences.txt`.
- [ ] Reject unknown action/scenario/subject, duplicate keys, invalid/overlapping ranges, wrong 600-video roster, wrong split, or wrong 2,391-subsequence count.
- [ ] Build archive/member byte identities from six local official ZIPs without decoding media; reject path traversal, duplicate/unexpected members, or non-AVI payload roster.
- [ ] Create protocol `v1-kth-yolo11n-real-ci` with KTH dataset identity fields initially nullable where official downloaded bytes must bind at runtime, independent from NTU success thresholds.
- [ ] Add direct verification tests using generated small ZIP/sequence fixtures plus parser-only checks for the frozen split and deterministic IDs.
- [ ] Run `python -m pytest tests/test_kth_source.py -q` and commit.

### Task 2: KTH action archive YOLO11 extraction

**Files:**
- Create: `python/yolo_flywire/kth_extract.py`
- Create: `tests/test_kth_extract.py`

**Interfaces:**
- Consumes one action ZIP, the parsed KTH sequence plan, one frozen runtime/model descriptor, and local `yolo11n-pose.pt`.
- Produces one action shard with `shard-report.json`, train/validation per-subsequence `observations.jsonl` and `timing.jsonl`, and URL-free byte evidence for all 100 parent videos in that action.
- `extract_action_shard(action, archive, sequence_file, protocol, weights, output)` is the primary API.

- [ ] Reuse hosted YOLO11 predictor validation (`17x3`, person-only, no tracking) without reusing NTU remote-manifest validation.
- [ ] Extract one parent AVI at a time from ZIP, hash it, then delete it before the next parent video.
- [ ] For final-test subjects, record member hash/size and delete AVI without calling decoder or predictor.
- [ ] For train/validation parent AVIs, decode once; route only official frame intervals to predictor; frames between official intervals advance the decoder but never enter predictor/features.
- [ ] Emit each official interval as sample `<video-key>#<NN>`, with sidecar frame indices starting at zero but source PTS/time-base preserved.
- [ ] Record archive SHA/size, member hashes, checkpoint/runtime/spec/schema hashes, and `final_test_decoded=false`.
- [ ] Add fixture tests proving one parent decode feeds multiple intervals and final-test never invokes decoder/predictor.
- [ ] Run `python -m pytest tests/test_kth_extract.py -q` and commit.

### Task 3: Deterministic KTH aggregation and indexed binding

**Files:**
- Create: `python/yolo_flywire/kth_aggregate.py`
- Create: `python/yolo_flywire/kth_pose_index.py`
- Create: `python/yolo_flywire/kth_indexed_development.py`
- Create: `python/yolo_flywire/kth_development.py`
- Create: `tests/test_kth_aggregate.py`
- Create: `tests/test_kth_indexed_development.py`

**Interfaces:**
- `aggregate_kth_shards(sequence_plan, shard_paths, protocol, output) -> dict[str, Any]` builds a canonical train/validation pose bundle and complete KTH byte-evidence manifest.
- `load_kth_indexed_pose_development(...) -> IndexedPoseDevelopment` returns the existing strict source type.
- `run_kth_development(...) -> dict[str, Any]` invokes existing `_graph_inputs`, `_preflight_models`, and `_execute_arm` rather than implementing a second trainer.

- [ ] Require exactly six action shards with common model/runtime/schema identities and complete 600-parent/2,391-subsequence coverage.
- [ ] Reject final-test pose directories, missing/duplicate samples, source hash drift, sidecar hash drift, and artifact-order dependence.
- [ ] Concatenate development sidecars in canonical official-subsequence order and create manifest hashes.
- [ ] Implement KTH-specific pose index validation using dataset-agnostic `_frame`, `ByteSpan`, `PoseSampleIndex`, and `IndexedPoseBundle`; do not call NTU parsers.
- [ ] Bind the KTH index through existing `IndexedPoseDevelopment` feature/batch semantics.
- [ ] Implement development orchestration that uses the existing graph/control validation and four-arm execution core and never applies the NTU confirmatory threshold.
- [ ] Add post-implementation fixture tests for aggregation, final-test sealing, order independence, strict existing indexed type, and four-arm report evidence scope.
- [ ] Run focused KTH tests and commit.

### Task 4: Public KTH real workflow and actual run

**Files:**
- Create: `.github/workflows/yolo-flywire-kth-real.yml`
- Create: `tests/test_kth_real_workflow.py`
- Create: `docs/kth-yolo11-real-ci.md`

**Interfaces:**
- Push trigger only for branch `research/yolo-flywire-behavior-v0` and KTH workflow/source/protocol files; also supports `workflow_dispatch`.
- Coordinator output: canonical sequence plan, checkpoint hash/runtime descriptor, six-action matrix.
- Action artifacts: URL-free byte evidence + train/validation pose sidecars only.
- Final artifact: KTH source manifest, aggregate/index/binding identities, FlyWire provenance, execution config, development report.

- [ ] Add coordinator job on `ubuntu-latest`: download/validate/hash `00sequences.txt`; download YOLO11n from official GitHub Release; emit six-action matrix.
- [ ] Add six matrix extraction jobs: each downloads one official KTH ZIP, runs `kth_extract`, deletes ZIP/AVI/model bytes before artifact upload.
- [ ] Add aggregate/comparison job: download all six artifacts, aggregate, download/verify FlyWire source and controls, derive complete-roster batch size for exactly 40 updates, run `kth_development`.
- [ ] Upload evidence only; forbid KTH ZIP/AVI, checkpoint, and final-test observations.
- [ ] Add workflow contract tests enforcing official KTH URLs, official YOLO GitHub asset, `ubuntu-latest`, no secret/self-hosted, branch push trigger, six action matrix, and no final-test selector.
- [ ] Document KTH non-commercial use and ICPR 2004 citation requirement.
- [ ] Run full ordinary exact-head CI; fix only observed regressions.
- [ ] Let the research-branch push trigger start the real KTH workflow; inspect coordinator, six action shards, aggregation and four-arm report.
- [ ] Record real run ID, exact head, archive/member hashes and real development metrics on Issue #68. Do not claim KTH final-test or NTU evidence.

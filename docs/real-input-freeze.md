# Byte-backed real-input freeze

Tracked by #68. This gate binds the already-reviewed Phase 2 experiment design to one legally acquired NTU RGB+D 120 RGB corpus and one trusted `yolo26n-pose.pt` checkpoint. It inventories and hashes inputs only. It does **not** decode final-test media, construct a YOLO predictor, train a classifier, compare model arms, or authorize final-test execution.

## Prerequisites

Use the repository's frozen real protocol `protocols/v0-real-ntu120-preflight.json`. The design is already fixed to the 10-class RGB-only NTU120 task, official X-Sub120 outer split, deterministic validation subjects drawn from the training side, FlyWire FAFB-v783 topology/control fingerprints, seeds `7, 11, 19, 23, 31`, and the 20-epoch / 40-update / 50,000-parameter budget.

The extractor package version is fixed to Ultralytics `8.4.146`, and the feature policy is fixed to `confidence_threshold=0.05`, `scale_epsilon=1e-6`. The checkpoint used by the freeze/extraction commands must be a local, nonempty, non-symlink regular file named exactly `yolo26n-pose.pt`. The local commands never substitute another checkpoint. The dedicated manual CI workflow may first obtain that exact model family through the pinned Ultralytics package and then freezes the downloaded bytes by SHA-256 before any classifier execution.

A legally acquired NTU RGB+D 120 RGB corpus is required. The repository does not redistribute NTU media and does not contain an anonymous NTU download path.

## Step 1: freeze the NTU byte inventory

```bash
python -m yolo_flywire.ntu_io freeze \
  --root "$NTU_RGB" \
  --output ntu120-rgb-inventory.json
```

The inventory records every selected train, validation, and final-test RGB file with byte size and SHA-256, plus aggregate `dataset_content_hash` and `split_hash`. Final-test files are hashed for identity, but this operation does not decode them or expose them to model selection.

The inventory is an exclusive output: an existing destination is not overwritten. Keep this file with the experiment evidence and treat changes to the underlying RGB tree as a new input identity.

## Step 2: bind the real protocol to local bytes and runtime

```bash
python -m yolo_flywire.real_freeze freeze \
  --protocol protocols/v0-real-ntu120-preflight.json \
  --root "$NTU_RGB" \
  --inventory ntu120-rgb-inventory.json \
  --weights /trusted/yolo26n-pose.pt \
  --output v0-real-ntu120-frozen.json
```

`real_freeze` re-verifies the RGB inventory against the current bytes and fails closed on any mismatch. It also verifies the exact checkpoint basename, regular-file/symlink policy, nonempty contents, and byte SHA-256. The installed runtime must report Ultralytics `8.4.146`; runtime identities for PyTorch, NumPy, PyAV, and OpenCV are captured in the extraction specification.

The output derives and binds:

- `dataset_content_hash` from the verified NTU RGB bytes;
- `split_hash` from the exact selected roster and deterministic split assignment;
- `input_inventory_hash` over the complete verified inventory;
- `yolo_weights_sha256` from the local checkpoint bytes;
- `observation_schema_hash` from the frozen COCO17 timed observation schema;
- `pose_encoder_hash` from the 121-dimensional timed pose encoder and explicit feature policy;
- `extraction_spec` and `extraction_spec_hash`, including checkpoint identity, exact runtime versions, frozen predictor options, and decoder policy.

The snapshot also records:

```text
evidence_scope = real_input_frozen_no_final_test_execution
final_test_decoded = false
classifier_evaluated = false
```

An existing non-null derived pin in the input protocol is accepted only when it exactly matches the locally verified value. A conflict is an error; the tool never silently repairs or replaces a reviewed pin. The output is canonical JSON and is never overwritten.

## Manual authorized GitHub Actions execution

`.github/workflows/yolo-flywire-real.yml` is the repository's heavyweight real-development workflow. It is `workflow_dispatch` only and deliberately does not run on push or pull requests. The workflow requires a dedicated Linux x64 self-hosted runner carrying the label `yolo-flywire-real` and at least 350 GiB free under `/var/tmp`.

The runner must have `curl`, `unzip`, `df`, and `sha256sum`. Configure these repository secrets from an account that is already authorized to obtain NTU RGB+D 120:

- `NTU120_RGB_ARCHIVE_URL_1`
- `NTU120_RGB_ARCHIVE_URL_2`

Each secret must be a direct authorized ZIP download URL for one of the two official RGB archive parts. The workflow does not manufacture credentials, scrape an authenticated portal, or bypass the dataset release agreement. Missing/expired secrets fail closed.

The workflow then:

1. downloads and unpacks both authorized NTU RGB archive parts into a private run directory;
2. installs the exact exercised extraction runtime (`ultralytics==8.4.146`, `av==15.1.0`, `torch==2.14.0`, `numpy==2.4.6`, `opencv-python==5.0.0.93`);
3. asks the pinned Ultralytics package to materialize `yolo26n-pose.pt` and immediately records its SHA-256;
4. freezes the NTU inventory and byte-backed real protocol;
5. downloads FlyWire connectivity from the frozen commit, generates the full five matched controls, and verifies their declared fingerprints/invariants;
6. derives the unique full-roster batch size that yields exactly two batches per epoch, preserving the frozen 20 epochs / 40 optimizer updates, while recording the explicit learning rate/model widths supplied to the manual dispatch;
7. runs `python -m yolo_flywire.real_development run`, which performs development extraction, indexed four-arm optimization/evaluation, and validation-only reporting;
8. uploads only compact evidence (`frozen-protocol`, execution config, extraction manifest, development report, graph verification records, and content hashes), never the licensed RGB media, pose observation/timing streams, or model checkpoint;
9. removes the downloaded and derived licensed run directory after evidence publication.

The long real-development job and the evidence-publication job are separate. This gives the publication job a fresh GitHub token even if the data acquisition/extraction job runs for a long time on the self-hosted runner. The `yolo-flywire-real` label is intended for a dedicated runner so the persistent `/var/tmp/yolo-flywire-real-<run-id>` handoff remains on the same machine.

The manual inputs `learning_rate`, `gru_hidden_dim`, and `graph_node_dim` are development conditions, not hidden defaults. They are written into `execution-config.json` and hashed into the final development report. Seeds and the epoch/update/parameter budget cannot be changed by workflow inputs.

## What this gate does not prove

A successful freeze establishes reproducible **input provenance**, not behavior-recognition performance. It does not establish that FlyWire improves accuracy, robustness, throughput, or latency. It does not open the sealed final test.

The next permitted execution stage is development-only:

1. extract the train/validation RGB clips with the exact frozen extraction specification;
2. bind them through `IndexedPoseDevelopment`;
3. run the four arms — GRU, random sparse, matched rewired, FlyWire — for all five fixed seeds using 20 epochs / 40 optimizer updates and validation-only checkpoint selection;
4. report development macro-F1 and the predeclared paired FlyWire-minus-rewired result without changing topology, threshold, seeds, or budget.

The manual real workflow executes exactly this development stage. Only after its real-input snapshot, extraction, indexed runner, model selection, and all provenance checks have been reviewed may a separate sealed confirmatory execution command be introduced or run. Confirmatory outcomes must be reported whether positive, null, or negative.

## Failure policy

There is no compatibility fallback. Freeze fails on changed NTU bytes, a changed inventory, wrong or empty checkpoint, checkpoint symlink, wrong Ultralytics package version, malformed feature policy, stale graph/control design, or conflicting pre-existing derived hashes. It also refuses to overwrite a previous frozen snapshot.

The manual CI additionally fails on absent authorization URLs, insufficient disk, failed archive transfer/extraction, runtime-version drift, graph/control mismatch, a complete-roster batch size incompatible with the frozen update budget, or any downstream real-development provenance failure.

Hashes protect identity and accidental/stale substitutions; they are not a sandbox against an actor who can replace the code and all independent pins. Run the freeze and real workflow on a trusted self-hosted runner with access restricted to the authorized experiment operators.

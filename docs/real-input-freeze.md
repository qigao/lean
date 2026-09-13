# Byte-backed real-input freeze

Tracked by #68. This gate binds the already-reviewed Phase 2 experiment design to one legally acquired local NTU RGB+D 120 RGB corpus and one trusted local `yolo26n-pose.pt` checkpoint. It inventories and hashes inputs only. It does **not** decode final-test media, construct a YOLO predictor, train a classifier, compare model arms, or authorize final-test execution.

## Prerequisites

Use the repository's frozen real protocol `protocols/v0-real-ntu120-preflight.json`. The design is already fixed to the 10-class RGB-only NTU120 task, official X-Sub120 outer split, deterministic validation subjects drawn from the training side, FlyWire FAFB-v783 topology/control fingerprints, seeds `7, 11, 19, 23, 31`, and the 20-epoch / 40-update / 50,000-parameter budget.

The extractor package version is fixed to Ultralytics `8.4.146`, and the feature policy is fixed to `confidence_threshold=0.05`, `scale_epsilon=1e-6`. The checkpoint must be a local, nonempty, non-symlink regular file named exactly `yolo26n-pose.pt`. The repository does not download or substitute a model checkpoint.

A legally acquired local NTU RGB+D 120 RGB tree is required. The repository does not redistribute NTU media.

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

## What this gate does not prove

A successful freeze establishes reproducible **input provenance**, not behavior-recognition performance. It does not establish that FlyWire improves accuracy, robustness, throughput, or latency. It does not open the sealed final test.

The next permitted execution stage is development-only:

1. extract the train/validation RGB clips with the exact frozen extraction specification;
2. bind them through `IndexedPoseDevelopment`;
3. run the four arms — GRU, random sparse, matched rewired, FlyWire — for all five fixed seeds using 20 epochs / 40 optimizer updates and validation-only checkpoint selection;
4. report development macro-F1 and the predeclared paired FlyWire-minus-rewired result without changing topology, threshold, seeds, or budget.

Only after the real-input snapshot, development extraction, indexed runner, model selection, and all provenance checks have been reviewed may a separate sealed final-test execution command be introduced or run. Final-test outcomes must be reported whether positive, null, or negative.

## Failure policy

There is no compatibility fallback. Freeze fails on changed NTU bytes, a changed inventory, wrong or empty checkpoint, checkpoint symlink, wrong Ultralytics package version, malformed feature policy, stale graph/control design, or conflicting pre-existing derived hashes. It also refuses to overwrite a previous frozen snapshot.

Hashes protect identity and accidental/stale substitutions; they are not a sandbox against an actor who can replace the code and all independent pins. Run the freeze in a trusted process with the source data and checkpoint kept read-only for the experiment.

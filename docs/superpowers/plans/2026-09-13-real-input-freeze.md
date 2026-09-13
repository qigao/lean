# Real Input Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed command that turns the already-frozen Phase 2 preflight plus verified local NTU RGB bytes and local `yolo26n-pose.pt` bytes into one immutable real-input protocol snapshot, without decoding final-test media or running any classifier.

**Architecture:** Reuse `ntu_io.verify_rgb_manifest` for byte-backed dataset/split identity, `pose_backend.runtime_versions` plus `pose_extract.ExtractionSpec`/`_check_weights` for exact extractor runtime and model-byte identity, `pose_extract._schema` for the point-set schema, and `pose_features.PoseFeatureSpec`/`pose_encoder_hash` for the classifier encoder. A new `real_freeze.py` module owns only protocol binding/publication; it never decodes video, loads YOLO, trains models, or authorizes final-test execution. `cli._validate_frozen_protocol` will additionally require the encoder/extraction hashes for real topology protocols.

**Tech Stack:** Python 3.11, hashlib/JSON/pathlib, existing PyTorch/Ultralytics/PyAV runtime adapters, pytest, GitHub Actions.

**Spec:** `docs/phase2-real-data-freeze.md`

## Global Constraints

- Dataset family remains NTU RGB+D 120, RGB-only, 10-class subset A8/A9/A22/A23/A26/A27/A31/A34/A35/A36.
- Official X-Sub120 is the outer boundary; validation remains the existing deterministic subject-hash subset of the training side.
- Final-test bytes may be hashed/inventoried but must not be decoded or used for model/graph/threshold selection.
- Extractor family remains local `yolo26n-pose.pt`; no download, alternate checkpoint, model fallback, ByteTrack, depth, RGB embedding, or NTU skeleton classifier input.
- Pin Ultralytics package version to the already exercised CI version `8.4.146`; a different installed version fails closed.
- Pose feature policy remains `confidence_threshold=0.05`, `scale_epsilon=1e-6`, 121-dimensional timed COCO17 encoder.
- FlyWire release, source bytes, topology selection, rewiring algorithm, control fingerprints, seeds `(7,11,19,23,31)`, 20 epochs / 40 updates / 50,000 parameter ceiling, macro-F1 primary metric, and +0.02 / 4-of-5 success rule do not change.
- No protocol output may overwrite an existing file or symlink.
- No generated freeze snapshot is final-test authorization; sealed confirmatory execution remains a later gate.

---

### Task 1: RED contracts for real-input protocol freezing

**Files:**
- Create: `tests/test_real_freeze.py`
- Modify: `tests/test_v0_protocol.py`
- Modify: `protocols/v0-real-ntu120-preflight.json`
- Modify: `protocols/v0-real-template.json`

**Interfaces:**
- Consumes: existing `build_rgb_manifest`, `verify_rgb_manifest`, `ExtractionSpec`, `_check_weights`, `_schema`, `runtime_versions`, `PoseFeatureSpec`, `pose_encoder_hash`.
- Produces requirement for `yolo_flywire.real_freeze.freeze_real_inputs(...)` and CLI `python -m yolo_flywire.real_freeze freeze ...`.

- [ ] **Step 1: Pin design-only extractor choices in both real protocols**

Set `ultralytics_package_version` to `"8.4.146"` and add:

```json
"pose_feature_spec": {
  "confidence_threshold": 0.05,
  "scale_epsilon": 1e-06
},
"pose_encoder_hash": null,
"extraction_spec_hash": null
```

Keep dataset, split, weights, schema, encoder, and extraction hashes null until verified local inputs are bound.

- [ ] **Step 2: Write failing freeze API/CLI tests**

Use generated NTU-style RGB fixture files and a dummy nonempty local `yolo26n-pose.pt`. Monkeypatch only `runtime_versions` to an explicit version map; never mock dataset-byte verification or weight hashing. Require the result to:

```python
assert frozen["dataset_content_hash"] == inventory["dataset_content_hash"]
assert frozen["split_hash"] == inventory["split_hash"]
assert frozen["ultralytics_package_version"] == "8.4.146"
assert frozen["yolo_weights_sha256"] == sha256(weights.read_bytes()).hexdigest()
assert frozen["observation_schema_hash"] == _hash_json(_schema())
assert frozen["pose_encoder_hash"] == pose_encoder_hash(PoseFeatureSpec(**frozen["pose_feature_spec"]))
assert frozen["extraction_spec_hash"] == _hash_json(frozen["extraction_spec"])
assert frozen["final_test_used_for_selection"] is False
assert frozen["evidence_scope"] == "real_input_frozen_no_final_test_execution"
```

Also require canonical deterministic output, no overwrite, rejection of changed RGB bytes/manifest, wrong weights basename, empty/mutated weights, wrong installed Ultralytics version, malformed/null feature spec, and any attempt to pass a protocol whose frozen graph/budget/seed contract differs from the preflight.

- [ ] **Step 3: Prove no forbidden execution occurs**

Monkeypatch `pose_extract.decode_video`, `pose_backend.load_predictor`, `pose_comparison.run_pose_comparison`, and training entry points to raise. Freeze must still succeed because it inventories/hashes only.

- [ ] **Step 4: Run exact RED**

Run:

```bash
python -m pytest tests/test_real_freeze.py tests/test_v0_protocol.py -q
```

Expected: new freeze tests fail because `yolo_flywire.real_freeze`/new real-protocol gate does not yet exist; existing unrelated tests remain green.

- [ ] **Step 5: Commit RED only**

```bash
git add tests/test_real_freeze.py tests/test_v0_protocol.py protocols/v0-real-ntu120-preflight.json protocols/v0-real-template.json
git commit -m "test: require byte-backed real input protocol freeze"
```

---

### Task 2: Implement byte-backed freeze without final-test execution

**Files:**
- Create: `python/yolo_flywire/real_freeze.py`
- Modify: `python/yolo_flywire/cli.py`

**Interfaces:**
- Produces:

```python
def freeze_real_inputs(
    protocol: dict[str, Any], *, root: str | Path, inventory: dict[str, Any],
    weights: str | Path,
) -> dict[str, Any]: ...
```

CLI:

```text
python -m yolo_flywire.real_freeze freeze \
  --protocol protocols/v0-real-ntu120-preflight.json \
  --root /path/to/ntu-rgb \
  --inventory ntu120-rgb-inventory.json \
  --weights /path/to/yolo26n-pose.pt \
  --output frozen-real-protocol.json
```

- [ ] **Step 1: Validate the base protocol before touching dataset/weight bytes**

Require exact Phase 2 design constants: dataset id, ordered task labels, split rule, `yolo_version`, `yolo_weights`, pinned Ultralytics `8.4.146`, explicit pose feature spec, FlyWire release/source/commit/connectivity hashes, graph stats/fingerprint, rewiring algorithm/fingerprints, five seeds, 20/40/50000 budget, metrics, threshold, and `final_test_used_for_selection=False`. Reject already-populated mutable real-input fields unless they exactly match the derived value; never silently repair conflicting pins.

- [ ] **Step 2: Verify local RGB bytes and split identity**

Call `verify_rgb_manifest(root, inventory)` and use only its returned `dataset_content_hash` / `split_hash`. Record an `input_inventory_hash = _hash_json(verified_inventory)` so the frozen protocol commits to the complete selected-file manifest as well as aggregate content/split hashes.

- [ ] **Step 3: Bind exact weight bytes and extraction runtime**

Call `runtime_versions()`. Require `versions["ultralytics"] == protocol["ultralytics_package_version"]`. Hash the local nonempty regular `yolo26n-pose.pt`, construct `ExtractionSpec` with that hash and all five exact runtime versions, then call `_check_weights` to reuse symlink/name/hash gates. Store `extraction_spec = spec.descriptor()` and `extraction_spec_hash = _hash_json(extraction_spec)`.

- [ ] **Step 4: Bind observation schema and classifier encoder**

Compute `observation_schema_hash = _hash_json(_schema())`. Construct `PoseFeatureSpec(**protocol["pose_feature_spec"])`, compute `pose_encoder_hash(spec)`, and retain the explicit feature parameters in the protocol.

- [ ] **Step 5: Publish a fresh immutable snapshot**

Return a deep JSON-compatible copy of the base protocol with the derived fields populated, set `evidence_scope = "real_input_frozen_no_final_test_execution"`, and add `final_test_decoded = False`, `classifier_evaluated = False`, plus a freeze record containing inventory/extraction/encoder identities. CLI output uses the existing exclusive-publish pattern and refuses overwrite/symlinks.

- [ ] **Step 6: Extend real protocol execution validation**

For non-synthetic topology protocols, `cli._validate_frozen_protocol` must additionally require valid non-null `pose_encoder_hash` and `extraction_spec_hash` before even protocol-validation output can be emitted.

- [ ] **Step 7: Run focused GREEN**

```bash
python -m pytest tests/test_real_freeze.py tests/test_v0_protocol.py -q
```

Expected: all focused tests pass with zero warnings/errors from the new code.

- [ ] **Step 8: Commit implementation**

```bash
git add python/yolo_flywire/real_freeze.py python/yolo_flywire/cli.py
git commit -m "feat: freeze verified real input protocol"
```

---

### Task 3: Document and verify the new Phase 2 gate

**Files:**
- Modify: `docs/phase2-real-data-freeze.md`
- Create: `docs/real-input-freeze.md`

**Interfaces:**
- Consumes: `freeze_real_inputs`, `ntu_io freeze/verify`, existing development extraction/indexed-runner pipeline.
- Produces: operator sequence for legally acquired NTU media and local trusted weights.

- [ ] **Step 1: Document the exact operator flow**

Document:

```bash
python -m yolo_flywire.ntu_io freeze --root "$NTU_RGB" --output ntu120-rgb-inventory.json
python -m yolo_flywire.real_freeze freeze \
  --protocol protocols/v0-real-ntu120-preflight.json \
  --root "$NTU_RGB" --inventory ntu120-rgb-inventory.json \
  --weights /trusted/yolo26n-pose.pt --output v0-real-ntu120-frozen.json
```

State that inventory hashes final-test bytes for identity but does not decode them, and that the frozen snapshot still does not authorize final-test execution.

- [ ] **Step 2: Clarify remaining gate after freeze**

After a real snapshot exists, development extraction must be executed with the exact extraction spec, the bound indexed four-arm runner must complete five-seed 20-epoch/40-update validation-only development execution, and only then may a separately reviewed final-test command be introduced/executed.

- [ ] **Step 3: Run full verification**

```bash
python -m pytest tests -q
lake build
```

Then require GitHub Actions exact-head success for Python, Lean, pose integration, and FlyWire provenance. Read complete failing logs if any; do not patch speculatively.

- [ ] **Step 4: Record evidence on issue #68**

Comment with RED and GREEN run IDs, exact head, focused/full counts, what fields the freeze command now derives, and the explicit statement that no real NTU data or final-test behavior result has been produced unless a legally acquired local dataset and trusted local weights were actually supplied.

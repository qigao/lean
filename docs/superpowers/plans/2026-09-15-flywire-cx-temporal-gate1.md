# FlyWire CX Temporal Gate 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, reproducible Gate 1 experiment that compares a fixed FAFB v783 central-complex LIF reservoir with degree-preserving and block-preserving null graphs on temporal-state metrics from native NTU skeleton sequences.

**Architecture:** New CX-specific modules sit beside the existing visual-graph pipeline without changing Issue #68 contracts. Codex static files are converted into a frozen neuron-level CX artifact with biological input/core/output roles; matched nulls preserve declared graph statistics; a signed-event encoder drives only biological input nodes; a fixed recurrent LIF core is read only from biological output nodes; evaluation reports early prediction, retention, recovery, and spike efficiency.

**Tech Stack:** Python 3.11, NumPy >=1.26, PyTorch >=2.3, pytest >=8.0, stdlib `csv`/`gzip`/`hashlib`/`json`.

**Spec:** `docs/superpowers/specs/2026-09-15-flywire-cx-temporal-gate1-design.md`

## Global Constraints

- Repository: `qigao/lean`; implementation branch: `research/flywire-cx-temporal-gate1`.
- Do not change Issue #68 visual-system selection, fingerprints, rewiring semantics, GRU semantics, or `GraphRecurrentClassifier` semantics.
- FAFB source is v783 Codex static downloads; required products are `consolidated_cell_types` and `connections_princeton`.
- Aggregate connection rows by directed neuron pair before applying the fixed 5-synapse minimum.
- Frozen CX family rule for Gate 1 V1: INPUT families `ER`, `ExR`, `PFN`; CORE families `EPG`, `PEG`, `PEN`, `Delta7`, `hDelta`, `vDelta`; OUTPUT family `PFL`.
- Family matching is case-sensitive prefix matching on the consolidated primary type after stripping surrounding whitespace; a type must match exactly one declared family.
- Neurotransmitter sign map for R0: `ACH`, `OCT`, `SER`, `DA` => `+1`; `GABA`, `GLUT` => `-1`; any other transmitter on a retained recurrent edge is a hard error.
- Recurrent weights are fixed in R0; only input projection, readout, and explicitly declared global gains are trainable.
- Seeds are exactly `7, 11, 19, 23, 31`.
- Observation ratios are exactly `0.10, 0.20, 0.40, 0.60, 0.80, 1.00`.
- Gate 1A success: mean paired Real-minus-Degree early-AUC >= 0.02, positive in >=4/5 seeds, plus either mean paired retention-AUC >= 0.05 positive in >=4/5 or mean paired recovery-rate >= 0.05 positive in >=4/5.
- Gate 1B uses the same thresholds for Real-minus-Block and is reported separately.
- No final-test execution until every source/data/graph/null/encoder/dynamics fingerprint is non-null and verified against bytes.
- No compatibility fallback to visual graphs, reduced swap budgets, all-node input injection, all-node mean readout, or unknown neurotransmitter signs.

---

### Task 1: Freeze the CX Gate 1 protocol contract

**Files:**
- Create: `python/yolo_flywire/cx_protocol.py`
- Create: `protocols/v2-cx-temporal-gate1-preflight.json`
- Test: `tests/test_cx_protocol.py`

**Interfaces:**
- Produces: `load_cx_protocol(path: str | Path) -> CxProtocol`
- Produces: `validate_cx_protocol_dict(value: dict[str, Any]) -> CxProtocol`
- `CxProtocol` exposes immutable fields used by every later task, including families, sign map, threshold, seeds, ratios, null policy, metric thresholds, and provenance hashes.

- [ ] **Step 1: Write the failing protocol tests**

```python
from pathlib import Path
import json
import pytest

from yolo_flywire.cx_protocol import load_cx_protocol, validate_cx_protocol_dict

ROOT = Path(__file__).resolve().parents[1]


def test_preflight_freezes_gate1_choices_without_claiming_final_readiness():
    protocol = load_cx_protocol(ROOT / "protocols/v2-cx-temporal-gate1-preflight.json")
    assert protocol.dataset == "FAFB"
    assert protocol.release == "v783"
    assert protocol.connection_threshold == 5
    assert protocol.input_families == ("ER", "ExR", "PFN")
    assert protocol.core_families == ("EPG", "PEG", "PEN", "Delta7", "hDelta", "vDelta")
    assert protocol.output_families == ("PFL",)
    assert protocol.seeds == (7, 11, 19, 23, 31)
    assert protocol.observation_ratios == (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
    assert protocol.final_test_enabled is False
    assert protocol.cx_artifact_fingerprint is None


def test_protocol_rejects_role_overlap():
    raw = json.loads((ROOT / "protocols/v2-cx-temporal-gate1-preflight.json").read_text())
    raw["core_families"].append("PFN")
    with pytest.raises(ValueError, match="family roles must be disjoint"):
        validate_cx_protocol_dict(raw)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_cx_protocol.py -q`

Expected: import failure because `yolo_flywire.cx_protocol` does not exist.

- [ ] **Step 3: Implement the immutable protocol dataclass and strict validator**

```python
@dataclass(frozen=True)
class CxProtocol:
    protocol_id: str
    dataset: str
    release: str
    cell_types_product: str
    connections_product: str
    connection_threshold: int
    input_families: tuple[str, ...]
    core_families: tuple[str, ...]
    output_families: tuple[str, ...]
    nt_signs: tuple[tuple[str, int], ...]
    seeds: tuple[int, ...]
    observation_ratios: tuple[float, ...]
    early_auc_effect: float
    temporal_effect: float
    positive_seed_count: int
    final_test_enabled: bool
    source_hashes: tuple[tuple[str, str | None], ...]
    cx_artifact_fingerprint: str | None
    degree_null_fingerprints: tuple[tuple[str, str], ...] | None
    block_null_fingerprints: tuple[tuple[str, str], ...] | None
```

Validator requirements: exact dataset/release/products, exact seeds/ratios, disjoint family roles, exact sign-map keys/values, lowercase 64-hex digests when present, and all final-test fingerprints non-null before `final_test_enabled=True` is admissible.

- [ ] **Step 4: Add the preflight JSON with null provenance fields**

The JSON must contain exact constants from Global Constraints and explicit null values for source/artifact/null/data fingerprints not yet measured. Set `final_test_enabled` to `false`.

- [ ] **Step 5: Run focused and existing protocol tests**

Run: `python -m pytest tests/test_cx_protocol.py tests/test_protocol_manifest.py tests/test_v0_protocol.py -q`

Expected: PASS; existing V0 protocol behavior unchanged.

- [ ] **Step 6: Commit**

```bash
git add python/yolo_flywire/cx_protocol.py protocols/v2-cx-temporal-gate1-preflight.json tests/test_cx_protocol.py
git commit -m "test: freeze CX temporal Gate 1 protocol"
```

---

### Task 2: Build a deterministic FAFB v783 CX artifact

**Files:**
- Create: `python/yolo_flywire/cx_artifact.py`
- Test: `tests/test_cx_artifact.py`
- Create: `tests/fixtures/cx/consolidated_cell_types.csv`
- Create: `tests/fixtures/cx/connections_princeton.csv`

**Interfaces:**
- Consumes: `CxProtocol`
- Produces: `CxNode(root_id: int, primary_type: str, family: str, role: str)`
- Produces: `CxArtifact(nodes: tuple[CxNode, ...], graph: DirectedGraph, input_indices: tuple[int, ...], core_indices: tuple[int, ...], output_indices: tuple[int, ...], edge_signs: tuple[int, ...], fingerprint: str)`
- Produces: `build_cx_artifact(cell_types_path, connections_path, protocol) -> CxArtifact`
- Produces: `write_cx_artifact(artifact, output_dir) -> dict[str, str]`

- [ ] **Step 1: Write failing fixture tests for family selection and pair aggregation**

Use fixture cell-type columns exactly:

```text
root_id,primary_type,additional_type(s)
```

Use fixture connection columns exactly:

```text
pre_root_id,post_root_id,neuropil,syn_count,nt_type
```

Test that two rows `1->2` with 3 PB synapses and 4 EB synapses aggregate to 7 and survive the threshold, while a total of 4 does not.

```python
def test_build_cx_artifact_aggregates_pair_before_threshold(protocol):
    artifact = build_cx_artifact(CELL_TYPES, CONNECTIONS, protocol)
    assert artifact.graph.num_nodes == 8
    edges = {(s, d): w for s, d, w in zip(artifact.graph.src, artifact.graph.dst, artifact.graph.weight)}
    assert 7.0 in edges.values()
    assert 4.0 not in edges.values()
```

- [ ] **Step 2: Add RED tests for role routing and transmitter consistency**

```python
def test_roles_are_family_driven_and_disjoint(protocol):
    artifact = build_cx_artifact(CELL_TYPES, CONNECTIONS, protocol)
    assert artifact.input_indices
    assert artifact.core_indices
    assert artifact.output_indices
    assert not (set(artifact.input_indices) & set(artifact.output_indices))


def test_unknown_retained_transmitter_fails(protocol, tmp_path):
    bad = tmp_path / "connections.csv"
    bad.write_text("pre_root_id,post_root_id,neuropil,syn_count,nt_type\n1,2,PB,6,UNKNOWN\n")
    with pytest.raises(ValueError, match="transmitter"):
        build_cx_artifact(CELL_TYPES, bad, protocol)
```

- [ ] **Step 3: Implement strict CSV loaders and canonical family matcher**

```python
def _family(primary_type: str, protocol: CxProtocol) -> tuple[str, str] | None:
    matches = [family for family in protocol.all_families if primary_type.startswith(family)]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"ambiguous CX family for {primary_type!r}")
    family = matches[0]
    return family, protocol.role_for_family(family)
```

Reject duplicate `root_id`, empty primary types for selected neurons, malformed integer IDs/counts, negative synapse counts, and selected edges referencing missing selected-node metadata.

- [ ] **Step 4: Aggregate connection rows and build signed metadata**

Aggregate `syn_count` by `(pre_root_id, post_root_id)` before thresholding. Require all rows contributing to one retained pair to agree on `nt_type`; otherwise fail. Preserve unsigned synapse count in `DirectedGraph.weight`; store sign in `edge_signs` so graph fingerprints remain raw-connectivity fingerprints and dynamics can apply sign explicitly.

- [ ] **Step 5: Implement canonical artifact serialization**

Write only deterministic UTF-8/JSON/CSV outputs:

```text
nodes.csv
edges.csv
roles.json
metadata.json
```

`metadata.json` includes source SHA-256, node/edge counts, exact matched type list, exact input/core/output root IDs, and artifact fingerprint computed from canonical serialized metadata plus `graph_fingerprint(graph)`.

- [ ] **Step 6: Run focused and legacy graph tests**

Run: `python -m pytest tests/test_cx_artifact.py tests/test_flywire.py tests/test_flywire_type_graph.py tests/test_graphs.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/yolo_flywire/cx_artifact.py tests/test_cx_artifact.py tests/fixtures/cx
git commit -m "feat: build deterministic FlyWire CX artifact"
```

---

### Task 3: Add degree-preserving and block-preserving CX null graphs

**Files:**
- Create: `python/yolo_flywire/cx_nulls.py`
- Test: `tests/test_cx_nulls.py`

**Interfaces:**
- Consumes: `CxArtifact`
- Produces: `rewire_cx_degree_preserving(artifact: CxArtifact, seed: int, swaps: int) -> CxArtifact`
- Produces: `rewire_cx_block_preserving(artifact: CxArtifact, seed: int, swaps: int) -> CxArtifact`
- Produces: `cx_null_report(real: CxArtifact, null: CxArtifact) -> dict[str, Any]`

- [ ] **Step 1: Write invariant tests for the degree null**

Assert equality of node count, edge count, per-node in/out-degree sequence, global unsigned weight multiset, role assignments, node metadata, and sign multiset. Assert fingerprint differs from Real for nonzero swap budget.

- [ ] **Step 2: Write invariant tests for the block null**

Define each node block as `(role, family)`. Assert exact equality of the matrix:

```python
Counter((block[src], block[dst]) for src, dst in edges)
```

before and after rewiring, in addition to the Task 3 degree-null invariants.

- [ ] **Step 3: Implement a shared directed-swap engine without modifying `graphs.rewire_degree_preserving`**

The CX engine carries edge weights and signs with edge slots while rewiring destinations. For block mode, accept a proposal only when both replacement edges preserve their original `(source_block, target_block)` categories. Reject new self-edges unless the original artifact contains and the protocol explicitly allows the same self-edge policy.

- [ ] **Step 4: Add deterministic fingerprint regression tests**

For the small fixture and seeds 7/11, pin exact fingerprints after the implementation is stable. The regression values are evidence only for implementation determinism, not biological performance.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_cx_nulls.py tests/test_frozen_rewired_contract.py tests/test_rewiring_performance.py -q`

Expected: PASS with existing visual rewiring fingerprints unchanged.

- [ ] **Step 6: Commit**

```bash
git add python/yolo_flywire/cx_nulls.py tests/test_cx_nulls.py
git commit -m "feat: add matched CX null graphs"
```

---

### Task 4: Add a signed kinematic event encoder

**Files:**
- Create: `python/yolo_flywire/cx_events.py`
- Test: `tests/test_cx_events.py`

**Interfaces:**
- Produces: `KinematicEventEncoder.fit(train_sequences) -> KinematicEventEncoder`
- Produces: `KinematicEventEncoder.transform(sequence: np.ndarray) -> np.ndarray`
- Input sequence shape: `[time, joints, coords]`
- Output event shape: `[time, event_features]`, values in `{-1.0, 0.0, +1.0}`

- [ ] **Step 1: Write RED tests for motion-only sparse encoding**

```python
def test_static_pose_emits_no_motion_events():
    seq = np.ones((8, 25, 3), dtype=np.float32)
    encoder = KinematicEventEncoder.fixed_for_test(threshold=0.1)
    events = encoder.transform(seq)
    assert np.count_nonzero(events) == 0


def test_positive_and_negative_motion_keep_sign():
    seq = np.zeros((3, 1, 1), dtype=np.float32)
    seq[:, 0, 0] = [0.0, 1.0, 0.0]
    events = KinematicEventEncoder.fixed_for_test(threshold=0.5).transform(seq)
    assert +1.0 in events
    assert -1.0 in events
```

- [ ] **Step 2: Implement normalized `JM`, `BM`, and acceleration features**

Keep spatial normalization separate from event thresholding. For Gate 1 V1, use joint motion, bone motion, and joint acceleration; static `J` is not spike-encoded into recurrent input.

- [ ] **Step 3: Fit thresholds on training data only**

Use a deterministic per-feature absolute-motion quantile declared in protocol-development config, then serialize fitted numeric thresholds and their SHA-256. Validation/final transforms require an already-fitted encoder and never refit.

- [ ] **Step 4: Add serialization and hash tests**

The same threshold array must serialize to the same canonical JSON hash across runs; NaN/Inf thresholds are rejected.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_cx_events.py tests/test_features.py tests/test_pose_features.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/yolo_flywire/cx_events.py tests/test_cx_events.py
git commit -m "feat: encode skeleton motion as signed events"
```

---

### Task 5: Implement the biologically routed fixed-weight LIF reservoir

**Files:**
- Create: `python/yolo_flywire/models/cx_lif.py`
- Modify: `python/yolo_flywire/models/__init__.py`
- Test: `tests/test_cx_lif.py`

**Interfaces:**
- Produces: `CxLifState(membrane, synaptic, refractory, spikes)`
- Produces: `CxLifClassifier(input_dim: int, artifact: CxArtifact, num_classes: int, dynamics: CxDynamics)`
- Produces: `step(events_t, state) -> CxLifState`
- Produces: `encode(events) -> tuple[torch.Tensor, CxRunStats]`
- Produces: `forward(events) -> torch.Tensor`
- Recurrent weights are registered non-persistent buffers and absent from `parameters()`.

- [ ] **Step 1: Write RED tests that input reaches only INPUT nodes and readout sees only OUTPUT nodes**

```python
def test_input_projection_targets_only_biological_inputs(artifact):
    model = CxLifClassifier(12, artifact, 4, dynamics=TEST_DYNAMICS)
    assert model.input_projection.out_features == len(artifact.input_indices)
    assert model.readout.in_features == len(artifact.output_indices)


def test_recurrent_weights_are_not_trainable(artifact):
    model = CxLifClassifier(12, artifact, 4, dynamics=TEST_DYNAMICS)
    names = {name for name, _ in model.named_parameters()}
    assert all("recurrent" not in name for name in names)
```

- [ ] **Step 2: Implement sparse recurrent operator construction**

Build source, target, signed-weight tensors from the artifact. Apply `log1p` magnitude for the initial development protocol; keep magnitude policy explicit in `CxDynamics`. Do not densify to an `N x N` matrix.

- [ ] **Step 3: Implement one-step deterministic LIF dynamics**

```python
syn = decay * state.synaptic
syn.index_add_(1, target, state.spikes[:, source] * recurrent_weight)
mem = membrane_decay * state.membrane + syn + injected
can_fire = state.refractory == 0
spikes = (mem >= threshold) & can_fire
mem = torch.where(spikes, reset_value, mem)
refractory = torch.where(spikes, refractory_steps, torch.clamp(state.refractory - 1, min=0))
```

Input projection output is scattered only to `artifact.input_indices`. Readout concatenates or averages only `artifact.output_indices`; V1 freezes one policy in protocol before final execution.

- [ ] **Step 4: Add topology-sensitivity and zero-input-retention unit tests**

Create two tiny artifacts with identical node/edge/weight counts but different recurrent topology; load identical trainable parameters and show their spike/state trajectories differ. Also show zero input advances recurrent state without injecting events.

- [ ] **Step 5: Add spike accounting**

`CxRunStats` records total spikes, active-neuron count per step, recurrent synaptic events computed as outgoing retained edges from spiking sources, and peak state bytes.

- [ ] **Step 6: Export model and run model regressions**

Run: `python -m pytest tests/test_cx_lif.py tests/test_models.py tests/test_padded_models.py -q`

Expected: PASS; legacy models unchanged.

- [ ] **Step 7: Commit**

```bash
git add python/yolo_flywire/models/cx_lif.py python/yolo_flywire/models/__init__.py tests/test_cx_lif.py
git commit -m "feat: add fixed FlyWire CX LIF reservoir"
```

---

### Task 6: Implement early-prediction, retention, and recovery metrics

**Files:**
- Create: `python/yolo_flywire/cx_evaluation.py`
- Test: `tests/test_cx_evaluation.py`

**Interfaces:**
- Produces: `macro_f1(y_true, y_pred, num_classes) -> float`
- Produces: `early_prediction_auc(ratios, macro_f1_values) -> float`
- Produces: `retention_metrics(model, observed_events, true_label, horizon) -> RetentionMetrics`
- Produces: `recovery_metrics(model, events, perturb_at, silence_fraction, horizon, seed) -> RecoveryMetrics`
- Produces: `evaluate_gate1_seed(...) -> dict[str, Any]`

- [ ] **Step 1: Write exact numeric RED tests for normalized early AUC**

```python
def test_early_auc_uses_trapezoids_and_normalizes_ratio_span():
    ratios = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
    values = (0.2, 0.3, 0.5, 0.7, 0.8, 0.9)
    expected = np.trapezoid(values, ratios) / (ratios[-1] - ratios[0])
    assert early_prediction_auc(ratios, values) == pytest.approx(expected)
```

- [ ] **Step 2: Implement macro F1 without adding sklearn**

Handle absent predicted classes deterministically; classes present in the frozen task label set remain in the macro average even when a seed predicts none of them.

- [ ] **Step 3: Implement zero-input retention at the 40% boundary**

Freeze the state after observed-prefix processing; clone it; evolve one copy with zero external events for `horizon` steps. Compute true-class logit margin against the largest competing logit. Normalize the margin trajectory by `max(abs(initial_margin), epsilon)` where `epsilon` is a protocol constant, then trapezoid-integrate and divide by horizon.

- [ ] **Step 4: Implement deterministic node-silencing recovery**

At the 40% boundary, choose the silenced recurrent-node indices by `random.Random(perturbation_seed).sample(...)`; zero membrane/synaptic/spike state for exactly those nodes for one step, then continue the same observed event stream. Recovery occurs when the perturbed predicted class equals the matched unperturbed predicted class; report first recovery step and fixed-horizon recovery rate.

- [ ] **Step 5: Add metric edge-case tests**

Reject empty sequences, invalid ratios, nonfinite logits, perturbation fractions outside `(0,1)`, and retention normalization with nonfinite state. No silent NaN replacement.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_cx_evaluation.py tests/test_robustness.py tests/test_end_to_end.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/yolo_flywire/cx_evaluation.py tests/test_cx_evaluation.py
git commit -m "feat: measure CX temporal-state behavior"
```

---

### Task 7: Add a native NTU skeleton reader isolated from Issue #68 RGB input

**Files:**
- Create: `python/yolo_flywire/ntu_skeleton.py`
- Test: `tests/test_ntu_skeleton.py`

**Interfaces:**
- Produces: `NtuSkeletonSample(sample_id, subject, action, bodies)`
- Produces: `parse_skeleton_file(path: str | Path) -> NtuSkeletonSample`
- Produces: `split_for_subject_gate1(subject: int) -> str`
- Produces: `build_gate1_skeleton_manifest(root: str | Path, actions: tuple[int, ...]) -> dict[str, Any]`

- [ ] **Step 1: Write a minimal canonical `.skeleton` fixture and parser RED tests**

The parser must check declared frame/body/joint counts against consumed records, require finite numeric coordinates, and reject trailing malformed records. Preserve 25 NTU joints in file order.

- [ ] **Step 2: Implement official X-Sub outer split constants in the new module**

Do not import or alter `ntu_io.py`'s Issue #68 RGB manifest logic. Reuse the same published subject boundary values only by copying them into the Gate 1 module with an explicit reference comment and independent tests.

- [ ] **Step 3: Choose one body deterministically**

For Gate 1 V1, select the body with the largest number of valid tracked joints summed across frames; ties break by the numeric body ID. Record body-selection rule in the manifest. Do not add tracking heuristics.

- [ ] **Step 4: Build a hash-verified skeleton manifest**

Manifest each selected file with SHA-256, sample ID, subject/action, split, frame count, and chosen body ID. Compute dataset-content and split hashes using canonical JSON.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_ntu_skeleton.py tests/test_ntu_io.py -q`

Expected: PASS and no RGB behavior change.

- [ ] **Step 6: Commit**

```bash
git add python/yolo_flywire/ntu_skeleton.py tests/test_ntu_skeleton.py
git commit -m "feat: add isolated NTU skeleton input for CX Gate 1"
```

---

### Task 8: Add the development runner without opening final test

**Files:**
- Create: `python/yolo_flywire/cx_development.py`
- Test: `tests/test_cx_development.py`
- Modify: `protocols/v2-cx-temporal-gate1-preflight.json`

**Interfaces:**
- Produces CLI: `python -m yolo_flywire.cx_development prepare ...`
- Produces CLI: `python -m yolo_flywire.cx_development compare-validation ...`
- Produces one result JSON per `(arm, seed)` plus paired summary.
- Arms: `real`, `degree`, `block`; optional reference arms may be added only in a later separately frozen protocol.

- [ ] **Step 1: Write RED tests for strict arm/seed/budget equality**

A comparison must reject missing seeds, extra seeds, different epoch/step budgets, different encoder hashes, different input/output node hashes, or mismatched artifact/null fingerprints.

- [ ] **Step 2: Implement training on train split and checkpoint selection on validation only**

Use identical optimizer family, learning-rate schedule, max updates, batch construction, and early-stopping rule for all three graph arms. The model recurrent weights stay fixed; optimizer receives only trainable input/readout/global-gain parameters.

- [ ] **Step 3: Emit full temporal scoreboard per seed**

Result JSON contains:

```text
full_macro_f1
early_macro_f1_by_ratio
early_prediction_auc
retention_auc
retention_t50
recovery_rate
median_recovery_steps
spike_rate
active_neuron_fraction
synaptic_events_per_frame
parameter_count
```

- [ ] **Step 4: Implement paired Gate 1A / Gate 1B decision logic**

Use exactly the thresholds in Global Constraints. Report `pass`, `null`, or `negative`; never convert a null into pass using full-sequence accuracy.

- [ ] **Step 5: Keep final-test execution disabled**

Any `--split final_test` request against the preflight protocol raises `ValueError("final test is sealed")` until the protocol has all verified provenance and `final_test_enabled=true` in a later freeze commit.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_cx_development.py tests/test_pose_development.py tests/test_real_workflow.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/yolo_flywire/cx_development.py tests/test_cx_development.py protocols/v2-cx-temporal-gate1-preflight.json
git commit -m "feat: run sealed CX Gate 1 development comparisons"
```

---

### Task 9: Freeze real Codex artifact/null provenance in CI

**Files:**
- Create: `.github/workflows/flywire-cx-gate1-provenance.yml`
- Create: `python/yolo_flywire/cx_provenance.py`
- Test: `tests/test_cx_provenance.py`
- Modify after measured run: `protocols/v2-cx-temporal-gate1-preflight.json`

**Interfaces:**
- CLI: `python -m yolo_flywire.cx_provenance build --protocol ... --cell-types ... --connections ... --output ...`
- CLI: `python -m yolo_flywire.cx_provenance verify --protocol ... --cell-types ... --connections ... --output ...`

- [ ] **Step 1: Write fail-closed provenance tests**

Test wrong source hash, wrong real graph fingerprint, wrong null seed map, changed swap budget, missing role fingerprint, changed sign map, and corrupted serialized artifact. Every case must fail before producing comparison evidence.

- [ ] **Step 2: Implement source-byte hashing and artifact verification**

`build` hashes the two actual static-download files, constructs Real CX, generates five degree and five block nulls at the exact declared swap budget, writes immutable reports, and refuses to overwrite an existing output directory.

- [ ] **Step 3: Create CI workflow to download only public Codex static products**

Use the documented resource form:

```text
https://codex.flywire.ai/api/download_resource?data_product=consolidated_cell_types&dataset=fafb
https://codex.flywire.ai/api/download_resource?data_product=connections_princeton&dataset=fafb
```

The workflow records response bytes and SHA-256 before decompression/reading, runs `build`, uploads the artifact bundle, then runs `verify` against that bundle in a fresh step.

- [ ] **Step 4: Run the first provenance workflow and record measured fingerprints**

After CI produces a GREEN measured artifact, update only the preflight protocol fields for source hashes, Real artifact fingerprint, role fingerprints, and the five degree/block null fingerprints. Do not enable final test.

- [ ] **Step 5: Add measured regression tests**

Pin the measured maps in `tests/test_cx_provenance.py` so any source/product/selection/rewiring drift causes RED.

- [ ] **Step 6: Run the complete lightweight suite**

Run: `python -m pytest tests -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/flywire-cx-gate1-provenance.yml python/yolo_flywire/cx_provenance.py tests/test_cx_provenance.py protocols/v2-cx-temporal-gate1-preflight.json
git commit -m "test: freeze measured FlyWire CX provenance"
```

---

### Task 10: Verification checkpoint before any real Gate 1 training

**Files:**
- Modify only if verification exposes a defect in Tasks 1-9; no new feature files are expected.

**Interfaces:**
- Consumes all prior task contracts.
- Produces a documented exact-head readiness result; it does not produce a biological-performance claim.

- [ ] **Step 1: Run exact-head unit suite**

Run: `python -m pytest tests -q`

Expected: all tests PASS.

- [ ] **Step 2: Run Lean contracts**

Run: `lake build`

Expected: PASS; existing formal contracts remain unaffected.

- [ ] **Step 3: Verify current CX provenance bundle from scratch**

Run the same `cx_provenance build` and `verify` commands as CI on newly downloaded source bytes. Expected fingerprints must equal the committed frozen map exactly.

- [ ] **Step 4: Confirm final test is still sealed**

Run a test command requesting `final_test`; expected failure text: `final test is sealed`.

- [ ] **Step 5: Review scope diff against parent branch**

Confirm no changes to Issue #68 visual fingerprints, `load_visual_type_graph`, `rewire_degree_preserving`, `GraphRecurrentClassifier`, or existing real YOLO/Pose protocol semantics.

- [ ] **Step 6: Commit any verification-only corrections, then stop**

At this checkpoint the repository is ready for development-only train/validation experiments. Do not run or unseal the final-test comparison until train/validation behavior, compute budget, dataset manifest, encoder hash, dynamics, perturbation settings, and all remaining provenance fields are frozen in a separate reviewed commit.

# Pose Temporal SNN V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, reproducible temporal-model benchmark that compares RSNN, human-skeleton SpikingGraph, and streaming GRU on early action prediction and temporal-state metrics without any FlyWire dependency.

**Architecture:** A new `pose_snn/` Python project owns the full experiment boundary: native NTU skeleton inventory/parsing, tracking-aware normalization, signed motion-event encoding, three streaming temporal cores behind one interface, shared evaluation, and a fail-closed development protocol. The old `python/yolo_flywire` package remains untouched and is never imported by the new project.

**Tech Stack:** Python 3.11, NumPy >=1.26, PyTorch >=2.3, pytest >=8.0, stdlib `csv`/`hashlib`/`json`/`pathlib`.

**Spec:** `docs/superpowers/specs/2026-09-15-pose-temporal-snn-v1-design.md`

## Global Constraints

- Repository: `qigao/lean`; branch: `research/pose-temporal-snn-v1`.
- New project root is exactly `pose_snn/`; no module under it may import `yolo_flywire`.
- Dataset V1 is the NTU120 10-class motion subset: actions `(8, 9, 22, 23, 26, 27, 31, 34, 35, 36)`.
- Official X-Sub outer-train subjects are exactly `{1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38,45,46,47,49,50,52,53,54,55,56,57,58,59,70,74,78,80,81,82,83,84,85,86,89,91,92,93,94,95,97,98,100,103}`; every subject in `1..106` outside that set is `final_test`.
- Frozen inner-validation subjects are exactly `(14, 28, 35, 46, 50, 54, 74, 83, 84, 86, 103)`; all are inside outer-train.
- Final-test coordinates are never normalized, event-encoded, used for model selection, or used to fit thresholds before an explicit later protocol-unseal commit.
- Primary-body selection is `max-fully-tracked-joints-then-lowest-body-id`.
- A joint observation is usable iff body is present and `tracking_state >= 1`.
- Normalization is linear temporal interpolation with endpoint hold, root joint `0` centering, and median torso scale `||joint20-joint0||`; degenerate/nonfinite samples fail closed.
- Event features per joint are exactly 9 channels: joint motion XYZ, bone motion XYZ assigned to the child joint, joint acceleration XYZ. Root bone-motion channels are zero. Flat model input dimension is `25 * 9 = 225`.
- Event threshold quantile is exactly `0.75`, fit on development-train normalized sequences only.
- Event values are exactly `{-1.0, 0.0, +1.0}`; no Poisson/rate encoding.
- Observation ratios are exactly `(0.10, 0.20, 0.40, 0.60, 0.80, 1.00)`.
- Seeds are exactly `(7, 11, 19, 23, 31)`.
- Optimizer is Adam, learning rate `1e-3`, batch size `1`, maximum `10` epochs, maximum `2000` optimizer updates, checkpoint rule `best-validation-macro-f1-over-all-epochs`.
- Parameter ceiling is `50_000` trainable parameters per model arm.
- RSNN V1: hidden size `96`, membrane decay `0.95`, synaptic decay `0.80`, threshold `1.0`, reset `0.0`, one recurrent delay step, fast-sigmoid surrogate slope `5.0`, dense trainable recurrent matrix.
- SpikingGraph V1: `25` NTU joint nodes, hidden channels `32`, same LIF dynamics as RSNN, frozen undirected NTU kinematic adjacency plus self loops, shared trainable input/self/neighbor transforms.
- GRU V1: hidden size `64`, one `torch.nn.GRUCell`, streaming one frame at a time.
- Retention boundary is observation ratio `0.40`; zero-input retention horizon is `20` steps.
- Perturbation is deterministic hidden-state silencing of `25%` of hidden units/nodes at the 40% boundary; recovery horizon is `10` observed steps.
- Primary development success: mean paired `(RSNN - GRU) early_prediction_auc >= 0.02` and positive difference in at least `4/5` seeds, plus either retention-AUC improvement `>= 0.05` or recovery-rate improvement `>= 0.05`, positive in at least `4/5` seeds.
- If RSNN is null/negative versus GRU, V1 stops before RTMW133. SpikingGraph may continue only as a human-skeleton graph result, not as evidence for a generic SNN advantage.

---

### Task 1: Create the isolated package and fail-closed protocol

**Files:**
- Create: `pose_snn/pyproject.toml`
- Create: `pose_snn/src/pose_temporal_snn/__init__.py`
- Create: `pose_snn/src/pose_temporal_snn/protocol.py`
- Create: `pose_snn/protocols/v1-development-preflight.json`
- Test: `pose_snn/tests/test_protocol.py`

**Interfaces:**
- Produces: `ExperimentProtocol`, `load_protocol(path)`, `validate_protocol_dict(raw)`.
- Later tasks consume protocol fields for dataset actions, split IDs, event quantile, seeds, model configs, training budget, evaluation horizons, and final-test sealing.

- [ ] **Step 1: Write the package metadata and failing protocol tests**

`pose_snn/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pose-temporal-snn"
version = "0.1.0"
description = "Streaming skeleton temporal SNN experiments"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "torch>=2.3"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

`pose_snn/tests/test_protocol.py` starts with:

```python
from pathlib import Path
import json
import pytest

from pose_temporal_snn.protocol import load_protocol, validate_protocol_dict

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "v1-development-preflight.json"


def test_preflight_freezes_v1_without_opening_final_test():
    p = load_protocol(PROTOCOL)
    assert p.actions == (8, 9, 22, 23, 26, 27, 31, 34, 35, 36)
    assert p.validation_subjects == (14, 28, 35, 46, 50, 54, 74, 83, 84, 86, 103)
    assert p.event_quantile == pytest.approx(0.75)
    assert p.observation_ratios == (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
    assert p.seeds == (7, 11, 19, 23, 31)
    assert p.parameter_ceiling == 50_000
    assert p.final_test_enabled is False


def test_final_test_requires_all_measured_pins():
    raw = json.loads(PROTOCOL.read_text())
    raw["final_test_enabled"] = True
    with pytest.raises(ValueError, match="final test provenance"):
        validate_protocol_dict(raw)


def test_flywire_fields_are_not_part_of_the_new_schema():
    raw = json.loads(PROTOCOL.read_text())
    raw["flywire_release"] = "FAFB-v783"
    with pytest.raises(ValueError, match="schema"):
        validate_protocol_dict(raw)
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
cd pose_snn
python -m pytest tests/test_protocol.py -q
```

Expected: import failure because `pose_temporal_snn.protocol` does not exist.

- [ ] **Step 3: Implement the immutable protocol schema**

Use a frozen dataclass with these public fields:

```python
@dataclass(frozen=True)
class ExperimentProtocol:
    protocol_id: str
    actions: tuple[int, ...]
    outer_train_subjects: tuple[int, ...]
    validation_subjects: tuple[int, ...]
    event_quantile: float
    observation_ratios: tuple[float, ...]
    seeds: tuple[int, ...]
    parameter_ceiling: int
    retention_ratio: float
    retention_horizon: int
    perturbation_fraction: float
    recovery_horizon: int
    rsnn: ModelConfig
    spiking_graph: ModelConfig
    gru: ModelConfig
    training: TrainingConfig
    final_test_enabled: bool
    dataset_content_hash: str | None
    split_hash: str | None
    encoder_hash: str | None
    label_map_hash: str | None
    model_fingerprints: tuple[tuple[str, str], ...] | None
```

`validate_protocol_dict` must reject extra keys, wrong frozen constants, malformed SHA-256 strings, validation subjects outside outer-train, and `final_test_enabled=True` while any measured pin is null.

- [ ] **Step 4: Add the preflight JSON**

The JSON freezes every Global Constraint constant and leaves only measured byte/data/model fingerprints null. Set `final_test_enabled` to `false`.

- [ ] **Step 5: Run protocol tests and commit**

Run:

```bash
cd pose_snn
python -m pytest tests/test_protocol.py -q
```

Expected: PASS.

Commit:

```bash
git add pose_snn
 git commit -m "test: freeze pose temporal SNN V1 protocol"
```

---

### Task 2: Parse, inventory, split, and normalize native NTU skeletons

**Files:**
- Create: `pose_snn/src/pose_temporal_snn/ntu.py`
- Create: `pose_snn/src/pose_temporal_snn/normalization.py`
- Test: `pose_snn/tests/test_ntu.py`
- Test: `pose_snn/tests/test_normalization.py`

**Interfaces:**
- Produces: `NtuBody`, `NtuSkeletonSample`, `parse_skeleton_file(path)`, `select_primary_body(sample)`, `split_for_subject(subject, protocol)`, `build_manifest(root, protocol)`.
- Produces: `normalize_body(body) -> np.ndarray` with shape `[time, 25, 3]`.

- [ ] **Step 1: Write parser RED tests with a canonical mini `.skeleton` fixture**

Test canonical filename parsing, exactly 25 joints/body, finite XYZ, body-presence tracking, and deterministic body selection. Include:

```python
def test_primary_body_tie_breaks_by_lowest_body_id(tmp_path):
    sample = parse_skeleton_file(write_two_body_fixture(tmp_path, ids=(20, 10), equal_tracked=True))
    assert select_primary_body(sample).body_id == 10


def test_outer_and_inner_subject_split_is_frozen(protocol):
    assert split_for_subject(56, protocol) == "train"
    assert split_for_subject(14, protocol) == "validation"
    assert split_for_subject(3, protocol) == "final_test"
```

- [ ] **Step 2: Implement parser and byte-backed manifest**

`build_manifest` must hash selected `.skeleton` bytes, reject duplicate sample IDs/content, record `relative_path`, `sha256`, `subject`, `action`, `selected_body_id`, and split, then compute:

```python
dataset_content_hash = sha256(canonical_json([
    {"sample_id": r["sample_id"], "size_bytes": r["size_bytes"], "sha256": r["sha256"]}
    for r in samples
]))
```

The final-test files are inventoried and hashed but are not normalized in this task.

- [ ] **Step 3: Write normalization RED tests**

Include exact invariance and missing-data behavior:

```python
def test_translation_and_uniform_scale_do_not_change_normalized_motion():
    a = synthetic_body(offset=(0,0,0), scale=1.0)
    b = synthetic_body(offset=(10,-3,5), scale=7.0)
    np.testing.assert_allclose(normalize_body(a), normalize_body(b), atol=1e-6)


def test_tracking_zero_is_interpolated_not_converted_to_origin_jump():
    body = synthetic_body_with_missing_joint_frame(joint=5, frame=2)
    out = normalize_body(body)
    np.testing.assert_allclose(out[2,5], (out[1,5] + out[3,5]) / 2.0, atol=1e-6)
```

- [ ] **Step 4: Implement deterministic interpolation/root-center/scale normalization**

For each joint/coordinate use `np.interp` over usable frame indices. Reject joints with zero usable observations. Center on joint 0 each frame. Compute scale as the median finite positive `norm(j20-j0)` over usable torso frames; reject scale `<= 1e-6`.

- [ ] **Step 5: Run focused tests and commit**

```bash
cd pose_snn
python -m pytest tests/test_ntu.py tests/test_normalization.py -q
git add pose_snn/src/pose_temporal_snn/ntu.py pose_snn/src/pose_temporal_snn/normalization.py pose_snn/tests
 git commit -m "feat: normalize native NTU skeleton streams"
```

---

### Task 3: Encode signed per-joint motion events without leakage

**Files:**
- Create: `pose_snn/src/pose_temporal_snn/events.py`
- Test: `pose_snn/tests/test_events.py`

**Interfaces:**
- Produces: `EventEncoder.fit(train_sequences, quantile=0.75)`, `transform(sequence)`.
- `transform` returns `EventSequence(flat, nodes)` where `flat.shape == [T,225]` and `nodes.shape == [T,25,9]`.
- Produces stable `fingerprint` and canonical `to_json()`/`from_json()`.

- [ ] **Step 1: Write RED tests for feature structure and training-only fit**

```python
def test_static_skeleton_emits_no_events():
    seq = np.ones((8,25,3), dtype=np.float32)
    enc = EventEncoder.fixed_for_test(0.1)
    out = enc.transform(seq)
    assert out.flat.shape == (8,225)
    assert out.nodes.shape == (8,25,9)
    assert np.count_nonzero(out.flat) == 0


def test_root_has_zero_bone_motion_channels():
    out = EventEncoder.fixed_for_test(0.1).transform(moving_fixture())
    assert np.count_nonzero(out.nodes[:,0,3:6]) == 0
```

- [ ] **Step 2: Implement NTU25 bone topology and 9-channel node features**

Freeze the parent map:

```python
NTU25_PARENT = (
    -1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10,
    0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11,
)
```

For each child with parent >=0 compute bone vector; root bone is zero. Compute `JM`, `BM`, `A` with first frame zero and concatenate per joint as `[JM_xyz, BM_xyz, A_xyz]`.

- [ ] **Step 3: Fit thresholds and signed events**

Flatten `[T,25,9]` to `[T,225]`, fit per-feature `quantile(abs(feature), 0.75)` over concatenated development-train sequences only, replace exact-zero fitted thresholds with the smallest positive finite training magnitude for that feature or fail if the feature is always zero; transform to `-1/0/+1`.

- [ ] **Step 4: Add deterministic serialization/fingerprint tests**

Fingerprint canonical JSON containing feature spec ID, quantile, parent map, and 225 thresholds. Validation transforms must not mutate thresholds.

- [ ] **Step 5: Run and commit**

```bash
cd pose_snn
python -m pytest tests/test_events.py -q
git add pose_snn/src/pose_temporal_snn/events.py pose_snn/tests/test_events.py
 git commit -m "feat: encode pose motion as signed events"
```

---

### Task 4: Implement the shared streaming interface and RSNN

**Files:**
- Create: `pose_snn/src/pose_temporal_snn/models/base.py`
- Create: `pose_snn/src/pose_temporal_snn/models/rsnn.py`
- Create: `pose_snn/src/pose_temporal_snn/models/__init__.py`
- Test: `pose_snn/tests/test_rsnn.py`

**Interfaces:**
- `StreamingTemporalModel.initial_state(batch_size, device, dtype)`.
- `step(events_t, state) -> state`.
- `logits(state) -> Tensor[B,C]`.
- `forward(events) -> Tensor[B,C]` loops only through `step`.
- RSNN state: `(membrane, synaptic, spikes)` each `[B,96]`.

- [ ] **Step 1: Write interface/stream equivalence RED tests**

```python
def test_forward_equals_manual_step_loop():
    model = RSNN(input_dim=225, num_classes=10)
    x = torch.zeros(2,12,225)
    state = model.initial_state(2, x.device, x.dtype)
    for t in range(x.shape[1]):
        state = model.step(x[:,t], state)
    torch.testing.assert_close(model(x), model.logits(state))
```

- [ ] **Step 2: Implement surrogate spike autograd**

Use hard forward spike and fast-sigmoid backward:

```python
class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        s = ctx.slope
        return grad / (1.0 + s * x.abs()).pow(2), None
```

- [ ] **Step 3: Implement RSNN dynamics**

```python
syn = 0.80 * state.synaptic + input_proj(events_t) + recurrent(state.spikes)
mem = 0.95 * state.membrane + syn
spikes = spike(mem - 1.0, slope=5.0)
mem = torch.where(spikes.bool(), torch.zeros_like(mem), mem)
```

`input_proj=Linear(225,96,bias=False)`, `recurrent=Linear(96,96,bias=False)`, `readout=Linear(96,10)`. `logits` reads `membrane + spikes`.

- [ ] **Step 4: Test trainability, statefulness, zero-input retention, and parameter ceiling**

Assert recurrent weight receives nonzero finite gradient on a tiny sequence; zero external input still evolves a nonzero recurrent state; parameter count <= 50_000.

- [ ] **Step 5: Run and commit**

```bash
cd pose_snn
python -m pytest tests/test_rsnn.py -q
git add pose_snn/src/pose_temporal_snn/models pose_snn/tests/test_rsnn.py
 git commit -m "feat: add streaming recurrent LIF baseline"
```

---

### Task 5: Implement human-skeleton SpikingGraph

**Files:**
- Create: `pose_snn/src/pose_temporal_snn/models/spiking_graph.py`
- Test: `pose_snn/tests/test_spiking_graph.py`

**Interfaces:**
- Consumes node events `[B,25,9]` per step.
- State tensors `[B,25,32]` for membrane/synaptic/spikes.
- Same `initial_state`, `step`, `logits`, `forward` interface.

- [ ] **Step 1: Write RED tests for adjacency-constrained message flow**

Use a toy state with one active joint and assert after one recurrent step only self and anatomical neighbors receive graph-message current before further temporal propagation.

- [ ] **Step 2: Freeze the NTU25 undirected adjacency**

Build from the parent map in Task 3. Add self loops. Store row-normalized adjacency as a non-trainable buffer. No learned/new edges are allowed.

- [ ] **Step 3: Implement shared graph-LIF transforms**

```python
input_current = input_proj(events_t)              # [B,25,32]
neighbor_spikes = torch.einsum("ij,bjh->bih", A, state.spikes)
syn = 0.80 * state.synaptic + input_current + self_recurrent(state.spikes) + neighbor_recurrent(neighbor_spikes)
mem = 0.95 * state.membrane + syn
spikes = spike(mem - 1.0, slope=5.0)
```

Use shared `Linear(9,32,bias=False)`, `Linear(32,32,bias=False)` self and neighbor transforms. Readout mean-pools `(membrane + spikes)` over 25 joints then `Linear(32,10)`.

- [ ] **Step 4: Test streaming equivalence and ceiling**

Assert no imports from `yolo_flywire`, deterministic output for fixed seed, forward/manual-step equivalence, and parameter count <= 50_000.

- [ ] **Step 5: Run and commit**

```bash
cd pose_snn
python -m pytest tests/test_spiking_graph.py -q
git add pose_snn/src/pose_temporal_snn/models/spiking_graph.py pose_snn/tests/test_spiking_graph.py
 git commit -m "feat: add human-skeleton spiking graph model"
```

---

### Task 6: Implement the decisive streaming GRU baseline

**Files:**
- Create: `pose_snn/src/pose_temporal_snn/models/gru.py`
- Test: `pose_snn/tests/test_gru.py`

**Interfaces:**
- Flat input `[B,225]` per step.
- State `[B,64]`.
- Same streaming interface.

- [ ] **Step 1: Write streaming equivalence RED test**

```python
def test_gru_forward_is_exact_manual_streaming():
    model = StreamingGRU(input_dim=225, hidden_size=64, num_classes=10)
    x = torch.randn(2,10,225)
    state = model.initial_state(2, x.device, x.dtype)
    for t in range(10):
        state = model.step(x[:,t], state)
    torch.testing.assert_close(model(x), model.logits(state))
```

- [ ] **Step 2: Implement `GRUCell(225,64)` and readout**

No bidirectional context, sequence packing, or hidden history cache. `forward` is a simple `for t` loop through `step`.

- [ ] **Step 3: Test parameter ceiling and no future-frame leakage**

Changing frames after an observation prefix must not change logits obtained by streaming only the prefix. Assert trainable parameters <= 50_000.

- [ ] **Step 4: Run and commit**

```bash
cd pose_snn
python -m pytest tests/test_gru.py -q
git add pose_snn/src/pose_temporal_snn/models/gru.py pose_snn/tests/test_gru.py
 git commit -m "feat: add streaming GRU control"
```

---

### Task 7: Add shared evaluation and temporal diagnostics

**Files:**
- Create: `pose_snn/src/pose_temporal_snn/evaluation.py`
- Test: `pose_snn/tests/test_evaluation.py`

**Interfaces:**
- Produces: `macro_f1`, `early_prediction_auc`, `retention_metrics`, `recovery_metrics`, `activity_metrics`.
- All model-specific access goes through `initial_state/step/logits`.

- [ ] **Step 1: Write exact numeric macro-F1 and early-AUC tests**

```python
def test_early_auc_is_normalized_trapezoid():
    ratios = (0.10,0.20,0.40,0.60,0.80,1.00)
    scores = (0.2,0.3,0.5,0.7,0.8,0.9)
    expected = np.trapezoid(scores, ratios) / 0.90
    assert early_prediction_auc(ratios, scores) == pytest.approx(expected)
```

- [ ] **Step 2: Implement prefix evaluation using streaming state**

For each ratio, consume exactly `ceil(T * ratio)` frames from a fresh initial state and compute logits. Never call a model-specific batch shortcut when measuring streaming metrics.

- [ ] **Step 3: Implement zero-input retention**

At the 40% boundary freeze the observed state, record true-class logit margin, then step 20 zero-input frames. Normalize the margin by `max(abs(initial_margin), 1e-6)` and integrate trapezoidally over the 20-step horizon. Report normalized retention AUC and T50.

- [ ] **Step 4: Implement deterministic perturbation recovery for heterogeneous state shapes**

Define a model method `silence_state(state, fraction, seed)` in `base.py`. RSNN/GRU silence a deterministic 25% hidden-unit subset; SpikingGraph silences 25% of `(joint,channel)` units. Continue with the same future observed events for 10 steps and measure return to the unperturbed predicted class.

- [ ] **Step 5: Add activity accounting**

SNN models expose total spikes, active-unit fraction, and recurrent/graph synaptic events per frame. GRU reports spike-specific metrics as `None` and reports streaming step count/latency only; do not fabricate equivalent spikes.

- [ ] **Step 6: Run and commit**

```bash
cd pose_snn
python -m pytest tests/test_evaluation.py tests/test_rsnn.py tests/test_spiking_graph.py tests/test_gru.py -q
git add pose_snn/src/pose_temporal_snn/evaluation.py pose_snn/src/pose_temporal_snn/models/base.py pose_snn/tests
 git commit -m "feat: measure streaming temporal behavior"
```

---

### Task 8: Add deterministic data preparation and fair training

**Files:**
- Create: `pose_snn/src/pose_temporal_snn/data.py`
- Create: `pose_snn/src/pose_temporal_snn/training.py`
- Test: `pose_snn/tests/test_training_fairness.py`
- Test: `pose_snn/tests/test_final_test_seal.py`

**Interfaces:**
- Produces `PreparedData(binding, encoder, train_samples, validation_samples, final_test_inventory)`.
- Produces `train_validation_arm(model_kind, seed, prepared, protocol) -> RunResult`.
- `RunResult` contains model fingerprint, initial parameter fingerprint, training-order fingerprint, best epoch, update count, parameter count, and evaluation scoreboard.

- [ ] **Step 1: Write leakage RED tests**

Create tiny local NTU fixtures where validation motion magnitude changes by 1000x. `prepare_data` must produce identical encoder fingerprints while dataset-content hash changes. Put an intentionally non-normalizable final-test sample in the fixture and assert preparation still succeeds because final-test coordinates are never parsed into normalized events.

- [ ] **Step 2: Implement `prepare_data`**

Call `build_manifest`; normalize/encode only train and validation rows. Fit `EventEncoder` only on development-train. Map actions to labels by sorted action ID. Compute canonical hashes for dataset bytes, split assignments, label map, encoder, and preparation config. Verify protocol final-test remains sealed.

- [ ] **Step 3: Write fairness RED tests for model arms**

For a fixed seed, assert all three models receive the same sample-order fingerprint. For repeated same-arm runs assert exact result determinism on CPU fixture. Assert each architecture fingerprint and trainable parameter count are recorded and <= ceiling.

- [ ] **Step 4: Implement deterministic training**

For each epoch use:

```python
order = list(range(len(train_samples)))
random.Random((seed << 32) + epoch).shuffle(order)
```

Reset `torch.manual_seed(seed)` before model construction. Train batch size 1 with Adam/lr `1e-3`; stop at epoch 10 or 2000 updates. Select checkpoint only by validation full-sequence macro-F1. Evaluate early/retention/recovery only after restoring the selected checkpoint.

- [ ] **Step 5: Implement model fingerprints**

Canonical fingerprint includes model kind plus all architecture/dynamics constants and, for SpikingGraph, the frozen parent/adjacency definition. It excludes learned parameter values.

- [ ] **Step 6: Run and commit**

```bash
cd pose_snn
python -m pytest tests/test_training_fairness.py tests/test_final_test_seal.py -q
git add pose_snn/src/pose_temporal_snn/data.py pose_snn/src/pose_temporal_snn/training.py pose_snn/tests
 git commit -m "feat: train temporal models under one fair protocol"
```

---

### Task 9: Add development comparison, stop rule, and CLI

**Files:**
- Create: `pose_snn/src/pose_temporal_snn/cli.py`
- Create: `pose_snn/src/pose_temporal_snn/results.py`
- Modify: `pose_snn/pyproject.toml`
- Test: `pose_snn/tests/test_development_runner.py`

**Interfaces:**
- CLI: `pose-temporal-snn prepare --root ... --protocol ... --output ...`.
- CLI: `pose-temporal-snn compare-validation --root ... --prepared ... --output ...`.
- Produces exactly 15 run cells: 3 model kinds x 5 seeds.
- Produces `summary.json` with paired RSNN-GRU primary decision and SpikingGraph secondary comparisons.

- [ ] **Step 1: Add CLI entry point**

In `pyproject.toml`:

```toml
[project.scripts]
pose-temporal-snn = "pose_temporal_snn.cli:main"
```

- [ ] **Step 2: Write complete-matrix and stop-rule RED tests**

A summary must reject missing/duplicate seed-model cells or any result whose dataset/split/encoder/model-budget pins differ. Primary status is `pass`, `null`, or `negative` using the exact Global Constraints thresholds; full-sequence F1 cannot turn a temporal null into pass.

- [ ] **Step 3: Implement paired decision logic**

For each seed compute RSNN minus GRU early-AUC, retention-AUC, and recovery-rate differences. Pass only on primary + one temporal condition. A reverse early effect <= -0.02 in at least 4/5 seeds is `negative`; otherwise `null`.

- [ ] **Step 4: Keep final test inaccessible**

No CLI command in V1 preflight accepts `--split final_test`. Any internal request for it raises `ValueError("final test is sealed")` while `protocol.final_test_enabled == False`.

- [ ] **Step 5: Run and commit**

```bash
cd pose_snn
python -m pytest tests/test_development_runner.py tests/test_final_test_seal.py -q
git add pose_snn
 git commit -m "feat: compare temporal SNN development arms"
```

---

### Task 10: Add independent CI and exact-head verification

**Files:**
- Create: `.github/workflows/pose-temporal-snn.yml`
- Test: all `pose_snn/tests`.

**Interfaces:**
- CI is independent of all FlyWire/Codex workflows and downloads no NTU data.

- [ ] **Step 1: Add lightweight data/protocol job**

Install `pytest numpy` plus `pip install --no-deps -e ./pose_snn`; run:

```bash
python -m pytest pose_snn/tests/test_protocol.py pose_snn/tests/test_ntu.py pose_snn/tests/test_normalization.py pose_snn/tests/test_events.py -q
```

- [ ] **Step 2: Add CPU-Torch model/training job**

Install CPU-only torch, then run model/evaluation/training/development tests.

- [ ] **Step 3: Add full suite and Lean regression jobs**

Full Python:

```bash
python -m pytest pose_snn/tests -q
```

Repository regression:

```bash
lake build
```

- [ ] **Step 4: Add isolation regression test**

Search every Python file under `pose_snn/src` and fail if source text contains `yolo_flywire`, `flywire`, `codex`, `connectome`, or imports outside `pose_temporal_snn` except stdlib/NumPy/Torch. The design/spec files are excluded from this source-code test.

- [ ] **Step 5: Run exact-head verification**

Require all four jobs GREEN on one commit. Then compare branch diff against `7b5de8176adf4beff5e2a46c8ff021caad6404ae` and verify no files under `python/yolo_flywire`, `YoloFlywire/`, or existing FlyWire protocols were modified.

- [ ] **Step 6: Stop before real development training if NTU bytes are absent**

At this checkpoint the code is ready for a legally obtained local NTU skeleton root. Do not fabricate data, download unauthorized mirrors, or claim RSNN/GRU performance from fixtures.

- [ ] **Step 7: Commit verification-only corrections and record readiness**

```bash
git add .github/workflows/pose-temporal-snn.yml pose_snn
 git commit -m "ci: verify isolated pose temporal SNN V1"
```

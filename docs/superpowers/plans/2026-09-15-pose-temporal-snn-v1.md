# Pose Temporal SNN V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, reproducible temporal-model benchmark that compares RSNN, human-skeleton SpikingGraph, and streaming GRU on early action prediction and temporal-state metrics without any FlyWire dependency.

**Architecture:** A new `pose_snn/` Python project owns the full experiment boundary: native NTU skeleton inventory/parsing, tracking-aware normalization, signed motion-event encoding, three streaming temporal cores behind one interface, shared evaluation, and a fail-closed development protocol. The old `python/yolo_flywire` package remains untouched and is never imported by the new project.

**Tech Stack:** Python 3.11, NumPy >=1.26, PyTorch >=2.3, pytest >=8.0, stdlib `hashlib`/`json`/`pathlib`/`random`.

**Spec:** `docs/superpowers/specs/2026-09-15-pose-temporal-snn-v1-design.md`

## Global Constraints

- Repository: `qigao/lean`; branch: `research/pose-temporal-snn-v1`.
- New project root is exactly `pose_snn/`; no module under it may import `yolo_flywire`.
- Dataset V1 is the NTU120 10-class motion subset: actions `(8, 9, 22, 23, 26, 27, 31, 34, 35, 36)`.
- Official X-Sub outer-train subjects are exactly `{1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38,45,46,47,49,50,52,53,54,55,56,57,58,59,70,74,78,80,81,82,83,84,85,86,89,91,92,93,94,95,97,98,100,103}`; every subject in `1..106` outside that set is `final_test`.
- Frozen inner-validation subjects are exactly `(14, 28, 35, 46, 50, 54, 74, 83, 84, 86, 103)`.
- Final-test coordinates are never normalized, event-encoded, used for model selection, or used to fit thresholds before a later explicit unseal commit.
- Primary-body selection is `max-fully-tracked-joints-then-lowest-body-id`.
- A joint observation is usable iff the body is present and `tracking_state >= 1`.
- Normalization is linear temporal interpolation with endpoint hold, root joint `0` centering, and median torso scale `||joint20-joint0||`; degenerate/nonfinite samples fail closed.
- Event features per joint are exactly 9 channels: joint motion XYZ, bone motion XYZ assigned to the child joint, joint acceleration XYZ. Root bone-motion channels are structurally zero. Flat model input dimension is `25 * 9 = 225`.
- Event threshold quantile is exactly `0.75`, fit on development-train normalized sequences only; each threshold is `max(q75(abs(feature)), 1e-6)` so structurally zero channels stay inactive without making the encoder undefined.
- Event values are exactly `{-1.0, 0.0, +1.0}`; no Poisson/rate encoding.
- Observation ratios are exactly `(0.10, 0.20, 0.40, 0.60, 0.80, 1.00)`.
- Seeds are exactly `(7, 11, 19, 23, 31)`.
- Optimizer is Adam, learning rate `1e-3`, batch size `1`, maximum `10` epochs, maximum `2000` optimizer updates, checkpoint rule `best-validation-macro-f1-over-all-epochs`.
- Parameter ceiling is `50_000` trainable parameters per model arm.
- RSNN V1: hidden size `96`, membrane decay `0.95`, synaptic decay `0.80`, threshold `1.0`, reset `0.0`, one recurrent delay step, fast-sigmoid surrogate slope `5.0`, dense trainable recurrent matrix.
- SpikingGraph V1: `25` NTU joint nodes, hidden channels `32`, same LIF dynamics as RSNN, frozen undirected NTU kinematic adjacency plus self loops, shared trainable input/self/neighbor transforms.
- GRU V1: hidden size `56`, one `torch.nn.GRUCell`, streaming one frame at a time. `56` is frozen because `64` would exceed the `50_000` trainable-parameter ceiling with 225 input features.
- Retention boundary is observation ratio `0.40`; zero-input retention horizon is `20` steps.
- Perturbation is deterministic hidden-state silencing of `25%` of hidden units at the 40% boundary; recovery horizon is `10` observed steps.
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
- Produces `ModelConfig`, `TrainingConfig`, `ExperimentProtocol`, `load_protocol(path)`, `validate_protocol_dict(raw)`.
- Later tasks consume only these protocol values; no task hard-codes a second copy of model/training constants.

- [ ] **Step 1: Write package metadata and failing protocol tests**

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

`pose_snn/tests/test_protocol.py` includes:

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
    assert p.seeds == (7, 11, 19, 23, 31)
    assert p.parameter_ceiling == 50_000
    assert p.gru.hidden_size == 56
    assert p.final_test_enabled is False


def test_final_test_requires_all_measured_pins():
    raw = json.loads(PROTOCOL.read_text())
    raw["final_test_enabled"] = True
    with pytest.raises(ValueError, match="final test provenance"):
        validate_protocol_dict(raw)


def test_flywire_fields_are_not_part_of_the_new_schema():
    raw = json.loads(PROTOCOL.read_text())
    raw["flywire_release"] = "forbidden"
    with pytest.raises(ValueError, match="schema"):
        validate_protocol_dict(raw)
```

- [ ] **Step 2: Run the test and verify RED**

```bash
cd pose_snn
python -m pytest tests/test_protocol.py -q
```

Expected: import failure because `pose_temporal_snn.protocol` does not exist.

- [ ] **Step 3: Implement frozen dataclasses and strict schema validation**

```python
@dataclass(frozen=True)
class ModelConfig:
    kind: str
    hidden_size: int
    membrane_decay: float | None = None
    synaptic_decay: float | None = None
    threshold: float | None = None
    reset: float | None = None
    surrogate_slope: float | None = None

@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str
    learning_rate: float
    batch_size: int
    epochs: int
    max_updates: int
    checkpoint_rule: str

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

Reject extra JSON keys, changed frozen constants, malformed digests, validation subjects outside outer-train, and `final_test_enabled=True` while any measured pin is null.

- [ ] **Step 4: Add `v1-development-preflight.json`**

Freeze every Global Constraint constant. Leave only byte/data/model fingerprints null and set `final_test_enabled=false`.

- [ ] **Step 5: Run and commit**

```bash
cd pose_snn
python -m pytest tests/test_protocol.py -q
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
- Produces `NtuBody`, `NtuSkeletonSample`, `parse_skeleton_file`, `select_primary_body`, `split_for_subject`, `build_manifest`.
- Produces `normalize_body(body) -> np.ndarray[T,25,3]`.

- [ ] **Step 1: Write parser RED tests**

Include canonical filename parsing, exactly 25 joints, finite XYZ, body presence, and tie-break behavior:

```python
def test_primary_body_tie_breaks_by_lowest_body_id(tmp_path):
    sample = parse_skeleton_file(write_two_body_fixture(tmp_path, ids=(20, 10), equal_tracked=True))
    assert select_primary_body(sample).body_id == 10


def test_outer_and_inner_split_is_frozen(protocol):
    assert split_for_subject(56, protocol) == "train"
    assert split_for_subject(14, protocol) == "validation"
    assert split_for_subject(3, protocol) == "final_test"
```

- [ ] **Step 2: Implement parser and byte-backed manifest**

`build_manifest` hashes selected `.skeleton` bytes, rejects duplicate IDs/content, records `relative_path`, `size_bytes`, `sha256`, `subject`, `action`, `selected_body_id`, and split, then hashes the canonical sample identity list. Final-test files are inventoried and hashed but not normalized.

- [ ] **Step 3: Write normalization RED tests**

```python
def test_translation_and_uniform_scale_are_removed():
    a = synthetic_body(offset=(0,0,0), scale=1.0)
    b = synthetic_body(offset=(10,-3,5), scale=7.0)
    np.testing.assert_allclose(normalize_body(a), normalize_body(b), atol=1e-6)


def test_tracking_zero_interpolates_instead_of_origin_jump():
    body = synthetic_body_with_missing_joint_frame(joint=5, frame=2)
    out = normalize_body(body)
    np.testing.assert_allclose(out[2,5], (out[1,5] + out[3,5]) / 2.0, atol=1e-6)
```

- [ ] **Step 4: Implement interpolation/root-center/scale**

For each joint coordinate, call `np.interp` over usable frame indices. Reject any joint with no usable observation. Center every frame at joint 0. Scale by the median positive finite `norm(j20-j0)` and reject `<=1e-6`.

- [ ] **Step 5: Run and commit**

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
- Produces `EventSequence(flat, nodes)` with shapes `[T,225]` and `[T,25,9]`.
- Produces `EventEncoder.fit`, `transform`, `fingerprint`, `to_json`, `from_json`.

- [ ] **Step 1: Write RED tests**

```python
def test_static_skeleton_emits_no_events():
    seq = np.ones((8,25,3), dtype=np.float32)
    out = EventEncoder.fixed_for_test(0.1).transform(seq)
    assert out.flat.shape == (8,225)
    assert out.nodes.shape == (8,25,9)
    assert np.count_nonzero(out.flat) == 0


def test_root_bone_motion_is_structurally_zero():
    out = EventEncoder.fixed_for_test(0.1).transform(moving_fixture())
    assert np.count_nonzero(out.nodes[:,0,3:6]) == 0
```

- [ ] **Step 2: Implement the frozen NTU25 parent map**

```python
NTU25_PARENT = (
    -1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10,
    0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11,
)
```

Compute joint motion `JM`, bone vectors, bone motion `BM`, and joint acceleration `A`; first-frame deltas are zero. Per-joint feature order is `[JM_xyz, BM_xyz, A_xyz]`.

- [ ] **Step 3: Fit thresholds and transform signed events**

Concatenate development-train features and compute:

```python
theta = np.maximum(np.quantile(np.abs(train_features), 0.75, axis=0), 1e-6)
```

This keeps structural-zero channels defined and inactive. Transform with strict `>` / `<` to `+1 / -1`, else `0`.

- [ ] **Step 4: Add serialization/fingerprint tests**

Canonical JSON contains feature-spec ID, quantile, parent map, and all 225 thresholds. Validation transforms must not mutate thresholds.

- [ ] **Step 5: Run and commit**

```bash
cd pose_snn
python -m pytest tests/test_events.py -q
git add pose_snn/src/pose_temporal_snn/events.py pose_snn/tests/test_events.py
git commit -m "feat: encode pose motion as signed events"
```

---

### Task 4: Implement shared streaming state contracts and RSNN

**Files:**
- Create: `pose_snn/src/pose_temporal_snn/models/base.py`
- Create: `pose_snn/src/pose_temporal_snn/models/rsnn.py`
- Create: `pose_snn/src/pose_temporal_snn/models/__init__.py`
- Test: `pose_snn/tests/test_rsnn.py`

**Interfaces:**
- Every model provides `initial_state(batch_size, device, dtype)`, `step(events_t, state)`, `logits(state)`, `forward(events)`, and `silence_state(state, fraction, seed)`.
- RSNN state contains `membrane`, `synaptic`, `spikes`, each `[B,96]`.

- [ ] **Step 1: Write streaming-equivalence RED test**

```python
def test_forward_equals_manual_step_loop():
    model = RSNN(input_dim=225, num_classes=10)
    x = torch.zeros(2,12,225)
    state = model.initial_state(2, x.device, x.dtype)
    for t in range(x.shape[1]):
        state = model.step(x[:,t], state)
    torch.testing.assert_close(model(x), model.logits(state))
```

- [ ] **Step 2: Implement hard-spike / fast-sigmoid surrogate**

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
        return grad / (1.0 + ctx.slope * x.abs()).pow(2), None
```

- [ ] **Step 3: Implement RSNN dynamics**

Use `Linear(225,96,bias=False)`, `Linear(96,96,bias=False)`, `Linear(96,10)`. Step:

```python
syn = 0.80 * state.synaptic + input_proj(events_t) + recurrent(state.spikes)
mem = 0.95 * state.membrane + syn
spikes = spike(mem - 1.0, 5.0)
mem = mem * (1.0 - spikes.detach())
```

`logits` reads `membrane + spikes`.

- [ ] **Step 4: Implement deterministic state silencing**

Use `random.Random(seed).sample(range(96), floor(96*fraction))`, minimum one unit. Zero those indices in membrane/synaptic/spikes and return a new state.

- [ ] **Step 5: Test gradients, statefulness, and ceiling**

Assert recurrent weight gets finite nonzero gradient on a tiny sequence, zero external input can evolve recurrent state, and parameter count <= 50_000.

- [ ] **Step 6: Run and commit**

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
- Modify: `pose_snn/src/pose_temporal_snn/models/__init__.py`
- Test: `pose_snn/tests/test_spiking_graph.py`

**Interfaces:**
- Input per step `[B,25,9]`.
- State membrane/synaptic/spikes `[B,25,32]`.
- Same streaming/silencing interface as Task 4.

- [ ] **Step 1: Write adjacency-constrained message-flow RED test**

Activate one joint in a synthetic state and assert direct graph message current is nonzero only at that node and its frozen anatomical neighbors before temporal recurrence propagates further.

- [ ] **Step 2: Build frozen normalized adjacency**

Create undirected edges from `NTU25_PARENT`, add self loops, row-normalize, and register the matrix as a non-trainable buffer.

- [ ] **Step 3: Implement graph-LIF step**

Use shared `Linear(9,32,bias=False)` input, `Linear(32,32,bias=False)` self-recurrent, `Linear(32,32,bias=False)` neighbor-recurrent. Compute:

```python
neighbor = torch.einsum("ij,bjh->bih", adjacency, state.spikes)
syn = 0.80 * state.synaptic + input_proj(events_t) + self_recurrent(state.spikes) + neighbor_recurrent(neighbor)
mem = 0.95 * state.membrane + syn
spikes = spike(mem - 1.0, 5.0)
mem = mem * (1.0 - spikes.detach())
```

Mean-pool `membrane + spikes` over 25 joints, then `Linear(32,10)`.

- [ ] **Step 4: Implement deterministic silencing**

Flatten 25*32 hidden units, sample 25% with `random.Random(seed)`, and zero those exact `(joint,channel)` entries in every state tensor.

- [ ] **Step 5: Test forward/manual equivalence and ceiling**

Assert deterministic fixed-seed output, no learned/new adjacency, and parameter count <= 50_000.

- [ ] **Step 6: Run and commit**

```bash
cd pose_snn
python -m pytest tests/test_spiking_graph.py -q
git add pose_snn/src/pose_temporal_snn/models pose_snn/tests/test_spiking_graph.py
git commit -m "feat: add human-skeleton spiking graph model"
```

---

### Task 6: Implement decisive streaming GRU baseline

**Files:**
- Create: `pose_snn/src/pose_temporal_snn/models/gru.py`
- Modify: `pose_snn/src/pose_temporal_snn/models/__init__.py`
- Test: `pose_snn/tests/test_gru.py`

**Interfaces:**
- Flat input `[B,225]`, state `[B,56]`, same streaming/silencing interface.

- [ ] **Step 1: Write streaming-equivalence RED test**

```python
def test_gru_forward_is_exact_manual_streaming():
    model = StreamingGRU(input_dim=225, hidden_size=56, num_classes=10)
    x = torch.randn(2,10,225)
    state = model.initial_state(2, x.device, x.dtype)
    for t in range(10):
        state = model.step(x[:,t], state)
    torch.testing.assert_close(model(x), model.logits(state))
```

- [ ] **Step 2: Implement `GRUCell(225,56)` plus readout**

No bidirectionality, attention, history cache, or future-frame access. `forward` loops over `step` only.

- [ ] **Step 3: Implement deterministic silencing and ceiling tests**

Sample 25% of 56 hidden units with `random.Random(seed)` and zero them. Assert parameter count <= 50_000 and prefix logits are unchanged by modifying unseen future frames.

- [ ] **Step 4: Run and commit**

```bash
cd pose_snn
python -m pytest tests/test_gru.py -q
git add pose_snn/src/pose_temporal_snn/models pose_snn/tests/test_gru.py
git commit -m "feat: add streaming GRU control"
```

---

### Task 7: Add shared evaluation and temporal diagnostics

**Files:**
- Create: `pose_snn/src/pose_temporal_snn/evaluation.py`
- Modify: `pose_snn/src/pose_temporal_snn/models/base.py`
- Modify: `pose_snn/src/pose_temporal_snn/models/rsnn.py`
- Modify: `pose_snn/src/pose_temporal_snn/models/spiking_graph.py`
- Modify: `pose_snn/src/pose_temporal_snn/models/gru.py`
- Test: `pose_snn/tests/test_evaluation.py`

**Interfaces:**
- Produces `macro_f1`, `early_prediction_auc`, `retention_metrics`, `recovery_metrics`, `activity_metrics`.
- Evaluation interacts with models only through the streaming interface.

- [ ] **Step 1: Write exact macro-F1 / AUC tests**

```python
def test_early_auc_is_normalized_trapezoid():
    ratios = (0.10,0.20,0.40,0.60,0.80,1.00)
    scores = (0.2,0.3,0.5,0.7,0.8,0.9)
    expected = np.trapezoid(scores, ratios) / 0.90
    assert early_prediction_auc(ratios, scores) == pytest.approx(expected)
```

- [ ] **Step 2: Implement prefix evaluation via `step` only**

For each ratio consume exactly `ceil(T*ratio)` frames from a fresh state and call `logits`; do not call model-specific `forward` for streaming measurements.

- [ ] **Step 3: Implement zero-input retention**

At 40%, record true-class logit margin; step 20 zero-input frames. Normalize by `max(abs(initial_margin),1e-6)` and trapezoid-integrate. Report retention AUC and T50.

- [ ] **Step 4: Implement perturbation recovery**

Call the model's `silence_state(state, 0.25, seed)` at 40%, continue the same future observed frames for 10 steps, and measure first return to the matched unperturbed predicted class and fixed-horizon recovery rate.

- [ ] **Step 5: Add activity accounting**

RSNN/SpikingGraph expose total spikes, active-unit fraction, and recurrent/graph synaptic events per frame. GRU returns `None` for spike-specific metrics; never fabricate spike equivalents.

- [ ] **Step 6: Run and commit**

```bash
cd pose_snn
python -m pytest tests/test_evaluation.py tests/test_rsnn.py tests/test_spiking_graph.py tests/test_gru.py -q
git add pose_snn/src/pose_temporal_snn pose_snn/tests
git commit -m "feat: measure streaming temporal behavior"
```

---

### Task 8: Add deterministic preparation and fair training

**Files:**
- Create: `pose_snn/src/pose_temporal_snn/data.py`
- Create: `pose_snn/src/pose_temporal_snn/training.py`
- Test: `pose_snn/tests/test_training_fairness.py`
- Test: `pose_snn/tests/test_final_test_seal.py`

**Interfaces:**
- Produces `PreparedSample(sample_id, label, flat_events, node_events)`.
- Produces `PreparedData(binding, encoder, train_samples, validation_samples, final_test_inventory)`.
- Produces `train_validation_arm(model_kind, seed, prepared, protocol) -> RunResult`.

- [ ] **Step 1: Write leakage RED tests**

Build tiny NTU fixtures where validation motion is changed 1000x. `prepare_data` must produce identical encoder fingerprints but different dataset-content hashes. Include an intentionally non-normalizable final-test sample and assert preparation succeeds because final-test rows are inventoried but never normalized/event-encoded.

- [ ] **Step 2: Implement `prepare_data`**

Build manifest; normalize only train/validation; fit encoder only on train; transform validation afterward. Labels are sorted action IDs mapped to `0..9`. Convert encoder outputs to immutable float32 Torch tensors in `PreparedSample`. Canonically hash dataset bytes, split assignments, label map, encoder, and preparation config.

- [ ] **Step 3: Write fairness RED tests**

For a fixed seed assert all model kinds receive the same sample-order fingerprint and update ceiling. Repeated same-arm fixture runs must be deterministic on CPU. Record architecture fingerprint and parameter count for every run.

- [ ] **Step 4: Implement deterministic training**

Before model construction call `torch.manual_seed(seed)`. For each epoch:

```python
order = list(range(len(train_samples)))
random.Random((seed << 32) + epoch).shuffle(order)
```

Train batch size 1 with Adam/lr `1e-3`; stop at 10 epochs or 2000 updates. Select checkpoint only by validation full-sequence macro-F1, then evaluate early/retention/recovery on that checkpoint.

- [ ] **Step 5: Implement architecture fingerprints**

Fingerprint model kind plus all frozen architecture/dynamics constants; SpikingGraph also includes parent map/adjacency definition. Exclude learned parameter values.

- [ ] **Step 6: Run and commit**

```bash
cd pose_snn
python -m pytest tests/test_training_fairness.py tests/test_final_test_seal.py -q
git add pose_snn/src/pose_temporal_snn pose_snn/tests
git commit -m "feat: train temporal models under one fair protocol"
```

---

### Task 9: Add development comparison, stop rule, and CLI

**Files:**
- Create: `pose_snn/src/pose_temporal_snn/results.py`
- Create: `pose_snn/src/pose_temporal_snn/cli.py`
- Modify: `pose_snn/pyproject.toml`
- Test: `pose_snn/tests/test_development_runner.py`

**Interfaces:**
- CLI `pose-temporal-snn prepare --root ... --protocol ... --output ...`.
- CLI `pose-temporal-snn compare-validation --root ... --prepared ... --output ...`.
- Produces exactly 15 cells: 3 models x 5 seeds, plus `summary.json`.

- [ ] **Step 1: Add CLI entry point**

```toml
[project.scripts]
pose-temporal-snn = "pose_temporal_snn.cli:main"
```

- [ ] **Step 2: Write complete-matrix / stop-rule RED tests**

Reject missing/duplicate model-seed cells or any result with mismatched dataset/split/encoder/training/model pins. Full-sequence accuracy cannot turn a temporal null into pass.

- [ ] **Step 3: Implement paired decisions**

Primary comparison is RSNN minus GRU by seed. `pass` requires early-AUC mean >=0.02, positive 4/5, plus retention or recovery mean >=0.05 positive 4/5. Reverse early effect <=-0.02 positive in the reverse direction for 4/5 is `negative`; otherwise `null`. SpikingGraph comparisons are secondary.

- [ ] **Step 4: Keep final test inaccessible**

V1 preflight exposes no final-test CLI path. Internal access while sealed raises `ValueError("final test is sealed")`.

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
- CI downloads no NTU data and runs no FlyWire/Codex work.

- [ ] **Step 1: Add lightweight job**

Install `pytest numpy` and `pip install --no-deps -e ./pose_snn`; run protocol/NTU/normalization/events tests.

- [ ] **Step 2: Add CPU-Torch job**

Install CPU-only Torch; run model/evaluation/training/development tests.

- [ ] **Step 3: Add full suite and Lean regression**

```bash
python -m pytest pose_snn/tests -q
lake build
```

Use separate jobs so an external elan-install failure is distinguishable from Python model/test failures.

- [ ] **Step 4: Add source-isolation regression**

Scan only `pose_snn/src/**/*.py` and fail if source contains `yolo_flywire`, `flywire`, `codex`, or `connectome`. Standard-library, NumPy, Torch, and intra-`pose_temporal_snn` imports are allowed.

- [ ] **Step 5: Verify exact-head scope**

Require all Python jobs GREEN on one commit. Compare against base `7b5de8176adf4beff5e2a46c8ff021caad6404ae` and verify no file under `python/yolo_flywire`, `YoloFlywire/`, or existing FlyWire protocols changed.

- [ ] **Step 6: Stop if real NTU bytes are absent**

At this checkpoint code is ready for a legally obtained local NTU skeleton root. Do not fabricate data, use unauthorized mirrors, or claim RSNN/GRU performance from fixtures.

- [ ] **Step 7: Commit verification corrections**

```bash
git add .github/workflows/pose-temporal-snn.yml pose_snn
git commit -m "ci: verify isolated pose temporal SNN V1"
```

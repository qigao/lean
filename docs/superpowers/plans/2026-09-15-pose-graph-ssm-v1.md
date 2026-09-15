# Pose Graph + SSM V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated causal streaming benchmark that compares GRU, SSMOnly, GraphTCN, and GraphSSM on NTU25 early action prediction and temporal robustness without any FlyWire/SNN dependency.

**Architecture:** A new `pose_graph_ssm/` Python project owns protocol, NTU skeleton parsing, tracking-aware normalization, continuous `[T,25,15]` kinematic features, frozen human adjacency, native selective diagonal SSM blocks, four causal streaming model arms, shared evaluation, and fair development training. Final test remains sealed until all measured provenance pins are frozen.

**Tech Stack:** Python 3.11, NumPy >=1.26, PyTorch >=2.3, pytest >=8.0, stdlib `hashlib`/`json`/`pathlib`/`random`.

**Spec:** `docs/superpowers/specs/2026-09-15-pose-graph-ssm-v1-design.md`

## Global Constraints

- Repository: `qigao/lean`; branch: `research/pose-graph-ssm-v1`.
- New project root is exactly `pose_graph_ssm/`; no source file may import or mention `yolo_flywire`, `flywire`, `codex`, `connectome`, `cx_`, `spiking`, `lif`, or `surrogate_spike`.
- V1 actions are exactly `(8, 9, 22, 23, 26, 27, 31, 34, 35, 36)`.
- Official NTU120 X-Sub outer-training subjects are exactly `{1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38,45,46,47,49,50,52,53,54,55,56,57,58,59,70,74,78,80,81,82,83,84,85,86,89,91,92,93,94,95,97,98,100,103}`.
- Inner-validation subjects are exactly `(14, 28, 35, 46, 50, 54, 74, 83, 84, 86, 103)`.
- Final-test skeleton bytes may be inventoried/hashed while sealed, but coordinates may not be normalized, feature-transformed, standardized, scored, or used for model selection.
- Primary-body rule is `max-fully-tracked-joints-then-lowest-body-id`.
- Normalization is tracking-aware temporal interpolation, joint-0 centering, and median `||joint20-joint0||` scale; degenerate/nonfinite samples fail closed.
- Features per joint are exactly `P, JM, B, BM, A`, each XYZ, for `15` channels and tensor `[T,25,15]`.
- Feature standardization is train-only per `(joint,channel)`: `mu[25,15]`, `sigma[25,15]`, transform denominator `max(sigma,1e-6)`.
- Canonical NTU25 parent map is `(-1,0,20,2,20,4,5,6,20,8,9,10,0,12,13,14,0,16,17,18,1,7,7,11,11)`.
- Human adjacency is undirected + self loops + row normalization; it is frozen and non-trainable.
- All models are causal streaming and expose `initial_state`, `step`, `logits`, `forward`; `forward` is repeated `step` only.
- Standardized all-zero frame is canonical no-observation input. Direct frame-entry projections and graph self/neighbor transforms use `bias=False`. SSM `W_B`, `W_C`, `W_out` are bias-free; `b_delta` is allowed only as intrinsic time-scale bias.
- Model arms are exactly `gru`, `ssm_only`, `graph_tcn`, `graph_ssm`.
- Shared hidden width is `64`; GRU uses `GRUCell(375,64)`.
- Selective SSM recurrence is exactly the design-spec recurrence with stable diagonal `A=-softplus(A_log)` and clamped exponent argument `[-20,0]`.
- GraphTCN uses two graph+causal-conv blocks, kernel `3`, dilations `1` then `2`, with fixed streaming caches.
- GraphSSM uses two graph+SelectiveSSM blocks, state independently per joint, parameters shared across joints.
- Seeds are exactly `(7,11,19,23,31)`.
- Training is exactly AdamW, lr `3e-4`, weight decay `1e-4`, batch size `1`, max `20` epochs, max `4000` updates, checkpoint by best validation full-sequence macro F1 only.
- Parameter ceiling is `120_000` trainable parameters per arm.
- Observation ratios are exactly `(0.10,0.20,0.40,0.60,0.80,1.00)`.
- Retention boundary `0.40`, horizon `20` standardized zero frames.
- Pose-dropout burst is `8` standardized zero frames at 40%, followed by real stream; recovery horizon is `10` restored frames.
- Primary pass: mean paired `(GraphSSM-GRU) early_prediction_auc >= 0.02`, positive in >=4/5 seeds, plus retention-AUC or dropout-recovery-rate mean improvement >=0.05 positive in >=4/5 seeds.
- Graph attribution: mean paired `(GraphSSM-SSMOnly) early_prediction_auc >= 0.01`, positive in >=4/5 seeds.
- SSM attribution: mean paired `(GraphSSM-GraphTCN) early_prediction_auc >= 0.01`, positive in >=4/5 seeds.
- Better full-sequence F1 alone cannot turn a temporal null into pass.

---

### Task 1: Create isolated package and fail-closed protocol

**Files:**
- Create: `pose_graph_ssm/pyproject.toml`
- Create: `pose_graph_ssm/src/pose_graph_ssm/__init__.py`
- Create: `pose_graph_ssm/src/pose_graph_ssm/protocol.py`
- Create: `pose_graph_ssm/protocols/v1-development-preflight.json`
- Test: `pose_graph_ssm/tests/test_protocol.py`

**Interfaces:**
- Produces `ModelConfig`, `TrainingConfig`, `ExperimentProtocol`, `load_protocol(path)`, `validate_protocol_dict(raw)`.
- Later tasks consume exact dataset/split/model/training/metric constants and measured provenance pins.

- [ ] **Step 1: Write package metadata and protocol RED tests**

`pose_graph_ssm/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pose-graph-ssm"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "torch>=2.3"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

`tests/test_protocol.py` must include:

```python
from pathlib import Path
import json
import pytest
from pose_graph_ssm.protocol import load_protocol, validate_protocol_dict

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/v1-development-preflight.json"


def test_preflight_freezes_graph_ssm_v1_and_seals_final_test():
    p = load_protocol(PROTOCOL)
    assert p.actions == (8,9,22,23,26,27,31,34,35,36)
    assert p.validation_subjects == (14,28,35,46,50,54,74,83,84,86,103)
    assert p.model_kinds == ("gru","ssm_only","graph_tcn","graph_ssm")
    assert p.seeds == (7,11,19,23,31)
    assert p.parameter_ceiling == 120_000
    assert p.final_test_enabled is False


def test_protocol_rejects_abandoned_stack_fields():
    raw = json.loads(PROTOCOL.read_text())
    raw["flywire_release"] = "v783"
    with pytest.raises(ValueError, match="schema"):
        validate_protocol_dict(raw)


def test_final_test_requires_every_measured_pin():
    raw = json.loads(PROTOCOL.read_text())
    raw["final_test_enabled"] = True
    with pytest.raises(ValueError, match="final test provenance"):
        validate_protocol_dict(raw)
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=pose_graph_ssm/src python -m pytest pose_graph_ssm/tests/test_protocol.py -q
```

Expected: import failure because `pose_graph_ssm.protocol` does not exist.

- [ ] **Step 3: Implement immutable protocol validator**

Public dataclasses:

```python
@dataclass(frozen=True)
class ModelConfig:
    kind: str
    hidden_size: int
    blocks: int

@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    max_updates: int
    checkpoint_rule: str

@dataclass(frozen=True)
class ExperimentProtocol:
    protocol_id: str
    actions: tuple[int,...]
    outer_train_subjects: tuple[int,...]
    validation_subjects: tuple[int,...]
    model_kinds: tuple[str,...]
    seeds: tuple[int,...]
    observation_ratios: tuple[float,...]
    retention_ratio: float
    retention_horizon: int
    dropout_burst: int
    recovery_horizon: int
    parameter_ceiling: int
    training: TrainingConfig
    models: tuple[ModelConfig,...]
    final_test_enabled: bool
    dataset_content_hash: str | None
    split_hash: str | None
    feature_stats_hash: str | None
    label_map_hash: str | None
    adjacency_hash: str | None
    ssm_spec_hash: str | None
    model_fingerprints: tuple[tuple[str,str],...] | None
    length_quartiles: tuple[int,int,int] | None
```

Reject extra keys, changed frozen constants, malformed SHA-256 digests, wrong model-kind roster, validation subjects outside outer train, and final-test enablement while any measured pin is null.

- [ ] **Step 4: Add preflight JSON with null measured pins and `final_test_enabled=false`**

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=pose_graph_ssm/src python -m pytest pose_graph_ssm/tests/test_protocol.py -q
git add pose_graph_ssm
git commit -m "test: freeze pose Graph SSM V1 protocol"
```

---

### Task 2: Parse, inventory, split, and normalize native NTU skeletons

**Files:**
- Create: `pose_graph_ssm/src/pose_graph_ssm/ntu.py`
- Create: `pose_graph_ssm/src/pose_graph_ssm/normalization.py`
- Test: `pose_graph_ssm/tests/test_ntu.py`
- Test: `pose_graph_ssm/tests/test_normalization.py`

**Interfaces:**
- Produces `NtuBody`, `NtuSkeletonSample`, `parse_skeleton_file`, `select_primary_body`, `split_for_subject`, `build_manifest`, `normalize_body`.

- [ ] **Step 1: Write parser/split RED tests**

Include canonical filename validation, exactly 25 joints, finite XYZ, tracking states in `{0,1,2}`, deterministic primary-body tie break, and:

```python
def test_subject_split_is_frozen(protocol):
    assert split_for_subject(56, protocol) == "train"
    assert split_for_subject(14, protocol) == "validation"
    assert split_for_subject(3, protocol) == "final_test"
```

- [ ] **Step 2: Implement strict parser and byte-backed manifest**

Manifest rows record sample ID, subject/action, split, selected body ID, frame count, relative path, size and SHA-256. Reject duplicate sample IDs/content and symlinks. Hash canonical byte inventory and split assignments.

- [ ] **Step 3: Write normalization RED tests**

```python
def test_translation_and_uniform_scale_are_removed():
    a = synthetic_body(offset=(0,0,0), scale=1.0)
    b = synthetic_body(offset=(10,-3,5), scale=7.0)
    np.testing.assert_allclose(normalize_body(a), normalize_body(b), atol=1e-6)


def test_tracking_zero_is_interpolated_not_origin_jump():
    body = body_with_missing_joint_frame(joint=5, frame=2)
    out = normalize_body(body)
    np.testing.assert_allclose(out[2,5], (out[1,5]+out[3,5])/2.0, atol=1e-6)
```

- [ ] **Step 4: Implement interpolation/root-center/torso-scale normalization**

Use `np.interp` with endpoint hold; reject any joint with no usable observation and torso scale `<=1e-6`.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=pose_graph_ssm/src python -m pytest pose_graph_ssm/tests/test_ntu.py pose_graph_ssm/tests/test_normalization.py -q
git add pose_graph_ssm/src/pose_graph_ssm/ntu.py pose_graph_ssm/src/pose_graph_ssm/normalization.py pose_graph_ssm/tests
git commit -m "feat: normalize native NTU skeleton streams"
```

---

### Task 3: Build continuous kinematic features and train-only standardization

**Files:**
- Create: `pose_graph_ssm/src/pose_graph_ssm/features.py`
- Test: `pose_graph_ssm/tests/test_features.py`

**Interfaces:**
- Produces `NTU25_PARENT`, `kinematic_features(sequence)->np.ndarray[T,25,15]`, `FeatureStandardizer.fit(train_features)`, `transform`, `fingerprint`, `to_json`, `from_json`.

- [ ] **Step 1: Write feature-structure RED tests**

```python
def test_features_keep_joint_axis_and_15_channels():
    x = moving_sequence()
    out = kinematic_features(x)
    assert out.shape == (x.shape[0],25,15)


def test_root_bone_and_root_bone_motion_are_zero():
    out = kinematic_features(moving_sequence())
    assert np.count_nonzero(out[:,0,6:12]) == 0
```

- [ ] **Step 2: Implement `P, JM, B, BM, A` exactly**

First-frame `JM`, `BM`, `A` are zero. Root bone is zero. Reject nonfinite or wrong-shaped input.

- [ ] **Step 3: Write train-only standardization RED tests**

Changing validation values by 1000x must not change `FeatureStandardizer.fit(train).fingerprint`. Test `mu.shape == sigma.shape == (25,15)` and serialized round trip.

- [ ] **Step 4: Implement per-joint/channel statistics**

Use population mean/std over all train frames; denominator is `np.maximum(sigma,1e-6)`. Fingerprint canonical JSON containing feature spec ID, parent map, mean and std arrays.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=pose_graph_ssm/src python -m pytest pose_graph_ssm/tests/test_features.py -q
git add pose_graph_ssm/src/pose_graph_ssm/features.py pose_graph_ssm/tests/test_features.py
git commit -m "feat: build continuous pose kinematics"
```

---

### Task 4: Freeze human adjacency and graph block

**Files:**
- Create: `pose_graph_ssm/src/pose_graph_ssm/graph.py`
- Create: `pose_graph_ssm/src/pose_graph_ssm/models/base.py`
- Test: `pose_graph_ssm/tests/test_graph.py`

**Interfaces:**
- Produces `ntu25_adjacency() -> torch.Tensor[25,25]`, `adjacency_fingerprint()`, `RMSNorm`, `HumanGraphBlock(width)`.

- [ ] **Step 1: Write adjacency RED tests**

Assert self loops, symmetric anatomical links before normalization, row sums equal 1 after normalization, no non-anatomical nonzero entries, and adjacency is registered as non-trainable buffer inside `HumanGraphBlock`.

- [ ] **Step 2: Implement canonical row-normalized adjacency from `NTU25_PARENT`**

- [ ] **Step 3: Write graph-block zero-drive and shape tests**

```python
def test_graph_block_preserves_shape_and_zero_input():
    block = HumanGraphBlock(64)
    x = torch.zeros(2,25,64)
    y = block(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y, torch.zeros_like(y))
```

- [ ] **Step 4: Implement bias-free self/neighbor graph projections and RMSNorm residual**

`W_self` and `W_neighbor` use `bias=False`. `RMSNorm` scale is trainable but has no additive bias.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=pose_graph_ssm/src python -m pytest pose_graph_ssm/tests/test_graph.py -q
git add pose_graph_ssm/src/pose_graph_ssm/graph.py pose_graph_ssm/src/pose_graph_ssm/models/base.py pose_graph_ssm/tests/test_graph.py
git commit -m "feat: add frozen human graph block"
```

---

### Task 5: Implement native selective diagonal SSM and SSMOnly

**Files:**
- Create: `pose_graph_ssm/src/pose_graph_ssm/models/selective_ssm.py`
- Create: `pose_graph_ssm/src/pose_graph_ssm/models/ssm_only.py`
- Test: `pose_graph_ssm/tests/test_selective_ssm.py`
- Test: `pose_graph_ssm/tests/test_models.py`

**Interfaces:**
- `SelectiveSSMBlock(width)` exposes `initial_state(shape,device,dtype)` and `step(u_t,state)->(output,state)`.
- `SSMOnly` exposes common streaming model interface and flattens `[B,25,15]` to 375 frame features.

- [ ] **Step 1: Write recurrence RED tests**

Check deterministic streaming equivalence, finite states over 1000 zero/random steps, zero standardized input produces no `B_t`/`C_t` external drive from biases, and exponent argument never exceeds `[−20,0]`.

- [ ] **Step 2: Implement exact selective SSM recurrence**

Use trainable `A_log`, `D`, bias-free `W_B/W_C/W_out`, `W_delta` with only `b_delta`; return RMS-normalized residual output.

- [ ] **Step 3: Write SSMOnly streaming RED tests**

`forward(sequence)` must equal manual repeated `step`; changing unseen future frames must not alter prefix logits; parameter count must be <=120k.

- [ ] **Step 4: Implement SSMOnly: `Linear(375,64,bias=False) -> RMSNorm -> 2xSelectiveSSMBlock -> Linear(64,10)`**

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=pose_graph_ssm/src python -m pytest pose_graph_ssm/tests/test_selective_ssm.py pose_graph_ssm/tests/test_models.py -q
git add pose_graph_ssm/src/pose_graph_ssm/models pose_graph_ssm/tests
git commit -m "feat: add causal selective SSM core"
```

---

### Task 6: Implement GRU, GraphTCN, and GraphSSM model arms

**Files:**
- Create: `pose_graph_ssm/src/pose_graph_ssm/models/gru.py`
- Create: `pose_graph_ssm/src/pose_graph_ssm/models/graph_tcn.py`
- Create: `pose_graph_ssm/src/pose_graph_ssm/models/graph_ssm.py`
- Create: `pose_graph_ssm/src/pose_graph_ssm/models/__init__.py`
- Test: `pose_graph_ssm/tests/test_streaming.py`
- Extend: `pose_graph_ssm/tests/test_models.py`

**Interfaces:**
- All four model kinds consume one standardized frame and expose common streaming interface.
- GraphTCN state is fixed causal caches for dilation 1/2 convs.
- GraphSSM state contains two independent `[B,25,64]` SSM states.

- [ ] **Step 1: Write common streaming RED tests for all four arms**

For each model: manual repeated `step` equals `forward`; prefix logits ignore changed future frames; parameter count <=120k; `forward` rejects nonfinite input.

- [ ] **Step 2: Implement GRU**

`GRUCell(375,64)` + readout. No bidirectionality/history cache.

- [ ] **Step 3: Implement causal temporal conv block**

Kernel 3, explicit left-history cache, dilation-specific offsets. Frame-entry transform is bias-free. Streaming step must match a causal reference convolution over the same prefix.

- [ ] **Step 4: Implement GraphTCN**

`Linear(15,64,bias=False)` then two `HumanGraphBlock + CausalTemporalConvBlock` stages with dilations 1 and 2, learned joint pooling, classifier.

- [ ] **Step 5: Implement GraphSSM**

`Linear(15,64,bias=False)` then two `HumanGraphBlock + SelectiveSSMBlock` stages. SSM weights shared across joints; state is separate per joint. Use same learned joint pooling as GraphTCN.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=pose_graph_ssm/src python -m pytest pose_graph_ssm/tests/test_models.py pose_graph_ssm/tests/test_streaming.py -q
git add pose_graph_ssm/src/pose_graph_ssm/models pose_graph_ssm/tests
git commit -m "feat: add Graph SSM comparison arms"
```

---

### Task 7: Add shared early/retention/dropout/length evaluation

**Files:**
- Create: `pose_graph_ssm/src/pose_graph_ssm/evaluation.py`
- Test: `pose_graph_ssm/tests/test_evaluation.py`

**Interfaces:**
- Produces `macro_f1`, `early_prediction_auc`, `prefix_predictions`, `retention_metrics`, `dropout_recovery_metrics`, `length_stratified_metrics`.

- [ ] **Step 1: Write exact macro-F1 and normalized-AUC tests**

```python
def test_early_auc_is_normalized_trapezoid():
    ratios=(0.10,0.20,0.40,0.60,0.80,1.00)
    scores=(0.2,0.3,0.5,0.7,0.8,0.9)
    expected=np.trapezoid(scores, ratios)/(ratios[-1]-ratios[0])
    assert early_prediction_auc(ratios,scores)==pytest.approx(expected)
```

- [ ] **Step 2: Implement prefix evaluation via `step` only**

For ratio `r`, consume exactly `ceil(T*r)` frames from fresh state and call `logits`.

- [ ] **Step 3: Implement zero-input retention**

At 40%, step 20 standardized zero frames. Use true-class margin normalized by `max(abs(initial_margin),1e-6)` and trapezoidal AUC.

- [ ] **Step 4: Implement pose-dropout recovery**

At 40%, replace exactly 8 input frames with standardized zeros in perturbed run while reference receives real frames; then restore real stream and report first stable return to reference predicted class plus recovery within 10 restored frames.

- [ ] **Step 5: Implement train-derived frame-count quartile stratification**

`length_quartile_boundaries(train_frame_counts)` returns three integer boundaries deterministically; validation metrics use those frozen boundaries only.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=pose_graph_ssm/src python -m pytest pose_graph_ssm/tests/test_evaluation.py -q
git add pose_graph_ssm/src/pose_graph_ssm/evaluation.py pose_graph_ssm/tests/test_evaluation.py
git commit -m "feat: evaluate causal temporal behavior"
```

---

### Task 8: Add leak-free preparation and fair four-arm training

**Files:**
- Create: `pose_graph_ssm/src/pose_graph_ssm/data.py`
- Create: `pose_graph_ssm/src/pose_graph_ssm/training.py`
- Test: `pose_graph_ssm/tests/test_training_fairness.py`
- Test: `pose_graph_ssm/tests/test_final_test_seal.py`

**Interfaces:**
- Produces `PreparedSample(sample_id,label,features)`, `PreparedData(binding,standardizer,train_samples,validation_samples,final_test_inventory)`, `prepare_data(root,protocol)`, `train_validation_arm(model_kind,seed,prepared,protocol)->RunResult`.

- [ ] **Step 1: Write leakage RED tests**

Changing validation coordinates by 1000x must change dataset-content hash but not feature-statistics fingerprint. Include a deliberately non-normalizable final-test skeleton and assert sealed preparation succeeds because final-test coordinates are never parsed into normalization/features.

- [ ] **Step 2: Implement preparation**

Build raw manifest; normalize/feature only train+validation; fit standardizer only on train; transform validation after fit. Label map is sorted frozen action IDs to `0..9`. Compute dataset/split/feature-stats/label-map/adjacency/SSM-spec hashes and train frame-count quartiles.

- [ ] **Step 3: Write training-fairness RED tests**

For same seed, all four model kinds must receive identical sample-order fingerprint and update ceiling. Repeated same-arm CPU fixture runs must be deterministic. Reject parameter count above ceiling or any binding hash mismatch.

- [ ] **Step 4: Implement deterministic training**

Before construction `torch.manual_seed(seed)`. Per epoch:

```python
order=list(range(len(train_samples)))
random.Random((seed<<32)+epoch).shuffle(order)
```

Train batch size 1 with AdamW/lr `3e-4`/weight decay `1e-4`, stop at 20 epochs or 4000 updates. Select checkpoint only by validation full-sequence macro F1; then compute all other metrics from chosen checkpoint.

- [ ] **Step 5: Implement architecture fingerprints**

Fingerprint model kind plus every frozen architecture/dynamics constant; Graph models include adjacency fingerprint; SSM models include SSM recurrence fingerprint. Exclude learned parameter values.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=pose_graph_ssm/src python -m pytest pose_graph_ssm/tests/test_training_fairness.py pose_graph_ssm/tests/test_final_test_seal.py -q
git add pose_graph_ssm/src/pose_graph_ssm pose_graph_ssm/tests
git commit -m "feat: train Graph SSM arms under one protocol"
```

---

### Task 9: Add paired decisions, CLI, CI, and exact-head verification

**Files:**
- Create: `pose_graph_ssm/src/pose_graph_ssm/results.py`
- Create: `pose_graph_ssm/src/pose_graph_ssm/cli.py`
- Modify: `pose_graph_ssm/pyproject.toml`
- Create: `.github/workflows/pose-graph-ssm.yml`
- Test: `pose_graph_ssm/tests/test_development_runner.py`

**Interfaces:**
- CLI `pose-graph-ssm prepare --root ... --protocol ... --output ...`.
- CLI `pose-graph-ssm compare-validation --root ... --prepared ... --output ...`.
- Produces exactly `4 models x 5 seeds = 20` result cells plus `summary.json`.

- [ ] **Step 1: Write result-matrix and stop-rule RED tests**

Reject missing/duplicate cells or mismatched dataset/split/stats/training/model pins. Primary decision uses exact GraphSSM-vs-GRU thresholds. Attribution decisions use exact GraphSSM-vs-SSMOnly and GraphSSM-vs-GraphTCN thresholds. Full-sequence F1 cannot override temporal null.

- [ ] **Step 2: Implement paired decision reducer and sealed CLI**

Preflight exposes no final-test scoring command; any internal sealed final-test request raises `ValueError("final test is sealed")`.

- [ ] **Step 3: Add project script**

```toml
[project.scripts]
pose-graph-ssm = "pose_graph_ssm.cli:main"
```

- [ ] **Step 4: Add independent CI**

Jobs:

```text
lightweight: pytest protocol/ntu/normalization/features with numpy
models: CPU torch graph/SSM/model/streaming tests
full: pytest pose_graph_ssm/tests -q
isolation: grep source for forbidden abandoned-stack tokens
lean: lake build
```

CI downloads no NTU data and runs no FlyWire/Codex workflow.

- [ ] **Step 5: Verify exact-head scope**

Compare against base `7b5de8176adf4beff5e2a46c8ff021caad6404ae`. Confirm no files under `python/yolo_flywire/`, `YoloFlywire/`, or existing FlyWire protocols changed.

- [ ] **Step 6: Run final code-only verification**

```bash
PYTHONPATH=pose_graph_ssm/src python -m pytest pose_graph_ssm/tests -q
lake build
```

Require all Python jobs GREEN on one commit. If NTU bytes are absent, stop after code readiness; do not fabricate performance evidence.

- [ ] **Step 7: Commit verification corrections**

```bash
git add pose_graph_ssm .github/workflows/pose-graph-ssm.yml
git commit -m "ci: verify isolated pose Graph SSM V1"
```
